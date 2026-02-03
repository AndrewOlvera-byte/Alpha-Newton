"""
RL Training Callbacks for RLVR

Provides specialized callbacks for:
- Top-K checkpoint management based on custom eval metrics
- Held-out evaluation with difficulty-weighted scoring
- WandB integration for custom metrics
"""

import heapq
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from datasets import load_dataset


class BaseTopKCheckpointCallback(TrainerCallback):
    """
    Base class for top-K checkpoint management with custom evaluation.

    Subclasses should implement:
    - _compute_metrics(prompts, completions, ground_truths, metadata) -> Dict[str, float]

    This base class handles:
    - Min-heap for tracking top-K checkpoints
    - Batched PyTorch generation with Flash Attention
    - Checkpoint deletion
    - JSON logging and WandB integration
    """

    def __init__(
        self,
        k: int = 5,
        eval_dataset_path: str = None,
        eval_every_n_steps: int = 50,
        metric_name: str = "held_out_accuracy",
        reward_function: str = "ppo_binary",
        source_type: str = "local",
        output_dir: str = None,
        trainer = None,
        tokenizer = None,
        eval_batch_size: int = 8,
        use_compile: bool = False,
    ):
        self.k = k
        self.eval_dataset_path = eval_dataset_path
        self.eval_every_n_steps = eval_every_n_steps
        self.metric_name = metric_name
        self.reward_function_name = reward_function
        self.source_type = source_type
        self.output_dir = output_dir
        self.eval_batch_size = eval_batch_size
        self.use_compile = use_compile

        # Store trainer and tokenizer references
        self.trainer = trainer
        self.tokenizer = tokenizer

        # Min-heap: stores tuples of (score, step, checkpoint_path)
        self.heap: List[Tuple[float, int, str]] = []

        # Tracking
        self.best_score = float('-inf')
        self.best_step = None
        self.eval_history = []

        # Dataset loaded lazily on first eval
        self.eval_dataset = None
        self.reward_fn = None
        self.last_eval_step = -1
        self.compiled_model = None

    def _load_eval_dataset(self):
        """Load held-out evaluation dataset."""
        if self.eval_dataset is not None:
            return

        if self.eval_dataset_path is None:
            print("[TopK Callback] No eval dataset path provided. Skipping custom eval.")
            return

        print(f"[TopK Callback] Loading held-out eval dataset from: {self.eval_dataset_path}")

        if self.source_type == "local":
            self.eval_dataset = load_dataset(
                "json",
                data_files=self.eval_dataset_path,
                split="train"
            )
        elif self.source_type == "hf":
            self.eval_dataset = load_dataset(
                self.eval_dataset_path,
                split="train"
            )
        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")

        print(f"[TopK Callback] Loaded {len(self.eval_dataset)} held-out samples")

        # Load reward function
        from src.rlvr.math_verifier import get_reward_function
        self.reward_fn = get_reward_function(self.reward_function_name)
        print(f"[TopK Callback] Using reward function: {self.reward_function_name}")

    def _run_custom_eval(self, trainer, tokenizer) -> Dict[str, float]:
        """
        Run custom evaluation on held-out set.
        Subclasses can override _compute_metrics for custom logic.
        """
        if self.eval_dataset is None:
            return {}

        print(f"[TopK Callback] Running custom evaluation on {len(self.eval_dataset)} samples...")

        # Extract data from dataset
        prompts = []
        ground_truths = []
        metadata = []

        for sample in self.eval_dataset:
            prompts.append(sample["prompt"])
            ground_truths.append(sample["answer"])
            metadata.append({
                "difficulty": sample.get("difficulty", "default"),
                "dataset_source": sample.get("dataset_source", "unknown"),
                "problem": sample.get("problem", ""),
            })

        # Generate completions using batched PyTorch generation
        completions = self._generate_completions(trainer.model, prompts, tokenizer)

        # Compute metrics (subclass-specific)
        metrics = self._compute_metrics(prompts, completions, ground_truths, metadata)

        print(f"[TopK Callback] Eval complete: {self.metric_name}={metrics.get(self.metric_name, 0.0):.4f}")

        return metrics

    def _compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: List[str],
        metadata: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics. Override in subclasses.

        Args:
            prompts: List of prompts
            completions: List of generated completions
            ground_truths: List of ground truth answers
            metadata: List of metadata dicts per sample

        Returns:
            Dict with metrics including self.metric_name
        """
        raise NotImplementedError("Subclasses must implement _compute_metrics")

    def _generate_completions(
        self,
        model,
        prompts: List[str],
        tokenizer,
    ) -> List[str]:
        """
        Generate completions using batched PyTorch generation with Flash Attention.

        Uses model.eval() and torch.no_grad() to avoid affecting training resources.
        """
        print(f"[TopK Callback] Generating with PyTorch (batch_size={self.eval_batch_size}, {len(prompts)} prompts)")

        # Store original training state
        was_training = model.training

        # Switch to eval mode
        model.eval()
        device = next(model.parameters()).device

        # Compile model for faster generation (PyTorch 2.0+)
        if self.use_compile and self.compiled_model is None:
            print("[TopK Callback] Compiling model for faster eval (first run only)...")
            try:
                self.compiled_model = torch.compile(model, mode="reduce-overhead")
                print("[TopK Callback] ✓ Model compiled successfully")
            except Exception as e:
                print(f"[TopK Callback] Warning: Could not compile model: {e}")
                self.compiled_model = model

        eval_model = self.compiled_model if self.use_compile and self.compiled_model else model
        completions = []

        with torch.no_grad():
            for i in range(0, len(prompts), self.eval_batch_size):
                batch_prompts = prompts[i:i + self.eval_batch_size]

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=384,  # Match training max_prompt_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = eval_model.generate(
                    **inputs,
                    max_new_tokens=400,  # Reduced from 768 (GSM8K rarely needs more)
                    do_sample=False,  # Greedy decoding for deterministic eval
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for efficiency
                    early_stopping=True,  # Stop at EOS token
                )

                for j, output_ids in enumerate(outputs):
                    prompt_len = inputs["input_ids"][j].shape[0]
                    completion_ids = output_ids[prompt_len:]
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    completions.append(completion_text)

                # Progress indicator for large eval sets
                if (i + self.eval_batch_size) % (self.eval_batch_size * 5) == 0:
                    print(f"[TopK Callback] Generated {min(i + self.eval_batch_size, len(prompts))}/{len(prompts)} completions")

        # Restore original training state
        if was_training:
            model.train()

        print(f"[TopK Callback] Generation complete ({len(completions)} samples)")
        return completions


    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of each training step."""
        current_step = state.global_step

        # Check if it's time to evaluate
        if current_step % self.eval_every_n_steps != 0:
            return control

        # Avoid re-evaluating same step
        if current_step == self.last_eval_step:
            return control

        self.last_eval_step = current_step

        # Load dataset if not loaded
        if self.eval_dataset is None:
            self._load_eval_dataset()

        if self.eval_dataset is None:
            return control

        # Use stored trainer and tokenizer references
        if self.trainer is None or self.tokenizer is None:
            print("[TopK Callback] ERROR: trainer/tokenizer not set. Pass them during callback initialization.")
            return control

        # Run custom evaluation
        print(f"\n[TopK Callback] Step {current_step}: Running custom held-out evaluation...")

        metrics = self._run_custom_eval(trainer=self.trainer, tokenizer=self.tokenizer)

        if not metrics:
            return control

        # Log metrics to WandB
        wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        wandb_metrics["eval/step"] = current_step

        if hasattr(state, "log_history"):
            state.log_history.append(wandb_metrics)

        if self.trainer is not None:
            self.trainer.log(wandb_metrics)

        main_score = metrics[self.metric_name]
        print(f"[TopK Callback] Step {current_step}: {self.metric_name} = {main_score:.4f}")

        # Track in eval history
        eval_entry = {
            "step": current_step,
            "score": main_score,
            "metrics": metrics,
            "checkpoint": f"checkpoint-{current_step}",
        }
        self.eval_history.append(eval_entry)

        # Track best score
        if main_score > self.best_score:
            self.best_score = main_score
            self.best_step = current_step
            print(f"[TopK Callback] NEW BEST: {main_score:.4f} at step {current_step}")

        # Manage top-K checkpoint saving and deletion
        output_dir = self.output_dir or args.output_dir
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{current_step}")

        print(f"[TopK Callback] Managing checkpoint-{current_step} with score {main_score:.4f}")

        # Save checkpoint using trainer
        if self.trainer is not None:
            print(f"[TopK Callback] Saving checkpoint-{current_step} to {checkpoint_path}")
            self.trainer.save_model(checkpoint_path)
            # Also save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(checkpoint_path)
            print(f"[TopK Callback] ✓ Checkpoint saved")

        # Add to heap
        heapq.heappush(self.heap, (main_score, current_step, checkpoint_path))

        # If heap size exceeds K, remove worst checkpoint
        if len(self.heap) > self.k:
            worst_score, worst_step, worst_path = heapq.heappop(self.heap)

            print(f"[TopK Callback] Removing checkpoint-{worst_step} (score: {worst_score:.4f})")
            print(f"[TopK Callback] Keeping top {self.k}: steps {sorted([s for _, s, _ in self.heap])}")

            if os.path.exists(worst_path):
                try:
                    shutil.rmtree(worst_path)
                    print(f"[TopK Callback] ✓ Deleted: {worst_path}")
                except Exception as e:
                    print(f"[TopK Callback] Failed to delete {worst_path}: {e}")

        # Save eval history
        self._save_eval_history()

        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called after a checkpoint is saved."""
        return control

    def _save_eval_history(self):
        """Save evaluation history and heap state to JSON."""
        if self.output_dir is None:
            return

        history_path = os.path.join(self.output_dir, "topk_eval_history.json")

        heap_info = [
            {"step": step, "score": score, "checkpoint": path}
            for score, step, path in sorted(self.heap, reverse=True)
        ]

        data = {
            "best_score": self.best_score,
            "best_step": self.best_step,
            "best_checkpoint": f"checkpoint-{self.best_step}" if self.best_step else None,
            "top_k_checkpoints": heap_info,
            "eval_history": self.eval_history,
            "config": {
                "k": self.k,
                "eval_every_n_steps": self.eval_every_n_steps,
                "metric_name": self.metric_name,
                "reward_function": self.reward_function_name,
            }
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(data, f, indent=2)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Print summary at end of training."""
        print(f"\n{'='*70}")
        print(f"[TopK Checkpoint Callback] TRAINING COMPLETE")
        print(f"{'='*70}")

        if self.best_step is not None:
            print(f"Best {self.metric_name}: {self.best_score:.4f}")
            print(f"Best Step: {self.best_step}")
            print(f"Best Checkpoint: checkpoint-{self.best_step}")

        if len(self.heap) > 0:
            print(f"\nTop-{self.k} Checkpoints (sorted by score):")
            for score, step, path in sorted(self.heap, reverse=True):
                print(f"  Step {step:5d}: {score:.4f} -> {path}")
        else:
            print("\nNo checkpoints in top-K heap.")

        history_path = os.path.join(self.output_dir or args.output_dir, "topk_eval_history.json")
        print(f"\nEvaluation history saved to: {history_path}")
        print(f"{'='*70}\n")

        return control


class GSMTopKCheckpointCallback(BaseTopKCheckpointCallback):
    """
    Top-K checkpoint callback for GSM8K-style datasets.

    Uses simple accuracy as the ranking metric.
    No difficulty weighting - just percentage correct.
    """

    def __init__(
        self,
        k: int = 5,
        eval_dataset_path: str = None,
        eval_every_n_steps: int = 50,
        reward_function: str = "ppo_binary",
        source_type: str = "local",
        output_dir: str = None,
        trainer = None,
        tokenizer = None,
        eval_batch_size: int = 8,
        use_compile: bool = False,
    ):
        super().__init__(
            k=k,
            eval_dataset_path=eval_dataset_path,
            eval_every_n_steps=eval_every_n_steps,
            metric_name="held_out_accuracy",
            reward_function=reward_function,
            source_type=source_type,
            output_dir=output_dir,
            trainer=trainer,
            tokenizer=tokenizer,
            eval_batch_size=eval_batch_size,
            use_compile=use_compile,
        )

    def _compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: List[str],
        metadata: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute simple accuracy for GSM8K.

        Uses the reward function to check correctness (handles answer extraction).
        """
        print(f"[GSM TopK] Computing rewards for {len(completions)} completions...")

        # Use reward function for correctness checking
        rewards = self.reward_fn(
            prompts=prompts,
            completions=completions,
            answer=ground_truths
        )

        # Count correct (reward > 0.5 threshold for binary rewards)
        correct = sum(1 for r in rewards if r > 0.5)
        total = len(rewards)
        accuracy = correct / total if total > 0 else 0.0

        # Compute format consistency (valid think tags and boxed answers)
        from src.rlvr.math_verifier import check_format_quality, extract_answer_typed

        valid_format_count = 0
        for completion in completions:
            fmt = check_format_quality(completion)
            _, _, _, used_boxed = extract_answer_typed(completion)

            # Valid format: proper think tags OR boxed answer (lenient for eval)
            has_valid_think = fmt["has_think_tags"] and not fmt["has_partial_tags"]
            has_valid_format = has_valid_think or used_boxed

            if has_valid_format:
                valid_format_count += 1

        format_consistency = valid_format_count / total if total > 0 else 0.0

        metrics = {
            "held_out_accuracy": accuracy,
            "held_out_correct": correct,
            "held_out_total": total,
            "format_consistency": format_consistency,
        }

        print(f"[GSM TopK] Results: {correct}/{total} correct ({accuracy*100:.1f}%), format: {format_consistency*100:.1f}%")

        return metrics


class MixedDifficultyTopKCheckpointCallback(BaseTopKCheckpointCallback):
    """
    Top-K checkpoint callback for mixed-difficulty datasets (GSM8K + MATH).

    Uses difficulty-weighted accuracy as the ranking metric.
    Harder problems contribute more to the final score.
    """

    def __init__(
        self,
        k: int = 5,
        eval_dataset_path: str = None,
        eval_every_n_steps: int = 50,
        difficulty_weights: Optional[Dict[str, float]] = None,
        reward_function: str = "ppo_binary",
        source_type: str = "local",
        output_dir: str = None,
        trainer = None,
        tokenizer = None,
        eval_batch_size: int = 8,
        use_compile: bool = False,
    ):
        super().__init__(
            k=k,
            eval_dataset_path=eval_dataset_path,
            eval_every_n_steps=eval_every_n_steps,
            metric_name="held_out_weighted_accuracy",
            reward_function=reward_function,
            source_type=source_type,
            output_dir=output_dir,
            trainer=trainer,
            tokenizer=tokenizer,
            eval_batch_size=eval_batch_size,
            use_compile=use_compile,
        )
        self.difficulty_weights = difficulty_weights or {"default": 1.0}

    def _compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: List[str],
        metadata: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute difficulty-weighted accuracy.

        Harder problems get higher weights in the final score.
        """
        print(f"[Mixed TopK] Computing rewards for {len(completions)} completions...")

        # Use reward function for correctness checking
        rewards = self.reward_fn(
            prompts=prompts,
            completions=completions,
            answer=ground_truths
        )

        # Aggregate by difficulty
        difficulty_scores = {}
        for i, (reward, meta) in enumerate(zip(rewards, metadata)):
            difficulty = meta.get("difficulty", "default")
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            difficulty_scores[difficulty].append(reward)

        # Compute weighted accuracy
        weighted_score = 0.0
        total_weight = 0.0

        for difficulty, diff_rewards in difficulty_scores.items():
            weight = self.difficulty_weights.get(difficulty, 1.0)
            avg_reward = sum(diff_rewards) / len(diff_rewards)
            weighted_score += avg_reward * weight
            total_weight += weight

        weighted_accuracy = weighted_score / total_weight if total_weight > 0 else 0.0

        # Per-difficulty accuracies
        difficulty_accuracies = {
            f"held_out_{diff}_acc": sum(r > 0.5 for r in diff_rewards) / len(diff_rewards)
            for diff, diff_rewards in difficulty_scores.items()
        }

        # Overall unweighted accuracy
        total_correct = sum(1 for r in rewards if r > 0.5)
        raw_accuracy = total_correct / len(rewards) if len(rewards) > 0 else 0.0

        # Compute format consistency
        from src.rlvr.math_verifier import check_format_quality, extract_answer_typed

        valid_format_count = 0
        for completion in completions:
            fmt = check_format_quality(completion)
            _, _, _, used_boxed = extract_answer_typed(completion)

            # Valid format: proper think tags OR boxed answer
            has_valid_think = fmt["has_think_tags"] and not fmt["has_partial_tags"]
            has_valid_format = has_valid_think or used_boxed

            if has_valid_format:
                valid_format_count += 1

        format_consistency = valid_format_count / len(rewards) if len(rewards) > 0 else 0.0

        metrics = {
            "held_out_weighted_accuracy": weighted_accuracy,
            "held_out_raw_accuracy": raw_accuracy,
            "held_out_total": len(rewards),
            "format_consistency": format_consistency,
            **difficulty_accuracies,
        }

        print(f"[Mixed TopK] Weighted: {weighted_accuracy:.4f}, Raw: {raw_accuracy:.4f}, Format: {format_consistency:.4f}")

        return metrics


# Alias for backward compatibility
TopKCheckpointCallback = GSMTopKCheckpointCallback

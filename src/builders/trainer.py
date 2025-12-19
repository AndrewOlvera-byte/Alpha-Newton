import wandb
from src.core.registry import register
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
import torch
from typing import Any, Dict, List, Optional


class DataCollatorForCausalLMWithLabels:
    """
    Pads a batch of tokenized CausalLM features while preserving precomputed `labels`.

    This is critical for SFT when we pre-mask prompt tokens with -100 and only train
    on assistant tokens. TRL/Transformers' default LM collators often overwrite labels.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]):
        labels = []
        features_wo_labels = []
        for f in features:
            # Keep original labels (may be list[int] or torch.Tensor)
            lab = f.get("labels")
            if isinstance(lab, torch.Tensor):
                lab = lab.tolist()
            labels.append(lab)
            features_wo_labels.append({k: v for k, v in f.items() if k != "labels"})

        batch = self.tokenizer.pad(
            features_wo_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        pad_side = getattr(self.tokenizer, "padding_side", "right")
        padded_labels = []
        for lab in labels:
            lab = lab or []
            if len(lab) > max_len:
                lab = lab[:max_len]
            pad_len = max_len - len(lab)
            if pad_side == "left":
                padded_labels.append(([-100] * pad_len) + lab)
            else:
                padded_labels.append(lab + ([-100] * pad_len))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


@register("trainer", "trl_sft")
def build_trl_sft_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg):
    """
    Build SFT trainer with WandB integration

    WandB logs all metrics automatically. Model checkpoints save to output_dir locally
    """
    # Initialize WandB run
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["run_name"],
        tags=wandb_cfg.get("tags", []),
        config={
            "model": training_cfg.get("output_dir", "").split("/")[-1],
            "learning_rate": training_cfg.get("learning_rate"),
            "batch_size": training_cfg.get("per_device_train_batch_size"),
            "max_steps": training_cfg.get("max_steps"),
        },
    )

    # Enable WandB reporting in trainer
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    # Remove DPO-specific params (beta is only for DPO)
    training_cfg.pop("beta", None)

    training_args = SFTConfig(**training_cfg)

    # IMPORTANT: preserve prompt-masked labels coming from the dataset builder.
    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Renamed from 'tokenizer' in TRL 0.12.0+
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
        data_collator=data_collator,
    )
    return trainer


@register("trainer", "trl_dpo")
def build_trl_dpo_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg, ref_model=None):
    """
    Build DPO (Direct Preference Optimization) trainer

    DPO directly optimizes on preference pairs without needing a reward model
    Works with datasets containing (prompt, chosen, rejected) tuples

    Args:
        model: Policy model to train
        tokenizer: Tokenizer
        dataset: Dict with 'train' and 'eval' datasets containing preference pairs
        training_cfg: Training configuration (must include 'beta' for DPO temperature)
        wandb_cfg: WandB configuration
        ref_model: Reference model (optional - will use frozen copy of model if None)
    """
    # Initialize WandB run
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["run_name"],
        tags=wandb_cfg.get("tags", []),
        config={
            "model": training_cfg.get("output_dir", "").split("/")[-1],
            "learning_rate": training_cfg.get("learning_rate"),
            "batch_size": training_cfg.get("per_device_train_batch_size"),
            "max_steps": training_cfg.get("max_steps"),
            "beta": training_cfg.get("beta", 0.1),
        },
    )

    # Enable WandB reporting
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]

    # Extract DPO-specific parameters
    beta = training_cfg.pop("beta", 0.1)
    max_length = training_cfg.pop("max_length", 2048)
    max_prompt_length = training_cfg.pop("max_prompt_length", max_length // 3)
    # Precompute ref log probs upfront - major speedup (avoids ref model forward pass each step)
    precompute_ref_log_probs = training_cfg.pop("precompute_ref_log_probs", True)

    training_args = DPOConfig(
        **training_cfg,
        beta=beta,  # DPO temperature (higher = more conservative, typical: 0.1-0.5)
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        # Precompute reference log probs once upfront instead of every training step
        # This is the main DPO speedup - eliminates ref model forward pass during training
        precompute_ref_log_probs=precompute_ref_log_probs,
        # Padding settings for efficient batching
        padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        truncation_mode="keep_end",  # Keep end of sequences (most relevant for responses)
    )

    if precompute_ref_log_probs:
        print("[DPO Trainer] Will precompute reference log probabilities (one-time cost, faster training)")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # If None, DPOTrainer creates a frozen copy automatically
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
    )
    return trainer


@register("trainer", "trl_grpo")
def build_trl_grpo_trainer(model, tokenizer, dataset, training_cfg, grpo_cfg, wandb_cfg, reward_funcs):
    """
    Build GRPO (Group Relative Policy Optimization) trainer for RLVR.
    
    GRPO is an online RL algorithm that:
    1. Generates multiple completions per prompt
    2. Scores them with reward functions
    3. Uses group-relative advantages (no value model needed)
    
    Args:
        model: Policy model to train
        tokenizer: Tokenizer
        dataset: Dict with 'train' and 'eval' datasets (must have 'prompt' column)
        training_cfg: Standard training config (lr, batch size, etc.)
        grpo_cfg: GRPO-specific config (num_generations, max_completion_length, etc.)
        wandb_cfg: WandB configuration
        reward_funcs: Reward function(s) - callable or list of callables
    """
    # Initialize WandB
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["run_name"],
        tags=wandb_cfg.get("tags", []),
        config={
            "model": training_cfg.get("output_dir", "").split("/")[-1],
            "learning_rate": training_cfg.get("learning_rate"),
            "batch_size": training_cfg.get("per_device_train_batch_size"),
            "max_steps": training_cfg.get("max_steps"),
            "num_generations": grpo_cfg.get("num_generations", 4),
        },
    )
    
    # Build training args - merge training_cfg with GRPO-specific settings
    training_cfg = {**training_cfg}
    training_cfg["report_to"] = ["wandb"]
    
    # Remove non-GRPO params
    training_cfg.pop("beta", None)
    
    # GRPO config with generation settings
    config = GRPOConfig(
        **training_cfg,
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 1024),
        # Generation settings
        temperature=grpo_cfg.get("temperature", 0.7),
        # Optional vLLM for faster generation
        use_vllm=grpo_cfg.get("use_vllm", False),
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        reward_funcs=reward_funcs,
        args=config,
    )
    
    return trainer

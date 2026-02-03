"""
RLVR (Reinforcement Learning with Verifiable Rewards) Training

Usage:
    python -m src.entrypoints.train_rlvr --exp qwen3_600M_rlvr_math
"""

import argparse
from src.core.config import Config
from src.core.registry import build

import src.builders.model
import src.builders.tokenizer
import src.builders.data
import src.builders.data_thinking
import src.builders.trainer

import glob
import os
import json
from src.rlvr.math_verifier import get_reward_function
from transformers import TrainerCallback
from src.rlvr.rl_callbacks import GSMTopKCheckpointCallback, MixedDifficultyTopKCheckpointCallback

def main(exp_name: str):
    cfg = Config.from_experiment(exp_name)

    print(f"[RLVR] Run: {cfg.run['name']}")
    print(f"[RLVR] Mode: {cfg.run['mode']}")
    print(f"[RLVR] Model: {cfg.model.get('id', 'N/A')}")
    print(f"[RLVR] Dataset: {cfg.data.get('train_path', 'N/A')}")
    print(f"[RLVR] Output: {cfg.training['output_dir']}")
    print()

    tokenizer = build("tokenizer", **cfg.tokenizer)
    model = build("model", **cfg.model)

    grpo_cfg = getattr(cfg, 'grpo', {})
    diverse_config = grpo_cfg.get('diverse_sampling', {})
    diverse_enabled = diverse_config.get('enabled', False)

    dataset = build("data", tokenizer=tokenizer, diverse_sampling=diverse_config, **cfg.data)

    reward_function_name = grpo_cfg.get('reward_function', 'dapo_advanced')
    reward_fn = get_reward_function(reward_function_name)

    print(f"[RLVR] GRPO Config:")
    print(f"  - reward_function: {reward_function_name}")
    print(f"  - num_generations: {grpo_cfg.get('num_generations', 4)}")
    print(f"  - max_completion_length: {grpo_cfg.get('max_completion_length', 512)}")
    print(f"  - use_vllm: {grpo_cfg.get('use_vllm', False)}")

    if diverse_enabled:
        print(f"  - diverse_sampling: ENABLED")
        sampling_configs = diverse_config.get('sampling_configs', [])
        print(f"    - sampling_configs: {len(sampling_configs)} discrete configs")
        for i, sampling_cfg in enumerate(sampling_configs):
            print(f"      [{i}] temp={sampling_cfg.get('temperature', 0.7):.2f}, top_p={sampling_cfg.get('top_p', 0.95):.2f}")
        print(f"    - system_prompts: {len(diverse_config.get('system_prompts', []))} modes")
    else:
        print(f"  - diverse_sampling: DISABLED")
        print(f"    - temperature: {grpo_cfg.get('temperature', 0.7)} (static)")
        print(f"    - top_p: {grpo_cfg.get('top_p', 1.0)} (static)")

    print()

    trainer_type = "trl_grpo_diverse" if diverse_enabled else "trl_grpo"
    print(f"[RLVR] Using trainer type: {trainer_type}")

    trainer = build(
        "trainer",
        type=trainer_type,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=cfg.training,
        grpo_cfg=grpo_cfg,
        wandb_cfg=cfg.wandb,
        reward_funcs=reward_fn,
    )

    output_dir = cfg.training['output_dir']

    if hasattr(cfg, 'topk_eval') and cfg.topk_eval.get('enabled', False):

        eval_dataset_path = cfg.topk_eval.get('eval_dataset_path')
        if eval_dataset_path is None and hasattr(cfg, 'data'):
            eval_dataset_path = cfg.data.get('eval_path')

        if eval_dataset_path is None:
            print("[RLVR] Warning: topk_eval enabled but no eval_dataset_path provided")
            print("[RLVR] Skipping TopK callback. Set topk_eval.eval_dataset_path or data.eval_path")
        else:
            callback_mode = cfg.topk_eval.get('mode', 'gsm')

            if callback_mode == 'gsm':
                topk_callback = GSMTopKCheckpointCallback(
                    k=cfg.topk_eval.get('k', 5),
                    eval_dataset_path=eval_dataset_path,
                    eval_every_n_steps=cfg.topk_eval.get('eval_every_n_steps', 50),
                    reward_function=cfg.topk_eval.get('reward_function', reward_function_name),
                    source_type=cfg.topk_eval.get('source_type', 'local'),
                    output_dir=output_dir,
                    trainer=trainer,
                    tokenizer=tokenizer,
                    eval_batch_size=cfg.topk_eval.get('eval_batch_size', 8),
                    use_compile=cfg.topk_eval.get('use_compile', False),
                )
                print(f"[RLVR] Added GSMTopKCheckpointCallback (simple accuracy)")
            elif callback_mode == 'mixed':
                topk_callback = MixedDifficultyTopKCheckpointCallback(
                    k=cfg.topk_eval.get('k', 5),
                    eval_dataset_path=eval_dataset_path,
                    eval_every_n_steps=cfg.topk_eval.get('eval_every_n_steps', 50),
                    difficulty_weights=cfg.topk_eval.get('difficulty_weights', {"default": 1.0}),
                    reward_function=cfg.topk_eval.get('reward_function', reward_function_name),
                    source_type=cfg.topk_eval.get('source_type', 'local'),
                    output_dir=output_dir,
                    trainer=trainer,
                    tokenizer=tokenizer,
                    eval_batch_size=cfg.topk_eval.get('eval_batch_size', 8),
                    use_compile=cfg.topk_eval.get('use_compile', False),
                )
                print(f"[RLVR] Added MixedDifficultyTopKCheckpointCallback (weighted accuracy)")
                print(f"[RLVR]   - Difficulty weights: {cfg.topk_eval.get('difficulty_weights')}")
            else:
                raise ValueError(f"Unknown topk_eval.mode: {callback_mode}. Use 'gsm' or 'mixed'")

            trainer.add_callback(topk_callback)
            print(f"[RLVR]   - K: {cfg.topk_eval.get('k', 5)} (keep top {cfg.topk_eval.get('k', 5)} checkpoints)")
            print(f"[RLVR]   - Eval dataset: {eval_dataset_path}")
            print(f"[RLVR]   - Eval every: {cfg.topk_eval.get('eval_every_n_steps', 50)} steps")
            print(f"[RLVR]   - Reward function: {cfg.topk_eval.get('reward_function', reward_function_name)}")
            print()

    checkpoint_path = None
    if os.path.exists(output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint_path = latest_checkpoint
            step_num = latest_checkpoint.split("-")[-1]
            print(f"[RLVR] Found checkpoint: {latest_checkpoint}")
            print(f"[RLVR] Resuming training from step {step_num}")
        else:
            print("[RLVR] Starting training from scratch (no checkpoints found)")
    else:
        print("[RLVR] Starting training from scratch (output directory created)")

    print(f"[RLVR] Training will save checkpoints every {cfg.training.get('save_steps', 500)} steps")
    print()

    print("[RLVR] Starting online GRPO training...")
    trainer.train(resume_from_checkpoint=checkpoint_path)

    print(f"\n[RLVR] Training completed!")
    print("[RLVR] Check eval_reward_history.json to see which checkpoint performed best.\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLVR model with math verification")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (loads from configs/exp/{exp}.yaml)",
    )
    
    args = parser.parse_args()
    main(exp_name=args.exp)

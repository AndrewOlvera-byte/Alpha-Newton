"""
RLVR (Reinforcement Learning with Verifiable Rewards) Training

Online RL training with GRPOTrainer:
- Model generates completions during training
- Math verifier scores correctness (1.0 / 0.0)
- Model learns from its own generations via group-relative policy optimization

Usage:
    python -m src.entrypoints.train_rlvr --exp qwen3_600M_rlvr_math
"""

import argparse
from src.core.config import Config
from src.core.registry import build

import src.builders.model
import src.builders.tokenizer
import src.builders.data
import src.builders.trainer

from src.rlvr import math_reward_fn


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
    dataset = build("data", tokenizer=tokenizer, **cfg.data)
    
    grpo_cfg = getattr(cfg, 'grpo', {})
    
    print(f"[RLVR] GRPO Config:")
    print(f"  - num_generations: {grpo_cfg.get('num_generations', 4)}")
    print(f"  - max_completion_length: {grpo_cfg.get('max_completion_length', 512)}")
    print(f"  - temperature: {grpo_cfg.get('temperature', 0.7)}")
    print(f"  - use_vllm: {grpo_cfg.get('use_vllm', False)}")
    print()
    
    trainer = build(
        "trainer",
        type="trl_grpo",
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=cfg.training,
        grpo_cfg=grpo_cfg,
        wandb_cfg=cfg.wandb,
        reward_funcs=math_reward_fn,
    )
    
    print("[RLVR] Starting online GRPO training...")
    trainer.train()
    
    print(f"\n[RLVR] Saving final model to: {cfg.training['output_dir']}")
    trainer.save_model(cfg.training['output_dir'])
    tokenizer.save_pretrained(cfg.training['output_dir'])
    print("[RLVR] Model and tokenizer saved successfully")


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

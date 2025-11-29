import argparse
from src.core.config import Config
from src.core.registry import build

import src.builders.model
import src.builders.tokenizer
import src.builders.data
import src.builders.trainer


def main(exp_name: str):
    """
    Train SFT model from experiment

    Loads: configs/base/common.yaml + configs/exp/{exp_name}.yaml
    """
    cfg = Config.from_experiment(exp_name)

    print(f"[Config] Run: {cfg.run['name']}")
    print(f"[Config] Mode: {cfg.run['mode']}")
    print(f"[Config] Model: {cfg.model.get('id', 'N/A')}")
    print(f"[Config] Output: {cfg.training['output_dir']}")
    print()

    # Build components via registry
    tokenizer = build("tokenizer", **cfg.tokenizer)
    model = build("model", **cfg.model)
    dataset = build("data", tokenizer=tokenizer, **cfg.data)

    # === Optimization Verification ===
    print("=" * 60)
    print("[OPTIMIZATION CHECK]")
    print("=" * 60)

    # Check Flash Attention 2
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"✓ Attention Implementation: {attn_impl}")
    if attn_impl == "flash_attention_2":
        print("  → Flash Attention 2 is ENABLED")
    else:
        print(f"  → WARNING: Flash Attention 2 NOT enabled (using {attn_impl})")

    # Check model dtype
    model_dtype = next(model.parameters()).dtype
    print(f"✓ Model dtype: {model_dtype}")

    # Check bf16 training
    bf16_enabled = cfg.training.get('bf16', False)
    print(f"✓ bf16 training: {bf16_enabled}")

    # Check torch.compile
    torch_compile = cfg.training.get('torch_compile', False)
    print(f"✓ torch.compile: {torch_compile}")

    # Check gradient checkpointing
    grad_ckpt = cfg.model.get('gradient_checkpointing', False)
    print(f"✓ Gradient checkpointing: {grad_ckpt}")

    print("=" * 60)
    print()

    trainer = build(
        "trainer",
        type="trl_sft",
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=cfg.training,
        wandb_cfg=cfg.wandb,
    )

    # Train model
    trainer.train()

    # Save final model to output directory root
    print(f"\n[Saving] Final model to: {cfg.training['output_dir']}")
    trainer.save_model(cfg.training['output_dir'])
    tokenizer.save_pretrained(cfg.training['output_dir'])
    print(f"[Saved] Model and tokenizer saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT model")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (loads from configs/exp/{exp}.yaml)",
    )

    args = parser.parse_args()
    main(exp_name=args.exp)

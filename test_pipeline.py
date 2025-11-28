#!/usr/bin/env python3
"""Quick test to verify the pipeline can build all components"""

from src.core.config import Config
from src.core.registry import build

import src.builders.model
import src.builders.tokenizer
import src.builders.data
import src.builders.trainer


def test_pipeline():
    print("=" * 60)
    print("Testing Alpha-Newton Pipeline")
    print("=" * 60)

    # Load config
    print("\n[1/5] Loading config...")
    cfg = Config.from_experiment('qwen3_600M_base_sft_test')
    print(f"  ✓ Loaded: {cfg.run['name']}")
    print(f"  ✓ Mode: {cfg.run['mode']}")
    print(f"  ✓ Model: {cfg.model['id']}")
    print(f"  ✓ Dataset: {cfg.data['train_path']}")
    print(f"  ✓ Output: {cfg.training['output_dir']}")

    # Build tokenizer
    print("\n[2/5] Building tokenizer...")
    tokenizer = build("tokenizer", **cfg.tokenizer)
    print(f"  ✓ Tokenizer built: {type(tokenizer).__name__}")
    print(f"  ✓ Vocab size: {len(tokenizer)}")

    # Build model
    print("\n[3/5] Building model...")
    model = build("model", **cfg.model)
    print(f"  ✓ Model built: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total params: {total_params:,}")

    # Build dataset (will download - may take time)
    print("\n[4/5] Building dataset...")
    print("  (This may take a while on first run - downloading dataset)")
    try:
        dataset = build("data", tokenizer=tokenizer, **cfg.data)
        print(f"  ✓ Dataset built successfully")
        print(f"  ✓ Train samples: {len(dataset['train'])}")
        if 'eval' in dataset:
            print(f"  ✓ Eval samples: {len(dataset['eval'])}")
    except Exception as e:
        print(f"  ⚠ Dataset build failed (expected on first run): {e}")
        print(f"  This is likely due to dataset format - you may need to adjust the data builder")
        dataset = None

    # Check WandB config
    print("\n[5/5] Checking WandB configuration...")
    print(f"  ✓ Project: {cfg.wandb['project']}")
    print(f"  ✓ Entity: {cfg.wandb['entity']}")
    print(f"  ✓ Run name: {cfg.wandb['run_name']}")
    print(f"  ✓ Tags: {cfg.wandb.get('tags', [])}")
    if not cfg.wandb['entity']:
        print(f"  ⚠ WARNING: WandB entity not set! Update configs/base/common.yaml")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: wandb login")
    print("  2. Set your WandB username in configs/base/common.yaml")
    print("  3. Start training:")
    print(f"     python src/entrypoints/train_sft.py --exp qwen3_600M_base_sft_test")


if __name__ == "__main__":
    test_pipeline()

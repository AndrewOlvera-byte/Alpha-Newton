import argparse
from src.core.config import Config
from src.core.registry import build

import src.builders.model
import src.builders.tokenizer
import src.builders.data
import src.builders.trainer


def main(exp_name: str):
    """
    Train DPO (Direct Preference Optimization) model from experiment

    Loads: configs/base/common.yaml + configs/exp/{exp_name}.yaml
    """
    cfg = Config.from_experiment(exp_name)

    print(f"[Config] Run: {cfg.run['name']}")
    print(f"[Config] Mode: {cfg.run['mode']}")
    print(f"[Config] Model: {cfg.model.get('id', 'N/A')}")
    print(f"[Config] Dataset: {cfg.data['train_path']}")
    print(f"[Config] Output: {cfg.training['output_dir']}")
    print(f"[Config] Beta: {cfg.training.get('beta', 0.1)}")
    print()

    # Build components via registry
    tokenizer = build("tokenizer", **cfg.tokenizer)
    model = build("model", **cfg.model)
    dataset = build("data", tokenizer=tokenizer, **cfg.data)

    trainer = build(
        "trainer",
        type="trl_dpo",
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
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (loads from configs/exp/{exp}.yaml)",
    )

    args = parser.parse_args()
    main(exp_name=args.exp)

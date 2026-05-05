"""
Behavioral Cloning (BC) training entrypoint for robotics.

Usage:
    python -m src.entrypoints.train_bc --exp bc_lift_ph
    python -m src.entrypoints.train_bc --exp bc_lift_ph --test
"""
import argparse
import torch

from src.core.config import Config
from src.core.registry import build

# Register builders
import src.robotics.data
import src.robotics.bc_trainer
import src.robotics.models.VLA_like
import src.robotics.models.VLA_MLP
import src.robotics.models.VLA_Gaussian
import src.robotics.models.flightmare
import scripts.flightmare_bc.dataset


def test_dataset(cfg: Config):
    """Validate dataset loading, config, and model construction."""
    print("=" * 60)
    print("[TEST] Dataset + Config + Model Validation")
    print("=" * 60)

    dataset = build("data", **cfg.data)
    train_ds = dataset["train"]
    eval_ds = dataset["eval"]

    print(f"\nTrain: {len(train_ds)} | Eval: {len(eval_ds)}")

    sample = train_ds[0]
    print(f"\n--- Sample 0 ---")
    print(f"Action shape: {sample['action'].shape} dtype: {sample['action'].dtype}")
    print(f"State shape:  {sample['state'].shape}")
    print(f"Prev actions: {sample['prev_actions'].shape}")
    for k, v in sample["images"].items():
        print(f"Image/{k}: {v.shape} ({v.dtype})")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\n--- Batch (bs=2) ---")
    print(f"Action: {batch['action'].shape}")
    print(f"State:  {batch['state'].shape}")
    for k, v in batch["images"].items():
        print(f"Image/{k}: {v.shape}")

    # Test model construction
    robotics_cfg = cfg.robotics or {}
    arch_cfg = robotics_cfg.get("architecture", {})
    if arch_cfg.get("type"):
        print(f"\n--- Model ---")
        model = build("architecture", **arch_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")

        # Test forward pass (CPU, small batch)
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()

        test_batch = {
            "images": {k: v[:1].to(device) for k, v in batch["images"].items()},
            "state": batch["state"][:1].to(device),
            "prev_actions": batch["prev_actions"][:1].to(device),
            "action": batch["action"][:1].to(device),
        }
        # Include task_id so task embeddings are exercised (multi-task models).
        if "task_id" in batch:
            test_batch["task_id"] = batch["task_id"][:1].to(device)

        print("Testing forward pass...")
        with torch.no_grad():
            output = model(test_batch)
        print(f"Loss: {output['loss'].item():.4f}")

        print("Testing predict (inference)...")
        with torch.no_grad():
            action = model.predict(test_batch, num_steps=3)
        print(f"Predicted action: {action.shape} range=[{action.min():.3f}, {action.max():.3f}]")
    else:
        print("\n[skip] No architecture.type set — skipping model test")

    print(f"\n{'=' * 60}")
    print("[TEST] All checks passed!")
    print(f"{'=' * 60}")


def main(exp_name: str, test_only: bool = False):
    cfg = Config.from_experiment(exp_name)
    if cfg.run.get("mode") != "bc":
        raise ValueError(
            f"train_bc.py only accepts run.mode='bc'. "
            f"Got mode={cfg.run.get('mode')!r} for exp={exp_name!r}."
        )
    if (cfg.robotics or {}).get("ppo") is not None:
        raise ValueError(
            f"train_bc.py refuses PPO configs. Use train_ppo.py for exp={exp_name!r}."
        )

    print(f"[Config] Run:   {cfg.run['name']}")
    print(f"[Config] Mode:  {cfg.run['mode']}")
    data_type = cfg.data.get("type", "?")
    print(f"[Config] Data:  type={data_type}")
    if data_type.startswith("robomimic"):
        print(f"[Config]        tasks={cfg.data.get('tasks', [])}  image_size={cfg.data.get('image_size', 'unset')}")
    elif data_type.startswith("flightmare"):
        print(f"[Config]        data_dir={cfg.data.get('data_dir', '?')}  "
              f"action={cfg.data.get('action_type', '?')}  "
              f"include_mission={cfg.data.get('include_mission', True)}")
    arch_type = (cfg.robotics or {}).get("architecture", {}).get("type", "?")
    print(f"[Config] Arch:  {arch_type}")
    print(f"[Config] Output: {cfg.training['output_dir']}")
    print()

    if test_only:
        test_dataset(cfg)
        return

    # Build components via registry
    dataset = build("data", **cfg.data)

    robotics_cfg = cfg.robotics or {}
    arch_cfg = robotics_cfg.get("architecture", {})
    if not arch_cfg.get("type"):
        raise ValueError("robotics.architecture.type must be set in config")

    model = build("architecture", **arch_cfg)

    trainer_type = cfg.training.get("trainer_type", "bc_flow_matching_robosuite")
    trainer = build(
        "trainer",
        type=trainer_type,
        model=model,
        dataset=dataset,
        training_cfg=cfg.training,
        robotics_cfg=robotics_cfg,
        wandb_cfg=cfg.wandb,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC policy")
    parser.add_argument("--exp", type=str, required=True, help="Experiment config name")
    parser.add_argument("--test", action="store_true", help="Test dataset + model only")
    args = parser.parse_args()
    main(exp_name=args.exp, test_only=args.test)

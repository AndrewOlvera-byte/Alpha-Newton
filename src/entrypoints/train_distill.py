"""DAgger-style distillation training entrypoint for Flightmare policies."""
from __future__ import annotations

import argparse

import torch

from src.core.config import Config
from src.core.registry import build

# Register builders.
import src.robotics.bc_trainer  # noqa: F401
import src.robotics.data  # noqa: F401
import src.robotics.distill_trainer  # noqa: F401
import src.robotics.models.flightmare  # noqa: F401
import scripts.flightmare_bc.dataset  # noqa: F401


def validate_distill_config(cfg: Config, exp_name: str = "<config>") -> None:
    if cfg.run.get("mode") != "distill":
        raise ValueError(
            f"train_distill.py only accepts run.mode='distill'. "
            f"Got mode={cfg.run.get('mode')!r} for exp={exp_name!r}."
        )
    if (cfg.robotics or {}).get("ppo") is not None:
        raise ValueError("train_distill.py refuses PPO configs; distillation starts from a BC actor.")
    if not (cfg.robotics or {}).get("architecture", {}).get("type"):
        raise ValueError("robotics.architecture.type must be set in distillation configs.")
    if not (cfg.training or {}).get("trainer_type"):
        raise ValueError("training.trainer_type must be set, usually 'flightmare_dagger_distill'.")
    if not cfg.distill:
        raise ValueError("distillation configs require a top-level distill block.")


def test_config(cfg: Config) -> None:
    print("=" * 60)
    print("[TEST] Distillation Config + Model Validation")
    print("=" * 60)
    arch_cfg = cfg.robotics.get("architecture", {})
    model = build("architecture", **arch_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch_cfg['type']} trainable={trainable:,}")

    state_dim = (
        int(arch_cfg.get("proprio_core_dim", 0))
        + int(arch_cfg.get("gate_dim", 0))
        + int(arch_cfg.get("aux_dim", 0))
    )
    if state_dim <= 0:
        state_dim = int(arch_cfg.get("state_dim", 13))
    action_dim = int(arch_cfg.get("action_dim", 4))
    batch = {
        "images": {},
        "state": torch.zeros(2, state_dim),
        "prev_actions": torch.zeros(2, action_dim),
        "action": torch.zeros(2, action_dim),
        "sample_weight": torch.ones(2),
        "expert_log_std": torch.zeros(2, action_dim),
    }
    with torch.no_grad():
        out = model(batch)
    print(f"Forward loss: {float(out['loss']):.4f}")
    print(f"Trainer: {cfg.training['trainer_type']}")
    print(f"Collection mode: {cfg.distill.get('collection_mode', 'flightmare')}")
    print("=" * 60)


def main(exp_name: str, test_only: bool = False, no_wandb: bool = False) -> None:
    cfg = Config.from_experiment(exp_name)
    validate_distill_config(cfg, exp_name)
    if no_wandb:
        cfg.wandb = {}

    print(f"[Config] Run:    {cfg.run['name']}")
    print(f"[Config] Mode:   {cfg.run['mode']}")
    print(f"[Config] Data:   type={cfg.data.get('type')} action={cfg.data.get('action_type')}")
    print(f"[Config] Arch:   {cfg.robotics.get('architecture', {}).get('type')}")
    print(f"[Config] Rounds: {cfg.distill.get('rounds', 1)}")
    print(f"[Config] Output: {cfg.training['output_dir']}")
    if no_wandb:
        print("[Config] Wandb:  disabled by --no-wandb")
    print()

    if test_only:
        test_config(cfg)
        return

    model = build("architecture", **cfg.robotics["architecture"])
    trainer = build(
        "trainer",
        type=cfg.training.get("trainer_type", "flightmare_dagger_distill"),
        model=model,
        training_cfg=cfg.training,
        robotics_cfg=cfg.robotics,
        wandb_cfg=cfg.wandb,
        data_cfg=cfg.data,
        distill_cfg=cfg.distill,
        full_cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flightmare policy by DAgger distillation")
    parser.add_argument("--exp", type=str, required=True, help="Experiment config name under configs/exp")
    parser.add_argument("--test", action="store_true", help="Validate config/model only")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging for this run")
    args = parser.parse_args()
    main(exp_name=args.exp, test_only=args.test, no_wandb=args.no_wandb)

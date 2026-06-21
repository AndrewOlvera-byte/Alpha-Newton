from __future__ import annotations

import json
import os

import pytest

from src.robotics.distill_trainer import DistillationTrainer
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import MLPFusionGaussianExpertActor


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FLIGHTMARE_INTEGRATION") != "1",
    reason="Set RUN_FLIGHTMARE_INTEGRATION=1 inside docker compose exec alpha-newton.",
)


def _tiny_actor() -> MLPFusionGaussianExpertActor:
    return MLPFusionGaussianExpertActor(
        action_dim=4,
        fusion="swift",
        proprio_core_dim=9,
        gate_dim=24,
        aux_dim=3,
        use_vision=False,
        include_prev_action=True,
        proprio_hidden_dim=8,
        proprio_embed_dim=8,
        gate_hidden_dim=8,
        gate_embed_dim=8,
        trunk_hidden_dim=16,
        trunk_depth=1,
        head_hidden_dim=8,
        head_depth=1,
        state_dependent_std=True,
        bc_loss_type="gnll",
    )


def test_geometric_controller_preflight_passes_single_flightmare_episode():
    from scripts.flightmare_bc.collection_config import load_collection_config
    from scripts.flightmare_bc.eval_controller import evaluate_controller_config

    cfg = load_collection_config("configs/flightmare_bc/ctbr_v6.yaml")
    result = evaluate_controller_config(cfg, episodes=1, speed_bins=1, seed=123)

    summary = result["summary"]
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["mean_gate_completion"] == pytest.approx(1.0)
    assert summary["gate_miss_rate"] == pytest.approx(0.0)


def test_distill_trainer_collects_one_geometric_bootstrap_round(tmp_path):
    trainer = DistillationTrainer(
        model=_tiny_actor(),
        training_cfg={
            "output_dir": str(tmp_path / "distill_out"),
            "trainer_type": "flightmare_dagger_distill",
            "rollout_eval": {"enabled": False},
        },
        robotics_cfg={"architecture": {"type": "flightmare_actor"}},
        wandb_cfg={},
        data_cfg={
            "type": "flightmare_bc_state_v3",
            "data_dir": str(tmp_path / "unused"),
            "action_type": "ctbr",
            "normalize_obs": True,
            "normalize_action": True,
            "action_normalization": "bounds",
            "preload": True,
        },
        distill_cfg={
            "collection_mode": "flightmare",
            "collection_config": "configs/flightmare_bc/ctbr_v6.yaml",
            "rounds": 1,
            "bootstrap_episodes": 1,
            "episodes_per_round": 1,
            "val_frac": 0.0,
            "inner_training": {"max_steps": 0},
            "collection_overrides": {
                "controller_eval": {
                    "enabled": True,
                    "episodes": 1,
                    "speed_bins": 1,
                    "min_success_rate": 1.0,
                    "min_gate_completion": 1.0,
                },
                "postprocess": {
                    "transform_to_v3": True,
                    "recompute_v3_norm_stats": True,
                    "diag_v3_geometry": False,
                },
            },
        },
    )

    trainer.train()

    out = tmp_path / "distill_out"
    assert (out / "best" / "model.pt").exists()
    aggregate = json.loads((out / "round_000" / "aggregate" / "index.json").read_text())
    assert len(aggregate["episodes"]) == 1
    assert aggregate["episodes"][0]["split"] == "train"

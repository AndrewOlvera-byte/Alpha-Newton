from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from scripts.flightmare_bc.hdf5_writer import EpisodeWriter
from src.core.config import Config
from src.entrypoints.train_distill import validate_distill_config
from src.robotics.data import FlightmareBCStateV3Dataset
from src.robotics.distill_trainer import DistillationTrainer
from src.robotics.flightmare_experts.labels import HybridExpertLabeler
from src.robotics.flightmare_experts.safety import SafetySupervisor
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import MLPFusionGaussianExpertActor


def _write_norm_stats(data_dir):
    np.savez(
        data_dir / "norm_stats.npz",
        state_mean=np.zeros(13, dtype=np.float32),
        state_std=np.ones(13, dtype=np.float32),
        ctbr_mean=np.zeros(4, dtype=np.float32),
        ctbr_std=np.ones(4, dtype=np.float32),
        ctbr_low=np.array([0.0, -10.0, -10.0, -5.0], dtype=np.float32),
        ctbr_high=np.array([1.0, 10.0, 10.0, 5.0], dtype=np.float32),
        action_normalization=np.array("bounds"),
        proprio_core_mean=np.zeros(9, dtype=np.float32),
        proprio_core_std=np.ones(9, dtype=np.float32),
        gate_mean=np.zeros(24, dtype=np.float32),
        gate_std=np.ones(24, dtype=np.float32),
        aux_mean=np.zeros(3, dtype=np.float32),
        aux_std=np.ones(3, dtype=np.float32),
    )


def _write_v3_distill_dataset(data_dir):
    (data_dir / "episodes").mkdir()
    with h5py.File(data_dir / "episodes" / "ep_000000.h5", "w") as h:
        h.create_dataset("obs/proprio_core", data=np.zeros((2, 9), dtype=np.float32))
        h.create_dataset("obs/gate", data=np.zeros((2, 24), dtype=np.float32))
        h.create_dataset("obs/aux", data=np.zeros((2, 3), dtype=np.float32))
        h.create_dataset("action/ctbr", data=np.array([[0.2, 1.0, 2.0, 3.0], [0.3, 4.0, 5.0, 6.0]], dtype=np.float32))
        h.create_dataset("action/ctbr_prev", data=np.array([[0.1, 0.0, 0.0, 0.0], [0.2, 1.0, 2.0, 3.0]], dtype=np.float32))
        h.create_dataset("expert/weight", data=np.array([1.0, 0.25], dtype=np.float32))
        h.create_dataset("expert/action_log_std", data=np.full((2, 4), -1.5, dtype=np.float32))
    (data_dir / "index.json").write_text(
        json.dumps(
            {
                "episodes": [{"episode_id": 0, "path": "episodes/ep_000000.h5", "length": 2, "split": "train"}],
                "obs_v3": {
                    "proprio_core": {"dim": 9},
                    "gate": {"dim": 24},
                    "aux": {"dim": 3},
                },
            }
        )
    )
    _write_norm_stats(data_dir)


def _write_v3_kind_dataset(data_dir):
    (data_dir / "episodes").mkdir()
    episodes = []
    specs = [("expert", 6), ("dagger", 2), ("synthetic_recovery", 2)]
    for ep_id, (kind, length) in enumerate(specs):
        with h5py.File(data_dir / "episodes" / f"ep_{ep_id:06d}.h5", "w") as h:
            h.create_dataset("obs/proprio_core", data=np.zeros((length, 9), dtype=np.float32))
            h.create_dataset("obs/gate", data=np.zeros((length, 24), dtype=np.float32))
            h.create_dataset("obs/aux", data=np.zeros((length, 3), dtype=np.float32))
            h.create_dataset("action/ctbr", data=np.zeros((length, 4), dtype=np.float32))
            h.create_dataset("action/ctbr_prev", data=np.zeros((length, 4), dtype=np.float32))
        episodes.append(
            {
                "episode_id": ep_id,
                "path": f"episodes/ep_{ep_id:06d}.h5",
                "length": length,
                "split": "train",
                "sample_kind": kind,
            }
        )
    (data_dir / "index.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "obs_v3": {
                    "proprio_core": {"dim": 9},
                    "gate": {"dim": 24},
                    "aux": {"dim": 3},
                },
            }
        )
    )
    _write_norm_stats(data_dir)


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


def test_distill_config_loads_robotics_base_and_validates():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_distill_mppi_v1")

    assert cfg.run["mode"] == "distill"
    assert cfg.distill["expert"]["primary"] == "geometric_minjerk"
    assert cfg.distill["experimental_mppi_collection_config"] == "configs/flightmare_bc/ctbr_v7_mpcc.yaml"
    assert cfg.robotics["architecture"]["type"] == "flightmare_actor"
    validate_distill_config(cfg, "flightmare/paper/teacher/ctbr_distill_mppi_v1")


def test_balanced_v2_config_enables_sample_kind_balancing_and_matched_eval():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_distill_geometric_balanced_v2")

    validate_distill_config(cfg, "flightmare/paper/teacher/ctbr_distill_geometric_balanced_v2")
    assert cfg.training["sample_kind_balancing"]["enabled"] is True
    assert cfg.training["sample_kind_balancing"]["target_mix"]["dagger"] == pytest.approx(0.65)
    assert "course_distribution" not in cfg.training["rollout_eval"]
    assert cfg.distill["aggregate"]["max_rounds_kept"] == 3


def test_train_distill_rejects_non_distill_mode():
    cfg = Config(
        run={"name": "bad", "mode": "bc"},
        model={},
        tokenizer={},
        data={},
        training={"trainer_type": "flightmare_dagger_distill"},
        wandb={},
        robotics={"architecture": {"type": "flightmare_actor"}},
        distill={"rounds": 1},
    )

    with pytest.raises(ValueError, match="run.mode='distill'"):
        validate_distill_config(cfg, "bad")


def test_episode_writer_persists_expert_distillation_fields(tmp_path):
    ep_path = tmp_path / "ep.h5"
    with EpisodeWriter(
        ep_path,
        image_size=1,
        cameras=[],
        state_dim=13,
        controller_name="test",
        dt=0.01,
        seed=0,
        episode_id=0,
    ) as writer:
        action = np.zeros(4, dtype=np.float32)
        writer.append(
            state=np.zeros(13, dtype=np.float32),
            images={},
            actions={"waypoint": action, "ctbr": action, "motor": action},
            prev_actions={"waypoint": action, "ctbr": action, "motor": action},
            ref_pos=np.zeros(3, dtype=np.float32),
            ref_vel=np.zeros(3, dtype=np.float32),
            ref_yaw=0.0,
            done=False,
            expert={
                "weight": 0.5,
                "confidence": 0.75,
                "action_log_std": np.full(4, -1.0, dtype=np.float32),
                "source": "mppi",
                "valid": True,
            },
        )

    with h5py.File(ep_path, "r") as h:
        assert h["expert/weight"][0] == pytest.approx(0.5)
        assert h["expert/confidence"][0] == pytest.approx(0.75)
        np.testing.assert_allclose(h["expert/action_log_std"][0], np.full(4, -1.0))
        assert h["expert/source"][0].decode() == "mppi"


def test_v3_dataset_emits_distillation_fields_preloaded_and_lazy(tmp_path):
    _write_v3_distill_dataset(tmp_path)

    lazy = FlightmareBCStateV3Dataset(
        data_dir=str(tmp_path),
        action_type="ctbr",
        split="train",
        normalize_obs=False,
        normalize_action=False,
        preload=False,
    )
    preloaded = FlightmareBCStateV3Dataset(
        data_dir=str(tmp_path),
        action_type="ctbr",
        split="train",
        normalize_obs=False,
        normalize_action=False,
        preload=True,
    )

    for ds in (lazy, preloaded):
        sample = ds[1]
        assert sample["sample_weight"].item() == pytest.approx(0.25)
        np.testing.assert_allclose(sample["expert_log_std"].numpy(), np.full(4, -1.5))


def test_v3_dataset_sample_kind_sampling_weights_target_mix(tmp_path):
    _write_v3_kind_dataset(tmp_path)
    ds = FlightmareBCStateV3Dataset(
        data_dir=str(tmp_path),
        action_type="ctbr",
        split="train",
        normalize_obs=False,
        normalize_action=False,
        preload=True,
    )

    assert ds.sample_kind_counts() == {"expert": 6, "dagger": 2, "synthetic_recovery": 2}
    weights = ds.sampling_weights_for_sample_kind_mix(
        {"expert": 0.15, "dagger": 0.65, "synthetic_recovery": 0.20}
    )
    by_kind = {name: 0.0 for name in ds.sample_kind_names}
    for idx, weight in enumerate(weights.tolist()):
        kind = ds.sample_kind_names[int(ds._sample_kind_ids[idx])]
        by_kind[kind] += weight
    total = sum(by_kind.values())

    assert by_kind["expert"] / total == pytest.approx(0.15)
    assert by_kind["dagger"] / total == pytest.approx(0.65)
    assert by_kind["synthetic_recovery"] / total == pytest.approx(0.20)


def test_weighted_gnll_uses_sample_weight():
    torch.manual_seed(0)
    model = _tiny_actor()
    batch = {
        "images": {},
        "state": torch.randn(2, 36),
        "prev_actions": torch.zeros(2, 4),
        "action": torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, -3.0, 2.0, -2.0]]),
    }
    unweighted = model(batch)
    weighted = model({**batch, "sample_weight": torch.tensor([1.0, 0.0])})

    assert unweighted["_loss_per_sample"][0] != pytest.approx(unweighted["_loss_per_sample"][1])
    assert weighted["loss"].item() == pytest.approx(unweighted["_loss_per_sample"][0].item())


def test_hybrid_labeler_falls_back_when_primary_is_invalid():
    class BadExpert:
        source = "bad_mppi"

        def compute(self, *_, **__):
            return {
                "thrust_normalized": float("nan"),
                "body_rates": np.zeros(3, dtype=np.float32),
                "motor_normalized": np.zeros(4, dtype=np.float32),
                "source": self.source,
            }

    class GoodExpert:
        source = "geometric_minjerk"

        def compute(self, *_, **__):
            return {
                "thrust_normalized": 0.5,
                "body_rates": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "motor_normalized": np.ones(4, dtype=np.float32) * 0.5,
                "source": self.source,
            }

    labeler = HybridExpertLabeler(
        BadExpert(),
        GoodExpert(),
        supervisor=SafetySupervisor(max_body_rate=5.0),
        max_body_rate=5.0,
    )
    label = labeler.compute()

    assert label.valid
    assert label.source == "geometric_minjerk_fallback"
    np.testing.assert_allclose(label.action_ctbr_raw, np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32))


def test_distillation_trainer_mock_round_writes_aggregate_and_best(tmp_path):
    model = _tiny_actor()
    trainer = DistillationTrainer(
        model=model,
        training_cfg={
            "output_dir": str(tmp_path / "out"),
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
            "collection_mode": "mock",
            "rounds": 1,
            "episodes_per_round": 2,
            "mock_episode_length": 3,
            "inner_training": {"max_steps": 0},
            "val_frac": 0.5,
        },
    )

    trainer.train()

    out = tmp_path / "out"
    assert (out / "best" / "model.pt").exists()
    aggregate_index = json.loads((out / "round_000" / "aggregate" / "index.json").read_text())
    assert len(aggregate_index["episodes"]) == 2
    ds = FlightmareBCStateV3Dataset(
        data_dir=str(out / "round_000" / "aggregate"),
        action_type="ctbr",
        split="train",
        preload=True,
    )
    assert "sample_weight" in ds[0]

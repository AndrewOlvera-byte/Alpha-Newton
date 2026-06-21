from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from scripts.flightmare_bc.collection_config import collect_cli_args, load_collection_config, mode_counts
from scripts.flightmare_bc.collect_dataset import _run_controller_preflight
from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.eval_controller import evaluate_controller_config
from src.robotics.bc_trainer import BCTrainer
from src.robotics.data import FlightmareBCStateV3Dataset
from src.robotics.flightmare_policy_eval import apply_plant_overrides, plant_config_from_manifest


def _write_v3_stats(data_dir):
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


def _write_v3_dataset(data_dir):
    (data_dir / "episodes").mkdir()
    with h5py.File(data_dir / "episodes" / "ep_000000.h5", "w") as h:
        h.create_dataset("obs/proprio_core", data=np.zeros((2, 9), dtype=np.float32))
        h.create_dataset("obs/gate", data=np.zeros((2, 24), dtype=np.float32))
        h.create_dataset("obs/aux", data=np.zeros((2, 3), dtype=np.float32))
        h.create_dataset(
            "action/ctbr",
            data=np.array([[0.2, 1.0, 2.0, 3.0], [0.3, 4.0, 5.0, 6.0]], dtype=np.float32),
        )
        h.create_dataset(
            "action/ctbr_prev",
            data=np.array([[0.1, -1.0, -2.0, -3.0], [0.2, 1.0, 2.0, 3.0]], dtype=np.float32),
        )
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
    _write_v3_stats(data_dir)


def test_v6_collection_config_is_explicit_about_controller_and_speed():
    cfg = load_collection_config("configs/flightmare_bc/ctbr_v6.yaml")

    assert cfg["expert"]["controller"] == "geometric_minjerk"
    assert cfg["expert"]["speed_range"] == [1.8, 2.6]
    assert cfg["expert"]["trajectory_min_segment_s"] == pytest.approx(0.8)
    assert cfg["controller_eval"]["enabled"] is True
    assert cfg["controller_eval"]["min_success_rate"] == pytest.approx(1.0)
    assert cfg["controller_eval"]["min_gate_completion"] == pytest.approx(1.0)
    assert cfg["mpcc"]["enabled"] is False
    counts = mode_counts(17, cfg)
    assert sum(counts.values()) == 17
    assert set(counts) == {"expert", "synthetic_recovery", "dagger"}

    args = collect_cli_args(cfg, episodes=3)
    assert "--random-start-gate" in args
    assert args[args.index("--speed-range") + 1: args.index("--speed-range") + 3] == ["1.8", "2.6"]
    assert args[args.index("--omega-max-body") + 1: args.index("--omega-max-body") + 4] == ["18.0", "18.0", "18.0"]


def test_v6_controller_eval_passes_strict_flightmare_gate_rollouts():
    cfg = load_collection_config("configs/flightmare_bc/ctbr_v6.yaml")

    result = evaluate_controller_config(cfg, episodes=2, speed_bins=2, seed=99)
    summary = result["summary"]

    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["mean_gate_completion"] == pytest.approx(1.0)
    assert summary["gate_miss_rate"] == pytest.approx(0.0)
    assert summary["mean_track_err_m"] < 0.10
    for row in result["by_speed_bin"]:
        assert row["success_rate"] == pytest.approx(1.0)
        assert row["mean_gate_completion"] == pytest.approx(1.0)


def test_controller_preflight_rejects_failed_eval(monkeypatch):
    cfg = load_collection_config("configs/flightmare_bc/ctbr_v6.yaml")

    def fake_eval_controller_config(*_args, **_kwargs):
        return {
            "summary": {
                "success_rate": 0.0,
                "mean_gate_completion": 0.25,
                "mean_track_err_m": 2.0,
            },
            "by_speed_bin": [],
            "episodes": [],
        }

    monkeypatch.setattr(
        "scripts.flightmare_bc.eval_controller.evaluate_controller_config",
        fake_eval_controller_config,
    )

    with pytest.raises(RuntimeError, match="Controller preflight failed"):
        _run_controller_preflight(cfg)


def test_mpcc_controller_requires_real_adapter(tmp_path):
    cfg_path = tmp_path / "bad_mpcc.yaml"
    cfg_path.write_text("expert:\n  controller: mpcc\nmpcc:\n  enabled: false\n")

    with pytest.raises(ValueError, match="mpcc.enabled=true"):
        load_collection_config(cfg_path)


def test_manifest_plant_config_applies_full_racing_plant():
    manifest = {
        "params": {
            "mass": 0.82,
            "arm_length": 0.18,
            "inertia": [0.003, 0.004, 0.005],
            "k_thrust": 2.0e-6,
            "k_torque": 3.0e-7,
            "motor_omega_min": 140.0,
            "motor_omega_max": 3200.0,
            "motor_tau": 0.0002,
            "thrust_map": [1.0e-6, 0.003, -1.4],
            "kappa": 0.018,
            "omega_max_body": [18.0, 17.0, 16.0],
        }
    }

    params = apply_plant_overrides(QuadParams(), plant_config_from_manifest(manifest))

    assert params.mass == pytest.approx(0.82)
    assert params.arm_length == pytest.approx(0.18)
    assert params.inertia == (0.003, 0.004, 0.005)
    assert params.k_thrust == pytest.approx(2.0e-6)
    assert params.k_torque == pytest.approx(3.0e-7)
    assert params.motor_omega_min == pytest.approx(140.0)
    assert params.motor_omega_max == pytest.approx(3200.0)
    assert params.motor_tau == pytest.approx(0.0002)
    assert params.thrust_map == (1.0e-6, 0.003, -1.4)
    assert params.kappa == pytest.approx(0.018)
    assert params.omega_max_body == (18.0, 17.0, 16.0)


def test_v3_dataset_prefers_stored_prev_action_for_recovery_samples(tmp_path):
    _write_v3_dataset(tmp_path)

    ds = FlightmareBCStateV3Dataset(
        data_dir=str(tmp_path),
        action_type="ctbr",
        split="train",
        normalize_obs=False,
        normalize_action=False,
        preload=False,
    )

    np.testing.assert_allclose(ds[0]["prev_actions"].numpy(), np.array([0.1, -1.0, -2.0, -3.0]))
    np.testing.assert_allclose(ds[1]["prev_actions"].numpy(), np.array([0.2, 1.0, 2.0, 3.0]))


def test_preloaded_v3_dataset_prefers_stored_prev_action(tmp_path):
    _write_v3_dataset(tmp_path)

    ds = FlightmareBCStateV3Dataset(
        data_dir=str(tmp_path),
        action_type="ctbr",
        split="train",
        normalize_obs=False,
        normalize_action=False,
        preload=True,
    )

    np.testing.assert_allclose(ds[0]["prev_actions"].numpy(), np.array([0.1, -1.0, -2.0, -3.0]))
    np.testing.assert_allclose(ds[1]["prev_actions"].numpy(), np.array([0.2, 1.0, 2.0, 3.0]))


def test_flightmare_rollout_selection_prefers_closed_loop_score_over_eval_loss():
    trainer = object.__new__(BCTrainer)
    trainer.best_flightmare_rollout_score = 2.0
    trainer._best_flightmare_gate_completion = 0.4
    trainer.best_eval_loss = 0.1

    assert not trainer._select_flightmare_rollout_checkpoint({"score": 1.9, "mean_gate_completion": 0.9}, 0.01)
    assert trainer.best_flightmare_rollout_score == pytest.approx(2.0)
    assert trainer.best_eval_loss == pytest.approx(0.1)

    assert trainer._select_flightmare_rollout_checkpoint({"score": 2.1, "mean_gate_completion": 0.5}, 0.5)
    assert trainer.best_flightmare_rollout_score == pytest.approx(2.1)
    assert trainer._best_flightmare_gate_completion == pytest.approx(0.5)


def test_rollout_selection_does_not_consume_patience_without_rollout_metrics():
    trainer = object.__new__(BCTrainer)

    trainer.selection_metric = "flightmare_rollout_score"
    assert not trainer._selection_metric_checked_for_early_stopping({})
    assert trainer._selection_metric_checked_for_early_stopping({"score": 0.0})

    trainer.selection_metric = "eval_loss"
    assert trainer._selection_metric_checked_for_early_stopping({})

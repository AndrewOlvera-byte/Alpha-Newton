from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
import yaml

from scripts.flightmare_bc.collect import STATE_DIM, write_norm_stats
from src.core.registry import build, register


def _deep_merge(base: dict, override: dict | None) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_OBS_V3_BLOCK = {
    "lookahead_gates": 2,
    "proprio_core": {
        "dim": 9,
        "layout": ["v_body(3)", "omega_body(3)", "gravity_body(3)"],
    },
    "gate": {
        "dim": 24,
        "layout": "2 gates x 4 corners x 3 body-frame xyz",
    },
    "aux": {
        "dim": 3,
        "layout": ["progress", "dist_to_current", "time_since_pass"],
    },
    "total_state_dim_no_prev_action": 36,
}


class DistillationTrainer:
    """Closed-loop DAgger-style distillation orchestrator for Flightmare."""

    def __init__(
        self,
        model: torch.nn.Module,
        training_cfg: Dict[str, Any],
        robotics_cfg: Dict[str, Any] | None = None,
        wandb_cfg: Dict[str, Any] | None = None,
        data_cfg: Dict[str, Any] | None = None,
        distill_cfg: Dict[str, Any] | None = None,
        full_cfg: Any | None = None,
    ):
        self.model = model
        self.training_cfg = copy.deepcopy(training_cfg or {})
        self.robotics_cfg = copy.deepcopy(robotics_cfg or {})
        self.wandb_cfg = copy.deepcopy(wandb_cfg or {})
        self.data_cfg = copy.deepcopy(data_cfg or {})
        self.distill_cfg = copy.deepcopy(distill_cfg or {})
        self.full_cfg = full_cfg

        self.output_dir = Path(self.training_cfg.get("output_dir", "outputs/distill"))
        self.rounds = int(self.distill_cfg.get("rounds", 1))
        self.episodes_per_round = int(self.distill_cfg.get("episodes_per_round", 128))
        self.bootstrap_episodes = int(self.distill_cfg.get("bootstrap_episodes", self.episodes_per_round))
        self.val_frac = float(self.distill_cfg.get("val_frac", 0.1))
        self.seed = int(self.distill_cfg.get("seed", self.training_cfg.get("seed", 0)))
        self.collection_mode = str(self.distill_cfg.get("collection_mode", "flightmare"))
        self.best_score = float("-inf")
        self.best_checkpoint: Path | None = None

    @staticmethod
    def _schedule_value(schedule: Any, round_idx: int, default: Any) -> Any:
        """Resolve a per-round scheduled value.

        ``schedule`` may be: None (use ``default``); a list/tuple (indexed by
        round, clamped to the last entry for later rounds); or a dict keyed by
        int/str round index with an optional ``"default"`` fallback.
        """
        if schedule is None:
            return default
        if isinstance(schedule, (list, tuple)):
            if not schedule:
                return default
            idx = min(int(round_idx), len(schedule) - 1)
            return schedule[idx]
        if isinstance(schedule, dict):
            for key in (round_idx, str(round_idx)):
                if key in schedule:
                    return schedule[key]
            return schedule.get("default", default)
        return schedule

    def _apply_loss_schedule(self, round_idx: int) -> None:
        """Set the model's BC loss type/delta for this round (Huber->GNLL anneal)."""
        schedule = self.distill_cfg.get("bc_loss_schedule")
        loss_type = self._schedule_value(schedule, round_idx, None)
        if loss_type is not None and hasattr(self.model, "bc_loss_type"):
            self.model.bc_loss_type = str(loss_type)
        delta = self._schedule_value(self.distill_cfg.get("bc_huber_delta_schedule"), round_idx, None)
        if delta is not None and hasattr(self.model, "bc_huber_delta"):
            self.model.bc_huber_delta = float(delta)
        if loss_type is not None:
            print(
                f"[Distill] round={round_idx} bc_loss_type={getattr(self.model, 'bc_loss_type', '?')}",
                flush=True,
            )

    def _apply_scenario_weight_schedule(self, cfg: dict, round_idx: int) -> None:
        """Optionally rescale (never zero) course scenario weights for this round.

        ``distill.scenario_weight_schedule`` maps scenario ``name`` or ``family``
        to a per-round multiplier (list/dict per :meth:`_schedule_value`). This
        only shifts the anchor-vs-procedural balance; it never removes a family,
        keeping the DAgger aggregate buffer representative.
        """
        schedule = self.distill_cfg.get("scenario_weight_schedule")
        if not schedule:
            return
        dist = cfg.get("course_distribution")
        if not isinstance(dist, dict):
            return
        for sc in dist.get("scenarios", []) or []:
            key = sc.get("name") if sc.get("name") in schedule else sc.get("family")
            if key not in schedule:
                continue
            mult = float(self._schedule_value(schedule[key], round_idx, 1.0))
            sc["weight"] = max(1e-6, float(sc.get("weight", 1.0)) * mult)

    def train(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.output_dir / "distill_config.json", self._jsonable(self.distill_cfg))
        current_checkpoint = self._load_initial_student()
        current_policy_config = self._initial_policy_config()
        current_data_dir: Path | None = None
        collected_dirs: list[Path] = []

        t0 = time.time()
        for round_idx in range(self.rounds):
            round_dir = self.output_dir / f"round_{round_idx:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Distill] round={round_idx} output={round_dir}", flush=True)

            dataset_dir = self._collect_round(
                round_idx,
                current_checkpoint=current_checkpoint,
                current_policy_config=current_policy_config,
                out_dir=round_dir / "dataset",
            )
            collected_dirs.append(dataset_dir)
            kept_dirs = self._kept_round_dirs(collected_dirs)
            aggregate_dir = self._write_aggregate_dataset(round_idx, kept_dirs, round_dir / "aggregate")
            current_data_dir = aggregate_dir

            self._apply_loss_schedule(round_idx)
            checkpoint_dir = self._run_inner_training(round_idx, aggregate_dir, round_dir / "train")
            current_checkpoint = checkpoint_dir / "model.pt"
            current_policy_config = self._write_policy_config(round_idx, aggregate_dir, round_dir / "policy_config.yaml")

            metrics = self._evaluate_round(round_idx, aggregate_dir, checkpoint_dir, round_dir)
            self._maybe_promote_best(checkpoint_dir, metrics)
            self._write_json(
                round_dir / "round_summary.json",
                {
                    "round": round_idx,
                    "dataset_dir": str(dataset_dir),
                    "aggregate_dir": str(aggregate_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                    "policy_config": str(current_policy_config),
                    "metrics": metrics,
                },
            )

        elapsed = time.time() - t0
        self._write_json(
            self.output_dir / "distill_state.json",
            {
                "rounds": self.rounds,
                "elapsed_s": elapsed,
                "best_score": self.best_score,
                "best_checkpoint": None if self.best_checkpoint is None else str(self.best_checkpoint),
                "latest_data_dir": None if current_data_dir is None else str(current_data_dir),
            },
        )
        print(f"[Distill] complete rounds={self.rounds} output={self.output_dir}", flush=True)

    def _load_initial_student(self) -> Path | None:
        student_cfg = self.distill_cfg.get("student", {}) or {}
        ckpt = student_cfg.get("init_checkpoint")
        if not ckpt:
            return None
        ckpt_path = self._model_path(Path(ckpt))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"student.init_checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)
        print(f"[Distill] loaded initial student checkpoint {ckpt_path}", flush=True)
        return ckpt_path

    def _initial_policy_config(self) -> Path | None:
        student_cfg = self.distill_cfg.get("student", {}) or {}
        cfg_path = student_cfg.get("init_policy_config")
        if not cfg_path:
            return None
        path = Path(cfg_path)
        if not path.exists():
            raise FileNotFoundError(f"student.init_policy_config not found: {path}")
        return path

    def _collect_round(
        self,
        round_idx: int,
        *,
        current_checkpoint: Path | None,
        current_policy_config: Path | None,
        out_dir: Path,
    ) -> Path:
        if self.collection_mode == "mock":
            self._write_mock_dataset(out_dir, round_idx)
            return out_dir
        self._run_flightmare_collection(
            round_idx,
            current_checkpoint=current_checkpoint,
            current_policy_config=current_policy_config,
            out_dir=out_dir,
        )
        return out_dir

    def _run_flightmare_collection(
        self,
        round_idx: int,
        *,
        current_checkpoint: Path | None,
        current_policy_config: Path | None,
        out_dir: Path,
    ) -> None:
        from scripts.flightmare_bc.collection_config import load_collection_config

        bootstrap_round = round_idx == 0 and self.bootstrap_episodes > 0
        collection_path = self._collection_config_path(bootstrap_round=bootstrap_round)
        if not collection_path:
            raise ValueError("distill.collection_config is required when collection_mode='flightmare'.")
        cfg = load_collection_config(collection_path)
        cfg = _deep_merge(cfg, self.distill_cfg.get("collection_overrides"))
        if bootstrap_round:
            cfg = _deep_merge(cfg, self.distill_cfg.get("bootstrap_collection_overrides"))
        else:
            cfg = _deep_merge(cfg, self.distill_cfg.get("dagger_collection_overrides"))
        cfg["output_dir"] = str(out_dir)
        cfg["seed"] = int(cfg.get("seed", 0)) + round_idx * 1009
        cfg["episodes"] = self.bootstrap_episodes if bootstrap_round else self.episodes_per_round
        self._apply_scenario_weight_schedule(cfg, round_idx)

        clean_frac = float(
            self._schedule_value(
                self.distill_cfg.get("clean_expert_fraction_schedule"),
                round_idx,
                self.distill_cfg.get("clean_expert_fraction", 0.1),
            )
        )
        recovery_frac = float(
            self._schedule_value(
                self.distill_cfg.get("synthetic_recovery_fraction_schedule"),
                round_idx,
                self.distill_cfg.get("synthetic_recovery_fraction", 0.0),
            )
        )
        if bootstrap_round or current_checkpoint is None:
            cfg["mix"] = {"expert": 1.0, "synthetic_recovery": 0.0, "dagger": 0.0}
            cfg["dagger"] = {"enabled": False}
        else:
            if current_policy_config is None:
                raise RuntimeError("DAgger collection requires a policy config from the previous round.")
            dagger_frac = max(0.0, 1.0 - clean_frac - recovery_frac)
            cfg["mix"] = {
                "expert": clean_frac,
                "synthetic_recovery": recovery_frac,
                "dagger": dagger_frac,
            }
            dagger_cfg = dict(cfg.get("dagger", {}) or {})
            dagger_cfg.update(
                {
                    "enabled": True,
                    "checkpoint": str(current_checkpoint),
                    "policy_config": str(current_policy_config),
                }
            )
            cfg["dagger"] = dagger_cfg

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "collection_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd = [sys.executable, "-m", "scripts.flightmare_bc.collect_dataset", "--config", str(cfg_path)]
        if bool(self.distill_cfg.get("skip_controller_eval", False)):
            cmd.append("--skip-controller-eval")
        print(f"[Distill] collecting round={round_idx} episodes={cfg['episodes']} -> {out_dir}", flush=True)
        subprocess.run(cmd, check=True)

    def _collection_config_path(self, *, bootstrap_round: bool) -> str | None:
        if bootstrap_round:
            return self.distill_cfg.get("bootstrap_collection_config") or self.distill_cfg.get("collection_config")
        return (
            self.distill_cfg.get("dagger_collection_config")
            or self.distill_cfg.get("collection_config")
            or self.distill_cfg.get("bootstrap_collection_config")
        )

    def _write_mock_dataset(self, out_dir: Path, round_idx: int) -> None:
        rng = np.random.default_rng(self.seed + round_idx)
        episodes = int(self.distill_cfg.get("mock_episodes", self.episodes_per_round))
        length = int(self.distill_cfg.get("mock_episode_length", 8))
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_dir = out_dir / "episodes"
        ep_dir.mkdir(exist_ok=True)
        manifest = {
            "version": 2,
            "collector": "src.robotics.distill_trainer.mock",
            "controller": "mock_hybrid",
            "image_size": 0,
            "cameras": [],
            "control_hz": 100.0,
            "state_dim": STATE_DIM,
            "action_normalization": "bounds",
            "skip_initial_frames": 0,
            "bounds_margin": 0.10,
            "obs_v3": _OBS_V3_BLOCK,
            "episodes": [],
        }
        for ep in range(episodes):
            path = ep_dir / f"ep_{ep:06d}.h5"
            proprio = rng.normal(0.0, 0.2, size=(length, 9)).astype(np.float32)
            gate = rng.normal(0.0, 0.5, size=(length, 24)).astype(np.float32)
            aux = rng.normal(0.0, 0.1, size=(length, 3)).astype(np.float32)
            state = rng.normal(0.0, 0.3, size=(length, STATE_DIM)).astype(np.float32)
            action = np.stack(
                [
                    rng.uniform(0.35, 0.65, size=length),
                    rng.normal(0.0, 1.0, size=length),
                    rng.normal(0.0, 1.0, size=length),
                    rng.normal(0.0, 0.5, size=length),
                ],
                axis=1,
            ).astype(np.float32)
            prev = np.concatenate([np.zeros((1, 4), dtype=np.float32), action[:-1]], axis=0)
            with h5py.File(path, "w") as h:
                h.create_dataset("obs/state", data=state)
                h.create_dataset("obs/proprio_core", data=proprio)
                h.create_dataset("obs/gate", data=gate)
                h.create_dataset("obs/aux", data=aux)
                h.create_dataset("action/ctbr", data=action)
                h.create_dataset("action/ctbr_prev", data=prev)
                h.create_dataset("mission/gate_index", data=np.zeros((length,), dtype=np.int32))
                h.create_dataset("expert/weight", data=np.linspace(1.0, 0.5, length, dtype=np.float32))
                h.create_dataset("expert/action_log_std", data=np.full((length, 4), -2.0, dtype=np.float32))
                h.create_dataset("expert/confidence", data=np.ones((length,), dtype=np.float32))
            manifest["episodes"].append(
                {
                    "episode_id": ep,
                    "path": f"episodes/ep_{ep:06d}.h5",
                    "length": length,
                    "split": "val" if ep == episodes - 1 and episodes > 1 else "train",
                    "sample_kind": "mock_dagger" if round_idx > 0 else "mock_expert",
                    "gates": [],
                }
            )
        (out_dir / "index.json").write_text(json.dumps(manifest, indent=2))
        train_files = [out_dir / ep["path"] for ep in manifest["episodes"] if ep.get("split") == "train"]
        write_norm_stats(out_dir, train_files, action_normalization="bounds")
        self._write_v3_norm_stats(out_dir, manifest)

    def _kept_round_dirs(self, collected_dirs: list[Path]) -> list[Path]:
        max_kept = self.distill_cfg.get("aggregate", {}).get("max_rounds_kept", "all")
        if max_kept == "all" or max_kept is None:
            return list(collected_dirs)
        return list(collected_dirs[-int(max_kept):])

    def _write_aggregate_dataset(self, round_idx: int, source_dirs: list[Path], out_dir: Path) -> Path:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "episodes").mkdir(parents=True)
        manifest: dict[str, Any] | None = None
        episodes_out: list[dict[str, Any]] = []
        next_ep_id = 0
        for source_dir in source_dirs:
            source_manifest = json.loads((source_dir / "index.json").read_text())
            if manifest is None:
                manifest = {k: copy.deepcopy(v) for k, v in source_manifest.items() if k != "episodes"}
            for ep in source_manifest.get("episodes", []):
                if ep.get("split") == "discard":
                    continue
                src = source_dir / ep["path"]
                dst_rel = Path("episodes") / f"r{round_idx:03d}_{next_ep_id:06d}.h5"
                dst = out_dir / dst_rel
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                ep_out = copy.deepcopy(ep)
                ep_out["episode_id"] = next_ep_id
                ep_out["path"] = str(dst_rel)
                episodes_out.append(ep_out)
                next_ep_id += 1
        if manifest is None or not episodes_out:
            raise RuntimeError("No distillation episodes were collected.")

        rng = np.random.default_rng(self.seed + 17 * round_idx)
        order = rng.permutation(len(episodes_out))
        n_val = int(round(self.val_frac * len(episodes_out)))
        if len(episodes_out) > 1:
            n_val = min(max(1, n_val), len(episodes_out) - 1)
        else:
            n_val = 0
        val_ids = set(int(i) for i in order[:n_val])
        for i, ep in enumerate(episodes_out):
            ep["split"] = "val" if i in val_ids else "train"

        manifest["collector"] = "src.robotics.distill_trainer.aggregate"
        manifest["distill_round"] = round_idx
        manifest["episodes"] = episodes_out
        manifest.setdefault("obs_v3", _OBS_V3_BLOCK)
        (out_dir / "index.json").write_text(json.dumps(manifest, indent=2))
        train_files = [out_dir / ep["path"] for ep in episodes_out if ep.get("split") == "train"]
        action_norm = str(manifest.get("action_normalization", self.data_cfg.get("action_normalization", "bounds")))
        write_norm_stats(
            out_dir,
            train_files,
            action_normalization=action_norm,
            skip_initial_frames=int(manifest.get("skip_initial_frames", 0)),
            bounds_margin=float(manifest.get("bounds_margin", 0.10)),
        )
        self._write_v3_norm_stats(out_dir, manifest)
        print(
            f"[Distill] aggregate round={round_idx} sources={len(source_dirs)} "
            f"episodes={len(episodes_out)} -> {out_dir}",
            flush=True,
        )
        return out_dir

    def _write_v3_norm_stats(self, data_dir: Path, manifest: dict) -> None:
        train_eps = [ep for ep in manifest["episodes"] if ep.get("split") == "train"]
        sums: dict[str, np.ndarray] = {}
        sq_sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        keys = {"proprio_core": "obs/proprio_core", "gate": "obs/gate", "aux": "obs/aux"}
        for ep in train_eps:
            with h5py.File(data_dir / ep["path"], "r") as h:
                for name, key in keys.items():
                    if key not in h:
                        raise RuntimeError(f"{data_dir / ep['path']} missing {key}; v3 distillation requires obs v3.")
                    arr = h[key][...].astype(np.float64, copy=False)
                    sums.setdefault(name, np.zeros(arr.shape[1], dtype=np.float64))
                    sq_sums.setdefault(name, np.zeros(arr.shape[1], dtype=np.float64))
                    counts[name] = counts.get(name, 0) + int(arr.shape[0])
                    sums[name] += arr.sum(axis=0)
                    sq_sums[name] += np.square(arr).sum(axis=0)
        existing = {}
        stats_path = data_dir / "norm_stats.npz"
        if stats_path.exists():
            with np.load(stats_path) as s:
                existing = {k: s[k] for k in s.files}
        for name in keys:
            mean = sums[name] / max(1, counts[name])
            var = sq_sums[name] / max(1, counts[name]) - np.square(mean)
            existing[f"{name}_mean"] = mean.astype(np.float32)
            existing[f"{name}_std"] = (np.sqrt(np.clip(var, 1e-8, None)) + 1e-6).astype(np.float32)
        np.savez(stats_path, **existing)

    def _run_inner_training(self, round_idx: int, data_dir: Path, train_dir: Path) -> Path:
        inner_cfg = copy.deepcopy(self.training_cfg)
        inner_cfg.update(copy.deepcopy(self.distill_cfg.get("inner_training", {}) or {}))
        inner_cfg["output_dir"] = str(train_dir)
        inner_cfg.setdefault("trainer_type", "bc_flow_matching_robosuite")
        max_steps = int(inner_cfg.get("max_steps", 0))
        if max_steps <= 0:
            ckpt_dir = train_dir / "final"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
            (ckpt_dir / "training_state.json").write_text(json.dumps({"step": 0, "round": round_idx}, indent=2))
            print(f"[Distill] skipped inner training for round={round_idx}; wrote {ckpt_dir}", flush=True)
            return ckpt_dir

        data_cfg = copy.deepcopy(self.data_cfg)
        data_cfg["data_dir"] = str(data_dir)
        dataset = build("data", **data_cfg)
        trainer_type = inner_cfg.get("trainer_type", "bc_flow_matching_robosuite")
        trainer = build(
            "trainer",
            type=trainer_type,
            model=self.model,
            dataset=dataset,
            training_cfg=inner_cfg,
            robotics_cfg=self.robotics_cfg,
            wandb_cfg=self.wandb_cfg,
            data_cfg=data_cfg,
        )
        trainer.train()
        best = train_dir / "best"
        final = train_dir / "final"
        ckpt_dir = best if (best / "model.pt").exists() else final
        self._load_checkpoint_into_model(ckpt_dir / "model.pt")
        return ckpt_dir

    def _evaluate_round(self, round_idx: int, data_dir: Path, checkpoint_dir: Path, round_dir: Path) -> dict:
        rollout_cfg = copy.deepcopy(self.training_cfg.get("rollout_eval", {}) or {})
        if not bool(rollout_cfg.get("enabled", False)):
            return {}
        from src.robotics.flightmare_policy_eval import evaluate_flightmare_policy

        data_cfg = copy.deepcopy(self.data_cfg)
        data_cfg["data_dir"] = str(data_dir)
        self._load_checkpoint_into_model(checkpoint_dir / "model.pt")
        metrics = evaluate_flightmare_policy(
            self.model,
            data_cfg=data_cfg,
            robotics_cfg=self.robotics_cfg,
            eval_cfg=rollout_cfg,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._write_json(round_dir / "rollout_metrics.json", metrics)
        print(
            f"[Distill] round={round_idx} rollout score={metrics.get('score', 0.0):.4f} "
            f"gate_completion={metrics.get('mean_gate_completion', 0.0):.3f}",
            flush=True,
        )
        return metrics

    def _maybe_promote_best(self, checkpoint_dir: Path, metrics: dict) -> None:
        score = float(metrics.get("score", 0.0)) if metrics else (0.0 if self.best_checkpoint is None else self.best_score)
        if self.best_checkpoint is not None and score < self.best_score:
            return
        self.best_score = score
        self.best_checkpoint = checkpoint_dir
        dst = self.output_dir / "best"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(checkpoint_dir, dst)
        print(f"[Distill] promoted best score={self.best_score:.4f} -> {dst}", flush=True)

    def _write_policy_config(self, round_idx: int, data_dir: Path, path: Path) -> Path:
        cfg = self._config_as_dict()
        cfg.setdefault("data", {})
        cfg["data"]["data_dir"] = str(data_dir)
        cfg.setdefault("training", {})
        cfg["training"]["resume_from"] = None
        cfg["run"] = dict(cfg.get("run", {}), mode="bc", name=f"{cfg.get('run', {}).get('name', 'distill')}_round_{round_idx:03d}")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    def _config_as_dict(self) -> dict:
        if self.full_cfg is None:
            return {
                "run": {},
                "data": copy.deepcopy(self.data_cfg),
                "robotics": copy.deepcopy(self.robotics_cfg),
                "training": copy.deepcopy(self.training_cfg),
                "wandb": copy.deepcopy(self.wandb_cfg),
            }
        if is_dataclass(self.full_cfg):
            return asdict(self.full_cfg)
        return copy.deepcopy(self.full_cfg)

    def _load_checkpoint_into_model(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)

    @staticmethod
    def _model_path(path: Path) -> Path:
        return path / "model.pt" if path.is_dir() else path

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _jsonable(value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(k): DistillationTrainer._jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [DistillationTrainer._jsonable(v) for v in value]
            return str(value)


@register("trainer", "flightmare_dagger_distill")
def build_distillation_trainer(
    model: torch.nn.Module,
    training_cfg: dict,
    robotics_cfg: dict = None,
    wandb_cfg: dict = None,
    data_cfg: dict = None,
    distill_cfg: dict = None,
    full_cfg: Any = None,
    **_: Any,
) -> DistillationTrainer:
    return DistillationTrainer(
        model=model,
        training_cfg=training_cfg,
        robotics_cfg=robotics_cfg,
        wandb_cfg=wandb_cfg,
        data_cfg=data_cfg,
        distill_cfg=distill_cfg,
        full_cfg=full_cfg,
    )

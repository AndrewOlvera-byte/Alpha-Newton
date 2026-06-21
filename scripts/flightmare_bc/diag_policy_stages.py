"""Evaluate a Flightmare PPO policy across curriculum stages with diagnostics.

This is intentionally policy/env local: it bypasses the autonomy graph and
steps ``FlightmareRacingEnv`` the same way PPO does. It reports whether a
checkpoint is using the v3 reference-gate observation, whether normalized obs
blocks are in a sane range, whether Gaussian samples are being clipped by the
env, and which reward terms dominate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.config import Config
from src.core.registry import build
from src.robotics.flightmare_envs import build_flightmare_env_config, FlightmareRacingEnv

# Register builders.
import src.robotics.models.flightmare  # noqa: F401
import src.robotics.flightmare_envs  # noqa: F401


def _stage_overrides(cfg: Config, stage_name: str | None) -> dict[str, Any]:
    ppo = ((cfg.robotics or {}).get("ppo", {}) or {})
    if not stage_name:
        return {}
    stages = ((ppo.get("curriculum", {}) or {}).get("stages", []) or [])
    curriculum_keys = {
        "name",
        "until_iter",
        "advance_metric",
        "advance_threshold",
        "advance_after_iter",
        "advance_after_stage_iters",
        "advance_patience",
        "advance_on_timeout",
        "duration_iters",
        "ent_coeff_override",
        "actor_lr_override",
    }
    for raw in stages:
        if str(raw.get("name")) == stage_name:
            return {k: v for k, v in dict(raw).items() if k not in curriculum_keys}
    raise SystemExit(f"stage {stage_name!r} not found")


def _base_env_kwargs(cfg: Config) -> dict[str, Any]:
    ppo = ((cfg.robotics or {}).get("ppo", {}) or {})
    data = cfg.data or {}
    return {
        "data_dir": data.get("data_dir", "data/flightmare/bc_v4"),
        "action_type": data.get("action_type", ppo.get("action_type", "ctbr")),
        "obs_schema": data.get("obs_schema", ppo.get("obs_schema", "v3")),
        "include_mission": data.get("include_mission", True),
        "normalize_state": data.get("normalize_state", True),
        "normalize_mission": data.get("normalize_mission", True),
        "normalize_action": data.get("normalize_action", True),
        "action_normalization": data.get("action_normalization", "auto"),
        "control_hz": ppo.get("control_hz", 100.0),
        "horizon": ppo.get("horizon", 1500),
        "scene": ppo.get("scene", "industrial"),
        "render": ppo.get("render", False),
        "backend": ppo.get("backend", "flightgym"),
        "course_mode": ppo.get("course_mode", "swift_v4"),
        "gate_layout": ppo.get("gate_layout"),
        "random_start_gate": ppo.get("random_start_gate", False),
        "fixed_gate_pos_noise": ppo.get("fixed_gate_pos_noise", 0.0),
        "fixed_gate_pos_noise_xyz": ppo.get("fixed_gate_pos_noise_xyz"),
        "fixed_gate_yaw_noise": ppo.get("fixed_gate_yaw_noise", 0.0),
        "num_gates": ppo.get("num_gates", 7),
        "gate_spacing_range": ppo.get("gate_spacing_range", [4.0, 9.0]),
        "gate_lateral_jitter": ppo.get("gate_lateral_jitter", 2.0),
        "gate_z_range": ppo.get("gate_z_range", [1.5, 4.0]),
        "gate_yaw_step": ppo.get("gate_yaw_step", 0.7),
        "gate_yaw_noise": ppo.get("gate_yaw_noise", 0.25),
        "gate_size": ppo.get("gate_size", 1.0),
        "gate_approach_m": ppo.get("gate_approach_m", 1.2),
        "inverted_roll_jitter_rad": ppo.get("inverted_roll_jitter_rad", 0.2618),
        "z_min": ppo.get("z_min", 1.0),
        "gate_vehicle_radius": ppo.get("gate_vehicle_radius", 0.15),
        "max_world_radius": ppo.get("max_world_radius", 120.0),
        "min_z": ppo.get("min_z", -0.25),
        "terminate_on_gate_miss": ppo.get("terminate_on_gate_miss", True),
        "terminate_on_crash": ppo.get("terminate_on_crash", True),
        "reward_fn": ppo.get("reward_fn", "flightmare_racing_v2"),
        "reward_kwargs": ppo.get("reward_kwargs", {}),
        "max_body_rate": ppo.get("max_body_rate", 8.0),
        "max_waypoint_speed": ppo.get("max_waypoint_speed", 15.0),
        "max_collective_thrust_g": ppo.get("max_collective_thrust_g", 4.0),
        "plant": ppo.get("plant"),
        "action_bounds_override": ppo.get("action_bounds_override"),
        "reset_mode": ppo.get("reset_mode", "course_start"),
        "course_start_sample_prob": ppo.get("course_start_sample_prob", 0.0),
        "start_gate_index": ppo.get("start_gate_index"),
        "start_gate_choices": ppo.get("start_gate_choices"),
        "start_offset_m": ppo.get("start_offset_m", 3.0),
        "start_offset_range": ppo.get("start_offset_range"),
        "start_lateral_range_m": ppo.get("start_lateral_range_m"),
        "start_vertical_range_m": ppo.get("start_vertical_range_m"),
        "start_yaw_noise_rad": ppo.get("start_yaw_noise_rad", 0.0),
        "start_speed_noise_mps": ppo.get("start_speed_noise_mps", 0.0),
        "reference_speed_mps": ppo.get("reference_speed_mps", 4.0),
        "goal_mode": ppo.get("goal_mode", "finish_remaining_course"),
        "goal_gate_span": ppo.get("goal_gate_span"),
        "post_pass_success_steps": ppo.get("post_pass_success_steps", 0),
        "post_pass_max_speed_mps": ppo.get("post_pass_max_speed_mps"),
        "post_pass_max_body_rate": ppo.get("post_pass_max_body_rate"),
        "post_pass_max_center_error_norm": ppo.get("post_pass_max_center_error_norm"),
        "post_pass_min_gate_signed_distance_m": ppo.get("post_pass_min_gate_signed_distance_m"),
        "trajectory_replay_capacity_per_gate": ppo.get("trajectory_replay_capacity_per_gate", 0),
        "trajectory_replay_sample_prob": ppo.get("trajectory_replay_sample_prob", 0.0),
        "trajectory_replay_min_samples_per_gate": ppo.get("trajectory_replay_min_samples_per_gate", 1),
        "trajectory_replay_pos_noise_m": ppo.get("trajectory_replay_pos_noise_m", 0.0),
        "trajectory_replay_vel_noise_mps": ppo.get("trajectory_replay_vel_noise_mps", 0.0),
        "trajectory_replay_omega_noise_radps": ppo.get("trajectory_replay_omega_noise_radps", 0.0),
        "trajectory_replay_yaw_noise_rad": ppo.get("trajectory_replay_yaw_noise_rad", 0.0),
        "action_clip": ppo.get(
            "action_clip",
            ((cfg.robotics or {}).get("architecture", {}) or {}).get("action_clip", 5.0),
        ),
    }


def _merge_stage_env_kwargs(base: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    stage = dict(stage)
    reward_overrides = stage.pop("reward_kwargs", None)
    plant_overrides = stage.pop("plant", None)
    bounds_overrides = stage.pop("action_bounds_override", None)
    merged.update(stage)
    if reward_overrides is not None:
        merged["reward_kwargs"] = {
            **(base.get("reward_kwargs", {}) or {}),
            **(reward_overrides or {}),
        }
    if plant_overrides is not None:
        merged["plant"] = {
            **(base.get("plant") or {}),
            **(plant_overrides or {}),
        }
    if bounds_overrides is not None:
        merged["action_bounds_override"] = {
            **(base.get("action_bounds_override") or {}),
            **(bounds_overrides or {}),
        }
    return merged


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
    return state


def _build_model(cfg: Config, checkpoint: Path, device: torch.device):
    arch = dict((cfg.robotics or {}).get("architecture", {}) or {})
    model = build("architecture", **arch)
    state = _load_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[model] missing={len(missing)} first={missing[:5]}")
    if unexpected:
        print(f"[model] unexpected={len(unexpected)} first={unexpected[:5]}")
    model.to(device).eval()
    return model


def _tensor_batch(obs: dict, prev_action: np.ndarray, device: torch.device) -> dict:
    return {
        "images": {},
        "state": torch.from_numpy(obs["state"][None].astype(np.float32)).to(device),
        "prev_actions": torch.from_numpy(prev_action[None].astype(np.float32)).to(device),
    }


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def run_one(
    *,
    cfg: Config,
    model,
    checkpoint: Path,
    stage_name: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    max_steps: int | None,
    start_gate: int | None,
    device: torch.device,
) -> None:
    kwargs = _merge_stage_env_kwargs(_base_env_kwargs(cfg), _stage_overrides(cfg, stage_name))
    if start_gate is not None:
        kwargs["start_gate_index"] = int(start_gate)
        kwargs["start_gate_choices"] = [int(start_gate)]
    if max_steps is not None:
        kwargs["horizon"] = int(max_steps)
    kwargs["seed"] = int(seed)

    env = FlightmareRacingEnv(build_flightmare_env_config(**kwargs))
    action_dim = int(model.action_dim)
    action_clip = float(kwargs.get("action_clip", 1.0))

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    success: list[float] = []
    gate_completion: list[float] = []
    goal_completion: list[float] = []
    gate_passes = 0
    gate_misses = 0
    crashes = 0
    reasons = Counter()
    start_gates = Counter()
    obs_abs_max: list[float] = []
    obs_abs_mean: list[float] = []
    block_abs = {"proprio": [], "gate": [], "aux": [], "prev": []}
    action_pre_abs_max: list[float] = []
    action_post_abs_max: list[float] = []
    action_clip_frac: list[float] = []
    action_mu_mean: list[np.ndarray] = []
    raw_action_mean: list[np.ndarray] = []
    reward_terms: dict[str, list[float]] = defaultdict(list)
    progress_terms = defaultdict(list)

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            prev = np.asarray(info.get("initial_prev_action_norm", np.zeros(action_dim)), dtype=np.float32)
            start_gates[int(info.get("start_gate_index", -1))] += 1
            total = 0.0
            done = False
            last_info = info
            steps = 0
            while not done:
                batch = _tensor_batch(obs, prev, device)
                with torch.no_grad():
                    out = model(batch)
                    mu = out["mu"]
                    if deterministic:
                        action_t = mu.clamp(-action_clip, action_clip)
                    else:
                        action_t, _, _ = model.act(batch)
                action = action_t.float().cpu().numpy()[0]
                mu_np = mu.float().cpu().numpy()[0]
                clipped = np.clip(action, -action_clip, action_clip)

                s = np.asarray(obs["state"], dtype=np.float32)
                obs_abs_max.append(float(np.max(np.abs(s))))
                obs_abs_mean.append(float(np.mean(np.abs(s))))
                block_abs["proprio"].append(float(np.mean(np.abs(s[:9]))))
                block_abs["gate"].append(float(np.mean(np.abs(s[9:33]))))
                block_abs["aux"].append(float(np.mean(np.abs(s[33:36]))))
                block_abs["prev"].append(float(np.mean(np.abs(prev))))
                action_pre_abs_max.append(float(np.max(np.abs(action))))
                action_post_abs_max.append(float(np.max(np.abs(clipped))))
                action_clip_frac.append(float(np.mean(np.abs(action) > action_clip)))
                action_mu_mean.append(mu_np)

                obs, rew, term, trunc, info = env.step(action)
                prev = np.asarray(info.get("action_norm", clipped), dtype=np.float32)
                raw = np.asarray(info.get("raw_action", np.zeros(action_dim)), dtype=np.float32)
                raw_action_mean.append(raw)
                total += float(rew)
                steps += 1
                last_info = info
                if info.get("gate_passed"):
                    gate_passes += 1
                if info.get("gate_missed"):
                    gate_misses += 1
                if info.get("crash"):
                    crashes += 1
                for key, value in (info.get("reward_terms") or {}).items():
                    reward_terms[key].append(float(value))
                for key in (
                    "gate_normal_progress_m",
                    "segment_progress_m",
                    "gate_normal_velocity_mps",
                    "gate_lateral_norm",
                    "gate_vertical_norm",
                    "gate_signed_distance_m",
                    "speed_mps",
                ):
                    progress_terms[key].append(float(info.get(key, 0.0)))
                done = bool(term or trunc)
            ep_returns.append(total)
            ep_lengths.append(steps)
            success.append(float(last_info.get("success", False)))
            gate_completion.append(float(last_info.get("gate_completion", 0.0)))
            goal_completion.append(float(last_info.get("goal_completion", 0.0)))
            reason = last_info.get("crash_reason") if last_info.get("crash") else None
            if last_info.get("gate_missed"):
                reason = "gate_miss"
            if last_info.get("success"):
                reason = "success"
            if reason is None and steps >= int(kwargs["horizon"]):
                reason = "horizon"
            reasons[str(reason or "other")] += 1
    finally:
        env.close()

    mu_arr = np.stack(action_mu_mean, axis=0) if action_mu_mean else np.zeros((1, action_dim), dtype=np.float32)
    raw_arr = np.stack(raw_action_mean, axis=0) if raw_action_mean else np.zeros((1, action_dim), dtype=np.float32)
    print(
        f"\n[diag-policy] ckpt={checkpoint} stage={stage_name} "
        f"mode={'det' if deterministic else 'stoch'} episodes={episodes}"
    )
    print(
        f"  reset_mode={kwargs.get('reset_mode')} goal_mode={kwargs.get('goal_mode')} "
        f"horizon={kwargs.get('horizon')} starts={dict(sorted(start_gates.items()))}"
    )
    print(
        "  outcome "
        f"success={_safe_mean(success):.3f} gate_completion={_safe_mean(gate_completion):.3f} "
        f"goal_completion={_safe_mean(goal_completion):.3f} "
        f"return={_safe_mean(ep_returns):.2f} len={_safe_mean(ep_lengths):.1f} "
        f"pass={gate_passes} miss={gate_misses} crash={crashes} reasons={dict(reasons)}"
    )
    print(
        "  obs_norm "
        f"abs_mean={_safe_mean(obs_abs_mean):.3f} abs_max_p95={np.percentile(obs_abs_max, 95) if obs_abs_max else 0:.3f} "
        f"proprio={_safe_mean(block_abs['proprio']):.3f} gate={_safe_mean(block_abs['gate']):.3f} "
        f"aux={_safe_mean(block_abs['aux']):.3f} prev={_safe_mean(block_abs['prev']):.3f}"
    )
    print(
        "  action_norm "
        f"pre_abs_max_p95={np.percentile(action_pre_abs_max, 95) if action_pre_abs_max else 0:.3f} "
        f"post_abs_max_p95={np.percentile(action_post_abs_max, 95) if action_post_abs_max else 0:.3f} "
        f"clip_frac={_safe_mean(action_clip_frac):.3f} "
        f"mu_mean={np.round(mu_arr.mean(axis=0), 3).tolist()} mu_std={np.round(mu_arr.std(axis=0), 3).tolist()}"
    )
    print(
        "  raw_action "
        f"mean={np.round(raw_arr.mean(axis=0), 3).tolist()} std={np.round(raw_arr.std(axis=0), 3).tolist()} "
        f"min={np.round(raw_arr.min(axis=0), 3).tolist()} max={np.round(raw_arr.max(axis=0), 3).tolist()}"
    )
    if progress_terms:
        msg = "  geometry "
        for key in ("gate_normal_progress_m", "segment_progress_m", "gate_normal_velocity_mps", "gate_lateral_norm", "gate_vertical_norm", "gate_signed_distance_m", "speed_mps"):
            vals = progress_terms.get(key, [])
            if vals:
                msg += f"{key}={_safe_mean(vals):+.3f} "
        print(msg.rstrip())
    if reward_terms:
        means = {k: _safe_mean(v) for k, v in reward_terms.items()}
        top = sorted(means.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
        print("  reward_terms " + " ".join(f"{k}={v:+.3f}" for k, v in top))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", default="flightmare/paper/ctbr_curriculum_ppo")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stage", action="append", required=True, help="Curriculum stage name; may be repeated.")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--start-gate", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = Config.from_experiment(args.exp)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = _build_model(cfg, args.checkpoint, device)
    for stage in args.stage:
        run_one(
            cfg=cfg,
            model=model,
            checkpoint=args.checkpoint,
            stage_name=stage,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            start_gate=args.start_gate,
            device=device,
        )


if __name__ == "__main__":
    main()

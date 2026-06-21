"""Evaluate the BC expert controller directly in Flightmare.

This is a data-quality gate for collection configs: before trusting labels from
an expert, verify the same controller/plant/course setup actually flies through
strict gate apertures in closed loop.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.flightmare_bc.collection_config import load_collection_config
from scripts.flightmare_bc.collect import (
    _gate_record,
    sample_course_with_meta,
    waypoints_from_gates,
)
from scripts.flightmare_bc.collect_dataset import _collector_namespace, _expert_controller, _quad_params
from scripts.flightmare_bc.controllers import GeometricSE3Controller
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from scripts.flightmare_bc.trajectories import MinJerkTrajectory
from src.robotics.flightmare_autonomy_fsw.gates import StrictGateTracker


def _speed_bins(speed_range: list[float], bins: int) -> list[tuple[float, float]]:
    lo, hi = float(speed_range[0]), float(speed_range[1])
    if bins <= 1:
        return [(lo, hi)]
    edges = np.linspace(lo, hi, int(bins) + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(int(bins))]


def _rollout_one(
    env: FlightmareExpertEnv,
    controller: GeometricSE3Controller,
    rng: np.random.Generator,
    cfg_ns: SimpleNamespace,
    *,
    speed: float,
    episode_id: int,
) -> dict:
    gates, scenario_meta = sample_course_with_meta(rng, cfg_ns)
    for gate in gates:
        gate.gate_id = f"controller_eval_ep{episode_id:06d}_{gate.gate_id}"
    env.add_gates(gates)
    waypoints = waypoints_from_gates(
        gates,
        cfg_ns.z_min,
        d_approach=float(cfg_ns.gate_approach_m),
    )
    obs = env.reset(init_pos=waypoints[0], yaw=float(gates[0].yaw) if gates else 0.0)
    traj = MinJerkTrajectory(
        waypoints,
        avg_speed=float(speed),
        yaw_mode="tangent",
        min_segment_s=float(cfg_ns.trajectory_min_segment_s),
    )
    strict = StrictGateTracker(gates, vehicle_radius=float(cfg_ns.gate_vehicle_radius))
    if hasattr(controller, "reset"):
        controller.reset()
    n_steps = max(1, min(int(traj.total_time / env.dt), int(cfg_ns.max_steps)))
    pos_errors: list[float] = []
    speeds: list[float] = []
    thrusts: list[float] = []
    body_rates: list[np.ndarray] = []
    first_event = None
    prev_pos = obs.pos.copy()
    termination_reason = "max_steps"
    for k in range(n_steps):
        t = k * env.dt
        ref = traj.sample(t)
        cmd = controller.compute(
            pos=obs.pos,
            vel=obs.vel,
            quat=obs.quat,
            pos_des=ref.pos,
            vel_des=ref.vel,
            acc_des=ref.acc,
            yaw_des=ref.yaw,
            omega=obs.omega,
        )
        pos_errors.append(float(np.linalg.norm(obs.pos - ref.pos)))
        speeds.append(float(np.linalg.norm(obs.vel)))
        thrusts.append(float(cmd["thrust_normalized"]))
        body_rates.append(np.asarray(cmd["body_rates"], dtype=np.float32))
        obs = env.step_ctbr(cmd["thrust_newton"], cmd["body_rates"])
        event = strict.update(prev_pos, obs.pos)
        if event is not None and first_event is None:
            first_event = event
        prev_pos = obs.pos.copy()
        if event is not None and event.missed:
            termination_reason = "gate_miss"
            break
        if strict.completed:
            termination_reason = "completed"
            break
        if obs.done:
            termination_reason = "env_done"
            break
        if not np.all(np.isfinite(obs.pos)):
            termination_reason = "nonfinite"
            break
        if float(np.linalg.norm(obs.pos)) > float(cfg_ns.max_world_radius):
            termination_reason = "out_of_bounds"
            break
    body = np.stack(body_rates, axis=0) if body_rates else np.zeros((0, 3), dtype=np.float32)
    return {
        "episode_id": int(episode_id),
        "speed": float(speed),
        "completed": bool(strict.completed),
        "termination_reason": termination_reason,
        "gates_completed": int(strict.current_index),
        "num_gates": int(len(gates)),
        "gate_completion": float(strict.current_index / max(1, len(gates))),
        "gate_misses": int(strict.misses),
        "first_gate_margin_m": (
            None if first_event is None else float(first_event.clearance_margin_m)
        ),
        "first_gate_lateral_m": None if first_event is None else float(first_event.lateral_m),
        "first_gate_vertical_m": None if first_event is None else float(first_event.vertical_m),
        "mean_track_err_m": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_track_err_m": float(np.max(pos_errors)) if pos_errors else 0.0,
        "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed_mps": float(np.max(speeds)) if speeds else 0.0,
        "thrust_saturation_frac": float(np.mean((np.asarray(thrusts) <= 0.02) | (np.asarray(thrusts) >= 0.98))) if thrusts else 0.0,
        "body_rate_abs_max": float(np.max(np.abs(body))) if body.size else 0.0,
        "scenario_family": (scenario_meta or {}).get("scenario_family", "legacy"),
        "scenario_name": (scenario_meta or {}).get("scenario_name", cfg_ns.course_mode),
        "gates": [_gate_record(g) for g in gates],
    }


def summarize_controller_results(results: list[dict]) -> dict:
    if not results:
        return {}
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r["completed"] for r in results])),
        "mean_gate_completion": float(np.mean([r["gate_completion"] for r in results])),
        "mean_gates_completed": float(np.mean([r["gates_completed"] for r in results])),
        "gate_miss_rate": float(np.mean([r["gate_misses"] > 0 for r in results])),
        "mean_track_err_m": float(np.mean([r["mean_track_err_m"] for r in results])),
        "max_track_err_m": float(np.max([r["max_track_err_m"] for r in results])),
        "mean_speed_mps": float(np.mean([r["mean_speed_mps"] for r in results])),
        "first_gate_margin_mean_m": float(np.mean([r["first_gate_margin_m"] for r in results if r["first_gate_margin_m"] is not None]))
        if any(r["first_gate_margin_m"] is not None for r in results)
        else None,
    }


def _scenario_variants(cfg_ns: SimpleNamespace) -> list[tuple[str, SimpleNamespace]]:
    """Per-scenario cfg variants so each family is evaluated independently.

    A distribution with N families is split into N single-family distributions
    (weight 1 each); legacy single-mode configs yield one variant.
    """
    from copy import copy

    from src.robotics.flightmare_courses import course_distribution_from_dict
    from src.robotics.flightmare_courses.scenarios import CourseDistributionConfig

    dist = getattr(cfg_ns, "course_distribution", None)
    if dist is None:
        return [("legacy", cfg_ns)]
    if not isinstance(dist, CourseDistributionConfig):
        dist = course_distribution_from_dict(dist)
    variants: list[tuple[str, SimpleNamespace]] = []
    for sc in dist.scenarios:
        ns = copy(cfg_ns)
        ns.course_distribution = CourseDistributionConfig(
            scenarios=[sc], bounds=dist.bounds, max_resample_tries=dist.max_resample_tries
        )
        variants.append((sc.name, ns))
    return variants


def evaluate_controller_config(
    cfg: dict,
    *,
    episodes: int | None = None,
    speed_range: list[float] | None = None,
    speed_bins: int = 1,
    seed: int | None = None,
) -> dict:
    params = _quad_params(cfg)
    controller = _expert_controller(cfg, params)
    cfg_ns = _collector_namespace(cfg)
    if speed_range is None:
        speed_range = list(cfg["expert"]["speed_range"])
    if episodes is None:
        episodes = int(cfg.get("controller_eval", {}).get("episodes", 8))
    rng = np.random.default_rng(int(cfg["seed"] if seed is None else seed))
    env = FlightmareExpertEnv(
        image_size=int(cfg["image_size"]),
        cameras=tuple(cfg.get("cameras", [])),
        control_hz=float(cfg["control_hz"]),
        params=params,
        scene=str(cfg["scene"]),
        render=bool(cfg.get("render", False)),
        seed=int(cfg["seed"] if seed is None else seed),
        visual_backend=True,
        backend=str(cfg["backend"]),
    )
    bins = _speed_bins(speed_range, speed_bins)
    variants = _scenario_variants(cfg_ns)
    results: list[dict] = []
    try:
        episode_id = 0
        for _family, variant_ns in variants:
            for lo, hi in bins:
                for _ in range(int(episodes)):
                    speed = float(rng.uniform(lo, hi))
                    results.append(
                        _rollout_one(
                            env,
                            controller,
                            rng,
                            variant_ns,
                            speed=speed,
                            episode_id=episode_id,
                        )
                    )
                    episode_id += 1
    finally:
        env.close()

    # Aggregate by speed bin (across scenarios) and by scenario family.
    by_bin = []
    for lo, hi in bins:
        chunk = [r for r in results if lo - 1e-9 <= r["speed"] <= hi + 1e-9]
        by_bin.append({"speed_min": lo, "speed_max": hi, **summarize_controller_results(chunk)})
    families = sorted({r.get("scenario_family", "legacy") for r in results})
    by_scenario = []
    for fam in families:
        chunk = [r for r in results if r.get("scenario_family", "legacy") == fam]
        by_scenario.append({"scenario_family": fam, **summarize_controller_results(chunk)})

    summary = summarize_controller_results(results)
    # Worst-family stats so the preflight gate can refuse on any single bad family.
    if by_scenario:
        summary["min_scenario_success_rate"] = float(min(s.get("success_rate", 0.0) for s in by_scenario))
        summary["min_scenario_gate_completion"] = float(min(s.get("mean_gate_completion", 0.0) for s in by_scenario))
        summary["worst_scenario_family"] = min(
            by_scenario, key=lambda s: (s.get("success_rate", 0.0), s.get("mean_gate_completion", 0.0))
        )["scenario_family"]
    return {
        "summary": summary,
        "by_speed_bin": by_bin,
        "by_scenario": by_scenario,
        "episodes": results,
        "plant": asdict(params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/flightmare_bc/ctbr_v6.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--speed-range", nargs=2, type=float, default=None)
    parser.add_argument("--speed-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_collection_config(args.config)
    result = evaluate_controller_config(
        cfg,
        episodes=args.episodes,
        speed_range=args.speed_range,
        speed_bins=args.speed_bins,
        seed=args.seed,
    )
    print("[controller-eval] summary")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print("[controller-eval] by speed bin")
    for row in result["by_speed_bin"]:
        print(
            f"  {row['speed_min']:.2f}-{row['speed_max']:.2f} m/s: "
            f"success={100.0 * row.get('success_rate', 0.0):.1f}% "
            f"gate_completion={100.0 * row.get('mean_gate_completion', 0.0):.1f}% "
            f"track_err={row.get('mean_track_err_m', 0.0):.3f}m"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"[controller-eval] wrote {args.output}")


if __name__ == "__main__":
    main()

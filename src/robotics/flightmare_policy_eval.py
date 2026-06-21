from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.mission import MISSION_DIM
from src.core.registry import build
from src.robotics.flightmare_autonomy_fsw.controllers import (
    BaseAutonomyController,
    BaseCTBRController,
    BaseMotorController,
    WaypointLQRController,
)
from src.robotics.flightmare_autonomy_fsw.graph import FlightmareAutonomyGraph
from src.robotics.flightmare_autonomy_fsw.nodes import (
    BaseControllerNode,
    CourseConfig,
    FlightmareStateNode,
    MissionWorldModelNode,
    PolicyPlannerNode,
)
from src.robotics.flightmare_autonomy_fsw.plotting import summarize_results
from src.robotics.models.flightmare.MissionWrapper import MissionWrapper


PLANT_KEYS = (
    "mass",
    "arm_length",
    "k_thrust",
    "k_torque",
    "inertia",
    "motor_omega_min",
    "motor_omega_max",
    "motor_tau",
    "thrust_map",
    "kappa",
    "omega_max_body",
)


def load_dataset_manifest(data_dir: str | Path) -> dict:
    index_path = Path(data_dir) / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        return json.load(f)


def plant_config_from_manifest(manifest: dict | None) -> dict:
    params = (manifest or {}).get("params", {}) or {}
    return {key: params[key] for key in PLANT_KEYS if key in params and params[key] is not None}


def apply_plant_overrides(params: QuadParams, plant: dict | None) -> QuadParams:
    if not plant:
        return params
    for key in PLANT_KEYS:
        if key not in plant or plant[key] is None:
            continue
        value = plant[key]
        if key in {"inertia", "thrust_map", "omega_max_body"}:
            value = tuple(float(x) for x in value)
        else:
            value = float(value)
        setattr(params, key, value)
    return params


def resolved_plant_config(ppo_cfg: dict | None, dataset_manifest: dict | None) -> dict:
    ppo_cfg = ppo_cfg or {}
    return dict(ppo_cfg.get("plant") or plant_config_from_manifest(dataset_manifest))


def manifest_first_gates(manifest: dict) -> list[dict]:
    for ep in manifest.get("episodes", []):
        gates = ep.get("gates", [])
        if gates:
            return gates
    return []


def manifest_num_gates(manifest: dict, default: int) -> int:
    gates = manifest_first_gates(manifest)
    return len(gates) if gates else int(default)


def manifest_gate_size(manifest: dict, default: float) -> float:
    gates = manifest_first_gates(manifest)
    if not gates:
        return float(default)
    size = gates[0].get("size", default)
    if isinstance(size, list):
        return float(size[0])
    return float(size)


def manifest_thrust_g(manifest: dict, default: float) -> float:
    params = manifest.get("params", {}) or {}
    try:
        return float(params["max_collective_thrust"]) / (float(params["mass"]) * float(params["g"]))
    except Exception:
        return float(default)


def infer_action_type(data_cfg: dict | None, robotics_cfg: dict | None) -> str:
    data_cfg = data_cfg or {}
    robotics_cfg = robotics_cfg or {}
    data_action = data_cfg.get("action_type")
    arch_cfg = robotics_cfg.get("architecture", {}) or {}
    arch_type = str(arch_cfg.get("type", ""))
    arch_action = None
    if "waypoint" in arch_type:
        arch_action = "waypoint"
    elif "ctbr" in arch_type:
        arch_action = "ctbr"
    elif "motor" in arch_type:
        arch_action = "motor"
    action_type = data_action or arch_cfg.get("action_type") or arch_action
    if action_type not in {"waypoint", "ctbr", "motor"}:
        raise ValueError("Flightmare action_type must be one of waypoint, ctbr, or motor.")
    if arch_action is not None and data_action is not None and arch_action != data_action:
        raise ValueError(
            f"Config action mismatch: data.action_type={data_action!r}, "
            f"architecture.type={arch_type!r} implies {arch_action!r}."
        )
    return str(action_type)


def _cfg_value(
    eval_cfg: dict,
    ppo_cfg: dict,
    manifest: dict,
    key: str,
    default: Any = None,
    *,
    ppo_key: str | None = None,
) -> Any:
    if key in eval_cfg and eval_cfg[key] is not None:
        return eval_cfg[key]
    return ppo_cfg.get(ppo_key or key, manifest.get(key, default))


def build_flightmare_model(arch_cfg: dict, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    cfg = dict(arch_cfg)
    if not cfg.get("type"):
        raise ValueError("robotics.architecture.type must be set in the config.")
    model = build("architecture", **cfg)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _saturation_metrics(
    results,
    *,
    max_body_rate: float,
    thrust_eps: float = 0.02,
    body_rate_margin: float = 0.98,
) -> dict[str, float]:
    ctbr = [r.ctbr_commands for r in results if getattr(r, "ctbr_commands", np.zeros((0, 4))).size]
    planner = [r.planner_actions for r in results if getattr(r, "planner_actions", np.zeros((0, 4))).size]
    if ctbr:
        commands = np.concatenate(ctbr, axis=0)
        thrust = commands[:, 0]
        body = np.abs(commands[:, 1:4])
        thrust_sat = np.mean((thrust <= thrust_eps) | (thrust >= 1.0 - thrust_eps))
        body_sat = np.mean(body >= float(max_body_rate) * float(body_rate_margin))
    else:
        thrust_sat = 0.0
        body_sat = 0.0
    if planner:
        actions = np.concatenate(planner, axis=0)
        action_abs_p95 = float(np.percentile(np.abs(actions), 95.0))
    else:
        action_abs_p95 = 0.0
    return {
        "thrust_saturation_frac": float(thrust_sat),
        "body_rate_saturation_frac": float(body_sat),
        "planner_action_abs_p95": action_abs_p95,
    }


def score_flightmare_rollout(summary: dict, weights: dict | None = None) -> float:
    weights = weights or {}
    w = {
        "success_rate": float(weights.get("success_rate", 4.0)),
        "mean_gate_completion": float(weights.get("mean_gate_completion", 8.0)),
        "mean_speed_mps": float(weights.get("mean_speed_mps", 0.05)),
        "gate_miss_rate": float(weights.get("gate_miss_rate", -2.0)),
        "thrust_saturation_frac": float(weights.get("thrust_saturation_frac", -0.5)),
        "body_rate_saturation_frac": float(weights.get("body_rate_saturation_frac", -0.5)),
        "mean_final_distance_to_gate_m": float(weights.get("mean_final_distance_to_gate_m", -0.02)),
    }
    return float(
        w["success_rate"] * float(summary.get("success_rate", 0.0))
        + w["mean_gate_completion"] * float(summary.get("mean_gate_completion", 0.0))
        + w["mean_speed_mps"] * float(summary.get("mean_speed_mps", 0.0))
        + w["gate_miss_rate"] * float(summary.get("gate_miss_rate", 0.0))
        + w["thrust_saturation_frac"] * float(summary.get("thrust_saturation_frac", 0.0))
        + w["body_rate_saturation_frac"] * float(summary.get("body_rate_saturation_frac", 0.0))
        + w["mean_final_distance_to_gate_m"] * float(summary.get("mean_final_distance_to_gate_m", 0.0))
    )


@torch.no_grad()
def evaluate_flightmare_policy(
    model: torch.nn.Module,
    *,
    data_cfg: dict,
    robotics_cfg: dict,
    eval_cfg: dict | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    eval_cfg = eval_cfg or {}
    robotics_cfg = robotics_cfg or {}
    ppo_cfg = robotics_cfg.get("ppo", {}) or {}
    arch_cfg = robotics_cfg.get("architecture", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", ""))
    manifest = load_dataset_manifest(data_dir)
    norm_stats = data_dir / "norm_stats.npz"
    if not norm_stats.exists():
        raise FileNotFoundError(f"norm_stats.npz not found: {norm_stats}")

    device = torch.device(device)
    action_type = infer_action_type(data_cfg, robotics_cfg)
    obs_schema = str(
        data_cfg.get("obs_schema")
        or arch_cfg.get("obs_schema")
        or ("v3" if str(arch_cfg.get("fusion", "")).lower() == "swift" else "v2")
    )
    include_mission = bool(data_cfg.get("include_mission", True))
    if obs_schema == "v3":
        concat_mission_to_state = False
        include_mission = False
    else:
        concat_mission_to_state = include_mission and int(arch_cfg.get("state_dim", 13)) == 13 + MISSION_DIM

    control_hz = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "control_hz", 100.0))
    backend = str(_cfg_value(eval_cfg, ppo_cfg, manifest, "backend", "flightgym"))
    scene = str(_cfg_value(eval_cfg, ppo_cfg, manifest, "scene", "industrial"))
    max_steps = int(_cfg_value(eval_cfg, ppo_cfg, manifest, "max_steps", 1500, ppo_key="horizon"))
    episodes = int(eval_cfg.get("episodes", 20))
    seed = int(eval_cfg.get("seed", 0))
    render = bool(eval_cfg.get("render", False))
    fail_on_fallback = bool(eval_cfg.get("fail_on_fallback", False))

    course_mode = str(_cfg_value(eval_cfg, ppo_cfg, manifest, "course_mode", "gates"))
    num_gates = int(_cfg_value(eval_cfg, ppo_cfg, manifest, "num_gates", manifest_num_gates(manifest, 8)))
    gate_spacing_range = list(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_spacing_range", [4.0, 9.0]))
    gate_lateral_jitter = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_lateral_jitter", 2.0))
    gate_z_range = list(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_z_range", [1.5, 4.0]))
    gate_yaw_step = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_yaw_step", 0.7))
    gate_yaw_noise = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_yaw_noise", 0.25))
    gate_size = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_size", manifest_gate_size(manifest, 1.0)))
    gate_approach_m = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_approach_m", 1.2))
    fixed_gate_pos_noise = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "fixed_gate_pos_noise", 0.0))
    fixed_gate_yaw_noise = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "fixed_gate_yaw_noise", 0.0))
    random_start_gate = bool(_cfg_value(eval_cfg, ppo_cfg, manifest, "random_start_gate", False))
    z_min = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "z_min", 1.0))
    max_world_radius = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "max_world_radius", 200.0))
    min_z = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "min_z", -0.25))
    max_body_rate = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "max_body_rate", 8.0))
    max_waypoint_speed = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "max_waypoint_speed", 15.0))
    gate_vehicle_radius = float(_cfg_value(eval_cfg, ppo_cfg, manifest, "gate_vehicle_radius", 0.15))
    max_collective_thrust_g = float(
        _cfg_value(eval_cfg, ppo_cfg, manifest, "max_collective_thrust_g", manifest_thrust_g(manifest, 4.0))
    )
    terminate_on_gate_miss = bool(_cfg_value(eval_cfg, ppo_cfg, manifest, "terminate_on_gate_miss", True))

    params = apply_plant_overrides(QuadParams(), resolved_plant_config(ppo_cfg, manifest))
    params.max_collective_thrust = max_collective_thrust_g * params.mass * params.g

    model_was_training = model.training
    model.eval()
    mission_wrapper = MissionWrapper(
        policy=model,
        gates=[],
        norm_stats=norm_stats,
        action_type=action_type,
        device=device,
        include_mission=include_mission,
        concat_mission_to_state=concat_mission_to_state,
        obs_schema=obs_schema,
    )
    mission_wrapper.set_dt(1.0 / control_hz)

    state_node = FlightmareStateNode(
        control_hz=control_hz,
        course_config=CourseConfig(
            course_mode=course_mode,
            num_gates=num_gates,
            z_min=z_min,
            gate_spacing_range=tuple(gate_spacing_range),
            gate_lateral_jitter=gate_lateral_jitter,
            gate_z_range=tuple(gate_z_range),
            gate_yaw_step=gate_yaw_step,
            gate_yaw_noise=gate_yaw_noise,
            gate_size=gate_size,
            gate_layout=eval_cfg.get("gate_layout") or ppo_cfg.get("gate_layout") or manifest.get("gate_layout"),
            random_start_gate=random_start_gate,
            fixed_gate_pos_noise=fixed_gate_pos_noise,
            fixed_gate_pos_noise_xyz=_cfg_value(eval_cfg, ppo_cfg, manifest, "fixed_gate_pos_noise_xyz", None),
            fixed_gate_yaw_noise=fixed_gate_yaw_noise,
            gate_approach_m=gate_approach_m,
            inverted_roll_jitter_rad=float(
                _cfg_value(eval_cfg, ppo_cfg, manifest, "inverted_roll_jitter_rad", 0.2618)
            ),
            # Use the same procedural scenario distribution as collection/PPO when present.
            course_distribution=_cfg_value(eval_cfg, ppo_cfg, manifest, "course_distribution", None),
        ),
        params=params,
        scene=scene,
        render=render,
        seed=seed,
        backend=backend,
    )
    graph = FlightmareAutonomyGraph(
        state_node=state_node,
        mission_node=MissionWorldModelNode(mission_wrapper),
        planner_node=PolicyPlannerNode(mission_wrapper, action_type=action_type),
        controller_node=BaseControllerNode(
            BaseAutonomyController(
                params=params,
                waypoint_controller=WaypointLQRController(params=params, max_speed=max_waypoint_speed),
                ctbr_controller=BaseCTBRController(params=params, max_body_rate=max_body_rate),
                motor_controller=BaseMotorController(params=params),
            )
        ),
        max_steps=max_steps,
        max_world_radius=max_world_radius,
        min_z=min_z,
        gate_vehicle_radius=gate_vehicle_radius,
        terminate_on_gate_miss=terminate_on_gate_miss,
    )
    try:
        if state_node.using_fallback and backend != "numpy" and fail_on_fallback:
            raise RuntimeError("Flightmare bindings unavailable; numpy fallback active.")
        results = graph.run(episodes=episodes, seed=seed)
    finally:
        graph.close()
        if model_was_training:
            model.train()

    summary = summarize_results(results)
    summary.update(_saturation_metrics(results, max_body_rate=max_body_rate))
    summary["mean_final_distance_to_gate_m"] = float(
        np.mean([r.final_distance_to_gate_m for r in results]) if results else 0.0
    )
    summary["score"] = score_flightmare_rollout(summary, eval_cfg.get("score_weights"))
    summary["plant_mass"] = float(params.mass)
    summary["plant_omega_max_body_x"] = float(params.omega_max_body[0])
    return summary

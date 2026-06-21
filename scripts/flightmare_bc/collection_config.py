from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


RACING_PLANT = {
    "mass": 0.73,
    "arm_length": 0.17,
    "inertia": [0.0025, 0.0021, 0.0043],
    "k_thrust": 1.91e-6,
    "k_torque": 2.6e-7,
    "motor_omega_min": 150.0,
    "motor_omega_max": 3000.0,
    "motor_tau": 0.0001,
    "thrust_map": [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325],
    "kappa": 0.016,
    "omega_max_body": [18.0, 18.0, 18.0],
    "max_collective_thrust_g": 5.5,
}


DEFAULT_COLLECTION_CONFIG = {
    "version": 2,
    "output_dir": "data/flightmare/bc_v6",
    "episodes": 4000,
    "seed": 0,
    "val_frac": 0.1,
    "image_size": 224,
    "cameras": [],
    "render": False,
    "backend": "flightgym",
    "scene": "industrial",
    "control_hz": 100.0,
    "max_steps": 5000,
    "max_world_radius": 350.0,
    "action_normalization": "bounds",
    "skip_initial_frames": 0,
    "bounds_margin": 0.10,
    "min_gate_completion": 0.99,
    "gate_vehicle_radius": 0.20,
    "z_min": 1.0,
    "plant": RACING_PLANT,
    "course": {
        "course_mode": "swift_v4",
        "num_gates": 7,
        "gate_size": 1.6,
        "gate_approach_m": 3.0,
        "fixed_gate_pos_noise": 0.20,
        "fixed_gate_yaw_noise": 0.05,
        "random_start_gate": True,
        "gate_spacing_range": [4.0, 9.0],
        "gate_lateral_jitter": 2.0,
        "gate_z_range": [1.5, 4.0],
        "gate_yaw_step": 0.7,
        "gate_yaw_noise": 0.25,
    },
    # Optional shared procedural scenario distribution. When present (non-null),
    # every collection mode samples courses from this weighted mixture via
    # ``src.robotics.flightmare_courses`` and the simple ``course`` block above is
    # only used as a fallback. Schema mirrors CourseDistributionConfig:
    #   course_distribution:
    #     max_resample_tries: 24
    #     bounds: {arena_radius: 60.0, z_min: 1.0, ...}
    #     scenarios:
    #       - {name: flow, family: procedural_flow, weight: 0.3, ...}
    "course_distribution": None,
    "expert": {
        # Default is a robust bootstrap expert. The config surface is explicit
        # because high-speed teacher data should eventually come from a real
        # MPCC/MPC backend, not from a hidden hard-coded controller.
        "controller": "geometric_minjerk",
        "speed_range": [1.8, 2.6],
        "lookahead_s": 0.30,
        "trajectory_min_segment_s": 0.80,
        "gains": {
            "kp_pos": [6.0, 6.0, 18.0],
            "kd_pos": [4.0, 4.0, 11.0],
            "kp_att": [10.0, 10.0, 4.0],
            "kp_rate": [28.0, 28.0, 12.0],
        },
    },
    "controller_eval": {
        "enabled": True,
        "episodes": 8,
        "speed_bins": 2,
        "min_success_rate": 1.0,
        "min_gate_completion": 1.0,
    },
    "mpcc": {
        "enabled": False,
        "backend": None,
        "binary": None,
        "config": None,
    },
    "mix": {
        "expert": 0.55,
        "synthetic_recovery": 0.30,
        "dagger": 0.15,
    },
    "recovery": {
        "samples_per_episode": 512,
        "gate_transition_probability": 0.60,
        "post_gate_probability": 0.45,
        "lateral_offset_m": [-1.2, 1.2],
        "vertical_offset_m": [-0.7, 0.7],
        "forward_offset_m": [-1.5, 2.0],
        "yaw_error_rad": [-0.9, 0.9],
        "velocity_error_mps": [-2.5, 2.5],
        "omega_error_radps": [-4.0, 4.0],
        "prev_action_noise": [0.04, 1.0, 1.0, 0.5],
    },
    "dagger": {
        "enabled": True,
        "policy_exp": "flightmare/paper/teacher/ctbr_bc",
        "checkpoint": "outputs/paper_teacher_ctbr_bc_v5/best/model.pt",
        "episodes": None,
        "record_deviation_threshold_m": 0.50,
        "max_record_deviation_m": 8.0,
        "terminate_deviation_m": 15.0,
        "record_clean_probability": 0.15,
        "terminate_on_gate_miss": True,
    },
    "postprocess": {
        "transform_to_v3": True,
        "recompute_v3_norm_stats": True,
        "diag_v3_geometry": True,
        "diag_episodes": 20,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_collection_config(path: str | Path) -> dict:
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_COLLECTION_CONFIG, user_cfg)
    if cfg.get("episodes", 0) <= 0:
        raise ValueError("episodes must be positive.")
    speed = cfg["expert"]["speed_range"]
    if len(speed) != 2 or float(speed[0]) <= 0 or float(speed[1]) < float(speed[0]):
        raise ValueError("expert.speed_range must be [min, max] with 0 < min <= max.")
    if not cfg.get("mix"):
        raise ValueError("mix must contain at least one collection mode.")
    controller = str(cfg.get("expert", {}).get("controller", "geometric_minjerk"))
    if controller not in {"geometric_minjerk", "mpcc"}:
        raise ValueError("expert.controller must be 'geometric_minjerk' or 'mpcc'.")
    if controller == "mpcc" and not cfg.get("mpcc", {}).get("enabled", False):
        raise ValueError("expert.controller='mpcc' requires mpcc.enabled=true and a real backend adapter.")
    _validate_course_distribution(cfg.get("course_distribution"))
    return cfg


def _validate_course_distribution(dist: Any | None) -> None:
    """Reject impossible/ambiguous course_distribution configs early.

    Building the dataclasses runs their ``__post_init__`` validators (unknown
    family, bad weight/ranges, missing layouts), so this surfaces schema errors
    at config-load time rather than mid-collection.
    """
    if dist is None:
        return
    from src.robotics.flightmare_courses import course_distribution_from_dict  # noqa: PLC0415

    try:
        built = course_distribution_from_dict(dist)
    except Exception as exc:  # noqa: BLE001 - re-raise as config error
        raise ValueError(f"Invalid course_distribution: {exc}") from exc
    if built is not None and float(built.weights.sum()) <= 0.0:
        raise ValueError("course_distribution scenario weights must sum to > 0.")


def normalized_mix(cfg: dict) -> list[tuple[str, float]]:
    items = [(str(k), float(v)) for k, v in cfg.get("mix", {}).items() if float(v) > 0.0]
    total = sum(v for _, v in items)
    if total <= 0.0:
        raise ValueError("At least one mix weight must be positive.")
    return [(name, weight / total) for name, weight in items]


def mode_counts(total_episodes: int, cfg: dict) -> dict[str, int]:
    mix = normalized_mix(cfg)
    raw = [(name, weight * int(total_episodes)) for name, weight in mix]
    counts = {name: int(value) for name, value in raw}
    remaining = int(total_episodes) - sum(counts.values())
    remainders = sorted(((value - int(value), name) for name, value in raw), reverse=True)
    for _, name in remainders[:remaining]:
        counts[name] += 1
    return counts


def plant_cli_args(cfg: dict) -> list[str]:
    plant = cfg.get("plant", {})
    args: list[str] = []
    scalar_flags = {
        "mass": "--mass",
        "arm_length": "--arm-length",
        "k_thrust": "--k-thrust",
        "k_torque": "--k-torque",
        "motor_omega_min": "--motor-omega-min",
        "motor_omega_max": "--motor-omega-max",
        "motor_tau": "--motor-tau",
        "kappa": "--kappa",
    }
    for key, flag in scalar_flags.items():
        if key in plant and plant[key] is not None:
            args.extend([flag, str(plant[key])])
    list_flags = {
        "inertia": "--inertia",
        "thrust_map": "--thrust-map",
        "omega_max_body": "--omega-max-body",
    }
    for key, flag in list_flags.items():
        if key in plant and plant[key] is not None:
            args.append(flag)
            args.extend(str(v) for v in plant[key])
    return args


def collect_cli_args(cfg: dict, *, out: str | Path | None = None, episodes: int | None = None) -> list[str]:
    course = cfg.get("course", {})
    expert = cfg.get("expert", {})
    plant = cfg.get("plant", {})
    args = [
        "--out", str(out or cfg["output_dir"]),
        "--episodes", str(int(episodes if episodes is not None else cfg["episodes"])),
        "--backend", str(cfg["backend"]),
        "--image-size", str(int(cfg["image_size"])),
        "--control-hz", str(float(cfg["control_hz"])),
        "--max-steps", str(int(cfg["max_steps"])),
        "--max-world-radius", str(float(cfg["max_world_radius"])),
        "--action-normalization", str(cfg["action_normalization"]),
        "--skip-initial-frames", str(int(cfg["skip_initial_frames"])),
        "--bounds-margin", str(float(cfg["bounds_margin"])),
        "--scene", str(cfg["scene"]),
        "--seed", str(int(cfg["seed"])),
        "--val-frac", str(float(cfg["val_frac"])),
        "--min-gate-completion", str(float(cfg["min_gate_completion"])),
        "--gate-vehicle-radius", str(float(cfg["gate_vehicle_radius"])),
        "--z-min", str(float(cfg["z_min"])),
        "--course-mode", str(course["course_mode"]),
        "--num-gates", str(int(course["num_gates"])),
        "--gate-size", str(float(course["gate_size"])),
        "--gate-approach-m", str(float(course["gate_approach_m"])),
        "--fixed-gate-pos-noise", str(float(course["fixed_gate_pos_noise"])),
        "--fixed-gate-yaw-noise", str(float(course["fixed_gate_yaw_noise"])),
        "--gate-yaw-step", str(float(course["gate_yaw_step"])),
        "--gate-yaw-noise", str(float(course["gate_yaw_noise"])),
        "--gate-lateral-jitter", str(float(course["gate_lateral_jitter"])),
        "--trajectory-min-segment-s", str(float(expert["trajectory_min_segment_s"])),
        "--lookahead-s", str(float(expert["lookahead_s"])),
        "--max-collective-thrust-g", str(float(plant["max_collective_thrust_g"])),
        "--no-render",
        "--speed-range", str(float(expert["speed_range"][0])), str(float(expert["speed_range"][1])),
        "--gate-spacing-range", str(float(course["gate_spacing_range"][0])), str(float(course["gate_spacing_range"][1])),
        "--gate-z-range", str(float(course["gate_z_range"][0])), str(float(course["gate_z_range"][1])),
    ]
    if course.get("random_start_gate", False):
        args.append("--random-start-gate")
    if course.get("gate_layout"):
        args.extend(["--gate-layout", str(course["gate_layout"])])
    args.extend(plant_cli_args(cfg))
    return args

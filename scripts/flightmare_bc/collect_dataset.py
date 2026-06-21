"""Config-driven Flightmare BC collection for v6 teacher seeds.

This is the focused replacement path for shell-script-as-config collection.
Legacy scripts still work, but new datasets should come through a YAML config
that records plant, course, expert, recovery, DAgger, and postprocess settings
in the manifest.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.flightmare_bc.collection_config import load_collection_config, mode_counts
from scripts.flightmare_bc.collect import (
    STATE_DIM,
    _gate_record,
    _quat_from_yaw_pitch_roll,
    build_state,
    collect_one_episode,
    ctbr_label,
    episode_gate_quality,
    motor_label,
    sample_course_with_meta,
    waypoint_speed_label,
    waypoints_from_gates,
    write_norm_stats,
)
from scripts.flightmare_bc.controllers import GeometricGains, GeometricSE3Controller, QuadParams
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from scripts.flightmare_bc.hdf5_writer import EpisodeWriter
from scripts.flightmare_bc.mission import LOOKAHEAD_GATES, MISSION_DIM, encode_mission, gates_to_views
from scripts.flightmare_bc.trajectories import MinJerkTrajectory


def _quad_params(cfg: dict) -> QuadParams:
    plant = cfg.get("plant", {})
    params = QuadParams()
    for key, value in plant.items():
        if key == "max_collective_thrust_g" or value is None:
            continue
        if key in {"inertia", "thrust_map", "omega_max_body"}:
            setattr(params, key, tuple(float(x) for x in value))
        elif hasattr(params, key):
            setattr(params, key, float(value))
    params.max_collective_thrust = float(plant.get("max_collective_thrust_g", 4.0)) * params.mass * params.g
    return params


def _collector_namespace(cfg: dict) -> SimpleNamespace:
    course = cfg.get("course", {})
    expert = cfg.get("expert", {})
    plant = cfg.get("plant", {})
    return SimpleNamespace(
        image_size=int(cfg["image_size"]),
        cameras=list(cfg.get("cameras", [])),
        control_hz=float(cfg["control_hz"]),
        max_steps=int(cfg["max_steps"]),
        n_waypoints=6,
        bbox=[8.0, 8.0, 3.5],
        z_min=float(cfg["z_min"]),
        min_step=1.5,
        speed_range=list(expert["speed_range"]),
        lookahead_s=float(expert["lookahead_s"]),
        scene=str(cfg["scene"]),
        backend=str(cfg["backend"]),
        action_normalization=str(cfg["action_normalization"]),
        skip_initial_frames=int(cfg["skip_initial_frames"]),
        bounds_margin=float(cfg["bounds_margin"]),
        course_mode=str(course.get("course_mode", "gates")),
        gate_layout=course.get("gate_layout"),
        random_start_gate=bool(course.get("random_start_gate", False)),
        fixed_gate_pos_noise=float(course.get("fixed_gate_pos_noise", 0.0)),
        fixed_gate_yaw_noise=float(course.get("fixed_gate_yaw_noise", 0.0)),
        fixed_gate_pos_noise_xyz=course.get("fixed_gate_pos_noise_xyz"),
        num_gates=int(course.get("num_gates", 7)),
        gate_spacing_range=list(course.get("gate_spacing_range", [4.0, 9.0])),
        gate_lateral_jitter=float(course.get("gate_lateral_jitter", 2.0)),
        gate_z_range=list(course.get("gate_z_range", [1.5, 4.0])),
        gate_yaw_step=float(course.get("gate_yaw_step", 0.7)),
        gate_yaw_noise=float(course.get("gate_yaw_noise", 0.25)),
        gate_size=float(course.get("gate_size", 1.6)),
        gate_approach_m=float(course.get("gate_approach_m", 3.0)),
        # Shared procedural scenario distribution (None => legacy single-mode).
        course_distribution=cfg.get("course_distribution"),
        trajectory_min_segment_s=float(expert["trajectory_min_segment_s"]),
        gate_vehicle_radius=float(cfg["gate_vehicle_radius"]),
        max_world_radius=float(cfg["max_world_radius"]),
        max_waypoint_speed=float(cfg.get("max_waypoint_speed", 15.0)),
        max_collective_thrust_g=float(plant["max_collective_thrust_g"]),
        render=bool(cfg.get("render", False)),
        min_gate_completion=float(cfg["min_gate_completion"]),
        shard_id=0,
    )


def _expert_controller(cfg: dict, params: QuadParams):
    """Select the expert backend (geometric SE(3) or MPPI) from the config."""
    from src.robotics.flightmare_experts import build_expert_controller

    return build_expert_controller(cfg, params)


def _dagger_labeler(cfg: dict, params: QuadParams, controller):
    """Build the distillation label adapter used for closed-loop DAgger data."""
    from src.robotics.flightmare_experts import HybridExpertLabeler, SafetySupervisor

    expert = cfg.get("expert", {}) or {}
    fallback = None
    if str(expert.get("controller", "geometric_minjerk")) == "mpcc":
        gains_cfg = expert.get("gains", {}) or {}
        gains = GeometricGains(
            kp_pos=tuple(float(x) for x in gains_cfg.get("kp_pos", GeometricGains.kp_pos)),
            kd_pos=tuple(float(x) for x in gains_cfg.get("kd_pos", GeometricGains.kd_pos)),
            kp_att=tuple(float(x) for x in gains_cfg.get("kp_att", GeometricGains.kp_att)),
            kp_rate=tuple(float(x) for x in gains_cfg.get("kp_rate", GeometricGains.kp_rate)),
        )
        fallback = GeometricSE3Controller(params=params, gains=gains)
        fallback.source = "geometric_minjerk"
    max_body_rate = float(
        (cfg.get("mpcc", {}) or {}).get("mppi", {}).get(
            "max_body_rate",
            cfg.get("dagger", {}).get("max_body_rate", 18.0),
        )
    )
    return HybridExpertLabeler(
        primary=controller,
        fallback=fallback,
        supervisor=SafetySupervisor(max_body_rate=max_body_rate),
        max_body_rate=max_body_rate,
    )


def _label_command(label) -> dict:
    if label.command is not None:
        return label.command
    return {
        "thrust_normalized": float(label.action_ctbr_raw[0]),
        "body_rates": np.asarray(label.action_ctbr_raw[1:4], dtype=np.float32),
        "motor_normalized": np.zeros(4, dtype=np.float32),
        "source": label.source,
    }


def _manifest_header(cfg: dict, params: QuadParams) -> dict:
    course = cfg.get("course", {})
    expert = cfg.get("expert", {})
    return {
        "version": int(cfg.get("version", 2)),
        "collector": "scripts.flightmare_bc.collect_dataset",
        "controller": str(expert.get("controller", "geometric_minjerk")),
        "image_size": int(cfg["image_size"]),
        "cameras": list(cfg.get("cameras", [])),
        "control_hz": float(cfg["control_hz"]),
        "state_dim": STATE_DIM,
        "state_layout": ["pos(3)", "vel(3)", "quat_wxyz(4)", "omega_body(3)"],
        "action_normalization": str(cfg["action_normalization"]),
        "skip_initial_frames": int(cfg["skip_initial_frames"]),
        "bounds_margin": float(cfg["bounds_margin"]),
        "backend": str(cfg["backend"]),
        "collection_config": cfg,
        "action_types": {
            "waypoint": {"dim": 4, "layout": ["dx_body", "dy_body", "dz_body", "speed"]},
            "ctbr": {"dim": 4, "layout": ["thrust_norm", "wx", "wy", "wz"]},
            "motor": {"dim": 4, "layout": ["m0_norm", "m1_norm", "m2_norm", "m3_norm"]},
        },
        "mission": {
            "dim": MISSION_DIM,
            "lookahead_gates": LOOKAHEAD_GATES,
        },
        "params": asdict(params),
        "lookahead_s": float(expert["lookahead_s"]),
        "speed_range": list(expert["speed_range"]),
        "course_mode": str(course["course_mode"]),
        # Surface the procedural distribution at the manifest top level so
        # closed-loop eval (flightmare_policy_eval._cfg_value) auto-matches the
        # collection course family instead of falling back to the default
        # `gates` course. None when collecting on a single course_mode.
        "course_distribution": cfg.get("course_distribution"),
        "gate_layout": str(course["gate_layout"]) if course.get("gate_layout") is not None else None,
        "random_start_gate": bool(course.get("random_start_gate", False)),
        "fixed_gate_pos_noise": float(course.get("fixed_gate_pos_noise", 0.0)),
        "fixed_gate_yaw_noise": float(course.get("fixed_gate_yaw_noise", 0.0)),
        "num_gates": int(course["num_gates"]),
        "gate_spacing_range": list(course.get("gate_spacing_range", [4.0, 9.0])),
        "gate_lateral_jitter": float(course.get("gate_lateral_jitter", 2.0)),
        "gate_z_range": list(course.get("gate_z_range", [1.5, 4.0])),
        "gate_yaw_step": float(course.get("gate_yaw_step", 0.7)),
        "gate_yaw_noise": float(course.get("gate_yaw_noise", 0.25)),
        "gate_size": float(course["gate_size"]),
        "z_min": float(cfg["z_min"]),
        "gate_approach_m": float(course["gate_approach_m"]),
        "trajectory_min_segment_s": float(expert["trajectory_min_segment_s"]),
        "gate_vehicle_radius": float(cfg["gate_vehicle_radius"]),
        "min_gate_completion": float(cfg["min_gate_completion"]),
        "seed": int(cfg["seed"]),
        "episodes": [],
    }


def _gate_frame(gate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quaternion-aware gate frame (forward, lateral, up).

    Delegates to the shared ``flightmare_autonomy_fsw.gates.gate_frame`` so
    recovery perturbations respect the real gate lateral/up axes for
    inverted/tilted gates (the previous world-up reconstruction was only
    correct for upright gates).
    """
    from src.robotics.flightmare_autonomy_fsw.gates import gate_frame as _gf  # noqa: PLC0415

    _, forward, lateral, up = _gf(gate)
    return (
        np.asarray(forward, dtype=np.float64),
        np.asarray(lateral, dtype=np.float64),
        np.asarray(up, dtype=np.float64),
    )


def _uniform_vec(rng: np.random.Generator, bounds: list[float], dim: int) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    return rng.uniform(lo, hi, size=dim).astype(np.float64)


def _nearest_ref(traj: MinJerkTrajectory, pos: np.ndarray, grid_t: np.ndarray, grid_pos: np.ndarray):
    idx = int(np.argmin(np.linalg.norm(grid_pos - pos[None, :], axis=1)))
    t = float(grid_t[idx])
    return t, traj.sample(t)


def collect_synthetic_recovery_episode(
    cfg: dict,
    params: QuadParams,
    controller: GeometricSE3Controller,
    rng: np.random.Generator,
    episode_id: int,
    writer: EpisodeWriter,
) -> dict:
    ns = _collector_namespace(cfg)
    rec_cfg = cfg.get("recovery", {})
    gates, scenario_meta = sample_course_with_meta(rng, ns)
    for gate in gates:
        gate.gate_id = f"ep{episode_id:06d}_{gate.gate_id}"
    waypoints = waypoints_from_gates(gates, ns.z_min, d_approach=float(ns.gate_approach_m))
    avg_speed = float(rng.uniform(ns.speed_range[0], ns.speed_range[1]))
    traj = MinJerkTrajectory(
        waypoints,
        avg_speed=avg_speed,
        yaw_mode="tangent",
        min_segment_s=float(ns.trajectory_min_segment_s),
    )
    grid_t = np.linspace(0.0, float(traj.total_time), 1024)
    grid_pos = np.stack([traj.sample(float(t)).pos for t in grid_t], axis=0)
    gate_views = gates_to_views(gates)

    samples = int(rec_cfg.get("samples_per_episode", 512))
    prev_noise = np.asarray(rec_cfg.get("prev_action_noise", [0.04, 1.0, 1.0, 0.5]), dtype=np.float64)
    post_prob = float(rec_cfg.get("post_gate_probability", 0.45))
    transition_prob = float(rec_cfg.get("gate_transition_probability", 0.60))
    pos_errors: list[float] = []
    for k in range(samples):
        gate_i = int(rng.integers(0, max(1, len(gates))))
        gate = gates[gate_i]
        forward, lateral, up = _gate_frame(gate)
        post_gate = bool(gate_i + 1 < len(gates) and rng.random() < post_prob)
        gate_transition = bool(rng.random() < transition_prob)
        forward_bounds = rec_cfg.get("forward_offset_m", [-1.5, 2.0])
        signed = float(rng.uniform(forward_bounds[0], forward_bounds[1]))
        if post_gate:
            signed = abs(signed) + float(rng.uniform(0.1, 1.2))
        elif gate_transition:
            signed = float(rng.uniform(-0.8, 0.8))

        pos = (
            np.asarray(gate.pos, dtype=np.float64)
            + signed * forward
            + float(rng.uniform(*rec_cfg.get("lateral_offset_m", [-1.2, 1.2]))) * lateral
            + float(rng.uniform(*rec_cfg.get("vertical_offset_m", [-0.7, 0.7]))) * up
        )
        pos[2] = max(pos[2], float(cfg["z_min"]))
        current_gate = min(gate_i + 1, len(gates) - 1) if post_gate else gate_i
        # Recovery samples are independent perturbed states; reset any stateful
        # expert (MPPI warm-start) so each label is planned from scratch.
        if hasattr(controller, "reset"):
            controller.reset()
        ref_t, ref = _nearest_ref(traj, pos, grid_t, grid_pos)
        yaw_err = float(rng.uniform(*rec_cfg.get("yaw_error_rad", [-0.9, 0.9])))
        quat = _quat_from_yaw_pitch_roll(float(ref.yaw) + yaw_err, 0.0, 0.0)
        vel = ref.vel + _uniform_vec(rng, rec_cfg.get("velocity_error_mps", [-2.5, 2.5]), 3)
        omega = _uniform_vec(rng, rec_cfg.get("omega_error_radps", [-4.0, 4.0]), 3)
        cmd = controller.compute(
            pos=pos,
            vel=vel,
            quat=quat,
            pos_des=ref.pos,
            vel_des=ref.vel,
            acc_des=ref.acc,
            yaw_des=ref.yaw,
            omega=omega,
        )
        lookahead_pos, speed_now = traj.lookahead(ref_t, float(ns.lookahead_s))
        waypoint_act = waypoint_speed_label(pos, quat, lookahead_pos, speed_now)
        ctbr_act = ctbr_label(cmd["thrust_normalized"], cmd["body_rates"])
        motor_act = motor_label(cmd["motor_normalized"])
        prev_ctbr = np.clip(ctbr_act + rng.normal(0.0, prev_noise, size=4), [0.0, -18.0, -18.0, -18.0], [1.0, 18.0, 18.0, 18.0])
        mission_vec = encode_mission(pos, quat, gate_views, current_gate, LOOKAHEAD_GATES)
        state_vec = build_state(pos, vel, quat, omega)
        writer.append(
            state=state_vec,
            images={},
            actions={"waypoint": waypoint_act, "ctbr": ctbr_act, "motor": motor_act},
            prev_actions={"waypoint": waypoint_act, "ctbr": prev_ctbr.astype(np.float32), "motor": motor_act},
            ref_pos=ref.pos.astype(np.float32),
            ref_vel=ref.vel.astype(np.float32),
            ref_yaw=float(ref.yaw),
            done=(k == samples - 1),
            mission=mission_vec,
            gate_index=current_gate,
        )
        pos_errors.append(float(np.linalg.norm(pos - ref.pos)))

    return {
        "episode_id": episode_id,
        "length": samples,
        "sample_kind": "synthetic_recovery",
        "avg_speed": avg_speed,
        "mean_track_err": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_track_err": float(np.max(pos_errors)) if pos_errors else 0.0,
        "n_waypoints": int(len(waypoints)),
        "gates": [_gate_record(g) for g in gates],
        "scenario": scenario_meta,
    }


def _load_dagger_policy(cfg: dict, device: str):
    import torch
    dagger = cfg.get("dagger", {})
    checkpoint = dagger.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        return None, None, None
    from src.core.config import Config
    from src.robotics.flightmare_policy_eval import build_flightmare_model, infer_action_type

    exp = dagger.get("policy_exp")
    config_path = dagger.get("policy_config")
    if exp:
        policy_cfg = Config.from_experiment(exp)
    elif config_path:
        policy_cfg = Config.load(config_path, base_configs=["configs/base/common.yaml", "configs/base/robotics.yaml"])
    else:
        raise ValueError("dagger requires policy_exp or policy_config when checkpoint is set.")
    action_type = infer_action_type(policy_cfg.data, policy_cfg.robotics)
    if action_type not in ("ctbr", "waypoint"):
        raise ValueError(
            f"DAgger collection supports CTBR or waypoint policies, got {action_type!r}."
        )
    model = build_flightmare_model(policy_cfg.robotics["architecture"], checkpoint, device=torch.device(device))
    return policy_cfg, model, Path(checkpoint)


def collect_dagger_episode(
    cfg: dict,
    params: QuadParams,
    controller: GeometricSE3Controller,
    env: FlightmareExpertEnv,
    policy_cfg,
    policy,
    rng: np.random.Generator,
    episode_id: int,
    writer: EpisodeWriter,
) -> dict:
    import torch
    from src.robotics.flightmare_autonomy_fsw.controllers import (
        BaseAutonomyController,
        BaseCTBRController,
        WaypointLQRController,
    )
    from src.robotics.flightmare_autonomy_fsw.messages import PlannerOutput, VehicleState
    from src.robotics.flightmare_policy_eval import infer_action_type
    from src.robotics.models.flightmare.MissionWrapper import MissionWrapper

    ns = _collector_namespace(cfg)
    dagger = cfg.get("dagger", {})
    # Student action interface: CTBR (raw body rates) or waypoint+speed (closed
    # via the LQR inner loop). The collector relabels with the privileged expert
    # the same way regardless; only the student's own rollout actuation differs.
    student_action_type = infer_action_type(policy_cfg.data, policy_cfg.robotics)
    gates, scenario_meta = sample_course_with_meta(rng, ns)
    for gate in gates:
        gate.gate_id = f"ep{episode_id:06d}_{gate.gate_id}"
    env.add_gates(gates)
    waypoints = waypoints_from_gates(gates, ns.z_min, d_approach=float(ns.gate_approach_m))
    obs = env.reset(init_pos=waypoints[0], yaw=float(gates[0].yaw))
    avg_speed = float(rng.uniform(ns.speed_range[0], ns.speed_range[1]))
    traj = MinJerkTrajectory(
        waypoints,
        avg_speed=avg_speed,
        yaw_mode="tangent",
        min_segment_s=float(ns.trajectory_min_segment_s),
    )
    device = next(policy.parameters()).device
    policy_data_dir = Path(policy_cfg.data["data_dir"])
    arch_cfg = policy_cfg.robotics.get("architecture", {})
    obs_schema = str(policy_cfg.data.get("obs_schema") or arch_cfg.get("obs_schema") or "v3")
    wrapper = MissionWrapper(
        policy=policy,
        gates=gates,
        norm_stats=policy_data_dir / "norm_stats.npz",
        action_type=student_action_type,
        device=device,
        include_mission=False if obs_schema == "v3" else bool(policy_cfg.data.get("include_mission", True)),
        obs_schema=obs_schema,
    )
    wrapper.set_dt(1.0 / float(ns.control_hz))
    max_body_rate = float(dagger.get("max_body_rate", 8.0))
    # CTBR students command body rates directly; waypoint students emit a
    # setpoint that the LQR inner loop stabilizes into CTBR. BaseAutonomyController
    # dispatches on planner.action_type, so a single actuator covers both.
    actuator = BaseAutonomyController(
        params=params,
        waypoint_controller=WaypointLQRController(
            params=params, max_speed=float(ns.max_waypoint_speed)
        ),
        ctbr_controller=BaseCTBRController(params=params, max_body_rate=max_body_rate),
    )
    gate_views = gates_to_views(gates)
    labeler = _dagger_labeler(cfg, params, controller)

    # DAgger relabels against the NEAREST point on the path (+ lookahead), not the
    # wall-clock-time-indexed point. With a time-indexed reference the target races
    # ahead as the student falls behind, so the expert relabels toward an
    # unreachable point and the recovery labels become physically inconsistent —
    # which makes the aggregate worse each round. Projecting to the closest path
    # point gives consistent "track the path from where you actually are" labels,
    # matching how synthetic_recovery already labels (see _nearest_ref above).
    grid_t = np.linspace(0.0, float(traj.total_time), 1024)
    grid_pos = np.stack([traj.sample(float(tt)).pos for tt in grid_t], axis=0)

    n_steps = max(1, min(int(traj.total_time / env.dt), int(ns.max_steps)))
    threshold = float(dagger.get("record_deviation_threshold_m", 0.50))
    max_record_deviation = float(dagger.get("max_record_deviation_m", 8.0))
    terminate_deviation = float(dagger.get("terminate_deviation_m", 15.0))
    clean_prob = float(dagger.get("record_clean_probability", 0.15))
    prev_policy_ctbr = np.zeros(4, dtype=np.float32)
    pos_errors: list[float] = []
    written = 0
    for k in range(n_steps):
        t = k * env.dt
        # Reference = nearest point on the path to the student, advanced by a short
        # lookahead so the target keeps progressing forward along the course.
        nearest_t, _ = _nearest_ref(traj, obs.pos, grid_t, grid_pos)
        ref_t = min(nearest_t + float(ns.lookahead_s), float(traj.total_time))
        ref = traj.sample(ref_t)
        label = labeler.compute(
            pos=obs.pos,
            vel=obs.vel,
            quat=obs.quat,
            pos_des=ref.pos,
            vel_des=ref.vel,
            acc_des=ref.acc,
            yaw_des=ref.yaw,
            omega=obs.omega,
        )
        cmd = _label_command(label)
        err = float(np.linalg.norm(obs.pos - ref.pos))
        if err > terminate_deviation:
            break
        wrapper.update_world_model(obs.pos, obs.vel, obs.quat, obs.omega)
        mission_obs = wrapper.last_observation
        current_gate = int(mission_obs.current_gate_index if mission_obs is not None else 0)
        should_record = err <= max_record_deviation and (err >= threshold or rng.random() < clean_prob)
        if should_record:
            # Waypoint label is also taken from the projected progress (ref_t),
            # so the recorded setpoint is consistent with the relabel reference.
            lookahead_pos, speed_now = traj.lookahead(nearest_t, float(ns.lookahead_s))
            waypoint_act = waypoint_speed_label(obs.pos, obs.quat, lookahead_pos, speed_now)
            ctbr_act = ctbr_label(cmd["thrust_normalized"], cmd["body_rates"])
            motor_act = motor_label(cmd["motor_normalized"])
            mission_vec = encode_mission(obs.pos, obs.quat, gate_views, current_gate, LOOKAHEAD_GATES)
            writer.append(
                state=build_state(obs.pos, obs.vel, obs.quat, obs.omega),
                images={},
                actions={"waypoint": waypoint_act, "ctbr": ctbr_act, "motor": motor_act},
                prev_actions={"waypoint": waypoint_act, "ctbr": prev_policy_ctbr, "motor": motor_act},
                ref_pos=ref.pos.astype(np.float32),
                ref_vel=ref.vel.astype(np.float32),
                ref_yaw=float(ref.yaw),
                done=False,
                mission=mission_vec,
                gate_index=current_gate,
                expert=label.as_hdf5_expert(),
            )
            written += 1
            pos_errors.append(err)

        with torch.no_grad():
            policy_action = wrapper.predict_from_world_model()
        planner = PlannerOutput(
            t=t,
            step=k,
            action_type=student_action_type,
            action=np.asarray(policy_action, dtype=np.float32),
            current_gate_index=current_gate,
        )
        veh_state = VehicleState(
            t=t, step=k, pos=obs.pos, vel=obs.vel, quat=obs.quat, omega=obs.omega
        )
        policy_cmd = actuator.compute(veh_state, planner)
        prev_policy_ctbr = policy_cmd.ctbr
        obs = env.step_ctbr(policy_cmd.thrust_newton, policy_cmd.body_rates)
        if obs.done or not np.all(np.isfinite(obs.pos)) or np.linalg.norm(obs.pos) > float(ns.max_world_radius):
            break
    if written == 0:
        ref = traj.sample(0.0)
        label = labeler.compute(obs.pos, obs.vel, obs.quat, ref.pos, ref.vel, ref.acc, ref.yaw, omega=obs.omega)
        cmd = _label_command(label)
        writer.append(
            state=build_state(obs.pos, obs.vel, obs.quat, obs.omega),
            images={},
            actions={
                "waypoint": waypoint_speed_label(obs.pos, obs.quat, ref.pos, avg_speed),
                "ctbr": ctbr_label(cmd["thrust_normalized"], cmd["body_rates"]),
                "motor": motor_label(cmd["motor_normalized"]),
            },
            prev_actions={"ctbr": prev_policy_ctbr},
            ref_pos=ref.pos.astype(np.float32),
            ref_vel=ref.vel.astype(np.float32),
            ref_yaw=float(ref.yaw),
            done=True,
            mission=encode_mission(obs.pos, obs.quat, gate_views, 0, LOOKAHEAD_GATES),
            gate_index=0,
            expert=label.as_hdf5_expert(),
        )
        written = 1
    return {
        "episode_id": episode_id,
        "length": written,
        "sample_kind": "dagger",
        "avg_speed": avg_speed,
        "mean_track_err": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_track_err": float(np.max(pos_errors)) if pos_errors else 0.0,
        "n_waypoints": int(len(waypoints)),
        "gates": [_gate_record(g) for g in gates],
        "scenario": scenario_meta,
    }


def _assign_splits(out_dir: Path, manifest: dict, cfg: dict) -> None:
    episodes = manifest["episodes"]
    keep_idx: list[int] = []
    for i, ep in enumerate(episodes):
        if ep.get("sample_kind", "expert") == "expert":
            ep.update(
                episode_gate_quality(
                    out_dir,
                    ep,
                    vehicle_radius=float(cfg["gate_vehicle_radius"]),
                )
            )
            if float(ep.get("gate_completion", 0.0)) < float(cfg["min_gate_completion"]):
                ep["split"] = "discard"
                continue
        keep_idx.append(i)

    rng = np.random.default_rng(int(cfg["seed"]))
    perm = rng.permutation(len(keep_idx))
    n_val = int(round(float(cfg["val_frac"]) * len(keep_idx)))
    val_local = set(int(x) for x in perm[:n_val])
    for local_i, global_i in enumerate(keep_idx):
        episodes[global_i]["split"] = "val" if local_i in val_local else "train"


def _run_postprocess(out_dir: Path, cfg: dict) -> None:
    post = cfg.get("postprocess", {})
    if post.get("transform_to_v3", True):
        subprocess.run(
            [sys.executable, "-m", "scripts.flightmare_bc.transform_to_v3", "--data-dir", str(out_dir), "--force"],
            check=True,
        )
    if post.get("recompute_v3_norm_stats", True):
        subprocess.run(
            [sys.executable, "-m", "scripts.flightmare_bc.recompute_v3_norm_stats", "--data-dir", str(out_dir), "--split", "train"],
            check=True,
        )
    if post.get("diag_v3_geometry", False):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.flightmare_bc.diag_bc_v3_geometry",
                "--data-dir",
                str(out_dir),
                "--episodes",
                str(int(post.get("diag_episodes", 20))),
                "--fail-on-missing-quat",
            ],
            check=True,
        )


def _run_controller_preflight(cfg: dict) -> dict | None:
    eval_cfg = cfg.get("controller_eval", {})
    if not bool(eval_cfg.get("enabled", False)):
        return None

    from scripts.flightmare_bc.eval_controller import evaluate_controller_config

    result = evaluate_controller_config(
        cfg,
        episodes=int(eval_cfg.get("episodes", 8)),
        speed_bins=int(eval_cfg.get("speed_bins", 1)),
        seed=int(cfg["seed"]),
    )
    summary = result["summary"]
    min_success = float(eval_cfg.get("min_success_rate", 0.0))
    min_gate_completion = float(eval_cfg.get("min_gate_completion", 0.0))
    success_rate = float(summary.get("success_rate", 0.0))
    gate_completion = float(summary.get("mean_gate_completion", 0.0))
    # Per-scenario worst-case gates the dataset: aggregate success can mask one
    # broken scenario family, so refuse if ANY family is below threshold.
    by_scenario = result.get("by_scenario") or []
    min_scenario_success = float(summary.get("min_scenario_success_rate", success_rate))
    min_scenario_completion = float(summary.get("min_scenario_gate_completion", gate_completion))
    print(
        "[collect_dataset] controller preflight: "
        f"success={success_rate:.3f} gate_completion={gate_completion:.3f} "
        f"track_err={float(summary.get('mean_track_err_m', 0.0)):.3f}m",
        flush=True,
    )
    for s in by_scenario:
        print(
            f"[collect_dataset]   scenario={s.get('scenario_family')}: "
            f"success={float(s.get('success_rate', 0.0)):.3f} "
            f"gate_completion={float(s.get('mean_gate_completion', 0.0)):.3f}",
            flush=True,
        )
    if (
        success_rate < min_success
        or gate_completion < min_gate_completion
        or min_scenario_success < min_success
        or min_scenario_completion < min_gate_completion
    ):
        worst = summary.get("worst_scenario_family", "aggregate")
        raise RuntimeError(
            "Controller preflight failed; refusing to collect labels from this expert. "
            f"aggregate success={success_rate:.3f}/{min_success:.3f}, "
            f"worst-family({worst}) success={min_scenario_success:.3f}/{min_success:.3f}, "
            f"completion={min_scenario_completion:.3f}/{min_gate_completion:.3f}. "
            "Run scripts.flightmare_bc.eval_controller to inspect failing scenarios/speeds."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-controller-eval", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    args = parser.parse_args()

    cfg = load_collection_config(args.config)
    if args.episodes is not None:
        cfg["episodes"] = int(args.episodes)
    if args.out is not None:
        cfg["output_dir"] = str(args.out)

    out_dir = Path(cfg["output_dir"])
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    params = _quad_params(cfg)
    controller = _expert_controller(cfg, params)
    manifest = _manifest_header(cfg, params)
    if args.skip_controller_eval:
        manifest["controller_eval"] = {"enabled": False, "skipped": True}
    else:
        manifest["controller_eval"] = _run_controller_preflight(cfg)
    ns = _collector_namespace(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    counts = mode_counts(int(cfg["episodes"]), cfg)
    modes = [mode for mode, count in counts.items() for _ in range(count)]
    rng.shuffle(modes)

    policy_cfg = policy = None
    if counts.get("dagger", 0) > 0 and cfg.get("dagger", {}).get("enabled", True):
        policy_cfg, policy, ckpt = _load_dagger_policy(cfg, "cuda" if __import__("torch").cuda.is_available() else "cpu")
        if policy is None:
            print("[collect_dataset] DAgger checkpoint missing; reassigning DAgger episodes to synthetic_recovery.")
            modes = ["synthetic_recovery" if m == "dagger" else m for m in modes]
        else:
            print(f"[collect_dataset] DAgger policy: {ckpt}")

    env = None
    needs_env = any(m in {"expert", "dagger"} for m in modes)
    if needs_env:
        env = FlightmareExpertEnv(
            image_size=int(cfg["image_size"]),
            cameras=tuple(cfg.get("cameras", [])),
            control_hz=float(cfg["control_hz"]),
            params=params,
            scene=str(cfg["scene"]),
            render=bool(cfg.get("render", False)),
            seed=int(cfg["seed"]),
            visual_backend=True,
            backend=str(cfg["backend"]),
        )
    t0 = time.time()
    try:
        for ep, mode in enumerate(modes):
            ep_path = out_dir / "episodes" / f"ep_{ep:06d}.h5"
            with EpisodeWriter(
                ep_path,
                image_size=int(cfg["image_size"]),
                cameras=cfg.get("cameras", []),
                state_dim=STATE_DIM,
                controller_name=f"{manifest['controller']}:{mode}",
                dt=1.0 / float(cfg["control_hz"]),
                seed=int(cfg["seed"]) + ep,
                episode_id=ep,
            ) as writer:
                if mode == "expert":
                    if env is None:
                        raise RuntimeError("expert collection requires FlightmareExpertEnv.")
                    info = collect_one_episode(env, controller, rng, ns, ep, writer, float(ns.lookahead_s))
                    info["sample_kind"] = "expert"
                elif mode == "dagger":
                    if env is None or policy is None:
                        raise RuntimeError("dagger collection requires env and loaded policy.")
                    info = collect_dagger_episode(cfg, params, controller, env, policy_cfg, policy, rng, ep, writer)
                elif mode == "synthetic_recovery":
                    info = collect_synthetic_recovery_episode(cfg, params, controller, rng, ep, writer)
                else:
                    raise ValueError(f"Unknown collection mode {mode!r}.")
            info["path"] = str(ep_path.relative_to(out_dir))
            manifest["episodes"].append(info)
            if (ep + 1) % 10 == 0 or ep == len(modes) - 1:
                elapsed = time.time() - t0
                print(
                    f"[collect_dataset] {ep + 1}/{len(modes)} mode={mode} "
                    f"len={info['length']} err={info['mean_track_err']:.2f}m "
                    f"v={info['avg_speed']:.1f}m/s ({elapsed:.0f}s)",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()

    _assign_splits(out_dir, manifest, cfg)
    with open(out_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    train_files = [out_dir / ep["path"] for ep in manifest["episodes"] if ep.get("split") == "train"]
    write_norm_stats(
        out_dir,
        train_files,
        action_normalization=str(cfg["action_normalization"]),
        skip_initial_frames=int(cfg["skip_initial_frames"]),
        bounds_margin=float(cfg["bounds_margin"]),
    )
    if not args.skip_postprocess:
        _run_postprocess(out_dir, cfg)

    print(f"[collect_dataset] wrote {len(manifest['episodes'])} episodes -> {out_dir}")


if __name__ == "__main__":
    main()

"""Evaluate state-only Flightmare BC policies in a ROS-style autonomy stack.

Examples:
    python -m src.entrypoints.eval_flightmare \
        --exp flightmare/flightmare_ctbr_bc_state \
        --episodes 20

    python -m src.entrypoints.eval_flightmare \
        --config configs/exp/flightmare/flightmare_waypoint_bc_state.yaml \
        --episodes 20
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.mission import MISSION_DIM
from src.core.config import Config
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
from src.robotics.flightmare_autonomy_fsw.plotting import (
    save_trajectory_plot,
    summarize_results,
    write_stats_json,
)
from src.robotics.models.flightmare.MissionWrapper import MissionWrapper

# Register Flightmare architecture builders.
import src.robotics.models.flightmare  # noqa: F401


def _load_config(args: argparse.Namespace) -> tuple[Config, str]:
    if args.exp:
        return Config.from_experiment(args.exp), args.exp

    config_path = Path(args.config)
    base_configs = [
        Path("configs/base/common.yaml"),
        Path("configs/base/robotics.yaml"),
    ]
    return Config.load(config_path, base_configs=base_configs), str(config_path)


def _infer_action_type(cfg: Config) -> str:
    data_action = (cfg.data or {}).get("action_type")
    arch_type = ((cfg.robotics or {}).get("architecture", {}) or {}).get("type", "")

    arch_action = None
    if "waypoint" in arch_type:
        arch_action = "waypoint"
    elif "ctbr" in arch_type:
        arch_action = "ctbr"
    elif "motor" in arch_type:
        arch_action = "motor"

    action_type = data_action or arch_action
    if action_type not in {"waypoint", "ctbr", "motor"}:
        raise ValueError(
            "Could not infer Flightmare action type. Set data.action_type to "
            "'waypoint', 'ctbr', or 'motor', or use an architecture type containing that token."
        )
    if arch_action is not None and data_action is not None and arch_action != data_action:
        raise ValueError(
            f"Config action mismatch: data.action_type={data_action!r}, "
            f"architecture.type={arch_type!r} implies {arch_action!r}."
        )
    return action_type


def _default_checkpoint(cfg: Config) -> Path:
    output_dir = Path(cfg.training["output_dir"])
    return output_dir / "best" / "model.pt"


def _load_state_dict(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _build_model(cfg: Config, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    arch_cfg = dict((cfg.robotics or {}).get("architecture", {}))
    if not arch_cfg.get("type"):
        raise ValueError("robotics.architecture.type must be set in the config.")
    model = build("architecture", **arch_cfg)
    state_dict = _load_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _ppo_cfg(cfg: Config) -> dict:
    return dict(((cfg.robotics or {}).get("ppo", {}) or {}))


def _arg_or_ppo(args: argparse.Namespace, attr: str, ppo_cfg: dict, ppo_key: str | None = None, default=None):
    value = getattr(args, attr)
    if value is not None:
        return value
    return ppo_cfg.get(ppo_key or attr, default)


def _print_episode(result) -> None:
    time_str = f"{result.completion_time_s:.2f}s" if result.completion_time_s is not None else "-"
    print(
        f"[Eval][ep {result.episode_id:03d}] "
        f"gates={result.gates_completed}/{result.num_gates} "
        f"completed={result.completed} "
        f"time={time_str} "
        f"elapsed={result.elapsed_time_s:.2f}s "
        f"mean_speed={result.mean_speed_mps:.2f}m/s "
        f"reason={result.termination_reason}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--exp", type=str, help="Experiment name under configs/exp, e.g. flightmare/flightmare_ctbr_bc_state")
    config_group.add_argument("--config", type=str, help="Direct path to the training YAML config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model.pt. Defaults to <training.output_dir>/best/model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="Defaults to robotics.ppo.horizon, then 1500.")
    parser.add_argument("--control-hz", type=float, default=None, help="Defaults to robotics.ppo.control_hz, then 100.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene", type=str, default=None, help="Defaults to robotics.ppo.scene, then industrial.")
    parser.add_argument("--backend", choices=["auto", "numpy", "visual", "flightgym"], default=None,
                        help="Defaults to robotics.ppo.backend, then auto.")
    parser.add_argument("--render", action="store_true", help="Connect/render Unity. Default is headless state-only evaluation.")
    parser.add_argument("--fail-on-fallback", action="store_true", help="Fail if Flightmare bindings are unavailable and numpy fallback is used.")
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--num-gates", type=int, default=None)
    parser.add_argument("--course-mode", choices=["gates", "swift_like", "fixed_gates"], default=None)
    parser.add_argument("--gate-layout", type=str, default=None)
    parser.add_argument("--random-start-gate", action="store_true", default=None)
    parser.add_argument("--fixed-gate-pos-noise", type=float, default=None)
    parser.add_argument("--fixed-gate-yaw-noise", type=float, default=None)
    parser.add_argument("--gate-spacing-range", nargs=2, type=float, default=None)
    parser.add_argument("--gate-lateral-jitter", type=float, default=None)
    parser.add_argument("--gate-z-range", nargs=2, type=float, default=None)
    parser.add_argument("--gate-yaw-step", type=float, default=None)
    parser.add_argument("--gate-yaw-noise", type=float, default=None)
    parser.add_argument("--gate-size", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)

    parser.add_argument("--max-body-rate", type=float, default=None)
    parser.add_argument("--max-waypoint-speed", type=float, default=None)
    parser.add_argument("--max-collective-thrust-g", type=float, default=None)
    parser.add_argument("--max-world-radius", type=float, default=None)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--gate-vehicle-radius", type=float, default=None)
    parser.add_argument("--allow-gate-miss", action="store_true", default=None,
                        help="Do not terminate when the current gate plane is crossed outside the aperture.")
    args = parser.parse_args()

    cfg, config_label = _load_config(args)
    ppo_cfg = _ppo_cfg(cfg)
    action_type = _infer_action_type(cfg)
    device = _device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_dir = Path((cfg.data or {}).get("data_dir", ""))
    norm_stats = data_dir / "norm_stats.npz"
    if not norm_stats.exists():
        raise FileNotFoundError(f"norm_stats.npz not found: {norm_stats}")

    run_name = cfg.run.get("name", Path(config_label).stem)
    output_dir = args.output_dir or (Path(cfg.training["output_dir"]) / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] {config_label}")
    print(f"[Run]    {run_name}")
    print(f"[Action] {action_type}")
    print(f"[Device] {device}")
    print(f"[Ckpt]   {checkpoint}")
    print(f"[Stats]  {norm_stats}")

    max_steps = int(_arg_or_ppo(args, "max_steps", ppo_cfg, "horizon", 1500))
    control_hz = float(_arg_or_ppo(args, "control_hz", ppo_cfg, "control_hz", 100.0))
    scene = str(_arg_or_ppo(args, "scene", ppo_cfg, "scene", "industrial"))
    backend = str(_arg_or_ppo(args, "backend", ppo_cfg, "backend", "auto"))
    render = bool(args.render or ppo_cfg.get("render", False))
    course_mode = str(_arg_or_ppo(args, "course_mode", ppo_cfg, "course_mode", "gates"))
    gate_layout = _arg_or_ppo(args, "gate_layout", ppo_cfg, "gate_layout", None)
    random_start_gate = (
        bool(args.random_start_gate)
        if args.random_start_gate is not None
        else bool(ppo_cfg.get("random_start_gate", False))
    )
    fixed_gate_pos_noise = float(_arg_or_ppo(args, "fixed_gate_pos_noise", ppo_cfg, "fixed_gate_pos_noise", 0.0))
    fixed_gate_yaw_noise = float(_arg_or_ppo(args, "fixed_gate_yaw_noise", ppo_cfg, "fixed_gate_yaw_noise", 0.0))
    num_gates = int(_arg_or_ppo(args, "num_gates", ppo_cfg, "num_gates", 8))
    gate_spacing_range = list(_arg_or_ppo(args, "gate_spacing_range", ppo_cfg, "gate_spacing_range", [4.0, 9.0]))
    gate_lateral_jitter = float(_arg_or_ppo(args, "gate_lateral_jitter", ppo_cfg, "gate_lateral_jitter", 2.0))
    gate_z_range = list(_arg_or_ppo(args, "gate_z_range", ppo_cfg, "gate_z_range", [1.5, 4.0]))
    gate_yaw_step = float(_arg_or_ppo(args, "gate_yaw_step", ppo_cfg, "gate_yaw_step", 0.7))
    gate_yaw_noise = float(_arg_or_ppo(args, "gate_yaw_noise", ppo_cfg, "gate_yaw_noise", 0.25))
    gate_size = float(_arg_or_ppo(args, "gate_size", ppo_cfg, "gate_size", 1.0))
    z_min = float(_arg_or_ppo(args, "z_min", ppo_cfg, "z_min", 1.0))
    max_body_rate = float(_arg_or_ppo(args, "max_body_rate", ppo_cfg, "max_body_rate", 8.0))
    max_waypoint_speed = float(_arg_or_ppo(args, "max_waypoint_speed", ppo_cfg, "max_waypoint_speed", 15.0))
    max_collective_thrust_g = float(_arg_or_ppo(args, "max_collective_thrust_g", ppo_cfg, "max_collective_thrust_g", 4.0))
    max_world_radius = float(_arg_or_ppo(args, "max_world_radius", ppo_cfg, "max_world_radius", 200.0))
    min_z = float(_arg_or_ppo(args, "min_z", ppo_cfg, "min_z", -0.25))
    gate_vehicle_radius = float(_arg_or_ppo(args, "gate_vehicle_radius", ppo_cfg, "gate_vehicle_radius", 0.15))
    terminate_on_gate_miss = (
        not args.allow_gate_miss
        if args.allow_gate_miss is not None
        else bool(ppo_cfg.get("terminate_on_gate_miss", True))
    )

    print(
        "[Eval Setup] "
        f"backend={backend} course={course_mode} gates={num_gates} spacing={gate_spacing_range} "
        f"yaw_step={gate_yaw_step} gate_size={gate_size} "
        f"max_waypoint_speed={max_waypoint_speed} max_body_rate={max_body_rate} "
        f"max_thrust={max_collective_thrust_g:.1f}g",
        flush=True,
    )

    model = _build_model(cfg, checkpoint, device)
    arch_cfg = (cfg.robotics or {}).get("architecture", {})
    include_mission = bool((cfg.data or {}).get("include_mission", True))
    concat_mission_to_state = include_mission and int(arch_cfg.get("state_dim", 13)) == 13 + MISSION_DIM

    mission_wrapper = MissionWrapper(
        policy=model,
        gates=[],
        norm_stats=norm_stats,
        action_type=action_type,
        device=device,
        include_mission=include_mission,
        concat_mission_to_state=concat_mission_to_state,
    )

    params = QuadParams()
    params.max_collective_thrust = max_collective_thrust_g * params.mass * params.g
    course_config = CourseConfig(
        course_mode=course_mode,
        num_gates=num_gates,
        z_min=z_min,
        gate_spacing_range=tuple(gate_spacing_range),
        gate_lateral_jitter=gate_lateral_jitter,
        gate_z_range=tuple(gate_z_range),
        gate_yaw_step=gate_yaw_step,
        gate_yaw_noise=gate_yaw_noise,
        gate_size=gate_size,
        gate_layout=gate_layout,
        random_start_gate=random_start_gate,
        fixed_gate_pos_noise=fixed_gate_pos_noise,
        fixed_gate_yaw_noise=fixed_gate_yaw_noise,
    )
    state_node = FlightmareStateNode(
        control_hz=control_hz,
        course_config=course_config,
        params=params,
        scene=scene,
        render=render,
        seed=args.seed,
        backend=backend,
    )
    if state_node.using_fallback:
        if backend == "numpy":
            msg = (
                "[Eval] Using configured numpy backend. This matches numpy-backend PPO "
                "rollouts but is not a real Flightmare/Unity dynamics validation."
            )
        else:
            msg = (
                "[Eval] WARNING: Flightmare bindings unavailable, using numpy fallback. "
                "This is useful for pipeline smoke tests, not authoritative racing metrics."
            )
            if args.fail_on_fallback:
                state_node.close()
                raise RuntimeError(msg)
        print(msg)

    controller = BaseAutonomyController(
        params=params,
        waypoint_controller=WaypointLQRController(params=params, max_speed=max_waypoint_speed),
        ctbr_controller=BaseCTBRController(params=params, max_body_rate=max_body_rate),
        motor_controller=BaseMotorController(params=params),
    )
    graph = FlightmareAutonomyGraph(
        state_node=state_node,
        mission_node=MissionWorldModelNode(mission_wrapper),
        planner_node=PolicyPlannerNode(mission_wrapper, action_type=action_type),
        controller_node=BaseControllerNode(controller),
        max_steps=max_steps,
        max_world_radius=max_world_radius,
        min_z=min_z,
        gate_vehicle_radius=gate_vehicle_radius,
        terminate_on_gate_miss=terminate_on_gate_miss,
    )

    rng = np.random.default_rng(args.seed)
    results = []
    try:
        for ep in range(args.episodes):
            result = graph.run_episode(ep, rng)
            results.append(result)
            _print_episode(result)
    finally:
        graph.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = output_dir / f"{timestamp}_stats.json"
    plot_path = output_dir / f"{timestamp}_trajectories.png"
    extra = {
        "config": config_label,
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "action_type": action_type,
        "backend": backend,
        "control_hz": control_hz,
        "max_steps": max_steps,
        "max_waypoint_speed": max_waypoint_speed,
        "max_body_rate": max_body_rate,
        "max_collective_thrust_g": max_collective_thrust_g,
        "seed": args.seed,
        "course": {
            "course_mode": course_mode,
            "num_gates": num_gates,
            "gate_layout": gate_layout,
            "random_start_gate": random_start_gate,
            "fixed_gate_pos_noise": fixed_gate_pos_noise,
            "fixed_gate_yaw_noise": fixed_gate_yaw_noise,
            "gate_spacing_range": gate_spacing_range,
            "gate_lateral_jitter": gate_lateral_jitter,
            "gate_z_range": gate_z_range,
            "gate_yaw_step": gate_yaw_step,
            "gate_yaw_noise": gate_yaw_noise,
            "gate_size": gate_size,
            "gate_vehicle_radius": gate_vehicle_radius,
            "strict_gate_aperture": terminate_on_gate_miss,
        },
    }
    write_stats_json(results, stats_path, extra=extra)
    if not args.no_plot:
        save_trajectory_plot(results, plot_path, title=f"{run_name} ({action_type})")

    summary = summarize_results(results)
    print("\n[Summary]")
    print(f"  success_rate: {100.0 * summary.get('success_rate', 0.0):.1f}%")
    print(f"  mean_gates_completed: {summary.get('mean_gates_completed', 0.0):.2f}")
    print(f"  mean_gate_completion: {100.0 * summary.get('mean_gate_completion', 0.0):.1f}%")
    print(f"  gate_miss_rate: {100.0 * summary.get('gate_miss_rate', 0.0):.1f}%")
    if summary.get("mean_completion_time_s") is not None:
        print(f"  mean_completion_time_s: {summary['mean_completion_time_s']:.2f}")
    print(f"  mean_speed_mps: {summary.get('mean_speed_mps', 0.0):.2f}")
    print(f"  stats: {stats_path}")
    if not args.no_plot:
        print(f"  plot:  {plot_path}")


if __name__ == "__main__":
    main()

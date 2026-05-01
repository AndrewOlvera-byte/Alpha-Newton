"""Collect privileged-expert BC episodes in Flightmare.

Pipeline (per episode):
  1. Sample a randomized gate course (or random waypoints in legacy mode).
  2. Fit a minimum-jerk polynomial through gate centers at a target avg speed.
  3. At each control tick, query the SE(3) geometric controller for a CTBR
     command, step Flightmare through the selected backend, and log
     waypoint+speed and CTBR action labels.
  4. Also log a per-step "mission" vector: the next K gates expressed in body
     frame, plus gate progress + distance to next gate. This is the
     map-relative perception prior a real autonomy stack would synthesize from
     a known track + VIO pose, and is what the deployment-time MissionWrapper
     reproduces from raw inputs.
  5. Save a per-episode HDF5 file under ``<out>/episodes/ep_NNNNNN.h5`` and
     update an ``index.json`` manifest with episode lengths + train/val split.
  6. After collection, scan the training split to write ``norm_stats.npz``
     (per-action-type mean/std + state mean/std + mission mean/std) for BC.

Run from repo root:

    python -m scripts.flightmare_bc.collect \\
        --out data/flightmare/expert_v1 \\
        --episodes 500 \\
        --image-size 224 \\
        --control-hz 100 \\
        --cameras forward \\
        --no-render \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.flightmare_bc.controllers import (
    GeometricSE3Controller,
    QuadParams,
    world_to_body,
)
from scripts.flightmare_bc.action_norms import ACTION_TYPES, DEFAULT_ACTION_BOUNDS
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv, GateSpec
from scripts.flightmare_bc.hdf5_writer import EpisodeWriter
from scripts.flightmare_bc.mission import (
    LOOKAHEAD_GATES,
    MISSION_DIM,
    MissionTracker,
)
from scripts.flightmare_bc.trajectories import (
    MinJerkTrajectory,
    random_waypoint_path,
)


# State vector layout used everywhere downstream (13-dim):
#   [pos(3), vel(3), quat(w,x,y,z) (4), omega_body(3)]
STATE_DIM = 13


def build_state(pos, vel, quat, omega) -> np.ndarray:
    return np.concatenate([pos, vel, quat, omega]).astype(np.float32)


def waypoint_speed_label(
    pos: np.ndarray, quat: np.ndarray, lookahead_pos: np.ndarray, speed: float,
) -> np.ndarray:
    """Body-frame waypoint vector + scalar reference speed - the high-level head's target."""
    delta_world = lookahead_pos - pos
    delta_body = world_to_body(delta_world, quat)
    return np.array([delta_body[0], delta_body[1], delta_body[2], speed], dtype=np.float32)


def ctbr_label(thrust_norm: float, omega_des: np.ndarray) -> np.ndarray:
    return np.array([thrust_norm, omega_des[0], omega_des[1], omega_des[2]], dtype=np.float32)


def motor_label(motor_normalized: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(motor_normalized, dtype=np.float32), 0.0, 1.0)


def _cfg(cfg: argparse.Namespace, name: str, default):
    return getattr(cfg, name, default)


def _gate_size_array(cfg: argparse.Namespace, gate_dict: dict | None = None) -> np.ndarray:
    if gate_dict is not None and "size" in gate_dict:
        size = gate_dict["size"]
        if isinstance(size, (int, float)):
            return np.array([size, size, size], dtype=np.float64)
        return np.asarray(size, dtype=np.float64)
    gate_size = float(_cfg(cfg, "gate_size", 1.0))
    return np.array([gate_size, gate_size, gate_size], dtype=np.float64)


def _compute_path_yaws(centers: np.ndarray) -> list[float]:
    yaws: list[float] = []
    n = int(len(centers))
    for i in range(n):
        if n <= 1:
            tangent = np.array([1.0, 0.0, 0.0])
        elif i == 0:
            tangent = centers[1] - centers[0]
        elif i == n - 1:
            tangent = centers[-1] - centers[-2]
        else:
            tangent = centers[i + 1] - centers[i - 1]
        yaws.append(float(np.arctan2(tangent[1], tangent[0])))
    return yaws


def _gates_from_centers(
    centers: np.ndarray,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
    *,
    prefix: str,
    yaws: list[float] | None = None,
    sizes: list[np.ndarray] | None = None,
) -> list[GateSpec]:
    centers = np.asarray(centers, dtype=np.float64).copy()
    if bool(_cfg(cfg, "random_start_gate", False)) and len(centers) > 1:
        offset = int(rng.integers(0, len(centers)))
        centers = np.roll(centers, -offset, axis=0)
        if yaws is not None:
            yaws = list(np.roll(np.asarray(yaws, dtype=np.float64), -offset))
        if sizes is not None:
            sizes = list(np.roll(np.asarray(sizes, dtype=np.float64), -offset, axis=0))

    pos_noise = float(_cfg(cfg, "fixed_gate_pos_noise", 0.0))
    if pos_noise > 0.0:
        centers += rng.normal(0.0, pos_noise, size=centers.shape)
        centers[:, 2] = np.maximum(centers[:, 2], float(_cfg(cfg, "z_min", 1.0)))

    if yaws is None:
        yaws = _compute_path_yaws(centers)

    yaw_noise = float(_cfg(cfg, "fixed_gate_yaw_noise", _cfg(cfg, "gate_yaw_noise", 0.0)))
    gates: list[GateSpec] = []
    for i, center in enumerate(centers):
        yaw = float(yaws[i])
        if yaw_noise > 0.0:
            yaw += float(rng.uniform(-yaw_noise, yaw_noise))
        size = sizes[i] if sizes is not None else _gate_size_array(cfg)
        gates.append(
            GateSpec(
                gate_id=f"{prefix}_{i:03d}",
                pos=center.copy(),
                yaw=yaw,
                size=np.asarray(size, dtype=np.float64).copy(),
            )
        )
    return gates


def _swift_like_gate_course(rng: np.random.Generator, cfg: argparse.Namespace) -> list[GateSpec]:
    """Seven-gate, ~75-80 m flowing course in a Swift-scale 30x30x8 m volume.

    This is intentionally named "swift_like" rather than "swift": it matches
    the published scale and gate count, while allowing a real measured layout
    to be supplied through --gate-layout when exact geometry is available.
    """
    centers = np.array([
        [3.0, 0.0, 2.0],
        [15.0, 0.0, 2.2],
        [27.0, 5.0, 3.0],
        [28.0, 18.0, 5.0],
        [17.0, 26.0, 3.4],
        [5.0, 21.0, 2.6],
        [8.0, 9.0, 2.8],
    ], dtype=np.float64)
    num_gates = int(_cfg(cfg, "num_gates", len(centers)))
    if 0 < num_gates < len(centers):
        # Keep the same scale while allowing shorter smoke tests.
        idx = np.linspace(0, len(centers) - 1, max(1, num_gates)).round().astype(int)
        centers = centers[idx]
    return _gates_from_centers(centers, rng, cfg, prefix="swift_like")


def _layout_gate_course(rng: np.random.Generator, cfg: argparse.Namespace) -> list[GateSpec]:
    layout_path = _cfg(cfg, "gate_layout", None)
    if not layout_path:
        raise ValueError("course_mode='fixed_gates' requires --gate-layout")
    with open(Path(layout_path)) as f:
        data = json.load(f)
    raw_gates = data.get("gates", data) if isinstance(data, dict) else data
    centers = []
    yaws: list[float | None] = []
    sizes: list[np.ndarray] = []
    for i, g in enumerate(raw_gates):
        if not isinstance(g, dict):
            raise ValueError(f"Gate layout entry {i} must be an object with pos/yaw/size fields.")
        pos = g.get("pos", g.get("position", g.get("center")))
        if pos is None:
            raise ValueError(f"Gate layout entry {i} missing pos/position/center.")
        centers.append(np.asarray(pos, dtype=np.float64))
        yaws.append(None if "yaw" not in g else float(g["yaw"]))
        sizes.append(_gate_size_array(cfg, g))
    centers_arr = np.stack(centers, axis=0)
    computed_yaws = _compute_path_yaws(centers_arr)
    resolved_yaws = [computed_yaws[i] if yaw is None else float(yaw) for i, yaw in enumerate(yaws)]
    return _gates_from_centers(centers_arr, rng, cfg, prefix="fixed_gate", yaws=resolved_yaws, sizes=sizes)


def sample_gate_course(rng: np.random.Generator, cfg: argparse.Namespace) -> list[GateSpec]:
    """Generate a forward-progressing randomized gate course."""
    course_mode = str(_cfg(cfg, "course_mode", "gates"))
    if course_mode == "swift_like":
        return _swift_like_gate_course(rng, cfg)
    if course_mode == "fixed_gates":
        return _layout_gate_course(rng, cfg)

    gates: list[GateSpec] = []
    gate_z_range = tuple(_cfg(cfg, "gate_z_range", (1.5, 4.0)))
    gate_spacing_range = tuple(_cfg(cfg, "gate_spacing_range", (4.0, 9.0)))
    z_min = float(_cfg(cfg, "z_min", 1.0))
    pos = np.array([0.0, 0.0, max(z_min, gate_z_range[0])], dtype=np.float64)
    heading = float(rng.uniform(-0.4, 0.4))
    for i in range(int(_cfg(cfg, "num_gates", 8))):
        spacing = float(rng.uniform(gate_spacing_range[0], gate_spacing_range[1]))
        heading += float(rng.uniform(-float(_cfg(cfg, "gate_yaw_step", 0.7)), float(_cfg(cfg, "gate_yaw_step", 0.7))))
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        pos = pos + spacing * forward
        jitter = float(_cfg(cfg, "gate_lateral_jitter", 2.0))
        pos = pos + rng.uniform(-jitter, jitter) * lateral
        pos[2] = float(rng.uniform(gate_z_range[0], gate_z_range[1]))
        gates.append(
            GateSpec(
                gate_id=f"gate_{i:03d}",
                pos=pos.copy(),
                yaw=heading + float(rng.uniform(-float(_cfg(cfg, "gate_yaw_noise", 0.25)), float(_cfg(cfg, "gate_yaw_noise", 0.25)))),
                size=_gate_size_array(cfg),
            )
        )
    return gates


def waypoints_from_gates(
    gates: list[GateSpec], z_min: float, d_approach: float = 1.2,
) -> np.ndarray:
    """Pre / center / exit waypoints aligned with each gate's forward axis.

    Inserting an approach point ``d_approach`` meters before the gate and an
    exit point ``d_approach`` meters after, both along the gate's forward
    direction, forces the min-jerk polynomial to be locally straight and
    aligned with the gate normal at the crossing. This eliminates corner-
    cutting through gate frames and (because tangent yaw equals gate yaw
    when the velocity is along the gate forward axis) auto-aligns the body
    yaw with each gate.
    """
    if not gates:
        return np.array([[0.0, 0.0, max(z_min, 2.0)]], dtype=np.float64)
    points: list[np.ndarray] = []
    f0 = np.array([np.cos(gates[0].yaw), np.sin(gates[0].yaw), 0.0])
    points.append(gates[0].pos - 3.0 * f0)  # lead-in
    for g in gates:
        f = np.array([np.cos(g.yaw), np.sin(g.yaw), 0.0])
        points.append(g.pos - d_approach * f)
        points.append(g.pos.copy())
        points.append(g.pos + d_approach * f)
    fN = np.array([np.cos(gates[-1].yaw), np.sin(gates[-1].yaw), 0.0])
    points.append(gates[-1].pos + 3.0 * fN)  # lead-out
    return np.stack(points, axis=0)


def launch_unity_if_requested(args: argparse.Namespace) -> subprocess.Popen | None:
    if not args.launch_unity:
        return None
    exe = Path(os.environ.get("FLIGHTMARE_UNITY_EXECUTABLE", "/opt/flightmare/flightrender/RPG_Flightmare.x86_64"))
    if not exe.exists():
        raise FileNotFoundError(f"Flightmare Unity executable not found: {exe}")
    extra_args = os.environ.get("FLIGHTMARE_UNITY_ARGS", "").split()
    cmd = [str(exe), *extra_args]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(max(0.0, float(args.unity_startup_s)))
    return proc


def collect_one_episode(
    env: FlightmareExpertEnv,
    controller: GeometricSE3Controller,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
    episode_id: int,
    writer: EpisodeWriter,
    lookahead_s: float,
) -> dict:
    gates: list[GateSpec] = []
    if cfg.course_mode in {"gates", "swift_like", "fixed_gates"}:
        gates = sample_gate_course(rng, cfg)
        # Prefix gate IDs with episode index so reused Unity instances don't
        # collide on object IDs across episodes (older gates remain in the
        # scene visually but are unused by the controller / mission tracker).
        for g in gates:
            g.gate_id = f"ep{episode_id:06d}_{g.gate_id}"
        env.add_gates(gates)
        waypoints = waypoints_from_gates(gates, cfg.z_min)
        obs = env.reset(init_pos=waypoints[0], yaw=float(gates[0].yaw))
    else:
        obs = env.reset(init_pos=np.zeros(3), yaw=0.0)
        waypoints = random_waypoint_path(
            rng,
            n_waypoints=cfg.n_waypoints,
            bbox=tuple(cfg.bbox),
            z_min=cfg.z_min,
            min_step=cfg.min_step,
            origin=obs.pos,
        )
    avg_speed = float(rng.uniform(cfg.speed_range[0], cfg.speed_range[1]))
    traj = MinJerkTrajectory(waypoints, avg_speed=avg_speed, yaw_mode="tangent")
    mission = MissionTracker(gates, lookahead=LOOKAHEAD_GATES)

    n_steps = int(traj.total_time / env.dt)
    n_steps = max(1, min(n_steps, cfg.max_steps))

    pos_errors: list[float] = []
    written = 0
    for k in range(n_steps):
        t = k * env.dt
        ref = traj.sample(t)
        cmd = controller.compute(
            pos=obs.pos, vel=obs.vel, quat=obs.quat,
            pos_des=ref.pos, vel_des=ref.vel, acc_des=ref.acc,
            yaw_des=ref.yaw,
        )

        lookahead_pos, speed_now = traj.lookahead(t, lookahead_s)
        waypoint_act = waypoint_speed_label(obs.pos, obs.quat, lookahead_pos, speed_now)
        ctbr_act = ctbr_label(cmd["thrust_normalized"], cmd["body_rates"])
        motor_act = motor_label(cmd["motor_normalized"])

        mission.update(obs.pos)
        mission_vec = mission.vector(obs.pos, obs.quat)

        state_vec = build_state(obs.pos, obs.vel, obs.quat, obs.omega)
        writer.append(
            state=state_vec,
            images=obs.images,
            actions={"waypoint": waypoint_act, "ctbr": ctbr_act, "motor": motor_act},
            ref_pos=ref.pos.astype(np.float32),
            ref_vel=ref.vel.astype(np.float32),
            ref_yaw=float(ref.yaw),
            done=(k == n_steps - 1),
            mission=mission_vec,
            gate_index=mission.current_index,
        )
        written += 1
        pos_errors.append(float(np.linalg.norm(obs.pos - ref.pos)))

        obs = env.step_ctbr(cmd["thrust_newton"], cmd["body_rates"])
        if obs.done or not np.all(np.isfinite(obs.pos)) or np.linalg.norm(obs.pos) > float(cfg.max_world_radius):
            break

    return {
        "episode_id": episode_id,
        "length": written,
        "avg_speed": avg_speed,
        "mean_track_err": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_track_err": float(np.max(pos_errors)) if pos_errors else 0.0,
        "n_waypoints": int(len(waypoints)),
        "gates": [
            {
                "id": g.gate_id,
                "pos": g.pos.astype(float).tolist(),
                "yaw": float(g.yaw),
                "size": g.size.astype(float).tolist(),
            }
            for g in gates
        ],
    }


def write_norm_stats(out_dir: Path, train_files: list[Path], action_normalization: str = "standard") -> None:
    """Per-channel mean/std for state and each action type, computed over train split."""
    import h5py

    sums = {"state": None, "mission": None, **{action_type: None for action_type in ACTION_TYPES}}
    sq_sums = {k: None for k in sums}
    counts = {k: 0 for k in sums}

    def acc(key: str, arr: np.ndarray) -> None:
        nonlocal sums, sq_sums, counts
        if sums[key] is None:
            sums[key] = np.zeros(arr.shape[1], dtype=np.float64)
            sq_sums[key] = np.zeros(arr.shape[1], dtype=np.float64)
        sums[key] += arr.sum(axis=0)
        sq_sums[key] += (arr.astype(np.float64) ** 2).sum(axis=0)
        counts[key] += arr.shape[0]

    for fp in train_files:
        with h5py.File(fp, "r") as f:
            acc("state", f["obs/state"][...])
            for action_type in ACTION_TYPES:
                key = f"action/{action_type}"
                if key in f:
                    acc(action_type, f[key][...])
            if "mission/vec" in f:
                acc("mission", f["mission/vec"][...])

    out = {}
    for k in sums:
        if counts[k] == 0:
            continue
        mean = sums[k] / counts[k]
        var = sq_sums[k] / counts[k] - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))
        out[f"{k}_mean"] = mean.astype(np.float32)
        out[f"{k}_std"] = std.astype(np.float32)
    out["action_normalization"] = np.array(str(action_normalization))
    for action_type, (low, high) in DEFAULT_ACTION_BOUNDS.items():
        out[f"{action_type}_low"] = low.astype(np.float32)
        out[f"{action_type}_high"] = high.astype(np.float32)
    np.savez(out_dir / "norm_stats.npz", **out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/flightmare/expert_v1"))
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cameras", nargs="*", default=[])
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--n-waypoints", type=int, default=6)
    p.add_argument("--bbox", nargs=3, type=float, default=[8.0, 8.0, 3.5])
    p.add_argument("--z-min", type=float, default=1.0)
    p.add_argument("--min-step", type=float, default=1.5)
    p.add_argument("--speed-range", nargs=2, type=float, default=[2.0, 8.0])
    p.add_argument("--lookahead-s", type=float, default=0.3,
                   help="Look-ahead horizon for the waypoint+speed action label.")
    p.add_argument("--scene", type=str, default="industrial")
    p.add_argument("--backend", choices=["numpy", "auto", "visual", "flightgym"], default="numpy",
                   help="Simulation backend. v3 state-only collection defaults to numpy for speed/reproducibility.")
    p.add_argument("--action-normalization", choices=["standard", "bounds"], default="standard",
                   help="Action normalization metadata written to norm_stats.npz. v3 configs use bounds.")
    p.add_argument("--course-mode", choices=["random", "gates", "swift_like", "fixed_gates"], default="random")
    p.add_argument("--gate-layout", type=Path, default=None,
                   help="JSON gate layout for --course-mode fixed_gates. Entries need pos and optional yaw/size.")
    p.add_argument("--random-start-gate", action="store_true",
                   help="Rotate fixed/swift-like gate order each episode for start-gate diversity.")
    p.add_argument("--fixed-gate-pos-noise", type=float, default=0.0,
                   help="Per-axis Gaussian position noise for fixed/swift-like gates.")
    p.add_argument("--fixed-gate-yaw-noise", type=float, default=0.0,
                   help="Uniform yaw noise for fixed/swift-like gates.")
    p.add_argument("--num-gates", type=int, default=8)
    p.add_argument("--gate-spacing-range", nargs=2, type=float, default=[4.0, 9.0])
    p.add_argument("--gate-lateral-jitter", type=float, default=2.0)
    p.add_argument("--gate-z-range", nargs=2, type=float, default=[1.5, 4.0])
    p.add_argument("--gate-yaw-step", type=float, default=0.7)
    p.add_argument("--gate-yaw-noise", type=float, default=0.25)
    p.add_argument("--gate-size", type=float, default=1.0)
    p.add_argument("--max-world-radius", type=float, default=350.0)
    p.add_argument("--max-collective-thrust-g", type=float, default=4.0,
                   help="Total thrust ceiling in multiples of vehicle weight.")
    p.add_argument("--launch-unity", action="store_true",
                   help="Launch FLIGHTMARE_UNITY_EXECUTABLE before connecting.")
    p.add_argument("--unity-startup-s", type=float, default=3.0,
                   help="Seconds to wait after --launch-unity before connecting.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--min-gate-completion", type=float, default=0.999,
                   help="Episodes that complete a smaller fraction of gates are "
                        "marked split='discard' and excluded from train/val + norm_stats.")
    p.add_argument("--render", action="store_true",
                   help="Enable Unity rendering and image collection (vision mode). "
                        "Default: state-only collection (no Unity, no image datasets).")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help=argparse.SUPPRESS)
    # Multi-process / shard mode (set by parallel_collect.py orchestrator).
    # When --shard-manifest is set, we skip the val-split + norm_stats step
    # (the orchestrator does both after all shards finish).
    p.add_argument("--ep-start", type=int, default=None,
                   help="Inclusive global episode id this shard begins at.")
    p.add_argument("--ep-end", type=int, default=None,
                   help="Exclusive global episode id this shard stops at.")
    p.add_argument("--shard-manifest", type=Path, default=None,
                   help="If set, write the shard's episode list to this JSON path "
                        "instead of the consolidated index.json.")
    p.add_argument("--shard-id", type=int, default=0)
    args = p.parse_args()
    state_only = not args.render
    if state_only:
        if args.cameras:
            print(f"[collect] state-only mode: ignoring --cameras {args.cameras}")
        args.cameras = []
        if args.course_mode in {"gates", "swift_like", "fixed_gates"}:
            print("[collect] state-only gate course: gates logged but not visualized.")

    out: Path = args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = QuadParams()
    params.max_collective_thrust = float(args.max_collective_thrust_g) * params.mass * params.g
    controller = GeometricSE3Controller(params=params)
    unity_proc = launch_unity_if_requested(args)

    manifest = {
        "version": 1,
        "controller": "se3_geometric+min_jerk",
        "image_size": args.image_size,
        "cameras": list(args.cameras),
        "control_hz": args.control_hz,
        "state_dim": STATE_DIM,
        "state_layout": ["pos(3)", "vel(3)", "quat_wxyz(4)", "omega_body(3)"],
        "action_normalization": args.action_normalization,
        "backend": args.backend,
        "action_types": {
            "waypoint": {"dim": 4, "layout": ["dx_body", "dy_body", "dz_body", "speed"]},
            "ctbr":     {"dim": 4, "layout": ["thrust_norm", "wx", "wy", "wz"]},
            "motor":    {"dim": 4, "layout": ["m0_norm", "m1_norm", "m2_norm", "m3_norm"]},
        },
        "mission": {
            "dim": MISSION_DIM,
            "lookahead_gates": LOOKAHEAD_GATES,
            "layout": (
                [f"gate{i}_{c}" for i in range(LOOKAHEAD_GATES) for c in ("dx_b", "dy_b", "dz_b", "yaw_rel")]
                + ["gate_progress", "dist_to_next"]
            ),
        },
        "params": asdict(params),
        "lookahead_s": args.lookahead_s,
        "course_mode": args.course_mode,
        "seed": args.seed,
        "episodes": [],
    }

    ep_start = 0 if args.ep_start is None else int(args.ep_start)
    ep_end = int(args.episodes) if args.ep_end is None else int(args.ep_end)

    t0 = time.time()
    # Build a single env and reuse it across episodes. Tearing down + rebuilding
    # the Unity bridge per episode put Unity in a half-connected state that
    # stalled subsequent renders.
    env = FlightmareExpertEnv(
        image_size=args.image_size,
        cameras=tuple(args.cameras),
        control_hz=args.control_hz,
        params=params,
        scene=args.scene,
        render=args.render,
        seed=args.seed,
        visual_backend=(args.course_mode in {"gates", "swift_like", "fixed_gates"}) or state_only,
        backend=args.backend,
    )
    if env.using_fallback:
        if args.backend == "numpy":
            print("[collect] using configured numpy backend - image datasets will be blank.")
        else:
            print("[collect] WARNING: numpy fallback active - images will be blank.")
    try:
        for ep in range(ep_start, ep_end):
            ep_path = out / "episodes" / f"ep_{ep:06d}.h5"
            with EpisodeWriter(
                ep_path,
                image_size=args.image_size,
                cameras=args.cameras,
                state_dim=STATE_DIM,
                controller_name="se3_geometric+min_jerk",
                dt=1.0 / args.control_hz,
                seed=args.seed + ep,
                episode_id=ep,
            ) as w:
                info = collect_one_episode(
                    env, controller, rng, args, ep, w, args.lookahead_s,
                )
            info["path"] = str(ep_path.relative_to(out))
            manifest["episodes"].append(info)
            done_in_shard = ep - ep_start + 1
            shard_total = ep_end - ep_start
            if done_in_shard % 10 == 0 or ep == ep_end - 1:
                elapsed = time.time() - t0
                print(f"[collect][shard{args.shard_id}] {done_in_shard}/{shard_total}  "
                      f"ep={ep} len={info['length']} err={info['mean_track_err']:.2f}m "
                      f"v={info['avg_speed']:.1f}m/s  ({elapsed:.0f}s elapsed)",
                      flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        if unity_proc is not None:
            unity_proc.terminate()
            try:
                unity_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                unity_proc.kill()
                unity_proc.wait()

    # Shard mode: write a partial manifest and exit. Orchestrator merges.
    if args.shard_manifest is not None:
        args.shard_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.shard_manifest, "w") as f:
            json.dump({"episodes": manifest["episodes"], "header": {
                k: manifest[k] for k in (
                    "version", "controller", "image_size", "cameras", "control_hz",
                    "state_dim", "state_layout", "action_types", "mission",
                    "params", "lookahead_s", "course_mode", "backend",
                    "action_normalization", "seed",
                )
            }}, f)
        print(f"[collect][shard{args.shard_id}] wrote {len(manifest['episodes'])} eps -> {args.shard_manifest}", flush=True)
        return

    # Quality filter: episodes that didn't pass enough gates are discarded.
    keep_idx: list[int] = []
    for i, ep in enumerate(manifest["episodes"]):
        n_gates = max(1, len(ep.get("gates", [])))
        # gates_passed is recorded as the final mission gate_index in the H5;
        # we approximate it via episode meta (recompute from H5 to be safe).
        import h5py
        with h5py.File(out / ep["path"], "r") as h:
            gi = h["mission/gate_index"][...]
            gates_passed = int(gi[-1]) if gi.size else 0
        ep["gates_passed"] = gates_passed
        completion = gates_passed / n_gates
        ep["gate_completion"] = float(completion)
        if completion >= args.min_gate_completion:
            keep_idx.append(i)
        else:
            ep["split"] = "discard"

    # Train/val split (episode-level, deterministic for given seed) over kept eps.
    n_keep = len(keep_idx)
    perm = np.random.default_rng(args.seed).permutation(n_keep)
    n_val = int(round(args.val_frac * n_keep))
    val_local = set(int(i) for i in perm[:n_val])
    for local_i, global_i in enumerate(keep_idx):
        manifest["episodes"][global_i]["split"] = "val" if local_i in val_local else "train"
    n_discard = len(manifest["episodes"]) - n_keep

    with open(out / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    train_files = [out / ep["path"] for ep in manifest["episodes"] if ep["split"] == "train"]
    if train_files:
        write_norm_stats(out, train_files, action_normalization=args.action_normalization)

    print(f"[collect] done. {len(manifest['episodes'])} episodes "
          f"(train+val={n_keep}, discarded={n_discard}) -> {out}")


if __name__ == "__main__":
    main()

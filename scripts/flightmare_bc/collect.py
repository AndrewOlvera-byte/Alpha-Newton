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


def sample_gate_course(rng: np.random.Generator, cfg: argparse.Namespace) -> list[GateSpec]:
    """Generate a forward-progressing randomized gate course."""
    gates: list[GateSpec] = []
    pos = np.array([0.0, 0.0, max(cfg.z_min, cfg.gate_z_range[0])], dtype=np.float64)
    heading = float(rng.uniform(-0.4, 0.4))
    for i in range(cfg.num_gates):
        spacing = float(rng.uniform(cfg.gate_spacing_range[0], cfg.gate_spacing_range[1]))
        heading += float(rng.uniform(-cfg.gate_yaw_step, cfg.gate_yaw_step))
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        pos = pos + spacing * forward
        pos = pos + rng.uniform(-cfg.gate_lateral_jitter, cfg.gate_lateral_jitter) * lateral
        pos[2] = float(rng.uniform(cfg.gate_z_range[0], cfg.gate_z_range[1]))
        gates.append(
            GateSpec(
                gate_id=f"gate_{i:03d}",
                pos=pos.copy(),
                yaw=heading + float(rng.uniform(-cfg.gate_yaw_noise, cfg.gate_yaw_noise)),
                size=np.array([cfg.gate_size, cfg.gate_size, cfg.gate_size], dtype=np.float64),
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
    proc = subprocess.Popen([str(exe)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
    if cfg.course_mode == "gates":
        gates = sample_gate_course(rng, cfg)
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

        mission.update(obs.pos)
        mission_vec = mission.vector(obs.pos, obs.quat)

        state_vec = build_state(obs.pos, obs.vel, obs.quat, obs.omega)
        writer.append(
            state=state_vec,
            images=obs.images,
            actions={"waypoint": waypoint_act, "ctbr": ctbr_act},
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
        if obs.done or not np.all(np.isfinite(obs.pos)) or np.linalg.norm(obs.pos) > 200.0:
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


def write_norm_stats(out_dir: Path, train_files: list[Path]) -> None:
    """Per-channel mean/std for state and each action type, computed over train split."""
    import h5py

    sums = {"state": None, "waypoint": None, "ctbr": None, "mission": None}
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
            acc("waypoint", f["action/waypoint"][...])
            acc("ctbr", f["action/ctbr"][...])
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
    np.savez(out_dir / "norm_stats.npz", **out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/flightmare/expert_v1"))
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cameras", nargs="+", default=["forward"])
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
    p.add_argument("--course-mode", choices=["random", "gates"], default="random")
    p.add_argument("--num-gates", type=int, default=8)
    p.add_argument("--gate-spacing-range", nargs=2, type=float, default=[4.0, 9.0])
    p.add_argument("--gate-lateral-jitter", type=float, default=2.0)
    p.add_argument("--gate-z-range", nargs=2, type=float, default=[1.5, 4.0])
    p.add_argument("--gate-yaw-step", type=float, default=0.7)
    p.add_argument("--gate-yaw-noise", type=float, default=0.25)
    p.add_argument("--gate-size", type=float, default=1.0)
    p.add_argument("--launch-unity", action="store_true",
                   help="Launch FLIGHTMARE_UNITY_EXECUTABLE before connecting.")
    p.add_argument("--unity-startup-s", type=float, default=3.0,
                   help="Seconds to wait after --launch-unity before connecting.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-render", action="store_true",
                   help="Skip Unity rendering (state-only dataset, for debugging).")
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
    if args.course_mode == "gates" and args.no_render:
        print("[collect] gate course requested with --no-render; gates are logged but not visualized.")

    out: Path = args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = QuadParams()
    controller = GeometricSE3Controller(params=params)
    unity_proc = launch_unity_if_requested(args)
    env = None

    manifest = {
        "version": 1,
        "controller": "se3_geometric+min_jerk",
        "image_size": args.image_size,
        "cameras": list(args.cameras),
        "control_hz": args.control_hz,
        "state_dim": STATE_DIM,
        "state_layout": ["pos(3)", "vel(3)", "quat_wxyz(4)", "omega_body(3)"],
        "action_types": {
            "waypoint": {"dim": 4, "layout": ["dx_body", "dy_body", "dz_body", "speed"]},
            "ctbr":     {"dim": 4, "layout": ["thrust_norm", "wx", "wy", "wz"]},
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
    try:
        for ep in range(ep_start, ep_end):
            env = FlightmareExpertEnv(
                image_size=args.image_size,
                cameras=tuple(args.cameras),
                control_hz=args.control_hz,
                params=params,
                scene=args.scene,
                render=not args.no_render,
                seed=args.seed + ep,
                visual_backend=args.course_mode == "gates",
            )
            if env.using_fallback:
                print("[collect] WARNING: numpy fallback active - images will be blank.")
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
            env.close()
            env = None
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
        if env is not None:
            env.close()
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
                    "params", "lookahead_s", "course_mode", "seed",
                )
            }}, f)
        print(f"[collect][shard{args.shard_id}] wrote {len(manifest['episodes'])} eps -> {args.shard_manifest}", flush=True)
        return

    # Train/val split (episode-level, deterministic for given seed).
    n = len(manifest["episodes"])
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = int(round(args.val_frac * n))
    val_ids = set(int(i) for i in perm[:n_val])
    for i, ep in enumerate(manifest["episodes"]):
        ep["split"] = "val" if i in val_ids else "train"

    with open(out / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    train_files = [out / ep["path"] for ep in manifest["episodes"] if ep["split"] == "train"]
    if train_files:
        write_norm_stats(out, train_files)

    print(f"[collect] done. {n} episodes -> {out}")


if __name__ == "__main__":
    main()

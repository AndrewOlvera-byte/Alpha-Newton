"""Collect privileged-expert BC episodes in Flightmare.

Pipeline (per episode):
  1. Sample random waypoints in a configurable bounding box.
  2. Fit a minimum-jerk polynomial through them at a target avg speed.
  3. At each control tick, query the SE(3) geometric controller for a CTBR
     command, step Flightmare with it, render the requested cameras, and
     log every action label (waypoint+speed, CTBR, per-rotor motor).
  4. Save a per-episode HDF5 file under ``<out>/episodes/ep_NNNNNN.h5`` and
     update an ``index.json`` manifest with episode lengths + train/val split.
  5. After collection, scan the training split to write ``norm_stats.npz``
     (per-action-type mean/std + state mean/std) for downstream BC.

Run from repo root:

    python -m scripts.flightmare_bc.collect \\
        --out data/flightmare/expert_v1 \\
        --episodes 500 \\
        --image-size 224 \\
        --control-hz 100 \\
        --cameras forward \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.flightmare_bc.controllers import (
    GeometricSE3Controller,
    QuadParams,
    quat_to_R,
    world_to_body,
)
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from scripts.flightmare_bc.hdf5_writer import EpisodeWriter
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


def collect_one_episode(
    env: FlightmareExpertEnv,
    controller: GeometricSE3Controller,
    rng: np.random.Generator,
    cfg: argparse.Namespace,
    episode_id: int,
    writer: EpisodeWriter,
    lookahead_s: float,
) -> dict:
    waypoints = random_waypoint_path(
        rng,
        n_waypoints=cfg.n_waypoints,
        bbox=tuple(cfg.bbox),
        z_min=cfg.z_min,
        min_step=cfg.min_step,
    )
    avg_speed = float(rng.uniform(cfg.speed_range[0], cfg.speed_range[1]))
    traj = MinJerkTrajectory(waypoints, avg_speed=avg_speed, yaw_mode="tangent")

    init_pos = waypoints[0]
    obs = env.reset(init_pos=init_pos, yaw=0.0)

    n_steps = int(traj.total_time / env.dt)
    n_steps = max(1, min(n_steps, cfg.max_steps))

    pos_errors: list[float] = []
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
        motor_act = cmd["motor_normalized"]

        state_vec = build_state(obs.pos, obs.vel, obs.quat, obs.omega)
        writer.append(
            state=state_vec,
            images=obs.images,
            actions={"waypoint": waypoint_act, "ctbr": ctbr_act, "motor": motor_act},
            ref_pos=ref.pos.astype(np.float32),
            ref_vel=ref.vel.astype(np.float32),
            ref_yaw=float(ref.yaw),
            done=(k == n_steps - 1),
        )
        pos_errors.append(float(np.linalg.norm(obs.pos - ref.pos)))

        obs = env.step_ctbr(cmd["thrust_newton"], cmd["body_rates"])
        if not np.all(np.isfinite(obs.pos)) or np.linalg.norm(obs.pos) > 200.0:
            break

    return {
        "episode_id": episode_id,
        "length": n_steps,
        "avg_speed": avg_speed,
        "mean_track_err": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "max_track_err": float(np.max(pos_errors)) if pos_errors else 0.0,
        "n_waypoints": int(len(waypoints)),
    }


def write_norm_stats(out_dir: Path, train_files: list[Path]) -> None:
    """Per-channel mean/std for state and each action type, computed over train split."""
    import h5py

    sums = {"state": None, "waypoint": None, "ctbr": None, "motor": None}
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
            acc("motor", f["action/motor"][...])

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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-render", action="store_true",
                   help="Skip Unity rendering (state-only dataset, for debugging).")
    args = p.parse_args()

    out: Path = args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = QuadParams()
    controller = GeometricSE3Controller(params=params)
    env = FlightmareExpertEnv(
        image_size=args.image_size,
        cameras=tuple(args.cameras),
        control_hz=args.control_hz,
        params=params,
        scene=args.scene,
        render=not args.no_render,
    )
    if env.using_fallback:
        print("[collect] WARNING: numpy fallback active - images will be blank.")

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
            "motor":    {"dim": 4, "layout": ["m0", "m1", "m2", "m3"]},
        },
        "params": asdict(params),
        "lookahead_s": args.lookahead_s,
        "seed": args.seed,
        "episodes": [],
    }

    t0 = time.time()
    for ep in range(args.episodes):
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
        if (ep + 1) % 10 == 0 or ep == args.episodes - 1:
            elapsed = time.time() - t0
            print(f"[collect] {ep + 1}/{args.episodes}  "
                  f"len={info['length']} err={info['mean_track_err']:.2f}m "
                  f"v={info['avg_speed']:.1f}m/s  ({elapsed:.0f}s elapsed)")

    env.close()

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

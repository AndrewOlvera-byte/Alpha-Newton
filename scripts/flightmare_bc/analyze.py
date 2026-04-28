"""Visual + statistical analysis of a state-only Flightmare BC dataset.

No Unity / no GPU required. Reads ``index.json`` + per-episode HDF5s and
produces:

  * ``<out>/episode_<i>_traj.png``   3D trajectory + top/side views + gates
  * ``<out>/episode_<i>_signals.png`` speed, tracking error, CTBR, mission
  * ``<out>/aggregate.png``          per-episode summary distributions
  * stdout report: tracking error, gate completion, action ranges, NaN check

This is the visual sanity check that replaces the "watch Unity render" loop
for state-only collection: if the trajectories thread the gates and tracking
error stays bounded, the SE(3) controller + min-jerk reference are good and
the BC dataset is healthy to warm-start a policy.

Usage:
    python -m scripts.flightmare_bc.analyze --data-dir data/flightmare/expert_v1
    python -m scripts.flightmare_bc.analyze --data-dir <dir> --episodes 0 3 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def _gate_corners(pos: np.ndarray, yaw: float, size: float) -> np.ndarray:
    """4 corners of a gate square in world frame, plus the center, for plotting."""
    half = 0.5 * size
    # Gate square is in the plane perpendicular to the gate's forward axis.
    # forward = (cos yaw, sin yaw, 0); right = (-sin yaw, cos yaw, 0); up = z.
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    u = np.array([0.0, 0.0, 1.0])
    corners = np.stack([
        pos + half * r + half * u,
        pos - half * r + half * u,
        pos - half * r - half * u,
        pos + half * r - half * u,
        pos + half * r + half * u,  # close loop
    ])
    return corners, f


def _load_episode(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "state": f["obs/state"][...],
            "waypoint": f["action/waypoint"][...],
            "ctbr": f["action/ctbr"][...],
            "mission": f["mission/vec"][...] if "mission/vec" in f else None,
            "gate_index": f["mission/gate_index"][...],
            "ref_pos": f["reference/pos_des"][...],
            "ref_vel": f["reference/vel_des"][...],
            "dt": float(f.attrs["dt"]),
        }


def plot_trajectory(ep: dict, gates: list[dict], title: str, out_path: Path) -> None:
    pos = ep["state"][:, 0:3]
    ref = ep["ref_pos"]

    fig = plt.figure(figsize=(14, 5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)

    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], "C0-", lw=1.4, label="actual")
    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", lw=0.7, alpha=0.6, label="ref")
    ax_xy.plot(pos[:, 0], pos[:, 1], "C0-", lw=1.4)
    ax_xy.plot(ref[:, 0], ref[:, 1], "k--", lw=0.7, alpha=0.6)
    ax_xz.plot(pos[:, 0], pos[:, 2], "C0-", lw=1.4)
    ax_xz.plot(ref[:, 0], ref[:, 2], "k--", lw=0.7, alpha=0.6)

    for g in gates:
        gpos = np.asarray(g["pos"])
        size = float(g["size"][0]) if isinstance(g["size"], list) else float(g["size"])
        corners, fwd = _gate_corners(gpos, float(g["yaw"]), size)
        ax3d.plot(corners[:, 0], corners[:, 1], corners[:, 2], "C3-", lw=1.5)
        ax3d.scatter(*gpos, c="C3", s=20)
        ax_xy.plot(corners[:, 0], corners[:, 1], "C3-", lw=1.2)
        ax_xy.scatter(*gpos[:2], c="C3", s=20)
        ax_xy.arrow(gpos[0], gpos[1], 0.6 * fwd[0], 0.6 * fwd[1],
                    head_width=0.15, color="C3", alpha=0.6)
        ax_xz.plot(corners[:, 0], corners[:, 2], "C3-", lw=1.2)
        ax_xz.scatter(gpos[0], gpos[2], c="C3", s=20)

    ax3d.set_title("3D"); ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.legend(loc="upper left", fontsize=8)
    ax_xy.set_title("top-down (xy)"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal"); ax_xy.grid(alpha=0.3)
    ax_xz.set_title("side (xz)"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_aspect("equal"); ax_xz.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_signals(ep: dict, gates: list[dict], title: str, out_path: Path) -> None:
    pos = ep["state"][:, 0:3]
    vel = ep["state"][:, 3:6]
    ref = ep["ref_pos"]
    ref_vel = ep["ref_vel"]
    ctbr = ep["ctbr"]
    waypoint = ep["waypoint"]
    gate_idx = ep["gate_index"]
    dt = ep["dt"]
    t = np.arange(pos.shape[0]) * dt

    speed = np.linalg.norm(vel, axis=1)
    speed_ref = np.linalg.norm(ref_vel, axis=1)
    track_err = np.linalg.norm(pos - ref, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    ax = axes[0, 0]
    ax.plot(t, speed, "C0", label="actual"); ax.plot(t, speed_ref, "k--", label="ref")
    ax.set_ylabel("speed [m/s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("speed tracking")

    ax = axes[0, 1]
    ax.plot(t, track_err, "C2"); ax.set_ylabel("|pos - ref| [m]")
    ax.grid(alpha=0.3); ax.set_title("position tracking error")

    ax = axes[1, 0]
    ax.plot(t, ctbr[:, 0], label="thrust_norm")
    ax.plot(t, ctbr[:, 1], label="wx"); ax.plot(t, ctbr[:, 2], label="wy")
    ax.plot(t, ctbr[:, 3], label="wz")
    ax.set_ylabel("ctbr"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3); ax.set_title("CTBR action label")

    ax = axes[1, 1]
    ax.plot(t, waypoint[:, 0], label="dx_b")
    ax.plot(t, waypoint[:, 1], label="dy_b")
    ax.plot(t, waypoint[:, 2], label="dz_b")
    ax.plot(t, waypoint[:, 3], label="speed", color="C3")
    ax.set_ylabel("waypoint+speed"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax.set_title(f"waypoint label (gates passed: {int(gate_idx[-1])}/{len(gates)})")

    for a in axes.ravel():
        # shade gate-passing transitions
        for k in range(1, len(gate_idx)):
            if gate_idx[k] > gate_idx[k - 1]:
                a.axvline(t[k], color="C3", alpha=0.25, lw=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_aggregate(per_ep: list[dict], out_path: Path) -> None:
    track_means = np.array([e["mean_track_err"] for e in per_ep])
    track_max = np.array([e["max_track_err"] for e in per_ep])
    gates_passed = np.array([e["gates_passed"] for e in per_ep])
    gates_total = np.array([e["gates_total"] for e in per_ep])
    speeds = np.array([e["mean_speed"] for e in per_ep])
    lengths = np.array([e["length"] for e in per_ep])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist(track_means, bins=20); axes[0, 0].set_title("mean tracking err [m]")
    axes[0, 0].set_xlabel("m"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].hist(track_max, bins=20); axes[0, 1].set_title("max tracking err [m]")
    axes[0, 1].set_xlabel("m"); axes[0, 1].grid(alpha=0.3)
    completion = gates_passed / np.maximum(gates_total, 1)
    axes[0, 2].hist(completion, bins=np.linspace(0, 1, 21)); axes[0, 2].set_title("gate completion frac")
    axes[0, 2].grid(alpha=0.3)
    axes[1, 0].hist(speeds, bins=20); axes[1, 0].set_title("mean speed [m/s]")
    axes[1, 0].set_xlabel("m/s"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].hist(lengths, bins=20); axes[1, 1].set_title("episode length [steps]")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 2].scatter(speeds, track_means, s=12)
    axes[1, 2].set_xlabel("mean speed [m/s]"); axes[1, 2].set_ylabel("mean track err [m]")
    axes[1, 2].set_title("speed vs tracking err"); axes[1, 2].grid(alpha=0.3)
    fig.suptitle(f"aggregate over {len(per_ep)} episodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir for plots; defaults to <data-dir>/_analysis")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Episode indices to render in detail (default: first 3 + worst-tracking)")
    args = p.parse_args()

    data_dir: Path = args.data_dir
    out_dir = args.out or (data_dir / "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "index.json") as f:
        manifest = json.load(f)
    ep_meta = manifest["episodes"]
    if not ep_meta:
        raise SystemExit("manifest has no episodes")

    per_ep: list[dict] = []
    for em in ep_meta:
        h5_path = data_dir / em["path"]
        ep = _load_episode(h5_path)
        pos = ep["state"][:, 0:3]
        vel = ep["state"][:, 3:6]
        ref = ep["ref_pos"]
        track = np.linalg.norm(pos - ref, axis=1)
        gates = em.get("gates", [])
        gates_passed = int(ep["gate_index"][-1]) if ep["gate_index"].size else 0
        per_ep.append({
            "episode_id": em["episode_id"],
            "h5": h5_path,
            "ep": ep,
            "gates": gates,
            "length": pos.shape[0],
            "mean_track_err": float(track.mean()),
            "max_track_err": float(track.max()),
            "mean_speed": float(np.linalg.norm(vel, axis=1).mean()),
            "gates_passed": gates_passed,
            "gates_total": len(gates),
            "ctbr_min": ep["ctbr"].min(axis=0).tolist(),
            "ctbr_max": ep["ctbr"].max(axis=0).tolist(),
            "waypoint_speed_range": (float(ep["waypoint"][:, 3].min()),
                                     float(ep["waypoint"][:, 3].max())),
            "nan_state": int(np.isnan(ep["state"]).sum()),
            "nan_ctbr": int(np.isnan(ep["ctbr"]).sum()),
        })

    # Pick episodes to render in detail
    if args.episodes is not None:
        chosen = sorted(set(args.episodes))
    else:
        first = list(range(min(3, len(per_ep))))
        worst = int(np.argmax([e["max_track_err"] for e in per_ep]))
        chosen = sorted(set(first + [worst]))

    # === Stdout report ===
    print(f"=== dataset {data_dir} ===")
    print(f"  episodes        : {len(per_ep)}")
    print(f"  course_mode     : {manifest.get('course_mode', '?')}")
    print(f"  control_hz      : {manifest.get('control_hz', '?')}")
    print(f"  state_dim       : {manifest.get('state_dim', '?')}")
    print(f"  mission.dim     : {manifest.get('mission', {}).get('dim', '?')}")
    print()
    print(f"=== aggregate quality ===")
    track_means = np.array([e["mean_track_err"] for e in per_ep])
    track_max = np.array([e["max_track_err"] for e in per_ep])
    completion = np.array([e["gates_passed"] / max(1, e["gates_total"]) for e in per_ep])
    speeds = np.array([e["mean_speed"] for e in per_ep])
    lengths = np.array([e["length"] for e in per_ep])
    nan_total = sum(e["nan_state"] + e["nan_ctbr"] for e in per_ep)
    print(f"  mean track err  : mean={track_means.mean():.3f} m  p50={np.median(track_means):.3f}  p95={np.percentile(track_means, 95):.3f}  max={track_means.max():.3f}")
    print(f"  max  track err  : mean={track_max.mean():.3f} m   p95={np.percentile(track_max, 95):.3f}  max={track_max.max():.3f}")
    print(f"  gate completion : mean={completion.mean()*100:.1f}%  median={np.median(completion)*100:.1f}%  full-runs={int((completion >= 0.999).sum())}/{len(per_ep)}")
    print(f"  mean speed      : mean={speeds.mean():.2f} m/s  range=[{speeds.min():.2f}, {speeds.max():.2f}]")
    print(f"  episode length  : mean={lengths.mean():.0f} steps  range=[{int(lengths.min())}, {int(lengths.max())}]")
    print(f"  NaN/Inf samples : {nan_total}")
    print()
    print(f"=== per-episode (first 10) ===")
    for e in per_ep[:10]:
        print(f"  ep{e['episode_id']:04d}  T={e['length']:4d}  track(mean/max)={e['mean_track_err']:.2f}/{e['max_track_err']:.2f}m  "
              f"v={e['mean_speed']:.1f}m/s  gates={e['gates_passed']}/{e['gates_total']}  "
              f"ctbr_thrust=[{e['ctbr_min'][0]:.2f},{e['ctbr_max'][0]:.2f}]")

    # === Plots ===
    print()
    print(f"=== writing plots to {out_dir} ===")
    for idx in chosen:
        e = per_ep[idx]
        title = (f"ep {e['episode_id']}  T={e['length']}  "
                 f"track={e['mean_track_err']:.2f}/{e['max_track_err']:.2f}m  "
                 f"v={e['mean_speed']:.1f}m/s  "
                 f"gates={e['gates_passed']}/{e['gates_total']}")
        plot_trajectory(e["ep"], e["gates"], title,
                        out_dir / f"episode_{e['episode_id']:04d}_traj.png")
        plot_signals(e["ep"], e["gates"], title,
                     out_dir / f"episode_{e['episode_id']:04d}_signals.png")
        print(f"  wrote ep{e['episode_id']:04d} (traj + signals)")
    plot_aggregate(per_ep, out_dir / "aggregate.png")
    print(f"  wrote aggregate.png")

    # === Quality verdict ===
    print()
    verdict_ok = (
        nan_total == 0
        and track_means.mean() < 1.0
        and completion.mean() >= 0.7
    )
    print(f"=== verdict ===")
    print(f"  no NaN/Inf            : {nan_total == 0}")
    print(f"  mean track err < 1m   : {track_means.mean() < 1.0}  ({track_means.mean():.3f})")
    print(f"  gate completion >=70% : {completion.mean() >= 0.7}  ({completion.mean()*100:.1f}%)")
    print(f"  -> {'PASS - dataset healthy for BC warmup' if verdict_ok else 'REVIEW - tune speed/max_steps/spacing or controller gains'}")


if __name__ == "__main__":
    main()

"""Multi-process BC collection with one Unity instance per worker.

Spawns ``--workers`` subprocesses. Each:
  * gets its own ZMQ port pair via ``parallel_unity.allocate_ports``
  * launches a private Unity instance on those ports (headless by default)
  * runs ``scripts.flightmare_bc.collect`` on its episode shard
  * writes a per-worker manifest JSON

When all workers finish, this script merges the shard manifests into the
canonical ``index.json`` (with deterministic train/val split across the full
episode set) and computes ``norm_stats.npz`` from the union of train shards.

Per-episode HDF5 files are independent — there is no shared writer, so the
parallelism is naturally safe.

Prerequisites
-------------
1. ``unity_bridge.cpp`` patched + flightlib rebuilt (``patch_unity_bridge.py``).
2. ``patch_flightmare_pybind.py`` already applied (you have this).
3. The Unity binary at ``$FLIGHTMARE_UNITY_EXECUTABLE`` (or default path)
   accepts ``-input-port`` / ``-output-port`` flags. If your build uses
   different flag names, set ``FLIGHTMARE_UNITY_ARGS`` to inject them.

Usage:
    python -m scripts.flightmare_bc.parallel_collect \\
        --out data/flightmare/gates_v1 \\
        --episodes 600 \\
        --workers 6 \\
        --course-mode gates --num-gates 8 \\
        --image-size 224 --max-steps 2000 \\
        --speed-range 2.5 5.5 \\
        --scene warehouse --val-frac 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.flightmare_bc.parallel_unity import (
    DEFAULT_UNITY_BIN,
    allocate_ports,
)


PASSTHROUGH_FLAGS = {
    # name -> (cli flag, nargs)
    "image_size": ("--image-size", 1),
    "control_hz": ("--control-hz", 1),
    "max_steps": ("--max-steps", 1),
    "n_waypoints": ("--n-waypoints", 1),
    "z_min": ("--z-min", 1),
    "min_step": ("--min-step", 1),
    "lookahead_s": ("--lookahead-s", 1),
    "scene": ("--scene", 1),
    "course_mode": ("--course-mode", 1),
    "num_gates": ("--num-gates", 1),
    "gate_lateral_jitter": ("--gate-lateral-jitter", 1),
    "gate_yaw_step": ("--gate-yaw-step", 1),
    "gate_yaw_noise": ("--gate-yaw-noise", 1),
    "gate_size": ("--gate-size", 1),
    "unity_startup_s": ("--unity-startup-s", 1),
}
PASSTHROUGH_LIST = {
    "cameras": "--cameras",
    "bbox": "--bbox",
    "speed_range": "--speed-range",
    "gate_spacing_range": "--gate-spacing-range",
    "gate_z_range": "--gate-z-range",
}


def shard_ranges(n_episodes: int, n_workers: int) -> list[tuple[int, int]]:
    """Even shard split (last worker absorbs the remainder)."""
    base = n_episodes // n_workers
    out = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < n_episodes - base * n_workers else 0)
        out.append((start, start + size))
        start += size
    return out


def build_worker_cmd(args: argparse.Namespace, worker_id: int,
                     ep_start: int, ep_end: int, shard_manifest: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.flightmare_bc.collect",
           "--out", str(args.out),
           "--episodes", str(args.episodes),
           "--seed", str(args.seed + worker_id * 10_000),
           "--ep-start", str(ep_start),
           "--ep-end", str(ep_end),
           "--shard-manifest", str(shard_manifest),
           "--shard-id", str(worker_id),
           "--val-frac", "0.0"]
    for attr, (flag, _) in PASSTHROUGH_FLAGS.items():
        cmd.extend([flag, str(getattr(args, attr))])
    for attr, flag in PASSTHROUGH_LIST.items():
        cmd.append(flag)
        cmd.extend(str(v) for v in getattr(args, attr))
    return cmd


def merge_manifests(out_dir: Path, shard_manifests: list[Path], seed: int,
                    val_frac: float) -> dict:
    episodes: list[dict] = []
    header = None
    for sm in shard_manifests:
        with open(sm) as f:
            data = json.load(f)
        if header is None:
            header = data["header"]
        episodes.extend(data["episodes"])
    episodes.sort(key=lambda e: e["episode_id"])

    n = len(episodes)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_frac * n))
    val_ids = set(int(i) for i in perm[:n_val])
    for i, ep in enumerate(episodes):
        ep["split"] = "val" if i in val_ids else "train"

    manifest = dict(header or {})
    manifest["episodes"] = episodes
    with open(out_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_norm_stats_from_index(out_dir: Path, manifest: dict) -> None:
    """Same logic as collect.write_norm_stats but reads the merged manifest."""
    import h5py

    train = [out_dir / e["path"] for e in manifest["episodes"] if e["split"] == "train"]
    if not train:
        print("[parallel] no train episodes; skipping norm_stats")
        return
    sums: dict = {"state": None, "waypoint": None, "ctbr": None, "mission": None}
    sq_sums = {k: None for k in sums}
    counts = {k: 0 for k in sums}

    def acc(key, arr):
        if sums[key] is None:
            sums[key] = np.zeros(arr.shape[1], dtype=np.float64)
            sq_sums[key] = np.zeros(arr.shape[1], dtype=np.float64)
        sums[key] += arr.sum(axis=0)
        sq_sums[key] += (arr.astype(np.float64) ** 2).sum(axis=0)
        counts[key] += arr.shape[0]

    for fp in train:
        with h5py.File(fp, "r") as f:
            acc("state", f["obs/state"][...])
            acc("waypoint", f["action/waypoint"][...])
            acc("ctbr", f["action/ctbr"][...])
            if "mission/vec" in f:
                acc("mission", f["mission/vec"][...])

    out: dict = {}
    for k in sums:
        if counts[k] == 0:
            continue
        mean = sums[k] / counts[k]
        var = sq_sums[k] / counts[k] - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))
        out[f"{k}_mean"] = mean.astype(np.float32)
        out[f"{k}_std"] = std.astype(np.float32)
    np.savez(out_dir / "norm_stats.npz", **out)
    print(f"[parallel] wrote norm_stats.npz ({len(train)} train files)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--unity-bin", type=str, default=DEFAULT_UNITY_BIN)
    p.add_argument("--unity-startup-s", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-render", action="store_true",
                   help="Run all workers state-only (sanity check; no Unity needed).")
    p.add_argument("--keep-shard-manifests", action="store_true")
    p.add_argument("--headless-unity", action="store_true",
                   help="Force Unity into -batchmode -nographics even with workers=1.")
    # collect.py passthrough
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cameras", nargs="+", default=["forward"])
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--n-waypoints", type=int, default=6)
    p.add_argument("--bbox", nargs=3, type=float, default=[8.0, 8.0, 3.5])
    p.add_argument("--z-min", type=float, default=1.0)
    p.add_argument("--min-step", type=float, default=1.5)
    p.add_argument("--speed-range", nargs=2, type=float, default=[2.5, 5.5])
    p.add_argument("--lookahead-s", type=float, default=0.3)
    p.add_argument("--scene", type=str, default="warehouse")
    p.add_argument("--course-mode", choices=["random", "gates"], default="gates")
    p.add_argument("--num-gates", type=int, default=8)
    p.add_argument("--gate-spacing-range", nargs=2, type=float, default=[4.0, 8.0])
    p.add_argument("--gate-lateral-jitter", type=float, default=2.0)
    p.add_argument("--gate-z-range", nargs=2, type=float, default=[1.5, 4.0])
    p.add_argument("--gate-yaw-step", type=float, default=0.5)
    p.add_argument("--gate-yaw-noise", type=float, default=0.25)
    p.add_argument("--gate-size", type=float, default=1.0)
    args = p.parse_args()

    out: Path = args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    shard_dir = out / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ranges = shard_ranges(args.episodes, args.workers)
    print(f"[parallel] {args.workers} workers, episodes {args.episodes}, shards={ranges}",
          flush=True)

    procs: list[subprocess.Popen] = []
    shard_manifests: list[Path] = []
    t0 = time.time()
    for wid, (s, e) in enumerate(ranges):
        if e <= s:
            continue
        sm = shard_dir / f"shard_{wid:02d}.json"
        shard_manifests.append(sm)
        cmd = build_worker_cmd(args, wid, s, e, sm)
        if args.no_render:
            cmd.append("--no-render")
        else:
            cmd.append("--render")
            cmd.append("--launch-unity")
            cmd.extend(["--unity-startup-s", str(args.unity_startup_s)])

        worker_env = dict(os.environ)
        if not args.no_render:
            pub, sub = allocate_ports(wid)
            worker_env["FLIGHTMARE_PUB_PORT"] = str(pub)
            worker_env["FLIGHTMARE_SUB_PORT"] = str(sub)
            worker_env["FLIGHTMARE_UNITY_EXECUTABLE"] = args.unity_bin
            # Pass port flags to the Unity binary that collect.py will spawn.
            unity_extra = f"-input-port {pub} -output-port {sub}"
            if args.workers > 1 or args.headless_unity:
                unity_extra += " -batchmode -nographics"
            worker_env["FLIGHTMARE_UNITY_ARGS"] = unity_extra
            print(f"[parallel] worker {wid}: ports pub={pub} sub={sub} eps=[{s},{e})",
                  flush=True)

        log = open(log_dir / f"worker_{wid:02d}.log", "w")
        procs.append(subprocess.Popen(cmd, env=worker_env, stdout=log,
                                      stderr=subprocess.STDOUT))

    rcs = [p.wait() for p in procs]
    elapsed = time.time() - t0
    print(f"[parallel] all workers exited (rc={rcs}) in {elapsed:.0f}s", flush=True)
    if any(rc != 0 for rc in rcs):
        print("[parallel] one or more workers failed; see logs in", log_dir)
        sys.exit(1)

    print(f"[parallel] merging {len(shard_manifests)} shard manifests")
    manifest = merge_manifests(out, shard_manifests, seed=args.seed,
                               val_frac=args.val_frac)
    write_norm_stats_from_index(out, manifest)
    if not args.keep_shard_manifests:
        for sm in shard_manifests:
            sm.unlink(missing_ok=True)
        try:
            shard_dir.rmdir()
        except OSError:
            pass
    print(f"[parallel] done. {len(manifest['episodes'])} episodes -> {out}")


if __name__ == "__main__":
    main()

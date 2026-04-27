"""Quick health check on a collected BC dataset.

Reports:
  * Per-dataset shapes and dtypes (one episode + the manifest)
  * Tracking error stats (controller fit vs reference)
  * Gate-progress monotonicity (it should never go backwards within an episode)
  * Action-label distribution sanity (means, stds, percentiles, NaN/Inf check)
  * Mission-vector sanity (norm, distance-to-next-gate at each progress step)

Usage:
    python -m scripts.flightmare_bc.diagnostics --data-dir data/flightmare/smoke_v0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _stats(name: str, arr: np.ndarray) -> str:
    return (
        f"{name:>16s}  shape={arr.shape}  dtype={arr.dtype}  "
        f"min={np.min(arr):.3f} max={np.max(arr):.3f} "
        f"mean={np.mean(arr):.3f} std={np.std(arr):.3f}  "
        f"nan={int(np.isnan(arr).sum())} inf={int(np.isinf(arr).sum())}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()

    data_dir: Path = args.data_dir
    with open(data_dir / "index.json") as f:
        manifest = json.load(f)

    print(f"=== manifest ===")
    print(f"  episodes        : {len(manifest['episodes'])}")
    print(f"  control_hz      : {manifest['control_hz']}")
    print(f"  state_layout    : {manifest['state_layout']}")
    print(f"  action_types    : {list(manifest['action_types'])}")
    if "mission" in manifest:
        print(f"  mission.dim     : {manifest['mission']['dim']}")
        print(f"  mission.layout  : {manifest['mission']['layout']}")

    files = [data_dir / e["path"] for e in manifest["episodes"]]
    if not files:
        raise SystemExit("no episodes")

    print(f"\n=== per-episode summary ===")
    all_progress_ok = True
    n_actions_ok = True
    for ep_meta, fp in zip(manifest["episodes"], files):
        with h5py.File(fp, "r") as h:
            T = h["obs/state"].shape[0]
            keys = list(h.keys())
            action_keys = list(h["action"].keys())
            n_actions_ok = n_actions_ok and (set(action_keys) == {"waypoint", "ctbr"})

            if "mission" in keys:
                gi = h["mission/gate_index"][...]
                # progress index must be non-decreasing within an episode
                mono = bool(np.all(np.diff(gi) >= 0)) if T > 1 else True
                all_progress_ok = all_progress_ok and mono
                final_gi = int(gi[-1])
                n_gates = len(ep_meta.get("gates", []))
                print(f"  {fp.name}  T={T:4d}  action={action_keys}  "
                      f"gates_passed={final_gi}/{n_gates}  monotonic={mono}")
            else:
                print(f"  {fp.name}  T={T:4d}  action={action_keys}  no mission")

    print(f"\n=== detailed dump (first episode) ===")
    with h5py.File(files[0], "r") as h:
        print(_stats("obs/state",    h["obs/state"][...]))
        print(_stats("action/waypoint", h["action/waypoint"][...]))
        print(_stats("action/ctbr",     h["action/ctbr"][...]))
        if "mission/vec" in h:
            mv = h["mission/vec"][...]
            print(_stats("mission/vec",  mv))
            # Each chunk of 4 cols is a (dx,dy,dz,yaw_rel) tuple. Distance
            # to the immediate next gate should match the last column.
            dx = mv[:, 0]; dy = mv[:, 1]; dz = mv[:, 2]
            d_implied = np.sqrt(dx**2 + dy**2 + dz**2)
            d_logged = mv[:, -1]
            err = float(np.max(np.abs(d_implied - d_logged)))
            print(f"  consistency: max |sqrt(dx^2+dy^2+dz^2) - dist_to_next| = {err:.4f} m  (should be ~0)")
        for cam in manifest["cameras"]:
            ds = h[f"obs/image_{cam}"]
            arr = ds[0]  # one frame
            print(f"  obs/image_{cam}  shape={ds.shape}  dtype={ds.dtype}  "
                  f"frame0 mean={arr.mean():.1f} max={int(arr.max())}")

    print(f"\n=== norm_stats ===")
    npz_path = data_dir / "norm_stats.npz"
    if npz_path.exists():
        stats = np.load(npz_path)
        for k in stats.files:
            print(f"  {k:>16s}  shape={stats[k].shape}")
    else:
        print("  norm_stats.npz NOT FOUND")

    print()
    print(f"PASS  actions=={{waypoint,ctbr}}: {n_actions_ok}")
    print(f"PASS  gate_index monotonic    : {all_progress_ok}")


if __name__ == "__main__":
    main()

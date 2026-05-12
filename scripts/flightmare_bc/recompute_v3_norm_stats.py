"""Recompute obs-v3 norm stats for an existing Flightmare BC dataset.

``transform_to_v3`` writes the v3 HDF5 overlays and can aggregate stats over
every processed episode. For BC training we want normalization from the train
split only, matching the action/state stats written by collection.

Example:
  python -m scripts.flightmare_bc.recompute_v3_norm_stats \
      --data-dir data/flightmare/bc_v4 --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


_V3_KEYS = {
    "proprio_core": "obs/proprio_core",
    "gate": "obs/gate",
    "aux": "obs/aux",
}


def _accumulate(paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    sums: dict[str, np.ndarray] = {}
    sq_sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for path in paths:
        with h5py.File(path, "r") as h:
            for name, key in _V3_KEYS.items():
                if key not in h:
                    raise RuntimeError(f"{path} missing {key}; rerun transform_to_v3 first.")
                arr = h[key][...].astype(np.float64, copy=False)
                if name not in sums:
                    sums[name] = np.zeros(arr.shape[1], dtype=np.float64)
                    sq_sums[name] = np.zeros(arr.shape[1], dtype=np.float64)
                    counts[name] = 0
                sums[name] += arr.sum(axis=0)
                sq_sums[name] += np.square(arr).sum(axis=0)
                counts[name] += int(arr.shape[0])

    return {name: (sums[name], sq_sums[name], counts[name]) for name in sums}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/flightmare/bc_v4"))
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()

    index_path = args.data_dir / "index.json"
    stats_path = args.data_dir / "norm_stats.npz"
    manifest = json.loads(index_path.read_text())
    episodes = [
        ep for ep in manifest.get("episodes", [])
        if ep.get("split", "train") == args.split
    ]
    if not episodes:
        raise RuntimeError(f"No episodes with split={args.split!r} in {index_path}.")
    paths = [args.data_dir / ep["path"] for ep in episodes]
    acc = _accumulate(paths)

    existing = {}
    if stats_path.exists():
        with np.load(stats_path) as s:
            existing = {k: s[k] for k in s.files}

    for name, (sum_, sq_sum, count) in acc.items():
        mean = sum_ / max(1, count)
        var = sq_sum / max(1, count) - np.square(mean)
        std = np.sqrt(np.clip(var, 1e-8, None)) + float(args.eps)
        key = "proprio_core" if name == "proprio_core" else name
        existing[f"{key}_mean"] = mean.astype(np.float32)
        existing[f"{key}_std"] = std.astype(np.float32)

    np.savez(stats_path, **existing)
    manifest["obs_v3_stats_split"] = args.split
    index_path.write_text(json.dumps(manifest, indent=2))

    print(f"[v3-stats] split={args.split} episodes={len(paths)} -> {stats_path}")
    for key in ("proprio_core", "gate", "aux"):
        print(
            f"  {key:13s} mean={existing[f'{key}_mean'].shape} "
            f"std={existing[f'{key}_std'].shape}"
        )


if __name__ == "__main__":
    main()

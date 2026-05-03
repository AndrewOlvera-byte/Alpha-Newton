"""Recompute norm_stats.npz for an existing Flightmare BC dataset.

Use this when an already-collected dataset has wrong action bounds (e.g. the
old [-50, 50] waypoint defaults that crushed bounds-mode normalization into a
[-0.02, 0.05] range and made BC GNLL trivially negative) or when you want to
trim a controller/sim warmup transient out of the train statistics without
recollecting.

The script also patches ``index.json`` so that
``FlightmareBCStateDataset`` skips the same transient at training time.

Example::

    python -m scripts.flightmare_bc.recompute_norm_stats \\
        --data-dir data/flightmare/bc_v3_swift_like \\
        --action-normalization bounds \\
        --skip-initial-frames 12 \\
        --bounds-margin 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.flightmare_bc.collect import write_norm_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--action-normalization", choices=["standard", "bounds"], default="bounds")
    p.add_argument("--skip-initial-frames", type=int, default=12,
                   help="Drop the first N samples per episode from norm-stats AND from the BC index.")
    p.add_argument("--bounds-margin", type=float, default=0.10)
    args = p.parse_args()

    data_dir: Path = args.data_dir
    index_path = data_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    with open(index_path) as f:
        manifest = json.load(f)

    train_files = [
        data_dir / ep["path"]
        for ep in manifest["episodes"]
        if ep.get("split", "train") == "train"
    ]
    if not train_files:
        raise RuntimeError(f"No train-split episodes found in {index_path}.")

    print(f"[recompute] data_dir={data_dir}")
    print(f"[recompute] train episodes: {len(train_files)}")
    print(f"[recompute] action_normalization={args.action_normalization}  "
          f"skip_initial_frames={args.skip_initial_frames}  bounds_margin={args.bounds_margin}")

    write_norm_stats(
        data_dir,
        train_files,
        action_normalization=args.action_normalization,
        skip_initial_frames=int(args.skip_initial_frames),
        bounds_margin=float(args.bounds_margin),
    )

    manifest["action_normalization"] = args.action_normalization
    manifest["skip_initial_frames"] = int(args.skip_initial_frames)
    manifest["bounds_margin"] = float(args.bounds_margin)
    with open(index_path, "w") as f:
        json.dump(manifest, f, indent=2)

    import numpy as np
    s = np.load(data_dir / "norm_stats.npz")
    print(f"[recompute] wrote {data_dir / 'norm_stats.npz'}")
    for at in ("waypoint", "ctbr", "motor"):
        lo = s[f"{at}_low"]; hi = s[f"{at}_high"]
        print(f"  {at}_low ={lo}")
        print(f"  {at}_high={hi}")


if __name__ == "__main__":
    main()

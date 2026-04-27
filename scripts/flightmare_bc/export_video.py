"""Export a Flightmare BC episode's stored camera frames to a video file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio


def _episode_path(data_dir: Path, episode_index: int) -> Path:
    with open(data_dir / "index.json") as f:
        manifest = json.load(f)
    return data_dir / manifest["episodes"][episode_index]["path"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, help="Dataset directory containing index.json.")
    p.add_argument("--episode", type=Path, help="Direct path to ep_*.h5.")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--camera", default="forward")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    if args.episode is None:
        if args.data_dir is None:
            raise SystemExit("Provide either --episode or --data-dir.")
        episode = _episode_path(args.data_dir, args.episode_index)
    else:
        episode = args.episode

    ds_key = f"obs/image_{args.camera}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(episode, "r") as f:
        if ds_key not in f:
            raise KeyError(f"{ds_key!r} not found in {episode}")
        frames = [frame for frame in f[ds_key][:: max(1, args.stride)]]

    imageio.mimwrite(args.out, frames, fps=args.fps)
    print(f"[export_video] wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download robomimic Proficient Human (PH) datasets and rebuild images.

Usage:
    # Download low-dim only
    python scripts/robomimic_download.py --tasks lift

    # Download low-dim + render images from sim states
    python scripts/robomimic_download.py --tasks lift --postprocess

    # All three tasks with images
    python scripts/robomimic_download.py --tasks lift can square --postprocess

Output structure:
    data/robomimic/
    ├── lift/ph/
    │   ├── low_dim.hdf5
    │   └── image.hdf5
    ├── can/ph/
    │   └── ...
    └── square/ph/
        └── ...
"""
import argparse
import os
import subprocess
import sys
import urllib.request

# Robomimic dataset URLs for PH (proficient human) variants
# These are the official URLs from the robomimic project
#
# Compatibility notes:
#   lift / can / square: single-arm, action_dim=7, obs_dim=19 — compatible with
#                        bc_multitask_ph.yaml (n_tasks=3).
#   transport:           bimanual (2 robots), action_dim=14, obs_dim varies.
#                        Download and render as normal; training config requires
#                        separate arch settings (action_dim=14, different obs_keys).
#   tool_hang:           single-arm but much larger obs_dim — same caveat.
DATASET_URLS = {
    "lift": {
        "low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/low_dim_v141.hdf5",
    },
    "can": {
        "low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim_v141.hdf5",
    },
    "square": {
        "low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim_v141.hdf5",
    },
    "transport": {
        "low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/ph/low_dim_v141.hdf5",
    },
    "tool_hang": {
        "low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang/ph/low_dim_v141.hdf5",
    },
}


def download_file(url: str, dest: str):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  [skip] Already exists: {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  Dest: {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress
    except Exception as e:
        # Clean up partial download
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Download failed: {e}")


def postprocess_images(
    hdf5_path: str,
    camera_names: list = None,
    camera_height: int = 84,
    camera_width: int = 84,
):
    """Rebuild images from low-dim dataset by replaying sim states through robosuite.

    The low_dim.hdf5 contains simulator states for each timestep. This function
    uses robomimic's dataset_states_to_obs.py to replay those states and render
    camera images, producing an image.hdf5 in the same directory.

    Requires: robomimic, robosuite
    """
    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]

    output_path = os.path.join(os.path.dirname(hdf5_path), "image.hdf5")
    if os.path.exists(output_path):
        print(f"  [skip] Image dataset already exists: {output_path}")
        return output_path

    print(f"  Rendering images from sim states: {hdf5_path}")
    print(f"  Cameras: {camera_names}, resolution: {camera_width}x{camera_height}")

    import robomimic
    script_path = os.path.join(
        os.path.dirname(robomimic.__file__),
        "scripts",
        "dataset_states_to_obs.py",
    )

    cmd = [
        sys.executable, script_path,
        "--dataset", hdf5_path,
        "--output_name", "image.hdf5",
        "--done_mode", "0",
        "--camera_names", *camera_names,
        "--camera_height", str(camera_height),
        "--camera_width", str(camera_width),
        "--compress",
        "--exclude-next-obs",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Image dataset written to: {output_path}")
    return output_path


def verify_dataset(hdf5_path: str):
    """Quick sanity check on a downloaded HDF5 file."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"Invalid HDF5: no 'data' group in {hdf5_path}")
        demos = sorted(f["data"].keys())
        n_demos = len(demos)
        first = f[f"data/{demos[0]}"]
        T = first["actions"].shape[0]
        action_dim = first["actions"].shape[1]
        obs_keys = list(first["obs"].keys()) if "obs" in first else []
        print(f"  Verified: {n_demos} demos, first demo T={T}, action_dim={action_dim}")
        print(f"  Obs keys: {obs_keys}")


def main():
    parser = argparse.ArgumentParser(description="Download robomimic PH datasets")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["lift", "can", "square", "transport", "tool_hang"],
        default=["lift"],
        help="Tasks to download (default: lift). "
             "For bc_multitask_ph training use: lift can square. "
             "transport and tool_hang have different action/obs dims and need separate configs.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/robomimic",
        help="Output directory (default: data/robomimic)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Run robomimic image postprocessing to generate images from states (requires robomimic+robosuite)",
    )
    args = parser.parse_args()

    print(f"Robomimic PH Dataset Download")
    print(f"Tasks: {args.tasks}")
    print(f"Output: {args.data_dir}")
    print()

    for task in args.tasks:
        print(f"=== {task.upper()} ===")
        urls = DATASET_URLS[task]

        low_dim_dest = os.path.join(args.data_dir, task, "ph", "low_dim.hdf5")
        download_file(urls["low_dim"], low_dim_dest)
        verify_dataset(low_dim_dest)

        if args.postprocess:
            postprocess_images(low_dim_dest)

        print()

    print("Done! Datasets ready at:", args.data_dir)


if __name__ == "__main__":
    main()

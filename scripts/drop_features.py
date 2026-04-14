"""Rewrite each robomimic image.hdf5 with *_features datasets stripped.

Preserves everything else verbatim — raw images, low-dim obs, actions,
dones, rewards, states, mask group, norm_stats group, env_args attrs.
Writes to image.hdf5.tmp first, verifies, then atomically swaps.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import h5py


TASKS = ["lift", "can", "square", "tool_hang"]  # ascending by size


def human(nbytes: float) -> str:
    return f"{nbytes / 1e9:.2f} GB"


def rewrite_without_features(src_path: str) -> str:
    tmp_path = src_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        # Root attrs.
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Top-level groups/datasets except 'data' (which needs per-demo filter).
        for key in src.keys():
            if key == "data":
                continue
            src.copy(key, dst)

        # 'data' group — copy attrs then iterate demos.
        data_dst = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data_dst.attrs[k] = v

        demos = list(src["data"].keys())
        total = len(demos)
        t0 = time.time()
        for i, demo in enumerate(demos):
            demo_src = src[f"data/{demo}"]
            demo_dst = data_dst.create_group(demo)
            for ak, av in demo_src.attrs.items():
                demo_dst.attrs[ak] = av

            for key in demo_src.keys():
                if key == "obs":
                    obs_dst = demo_dst.create_group("obs")
                    for ak, av in demo_src["obs"].attrs.items():
                        obs_dst.attrs[ak] = av
                    for obs_key in demo_src["obs"].keys():
                        if obs_key.endswith("_features"):
                            continue
                        demo_src["obs"].copy(obs_key, obs_dst)
                else:
                    demo_src.copy(key, demo_dst)

            if (i + 1) % 25 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - (i + 1)) / rate if rate > 0 else 0
                print(
                    f"  demo {i+1}/{total}  elapsed={elapsed:5.1f}s  "
                    f"rate={rate:.2f}/s  eta={eta:5.1f}s",
                    flush=True,
                )

    return tmp_path


def verify(tmp_path: str, src_path: str) -> None:
    """Sanity-check the new file: same demo count, same non-feature obs keys,
    same demo lengths, and no *_features datasets anywhere."""
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "r") as dst:
        src_demos = set(src["data"].keys())
        dst_demos = set(dst["data"].keys())
        assert src_demos == dst_demos, f"demo set differs: missing {src_demos - dst_demos}"

        # Compare the first demo in detail + a random later one.
        for demo in list(src_demos)[:1] + list(src_demos)[-1:]:
            sobs = src[f"data/{demo}/obs"]
            dobs = dst[f"data/{demo}/obs"]

            expected = {k for k in sobs.keys() if not k.endswith("_features")}
            got = set(dobs.keys())
            assert expected == got, f"{demo} obs keys differ: {expected ^ got}"

            # Random length check on one non-feature key.
            probe = next(iter(expected))
            assert sobs[probe].shape == dobs[probe].shape, (
                f"{demo}/{probe} shape mismatch"
            )

            # Assert actions/dones/rewards lengths match.
            for k in ("actions", "dones", "rewards", "states"):
                if k in src[f"data/{demo}"]:
                    assert src[f"data/{demo}/{k}"].shape == dst[f"data/{demo}/{k}"].shape

            # Confirm no features slipped through.
            assert not any(k.endswith("_features") for k in dobs.keys())

        # Top-level groups preserved.
        for key in ("mask", "norm_stats"):
            if key in src:
                assert key in dst, f"top-level group {key!r} missing from rewrite"


def main():
    for task in TASKS:
        src = f"data/robomimic/{task}/ph/image.hdf5"
        if not os.path.exists(src):
            print(f"[{task}] skip — {src} not found")
            continue

        orig = os.path.getsize(src)
        print(f"\n[{task}] rewriting {src}")
        print(f"  original: {human(orig)}")

        t0 = time.time()
        tmp = rewrite_without_features(src)
        rewrite_time = time.time() - t0
        new = os.path.getsize(tmp)
        print(f"  new:      {human(new)}  (rewrite took {rewrite_time:.1f}s)")
        print(f"  saved:    {human(orig - new)}")

        print(f"  verifying...")
        verify(tmp, src)
        os.replace(tmp, src)
        print(f"  [{task}] swapped in ✓")

    print("\nAll done.")


if __name__ == "__main__":
    main()

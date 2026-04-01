#!/usr/bin/env python3
"""
Render camera images from robomimic low_dim.hdf5 sim states using robosuite 1.5+.

Bypasses robomimic's dataset_states_to_obs.py which requires the old mujoco_py.
Directly replays states through robosuite and captures camera observations.

Usage:
    python scripts/render_images.py --dataset data/robomimic/lift/ph/low_dim.hdf5
    python scripts/render_images.py --dataset data/robomimic/lift/ph/low_dim.hdf5 --cameras agentview robot0_eye_in_hand --size 84
"""
import argparse
import json
import os

import h5py
import numpy as np


def create_env(env_meta, camera_names, camera_size):
    """Create a robosuite env from stored metadata, with offscreen rendering.

    Handles the v1.4 → v1.5 controller config migration:
    v1.4 stored controller_configs as a flat dict with type="OSC_POSE",
    v1.5 expects composite controller structure.
    """
    import robosuite as suite

    env_kwargs = env_meta.get("env_kwargs", {})

    # Remove v1.4-style controller_configs — let robosuite 1.5 use its defaults
    env_kwargs.pop("controller_configs", None)

    # Enable offscreen rendering for image capture
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = camera_names
    env_kwargs["camera_heights"] = camera_size
    env_kwargs["camera_widths"] = camera_size
    env_kwargs["camera_depths"] = False
    env_kwargs["ignore_done"] = True

    env_name = env_meta["env_name"]
    env = suite.make(env_name, **env_kwargs)
    return env


def render_dataset(dataset_path, output_name, camera_names, camera_size, n_demos=None, compress=True):
    """Replay sim states and render camera images into a new HDF5."""
    output_path = os.path.join(os.path.dirname(dataset_path), output_name)
    if os.path.exists(output_path):
        print(f"[skip] Output already exists: {output_path}")
        return output_path

    f_in = h5py.File(dataset_path, "r")
    env_meta = json.loads(f_in["data"].attrs["env_args"])

    print(f"Env: {env_meta['env_name']} (v{env_meta.get('env_version', '?')})")
    print(f"Cameras: {camera_names}, size: {camera_size}x{camera_size}")

    env = create_env(env_meta, camera_names, camera_size)

    # Sort demo keys numerically
    demos = sorted(f_in["data"].keys(), key=lambda x: int(x.split("_")[1]))
    if n_demos is not None:
        demos = demos[:n_demos]

    print(f"Processing {len(demos)} demos...")

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    total_samples = 0

    for i, ep in enumerate(demos):
        ep_grp = f_in[f"data/{ep}"]
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]
        T = states.shape[0]

        # Reset env and load initial state
        env.reset()
        # Set the initial sim state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # Collect observations for each timestep
        all_obs = {cam: [] for cam in camera_names}
        low_dim_obs = {}

        for t in range(T):
            # Set sim state
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()

            # Get observation dict from env
            obs = env._get_observations()

            # Capture camera images
            for cam in camera_names:
                img_key = f"{cam}_image"
                if img_key in obs:
                    all_obs[cam].append(obs[img_key])
                else:
                    # Try direct rendering
                    img = env.sim.render(
                        camera_name=cam,
                        height=camera_size,
                        width=camera_size,
                    )
                    all_obs[cam].append(img[::-1])  # flip vertically (mujoco convention)

            # Capture low-dim obs on first pass to know what keys exist
            if t == 0:
                for k, v in obs.items():
                    if isinstance(v, np.ndarray) and v.ndim == 1:
                        low_dim_obs[k] = [v]
                    elif isinstance(v, (int, float)):
                        low_dim_obs[k] = [np.array([v])]
            else:
                for k in low_dim_obs:
                    if k in obs:
                        v = obs[k]
                        if isinstance(v, np.ndarray) and v.ndim == 1:
                            low_dim_obs[k].append(v)
                        elif isinstance(v, (int, float)):
                            low_dim_obs[k].append(np.array([v]))

        # Write to output HDF5
        ep_out = data_grp.create_group(ep)
        ep_out.create_dataset("actions", data=actions)
        ep_out.create_dataset("states", data=states)

        # Copy rewards/dones from source
        if "rewards" in ep_grp:
            ep_out.create_dataset("rewards", data=ep_grp["rewards"][()])
        if "dones" in ep_grp:
            ep_out.create_dataset("dones", data=ep_grp["dones"][()])

        # Write image obs
        for cam in camera_names:
            img_data = np.array(all_obs[cam], dtype=np.uint8)  # (T, H, W, C)
            key = f"obs/{cam}_image"
            if compress:
                ep_out.create_dataset(key, data=img_data, compression="gzip")
            else:
                ep_out.create_dataset(key, data=img_data)

        # Write low-dim obs
        for k, v_list in low_dim_obs.items():
            if k.endswith("_image") or k.endswith("_depth"):
                continue
            try:
                arr = np.array(v_list)
                ep_out.create_dataset(f"obs/{k}", data=arr)
            except Exception:
                pass

        # Copy model_file attr if present
        if "model_file" in ep_grp.attrs:
            ep_out.attrs["model_file"] = ep_grp.attrs["model_file"]
        ep_out.attrs["num_samples"] = T
        total_samples += T

        if (i + 1) % 20 == 0 or i == len(demos) - 1:
            print(f"  [{i+1}/{len(demos)}] {ep}: T={T}")

    # Copy mask group
    if "mask" in f_in:
        f_in.copy("mask", f_out)

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = f_in["data"].attrs["env_args"]

    f_in.close()
    f_out.close()
    env.close()

    print(f"Wrote {len(demos)} demos ({total_samples} frames) to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render images from robomimic sim states")
    parser.add_argument("--dataset", type=str, required=True, help="Path to low_dim.hdf5")
    parser.add_argument("--output-name", type=str, default="image.hdf5", help="Output filename")
    parser.add_argument("--cameras", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    parser.add_argument("--size", type=int, default=84, help="Image resolution")
    parser.add_argument("--n", type=int, default=None, help="Process only first N demos")
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()

    render_dataset(
        dataset_path=args.dataset,
        output_name=args.output_name,
        camera_names=args.cameras,
        camera_size=args.size,
        n_demos=args.n,
        compress=not args.no_compress,
    )


if __name__ == "__main__":
    main()

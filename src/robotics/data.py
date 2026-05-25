"""Robotics imitation-learning datasets.

Two families live in this file:

* ``robomimic_*``: per-step / per-window datasets backed by robomimic HDF5
  files (vision + low-dim state + actions, multi-task).

* ``flightmare_bc_state``: per-step state-only dataset for Flightmare drone
  BC. Inputs are the drone state (13-dim) concatenated with the privileged
  mission vector (next-K-gates body-frame + progress + dist; 14-dim) into a
  single 27-dim feature, so the existing
  ``MLPFusionGaussianExpertActor(use_vision=False)`` can be used without
  any model-side plumbing changes. Targets are either the controller's CTBR
  output or the reference-derived waypoint+speed label.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.registry import register
from scripts.flightmare_bc.action_norms import action_bounds, stats_mode


class RobomimicDataset(Dataset):
    """Torch Dataset over robomimic HDF5 demos.

    Each sample is a dict:
        obs:     dict of tensors (low_dim: [seq_len, obs_dim], images: [seq_len, C, H, W])
        actions: [seq_length, action_dim]
    """

    def __init__(
        self,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys_low_dim: List[str],
        obs_keys_image: List[str],
        action_keys: List[str],
        seq_length: int = 10,
        frame_stack: int = 1,
        pad_frame_stack: bool = True,
        pad_seq_length: bool = True,
        image_size: int = 84,
        hdf5_normalize_obs: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.demo_keys = sorted(demo_keys)
        self.obs_keys_low_dim = obs_keys_low_dim
        self.obs_keys_image = obs_keys_image
        self.action_keys = action_keys
        self.seq_length = seq_length
        self.frame_stack = frame_stack
        self.pad_frame_stack = pad_frame_stack
        self.pad_seq_length = pad_seq_length
        self.image_size = image_size
        self.hdf5_normalize_obs = hdf5_normalize_obs

        self._index = []
        self._demo_lengths = {}

        with h5py.File(self.hdf5_path, "r") as f:
            for dk in self.demo_keys:
                demo_grp = f[f"data/{dk}"]
                T = demo_grp["actions"].shape[0]
                self._demo_lengths[dk] = T
                for t in range(T):
                    self._index.append((dk, t))

        self._hdf5_file = None

    def __len__(self):
        return len(self._index)

    def _open(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")

    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._open()
        demo_key, start_t = self._index[idx]
        demo_grp = self._hdf5_file[f"data/{demo_key}"]
        T = self._demo_lengths[demo_key]

        end_t = min(start_t + self.seq_length, T)
        actual_len = end_t - start_t

        actions = demo_grp["actions"][start_t:end_t]
        if self.pad_seq_length and actual_len < self.seq_length:
            pad_len = self.seq_length - actual_len
            actions = np.concatenate(
                [actions, np.zeros((pad_len, actions.shape[-1]), dtype=actions.dtype)],
                axis=0,
            )

        low_dim_parts = []
        for key in self.obs_keys_low_dim:
            obs_data = demo_grp[f"obs/{key}"][start_t:end_t]
            if obs_data.ndim == 1:
                obs_data = obs_data[:, None]
            if self.pad_seq_length and obs_data.shape[0] < self.seq_length:
                pad_len = self.seq_length - obs_data.shape[0]
                obs_data = np.concatenate(
                    [obs_data, np.zeros((pad_len, *obs_data.shape[1:]), dtype=obs_data.dtype)],
                    axis=0,
                )
            low_dim_parts.append(obs_data)

        low_dim_obs = np.concatenate(low_dim_parts, axis=-1) if low_dim_parts else np.zeros((self.seq_length, 0))

        image_obs = {}
        for key in self.obs_keys_image:
            img = demo_grp[f"obs/{key}"][start_t:end_t]
            if self.pad_seq_length and img.shape[0] < self.seq_length:
                pad_len = self.seq_length - img.shape[0]
                img = np.concatenate(
                    [img, np.zeros((pad_len, *img.shape[1:]), dtype=img.dtype)],
                    axis=0,
                )

            img = img.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            image_obs[key] = torch.from_numpy(img)

        mask = np.zeros(self.seq_length if self.pad_seq_length else actual_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        return {
            "obs": {
                "low_dim": torch.from_numpy(low_dim_obs.astype(np.float32)),
                **image_obs,
            },
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "mask": torch.from_numpy(mask),
        }


def _get_demo_keys(hdf5_path: str, filter_key: Optional[str] = None) -> List[str]:
    """Get sorted list of demo keys from HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        if filter_key is not None and f"mask/{filter_key}" in f:
            mask = f[f"mask/{filter_key}"][:]
            all_demos = sorted(f["data"].keys())
            return [dk for dk, m in zip(all_demos, mask) if m]
        return sorted(f["data"].keys())


@register("data", "robomimic_ph_robosuite")
def build_robomimic_ph_dataset(
    data_dir: str,
    tasks: List[str],
    obs_keys: Dict[str, List[str]],
    action_keys: List[str],
    seq_length: int = 10,
    frame_stack: int = 1,
    pad_frame_stack: bool = True,
    pad_seq_length: bool = True,
    image_size: int = 84,
    hdf5_normalize_obs: bool = False,
    hdf5_filter_key: Optional[str] = None,
    eval_ratio: float = 0.1,
    horizon: int = 400,
    dataset_keys: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Build train/eval sequence datasets from robomimic PH demos."""
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])

    obs_keys_image = [k for k in obs_keys_image if k]
    use_images = len(obs_keys_image) > 0
    hdf5_name = "image.hdf5" if use_images else "low_dim.hdf5"

    all_train_demos = []
    all_eval_demos = []
    hdf5_paths = []

    for task in tasks:
        hdf5_path = os.path.join(data_dir, task, "ph", hdf5_name)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Dataset not found: {hdf5_path}\n"
                f"Run: python scripts/robomimic_download.py --tasks {task}"
            )

        demo_keys = _get_demo_keys(hdf5_path, filter_key=hdf5_filter_key)
        n_eval = max(1, int(len(demo_keys) * eval_ratio))
        n_train = len(demo_keys) - n_eval

        train_demos = demo_keys[:n_train]
        eval_demos = demo_keys[n_train:]

        print(f"[Robomimic] {task}/ph: {len(demo_keys)} demos "
              f"(train={len(train_demos)}, eval={len(eval_demos)})")

        all_train_demos.append((hdf5_path, train_demos))
        all_eval_demos.append((hdf5_path, eval_demos))
        hdf5_paths.append(hdf5_path)

    if len(tasks) == 1:
        hdf5_path, train_demos = all_train_demos[0]
        _, eval_demos = all_eval_demos[0]

        train_ds = RobomimicDataset(
            hdf5_path=hdf5_path,
            demo_keys=train_demos,
            obs_keys_low_dim=obs_keys_low_dim,
            obs_keys_image=obs_keys_image,
            action_keys=action_keys,
            seq_length=seq_length,
            frame_stack=frame_stack,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            image_size=image_size,
            hdf5_normalize_obs=hdf5_normalize_obs,
        )
        eval_ds = RobomimicDataset(
            hdf5_path=hdf5_path,
            demo_keys=eval_demos,
            obs_keys_low_dim=obs_keys_low_dim,
            obs_keys_image=obs_keys_image,
            action_keys=action_keys,
            seq_length=seq_length,
            frame_stack=frame_stack,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            image_size=image_size,
            hdf5_normalize_obs=hdf5_normalize_obs,
        )
    else:
        from torch.utils.data import ConcatDataset

        train_datasets = []
        eval_datasets = []
        for (hp, td), (_, ed) in zip(all_train_demos, all_eval_demos):
            common_kwargs = dict(
                obs_keys_low_dim=obs_keys_low_dim,
                obs_keys_image=obs_keys_image,
                action_keys=action_keys,
                seq_length=seq_length,
                frame_stack=frame_stack,
                pad_frame_stack=pad_frame_stack,
                pad_seq_length=pad_seq_length,
                image_size=image_size,
                hdf5_normalize_obs=hdf5_normalize_obs,
            )
            train_datasets.append(RobomimicDataset(hdf5_path=hp, demo_keys=td, **common_kwargs))
            eval_datasets.append(RobomimicDataset(hdf5_path=hp, demo_keys=ed, **common_kwargs))

        train_ds = ConcatDataset(train_datasets)
        eval_ds = ConcatDataset(eval_datasets)

    print(f"[Robomimic] Total: train={len(train_ds)}, eval={len(eval_ds)} samples")
    return {"train": train_ds, "eval": eval_ds}


class RobomimicBCDataset(Dataset):
    """Per-timestep dataset with history window for BC training.

    Each sample:
        images:       {cam_name: [H, C, h, w]}  float32 [0,1]  (or precomputed features)
        state:        [H, state_dim]             float32  (normalized if norm_stats provided)
        prev_actions: [H, action_dim]            float32  (normalized if norm_stats provided)
        action:       [action_dim]               float32  (normalized target)
        task_id:      scalar int64               task index (0-indexed, for task embedding)

    Early timesteps pad by repeating the first observation.

    Precomputed ViT features are read from ``obs/{cam_key}_features`` when
    present.
    """

    def __init__(
        self,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys_low_dim: List[str],
        obs_keys_image: List[str],
        history_length: int = 3,
        image_size: int = 84,
        norm_stats=None,
        action_chunk_size: int = 1,
        task_id: int = 0,
        target_state_dim: Optional[int] = None,
        feature_aug_mode: str = "random",
    ):
        self.hdf5_path = hdf5_path
        self.demo_keys = sorted(demo_keys)
        self.obs_keys_low_dim = obs_keys_low_dim
        self.obs_keys_image = [k for k in obs_keys_image if k]
        self.history_length = history_length
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.action_chunk_size = action_chunk_size
        self.task_id = task_id
        self.target_state_dim = target_state_dim
        self.feature_aug_mode = feature_aug_mode

        self._index = []
        self._demo_lengths = {}

        self._feature_keys = {}
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in self.demo_keys:
                T = f[f"data/{dk}/actions"].shape[0]
                self._demo_lengths[dk] = T
                for t in range(T):
                    self._index.append((dk, t))

            first_dk = self.demo_keys[0] if self.demo_keys else None
            if first_dk:
                for img_key in self.obs_keys_image:
                    feat_key = img_key.replace("_image", "_features")
                    if f"data/{first_dk}/obs/{feat_key}" in f:
                        self._feature_keys[img_key] = feat_key
                        print(f"[Dataset] Using precomputed features: {feat_key}")
                    else:
                        self._feature_keys[img_key] = None

                obs_grp = f[f"data/{first_dk}/obs"]
                self.state_dim = sum(
                    obs_grp[key].shape[-1] if obs_grp[key].ndim > 1 else 1
                    for key in self.obs_keys_low_dim
                    if key in obs_grp
                )
            else:
                self.state_dim = 0

            if first_dk:
                self._warn_if_constant_demo(f, first_dk)

        self._hdf5_file = None

    def __len__(self):
        return len(self._index)

    def _open(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")

    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def _warn_if_constant_demo(self, f: h5py.File, demo_key: str) -> None:
        """Warn if a rendered demo appears time-constant despite varying actions."""
        demo_grp = f[f"data/{demo_key}"]
        if demo_grp["actions"].shape[0] < 2:
            return
        action_delta = np.abs(np.diff(demo_grp["actions"][:], axis=0)).mean()
        if action_delta < 1e-6:
            return

        checks = []
        for key in self.obs_keys_low_dim:
            obs_path = f"obs/{key}"
            if obs_path in demo_grp:
                arr = demo_grp[obs_path][:]
                checks.append((key, float(np.abs(np.diff(arr.astype(np.float32), axis=0)).mean())))
        for img_key in self.obs_keys_image:
            feat_key = self._feature_keys.get(img_key)
            obs_path = f"obs/{feat_key or img_key}"
            if obs_path in demo_grp:
                arr = demo_grp[obs_path][:]
                checks.append((feat_key or img_key, float(np.abs(np.diff(arr.astype(np.float32), axis=0)).mean())))

        if checks and all(delta < 1e-6 for _, delta in checks):
            checked = ", ".join(name for name, _ in checks)
            print(
                f"[Dataset] WARNING: {self.hdf5_path} {demo_key} has varying actions "
                f"but time-constant observations/features ({checked}). If this is a "
                f"rendered robomimic image dataset, regenerate it with "
                f"`python scripts/render_images.py --dataset ... --overwrite` and "
                f"then rerun `scripts/precompute_features.py`."
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._open()
        demo_key, t = self._index[idx]
        demo_grp = self._hdf5_file[f"data/{demo_key}"]
        H = self.history_length

        hist_indices = [max(0, t - (H - 1 - i)) for i in range(H)]

        low_dim_parts = []
        for key in self.obs_keys_low_dim:
            obs_all = demo_grp[f"obs/{key}"]
            parts = np.stack([obs_all[hi] for hi in hist_indices])
            if parts.ndim == 1:
                parts = parts[:, None]
            low_dim_parts.append(parts)
        state = np.concatenate(low_dim_parts, axis=-1).astype(np.float32) if low_dim_parts else np.zeros((H, 0), dtype=np.float32)

        images = {}
        for img_key in self.obs_keys_image:
            feat_key = self._feature_keys.get(img_key)
            if feat_key is not None:
                feat_all = demo_grp[f"obs/{feat_key}"]
                feat_stack = np.stack([feat_all[hi] for hi in hist_indices])
                if feat_stack.ndim == 4:
                    n_aug = feat_stack.shape[1]
                    if self.feature_aug_mode == "canonical":
                        aug_idx = 0
                    else:
                        aug_idx = np.random.randint(0, n_aug)
                    feat_stack = feat_stack[:, aug_idx]
                images[img_key] = torch.from_numpy(feat_stack.astype(np.float32))
            else:
                img_all = demo_grp[f"obs/{img_key}"]
                img_stack = np.stack([img_all[hi] for hi in hist_indices])
                img_stack = img_stack.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                images[img_key] = torch.from_numpy(img_stack)

        action_dim = demo_grp["actions"].shape[1]
        prev_actions = []
        for i in range(H):
            pa_t = hist_indices[i] - 1
            if pa_t < 0:
                prev_actions.append(np.zeros(action_dim, dtype=np.float32))
            else:
                prev_actions.append(demo_grp["actions"][pa_t].astype(np.float32))
        prev_actions = np.stack(prev_actions)

        K = self.action_chunk_size
        T = self._demo_lengths[demo_key]
        if K > 1:
            action_indices = [min(t + k, T - 1) for k in range(K)]
            action = np.stack([demo_grp["actions"][ai] for ai in action_indices]).astype(np.float32)
        else:
            action = demo_grp["actions"][t].astype(np.float32)

        if self.norm_stats is not None:
            state = self.norm_stats.normalize_state(state)
            action = self.norm_stats.normalize_action(action)
            prev_actions = self.norm_stats.normalize_action(prev_actions)

        if self.target_state_dim is not None and state.shape[-1] < self.target_state_dim:
            pad = np.zeros((state.shape[0], self.target_state_dim - state.shape[-1]), dtype=np.float32)
            state = np.concatenate([state, pad], axis=-1)

        return {
            "images": {k: v.clone() for k, v in images.items()},
            "state": torch.from_numpy(state).clone(),
            "prev_actions": torch.from_numpy(prev_actions).clone(),
            "action": torch.from_numpy(action).clone(),
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }


@register("data", "robomimic_bc_robosuite")
def build_robomimic_bc_dataset(
    data_dir: str,
    tasks: List[str],
    obs_keys: Dict[str, List[str]],
    history_length: int = 3,
    image_size: int = 84,
    hdf5_filter_key: Optional[str] = None,
    eval_ratio: float = 0.1,
    norm_stats=None,
    action_chunk_size: int = 1,
    feature_aug_train_mode: str = "random",
    feature_aug_eval_mode: str = "canonical",
    **kwargs,
) -> Dict[str, Any]:
    """Build train/eval history-window datasets for robosuite BC."""
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])
    obs_keys_image = [k for k in obs_keys_image if k]

    use_images = len(obs_keys_image) > 0
    hdf5_name = "image.hdf5" if use_images else "low_dim.hdf5"

    all_train = []
    all_eval = []
    train_demo_info = []

    for i, task in enumerate(tasks):
        hdf5_path = os.path.join(data_dir, task, "ph", hdf5_name)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Dataset not found: {hdf5_path}\n"
                f"Run: python scripts/robomimic_download.py --tasks {task}\n"
                f"     python scripts/render_images.py --dataset data/robomimic/{task}/ph/low_dim.hdf5"
            )

        demo_keys = _get_demo_keys(hdf5_path, filter_key=hdf5_filter_key)
        n_eval = max(1, int(len(demo_keys) * eval_ratio))
        train_demos = demo_keys[: len(demo_keys) - n_eval]
        eval_demos = demo_keys[len(demo_keys) - n_eval :]
        train_demo_info.append((hdf5_path, train_demos))

        print(f"[Robomimic BC] {task}/ph (id={i}): {len(demo_keys)} demos "
              f"(train={len(train_demos)}, eval={len(eval_demos)}), "
              f"history={history_length}")

        common = dict(
            obs_keys_low_dim=obs_keys_low_dim,
            obs_keys_image=obs_keys_image,
            history_length=history_length,
            image_size=image_size,
            norm_stats=norm_stats,
            action_chunk_size=action_chunk_size,
            task_id=i,
        )
        all_train.append(RobomimicBCDataset(
            hdf5_path=hdf5_path,
            demo_keys=train_demos,
            feature_aug_mode=feature_aug_train_mode,
            **common,
        ))
        all_eval.append(RobomimicBCDataset(
            hdf5_path=hdf5_path,
            demo_keys=eval_demos,
            feature_aug_mode=feature_aug_eval_mode,
            **common,
        ))

    if len(tasks) > 1:
        max_state_dim = max(ds.state_dim for ds in all_train)
        for ds in all_train + all_eval:
            ds.target_state_dim = max_state_dim
        print(f"[Robomimic BC] Multi-task state dims: "
              f"{ {tasks[i]: ds.state_dim for i, ds in enumerate(all_train)} } -> padded to {max_state_dim}")

    all_ds = all_train + all_eval
    if len(tasks) > 1 and obs_keys_image:
        for img_key in obs_keys_image:
            has_features = [ds._feature_keys.get(img_key) is not None for ds in all_train]
            if any(has_features) and not all(has_features):
                missing = [tasks[i] for i, ok in enumerate(has_features) if not ok]
                print(f"[Robomimic BC] WARNING: precomputed features missing for "
                      f"'{img_key}' in tasks {missing}. Falling back to raw images for all tasks.")
                for ds in all_ds:
                    ds._feature_keys[img_key] = None

    if len(tasks) == 1:
        train_ds, eval_ds = all_train[0], all_eval[0]
    else:
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset(all_train)
        eval_ds = ConcatDataset(all_eval)

    print(f"[Robomimic BC] Total: train={len(train_ds)}, eval={len(eval_ds)} samples")
    print(f"[Robomimic BC] Task map: { {i: t for i, t in enumerate(tasks)} }")
    return {
        "train": train_ds,
        "eval": eval_ds,
        "_train_demo_info": train_demo_info,
        "_obs_keys_low_dim": obs_keys_low_dim,
        "_obs_keys_image": obs_keys_image,
        "_image_size": image_size,
        "_task_names": list(tasks),
    }


# ---------------------------------------------------------------------------
# Flightmare drone BC (state-only)
# ---------------------------------------------------------------------------
class FlightmareBCStateDataset(Dataset):
    """Per-timestep state-only Flightmare BC dataset.

    Each sample is a flat dict consumed by ``MLPFusionGaussianExpertActor``:

        state:        [state_dim]                  float32  (drone state ++ mission)
        prev_actions: [action_dim]                 float32
        action:       [action_dim]                 float32
        images:       {}                           (empty - kept for trainer compat)

    All values are mean/std normalized using ``norm_stats.npz`` written by the
    collector. ``state_dim = 13 (pos+vel+quat+omega) + 20 (mission v2:
    3-lookahead-gates body frame [pos:3, fwd:3] each + progress + dist) = 33``
    when ``include_mission=True`` (default).

    HDF5 handles are opened lazily per worker and cached, so this composes
    safely with ``DataLoader(num_workers>0)``.
    """

    ACTION_TYPES = ("waypoint", "ctbr", "motor")

    def __init__(
        self,
        data_dir: str,
        action_type: Literal["waypoint", "ctbr", "motor"] = "ctbr",
        split: Literal["train", "val", "all"] = "train",
        include_mission: bool = True,
        normalize_state: bool = True,
        normalize_action: bool = True,
        action_normalization: Literal["auto", "standard", "bounds"] = "auto",
        normalize_mission: bool = True,
    ):
        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"action_type must be one of {self.ACTION_TYPES}, got {action_type!r}")
        self.data_dir = Path(data_dir)
        self.action_type = action_type
        self.split = split
        self.include_mission = include_mission
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action
        self.normalize_mission = normalize_mission and include_mission
        self.action_normalization = action_normalization

        with open(self.data_dir / "index.json") as f:
            manifest = json.load(f)
        self._meta = manifest

        if split == "all":
            episodes = manifest["episodes"]
        else:
            episodes = [e for e in manifest["episodes"] if e.get("split", "train") == split]
        if not episodes:
            raise RuntimeError(
                f"No episodes for split={split!r} under {data_dir}. "
                f"Counts: {self._split_counts(manifest)}"
            )

        self._episode_paths = [str(self.data_dir / e["path"]) for e in episodes]
        # Drop the controller/sim warmup transient (state at rest, action
        # already non-hover) from the BC index. Recorded at collection time so
        # PPO/eval don't need to know - only training samples are filtered.
        skip = int(manifest.get("skip_initial_frames", 0))
        self.skip_initial_frames = max(0, skip)
        # Flat (episode_idx, t) sample index.
        self._index: List[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            T = int(ep["length"])
            t0 = min(self.skip_initial_frames, T)
            self._index.extend((ei, t) for t in range(t0, T))

        # Load + cache normalization stats as torch tensors (cheap; reused per item).
        stats_path = self.data_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"norm_stats.npz missing in {data_dir} - rerun collect.py to (re)build it."
            )
        s = np.load(stats_path)
        self.state_mean = torch.from_numpy(s["state_mean"]).float()
        self.state_std = torch.from_numpy(s["state_std"]).float()
        self.action_mean = torch.from_numpy(s[f"{action_type}_mean"]).float()
        self.action_std = torch.from_numpy(s[f"{action_type}_std"]).float()
        self.action_norm_mode = stats_mode(s, default="standard") if action_normalization == "auto" else str(action_normalization)
        low, high = action_bounds(action_type, s)
        self.action_low = torch.from_numpy(low).float()
        self.action_high = torch.from_numpy(high).float()
        if self.include_mission and "mission_mean" in s.files:
            self.mission_mean = torch.from_numpy(s["mission_mean"]).float()
            self.mission_std = torch.from_numpy(s["mission_std"]).float()
        else:
            self.mission_mean = self.mission_std = None
            if self.include_mission:
                print(f"[FlightmareBCState] mission stats missing in {stats_path}; "
                      f"skipping mission normalization.")

        # Resolved by first sample (can also pull from manifest).
        self.state_dim_drone = int(manifest.get("state_dim", 13))
        self.mission_dim = int(manifest.get("mission", {}).get("dim", 0))
        self.state_dim = self.state_dim_drone + (self.mission_dim if self.include_mission else 0)
        self.action_dim = 4

        self._handles: Dict[int, h5py.File] = {}
        self._owner_pid: Optional[int] = None

    @staticmethod
    def _split_counts(manifest: dict) -> dict:
        counts: Dict[str, int] = {}
        for ep in manifest["episodes"]:
            counts[ep.get("split", "train")] = counts.get(ep.get("split", "train"), 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._index)

    def _get_handle(self, ep_idx: int) -> h5py.File:
        pid = os.getpid()
        if self._owner_pid is None:
            self._owner_pid = pid
        elif self._owner_pid != pid:
            self._handles = {}
            self._owner_pid = pid
        h = self._handles.get(ep_idx)
        if h is None:
            h = h5py.File(self._episode_paths[ep_idx], "r", swmr=False, libver="latest")
            self._handles[ep_idx] = h
        return h

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, t = self._index[idx]
        h = self._get_handle(ep_idx)

        drone_state = torch.from_numpy(h["obs/state"][t]).float()
        if self.normalize_state:
            drone_state = (drone_state - self.state_mean) / self.state_std

        if self.include_mission and "mission/vec" in h:
            mission = torch.from_numpy(h["mission/vec"][t]).float()
            if self.normalize_mission and self.mission_mean is not None:
                mission = (mission - self.mission_mean) / self.mission_std
            state = torch.cat([drone_state, mission], dim=0)
        else:
            state = drone_state

        action_key = f"action/{self.action_type}"
        action = torch.from_numpy(h[action_key][t]).float()
        if t == 0:
            prev_action = torch.zeros_like(action)
        else:
            prev_action = torch.from_numpy(h[action_key][t - 1]).float()
        if self.normalize_action:
            if self.action_norm_mode == "bounds":
                scale = torch.clamp(self.action_high - self.action_low, min=1e-6)
                action = 2.0 * (action - self.action_low) / scale - 1.0
                prev_action = 2.0 * (prev_action - self.action_low) / scale - 1.0
            else:
                action = (action - self.action_mean) / self.action_std
                prev_action = (prev_action - self.action_mean) / self.action_std

        return {
            "images": {},  # kept for trainer-side compat (empty -> ignored)
            "state": state,
            "prev_actions": prev_action,
            "action": action,
        }

    # Hook for normalization-injection from trainer; this dataset already
    # normalizes inline so we accept any value silently.
    @property
    def task_id(self) -> int:
        return 0

    @task_id.setter
    def task_id(self, value: int) -> None:  # pragma: no cover - trivial
        pass


@register("data", "flightmare_bc_state")
def build_flightmare_bc_state(
    data_dir: str,
    action_type: Literal["waypoint", "ctbr", "motor"] = "ctbr",
    include_mission: bool = True,
    normalize_state: bool = True,
    normalize_action: bool = True,
    action_normalization: Literal["auto", "standard", "bounds"] = "auto",
    normalize_mission: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """Build train/val state-only Flightmare BC datasets.

    Returns the trainer-expected dict ``{"train": ..., "eval": ...}``. The
    BCTrainer skips its own norm-stats step because we don't include
    ``_train_demo_info`` (the dataset already normalizes inline using the
    collector's ``norm_stats.npz`` - that's the right behavior here, since
    Flightmare data has a single self-contained train split).
    """
    common = dict(
        data_dir=data_dir,
        action_type=action_type,
        include_mission=include_mission,
        normalize_state=normalize_state,
        normalize_action=normalize_action,
        action_normalization=action_normalization,
        normalize_mission=normalize_mission,
    )
    train_ds = FlightmareBCStateDataset(split="train", **common)
    try:
        eval_ds = FlightmareBCStateDataset(split="val", **common)
    except RuntimeError:
        # No val split (e.g. val_frac=0.0 collection). Fall back to a tiny
        # held-out slice from train so eval/loss/diversity still emit metrics.
        print("[FlightmareBCState] No val split in manifest; using last 10% of train as eval.")
        n = len(train_ds)
        n_eval = max(1, int(0.1 * n))
        # Use Subset wrappers to avoid double-loading; cheap.
        from torch.utils.data import Subset
        eval_idx = list(range(n - n_eval, n))
        train_idx = list(range(0, n - n_eval))
        eval_ds = Subset(train_ds, eval_idx)
        train_ds = Subset(train_ds, train_idx)

    print(f"[FlightmareBCState] {data_dir}  action={action_type}  "
          f"state_dim={getattr(train_ds, 'state_dim', '?')}  "
          f"action_dim=4  train={len(train_ds)}  eval={len(eval_ds)}")
    return {
        "train": train_ds,
        "eval": eval_ds,
        "_obs_keys_low_dim": [],
        "_obs_keys_image": [],
        "_image_size": 0,
        "_task_names": ["flightmare"],
        "_state_dim": getattr(train_ds, "state_dim", 27),
        "_action_dim": getattr(train_ds, "action_dim", 4),
    }


# ---------------------------------------------------------------------------
# Flightmare drone BC (state-only, obs v3 — Swift-style split-then-fuse)
# ---------------------------------------------------------------------------
class FlightmareBCStateV3Dataset(Dataset):
    """Per-timestep Flightmare BC dataset emitting Swift-style obs v3 blocks.

    Each sample is a flat dict consumed by ``MLPFusionGaussianExpertActor``
    when configured with ``fusion=swift``:

        proprio_core: [9]      v_body, omega_body, gravity_body
        gate:         [24]     2 lookahead gates × 4 corners × 3 (body-frame)
        aux:          [3]      progress, dist_to_current, time_since_pass
        prev_actions: [4]      last commanded action (per action_type)
        action:       [4]      target action

    Mean/std normalization uses the v3 stats appended to ``norm_stats.npz``
    by ``transform_to_v3.py``. Action normalization shares the v2 path
    (``bounds`` or ``standard``) so action-type ablations stay comparable
    between v2 and v3 datasets.
    """

    ACTION_TYPES = ("waypoint", "ctbr", "motor")

    def __init__(
        self,
        data_dir: str,
        action_type: Literal["waypoint", "ctbr", "motor"] = "ctbr",
        split: Literal["train", "val", "all"] = "train",
        normalize_obs: bool = True,
        normalize_action: bool = True,
        action_normalization: Literal["auto", "standard", "bounds"] = "auto",
        preload: bool = True,
    ):
        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"action_type must be one of {self.ACTION_TYPES}, got {action_type!r}")
        self.data_dir = Path(data_dir)
        self.action_type = action_type
        self.split = split
        self.normalize_obs = normalize_obs
        self.normalize_action = normalize_action
        self.action_normalization = action_normalization
        self.preload = preload

        with open(self.data_dir / "index.json") as f:
            manifest = json.load(f)
        self._meta = manifest

        if "obs_v3" not in manifest:
            raise RuntimeError(
                f"{data_dir} has no obs_v3 block in index.json — run "
                f"`python -m scripts.flightmare_bc.transform_to_v3 --data-dir {data_dir}` first."
            )
        v3 = manifest["obs_v3"]
        self.proprio_core_dim = int(v3["proprio_core"]["dim"])
        self.gate_dim = int(v3["gate"]["dim"])
        self.aux_dim = int(v3["aux"]["dim"])

        if split == "all":
            episodes = manifest["episodes"]
        else:
            episodes = [e for e in manifest["episodes"] if e.get("split", "train") == split]
        if not episodes:
            raise RuntimeError(
                f"No episodes for split={split!r} under {data_dir}. "
                f"Counts: {FlightmareBCStateDataset._split_counts(manifest)}"
            )

        self._episode_paths = [str(self.data_dir / e["path"]) for e in episodes]
        skip = int(manifest.get("skip_initial_frames", 0))
        self.skip_initial_frames = max(0, skip)
        self._index: List[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            T = int(ep["length"])
            t0 = min(self.skip_initial_frames, T)
            self._index.extend((ei, t) for t in range(t0, T))

        stats_path = self.data_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"norm_stats.npz missing in {data_dir}.")
        s = np.load(stats_path)
        for needed in ("proprio_core_mean", "gate_mean", "aux_mean"):
            if needed not in s.files:
                raise RuntimeError(
                    f"{stats_path} missing v3 stats key {needed!r} — rerun transform_to_v3.py."
                )
        self.proprio_mean = torch.from_numpy(s["proprio_core_mean"]).float()
        self.proprio_std = torch.from_numpy(s["proprio_core_std"]).float()
        self.gate_mean = torch.from_numpy(s["gate_mean"]).float()
        self.gate_std = torch.from_numpy(s["gate_std"]).float()
        self.aux_mean = torch.from_numpy(s["aux_mean"]).float()
        self.aux_std = torch.from_numpy(s["aux_std"]).float()

        self.action_mean = torch.from_numpy(s[f"{action_type}_mean"]).float()
        self.action_std = torch.from_numpy(s[f"{action_type}_std"]).float()
        self.action_norm_mode = (
            stats_mode(s, default="standard")
            if action_normalization == "auto"
            else str(action_normalization)
        )
        low, high = action_bounds(action_type, s)
        self.action_low = torch.from_numpy(low).float()
        self.action_high = torch.from_numpy(high).float()

        # Composite "state" dim used by the architecture builder when it wants
        # a single number to log. The Swift fusion model consumes the three
        # blocks separately, but trainers/loggers like a scalar.
        self.state_dim = self.proprio_core_dim + self.gate_dim + self.aux_dim
        self.action_dim = 4

        self._handles: Dict[int, h5py.File] = {}
        self._owner_pid: Optional[int] = None

        # Preload everything into flat tensors. The whole BC dataset is small
        # (~800k samples * ~50 floats = ~160 MB), and the model is tiny, so
        # h5 random reads in __getitem__ dominate the step time. After preload
        # we apply normalization once, then __getitem__ is pure tensor indexing
        # and worker processes can run with num_workers=0.
        self._pre_state: Optional[torch.Tensor] = None
        self._pre_prev: Optional[torch.Tensor] = None
        self._pre_action: Optional[torch.Tensor] = None
        if self.preload:
            self._build_preload(episodes)

    def _build_preload(self, episodes: list) -> None:
        proprio_blocks: List[np.ndarray] = []
        gate_blocks: List[np.ndarray] = []
        aux_blocks: List[np.ndarray] = []
        action_blocks: List[np.ndarray] = []
        prev_action_blocks: List[np.ndarray] = []
        action_key = f"action/{self.action_type}"
        for ei, ep in enumerate(episodes):
            T = int(ep["length"])
            t0 = min(self.skip_initial_frames, T)
            if T <= t0:
                continue
            with h5py.File(self._episode_paths[ei], "r", libver="latest") as h:
                proprio_blocks.append(h["obs/proprio_core"][t0:T].astype(np.float32, copy=False))
                gate_blocks.append(h["obs/gate"][t0:T].astype(np.float32, copy=False))
                aux_blocks.append(h["obs/aux"][t0:T].astype(np.float32, copy=False))
                a = h[action_key][:T].astype(np.float32, copy=False)
                action_blocks.append(a[t0:T])
                prev = np.empty_like(a[t0:T])
                if t0 == 0:
                    prev[0] = 0.0
                    prev[1:] = a[: T - 1]
                else:
                    prev[:] = a[t0 - 1 : T - 1]
                prev_action_blocks.append(prev)

        proprio = torch.from_numpy(np.concatenate(proprio_blocks, axis=0))
        gate = torch.from_numpy(np.concatenate(gate_blocks, axis=0))
        aux = torch.from_numpy(np.concatenate(aux_blocks, axis=0))
        action = torch.from_numpy(np.concatenate(action_blocks, axis=0))
        prev_action = torch.from_numpy(np.concatenate(prev_action_blocks, axis=0))

        if self.normalize_obs:
            proprio = (proprio - self.proprio_mean) / self.proprio_std
            gate = (gate - self.gate_mean) / self.gate_std
            aux = (aux - self.aux_mean) / self.aux_std
        if self.normalize_action:
            if self.action_norm_mode == "bounds":
                scale = torch.clamp(self.action_high - self.action_low, min=1e-6)
                action = 2.0 * (action - self.action_low) / scale - 1.0
                prev_action = 2.0 * (prev_action - self.action_low) / scale - 1.0
            else:
                action = (action - self.action_mean) / self.action_std
                prev_action = (prev_action - self.action_mean) / self.action_std

        self._pre_state = torch.cat([proprio, gate, aux], dim=1).contiguous()
        self._pre_prev = prev_action.contiguous()
        self._pre_action = action.contiguous()
        N = self._pre_state.shape[0]
        if N != len(self._index):
            # Replace index with simple [0..N) since preload already applied skip.
            self._index = [(0, i) for i in range(N)]
        bytes_total = sum(
            t.numel() * t.element_size()
            for t in (self._pre_state, self._pre_prev, self._pre_action)
        )
        print(
            f"[FlightmareBCStateV3] preloaded split={self.split!r} "
            f"samples={N} state_dim={self._pre_state.shape[1]} "
            f"action_dim={self._pre_action.shape[1]} "
            f"mem={bytes_total / 1e6:.1f} MB"
        )

    def __len__(self) -> int:
        if self._pre_state is not None:
            return self._pre_state.shape[0]
        return len(self._index)

    def _get_handle(self, ep_idx: int) -> h5py.File:
        pid = os.getpid()
        if self._owner_pid is None:
            self._owner_pid = pid
        elif self._owner_pid != pid:
            self._handles = {}
            self._owner_pid = pid
        h = self._handles.get(ep_idx)
        if h is None:
            h = h5py.File(self._episode_paths[ep_idx], "r", swmr=False, libver="latest")
            self._handles[ep_idx] = h
        return h

    def collate_preloaded(self, batch: list) -> Dict[str, Any]:
        """Custom collate used when preload is on: avoids torch.stack of
        16k tiny tensors by doing one index_select per field. Pass this via
        DataLoader(collate_fn=ds.collate_preloaded) — see the trainer.
        """
        # Bypass default collate; recover the indices from the dict entries
        # is not possible, so the trainer is expected to use a BatchSampler
        # that yields a list[int] AND set collate_fn=self._collate_indices.
        states = torch.stack([b["state"] for b in batch], dim=0)
        prevs = torch.stack([b["prev_actions"] for b in batch], dim=0)
        acts = torch.stack([b["action"] for b in batch], dim=0)
        return {"images": {}, "state": states, "prev_actions": prevs, "action": acts}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._pre_state is not None:
            return {
                "images": {},
                "state": self._pre_state[idx],
                "prev_actions": self._pre_prev[idx],
                "action": self._pre_action[idx],
            }
        ep_idx, t = self._index[idx]
        h = self._get_handle(ep_idx)

        proprio = torch.from_numpy(h["obs/proprio_core"][t]).float()
        gate = torch.from_numpy(h["obs/gate"][t]).float()
        aux = torch.from_numpy(h["obs/aux"][t]).float()
        if self.normalize_obs:
            proprio = (proprio - self.proprio_mean) / self.proprio_std
            gate = (gate - self.gate_mean) / self.gate_std
            aux = (aux - self.aux_mean) / self.aux_std

        action_key = f"action/{self.action_type}"
        action = torch.from_numpy(h[action_key][t]).float()
        if t == 0:
            prev_action = torch.zeros_like(action)
        else:
            prev_action = torch.from_numpy(h[action_key][t - 1]).float()
        if self.normalize_action:
            if self.action_norm_mode == "bounds":
                scale = torch.clamp(self.action_high - self.action_low, min=1e-6)
                action = 2.0 * (action - self.action_low) / scale - 1.0
                prev_action = 2.0 * (prev_action - self.action_low) / scale - 1.0
            else:
                action = (action - self.action_mean) / self.action_std
                prev_action = (prev_action - self.action_mean) / self.action_std

        # Single flat ``state`` tensor (proprio_core ++ gate ++ aux). The
        # Swift fusion model splits it internally using stored dims. Emitting
        # one tensor here keeps the rollout collector / history buffer paths
        # unchanged from the v2 schema.
        state = torch.cat([proprio, gate, aux], dim=0)
        return {
            "images": {},
            "state": state,
            "prev_actions": prev_action,
            "action": action,
        }

    @property
    def task_id(self) -> int:
        return 0

    @task_id.setter
    def task_id(self, value: int) -> None:
        pass


@register("data", "flightmare_bc_state_v3")
def build_flightmare_bc_state_v3(
    data_dir: str,
    action_type: Literal["waypoint", "ctbr", "motor"] = "ctbr",
    normalize_obs: bool = True,
    normalize_action: bool = True,
    action_normalization: Literal["auto", "standard", "bounds"] = "auto",
    preload: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """Build train/val Swift-obs-v3 Flightmare BC datasets."""
    common = dict(
        data_dir=data_dir,
        action_type=action_type,
        normalize_obs=normalize_obs,
        normalize_action=normalize_action,
        action_normalization=action_normalization,
        preload=preload,
    )
    train_ds = FlightmareBCStateV3Dataset(split="train", **common)
    try:
        eval_ds = FlightmareBCStateV3Dataset(split="val", **common)
    except RuntimeError:
        print("[FlightmareBCStateV3] No val split in manifest; using last 10% of train as eval.")
        from torch.utils.data import Subset
        n = len(train_ds)
        n_eval = max(1, int(0.1 * n))
        eval_idx = list(range(n - n_eval, n))
        train_idx = list(range(0, n - n_eval))
        eval_ds = Subset(train_ds, eval_idx)
        train_ds = Subset(train_ds, train_idx)

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    print(
        f"[FlightmareBCStateV3] {data_dir}  action={action_type}  "
        f"proprio={base_ds.proprio_core_dim}  gate={base_ds.gate_dim}  aux={base_ds.aux_dim}  "
        f"action_dim=4  train={len(train_ds)}  eval={len(eval_ds)}"
    )
    return {
        "train": train_ds,
        "eval": eval_ds,
        "_obs_keys_low_dim": [],
        "_obs_keys_image": [],
        "_image_size": 0,
        "_task_names": ["flightmare"],
        "_state_dim": base_ds.state_dim,
        "_action_dim": base_ds.action_dim,
        "_proprio_core_dim": base_ds.proprio_core_dim,
        "_gate_dim": base_ds.gate_dim,
        "_aux_dim": base_ds.aux_dim,
    }

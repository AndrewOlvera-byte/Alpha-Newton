"""
Robomimic PH (Proficient Human) dataset builder.

Registered as data type "robomimic_ph" in the registry.
Loads HDF5 demos from robomimic, extracts low-dim state + images + actions,
and returns torch-ready train/eval splits.
"""
from __future__ import annotations

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional

from src.core.registry import register


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

        # Build index: list of (demo_key, start_idx) for each valid window
        self._index = []
        self._demo_lengths = {}

        with h5py.File(self.hdf5_path, "r") as f:
            for dk in self.demo_keys:
                demo_grp = f[f"data/{dk}"]
                T = demo_grp["actions"].shape[0]
                self._demo_lengths[dk] = T
                for t in range(T):
                    self._index.append((dk, t))

        # Lazy-opened file handle (opened on first __getitem__)
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

        # Determine sequence range
        end_t = min(start_t + self.seq_length, T)
        actual_len = end_t - start_t

        # --- Actions ---
        actions = demo_grp["actions"][start_t:end_t]  # (actual_len, action_dim)
        if self.pad_seq_length and actual_len < self.seq_length:
            pad_len = self.seq_length - actual_len
            actions = np.concatenate(
                [actions, np.zeros((pad_len, actions.shape[-1]), dtype=actions.dtype)],
                axis=0,
            )

        # --- Low-dim obs ---
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

        # --- Image obs ---
        image_obs = {}
        for key in self.obs_keys_image:
            img = demo_grp[f"obs/{key}"][start_t:end_t]  # (actual_len, H, W, C)
            if self.pad_seq_length and img.shape[0] < self.seq_length:
                pad_len = self.seq_length - img.shape[0]
                img = np.concatenate(
                    [img, np.zeros((pad_len, *img.shape[1:]), dtype=img.dtype)],
                    axis=0,
                )
            # HWC -> CHW, normalize to [0, 1]
            img = img.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            image_obs[key] = torch.from_numpy(img)

        # Build mask for valid timesteps
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


@register("data", "robomimic_ph")
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
    """Build robomimic PH dataset for BC training.

    Args:
        data_dir: Root directory containing task HDF5 files
                  (e.g., data/robomimic/lift/ph/low_dim.hdf5 or image.hdf5)
        tasks: List of task names ["lift", "can", "square"]
        obs_keys: {"low_dim": [...], "image": [...]}
        action_keys: Keys for action data
        seq_length: Length of each training subsequence
        frame_stack: Number of frames to stack (history)
        image_size: Expected image resolution
        eval_ratio: Fraction of demos held out for eval
        horizon: Max episode horizon (for reference)

    Returns:
        {"train": RobomimicDataset, "eval": RobomimicDataset}
    """
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])

    obs_keys_image = [k for k in obs_keys_image if k]  # filter empty strings
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

    # For single-task, return directly; multi-task uses ConcatDataset
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


# ─────────────────────────────────────────────────────────────
# BC History Dataset — per-timestep samples with history window
# ─────────────────────────────────────────────────────────────

class RobomimicBCDataset(Dataset):
    """Per-timestep dataset with history window for BC training.

    Each sample:
        images:       {cam_name: [H, C, h, w]}  float32 [0,1]  (or precomputed features)
        state:        [H, state_dim]             float32  (normalized if norm_stats provided)
        prev_actions: [H, action_dim]            float32  (normalized if norm_stats provided)
        action:       [action_dim]               float32  (normalized target)
        task_id:      scalar int64               task index (0-indexed, for task embedding)

    Early timesteps pad by repeating the first observation.

    If precomputed ViT features exist in the HDF5 under
    obs/{cam_key}_features, they are loaded instead of raw images,
    bypassing the ViT entirely during training (huge throughput gain).
    """

    def __init__(
        self,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys_low_dim: List[str],
        obs_keys_image: List[str],
        history_length: int = 3,
        image_size: int = 84,
        norm_stats=None,  # NormStats instance or None
        action_chunk_size: int = 1,
        task_id: int = 0,  # integer task index — embedded by the model for multi-task conditioning
        target_state_dim: Optional[int] = None,  # pad state to this dim for multi-task batching
        feature_aug_mode: str = "random",  # "random" for train, "canonical" for eval
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

        # Detect whether precomputed ViT features are available.
        # Feature key: replace '_image' suffix with '_features'.
        self._feature_keys = {}  # {image_key: feature_key or None}
        with h5py.File(self.hdf5_path, "r") as f:
            for dk in self.demo_keys:
                T = f[f"data/{dk}/actions"].shape[0]
                self._demo_lengths[dk] = T
                for t in range(T):
                    self._index.append((dk, t))

            # Check for precomputed features on first demo
            first_dk = self.demo_keys[0] if self.demo_keys else None
            if first_dk:
                for img_key in self.obs_keys_image:
                    feat_key = img_key.replace("_image", "_features")
                    if f"data/{first_dk}/obs/{feat_key}" in f:
                        self._feature_keys[img_key] = feat_key
                        print(f"[Dataset] Using precomputed features: {feat_key}")
                    else:
                        self._feature_keys[img_key] = None

                # Compute actual state dim for this task
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

        # History frame indices — pad early timesteps by repeating first frame
        hist_indices = [max(0, t - (H - 1 - i)) for i in range(H)]

        # --- Low-dim state: [H, state_dim] ---
        low_dim_parts = []
        for key in self.obs_keys_low_dim:
            obs_all = demo_grp[f"obs/{key}"]
            parts = np.stack([obs_all[hi] for hi in hist_indices])
            if parts.ndim == 1:
                parts = parts[:, None]
            low_dim_parts.append(parts)
        state = np.concatenate(low_dim_parts, axis=-1).astype(np.float32) if low_dim_parts else np.zeros((H, 0), dtype=np.float32)

        # --- Image obs (or precomputed features): {cam: tensor} ---
        images = {}
        for img_key in self.obs_keys_image:
            feat_key = self._feature_keys.get(img_key)
            if feat_key is not None:
                # Precomputed ViT features.
                # Shape [T, n_patches, d] (no aug) or [T, N_aug, n_patches, d] (with aug).
                feat_all = demo_grp[f"obs/{feat_key}"]
                feat_stack = np.stack([feat_all[hi] for hi in hist_indices])  # [H, ...]
                if feat_stack.ndim == 4:
                    # Has aug dimension [H, N_aug, n_patches, d].
                    # Train uses random augmented views; eval uses the canonical
                    # first view to keep checkpoint selection deterministic.
                    n_aug = feat_stack.shape[1]
                    if self.feature_aug_mode == "canonical":
                        aug_idx = 0
                    else:
                        aug_idx = np.random.randint(0, n_aug)
                    feat_stack = feat_stack[:, aug_idx]  # [H, n_patches, d]
                images[img_key] = torch.from_numpy(feat_stack.astype(np.float32))
            else:
                # Raw images: [T, H, W, C] → [H, C, H, W] in [0, 1]
                img_all = demo_grp[f"obs/{img_key}"]
                img_stack = np.stack([img_all[hi] for hi in hist_indices])  # [H, h, w, C]
                img_stack = img_stack.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                images[img_key] = torch.from_numpy(img_stack)

        # --- Previous actions: [H, action_dim] ---
        action_dim = demo_grp["actions"].shape[1]
        prev_actions = []
        for i in range(H):
            pa_t = hist_indices[i] - 1
            if pa_t < 0:
                prev_actions.append(np.zeros(action_dim, dtype=np.float32))
            else:
                prev_actions.append(demo_grp["actions"][pa_t].astype(np.float32))
        prev_actions = np.stack(prev_actions)

        # --- Target action(s) at t — supports action chunking ---
        K = self.action_chunk_size
        T = self._demo_lengths[demo_key]
        if K > 1:
            # Grab K future actions, pad with last action if near end of demo
            action_indices = [min(t + k, T - 1) for k in range(K)]
            action = np.stack([demo_grp["actions"][ai] for ai in action_indices]).astype(np.float32)  # [K, action_dim]
        else:
            action = demo_grp["actions"][t].astype(np.float32)

        # --- Normalization ---
        if self.norm_stats is not None:
            state = self.norm_stats.normalize_state(state)
            action = self.norm_stats.normalize_action(action)
            prev_actions = self.norm_stats.normalize_action(prev_actions)

        # --- Pad state to target_state_dim for multi-task batching ---
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


@register("data", "robomimic_bc")
def build_robomimic_bc_dataset(
    data_dir: str,
    tasks: List[str],
    obs_keys: Dict[str, List[str]],
    history_length: int = 3,
    image_size: int = 84,
    hdf5_filter_key: Optional[str] = None,
    eval_ratio: float = 0.1,
    norm_stats=None,  # NormStats instance (computed by trainer, then injected)
    action_chunk_size: int = 1,
    feature_aug_train_mode: str = "random",
    feature_aug_eval_mode: str = "canonical",
    **kwargs,
) -> Dict[str, Any]:
    """Build robomimic BC dataset with per-timestep history windows.

    Returns {
        "train": RobomimicBCDataset,
        "eval":  RobomimicBCDataset,
        "train_demos": List[Tuple[hdf5_path, List[demo_keys]]],  # for norm stats
    }.
    """
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])
    obs_keys_image = [k for k in obs_keys_image if k]

    use_images = len(obs_keys_image) > 0
    hdf5_name = "image.hdf5" if use_images else "low_dim.hdf5"

    all_train = []
    all_eval = []
    train_demo_info = []  # list of (hdf5_path, demo_keys) for norm stats

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
            task_id=i,  # 0-indexed — matches n_tasks in architecture config
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

    # For multi-task: tasks may have different object-state dims (e.g. lift=10, can/square=14).
    # Pad all state tensors to the max state_dim so the default collate can stack them.
    if len(tasks) > 1:
        max_state_dim = max(ds.state_dim for ds in all_train)
        for ds in all_train + all_eval:
            ds.target_state_dim = max_state_dim
        print(f"[Robomimic BC] Multi-task state dims: "
              f"{ {tasks[i]: ds.state_dim for i, ds in enumerate(all_train)} } → padded to {max_state_dim}")

    # For multi-task: enforce consistent feature strategy across all task datasets.
    # If any task is missing precomputed features for a camera, fall back to raw images
    # for that camera in ALL tasks — otherwise mixed [H,256,384] vs [H,3,84,84] tensors
    # in the same batch cause "Trying to resize storage that is not resizable" in collate.
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
        "_task_names": list(tasks),  # index → task name (for logging/inference)
    }

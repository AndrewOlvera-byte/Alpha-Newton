"""Normalization statistics for robotics demonstrations."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import h5py
import numpy as np
import torch

_MIN_STD = 1e-3


@dataclass
class NormStats:
    """Per-dimension normalization statistics."""

    action_mean: List[float]
    action_std: List[float]
    state_mean: List[float]
    state_std: List[float]

    @classmethod
    def compute(
        cls,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys_low_dim: List[str],
        min_std: float = _MIN_STD,
        clip_percentile: float = 1.0,
    ) -> "NormStats":
        """Compute per-dimension action and state statistics."""
        with h5py.File(hdf5_path, "r") as f:
            actions = np.concatenate(
                [f[f"data/{dk}/actions"][:] for dk in demo_keys], axis=0
            )

            state_parts = []
            for key in obs_keys_low_dim:
                obs = np.concatenate(
                    [f[f"data/{dk}/obs/{key}"][:] for dk in demo_keys], axis=0
                )
                if obs.ndim == 1:
                    obs = obs[:, None]
                state_parts.append(obs)
            state = np.concatenate(state_parts, axis=-1)

        if clip_percentile > 0.0:
            lo_a = np.percentile(actions, clip_percentile, axis=0)
            hi_a = np.percentile(actions, 100.0 - clip_percentile, axis=0)
            actions = np.clip(actions, lo_a, hi_a)
            lo_s = np.percentile(state, clip_percentile, axis=0)
            hi_s = np.percentile(state, 100.0 - clip_percentile, axis=0)
            state = np.clip(state, lo_s, hi_s)

        action_mean = actions.mean(0).tolist()
        action_std = np.maximum(actions.std(0), min_std).tolist()
        state_mean = state.mean(0).tolist()
        state_std = np.maximum(state.std(0), min_std).tolist()

        raw_action_std = actions.std(0)
        raw_state_std = state.std(0)
        n_clamped_a = int((raw_action_std < min_std).sum())
        n_clamped_s = int((raw_state_std < min_std).sum())
        if n_clamped_a > 0:
            print(f"[NormStats] Clamped {n_clamped_a} near-zero action std dims to {min_std}")
        if n_clamped_s > 0:
            print(f"[NormStats] Clamped {n_clamped_s} near-zero state std dims to {min_std}")

        return cls(
            action_mean=action_mean,
            action_std=action_std,
            state_mean=state_mean,
            state_std=state_std,
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormStats":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def save_to_hdf5(self, hdf5_path: str, n_train_demos: int = 0) -> None:
        """Write stats to norm_stats/ group in the dataset HDF5. Idempotent."""
        with h5py.File(hdf5_path, "a") as f:
            if "norm_stats" in f:
                del f["norm_stats"]
            grp = f.create_group("norm_stats")
            grp.create_dataset("action_mean", data=np.array(self.action_mean, dtype=np.float32))
            grp.create_dataset("action_std",  data=np.array(self.action_std,  dtype=np.float32))
            grp.create_dataset("state_mean",  data=np.array(self.state_mean,  dtype=np.float32))
            grp.create_dataset("state_std",   data=np.array(self.state_std,   dtype=np.float32))
            grp.attrs["n_train_demos"] = n_train_demos

    @classmethod
    def load_from_hdf5(cls, hdf5_path: str) -> "NormStats":
        with h5py.File(hdf5_path, "r") as f:
            grp = f["norm_stats"]
            return cls(
                action_mean=grp["action_mean"][:].tolist(),
                action_std=grp["action_std"][:].tolist(),
                state_mean=grp["state_mean"][:].tolist(),
                state_std=grp["state_std"][:].tolist(),
            )

    @classmethod
    def exists_in_hdf5(cls, hdf5_path: str) -> bool:
        """Return True if norm_stats/ group is present in the HDF5."""
        with h5py.File(hdf5_path, "r") as f:
            return "norm_stats" in f

    def _to_arrays(self):
        """Return mean/std as numpy float32 arrays."""
        return (
            np.array(self.action_mean, dtype=np.float32),
            np.array(self.action_std, dtype=np.float32),
            np.array(self.state_mean, dtype=np.float32),
            np.array(self.state_std, dtype=np.float32),
        )

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        mean = np.array(self.action_mean, dtype=np.float32)
        std = np.array(self.action_std, dtype=np.float32)
        return (action - mean) / std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        mean = np.array(self.action_mean, dtype=np.float32)
        std = np.array(self.action_std, dtype=np.float32)
        return action * std + mean

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        mean = np.array(self.state_mean, dtype=np.float32)
        std = np.array(self.state_std, dtype=np.float32)
        return (state - mean) / std

    def to_torch_buffers(self, device: torch.device):
        """Return (action_mean, action_std, state_mean, state_std) as torch tensors."""
        a_mean, a_std, s_mean, s_std = self._to_arrays()
        return (
            torch.from_numpy(a_mean).to(device),
            torch.from_numpy(a_std).to(device),
            torch.from_numpy(s_mean).to(device),
            torch.from_numpy(s_std).to(device),
        )

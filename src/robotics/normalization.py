"""
Action and state normalization for BC training.

Best practice (from Diffusion Policy, ACT, π0):
- Compute per-dimension mean/std from training data
- Normalize: (x - mean) / (std + eps)
- Handle zero-variance dims by clamping std to min_std
- Save stats to JSON alongside checkpoint for inference

Usage:
    stats = NormStats.compute(hdf5_path, demo_keys, obs_keys_low_dim)
    stats.save("outputs/bc_lift_ph/norm_stats.json")
    stats = NormStats.load("outputs/bc_lift_ph/norm_stats.json")

    norm_action = stats.normalize_action(action)   # [B, action_dim] or [action_dim]
    raw_action  = stats.denormalize_action(norm)
    norm_state  = stats.normalize_state(state)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import h5py
import numpy as np
import torch


# Minimum std to avoid division by near-zero for nearly-constant dimensions
_MIN_STD = 1e-3


@dataclass
class NormStats:
    """Per-dimension normalization statistics."""

    action_mean: List[float]
    action_std: List[float]
    state_mean: List[float]
    state_std: List[float]

    # ------------------------------------------------------------------
    # Compute from HDF5
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys_low_dim: List[str],
        min_std: float = _MIN_STD,
        clip_percentile: float = 1.0,
    ) -> "NormStats":
        """Compute mean/std from training demos with outlier clipping.

        Args:
            hdf5_path: Path to robomimic HDF5 file
            demo_keys: List of demo keys to include (train split only)
            obs_keys_low_dim: Ordered list of low-dim obs keys (must match dataset construction)
            min_std: Floor for std — prevents divide-by-zero on constant dims
            clip_percentile: Clip values outside [p, 100-p] percentile before computing
                             stats. 1.0 clips bottom/top 1% — prevents outliers from
                             skewing mean/std. Set to 0.0 to disable.
        """
        with h5py.File(hdf5_path, "r") as f:
            # Collect actions
            actions = np.concatenate(
                [f[f"data/{dk}/actions"][:] for dk in demo_keys], axis=0
            )  # [N, action_dim]

            # Collect state (concatenate obs keys in order)
            state_parts = []
            for key in obs_keys_low_dim:
                obs = np.concatenate(
                    [f[f"data/{dk}/obs/{key}"][:] for dk in demo_keys], axis=0
                )  # [N, dim] or [N]
                if obs.ndim == 1:
                    obs = obs[:, None]
                state_parts.append(obs)
            state = np.concatenate(state_parts, axis=-1)  # [N, state_dim]

        # Clip per-dimension outliers before computing stats
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

        # Report any clamped dims
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

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormStats":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    # ------------------------------------------------------------------
    # Normalize / denormalize (numpy or torch)
    # ------------------------------------------------------------------

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

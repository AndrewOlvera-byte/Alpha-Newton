"""Torch Dataset for Flightmare BC HDF5 episodes.

* Episode files are opened **lazily per worker** - no h5py handle is created
  in ``__init__`` or pickled by the DataLoader. Each worker maintains its own
  cache of file handles keyed by absolute path. This is the only safe way to
  combine h5py with multi-worker DataLoaders.
* The flat sample index is materialized once at construction (cheap: just a
  list of ``(file_idx, t)`` tuples), so ``__getitem__`` does a single
  dictionary lookup + a few HDF5 dataset reads.
* Augmentation is applied on the GPU side by the trainer is preferred, but a
  CPU-side ``transforms.v2`` pipeline is wired up here too (see ``augment``).

Registered as ``data.type = "flightmare_bc"`` so existing ``train_bc.py`` can
build it via the ``build("data", **cfg.data)`` registry call.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.registry import register
from scripts.flightmare_bc.action_norms import action_bounds, stats_mode
from scripts.flightmare_bc.augmentations import build_eval_aug, build_train_aug


_ActionT = Literal["waypoint", "ctbr", "motor"]


class FlightmareBCDataset(Dataset):
    """Per-step BC dataset.

    Args
    ----
    data_dir: directory containing ``index.json`` + ``episodes/``.
    action_type: which action label to expose as the BC target.
    split: ``"train"`` / ``"val"`` / ``"all"`` based on ``index.json`` splits.
    image_size: target square size; images are resized in the aug pipeline.
    augment: enable training-time photometric aug (ignored on val).
    history_length: currently fixed to 1 (single-frame, see README rationale).
    include_prev_action: include the previous-step action in the batch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        action_type: _ActionT = "ctbr",
        split: Literal["train", "val", "all"] = "train",
        image_size: int = 224,
        augment: bool = True,
        history_length: int = 1,
        include_prev_action: bool = True,
        cameras: list[str] | None = None,
        normalize_state: bool = True,
        normalize_action: bool = True,
        action_normalization: Literal["auto", "standard", "bounds"] = "auto",
        include_mission: bool = True,
        normalize_mission: bool = True,
    ):
        if history_length != 1:
            raise ValueError(
                "FlightmareBCDataset is single-frame Markov by design "
                "(see README); set history_length=1."
            )
        self.data_dir = Path(data_dir)
        self.action_type = action_type
        self.split = split
        self.image_size = image_size
        self.include_prev_action = include_prev_action
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action
        self.action_normalization = action_normalization
        self.include_mission = include_mission
        self.normalize_mission = normalize_mission and include_mission

        with open(self.data_dir / "index.json") as f:
            manifest = json.load(f)
        self._meta = manifest
        all_episodes = manifest["episodes"]
        if split == "all":
            episodes = all_episodes
        else:
            episodes = [e for e in all_episodes if e.get("split", "train") == split]
        if not episodes:
            raise RuntimeError(f"No episodes for split={split} under {data_dir}")

        self._episode_paths = [str(self.data_dir / e["path"]) for e in episodes]
        self.cameras = cameras or manifest["cameras"]

        # Flat (episode_idx, t) sample index.
        self._index: list[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            T = int(ep["length"])
            self._index.extend((ei, t) for t in range(T))

        self.aug = build_train_aug(image_size) if (augment and split == "train") else build_eval_aug(image_size)

        norm_path = self.data_dir / "norm_stats.npz"
        if normalize_state or normalize_action:
            if not norm_path.exists():
                raise FileNotFoundError(
                    f"norm_stats.npz missing in {data_dir}; rerun collect.py."
                )
            stats = np.load(norm_path)
            self.state_mean = torch.from_numpy(stats["state_mean"]).float()
            self.state_std = torch.from_numpy(stats["state_std"]).float()
            self.action_mean = torch.from_numpy(stats[f"{action_type}_mean"]).float()
            self.action_std = torch.from_numpy(stats[f"{action_type}_std"]).float()
            self.action_norm_mode = stats_mode(stats, default="standard") if action_normalization == "auto" else str(action_normalization)
            low, high = action_bounds(action_type, stats)
            self.action_low = torch.from_numpy(low).float()
            self.action_high = torch.from_numpy(high).float()
            if self.normalize_mission and "mission_mean" in stats.files:
                self.mission_mean = torch.from_numpy(stats["mission_mean"]).float()
                self.mission_std = torch.from_numpy(stats["mission_std"]).float()
            else:
                self.mission_mean = self.mission_std = None
        else:
            self.state_mean = self.state_std = self.action_mean = self.action_std = None
            self.mission_mean = self.mission_std = None
            self.action_norm_mode = "standard"
            self.action_low = self.action_high = None

        self._handles: dict[int, h5py.File] = {}
        self._owner_pid: int | None = None

    # -- handle management ------------------------------------------------
    def _get_handle(self, episode_idx: int) -> h5py.File:
        pid = os.getpid()
        if self._owner_pid is None:
            self._owner_pid = pid
        elif self._owner_pid != pid:
            self._handles = {}
            self._owner_pid = pid
        h = self._handles.get(episode_idx)
        if h is None:
            h = h5py.File(self._episode_paths[episode_idx], "r", swmr=False, libver="latest")
            self._handles[episode_idx] = h
        return h

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self._index[idx]
        h = self._get_handle(ep_idx)

        state = torch.from_numpy(h["obs/state"][t]).float()
        action_key = f"action/{self.action_type}"
        action = torch.from_numpy(h[action_key][t]).float()
        if t == 0:
            prev_action = torch.zeros_like(action)
        else:
            prev_action = torch.from_numpy(h[action_key][t - 1]).float()

        images: dict[str, torch.Tensor] = {}
        for cam in self.cameras:
            ds_key = f"obs/image_{cam}"
            img = h[ds_key][t]                         # uint8 (H, W, 3)
            img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 (3, H, W)
            img_t = self.aug(img_t)
            images[cam] = img_t.float()

        if self.normalize_state and self.state_mean is not None:
            state = (state - self.state_mean) / self.state_std
        if self.normalize_action and self.action_mean is not None:
            if self.action_norm_mode == "bounds":
                scale = torch.clamp(self.action_high - self.action_low, min=1e-6)
                action = 2.0 * (action - self.action_low) / scale - 1.0
                prev_action = 2.0 * (prev_action - self.action_low) / scale - 1.0
            else:
                action = (action - self.action_mean) / self.action_std
                prev_action = (prev_action - self.action_mean) / self.action_std

        out = {
            "images": images,
            "state": state,
            "prev_actions": prev_action,
            "action": action,
        }
        if self.include_mission and "mission/vec" in h:
            mission = torch.from_numpy(h["mission/vec"][t]).float()
            if self.normalize_mission and self.mission_mean is not None:
                mission = (mission - self.mission_mean) / self.mission_std
            out["mission"] = mission
        return out


def _collate(batch: list[dict]) -> dict:
    """Stack same-key tensors; preserve dict-of-images structure if multi-cam."""
    out: dict = {}
    sample0 = batch[0]
    for k, v in sample0.items():
        if isinstance(v, dict):
            out[k] = {ck: torch.stack([b[k][ck] for b in batch], 0) for ck in v}
        else:
            out[k] = torch.stack([b[k] for b in batch], 0)
    return out


@register("data", "flightmare_bc")
def build_flightmare_bc(**kwargs):
    kwargs.pop("type", None)
    split = kwargs.pop("split", "train")
    if split == "val":
        return FlightmareBCDataset(split="val", **{**kwargs, "augment": False})
    train_ds = FlightmareBCDataset(split="train", **kwargs)
    val_ds = None
    if split in ("train", "all"):
        try:
            val_ds = FlightmareBCDataset(split="val", **{**kwargs, "augment": False})
        except RuntimeError as e:
            if "No episodes for split=val" not in str(e):
                raise
    return {"train": train_ds, "eval": val_ds, "collate_fn": _collate}

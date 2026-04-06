"""Inference-time history buffer for maintaining observation history.

Used during both eval rollouts and RL fine-tuning.  Accepts raw (unnormalized)
observations and handles normalization internally so the calling code only needs
to know the HDF5 path, not the training output directory.

Typical RL usage:
    from src.robotics.normalization import NormStats
    ns = NormStats.load_from_hdf5("data/robomimic/lift/ph/image.hdf5")
    buf = HistoryBuffer(history_length=3, action_dim=7, task_id=0, norm_stats=ns)

    obs, _ = env.reset()
    buf.reset()
    while not done:
        buf.push(obs["images"], obs["state"])          # raw obs, no norm needed
        batch = buf.get_batch(device)                  # normalized + task_id ready
        action_norm = model.predict(batch).cpu().numpy()[0]
        action_world = ns.denormalize_action(action_norm)
        obs, reward, done, _ = env.step(action_world)
        buf.push_action(action_world)                  # record executed action
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch


class HistoryBuffer:
    """Maintains a fixed-size buffer of past observations and actions for inference.

    At each step:
      1. Call .push(obs_images, obs_state) with the current (raw) observation.
      2. Call .get_batch(device) to build the model-ready batch.
      3. Run model.predict(batch) to get a normalized action.
      4. Denormalize with norm_stats.denormalize_action() and apply to the env.
      5. Call .record_action(raw_action) with the action actually executed.

    task_id  — integer matching the task index used during BC training (0-indexed,
               same as the tasks list in the config).  Required for multi-task
               models so the task embedding fires correctly.

    norm_stats — NormStats for this task (load from the task's HDF5 via
                 NormStats.load_from_hdf5).  When provided, state and prev_actions
                 are normalized before being returned by get_batch(), exactly
                 mirroring what the BC dataset did during training.
    """

    def __init__(
        self,
        history_length: int,
        action_dim: int,
        task_id: int = 0,
        norm_stats=None,  # NormStats instance for this task, or None
        target_state_dim: int = None,  # pad state to this dim after normalization (multi-task)
    ):
        self.history_length = history_length
        self.action_dim = action_dim
        self.task_id = task_id
        self.norm_stats = norm_stats
        self.target_state_dim = target_state_dim

        self._images: deque = deque(maxlen=history_length)
        self._states: deque = deque(maxlen=history_length)
        self._prev_actions: deque = deque(maxlen=history_length)
        self._step = 0

    def reset(self):
        self._images.clear()
        self._states.clear()
        self._prev_actions.clear()
        self._step = 0

    def push(self, obs_images: Dict[str, np.ndarray], obs_state: np.ndarray):
        """Push a new raw observation (before normalization).

        On the very first step prev_action defaults to zeros (no prior action).
        Call record_action() after each env step to register the executed action.
        """
        prev_action = (
            list(self._prev_actions)[-1]
            if self._prev_actions
            else np.zeros(self.action_dim, dtype=np.float32)
        )
        self._images.append(obs_images)
        self._states.append(obs_state.astype(np.float32))
        self._prev_actions.append(prev_action)
        self._step += 1

    def record_action(self, raw_action: np.ndarray):
        """Record the action actually executed so the next push uses it as prev_action.

        Call this AFTER env.step() with the raw (world-space) action.
        The buffer stores it raw; normalization happens in get_batch().
        """
        if self._prev_actions:
            # Replace the last entry (which was the stale prev from push())
            actions_list = list(self._prev_actions)
            actions_list[-1] = raw_action.astype(np.float32)
            self._prev_actions.clear()
            self._prev_actions.extend(actions_list)

    def get_batch(self, device: Optional[torch.device] = None) -> dict:
        """Build a model-ready batch dict with B=1, normalized if norm_stats set.

        Returns keys: images, state, prev_actions, task_id.
        All values are on `device` if specified.
        """
        H = self.history_length

        # Pad by repeating the first available frame
        images_list = list(self._images)
        states_list = list(self._states)
        actions_list = list(self._prev_actions)

        while len(images_list) < H:
            images_list.insert(0, images_list[0])
            states_list.insert(0, states_list[0])
            actions_list.insert(0, np.zeros(self.action_dim, dtype=np.float32))

        # ── State: [H, state_dim] ──────────────────────────────────────
        state_np = np.stack(states_list).astype(np.float32)  # [H, state_dim]
        if self.norm_stats is not None:
            state_np = self.norm_stats.normalize_state(state_np)
        # Pad to target_state_dim after normalization (mirrors training dataset behavior)
        if self.target_state_dim is not None and state_np.shape[-1] < self.target_state_dim:
            pad = np.zeros((state_np.shape[0], self.target_state_dim - state_np.shape[-1]), dtype=np.float32)
            state_np = np.concatenate([state_np, pad], axis=-1)

        # ── Prev actions: [H, action_dim] ─────────────────────────────
        prev_actions_np = np.stack(actions_list).astype(np.float32)  # [H, action_dim]
        if self.norm_stats is not None:
            prev_actions_np = self.norm_stats.normalize_action(prev_actions_np)

        # ── Images: {cam: [1, H, C, h, w]} ────────────────────────────
        cam_keys = images_list[0].keys()
        images = {}
        for cam in cam_keys:
            imgs = np.stack([frame[cam] for frame in images_list])  # [H, ...]
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                # HWC → CHW and normalize to [0, 1]
                imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            t = torch.from_numpy(imgs).unsqueeze(0)  # [1, H, C, h, w]
            images[cam] = t.to(device) if device is not None else t

        state = torch.from_numpy(state_np).unsqueeze(0)          # [1, H, state_dim]
        prev_actions = torch.from_numpy(prev_actions_np).unsqueeze(0)  # [1, H, action_dim]
        task_id = torch.tensor([self.task_id], dtype=torch.long)

        if device is not None:
            state = state.to(device)
            prev_actions = prev_actions.to(device)
            task_id = task_id.to(device)

        return {
            "images": images,
            "state": state,
            "prev_actions": prev_actions,
            "task_id": task_id,
        }

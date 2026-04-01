"""Inference-time history buffer for maintaining observation history."""
from __future__ import annotations
from collections import deque
from typing import Dict

import numpy as np
import torch


class HistoryBuffer:
    """Maintains a fixed-size buffer of past observations and actions for inference.

    At each step, call .push(obs_images, obs_state, action) then .get_batch()
    to construct the model input. Early timesteps are padded by repeating
    the first observation.
    """

    def __init__(self, history_length: int, action_dim: int):
        self.history_length = history_length
        self.action_dim = action_dim
        self._images: deque = deque(maxlen=history_length)
        self._states: deque = deque(maxlen=history_length)
        self._prev_actions: deque = deque(maxlen=history_length)
        self._step = 0

    def reset(self):
        self._images.clear()
        self._states.clear()
        self._prev_actions.clear()
        self._step = 0

    def push(self, obs_images: Dict[str, np.ndarray], obs_state: np.ndarray, prev_action: np.ndarray = None):
        """Push a new observation. prev_action is the action taken BEFORE this observation."""
        if self._step == 0 and prev_action is None:
            prev_action = np.zeros(self.action_dim, dtype=np.float32)

        self._images.append(obs_images)
        self._states.append(obs_state)
        self._prev_actions.append(prev_action if prev_action is not None else np.zeros(self.action_dim, dtype=np.float32))
        self._step += 1

    def get_batch(self, device: torch.device = None) -> dict:
        """Get model-ready batch with history padding. Returns batch with B=1."""
        H = self.history_length

        # Pad by repeating first frame
        images_list = list(self._images)
        states_list = list(self._states)
        actions_list = list(self._prev_actions)

        while len(images_list) < H:
            images_list.insert(0, images_list[0])
            states_list.insert(0, states_list[0])
            actions_list.insert(0, np.zeros(self.action_dim, dtype=np.float32))

        # Build tensors
        cam_keys = images_list[0].keys()
        images = {}
        for cam in cam_keys:
            imgs = np.stack([frame[cam] for frame in images_list])
            # HWC → CHW if needed, normalize [0,1]
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            t = torch.from_numpy(imgs).unsqueeze(0)  # [1, H, C, h, w]
            if device is not None:
                t = t.to(device)
            images[cam] = t

        state = torch.from_numpy(np.stack(states_list).astype(np.float32)).unsqueeze(0)
        prev_actions = torch.from_numpy(np.stack(actions_list).astype(np.float32)).unsqueeze(0)

        if device is not None:
            state = state.to(device)
            prev_actions = prev_actions.to(device)

        return {
            "images": images,
            "state": state,
            "prev_actions": prev_actions,
        }

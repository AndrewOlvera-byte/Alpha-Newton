"""MissionWrapper: deployment-time adapter from raw drone I/O to policy I/O.

At training time, our BC dataset contains a per-step ``mission`` vector that
encodes "where the next K gates are, in body frame, plus how far through the
course we are." This is the perception prior that a real autonomy stack would
synthesize from a *known track map* + an estimated drone pose (VIO / GPS+IMU
fusion / motion capture, depending on platform).

At deploy time we don't have a sim-side oracle that hands us this vector. We
must reproduce it from:

  * the gate map (loaded once per course, world-frame)
  * the current estimated pose (pos, quat, vel, omega) -- whatever the
    state-estimation stack produces

This module is the only piece of the policy stack that depends on the
existence of a track map and a pose estimate. The neural policy itself is
agnostic: it just consumes ``state``, ``mission``, and ``image``.

Interface mirrors what the BC dataset emits:
    batch = {
        "images":      {cam: tensor[B,3,H,W]},
        "state":       tensor[B,13],   # pos, vel, quat, omega
        "mission":     tensor[B,M],    # MISSION_DIM
        "prev_actions": tensor[B,A],   # action_dim
    }

Example:
    wrapper = MissionWrapper(
        policy=trained_policy,
        gates=gate_list_from_track_map,
        norm_stats="data/flightmare/gates_v1/norm_stats.npz",
    )
    action = wrapper.act(pos, vel, quat, omega, image)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.flightmare_bc.mission import (
    LOOKAHEAD_GATES,
    MISSION_DIM,
    MissionTracker,
    encode_mission,
    gates_to_views,
)


@dataclass
class GateMap:
    """A loaded track map. Same fields as ``GateSpec`` but deploy-side."""
    pos: np.ndarray
    yaw: float
    size: np.ndarray = None  # type: ignore[assignment]


def load_gates_from_manifest(index_json_path: str | Path, episode_idx: int = 0) -> list[GateMap]:
    """Convenience: pull a gate course out of a collector ``index.json``."""
    import json
    with open(index_json_path) as f:
        manifest = json.load(f)
    ep = manifest["episodes"][episode_idx]
    return [
        GateMap(
            pos=np.asarray(g["pos"], dtype=np.float64),
            yaw=float(g["yaw"]),
            size=np.asarray(g["size"], dtype=np.float64),
        )
        for g in ep.get("gates", [])
    ]


class MissionWrapper:
    """Adapter between raw drone state + image and a trained policy.

    Parameters
    ----------
    policy : torch.nn.Module
        Loaded BC policy expecting ``forward(batch_dict)`` and returning an
        action tensor (or distribution mean) of shape ``[B, action_dim]``.
    gates : list[GateMap | GateSpec | dict]
        World-frame gate map. Anything with ``.pos`` and ``.yaw`` (or matching
        dict keys) works.
    norm_stats : path or None
        ``norm_stats.npz`` from collection. Used to apply the same
        normalization the dataset applied during training. If ``None``, no
        normalization is applied (the policy must have been trained that way).
    action_mean / action_std : optional
        If passed, the wrapper denormalizes the policy's output using these
        before returning real-world actions. Otherwise it is up to the caller.
    device : torch device
        Where the policy lives.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        gates,
        norm_stats: str | Path | None = None,
        action_type: str = "ctbr",
        device: str | torch.device = "cpu",
        lookahead: int = LOOKAHEAD_GATES,
    ):
        self.policy = policy
        self.action_type = action_type
        self.device = torch.device(device)
        self.tracker = MissionTracker(gates, lookahead=lookahead)

        self._state_mean = self._state_std = None
        self._mission_mean = self._mission_std = None
        self._action_mean = self._action_std = None
        if norm_stats is not None:
            stats = np.load(norm_stats)
            self._state_mean = torch.from_numpy(stats["state_mean"]).float().to(self.device)
            self._state_std = torch.from_numpy(stats["state_std"]).float().to(self.device)
            if "mission_mean" in stats.files:
                self._mission_mean = torch.from_numpy(stats["mission_mean"]).float().to(self.device)
                self._mission_std = torch.from_numpy(stats["mission_std"]).float().to(self.device)
            mkey = f"{action_type}_mean"
            if mkey in stats.files:
                self._action_mean = torch.from_numpy(stats[mkey]).float().to(self.device)
                self._action_std = torch.from_numpy(stats[f"{action_type}_std"]).float().to(self.device)

        self._prev_action = None  # populated on first act()

    @property
    def current_gate_index(self) -> int:
        return self.tracker.current_index

    @property
    def n_gates(self) -> int:
        return len(self.tracker.gates)

    def reset(self) -> None:
        self.tracker.current_index = 0
        self._prev_action = None

    @torch.no_grad()
    def act(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        image: np.ndarray | torch.Tensor,
        camera_name: str = "forward",
    ) -> np.ndarray:
        """Run one forward pass. Returns the (denormalized) action."""
        self.tracker.update(np.asarray(pos))
        mission_np = encode_mission(
            np.asarray(pos), np.asarray(quat),
            self.tracker.gates, self.tracker.current_index, self.tracker.lookahead,
        )

        state_np = np.concatenate([pos, vel, quat, omega]).astype(np.float32)
        state = torch.from_numpy(state_np).to(self.device).unsqueeze(0)
        mission = torch.from_numpy(mission_np).to(self.device).unsqueeze(0)

        if isinstance(image, np.ndarray):
            img_t = torch.from_numpy(image)
            if img_t.ndim == 3 and img_t.shape[-1] == 3:  # H,W,3 uint8
                img_t = img_t.permute(2, 0, 1)
            img_t = img_t.float() / 255.0 if img_t.dtype == torch.uint8 else img_t.float()
        else:
            img_t = image.float()
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.device)

        if self._state_mean is not None:
            state = (state - self._state_mean) / self._state_std
        if self._mission_mean is not None:
            mission = (mission - self._mission_mean) / self._mission_std

        if self._prev_action is None:
            action_dim = self._action_mean.shape[0] if self._action_mean is not None else 4
            self._prev_action = torch.zeros(1, action_dim, device=self.device)

        batch = {
            "images": {camera_name: img_t},
            "state": state,
            "mission": mission,
            "prev_actions": self._prev_action,
        }
        out = self.policy(batch)
        if isinstance(out, dict):
            out = out.get("action", out.get("mean"))
        action = out.detach().float()

        # Cache for next call (in normalized space, matches training).
        self._prev_action = action

        if self._action_mean is not None:
            action = action * self._action_std + self._action_mean
        return action.squeeze(0).cpu().numpy()

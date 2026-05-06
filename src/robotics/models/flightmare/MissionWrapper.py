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

from scripts.flightmare_bc.action_norms import action_bounds, stats_mode
from scripts.flightmare_bc.mission import (
    LOOKAHEAD_GATES,
    MISSION_DIM,
    MissionTracker,
    encode_mission,
    gates_to_views,
)
from scripts.flightmare_bc.obs_v3 import (
    AUX_DIM,
    GATE_DIM,
    LOOKAHEAD_GATES_V3,
    PROPRIO_CORE_DIM,
    GateSpec,
    build_proprio_core,
    encode_aux,
    encode_gate_corners,
)


@dataclass
class GateMap:
    """A loaded track map. Same fields as ``GateSpec`` but deploy-side."""
    pos: np.ndarray
    yaw: float
    size: np.ndarray = None  # type: ignore[assignment]


@dataclass
class MissionObservation:
    """Latest world-model state produced by ``MissionWrapper``."""

    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    drone_state: np.ndarray
    mission: np.ndarray
    current_gate_index: int
    n_gates: int

    @property
    def completed(self) -> bool:
        return self.n_gates > 0 and self.current_gate_index >= self.n_gates

    @property
    def dist_to_next_gate(self) -> float:
        return float(self.mission[-1]) if self.mission.size else 0.0


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
        policy: torch.nn.Module | None,
        gates,
        norm_stats: str | Path | None = None,
        action_type: str = "ctbr",
        device: str | torch.device = "cpu",
        lookahead: int = LOOKAHEAD_GATES,
        include_mission: bool = True,
        concat_mission_to_state: bool = False,
        obs_schema: str = "v2",
    ):
        if obs_schema not in {"v2", "v3"}:
            raise ValueError(f"obs_schema must be 'v2' or 'v3', got {obs_schema!r}")
        self.policy = policy
        self.action_type = action_type
        self.device = torch.device(device)
        self.lookahead = int(lookahead) if obs_schema == "v2" else LOOKAHEAD_GATES_V3
        self.include_mission = bool(include_mission)
        self.concat_mission_to_state = bool(concat_mission_to_state)
        self.obs_schema = obs_schema
        self.tracker = MissionTracker(gates, lookahead=self.lookahead)
        self._gate_specs: list[GateSpec] = self._gates_to_specs(gates)
        # v3 time_since_pass tracker (in seconds; advanced by act/update at
        # caller cadence — eval driver tells us via update_world_model).
        self._step_count = 0
        self._last_pass_step = 0
        self._gate_index_v3 = 0
        self._dt = 0.01  # default 100 Hz; eval driver may set via set_dt().

        self._state_mean = self._state_std = None
        self._mission_mean = self._mission_std = None
        self._proprio_mean = self._proprio_std = None
        self._gate_mean = self._gate_std = None
        self._aux_mean = self._aux_std = None
        self._action_mean = self._action_std = None
        self._action_low = self._action_high = None
        self._action_norm_mode = "standard"
        if norm_stats is not None:
            stats = np.load(norm_stats)
            self._state_mean = torch.from_numpy(stats["state_mean"]).float().to(self.device)
            self._state_std = torch.from_numpy(stats["state_std"]).float().to(self.device)
            if "mission_mean" in stats.files:
                self._mission_mean = torch.from_numpy(stats["mission_mean"]).float().to(self.device)
                self._mission_std = torch.from_numpy(stats["mission_std"]).float().to(self.device)
            if obs_schema == "v3":
                for k in ("proprio_core_mean", "gate_mean", "aux_mean"):
                    if k not in stats.files:
                        raise RuntimeError(
                            f"{norm_stats} missing v3 stats {k!r} — run "
                            f"`python -m scripts.flightmare_bc.transform_to_v3` first."
                        )
                self._proprio_mean = torch.from_numpy(stats["proprio_core_mean"]).float().to(self.device)
                self._proprio_std = torch.from_numpy(stats["proprio_core_std"]).float().to(self.device)
                self._gate_mean = torch.from_numpy(stats["gate_mean"]).float().to(self.device)
                self._gate_std = torch.from_numpy(stats["gate_std"]).float().to(self.device)
                self._aux_mean = torch.from_numpy(stats["aux_mean"]).float().to(self.device)
                self._aux_std = torch.from_numpy(stats["aux_std"]).float().to(self.device)
            mkey = f"{action_type}_mean"
            if mkey in stats.files:
                self._action_mean = torch.from_numpy(stats[mkey]).float().to(self.device)
                self._action_std = torch.from_numpy(stats[f"{action_type}_std"]).float().to(self.device)
                low, high = action_bounds(action_type, stats)
                self._action_low = torch.from_numpy(low).float().to(self.device)
                self._action_high = torch.from_numpy(high).float().to(self.device)
                self._action_norm_mode = stats_mode(stats, default="standard")

        self._prev_action = None  # populated on first act()
        self._last_observation: MissionObservation | None = None

    @staticmethod
    def _gates_to_specs(gates) -> list[GateSpec]:
        specs: list[GateSpec] = []
        for g in gates:
            if hasattr(g, "pos"):
                pos = np.asarray(g.pos, dtype=np.float64)
                yaw = float(getattr(g, "yaw", 0.0))
                size = np.asarray(getattr(g, "size", (1.6, 1.6, 1.6)), dtype=np.float64)
                quat = getattr(g, "quat", None)
            else:
                pos = np.asarray(g["pos"], dtype=np.float64)
                yaw = float(g.get("yaw", 0.0))
                size = np.asarray(g.get("size", (1.6, 1.6, 1.6)), dtype=np.float64)
                quat = g.get("quat")
            if quat is not None:
                quat = np.asarray(quat, dtype=np.float64).reshape(4)
            specs.append(GateSpec(pos=pos, yaw=yaw, size=size, quat=quat))
        return specs

    def set_dt(self, dt: float) -> None:
        self._dt = float(dt)

    @property
    def current_gate_index(self) -> int:
        return self.tracker.current_index

    @property
    def n_gates(self) -> int:
        return len(self.tracker.gates)

    def set_gates(self, gates) -> None:
        """Replace the active track map and clear per-episode state."""
        self.tracker = MissionTracker(gates, lookahead=self.lookahead)
        self._gate_specs = self._gates_to_specs(gates)
        self.reset()

    def reset(self) -> None:
        self.tracker.current_index = 0
        self._prev_action = None
        self._last_observation = None
        self._step_count = 0
        self._last_pass_step = 0
        self._gate_index_v3 = 0

    def update_world_model(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
    ) -> MissionObservation:
        """Update gate progress and synthesize the mission vector from raw state."""
        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        omega = np.asarray(omega, dtype=np.float32)

        self.tracker.update(pos)
        self._step_count += 1
        mission_np = encode_mission(
            pos,
            quat,
            self.tracker.gates,
            self.tracker.current_index,
            self.tracker.lookahead,
        )
        state_np = np.concatenate([pos, vel, quat, omega]).astype(np.float32)
        self._last_observation = MissionObservation(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            drone_state=state_np,
            mission=mission_np,
            current_gate_index=int(self.tracker.current_index),
            n_gates=len(self.tracker.gates),
        )
        return self._last_observation

    @property
    def last_observation(self) -> MissionObservation | None:
        return self._last_observation

    def _action_dim(self) -> int:
        if self._action_mean is not None:
            return int(self._action_mean.shape[0])
        if self.policy is not None and hasattr(self.policy, "action_dim"):
            return int(getattr(self.policy, "action_dim"))
        return 4

    def _prepare_prev_action(
        self,
        prev_action: np.ndarray | torch.Tensor | None,
        prev_action_normalized: bool,
    ) -> torch.Tensor:
        if prev_action is None:
            if self._prev_action is None:
                prev = torch.zeros(1, self._action_dim(), device=self.device)
                if self._action_mean is not None:
                    prev = self._normalize_action(prev)
                return prev
            return self._prev_action.to(self.device)

        if isinstance(prev_action, torch.Tensor):
            prev = prev_action.detach().float().to(self.device)
        else:
            prev = torch.from_numpy(np.asarray(prev_action, dtype=np.float32)).to(self.device)
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
        if not prev_action_normalized and self._action_mean is not None:
            prev = self._normalize_action(prev)
        return prev

    def make_policy_batch(
        self,
        image: np.ndarray | torch.Tensor | None = None,
        camera_name: str = "forward",
        prev_action: np.ndarray | torch.Tensor | None = None,
        prev_action_normalized: bool = False,
    ) -> dict:
        """Build the normalized actor batch from the latest world-model state."""
        if self._last_observation is None:
            raise RuntimeError("update_world_model() must be called before make_policy_batch().")

        obs = self._last_observation

        if self.obs_schema == "v3":
            current = int(self.tracker.current_index)
            if current != self._gate_index_v3:
                self._last_pass_step = self._step_count
                self._gate_index_v3 = current
            time_since_pass = float(self._step_count - self._last_pass_step) * self._dt
            proprio_np = build_proprio_core(obs.vel, obs.quat, obs.omega)
            gate_np = encode_gate_corners(
                obs.pos, obs.quat, self._gate_specs, current, LOOKAHEAD_GATES_V3
            )
            aux_np = encode_aux(obs.pos, self._gate_specs, current, time_since_pass)
            proprio = torch.from_numpy(proprio_np).float().to(self.device).unsqueeze(0)
            gate = torch.from_numpy(gate_np).float().to(self.device).unsqueeze(0)
            aux = torch.from_numpy(aux_np).float().to(self.device).unsqueeze(0)
            if self._proprio_mean is not None:
                proprio = (proprio - self._proprio_mean) / self._proprio_std
                gate = (gate - self._gate_mean) / self._gate_std
                aux = (aux - self._aux_mean) / self._aux_std
            state = torch.cat([proprio, gate, aux], dim=-1)
            batch = {
                "images": {},
                "state": state,
                "prev_actions": self._prepare_prev_action(prev_action, prev_action_normalized),
            }
        else:
            state = torch.from_numpy(obs.drone_state).float().to(self.device).unsqueeze(0)
            mission = torch.from_numpy(obs.mission).float().to(self.device).unsqueeze(0)

            if self._state_mean is not None:
                state = (state - self._state_mean) / self._state_std
            if self._mission_mean is not None:
                mission = (mission - self._mission_mean) / self._mission_std

            batch = {
                "images": {},
                "state": state,
                "prev_actions": self._prepare_prev_action(prev_action, prev_action_normalized),
            }

            if self.include_mission:
                if self.concat_mission_to_state:
                    batch["state"] = torch.cat([state, mission], dim=-1)
                else:
                    batch["mission"] = mission

        if image is not None:
            if isinstance(image, np.ndarray):
                img_t = torch.from_numpy(image)
                if img_t.ndim == 3 and img_t.shape[-1] == 3:  # H,W,3 uint8
                    img_t = img_t.permute(2, 0, 1)
                img_t = img_t.float() / 255.0 if img_t.dtype == torch.uint8 else img_t.float()
            else:
                img_t = image.float()
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            batch["images"] = {camera_name: img_t.to(self.device)}
        return batch

    def record_policy_action(self, action: torch.Tensor | np.ndarray, normalized: bool = True) -> None:
        """Cache the previous action in the normalized space expected by training."""
        if isinstance(action, torch.Tensor):
            action_t = action.detach().float().to(self.device)
        else:
            action_t = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(self.device)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        if not normalized and self._action_mean is not None:
            action_t = self._normalize_action(action_t)
        self._prev_action = action_t

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._action_mean is None:
            return action
        if self._action_norm_mode == "bounds":
            if self._action_low is None or self._action_high is None:
                raise RuntimeError("Bounds action normalization selected but action bounds are missing.")
            scale = torch.clamp(self._action_high - self._action_low, min=1e-6)
            return 2.0 * (action - self._action_low) / scale - 1.0
        return (action - self._action_mean) / self._action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._action_mean is not None:
            if self._action_norm_mode == "bounds":
                if self._action_low is None or self._action_high is None:
                    raise RuntimeError("Bounds action normalization selected but action bounds are missing.")
                scale = torch.clamp(self._action_high - self._action_low, min=1e-6)
                return self._action_low + 0.5 * (action + 1.0) * scale
            return action * self._action_std + self._action_mean
        return action

    @torch.no_grad()
    def predict_from_world_model(
        self,
        image: np.ndarray | torch.Tensor | None = None,
        camera_name: str = "forward",
    ) -> np.ndarray:
        """Run ``policy.predict(batch)`` if available and return real-world actions."""
        if self.policy is None:
            raise RuntimeError("MissionWrapper was constructed without a policy.")

        batch = self.make_policy_batch(image=image, camera_name=camera_name)
        if hasattr(self.policy, "predict"):
            action_norm = self.policy.predict(batch)
        else:
            out = self.policy(batch)
            if isinstance(out, dict):
                action_norm = out.get("action", out.get("mean", out.get("mu")))
            else:
                action_norm = out
        if action_norm is None:
            raise RuntimeError("Policy did not return an action, mean, or mu tensor.")
        action_norm = action_norm.detach().float()
        self.record_policy_action(action_norm, normalized=True)
        action = self.denormalize_action(action_norm)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        image: np.ndarray | torch.Tensor | None = None,
        camera_name: str = "forward",
    ) -> np.ndarray:
        """Run one forward pass. Returns the (denormalized) action."""
        self.update_world_model(pos=pos, vel=vel, quat=quat, omega=omega)
        return self.predict_from_world_model(image=image, camera_name=camera_name)

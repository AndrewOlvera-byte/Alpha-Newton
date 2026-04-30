"""Flightmare state-only racing environments for PPO.

These wrappers expose a Gymnasium-like vector API without requiring ROS or
Unity rendering. Observations match the state-only BC policy input:

    state = normalized([pos, vel, quat, omega] ++ mission)

Actions passed to ``step`` are normalized policy actions. The env denormalizes
them with the collector ``norm_stats.npz`` and applies either direct CTBR or
waypoint-LQR command generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from scripts.flightmare_bc.collect import build_state, sample_gate_course, waypoints_from_gates
from scripts.flightmare_bc.controllers import QuadParams, quat_to_R
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from scripts.flightmare_bc.mission import LOOKAHEAD_GATES, MISSION_DIM, encode_mission, gates_to_views
from src.core.registry import register
from src.robotics.flightmare_autonomy_fsw.controllers import (
    BaseCTBRController,
    WaypointLQRController,
)
from src.robotics.flightmare_autonomy_fsw.gates import StrictGateTracker, gate_frame
from src.robotics.flightmare_autonomy_fsw.messages import PlannerOutput, VehicleState
from src.robotics.flightmare_autonomy_fsw.nodes import CourseConfig
from src.robotics.rewards import get_reward_function


@dataclass
class FlightmareNormStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    mission_mean: np.ndarray | None
    mission_std: np.ndarray | None
    action_mean: np.ndarray
    action_std: np.ndarray
    include_mission: bool = True
    normalize_state: bool = True
    normalize_mission: bool = True
    normalize_action: bool = True

    @classmethod
    def load(
        cls,
        path: str | Path,
        action_type: str,
        include_mission: bool = True,
        normalize_state: bool = True,
        normalize_mission: bool = True,
        normalize_action: bool = True,
    ) -> "FlightmareNormStats":
        stats = np.load(path)
        mission_mean = stats["mission_mean"].astype(np.float32) if "mission_mean" in stats.files else None
        mission_std = stats["mission_std"].astype(np.float32) if "mission_std" in stats.files else None
        return cls(
            state_mean=stats["state_mean"].astype(np.float32),
            state_std=stats["state_std"].astype(np.float32),
            mission_mean=mission_mean,
            mission_std=mission_std,
            action_mean=stats[f"{action_type}_mean"].astype(np.float32),
            action_std=stats[f"{action_type}_std"].astype(np.float32),
            include_mission=include_mission,
            normalize_state=normalize_state,
            normalize_mission=normalize_mission and include_mission,
            normalize_action=normalize_action,
        )

    @property
    def state_dim(self) -> int:
        return int(self.state_mean.shape[0] + (MISSION_DIM if self.include_mission else 0))

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.shape[0])

    def normalize_observation(self, drone_state: np.ndarray, mission: np.ndarray) -> np.ndarray:
        state = np.asarray(drone_state, dtype=np.float32)
        if self.normalize_state:
            state = (state - self.state_mean) / self.state_std
        if self.include_mission:
            mission = np.asarray(mission, dtype=np.float32)
            if self.normalize_mission and self.mission_mean is not None and self.mission_std is not None:
                mission = (mission - self.mission_mean) / self.mission_std
            state = np.concatenate([state, mission], axis=0)
        return state.astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.normalize_action:
            action = action * self.action_std + self.action_mean
        return action.astype(np.float32)


@dataclass
class FlightmareRacingEnvConfig:
    data_dir: str = "data/flightmare/bc_v1"
    action_type: str = "ctbr"
    include_mission: bool = True
    normalize_state: bool = True
    normalize_mission: bool = True
    normalize_action: bool = True
    control_hz: float = 100.0
    horizon: int = 1500
    scene: str = "industrial"
    render: bool = False
    backend: str = "auto"
    image_size: int = 224
    seed: int = 0
    course: CourseConfig = field(default_factory=CourseConfig)
    gate_vehicle_radius: float = 0.15
    max_world_radius: float = 120.0
    min_z: float = -0.25
    terminate_on_gate_miss: bool = True
    terminate_on_crash: bool = True
    reward_fn: str = "flightmare_racing_v1"
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    max_body_rate: float = 8.0
    max_waypoint_speed: float = 15.0


class FlightmareRacingEnv(gymnasium.Env):
    """State-only Flightmare racing env with strict gate-aperture validation."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: FlightmareRacingEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.params = QuadParams()
        self.dt = 1.0 / float(cfg.control_hz)
        self.rng = np.random.default_rng(cfg.seed)
        self.norm = FlightmareNormStats.load(
            Path(cfg.data_dir) / "norm_stats.npz",
            action_type=cfg.action_type,
            include_mission=cfg.include_mission,
            normalize_state=cfg.normalize_state,
            normalize_mission=cfg.normalize_mission,
            normalize_action=cfg.normalize_action,
        )
        self.reward_fn = get_reward_function(cfg.reward_fn)
        self.env = FlightmareExpertEnv(
            image_size=cfg.image_size,
            cameras=(),
            control_hz=cfg.control_hz,
            params=self.params,
            scene=cfg.scene,
            render=cfg.render,
            seed=cfg.seed,
            visual_backend=True,
            backend=cfg.backend,
        )
        self.ctbr_controller = BaseCTBRController(params=self.params, max_body_rate=cfg.max_body_rate)
        self.waypoint_controller = WaypointLQRController(params=self.params, max_speed=cfg.max_waypoint_speed)

        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(self.norm.state_dim,), dtype=np.float32),
            "images": spaces.Dict({}),
        })
        self.action_space = spaces.Box(-5.0, 5.0, shape=(self.norm.action_dim,), dtype=np.float32)

        self._step_count = 0
        self._episode_id = 0
        self._obs = None
        self._gates = []
        self._gate_views = []
        self._strict_tracker: StrictGateTracker | None = None
        self._prev_raw_action = np.zeros(self.norm.action_dim, dtype=np.float32)
        self._prev_pos = None

    @property
    def using_fallback(self) -> bool:
        return bool(self.env.using_fallback)

    def _sample_course(self):
        gates = sample_gate_course(self.rng, self.cfg.course)
        for gate in gates:
            gate.gate_id = f"ppo_ep{self._episode_id:08d}_{gate.gate_id}"
        waypoints = waypoints_from_gates(gates, self.cfg.course.z_min)
        return gates, waypoints

    def _mission(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        current = 0 if self._strict_tracker is None else self._strict_tracker.current_index
        return encode_mission(pos, quat, self._gate_views, current, LOOKAHEAD_GATES)

    def _make_obs(self) -> dict:
        mission = self._mission(self._obs.pos, self._obs.quat)
        state = build_state(self._obs.pos, self._obs.vel, self._obs.quat, self._obs.omega)
        return {
            "state": self.norm.normalize_observation(state, mission),
            "images": {},
        }

    def _vehicle_state(self) -> VehicleState:
        return VehicleState(
            t=self._step_count * self.dt,
            step=self._step_count,
            pos=self._obs.pos.astype(np.float32),
            vel=self._obs.vel.astype(np.float32),
            quat=self._obs.quat.astype(np.float32),
            omega=self._obs.omega.astype(np.float32),
            done=bool(self._obs.done),
        )

    def _apply_action(self, raw_action: np.ndarray):
        if self.cfg.action_type == "ctbr":
            planner = PlannerOutput(
                t=self._step_count * self.dt,
                step=self._step_count,
                action_type="ctbr",
                action=raw_action,
                current_gate_index=self._strict_tracker.current_index,
            )
            command = self.ctbr_controller.compute(planner)
        elif self.cfg.action_type == "waypoint":
            planner = PlannerOutput(
                t=self._step_count * self.dt,
                step=self._step_count,
                action_type="waypoint",
                action=raw_action,
                current_gate_index=self._strict_tracker.current_index,
            )
            command = self.waypoint_controller.compute(self._vehicle_state(), planner)
        else:
            raise ValueError(f"Unsupported Flightmare action_type={self.cfg.action_type!r}")
        return self.env.step_ctbr(command.thrust_newton, command.body_rates), command

    def _crash_reason(self) -> str | None:
        if self._obs.done:
            return "env_done"
        arrays = (self._obs.pos, self._obs.vel, self._obs.quat, self._obs.omega)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            return "nonfinite_state"
        if float(np.linalg.norm(self._obs.pos)) > self.cfg.max_world_radius:
            return "out_of_bounds"
        if float(self._obs.pos[2]) < self.cfg.min_z:
            return "ground"
        return None

    def _target_center_before_step(self) -> np.ndarray:
        if not self._gates:
            return np.zeros(3, dtype=np.float64)
        idx = min(self._strict_tracker.current_index, len(self._gates) - 1)
        center, _, _, _ = gate_frame(self._gates[idx])
        return center

    def _alignment_to_gate(self, pos: np.ndarray, quat: np.ndarray, target_center: np.ndarray) -> float:
        to_gate = target_center - np.asarray(pos, dtype=np.float64)
        n = float(np.linalg.norm(to_gate))
        if n < 1e-6:
            return 1.0
        body_forward = quat_to_R(quat)[:, 0]
        return float(np.clip(np.dot(body_forward, to_gate / n), 0.0, 1.0))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._gates, waypoints = self._sample_course()
        self.env.add_gates(self._gates)
        yaw = float(self._gates[0].yaw) if self._gates else 0.0
        self._obs = self.env.reset(init_pos=waypoints[0], yaw=yaw)
        self._gate_views = gates_to_views(self._gates)
        self._strict_tracker = StrictGateTracker(self._gates, vehicle_radius=self.cfg.gate_vehicle_radius)
        self._prev_raw_action = np.zeros(self.norm.action_dim, dtype=np.float32)
        self._prev_pos = self._obs.pos.copy()
        self._step_count = 0
        self._episode_id += 1
        return self._make_obs(), {
            "using_fallback": self.using_fallback,
            "num_gates": len(self._gates),
        }

    def step(self, action: np.ndarray):
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        action_norm = np.clip(np.asarray(action, dtype=np.float32), -5.0, 5.0)
        raw_action = self.norm.denormalize_action(action_norm)
        target_center = self._target_center_before_step()
        prev_pos = self._obs.pos.copy()
        prev_dist = float(np.linalg.norm(prev_pos - target_center))
        segment_progress = self._strict_tracker.segment_progress(prev_pos, prev_pos)
        alignment = self._alignment_to_gate(prev_pos, self._obs.quat, target_center)

        self._obs, command = self._apply_action(raw_action)
        self._step_count += 1

        curr_pos = self._obs.pos.copy()
        curr_dist_to_same_gate = float(np.linalg.norm(curr_pos - target_center))
        distance_progress = prev_dist - curr_dist_to_same_gate
        # Recompute segment progress over the actual movement and pre-step target.
        if self._gates and not self._strict_tracker.completed:
            idx = self._strict_tracker.current_index
            curr_center = target_center
            prev_center = prev_pos if idx == 0 else gate_frame(self._gates[idx - 1])[0]
            seg = curr_center - prev_center
            denom = float(np.linalg.norm(seg))
            segment_progress = 0.0 if denom < 1e-6 else float(np.dot(curr_pos - prev_pos, seg / denom))

        event = self._strict_tracker.update(prev_pos, curr_pos)
        crash_reason = self._crash_reason()
        gate_missed = bool(event is not None and event.missed)
        gate_passed = bool(event is not None and event.passed)
        success = bool(self._strict_tracker.completed)
        crash = crash_reason is not None

        terminated = success
        if gate_missed and self.cfg.terminate_on_gate_miss:
            terminated = True
        if crash and self.cfg.terminate_on_crash:
            terminated = True
        truncated = bool(self._step_count >= self.cfg.horizon and not terminated)

        info = {
            "success": success,
            "gate_passed": gate_passed,
            "gate_missed": gate_missed,
            "gate_margin_m": float(event.clearance_margin_m) if event is not None else 0.0,
            "gates_completed": int(self._strict_tracker.current_index),
            "num_gates": len(self._gates),
            "gate_completion": float(self._strict_tracker.current_index / max(1, len(self._gates))),
            "distance_progress_m": float(distance_progress),
            "segment_progress_m": float(segment_progress),
            "gate_alignment": float(alignment),
            "crash": crash,
            "crash_reason": crash_reason,
            "raw_action": raw_action.astype(np.float32),
            "action_norm": action_norm.astype(np.float32),
            "ctbr_command": command.ctbr.astype(np.float32),
            "step": self._step_count,
            "dt": self.dt,
            "speed_mps": float(np.linalg.norm(self._obs.vel)),
        }
        reward = self.reward_fn(
            info=info,
            action=raw_action,
            prev_action=self._prev_raw_action,
            dt=self.dt,
            **self.cfg.reward_kwargs,
        )
        self._prev_raw_action = raw_action.astype(np.float32)
        obs = self._make_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()


class FlightmareVecEnv:
    """Synchronous vector env for Flightmare state-only PPO."""

    def __init__(self, envs: list[FlightmareRacingEnv]):
        self.envs = envs
        self.n_envs = len(envs)

    def reset(self, *, seed: Optional[int] = None):
        obs_infos = [
            env.reset(seed=None if seed is None else seed + i)
            for i, env in enumerate(self.envs)
        ]
        return [x[0] for x in obs_infos], [x[1] for x in obs_infos]

    def reset_at(self, indices: list[int], *, seed: Optional[int] = None):
        obs_infos = [
            self.envs[i].reset(seed=None if seed is None else seed + i)
            for i in indices
        ]
        return [x[0] for x in obs_infos], [x[1] for x in obs_infos]

    def step(self, actions: np.ndarray):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs = [r[0] for r in results]
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        terms = np.asarray([r[2] for r in results], dtype=bool)
        truncs = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, rewards, terms, truncs, infos

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


def build_flightmare_env_config(**kwargs) -> FlightmareRacingEnvConfig:
    course_kwargs = kwargs.pop("course", {}) or {}
    if "gate_spacing_range" in kwargs:
        course_kwargs["gate_spacing_range"] = kwargs.pop("gate_spacing_range")
    if "gate_lateral_jitter" in kwargs:
        course_kwargs["gate_lateral_jitter"] = kwargs.pop("gate_lateral_jitter")
    if "gate_z_range" in kwargs:
        course_kwargs["gate_z_range"] = kwargs.pop("gate_z_range")
    if "gate_yaw_step" in kwargs:
        course_kwargs["gate_yaw_step"] = kwargs.pop("gate_yaw_step")
    if "gate_yaw_noise" in kwargs:
        course_kwargs["gate_yaw_noise"] = kwargs.pop("gate_yaw_noise")
    if "gate_size" in kwargs:
        course_kwargs["gate_size"] = kwargs.pop("gate_size")
    if "num_gates" in kwargs:
        course_kwargs["num_gates"] = kwargs.pop("num_gates")
    if "z_min" in kwargs:
        course_kwargs["z_min"] = kwargs.pop("z_min")
    course = CourseConfig(**course_kwargs)
    return FlightmareRacingEnvConfig(course=course, **kwargs)


def make_flightmare_vec_env(n_envs: int, seed: int = 0, **kwargs) -> FlightmareVecEnv:
    envs = []
    for i in range(int(n_envs)):
        cfg = build_flightmare_env_config(**{**kwargs, "seed": seed + i})
        envs.append(FlightmareRacingEnv(cfg))
    return FlightmareVecEnv(envs)


@register("env", "flightmare_racing")
def build_registered_flightmare_vec_env(**kwargs):
    kwargs.pop("type", None)
    n_envs = int(kwargs.pop("n_envs", 1))
    seed = int(kwargs.pop("seed", 0))
    return make_flightmare_vec_env(n_envs=n_envs, seed=seed, **kwargs)

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
from scripts.flightmare_bc.action_norms import action_bounds, stats_mode
from scripts.flightmare_bc.controllers import QuadParams, quat_to_R
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from scripts.flightmare_bc.mission import LOOKAHEAD_GATES, MISSION_DIM, encode_mission, gates_to_views
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
from src.core.registry import register
from src.robotics.flightmare_autonomy_fsw.controllers import (
    BaseCTBRController,
    BaseMotorController,
    WaypointLQRController,
)
from src.robotics.flightmare_autonomy_fsw.gates import (
    StrictGateTracker,
    gate_frame,
    gate_half_extents,
    gate_offsets,
    signed_gate_distance,
)
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
    action_low: np.ndarray | None = None
    action_high: np.ndarray | None = None
    action_normalization: str = "standard"
    include_mission: bool = True
    normalize_state: bool = True
    normalize_mission: bool = True
    normalize_action: bool = True
    # --- obs v3 (Swift split-then-fuse) ---
    obs_schema: str = "v2"  # "v2" or "v3"
    proprio_mean: np.ndarray | None = None
    proprio_std: np.ndarray | None = None
    gate_mean: np.ndarray | None = None
    gate_std: np.ndarray | None = None
    aux_mean: np.ndarray | None = None
    aux_std: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        action_type: str,
        include_mission: bool = True,
        normalize_state: bool = True,
        normalize_mission: bool = True,
        normalize_action: bool = True,
        action_normalization: str = "auto",
        obs_schema: str = "v2",
    ) -> "FlightmareNormStats":
        stats = np.load(path)
        mission_mean = stats["mission_mean"].astype(np.float32) if "mission_mean" in stats.files else None
        mission_std = stats["mission_std"].astype(np.float32) if "mission_std" in stats.files else None
        mode = stats_mode(stats, default="standard") if action_normalization == "auto" else str(action_normalization)
        low, high = action_bounds(action_type, stats)

        v3_kwargs: dict[str, Any] = {}
        if obs_schema == "v3":
            for needed in ("proprio_core_mean", "gate_mean", "aux_mean"):
                if needed not in stats.files:
                    raise RuntimeError(
                        f"{path} missing v3 stats {needed!r} — rerun "
                        f"`python -m scripts.flightmare_bc.transform_to_v3`."
                    )
            v3_kwargs = dict(
                proprio_mean=stats["proprio_core_mean"].astype(np.float32),
                proprio_std=stats["proprio_core_std"].astype(np.float32),
                gate_mean=stats["gate_mean"].astype(np.float32),
                gate_std=stats["gate_std"].astype(np.float32),
                aux_mean=stats["aux_mean"].astype(np.float32),
                aux_std=stats["aux_std"].astype(np.float32),
            )

        return cls(
            state_mean=stats["state_mean"].astype(np.float32),
            state_std=stats["state_std"].astype(np.float32),
            mission_mean=mission_mean,
            mission_std=mission_std,
            action_mean=stats[f"{action_type}_mean"].astype(np.float32),
            action_std=stats[f"{action_type}_std"].astype(np.float32),
            action_low=low,
            action_high=high,
            action_normalization=mode,
            include_mission=include_mission,
            normalize_state=normalize_state,
            normalize_mission=normalize_mission and include_mission,
            normalize_action=normalize_action,
            obs_schema=obs_schema,
            **v3_kwargs,
        )

    @property
    def state_dim(self) -> int:
        if self.obs_schema == "v3":
            return PROPRIO_CORE_DIM + GATE_DIM + AUX_DIM
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

    def normalize_observation_v3(
        self, proprio_core: np.ndarray, gate: np.ndarray, aux: np.ndarray
    ) -> np.ndarray:
        """Normalize and concat the three v3 blocks into a 36-d state vector."""
        if self.normalize_state:
            proprio_core = (proprio_core - self.proprio_mean) / self.proprio_std
            gate = (gate - self.gate_mean) / self.gate_std
            aux = (aux - self.aux_mean) / self.aux_std
        return np.concatenate([proprio_core, gate, aux], axis=0).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.normalize_action:
            if self.action_normalization == "bounds":
                low = self.action_low
                high = self.action_high
                if low is None or high is None:
                    raise RuntimeError("Bounds action normalization selected but action bounds are missing.")
                action = low + 0.5 * (action + 1.0) * np.maximum(high - low, 1e-6)
            else:
                action = action * self.action_std + self.action_mean
        return action.astype(np.float32)

    def normalize_action_value(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.normalize_action:
            if self.action_normalization == "bounds":
                low = self.action_low
                high = self.action_high
                if low is None or high is None:
                    raise RuntimeError("Bounds action normalization selected but action bounds are missing.")
                action = 2.0 * (action - low) / np.maximum(high - low, 1e-6) - 1.0
            else:
                action = (action - self.action_mean) / self.action_std
        return action.astype(np.float32)


@dataclass
class FlightmareRacingEnvConfig:
    data_dir: str = "data/flightmare/bc_v1"
    action_type: str = "ctbr"
    obs_schema: str = "v2"  # "v2" (state+mission, 33-d) or "v3" (Swift split-then-fuse, 36-d)
    include_mission: bool = True
    normalize_state: bool = True
    normalize_mission: bool = True
    normalize_action: bool = True
    action_normalization: str = "auto"
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
    max_collective_thrust_g: float = 4.0
    action_clip: float = 5.0
    # Plant overrides: forwarded to QuadParams + the Flightmare C++ dynamics
    # YAML. Keys (all optional, default None = keep historical Flightmare
    # values): mass, arm_length, inertia, k_thrust, k_torque, motor_omega_min,
    # motor_omega_max, motor_tau, thrust_map, kappa, omega_max_body.
    plant: dict[str, Any] | None = None
    # Override the action bounds loaded from norm_stats. Useful when the BC
    # data was collected at conservative dynamics but the PPO env runs an
    # aggressive racing plant. Schema: {"ctbr": {"low": [...], "high": [...]},
    # "motor": {...}, "waypoint": {...}}.
    action_bounds_override: dict[str, Any] | None = None
    reset_mode: str = "course_start"  # course_start|gate_pre_entry|reference_state|segment_window
    start_gate_index: int | None = None
    start_gate_choices: list[int] | None = None
    start_offset_m: float = 3.0
    start_offset_range: tuple[float, float] | None = None
    reference_speed_mps: float = 4.0
    goal_mode: str = "finish_remaining_course"  # pass_gate_i|gate_i_to_i+1|gate_i_to_i+2|finish_remaining_course
    goal_gate_span: int | None = None


class FlightmareRacingEnv(gymnasium.Env):
    """State-only Flightmare racing env with strict gate-aperture validation."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: FlightmareRacingEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.params = QuadParams()
        if cfg.plant:
            _PLANT_KEYS = (
                "mass", "arm_length", "k_thrust", "k_torque", "inertia",
                "motor_omega_min", "motor_omega_max", "motor_tau",
                "thrust_map", "kappa", "omega_max_body",
            )
            for key in _PLANT_KEYS:
                if key in cfg.plant and cfg.plant[key] is not None:
                    value = cfg.plant[key]
                    if key in ("inertia", "thrust_map", "omega_max_body"):
                        value = tuple(float(x) for x in value)
                    else:
                        value = float(value)
                    setattr(self.params, key, value)
        self.params.max_collective_thrust = float(cfg.max_collective_thrust_g) * self.params.mass * self.params.g
        self.dt = 1.0 / float(cfg.control_hz)
        self.rng = np.random.default_rng(cfg.seed)
        self.norm = FlightmareNormStats.load(
            Path(cfg.data_dir) / "norm_stats.npz",
            action_type=cfg.action_type,
            include_mission=cfg.include_mission,
            normalize_state=cfg.normalize_state,
            normalize_mission=cfg.normalize_mission,
            normalize_action=cfg.normalize_action,
            action_normalization=cfg.action_normalization,
            obs_schema=cfg.obs_schema,
        )
        if cfg.action_bounds_override:
            bounds = cfg.action_bounds_override.get(cfg.action_type)
            if bounds:
                low = np.asarray(bounds["low"], dtype=np.float32)
                high = np.asarray(bounds["high"], dtype=np.float32)
                if low.shape != self.norm.action_low.shape or high.shape != self.norm.action_high.shape:
                    raise ValueError(
                        f"action_bounds_override[{cfg.action_type}] shape mismatch: "
                        f"got low={low.shape} high={high.shape}, expected {self.norm.action_low.shape}"
                    )
                self.norm.action_low = low
                self.norm.action_high = high
        self._gate_specs: list[GateSpec] = []
        self._gate_index_v3 = 0
        self._last_pass_step = 0
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
        self.motor_controller = BaseMotorController(params=self.params)

        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(self.norm.state_dim,), dtype=np.float32),
            "images": spaces.Dict({}),
        })
        self.action_space = spaces.Box(
            -float(cfg.action_clip),
            float(cfg.action_clip),
            shape=(self.norm.action_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._episode_id = 0
        self._obs = None
        self._gates = []
        self._gate_views = []
        self._strict_tracker: StrictGateTracker | None = None
        self._prev_raw_action = np.zeros(self.norm.action_dim, dtype=np.float32)
        self._prev_pos = None
        self._start_gate_index = 0
        self._goal_terminal_gate_index = 0

    @property
    def using_fallback(self) -> bool:
        return bool(self.env.using_fallback)

    def _sample_course(self):
        gates = sample_gate_course(self.rng, self.cfg.course)
        for gate in gates:
            gate.gate_id = f"ppo_ep{self._episode_id:08d}_{gate.gate_id}"
        waypoints = waypoints_from_gates(
            gates,
            self.cfg.course.z_min,
            d_approach=float(self.cfg.course.gate_approach_m),
        )
        return gates, waypoints

    def _mission(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        current = 0 if self._strict_tracker is None else self._strict_tracker.current_index
        return encode_mission(pos, quat, self._gate_views, current, LOOKAHEAD_GATES)

    def _make_obs(self) -> dict:
        if self.cfg.obs_schema == "v3":
            current = 0 if self._strict_tracker is None else self._strict_tracker.current_index
            # Update time_since_pass tracker on gate-index advance.
            if current != self._gate_index_v3:
                self._last_pass_step = self._step_count
                self._gate_index_v3 = current
            time_since_pass = float(self._step_count - self._last_pass_step) * self.dt
            proprio = build_proprio_core(self._obs.vel, self._obs.quat, self._obs.omega)
            gate = encode_gate_corners(
                self._obs.pos, self._obs.quat, self._gate_specs, current, LOOKAHEAD_GATES_V3
            )
            aux = encode_aux(self._obs.pos, self._gate_specs, current, time_since_pass)
            state = self.norm.normalize_observation_v3(proprio, gate, aux)
            return {"state": state, "images": {}}
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
        elif self.cfg.action_type == "motor":
            planner = PlannerOutput(
                t=self._step_count * self.dt,
                step=self._step_count,
                action_type="motor",
                action=raw_action,
                current_gate_index=self._strict_tracker.current_index,
            )
            command = self.motor_controller.compute(planner)
        else:
            raise ValueError(f"Unsupported Flightmare action_type={self.cfg.action_type!r}")
        if command.source_action_type == "motor":
            return self.env.step_motor(command.motor), command
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

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)

    def _select_start_gate_index(self) -> int:
        n = len(self._gates)
        if n <= 0:
            return 0
        choices = self.cfg.start_gate_choices
        if choices:
            valid = [int(i) for i in choices if 0 <= int(i) < n]
            if valid:
                return int(valid[int(self.rng.integers(0, len(valid)))])
        if self.cfg.start_gate_index is not None:
            return int(np.clip(int(self.cfg.start_gate_index), 0, n - 1))
        if self.cfg.reset_mode in {"gate_pre_entry", "reference_state", "segment_window"}:
            return int(self.rng.integers(0, n))
        return 0

    def _start_offset(self) -> float:
        rng = self.cfg.start_offset_range
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
            if hi < lo:
                lo, hi = hi, lo
            return float(self.rng.uniform(lo, hi))
        return float(self.cfg.start_offset_m)

    def _goal_terminal_index(self, start_gate: int) -> int:
        n = len(self._gates)
        if n <= 0:
            return 0
        if self.cfg.goal_gate_span is not None:
            return int(np.clip(start_gate + int(self.cfg.goal_gate_span), start_gate + 1, n))
        mode = str(self.cfg.goal_mode)
        if mode == "pass_gate_i":
            return min(n, start_gate + 1)
        if mode == "gate_i_to_i+1":
            return min(n, start_gate + 2)
        if mode == "gate_i_to_i+2":
            return min(n, start_gate + 3)
        if mode == "finish_remaining_course":
            return n
        raise ValueError(f"Unsupported goal_mode={self.cfg.goal_mode!r}")

    def _reference_start_state(self, start_gate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gate = self._gates[start_gate]
        center, forward, _, _ = gate_frame(gate)
        offset = self._start_offset()
        pos = center - offset * forward
        yaw = float(getattr(gate, "yaw", np.arctan2(forward[1], forward[0])))
        quat = self._quat_from_yaw(yaw)
        vel = float(self.cfg.reference_speed_mps) * forward
        omega = np.zeros(3, dtype=np.float64)
        return pos.astype(np.float64), vel.astype(np.float64), quat.astype(np.float64), omega

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._gates, waypoints = self._sample_course()
        self.env.add_gates(self._gates)
        self._gate_views = gates_to_views(self._gates)
        self._strict_tracker = StrictGateTracker(self._gates, vehicle_radius=self.cfg.gate_vehicle_radius)
        self._gate_specs = [
            GateSpec(
                pos=np.asarray(g.pos, dtype=np.float64),
                yaw=float(getattr(g, "yaw", 0.0)),
                size=np.asarray(getattr(g, "size", (1.6, 1.6, 1.6)), dtype=np.float64),
                quat=(np.asarray(g.quat, dtype=np.float64) if getattr(g, "quat", None) is not None else None),
            )
            for g in self._gates
        ]
        self._start_gate_index = self._select_start_gate_index()
        self._goal_terminal_gate_index = self._goal_terminal_index(self._start_gate_index)
        reset_mode = str(self.cfg.reset_mode)
        if reset_mode == "course_start" or not self._gates:
            yaw = float(self._gates[0].yaw) if self._gates else 0.0
            self._obs = self.env.reset(init_pos=waypoints[0], yaw=yaw)
            self._start_gate_index = 0
            self._goal_terminal_gate_index = self._goal_terminal_index(0)
        elif reset_mode in {"gate_pre_entry", "reference_state", "segment_window"}:
            pos, vel, quat, omega = self._reference_start_state(self._start_gate_index)
            yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
            self._obs = self.env.reset(init_pos=pos, yaw=yaw)
            self._obs = self.env.set_state(pos=pos, quat=quat, vel=vel, omega=omega)
            self._strict_tracker.current_index = int(self._start_gate_index)
        else:
            raise ValueError(f"Unsupported reset_mode={self.cfg.reset_mode!r}")
        self._gate_index_v3 = int(self._start_gate_index)
        self._last_pass_step = 0
        self._prev_raw_action = np.zeros(self.norm.action_dim, dtype=np.float32)
        self._prev_pos = self._obs.pos.copy()
        self._step_count = 0
        self._episode_id += 1
        initial_prev_action = self.norm.normalize_action_value(
            np.zeros(self.norm.action_dim, dtype=np.float32)
        )
        return self._make_obs(), {
            "using_fallback": self.using_fallback,
            "num_gates": len(self._gates),
            "reset_mode": reset_mode,
            "start_gate_index": int(self._start_gate_index),
            "goal_terminal_gate_index": int(self._goal_terminal_gate_index),
            "goal_mode": str(self.cfg.goal_mode),
            "initial_prev_action_norm": initial_prev_action,
        }

    def step(self, action: np.ndarray):
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        action_norm = np.clip(
            np.asarray(action, dtype=np.float32),
            -float(self.cfg.action_clip),
            float(self.cfg.action_clip),
        )
        raw_action = self.norm.denormalize_action(action_norm)
        target_center = self._target_center_before_step()
        prev_pos = self._obs.pos.copy()
        prev_dist = float(np.linalg.norm(prev_pos - target_center))
        segment_progress = self._strict_tracker.segment_progress(prev_pos, prev_pos)
        alignment = self._alignment_to_gate(prev_pos, self._obs.quat, target_center)
        gate_signed_distance = 0.0
        gate_normal_progress = 0.0
        gate_normal_velocity = 0.0
        gate_lateral_norm = 0.0
        gate_vertical_norm = 0.0
        gate_lateral_norm_next = 0.0
        gate_vertical_norm_next = 0.0
        gate_center_error_norm = 0.0
        gate_center_error_norm_next = 0.0
        gate_center_error_progress = 0.0
        gate_signed_distance_next = 0.0
        target_gate = None
        if self._gates and not self._strict_tracker.completed:
            idx = self._strict_tracker.current_index
            target_gate = self._gates[idx]
            gate_signed_distance = signed_gate_distance(prev_pos, target_gate)
            lateral_m, vertical_m = gate_offsets(prev_pos, target_gate)
            half_w, half_h = gate_half_extents(target_gate, vehicle_radius=self.cfg.gate_vehicle_radius)
            gate_lateral_norm = float(lateral_m / max(half_w, 1e-6))
            gate_vertical_norm = float(vertical_m / max(half_h, 1e-6))
            gate_center_error_norm = float(np.hypot(gate_lateral_norm, gate_vertical_norm))

        self._obs, command = self._apply_action(raw_action)
        self._step_count += 1

        curr_pos = self._obs.pos.copy()
        curr_dist_to_same_gate = float(np.linalg.norm(curr_pos - target_center))
        distance_progress = prev_dist - curr_dist_to_same_gate
        if target_gate is not None:
            curr_signed = signed_gate_distance(curr_pos, target_gate)
            gate_normal_progress = float(curr_signed - gate_signed_distance)
            gate_signed_distance_next = float(curr_signed)
            _, gate_forward, _, _ = gate_frame(target_gate)
            gate_normal_velocity = float(np.dot(self._obs.vel, gate_forward))
            lateral_m, vertical_m = gate_offsets(curr_pos, target_gate)
            half_w, half_h = gate_half_extents(target_gate, vehicle_radius=self.cfg.gate_vehicle_radius)
            gate_lateral_norm_next = float(lateral_m / max(half_w, 1e-6))
            gate_vertical_norm_next = float(vertical_m / max(half_h, 1e-6))
            gate_center_error_norm_next = float(np.hypot(gate_lateral_norm_next, gate_vertical_norm_next))
            gate_center_error_progress = float(gate_center_error_norm - gate_center_error_norm_next)
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
        success = bool(self._strict_tracker.current_index >= self._goal_terminal_gate_index)
        crash = crash_reason is not None
        body_z_world = quat_to_R(self._obs.quat)[:, 2]

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
            "goal_completion": float(
                (self._strict_tracker.current_index - self._start_gate_index)
                / max(1, self._goal_terminal_gate_index - self._start_gate_index)
            ),
            "start_gate_index": int(self._start_gate_index),
            "goal_terminal_gate_index": int(self._goal_terminal_gate_index),
            "goal_mode": str(self.cfg.goal_mode),
            "reset_mode": str(self.cfg.reset_mode),
            "distance_progress_m": float(distance_progress),
            "segment_progress_m": float(segment_progress),
            "gate_alignment": float(alignment),
            "gate_signed_distance_m": float(gate_signed_distance),
            "gate_signed_distance_next_m": float(gate_signed_distance_next),
            "gate_normal_progress_m": float(gate_normal_progress),
            "gate_normal_velocity_mps": float(gate_normal_velocity),
            "gate_lateral_norm": float(gate_lateral_norm),
            "gate_vertical_norm": float(gate_vertical_norm),
            "gate_lateral_norm_next": float(gate_lateral_norm_next),
            "gate_vertical_norm_next": float(gate_vertical_norm_next),
            "gate_center_error_norm": float(gate_center_error_norm),
            "gate_center_error_norm_next": float(gate_center_error_norm_next),
            "gate_center_error_progress": float(gate_center_error_progress),
            "crash": crash,
            "crash_reason": crash_reason,
            "raw_action": raw_action.astype(np.float32),
            "action_norm": action_norm.astype(np.float32),
            "source_action_type": command.source_action_type,
            "ctbr_command": command.ctbr.astype(np.float32),
            "motor_command": command.motor.astype(np.float32),
            "quat": self._obs.quat.astype(np.float32),
            "omega": self._obs.omega.astype(np.float32),
            "body_z_world_z": float(body_z_world[2]),
            "vertical_speed_mps": float(self._obs.vel[2]),
            "angular_rate_norm": float(np.linalg.norm(self._obs.omega)),
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
    if "course_mode" in kwargs:
        course_kwargs["course_mode"] = kwargs.pop("course_mode")
    if "gate_layout" in kwargs:
        course_kwargs["gate_layout"] = kwargs.pop("gate_layout")
    if "random_start_gate" in kwargs:
        course_kwargs["random_start_gate"] = kwargs.pop("random_start_gate")
    if "fixed_gate_pos_noise" in kwargs:
        course_kwargs["fixed_gate_pos_noise"] = kwargs.pop("fixed_gate_pos_noise")
    if "fixed_gate_pos_noise_xyz" in kwargs:
        course_kwargs["fixed_gate_pos_noise_xyz"] = kwargs.pop("fixed_gate_pos_noise_xyz")
    if "fixed_gate_yaw_noise" in kwargs:
        course_kwargs["fixed_gate_yaw_noise"] = kwargs.pop("fixed_gate_yaw_noise")
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
    if "gate_approach_m" in kwargs:
        course_kwargs["gate_approach_m"] = kwargs.pop("gate_approach_m")
    course = CourseConfig(**course_kwargs)
    return FlightmareRacingEnvConfig(course=course, **kwargs)


def _subproc_worker(remote, env_kwargs_list: list[dict], seeds: list[int], worker_id: int = 0) -> None:
    """Subprocess worker: holds a slice of envs, services step/reset commands.

    Lives at module level so it pickles under the 'spawn' start method.
    Sets ``FLIGHTMARE_PUB_PORT``/``FLIGHTMARE_SUB_PORT`` per worker so each
    flightgym instance binds its own ZMQ port pair (otherwise every worker
    would race for the default 10253/10254 and crash with ``Address already
    in use``). Requires the patched ``unity_bridge.cpp`` that reads these
    env vars.
    """
    import os, sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # Self-contained port allocation (no import dependency). Matches
    # parallel_unity.allocate_ports: base 10253/10254, stride 4 per slot.
    _BASE_PUB, _BASE_SUB, _STRIDE = 10253, 10254, 4
    _ENV_SLOTS_PER_WORKER = 64  # upper bound; gives disjoint port ranges across workers
    envs: list[FlightmareRacingEnv] = []
    for local_i, (kw, seed) in enumerate(zip(env_kwargs_list, seeds)):
        slot = int(worker_id) * _ENV_SLOTS_PER_WORKER + local_i
        pub = _BASE_PUB + slot * _STRIDE
        sub = _BASE_SUB + slot * _STRIDE
        os.environ["FLIGHTMARE_PUB_PORT"] = str(pub)
        os.environ["FLIGHTMARE_SUB_PORT"] = str(sub)
        print(f"[subproc_worker w={worker_id} local={local_i}] FLIGHTMARE_PUB_PORT={pub} SUB_PORT={sub}", flush=True)
        cfg = build_flightmare_env_config(**{**kw, "seed": int(seed)})
        envs.append(FlightmareRacingEnv(cfg))
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "step":
                actions = payload  # ndarray (m, action_dim)
                remote.send([env.step(actions[i]) for i, env in enumerate(envs)])
            elif cmd == "reset":
                seed = payload
                remote.send([
                    env.reset(seed=None if seed is None else int(seed) + i)
                    for i, env in enumerate(envs)
                ])
            elif cmd == "reset_at":
                local_indices, seed = payload
                remote.send([
                    envs[li].reset(seed=None if seed is None else int(seed) + li)
                    for li in local_indices
                ])
            elif cmd == "close":
                for env in envs:
                    try:
                        env.close()
                    except Exception:
                        pass
                remote.close()
                return
            else:
                raise RuntimeError(f"SubprocFlightmareVecEnv: unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass


class SubprocFlightmareVecEnv:
    """Multiprocess vector env with the same API as ``FlightmareVecEnv``.

    Each worker holds a contiguous slice of envs and steps them serially.
    The parent scatters actions, gathers results, and re-orders into the
    canonical [0..n_envs) layout so the rollout collector's episode logic
    (``reset_at(done_indices)``) is unchanged.
    """

    def __init__(self, env_kwargs_list: list[dict], seeds: list[int], n_workers: int):
        import multiprocessing as mp
        n = len(env_kwargs_list)
        # Flightmare/UnityBridge binds a ZMQ port per env construction. The
        # patched bridge reads FLIGHTMARE_PUB_PORT/SUB_PORT once at bridge
        # construction time, so as long as the worker sets these env vars
        # before each FlightmareRacingEnv() call (see _subproc_worker), packing
        # multiple envs in one process is safe and dramatically reduces IPC
        # overhead — a 1-env-per-proc setup spends most of its wall time in
        # the parent's serial recv() loop over remotes, not in physics.
        # Direct profiling: 4 envs sequential in one process hits ~6000 fps,
        # vs ~5000 fps aggregate at 32 procs × 1 env.
        n_workers = max(1, min(int(n_workers) or n, n))
        bins = np.array_split(np.arange(n, dtype=np.int64), n_workers)
        self._chunks: list[list[int]] = [list(map(int, b)) for b in bins]
        self._owner: list[tuple[int, int]] = [(-1, -1)] * n
        for w, idxs in enumerate(self._chunks):
            for li, gi in enumerate(idxs):
                self._owner[gi] = (w, li)
        ctx = mp.get_context("spawn")
        self._procs = []
        self._remotes = []
        for w, idxs in enumerate(self._chunks):
            sub_kwargs = [env_kwargs_list[i] for i in idxs]
            sub_seeds = [seeds[i] for i in idxs]
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_subproc_worker,
                args=(child_conn, sub_kwargs, sub_seeds, w),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._procs.append(p)
            self._remotes.append(parent_conn)
        self.n_envs = n
        self._closed = False

    def reset(self, *, seed: Optional[int] = None):
        for r in self._remotes:
            r.send(("reset", seed))
        all_obs: list = [None] * self.n_envs
        all_info: list = [None] * self.n_envs
        for w, r in enumerate(self._remotes):
            results = r.recv()
            for li, gi in enumerate(self._chunks[w]):
                all_obs[gi] = results[li][0]
                all_info[gi] = results[li][1]
        return all_obs, all_info

    def reset_at(self, indices: list[int], *, seed: Optional[int] = None):
        per_worker_local: list[list[int]] = [[] for _ in self._remotes]
        per_worker_global: list[list[int]] = [[] for _ in self._remotes]
        for gi in indices:
            w, li = self._owner[int(gi)]
            per_worker_local[w].append(li)
            per_worker_global[w].append(int(gi))
        active = [w for w, lst in enumerate(per_worker_local) if lst]
        for w in active:
            self._remotes[w].send(("reset_at", (per_worker_local[w], seed)))
        out_obs: list = [None] * len(indices)
        out_info: list = [None] * len(indices)
        pos = {int(gi): i for i, gi in enumerate(indices)}
        for w in active:
            results = self._remotes[w].recv()
            for k, gi in enumerate(per_worker_global[w]):
                p = pos[gi]
                out_obs[p] = results[k][0]
                out_info[p] = results[k][1]
        return out_obs, out_info

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        for w, r in enumerate(self._remotes):
            r.send(("step", actions[self._chunks[w]]))
        all_obs: list = [None] * self.n_envs
        all_rew = np.zeros(self.n_envs, dtype=np.float32)
        all_term = np.zeros(self.n_envs, dtype=bool)
        all_trunc = np.zeros(self.n_envs, dtype=bool)
        all_info: list = [None] * self.n_envs
        for w, r in enumerate(self._remotes):
            results = r.recv()
            for li, gi in enumerate(self._chunks[w]):
                obs, rew, term, trunc, info = results[li]
                all_obs[gi] = obs
                all_rew[gi] = float(rew)
                all_term[gi] = bool(term)
                all_trunc[gi] = bool(trunc)
                all_info[gi] = info
        return all_obs, all_rew, all_term, all_trunc, all_info

    def close(self):
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        for r in self._remotes:
            try:
                r.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_flightmare_vec_env(
    n_envs: int,
    seed: int = 0,
    n_workers: int = 0,
    **kwargs,
) -> FlightmareVecEnv | SubprocFlightmareVecEnv:
    n_envs = int(n_envs)
    n_workers = int(n_workers or 0)
    base_kwargs = [dict(kwargs) for _ in range(n_envs)]
    seeds = [int(seed) + i for i in range(n_envs)]
    if n_workers <= 1 or n_envs <= 1:
        envs = [
            FlightmareRacingEnv(build_flightmare_env_config(**{**base_kwargs[i], "seed": seeds[i]}))
            for i in range(n_envs)
        ]
        return FlightmareVecEnv(envs)
    return SubprocFlightmareVecEnv(base_kwargs, seeds, n_workers=n_workers)


@register("env", "flightmare_racing")
def build_registered_flightmare_vec_env(**kwargs):
    kwargs.pop("type", None)
    n_envs = int(kwargs.pop("n_envs", 1))
    seed = int(kwargs.pop("seed", 0))
    return make_flightmare_vec_env(n_envs=n_envs, seed=seed, **kwargs)

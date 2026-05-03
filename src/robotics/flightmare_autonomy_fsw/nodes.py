from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from scripts.flightmare_bc.collect import sample_gate_course, waypoints_from_gates
from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.expert_env import FlightmareExpertEnv
from src.robotics.flightmare_autonomy_fsw.controllers import BaseAutonomyController
from src.robotics.flightmare_autonomy_fsw.messages import (
    ActionType,
    ControlCommand,
    GateCourse,
    PlannerOutput,
    VehicleState,
)
from src.robotics.models.flightmare.MissionWrapper import MissionObservation, MissionWrapper


@dataclass
class CourseConfig:
    """Procedural gate-course parameters shared with ``scripts.flightmare_bc.collect``."""

    course_mode: str = "gates"
    num_gates: int = 8
    z_min: float = 1.0
    gate_spacing_range: Sequence[float] = (4.0, 9.0)
    gate_lateral_jitter: float = 2.0
    gate_z_range: Sequence[float] = (1.5, 4.0)
    gate_yaw_step: float = 0.7
    gate_yaw_noise: float = 0.25
    gate_size: float = 1.0
    gate_layout: str | None = None
    random_start_gate: bool = False
    fixed_gate_pos_noise: float = 0.0
    fixed_gate_pos_noise_xyz: Sequence[float] | None = None
    fixed_gate_yaw_noise: float = 0.0
    gate_approach_m: float = 1.2


class FlightmareStateNode:
    """State-estimator/simulator node: owns Flightmare and publishes state."""

    def __init__(
        self,
        control_hz: float = 100.0,
        course_config: CourseConfig | None = None,
        params: QuadParams | None = None,
        scene: str = "industrial",
        render: bool = False,
        seed: int = 0,
        image_size: int = 224,
        backend: str = "auto",
    ):
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz
        self.course_config = course_config or CourseConfig()
        self.params = params or QuadParams()
        self.scene = scene
        self.render = bool(render)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.backend = str(backend)
        self.env = FlightmareExpertEnv(
            image_size=self.image_size,
            cameras=(),
            control_hz=self.control_hz,
            params=self.params,
            scene=self.scene,
            render=self.render,
            seed=self.seed,
            visual_backend=True,
            backend=self.backend,
        )
        self._step = 0
        self._t = 0.0
        self._course: GateCourse | None = None

    @property
    def using_fallback(self) -> bool:
        return bool(self.env.using_fallback)

    @property
    def course(self) -> GateCourse:
        if self._course is None:
            raise RuntimeError("FlightmareStateNode.reset() must be called before accessing course.")
        return self._course

    def reset(self, episode_id: int, rng: np.random.Generator) -> tuple[GateCourse, VehicleState]:
        gates = sample_gate_course(rng, self.course_config)
        for gate in gates:
            gate.gate_id = f"eval_ep{episode_id:06d}_{gate.gate_id}"
        self.env.add_gates(gates)
        waypoints = waypoints_from_gates(
            gates,
            self.course_config.z_min,
            d_approach=float(self.course_config.gate_approach_m),
        )
        yaw = float(gates[0].yaw) if gates else 0.0
        obs = self.env.reset(init_pos=waypoints[0], yaw=yaw)

        self._step = 0
        self._t = 0.0
        self._course = GateCourse(episode_id=int(episode_id), gates=gates, waypoints=waypoints)
        return self._course, VehicleState.from_step_result(obs, t=self._t, step=self._step)

    def step(self, command: ControlCommand) -> VehicleState:
        if command.source_action_type == "motor":
            obs = self.env.step_motor(command.motor)
        else:
            obs = self.env.step_ctbr(command.thrust_newton, command.body_rates)
        self._step += 1
        self._t = self._step * self.dt
        return VehicleState.from_step_result(obs, t=self._t, step=self._step)

    def close(self) -> None:
        self.env.close()


class MissionWorldModelNode:
    """Mission/world-model node backed by ``MissionWrapper``."""

    def __init__(self, mission_wrapper: MissionWrapper):
        self.wrapper = mission_wrapper

    def reset(self, gates) -> None:
        self.wrapper.set_gates(gates)

    def tick(self, state: VehicleState) -> MissionObservation:
        return self.wrapper.update_world_model(
            pos=state.pos,
            vel=state.vel,
            quat=state.quat,
            omega=state.omega,
        )


class PolicyPlannerNode:
    """Planner node: runs the registered Flightmare actor through ``model.predict``."""

    def __init__(
        self,
        mission_wrapper: MissionWrapper,
        action_type: ActionType,
    ):
        if mission_wrapper.policy is None:
            raise ValueError("PolicyPlannerNode requires a MissionWrapper with a policy.")
        self.wrapper = mission_wrapper
        self.policy = mission_wrapper.policy
        self.action_type = action_type

    @torch.no_grad()
    def tick(self, state: VehicleState, mission: MissionObservation) -> PlannerOutput:
        batch = self.wrapper.make_policy_batch()
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
        self.wrapper.record_policy_action(action_norm, normalized=True)
        action = self.wrapper.denormalize_action(action_norm).squeeze(0).cpu().numpy()
        return PlannerOutput(
            t=state.t,
            step=state.step,
            action_type=self.action_type,
            action=np.asarray(action, dtype=np.float32),
            current_gate_index=int(mission.current_gate_index),
        )


class BaseControllerNode:
    """Controller node that turns planner outputs into Flightmare CTBR commands."""

    def __init__(self, controller: BaseAutonomyController | None = None):
        self.controller = controller or BaseAutonomyController()

    def tick(self, state: VehicleState, planner: PlannerOutput) -> ControlCommand:
        return self.controller.compute(state, planner)

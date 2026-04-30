from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.flightmare_bc.controllers import (
    GeometricGains,
    GeometricSE3Controller,
    QuadParams,
    quat_to_R,
)
from src.robotics.flightmare_autonomy_fsw.messages import ControlCommand, PlannerOutput, VehicleState


def _double_integrator_lqr_gain(q_pos: np.ndarray, q_vel: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form continuous-time LQR gains for independent double integrators."""
    q_pos = np.asarray(q_pos, dtype=np.float64)
    q_vel = np.asarray(q_vel, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if np.any(q_pos <= 0.0) or np.any(r <= 0.0):
        raise ValueError("q_pos and r must be positive for LQR gain construction.")
    kp = np.sqrt(q_pos / r)
    kd = np.sqrt((q_vel + 2.0 * np.sqrt(q_pos * r)) / r)
    return kp, kd


@dataclass
class WaypointLQRGains:
    q_pos: tuple[float, float, float] = (12.0, 12.0, 16.0)
    q_vel: tuple[float, float, float] = (3.0, 3.0, 4.0)
    r: tuple[float, float, float] = (1.0, 1.0, 1.0)


class WaypointLQRController:
    """High-level waypoint+speed tracker that emits CTBR commands."""

    def __init__(
        self,
        params: QuadParams | None = None,
        gains: WaypointLQRGains | None = None,
        min_speed: float = 0.0,
        max_speed: float = 15.0,
    ):
        self.params = params or QuadParams()
        self.lqr_gains = gains or WaypointLQRGains()
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        kp, kd = _double_integrator_lqr_gain(
            np.asarray(self.lqr_gains.q_pos),
            np.asarray(self.lqr_gains.q_vel),
            np.asarray(self.lqr_gains.r),
        )
        self.inner = GeometricSE3Controller(
            params=self.params,
            gains=GeometricGains(kp_pos=tuple(kp), kd_pos=tuple(kd), kp_att=(8.0, 8.0, 3.0)),
        )

    def compute(self, state: VehicleState, planner: PlannerOutput) -> ControlCommand:
        action = np.asarray(planner.action, dtype=np.float32)
        delta_body = action[:3].astype(np.float64)
        speed = float(np.clip(action[3], self.min_speed, self.max_speed))

        R = quat_to_R(state.quat)
        target_world = state.pos.astype(np.float64) + R @ delta_body
        to_target = target_world - state.pos.astype(np.float64)
        dist = float(np.linalg.norm(to_target))
        if dist > 1e-6:
            direction = to_target / dist
        else:
            direction = R[:, 0]
        vel_des = direction * speed

        horizontal = direction[:2]
        if np.linalg.norm(horizontal) > 1e-6:
            yaw_des = float(np.arctan2(horizontal[1], horizontal[0]))
        else:
            yaw_des = float(np.arctan2(R[1, 0], R[0, 0]))

        cmd = self.inner.compute(
            pos=state.pos,
            vel=state.vel,
            quat=state.quat,
            pos_des=target_world,
            vel_des=vel_des,
            acc_des=np.zeros(3, dtype=np.float64),
            yaw_des=yaw_des,
        )
        return ControlCommand(
            t=planner.t,
            step=planner.step,
            thrust_newton=float(cmd["thrust_newton"]),
            body_rates=np.asarray(cmd["body_rates"], dtype=np.float32),
            thrust_normalized=float(cmd["thrust_normalized"]),
            source_action_type="waypoint",
            planner_action=action,
        )


class BaseCTBRController:
    """Thin actuator adapter for policy-predicted collective thrust/body rates."""

    def __init__(
        self,
        params: QuadParams | None = None,
        min_thrust_norm: float = 0.0,
        max_thrust_norm: float = 1.0,
        max_body_rate: float = 8.0,
    ):
        self.params = params or QuadParams()
        self.min_thrust_norm = float(min_thrust_norm)
        self.max_thrust_norm = float(max_thrust_norm)
        self.max_body_rate = float(max_body_rate)

    def compute(self, planner: PlannerOutput) -> ControlCommand:
        action = np.asarray(planner.action, dtype=np.float32)
        thrust_norm = float(np.clip(action[0], self.min_thrust_norm, self.max_thrust_norm))
        body_rates = np.clip(action[1:4], -self.max_body_rate, self.max_body_rate).astype(np.float32)
        return ControlCommand(
            t=planner.t,
            step=planner.step,
            thrust_newton=thrust_norm * float(self.params.max_collective_thrust),
            body_rates=body_rates,
            thrust_normalized=thrust_norm,
            source_action_type="ctbr",
            planner_action=action,
        )


class BaseAutonomyController:
    """Dispatch planner outputs to the appropriate command-generation controller."""

    def __init__(
        self,
        params: QuadParams | None = None,
        waypoint_controller: WaypointLQRController | None = None,
        ctbr_controller: BaseCTBRController | None = None,
    ):
        self.params = params or QuadParams()
        self.waypoint_controller = waypoint_controller or WaypointLQRController(params=self.params)
        self.ctbr_controller = ctbr_controller or BaseCTBRController(params=self.params)

    def compute(self, state: VehicleState, planner: PlannerOutput) -> ControlCommand:
        if planner.action_type == "waypoint":
            return self.waypoint_controller.compute(state, planner)
        if planner.action_type == "ctbr":
            return self.ctbr_controller.compute(planner)
        raise ValueError(f"Unsupported planner action_type={planner.action_type!r}")

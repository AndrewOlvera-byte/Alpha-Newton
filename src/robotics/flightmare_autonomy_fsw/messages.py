from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from scripts.flightmare_bc.expert_env import GateSpec, StepResult


ActionType = Literal["ctbr", "waypoint", "motor"]


@dataclass
class VehicleState:
    """State-estimator output published by the Flightmare state node."""

    t: float
    step: int
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    done: bool = False

    @classmethod
    def from_step_result(cls, result: StepResult, t: float, step: int) -> "VehicleState":
        return cls(
            t=float(t),
            step=int(step),
            pos=np.asarray(result.pos, dtype=np.float32),
            vel=np.asarray(result.vel, dtype=np.float32),
            quat=np.asarray(result.quat, dtype=np.float32),
            omega=np.asarray(result.omega, dtype=np.float32),
            done=bool(result.done),
        )

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.quat, self.omega]).astype(np.float32)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


@dataclass
class GateCourse:
    """Procedurally generated gate course for one episode."""

    episode_id: int
    gates: list[GateSpec]
    waypoints: np.ndarray

    @property
    def gate_positions(self) -> np.ndarray:
        if not self.gates:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([np.asarray(g.pos, dtype=np.float32) for g in self.gates], axis=0)


@dataclass
class PlannerOutput:
    """Policy output published by the planner node."""

    t: float
    step: int
    action_type: ActionType
    action: np.ndarray
    current_gate_index: int


@dataclass
class ControlCommand:
    """Command consumed by the Flightmare actuator interface."""

    t: float
    step: int
    thrust_newton: float
    body_rates: np.ndarray
    thrust_normalized: float
    source_action_type: ActionType
    planner_action: np.ndarray
    motor_normalized: np.ndarray | None = None

    @property
    def ctbr(self) -> np.ndarray:
        return np.array(
            [
                self.thrust_normalized,
                self.body_rates[0],
                self.body_rates[1],
                self.body_rates[2],
            ],
            dtype=np.float32,
        )

    @property
    def motor(self) -> np.ndarray:
        if self.motor_normalized is not None:
            return np.asarray(self.motor_normalized, dtype=np.float32)
        return np.zeros(4, dtype=np.float32)

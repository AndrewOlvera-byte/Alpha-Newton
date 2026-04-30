from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.flightmare_autonomy_fsw.gates import StrictGateTracker
from src.robotics.flightmare_autonomy_fsw.nodes import (
    BaseControllerNode,
    FlightmareStateNode,
    MissionWorldModelNode,
    PolicyPlannerNode,
)


@dataclass
class EpisodeResult:
    episode_id: int
    completed: bool
    termination_reason: str
    gates_completed: int
    num_gates: int
    completion_time_s: float | None
    elapsed_time_s: float
    steps: int
    mean_speed_mps: float
    max_speed_mps: float
    path_length_m: float
    final_distance_to_gate_m: float
    using_fallback: bool
    trajectory: np.ndarray
    gate_positions: np.ndarray
    gate_yaws: np.ndarray
    gate_indices: np.ndarray
    plane_gate_indices: np.ndarray
    gate_misses: int
    last_gate_margin_m: float | None
    planner_actions: np.ndarray
    ctbr_commands: np.ndarray

    @property
    def gate_completion(self) -> float:
        return float(self.gates_completed / max(1, self.num_gates))

    def metrics_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "gates_completed": self.gates_completed,
            "num_gates": self.num_gates,
            "gate_completion": self.gate_completion,
            "gate_misses": self.gate_misses,
            "last_gate_margin_m": self.last_gate_margin_m,
            "completion_time_s": self.completion_time_s,
            "elapsed_time_s": self.elapsed_time_s,
            "steps": self.steps,
            "mean_speed_mps": self.mean_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "path_length_m": self.path_length_m,
            "final_distance_to_gate_m": self.final_distance_to_gate_m,
            "using_fallback": self.using_fallback,
        }


class FlightmareAutonomyGraph:
    """Deterministic ROS-style graph runner for one Flightmare autonomy stack."""

    def __init__(
        self,
        state_node: FlightmareStateNode,
        mission_node: MissionWorldModelNode,
        planner_node: PolicyPlannerNode,
        controller_node: BaseControllerNode,
        max_steps: int = 1500,
        max_world_radius: float = 200.0,
        min_z: float = -0.25,
        gate_vehicle_radius: float = 0.15,
        terminate_on_gate_miss: bool = True,
    ):
        self.state_node = state_node
        self.mission_node = mission_node
        self.planner_node = planner_node
        self.controller_node = controller_node
        self.max_steps = int(max_steps)
        self.max_world_radius = float(max_world_radius)
        self.min_z = float(min_z)
        self.gate_vehicle_radius = float(gate_vehicle_radius)
        self.terminate_on_gate_miss = bool(terminate_on_gate_miss)

    def _invalid_reason(self, state) -> str | None:
        if state.done:
            return "env_done"
        if not (
            np.all(np.isfinite(state.pos))
            and np.all(np.isfinite(state.vel))
            and np.all(np.isfinite(state.quat))
            and np.all(np.isfinite(state.omega))
        ):
            return "nonfinite_state"
        if float(np.linalg.norm(state.pos)) > self.max_world_radius:
            return "out_of_bounds"
        if float(state.pos[2]) < self.min_z:
            return "ground"
        return None

    def run_episode(self, episode_id: int, rng: np.random.Generator) -> EpisodeResult:
        course, state = self.state_node.reset(episode_id=episode_id, rng=rng)
        self.mission_node.reset(course.gates)
        strict_tracker = StrictGateTracker(course.gates, vehicle_radius=self.gate_vehicle_radius)

        positions: list[np.ndarray] = []
        gate_indices: list[int] = []
        plane_gate_indices: list[int] = []
        distances: list[float] = []
        speeds: list[float] = []
        planner_actions: list[np.ndarray] = []
        ctbr_commands: list[np.ndarray] = []

        completed = False
        termination_reason = "max_steps"
        mission = None
        prev_state = None
        last_gate_margin = None

        while True:
            invalid_reason = self._invalid_reason(state)
            if invalid_reason is not None:
                termination_reason = invalid_reason
                break

            if prev_state is not None:
                event = strict_tracker.update(prev_state.pos, state.pos)
                if event is not None:
                    last_gate_margin = event.clearance_margin_m
                    if event.missed and self.terminate_on_gate_miss:
                        termination_reason = "gate_miss"
                        positions.append(state.pos.copy())
                        gate_indices.append(int(strict_tracker.current_index))
                        plane_gate_indices.append(int(self.mission_node.wrapper.current_gate_index))
                        distances.append(float(strict_tracker.distance_to_next_gate(state.pos)))
                        speeds.append(float(state.speed))
                        break

            mission = self.mission_node.tick(state)
            positions.append(state.pos.copy())
            gate_indices.append(int(strict_tracker.current_index))
            plane_gate_indices.append(int(mission.current_gate_index))
            distances.append(float(strict_tracker.distance_to_next_gate(state.pos)))
            speeds.append(float(state.speed))

            if strict_tracker.completed:
                completed = True
                termination_reason = "completed"
                break
            if state.step >= self.max_steps:
                termination_reason = "max_steps"
                break

            planner = self.planner_node.tick(state, mission)
            command = self.controller_node.tick(state, planner)
            planner_actions.append(planner.action.copy())
            ctbr_commands.append(command.ctbr.copy())
            prev_state = state
            state = self.state_node.step(command)

        traj = np.stack(positions, axis=0) if positions else np.zeros((0, 3), dtype=np.float32)
        if len(traj) > 1:
            path_length = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
        else:
            path_length = 0.0
        gates_completed = int(min(strict_tracker.current_index, len(course.gates)))
        elapsed_time = float(state.t)

        return EpisodeResult(
            episode_id=int(episode_id),
            completed=completed,
            termination_reason=termination_reason,
            gates_completed=gates_completed,
            num_gates=len(course.gates),
            completion_time_s=elapsed_time if completed else None,
            elapsed_time_s=elapsed_time,
            steps=int(state.step),
            mean_speed_mps=float(np.mean(speeds)) if speeds else 0.0,
            max_speed_mps=float(np.max(speeds)) if speeds else 0.0,
            path_length_m=path_length,
            final_distance_to_gate_m=float(distances[-1]) if distances else 0.0,
            using_fallback=self.state_node.using_fallback,
            trajectory=traj,
            gate_positions=course.gate_positions,
            gate_yaws=np.asarray([g.yaw for g in course.gates], dtype=np.float32),
            gate_indices=np.asarray(gate_indices, dtype=np.int32),
            plane_gate_indices=np.asarray(plane_gate_indices, dtype=np.int32),
            gate_misses=int(strict_tracker.misses),
            last_gate_margin_m=last_gate_margin,
            planner_actions=(
                np.stack(planner_actions, axis=0)
                if planner_actions
                else np.zeros((0, 4), dtype=np.float32)
            ),
            ctbr_commands=(
                np.stack(ctbr_commands, axis=0)
                if ctbr_commands
                else np.zeros((0, 4), dtype=np.float32)
            ),
        )

    def run(self, episodes: int, seed: int = 0) -> list[EpisodeResult]:
        rng = np.random.default_rng(seed)
        return [self.run_episode(ep, rng) for ep in range(int(episodes))]

    def close(self) -> None:
        self.state_node.close()

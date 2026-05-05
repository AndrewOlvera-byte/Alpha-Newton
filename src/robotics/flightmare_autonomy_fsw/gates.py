from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateCrossingEvent:
    gate_index: int
    crossed: bool
    passed: bool
    missed: bool
    crossing_pos: np.ndarray | None = None
    lateral_m: float = 0.0
    vertical_m: float = 0.0
    clearance_margin_m: float = 0.0


def _gate_pos_yaw_size_quat(gate):
    if hasattr(gate, "pos"):
        pos = np.asarray(gate.pos, dtype=np.float64)
        yaw = float(gate.yaw)
        size = getattr(gate, "size", None)
        quat = getattr(gate, "quat", None)
    else:
        pos = np.asarray(gate["pos"], dtype=np.float64)
        yaw = float(gate["yaw"])
        size = gate.get("size")
        quat = gate.get("quat")
    if size is None:
        size_arr = np.ones(3, dtype=np.float64)
    else:
        size_arr = np.asarray(size, dtype=np.float64).reshape(-1)
        if size_arr.size == 1:
            size_arr = np.repeat(size_arr, 3)
    if quat is not None:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
    return pos, yaw, size_arr, quat


def _gate_pos_yaw_size(gate):
    pos, yaw, size, _ = _gate_pos_yaw_size_quat(gate)
    return pos, yaw, size


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def gate_quat(gate) -> np.ndarray:
    """World-from-gate quaternion (w, x, y, z). Falls back to yaw-only when
    the gate has no full 3D orientation set."""
    _, yaw, _, quat = _gate_pos_yaw_size_quat(gate)
    return quat if quat is not None else _quat_yaw(yaw)


def gate_frame(gate) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gate center and world-frame forward/lateral/up axes.

    The gate's body +x is forward (the approach normal), +y is lateral, +z
    is up (top of the gate). For an upright yaw-only gate this collapses to
    ``[cos(yaw), sin(yaw), 0]`` etc. For a Split-S inverted gate the up axis
    points world-down — which lets the planner / aperture check / mission
    obs all behave correctly without special-casing inverted gates."""
    pos, yaw, _, quat = _gate_pos_yaw_size_quat(gate)
    if quat is None:
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return pos, forward, lateral, up
    R = _quat_to_R(quat)
    return pos, R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()


def gate_half_extents(gate, vehicle_radius: float = 0.0) -> tuple[float, float]:
    """Return usable lateral/vertical half extents after vehicle-radius clearance."""
    _, _, size = _gate_pos_yaw_size(gate)
    width = float(size[0])
    height = float(size[2] if size.size >= 3 else size[0])
    clearance = max(0.0, float(vehicle_radius))
    return max(0.0, 0.5 * width - clearance), max(0.0, 0.5 * height - clearance)


def signed_gate_distance(pos: np.ndarray, gate) -> float:
    center, forward, _, _ = gate_frame(gate)
    return float(np.dot(np.asarray(pos, dtype=np.float64) - center, forward))


def gate_offsets(pos: np.ndarray, gate) -> tuple[float, float]:
    center, _, lateral, up = gate_frame(gate)
    delta = np.asarray(pos, dtype=np.float64) - center
    return float(np.dot(delta, lateral)), float(np.dot(delta, up))


def segment_gate_crossing(
    prev_pos: np.ndarray,
    pos: np.ndarray,
    gate,
    vehicle_radius: float = 0.0,
) -> GateCrossingEvent:
    """Check whether a segment crosses the gate plane inside the aperture."""
    prev_s = signed_gate_distance(prev_pos, gate)
    curr_s = signed_gate_distance(pos, gate)
    if not (prev_s <= 0.0 < curr_s):
        return GateCrossingEvent(-1, crossed=False, passed=False, missed=False)

    denom = curr_s - prev_s
    alpha = 1.0 if abs(denom) < 1e-9 else float(np.clip(-prev_s / denom, 0.0, 1.0))
    crossing = np.asarray(prev_pos, dtype=np.float64) + alpha * (
        np.asarray(pos, dtype=np.float64) - np.asarray(prev_pos, dtype=np.float64)
    )
    lat, vert = gate_offsets(crossing, gate)
    half_w, half_h = gate_half_extents(gate, vehicle_radius=vehicle_radius)
    margin = min(half_w - abs(lat), half_h - abs(vert))
    passed = margin >= 0.0
    return GateCrossingEvent(
        gate_index=-1,
        crossed=True,
        passed=passed,
        missed=not passed,
        crossing_pos=crossing.astype(np.float32),
        lateral_m=float(lat),
        vertical_m=float(vert),
        clearance_margin_m=float(margin),
    )


class StrictGateTracker:
    """Race validator: gates count only when crossed inside the aperture."""

    def __init__(self, gates, vehicle_radius: float = 0.15):
        self.gates = list(gates)
        self.vehicle_radius = float(vehicle_radius)
        self.current_index = 0
        self.misses = 0
        self.last_event: GateCrossingEvent | None = None

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @property
    def completed(self) -> bool:
        return self.n_gates > 0 and self.current_index >= self.n_gates

    def reset(self) -> None:
        self.current_index = 0
        self.misses = 0
        self.last_event = None

    def update(self, prev_pos: np.ndarray, pos: np.ndarray) -> GateCrossingEvent | None:
        if self.completed or not self.gates:
            return None
        gate_index = self.current_index
        event = segment_gate_crossing(
            prev_pos,
            pos,
            self.gates[gate_index],
            vehicle_radius=self.vehicle_radius,
        )
        if not event.crossed:
            return None
        event.gate_index = gate_index
        if event.passed:
            self.current_index += 1
        else:
            self.misses += 1
        self.last_event = event
        return event

    def distance_to_next_gate(self, pos: np.ndarray) -> float:
        if not self.gates:
            return 0.0
        idx = min(self.current_index, self.n_gates - 1)
        center, _, _, _ = gate_frame(self.gates[idx])
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - center))

    def segment_progress(self, prev_pos: np.ndarray, pos: np.ndarray) -> float:
        """Progress along the previous-gate-center to current-gate-center segment."""
        if not self.gates or self.completed:
            return 0.0
        idx = self.current_index
        curr, _, _, _ = gate_frame(self.gates[idx])
        if idx == 0:
            prev_center = np.asarray(prev_pos, dtype=np.float64)
        else:
            prev_center, _, _, _ = gate_frame(self.gates[idx - 1])
        seg = curr - prev_center
        denom = float(np.linalg.norm(seg))
        if denom < 1e-6:
            return 0.0
        direction = seg / denom
        return float(np.dot(np.asarray(pos, dtype=np.float64) - np.asarray(prev_pos, dtype=np.float64), direction))

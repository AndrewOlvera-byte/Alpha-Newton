"""Body-frame mission encoding (v2: full 3D gate orientation).

Per-step mission vector:

    [ for k in 0..LOOKAHEAD_GATES-1:
        dx_body, dy_body, dz_body,   # gate position relative to drone, in drone body frame
        fx_body, fy_body, fz_body ]  # gate forward (approach normal) in drone body frame
    gate_progress    # current_index / total_gates  (0 if no gates)
    dist_to_next     # ||gate_curr.pos - drone.pos||

This encodes the full 3D gate frame (not just yaw_rel like v1), so inverted
and arbitrarily-tilted gates are first-class — required for canonical
Split-S maneuvers. The gate-up axis is reconstructible from the gate forward
plus the assumption that the gate's lateral axis is perpendicular to forward
in the gate's plane; in the cases where lateral is not deducible (gate normal
exactly along world-up), the policy can still decide by relying on consecutive
gate positions.

If fewer than LOOKAHEAD_GATES gates remain ahead, the last gate is repeated
so the policy sees a continuous signal instead of a discontinuity at the
final gate.

``MissionTracker`` owns the gate-progress state machine and advances the
current target gate when the drone crosses the gate's forward plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.flightmare_bc.controllers import quat_to_R


LOOKAHEAD_GATES = 3
MISSION_DIM = LOOKAHEAD_GATES * 6 + 2  # 20


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


@dataclass
class _GateView:
    pos: np.ndarray            # world position
    yaw: float                 # legacy z-yaw (preserved for upright-gate paths)
    forward: np.ndarray        # unit world-frame approach normal (gate body +x)
    quat: np.ndarray | None = None  # full world-from-gate (w,x,y,z); None == yaw-only


def gates_to_views(gates) -> list[_GateView]:
    views = []
    for g in gates:
        if hasattr(g, "pos"):
            pos = np.asarray(g.pos, dtype=np.float64)
            yaw = float(g.yaw)
            quat = getattr(g, "quat", None)
        else:
            pos = np.asarray(g["pos"], dtype=np.float64)
            yaw = float(g["yaw"])
            quat = g.get("quat")
        if quat is not None:
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            R = quat_to_R(quat)
            forward = R[:, 0].copy()
        else:
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        views.append(_GateView(pos=pos, yaw=yaw, forward=forward, quat=quat))
    return views


def encode_mission(
    pos: np.ndarray,
    quat: np.ndarray,
    gates: list[_GateView],
    current_index: int,
    lookahead: int = LOOKAHEAD_GATES,
) -> np.ndarray:
    """Pure function: produce the per-step mission vector (20-d).

    No state — pulls the next ``lookahead`` gates starting at ``current_index``.
    Used both at data collection time and inside ``MissionWrapper`` at deploy.
    """
    out = np.zeros(lookahead * 6 + 2, dtype=np.float32)
    if not gates:
        return out

    R_wb = quat_to_R(np.asarray(quat, dtype=np.float64))
    R_bw = R_wb.T  # world -> body

    n = len(gates)
    last = max(0, min(current_index, n - 1))
    for k in range(lookahead):
        idx = min(current_index + k, n - 1)
        g = gates[idx]
        delta_world = g.pos - np.asarray(pos, dtype=np.float64)
        delta_body = R_bw @ delta_world
        forward_body = R_bw @ g.forward
        base = k * 6
        out[base + 0] = delta_body[0]
        out[base + 1] = delta_body[1]
        out[base + 2] = delta_body[2]
        out[base + 3] = forward_body[0]
        out[base + 4] = forward_body[1]
        out[base + 5] = forward_body[2]

    progress = float(current_index) / float(max(1, n))
    out[lookahead * 6 + 0] = np.clip(progress, 0.0, 1.0)

    g_curr = gates[last]
    out[lookahead * 6 + 1] = float(np.linalg.norm(g_curr.pos - np.asarray(pos, dtype=np.float64)))
    return out


class MissionTracker:
    """Owns the gate-progress state machine + emits mission vectors.

    Progress rule: the drone has "passed" gate k once the signed distance
    along the gate's forward axis becomes positive
    (``dot(pos - gate.pos, forward) > 0``). At that point we advance the
    current target to k+1.
    """

    def __init__(self, gates, lookahead: int = LOOKAHEAD_GATES):
        self.gates: list[_GateView] = gates_to_views(gates)
        self.lookahead = int(lookahead)
        self.current_index = 0

    def update(self, pos: np.ndarray) -> None:
        if not self.gates:
            return
        while self.current_index < len(self.gates):
            g = self.gates[self.current_index]
            signed = float(np.dot(np.asarray(pos, dtype=np.float64) - g.pos, g.forward))
            if signed > 0.0:
                self.current_index += 1
            else:
                break

    def vector(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return encode_mission(pos, quat, self.gates, self.current_index, self.lookahead)

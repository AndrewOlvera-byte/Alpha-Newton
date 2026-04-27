"""Body-frame mission encoding shared by collector + deployment wrapper.

The "mission" vector is the perception prior a real autonomy stack would
synthesize from a known track map + estimated drone pose:

    [ for k in 0..LOOKAHEAD_GATES-1:
        dx_body, dy_body, dz_body, yaw_rel ]
    gate_progress    # current_index / total_gates  (0 if no gates)
    dist_to_next     # ||gate_curr.pos - drone.pos||

If fewer than LOOKAHEAD_GATES gates remain ahead, the last gate is repeated
so the policy sees a continuous signal instead of a discontinuity at the
final gate.

``MissionTracker`` also owns the gate-progress state machine: it advances the
current target gate when the drone crosses the gate's forward plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.flightmare_bc.controllers import quat_to_R


LOOKAHEAD_GATES = 3
MISSION_DIM = LOOKAHEAD_GATES * 4 + 2


@dataclass
class _GateView:
    pos: np.ndarray
    yaw: float
    forward: np.ndarray  # unit world-frame +x of the gate (from yaw)


def _wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def gates_to_views(gates) -> list[_GateView]:
    views = []
    for g in gates:
        if hasattr(g, "pos"):
            pos = np.asarray(g.pos, dtype=np.float64)
            yaw = float(g.yaw)
        else:
            pos = np.asarray(g["pos"], dtype=np.float64)
            yaw = float(g["yaw"])
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        views.append(_GateView(pos=pos, yaw=yaw, forward=forward))
    return views


def encode_mission(
    pos: np.ndarray,
    quat: np.ndarray,
    gates: list[_GateView],
    current_index: int,
    lookahead: int = LOOKAHEAD_GATES,
) -> np.ndarray:
    """Pure function: produce the per-step mission vector.

    No state — pulls the next ``lookahead`` gates starting at ``current_index``.
    Used both at data collection time and inside ``MissionWrapper`` at deploy.
    """
    out = np.zeros(lookahead * 4 + 2, dtype=np.float32)
    if not gates:
        return out

    R_wb = quat_to_R(np.asarray(quat, dtype=np.float64))
    R_bw = R_wb.T  # world -> body
    drone_yaw = float(np.arctan2(R_wb[1, 0], R_wb[0, 0]))

    n = len(gates)
    last = max(0, min(current_index, n - 1))
    for k in range(lookahead):
        idx = min(current_index + k, n - 1)
        g = gates[idx]
        delta_world = g.pos - np.asarray(pos, dtype=np.float64)
        delta_body = R_bw @ delta_world
        yaw_rel = _wrap_to_pi(g.yaw - drone_yaw)
        base = k * 4
        out[base + 0] = delta_body[0]
        out[base + 1] = delta_body[1]
        out[base + 2] = delta_body[2]
        out[base + 3] = yaw_rel

    progress = float(current_index) / float(max(1, n))
    out[lookahead * 4 + 0] = np.clip(progress, 0.0, 1.0)

    g_curr = gates[last]
    out[lookahead * 4 + 1] = float(np.linalg.norm(g_curr.pos - np.asarray(pos, dtype=np.float64)))
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

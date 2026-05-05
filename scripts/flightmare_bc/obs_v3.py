"""Swift-style obs v3 encoders (40-d, MLP-fusion friendly).

Layout (single source of truth — used by collect-time transform, BC dataset,
and PPO env):

  proprio_core (9):
    v_body       (3)   linear velocity in body frame
    omega_body   (3)   angular velocity in body frame (already body-frame in sim)
    gravity_body (3)   gravity vector expressed in body frame; encodes roll+pitch
                       without the quaternion sign / yaw ambiguity

  prev_action  (4):   last commanded action (stored separately so the same
                      proprio_core is reusable across action-type ablations).

  gate (24):  for k in 0..LOOKAHEAD_GATES_V3-1:
    corner_TL_body (3), corner_TR_body (3), corner_BR_body (3), corner_BL_body (3)
    Four world corners of gate k transformed into the drone body frame.
    Implicit encoding of gate position, full 3D orientation (incl. roll —
    a 180° flip about the forward axis swaps top/bottom corners), and size.

  aux (3):
    progress         (1)   current_gate_index / max(1, total_gates)
    dist_to_current  (1)   ||gate_curr.pos - drone.pos||
    time_since_pass  (1)   seconds since the most recent gate index advance

Total state-side dim: 9 + 4 + 24 + 3 = 40 (matches Swift's published policy obs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.flightmare_autonomy_fsw.gates import gate_frame, gate_half_extents


LOOKAHEAD_GATES_V3 = 2
PROPRIO_CORE_DIM = 9
GATE_DIM = LOOKAHEAD_GATES_V3 * 4 * 3  # 24
AUX_DIM = 3
STATE_V3_DIM = PROPRIO_CORE_DIM + GATE_DIM + AUX_DIM  # 36 (without prev_action)
GRAVITY = 9.81


def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def build_proprio_core(
    vel_world: np.ndarray, quat_wxyz: np.ndarray, omega_body: np.ndarray
) -> np.ndarray:
    """9-d proprioception block: [v_body, omega_body, gravity_body]."""
    R_wb = _quat_to_R(np.asarray(quat_wxyz, dtype=np.float64))
    R_bw = R_wb.T
    v_body = R_bw @ np.asarray(vel_world, dtype=np.float64)
    g_world = np.array([0.0, 0.0, -GRAVITY], dtype=np.float64)
    g_body = R_bw @ g_world
    out = np.empty(PROPRIO_CORE_DIM, dtype=np.float32)
    out[0:3] = v_body
    out[3:6] = np.asarray(omega_body, dtype=np.float64)
    out[6:9] = g_body
    return out


@dataclass
class GateSpec:
    """Minimal gate descriptor expected by the corner encoder."""
    pos: np.ndarray            # (3,) world position
    yaw: float                 # legacy yaw (ignored if quat provided)
    size: np.ndarray           # (3,) [w, _, h]; we use 0 (width) and 2 (height)
    quat: np.ndarray | None = None  # full world-from-gate (w,x,y,z) optional


def gates_from_records(records) -> list[GateSpec]:
    """Coerce dataclass / dict / sim-object gate records into ``GateSpec``."""
    out: list[GateSpec] = []
    for g in records:
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
        out.append(GateSpec(pos=pos, yaw=yaw, size=size, quat=quat))
    return out


def _gate_world_corners(g: GateSpec) -> np.ndarray:
    """Return (4, 3) world-frame corner positions in order TL, TR, BR, BL.

    TL = center + (+up)*half_h + (-lateral)*half_w   (gate-frame's "top left"
                                                      from the drone's approach
                                                      side, with +lateral on
                                                      the right)
    TR = center + (+up)*half_h + (+lateral)*half_w
    BR = center + (-up)*half_h + (+lateral)*half_w
    BL = center + (-up)*half_h + (-lateral)*half_w
    """
    center, _forward, lateral, up = gate_frame(g)
    half_w, half_h = gate_half_extents(g, vehicle_radius=0.0)
    tl = center + up * half_h - lateral * half_w
    tr = center + up * half_h + lateral * half_w
    br = center - up * half_h + lateral * half_w
    bl = center - up * half_h - lateral * half_w
    return np.stack([tl, tr, br, bl], axis=0)


def encode_gate_corners(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    gates: list[GateSpec],
    current_index: int,
    lookahead: int = LOOKAHEAD_GATES_V3,
) -> np.ndarray:
    """24-d gate block: ``lookahead`` gates × 4 corners × 3 (body-frame).

    If fewer than ``lookahead`` gates remain ahead, the last gate is repeated
    so the policy sees a continuous signal at the end of the course.
    """
    out = np.zeros(lookahead * 4 * 3, dtype=np.float32)
    if not gates:
        return out
    R_bw = _quat_to_R(np.asarray(quat_wxyz, dtype=np.float64)).T
    pos_w = np.asarray(pos, dtype=np.float64)
    n = len(gates)
    for k in range(lookahead):
        idx = min(current_index + k, n - 1)
        corners_w = _gate_world_corners(gates[idx])     # (4,3)
        corners_b = (R_bw @ (corners_w - pos_w).T).T    # (4,3)
        out[k * 12 : (k + 1) * 12] = corners_b.reshape(-1)
    return out


def encode_aux(
    pos: np.ndarray,
    gates: list[GateSpec],
    current_index: int,
    time_since_pass: float,
) -> np.ndarray:
    """3-d aux block: [progress, dist_to_current, time_since_pass]."""
    out = np.zeros(AUX_DIM, dtype=np.float32)
    if not gates:
        return out
    n = len(gates)
    last = max(0, min(current_index, n - 1))
    out[0] = float(np.clip(current_index / max(1, n), 0.0, 1.0))
    out[1] = float(np.linalg.norm(gates[last].pos - np.asarray(pos, dtype=np.float64)))
    out[2] = float(time_since_pass)
    return out


def encode_obs_v3(
    pos: np.ndarray,
    vel_world: np.ndarray,
    quat_wxyz: np.ndarray,
    omega_body: np.ndarray,
    gates: list[GateSpec],
    current_index: int,
    time_since_pass: float,
    lookahead: int = LOOKAHEAD_GATES_V3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot helper. Returns (proprio_core[9], gate[24], aux[3])."""
    proprio = build_proprio_core(vel_world, quat_wxyz, omega_body)
    gate = encode_gate_corners(pos, quat_wxyz, gates, current_index, lookahead)
    aux = encode_aux(pos, gates, current_index, time_since_pass)
    return proprio, gate, aux

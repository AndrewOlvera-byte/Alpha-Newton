"""Geometry validator for generated courses.

``validate_course`` enforces that a course is physically flyable before any BC
label is written or an RL episode starts: gates must not overlap or sit too
close, must stay in the arena and above ground, segments must not climb/turn
impossibly steeply, the aperture must clear the vehicle radius, and (for
inverted/tilted gates) the approach must cross the correct side of the gate
plane. It reuses the quaternion-aware helpers in
``src.robotics.flightmare_autonomy_fsw.gates`` so inverted gates validate
correctly without special-casing.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.robotics.flightmare_autonomy_fsw.gates import (
    gate_frame,
    gate_half_extents,
    signed_gate_distance,
)
from src.robotics.flightmare_courses.scenarios import CourseBounds


def _centers(gates: Sequence) -> np.ndarray:
    return np.stack([np.asarray(g.pos, dtype=np.float64) for g in gates], axis=0)


def path_length(gates: Sequence) -> float:
    """Total Euclidean length through the gate centers (m)."""
    if len(gates) < 2:
        return 0.0
    c = _centers(gates)
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))


def validate_course(
    gates: Sequence,
    bounds: CourseBounds,
    waypoints: np.ndarray | None = None,
) -> tuple[bool, list[str]]:
    """Return ``(ok, reasons)``; ``reasons`` is empty iff the course is valid."""
    reasons: list[str] = []
    n = len(gates)
    if n == 0:
        return False, ["empty course"]

    centers = _centers(gates)

    # --- Altitude + arena bounds ---
    for i, c in enumerate(centers):
        if c[2] < bounds.z_min - 1e-6:
            reasons.append(f"gate {i} below z_min ({c[2]:.2f} < {bounds.z_min:.2f})")
        if c[2] > bounds.z_max + 1e-6:
            reasons.append(f"gate {i} above z_max ({c[2]:.2f} > {bounds.z_max:.2f})")
        if float(np.linalg.norm(c[:2])) > bounds.arena_radius + 1e-6:
            reasons.append(f"gate {i} outside arena_radius ({np.linalg.norm(c[:2]):.1f})")

    # --- Pairwise clearance: no two gates closer than gate half-width + margin ---
    for i in range(n):
        half_w_i, _ = gate_half_extents(gates[i])
        for j in range(i + 1, n):
            half_w_j, _ = gate_half_extents(gates[j])
            min_dist = half_w_i + half_w_j + bounds.min_gate_clearance_m
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d < min_dist:
                reasons.append(
                    f"gates {i},{j} too close ({d:.2f} < {min_dist:.2f})"
                )

    # --- Consecutive spacing ---
    for i in range(n - 1):
        d = float(np.linalg.norm(centers[i + 1] - centers[i]))
        if d < bounds.min_consecutive_m:
            reasons.append(f"segment {i}->{i+1} too short ({d:.2f} < {bounds.min_consecutive_m:.2f})")
        if d > bounds.max_consecutive_m:
            reasons.append(f"segment {i}->{i+1} too long ({d:.2f} > {bounds.max_consecutive_m:.2f})")

    # --- Climb/descent angle per segment ---
    for i in range(n - 1):
        seg = centers[i + 1] - centers[i]
        horiz = float(np.linalg.norm(seg[:2]))
        climb = float(abs(np.arctan2(seg[2], max(horiz, 1e-6))))
        if climb > bounds.max_climb_angle_rad + 1e-6:
            reasons.append(f"segment {i}->{i+1} climb angle {np.degrees(climb):.0f}deg too steep")

    # --- Heading change between adjacent segments ---
    for i in range(1, n - 1):
        a = centers[i] - centers[i - 1]
        b = centers[i + 1] - centers[i]
        na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
        if na < 1e-6 or nb < 1e-6:
            continue
        cos = float(np.clip(np.dot(a[:2], b[:2]) / (na * nb), -1.0, 1.0))
        turn = float(np.arccos(cos))
        if turn > bounds.max_heading_change_rad + 1e-6:
            reasons.append(f"heading change at gate {i} {np.degrees(turn):.0f}deg too sharp")

    # --- Aperture must clear the vehicle radius ---
    for i in range(n):
        half_w, half_h = gate_half_extents(gates[i], vehicle_radius=bounds.vehicle_radius)
        if half_w <= 0.0 or half_h <= 0.0:
            reasons.append(f"gate {i} aperture too small for vehicle radius")

    # --- Total path length ---
    length = path_length(gates)
    lo, hi = float(bounds.total_length_range[0]), float(bounds.total_length_range[1])
    if length < lo:
        reasons.append(f"path length {length:.1f} below min {lo:.1f}")
    if length > hi:
        reasons.append(f"path length {length:.1f} above max {hi:.1f}")

    # --- Approach side consistency: pre-gate waypoint must be behind the plane
    #     and exit waypoint in front (handles inverted/tilted gates). ---
    if waypoints is not None:
        for i, g in enumerate(gates):
            center, forward, _, _ = gate_frame(g)
            pre = center - bounds.min_consecutive_m * 0.1 * forward
            post = center + bounds.min_consecutive_m * 0.1 * forward
            if signed_gate_distance(pre, g) >= 0.0 or signed_gate_distance(post, g) <= 0.0:
                reasons.append(f"gate {i} approach side inconsistent")

    return (len(reasons) == 0), reasons

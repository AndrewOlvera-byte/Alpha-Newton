"""Random-waypoint minimum-jerk polynomial trajectories.

Reference: Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control
for Quadrotors", ICRA 2011 - here we use the simpler minimum-jerk variant
(5th-order polynomial per segment with continuity in pos/vel/acc), which is
sufficient for BC data and avoids solving the full QP. This matches the
trajectory style used in ``rotorpy`` and ``gym-pybullet-drones``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectorySample:
    pos: np.ndarray   # (3,)
    vel: np.ndarray   # (3,)
    acc: np.ndarray   # (3,)
    jerk: np.ndarray  # (3,)
    yaw: float
    yaw_rate: float


def _quintic_coeffs(p0, v0, a0, pT, vT, aT, T):
    """5th-order polynomial coefficients with full pos/vel/acc continuity at both ends."""
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    A = np.array([
        [1, 0, 0, 0,    0,     0],
        [0, 1, 0, 0,    0,     0],
        [0, 0, 2, 0,    0,     0],
        [1, T, T2, T3,  T4,    T5],
        [0, 1, 2*T, 3*T2, 4*T3, 5*T4],
        [0, 0, 2,   6*T,  12*T2, 20*T3],
    ], dtype=np.float64)
    b = np.stack([p0, v0, a0, pT, vT, aT], axis=0)  # (6, 3) or (6,)
    return np.linalg.solve(A, b)  # (6, 3) per-axis


class MinJerkTrajectory:
    """Piecewise quintic minimum-jerk trajectory through random waypoints."""

    def __init__(
        self,
        waypoints: np.ndarray,        # (K, 3)
        avg_speed: float,
        yaw_mode: str = "tangent",    # "tangent" or "zero"
        boundary_vel: np.ndarray | None = None,
    ):
        assert waypoints.ndim == 2 and waypoints.shape[1] == 3
        self.waypoints = waypoints.astype(np.float64)
        self.yaw_mode = yaw_mode

        seg_lens = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        seg_durations = np.maximum(seg_lens / max(avg_speed, 0.1), 1e-2)
        self.seg_t0 = np.concatenate([[0.0], np.cumsum(seg_durations)])
        self.seg_dur = seg_durations
        self.total_time = float(self.seg_t0[-1])

        # Segment-end velocities: aim along the chord at avg_speed (taper at endpoints).
        K = len(self.waypoints)
        vels = np.zeros_like(self.waypoints)
        for i in range(1, K - 1):
            chord = self.waypoints[i + 1] - self.waypoints[i - 1]
            n = np.linalg.norm(chord) + 1e-9
            vels[i] = chord / n * avg_speed
        if boundary_vel is not None:
            vels[0] = boundary_vel
            vels[-1] = boundary_vel
        self.vels = vels

        accs = np.zeros_like(self.waypoints)
        self.coeffs = []
        for i in range(K - 1):
            c = _quintic_coeffs(
                self.waypoints[i], vels[i], accs[i],
                self.waypoints[i + 1], vels[i + 1], accs[i + 1],
                seg_durations[i],
            )
            self.coeffs.append(c)

    def _segment(self, t: float) -> tuple[int, float]:
        t = float(np.clip(t, 0.0, self.total_time - 1e-9))
        idx = int(np.searchsorted(self.seg_t0, t, side="right") - 1)
        idx = max(0, min(idx, len(self.coeffs) - 1))
        return idx, t - self.seg_t0[idx]

    def sample(self, t: float) -> TrajectorySample:
        idx, tau = self._segment(t)
        c = self.coeffs[idx]  # (6, 3): rows are powers 0..5
        powers = np.array([1.0, tau, tau ** 2, tau ** 3, tau ** 4, tau ** 5])
        d_powers = np.array([0.0, 1.0, 2 * tau, 3 * tau ** 2, 4 * tau ** 3, 5 * tau ** 4])
        dd_powers = np.array([0.0, 0.0, 2.0, 6 * tau, 12 * tau ** 2, 20 * tau ** 3])
        ddd_powers = np.array([0.0, 0.0, 0.0, 6.0, 24 * tau, 60 * tau ** 2])
        pos = powers @ c
        vel = d_powers @ c
        acc = dd_powers @ c
        jerk = ddd_powers @ c

        if self.yaw_mode == "tangent":
            speed_xy = np.linalg.norm(vel[:2])
            yaw = float(np.arctan2(vel[1], vel[0])) if speed_xy > 0.5 else 0.0
        else:
            yaw = 0.0
        return TrajectorySample(pos=pos, vel=vel, acc=acc, jerk=jerk, yaw=yaw, yaw_rate=0.0)

    def lookahead(self, t: float, horizon_s: float) -> tuple[np.ndarray, float]:
        """Position at ``t + horizon_s`` and instantaneous tangential speed at ``t``."""
        s_ahead = self.sample(t + horizon_s)
        s_now = self.sample(t)
        speed = float(np.linalg.norm(s_now.vel))
        return s_ahead.pos, speed


def random_waypoint_path(
    rng: np.random.Generator,
    n_waypoints: int = 6,
    bbox: tuple[float, float, float] = (8.0, 8.0, 3.0),
    z_min: float = 1.0,
    min_step: float = 1.5,
) -> np.ndarray:
    """Sample a random waypoint sequence inside an XYZ bounding box.

    Successive waypoints are at least ``min_step`` apart so the polynomial
    segments are well-conditioned.
    """
    bx, by, bz = bbox
    out = [np.array([0.0, 0.0, max(z_min, bz * 0.5)])]
    for _ in range(n_waypoints):
        for _ in range(40):
            cand = np.array([
                rng.uniform(-bx, bx),
                rng.uniform(-by, by),
                rng.uniform(z_min, bz),
            ])
            if np.linalg.norm(cand - out[-1]) >= min_step:
                out.append(cand)
                break
        else:
            out.append(out[-1] + rng.normal(scale=min_step, size=3))
    return np.stack(out, axis=0)

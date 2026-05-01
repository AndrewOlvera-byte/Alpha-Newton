"""SE(3) geometric controller (Lee, Leok, McClamroch 2010).

Given a desired pos/vel/acc/yaw reference and the current quadrotor state, it
returns a `(thrust_collective, omega_body)` command - the standard CTBR
interface used in real autonomous-drone-racing inner-loop controllers.

Reference Python implementations: ``rotorpy/controllers/quadrotor_control.py``
and ``gym-pybullet-drones`` (Lee2010 controller). This module is a clean
re-derivation; no external code is required.

Conventions
-----------
* World frame: ``z`` up, gravity acts as ``-g e_z``.
* Quadrotor orientation as a rotation matrix ``R_wb`` (world <- body).
* Thrust along body ``+z``.
* Collective thrust output is normalized to ``[0, 1]`` by ``hover_thrust_ratio``
  so it composes with PPO's tanh-style action assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hat(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def quat_to_R(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix R_wb."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


@dataclass
class QuadParams:
    mass: float = 0.73                   # kg, Flightmare default
    g: float = 9.81
    arm_length: float = 0.17
    k_thrust: float = 1.91e-6
    k_torque: float = 2.6e-7
    max_collective_thrust: float = 4.0 * 9.81 * 0.73  # ~4 g of total thrust
    inertia: tuple[float, float, float] = (0.0025, 0.0021, 0.0043)


@dataclass
class GeometricGains:
    kp_pos: tuple[float, float, float] = (6.0, 6.0, 8.0)
    kd_pos: tuple[float, float, float] = (4.0, 4.0, 5.0)
    kp_att: tuple[float, float, float] = (8.0, 8.0, 3.0)


class GeometricSE3Controller:
    """Position + attitude controller producing collective thrust and body rates."""

    def __init__(self, params: QuadParams = QuadParams(), gains: GeometricGains = GeometricGains()):
        self.params = params
        self.gains = gains
        self.kp_pos = np.asarray(gains.kp_pos)
        self.kd_pos = np.asarray(gains.kd_pos)
        self.kp_att = np.asarray(gains.kp_att)

    @property
    def hover_thrust(self) -> float:
        return self.params.mass * self.params.g

    def compute(
        self,
        pos: np.ndarray, vel: np.ndarray, quat: np.ndarray,
        pos_des: np.ndarray, vel_des: np.ndarray, acc_des: np.ndarray,
        yaw_des: float,
    ) -> dict:
        """Return CTBR command and intermediate references.

        Outputs
        -------
        ``thrust_normalized``: collective thrust / max_collective_thrust in [0, 1].
        ``thrust_newton``:     collective thrust in N.
        ``body_rates``:        desired body angular velocity (rad/s), 3-vector.
        ``motor_normalized``:  per-rotor thrust in [0, 1] via standard X-mixer.
        """
        m, g = self.params.mass, self.params.g

        # --- Position loop -> desired thrust direction ------------------------
        e_p = pos - pos_des
        e_v = vel - vel_des
        F_des = -self.kp_pos * e_p - self.kd_pos * e_v + m * acc_des + np.array([0.0, 0.0, m * g])

        R = quat_to_R(quat)
        b3 = R[:, 2]
        thrust_newton = float(np.dot(F_des, b3))
        thrust_newton = max(thrust_newton, 0.05 * m * g)

        # --- Attitude loop ----------------------------------------------------
        b3_des = F_des / (np.linalg.norm(F_des) + 1e-9)
        c_yaw = np.cos(yaw_des)
        s_yaw = np.sin(yaw_des)
        b1c = np.array([c_yaw, s_yaw, 0.0])
        b2_des = np.cross(b3_des, b1c)
        b2_des /= (np.linalg.norm(b2_des) + 1e-9)
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des], axis=1)

        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        # Geometric controller normally outputs torque; we expose body rates
        # directly (a tracking-rate controller in the inner loop). Mapping
        # ``omega_des = -kp_att * e_R`` is the standard rate-reference used by
        # CTBR-interface controllers (Foehn et al., Kaufmann et al.).
        omega_des = -self.kp_att * e_R

        max_T = self.params.max_collective_thrust
        thrust_norm = float(np.clip(thrust_newton / max_T, 0.0, 1.0))

        motor_norm = self._mix_to_motors(thrust_newton, omega_des)
        return {
            "thrust_newton": thrust_newton,
            "thrust_normalized": thrust_norm,
            "body_rates": omega_des.astype(np.float32),
            "motor_normalized": motor_norm.astype(np.float32),
            "R_des": R_des,
            "F_des": F_des,
        }

    def _mix_to_motors(self, thrust_newton: float, omega_des: np.ndarray) -> np.ndarray:
        """Approximate X-config mixer: T and torques -> per-rotor thrust normalized to [0,1].

        We use a body-rate proxy for torque (omega_des * inertia) which is a
        close-enough approximation when the inner-loop attitude rate tracks
        omega_des at high bandwidth - good enough for BC labels.
        """
        Ix, Iy, Iz = self.params.inertia
        tau = np.array([Ix * omega_des[0], Iy * omega_des[1], Iz * omega_des[2]])
        L = self.params.arm_length
        kT = self.params.k_thrust
        kQ = self.params.k_torque

        # X mixer (standard quadcopter): motor 0 FR, 1 BL, 2 FL, 3 BR.
        # Motor-normalized thrust is f_i / f_i,max, so 1.0 on all four rotors
        # gives the configured collective thrust ceiling.
        s = np.sqrt(0.5)
        kappa = kQ / kT
        mixer = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-L * s, L * s, L * s, -L * s],
            [-L * s, L * s, -L * s, L * s],
            [-kappa, -kappa, kappa, kappa],
        ])
        rhs = np.array([thrust_newton, tau[0], tau[1], tau[2]])
        f = np.linalg.solve(mixer, rhs)
        f_max = self.params.max_collective_thrust / 4.0
        return np.clip(f / max(f_max, 1e-6), 0.0, 1.0)


def world_to_body(vec_world: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = quat_to_R(quat)
    return R.T @ vec_world

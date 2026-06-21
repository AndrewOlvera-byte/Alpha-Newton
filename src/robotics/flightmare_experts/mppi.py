"""Self-contained MPPI (Model Predictive Path Integral) CTBR expert.

A sampling-based receding-horizon controller over the reduced CTBR quadrotor
model (ideal body-rate tracking + collective thrust along body-z under the
shared ``QuadParams``). Each control step it samples ``num_samples`` action
sequences around a warm-started nominal, rolls them out ``horizon`` steps,
scores them against a reference (position + feed-forward velocity) with
actuator-saturation penalties, and returns the MPPI-weighted first action as a
CTBR command.

Unlike the pure geometric P-tracker, MPPI plans over a horizon under the body-
rate/thrust limits, so it accelerates on straights and brakes into gates rather
than tracking instantaneously — the realism needed for high-quality high-speed
BC labels. It is a drop-in for ``GeometricSE3Controller.compute`` (same args and
return dict) so it slots into the existing collection/eval loops.

Pure NumPy, no native deps; rollouts are vectorized across samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.flightmare_bc.controllers import GeometricSE3Controller, QuadParams


@dataclass
class MPPIConfig:
    horizon: int = 25                 # H steps
    num_samples: int = 384            # K rollouts
    dt: float = 0.01                  # control period (s)
    temperature: float = 0.4          # lambda in the softmax
    # Sampling std per action dim [thrust_norm, wx, wy, wz].
    noise_std: tuple[float, float, float, float] = (0.05, 2.2, 2.2, 1.2)
    max_body_rate: float = 18.0       # rad/s clamp on sampled rates
    # Cost weights (tuned for ~0.14 m tracking at 4 m/s on the reduced model).
    w_pos: float = 20.0               # running position tracking
    w_vel: float = 2.0                # running velocity match
    w_term: float = 30.0              # terminal position
    w_ctrl: float = 0.01              # body-rate effort
    w_smooth: float = 0.02            # action smoothness
    w_sat: float = 5.0                # actuator-saturation penalty
    elite_frac: float = 1.0           # 1.0 = standard MPPI over all samples


def _quat_to_R_batch(q: np.ndarray) -> np.ndarray:
    """(K,4) wxyz -> (K,3,3) world-from-body."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    K = q.shape[0]
    R = np.empty((K, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _quat_integrate_batch(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate (K,4) quats by body rates (K,3) over dt; returns normalized (K,4)."""
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # q_dot = 0.5 * q ⊗ (0, omega_body)
    dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
    dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
    dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
    dqz = 0.5 * (qw * wz + qx * wy - qy * wx)
    out = q + dt * np.stack([dqw, dqx, dqy, dqz], axis=1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


class MPPIController:
    """MPPI CTBR expert; drop-in for GeometricSE3Controller.compute."""

    def __init__(self, params: QuadParams | None = None, config: MPPIConfig | None = None, seed: int = 0):
        self.params = params or QuadParams()
        self.cfg = config or MPPIConfig()
        self.rng = np.random.default_rng(seed)
        self.source = "mppi"
        self._mixer = GeometricSE3Controller(params=self.params)
        self._hover_norm = float(np.clip((self.params.mass * self.params.g) / max(self.params.max_collective_thrust, 1e-6), 0.0, 1.0))
        # Warm-started nominal action sequence (H, 4): [thrust_norm, wx, wy, wz].
        self._nominal = np.zeros((self.cfg.horizon, 4), dtype=np.float64)
        self._nominal[:, 0] = self._hover_norm

    def reset(self) -> None:
        self._nominal[:] = 0.0
        self._nominal[:, 0] = self._hover_norm

    def _rollout_cost(
        self,
        actions: np.ndarray,        # (K, H, 4)
        pos0: np.ndarray, vel0: np.ndarray, quat0: np.ndarray,
        ref_pos: np.ndarray,        # (H, 3)
        ref_vel: np.ndarray,        # (3,)
    ) -> np.ndarray:
        cfg = self.cfg
        K = actions.shape[0]
        m, g = self.params.mass, self.params.g
        max_T = self.params.max_collective_thrust
        pos = np.tile(pos0.astype(np.float64), (K, 1))
        vel = np.tile(vel0.astype(np.float64), (K, 1))
        quat = np.tile(quat0.astype(np.float64), (K, 1))
        grav = np.array([0.0, 0.0, g], dtype=np.float64)
        cost = np.zeros(K, dtype=np.float64)
        prev_a = None
        for t in range(cfg.horizon):
            a = actions[:, t, :]
            thrust_norm = np.clip(a[:, 0], 0.0, 1.0)
            omega = np.clip(a[:, 1:4], -cfg.max_body_rate, cfg.max_body_rate)
            quat = _quat_integrate_batch(quat, omega, cfg.dt)
            R = _quat_to_R_batch(quat)
            b3 = R[:, :, 2]
            acc = (thrust_norm[:, None] * max_T / m) * b3 - grav[None, :]
            vel = vel + acc * cfg.dt
            pos = pos + vel * cfg.dt
            # Running costs.
            dpos = pos - ref_pos[t][None, :]
            cost += cfg.w_pos * np.sum(dpos * dpos, axis=1)
            dvel = vel - ref_vel[None, :]
            cost += cfg.w_vel * np.sum(dvel * dvel, axis=1)
            cost += cfg.w_ctrl * np.sum(omega * omega, axis=1)
            # Saturation penalty (thrust pinned or rate near clamp).
            sat = (thrust_norm <= 0.02) | (thrust_norm >= 0.98)
            cost += cfg.w_sat * sat.astype(np.float64)
            cost += cfg.w_sat * np.sum((np.abs(omega) >= cfg.max_body_rate - 1e-3), axis=1)
            if prev_a is not None:
                da = a - prev_a
                cost += cfg.w_smooth * np.sum(da * da, axis=1)
            prev_a = a
        # Terminal cost.
        dterm = pos - ref_pos[-1][None, :]
        cost += cfg.w_term * np.sum(dterm * dterm, axis=1)
        return cost

    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        pos_des: np.ndarray,
        vel_des: np.ndarray,
        acc_des: np.ndarray,
        yaw_des: float,
        omega: np.ndarray | None = None,
    ) -> dict:
        cfg = self.cfg
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        pos_des = np.asarray(pos_des, dtype=np.float64)
        vel_des = np.asarray(vel_des, dtype=np.float64)
        # Feed-forward reference horizon: extrapolate the desired position along
        # the desired velocity so MPPI matches the reference SPEED, not just a
        # static point (this is what yields accelerate-then-brake behavior).
        steps = np.arange(1, cfg.horizon + 1, dtype=np.float64)
        ref_pos = pos_des[None, :] + np.outer(steps * cfg.dt, vel_des)

        noise_std = np.asarray(cfg.noise_std, dtype=np.float64)
        noise = self.rng.normal(0.0, 1.0, size=(cfg.num_samples, cfg.horizon, 4)) * noise_std[None, None, :]
        actions = self._nominal[None, :, :] + noise
        actions[:, :, 0] = np.clip(actions[:, :, 0], 0.0, 1.0)
        actions[:, :, 1:4] = np.clip(actions[:, :, 1:4], -cfg.max_body_rate, cfg.max_body_rate)

        cost = self._rollout_cost(actions, pos, vel, quat, ref_pos, vel_des)
        beta = cost.min()
        weights = np.exp(-(cost - beta) / max(cfg.temperature, 1e-6))
        weights /= weights.sum() + 1e-12
        new_nominal = np.sum(weights[:, None, None] * actions, axis=0)
        first_actions = actions[:, 0, :]
        first_mean = new_nominal[0]
        first_diff = first_actions - first_mean[None, :]
        action_cov_diag = np.sum(weights[:, None] * first_diff * first_diff, axis=0)
        if cost.shape[0] > 1:
            best_two = np.partition(cost, 1)[:2]
            cost_margin = float(best_two[1] - best_two[0])
        else:
            cost_margin = 0.0

        # Warm start: shift the solution forward one step.
        self._nominal = np.roll(new_nominal, -1, axis=0)
        self._nominal[-1] = new_nominal[-1]

        a0 = first_mean
        thrust_norm = float(np.clip(a0[0], 0.0, 1.0))
        body_rates = np.clip(a0[1:4], -cfg.max_body_rate, cfg.max_body_rate).astype(np.float64)
        thrust_newton = thrust_norm * self.params.max_collective_thrust
        omega_actual = np.zeros(3) if omega is None else np.asarray(omega, dtype=np.float64)
        motor_norm = self._mixer._mix_to_motors(thrust_newton, body_rates, omega_actual)
        return {
            "thrust_newton": float(thrust_newton),
            "thrust_normalized": thrust_norm,
            "body_rates": body_rates.astype(np.float32),
            "motor_normalized": motor_norm.astype(np.float32),
            "source": self.source,
            "mppi_cost": float(beta),
            "cost_margin": float(cost_margin),
            "action_cov_diag": action_cov_diag.astype(np.float32),
        }

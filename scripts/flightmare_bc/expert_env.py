"""Adapter around Flightmare for BC data collection.

Wraps the C++/Python `flightlib` quadrotor + Unity bridge (built into the
docker image via ``pip install flightmare/flightlib``) and exposes a tiny
``reset / step_ctbr / render`` API that the collector uses. If the binary
extensions are unavailable (e.g. running outside the docker), a pure-numpy
rigid-body fallback is used so this script can still be developed and
unit-tested. The fallback never produces real images - it returns zeros - and
prints a one-time warning.

This keeps `controllers.py` and `trajectories.py` independent of any specific
simulator binding.
"""
from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.flightmare_bc.controllers import QuadParams, quat_to_R


_SCENE_IDS = {
    "industrial": 0,
    "warehouse": 1,
    "garage": 2,
    "natureforest": 3,
    "forest": 3,
    "tunnels": 4,
}


@dataclass
class StepResult:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray            # (w, x, y, z)
    omega: np.ndarray           # body angular velocity
    images: dict[str, np.ndarray]   # {cam_name: uint8 [H, W, 3]}
    done: bool


@dataclass
class GateSpec:
    """3D gate spec.

    ``yaw`` is preserved for upright gates and Unity rendering. ``quat``
    (w, x, y, z) is the full 3D orientation of the gate frame in world coords:
    its body +x axis is the **forward** direction (the side approached from),
    +z is the **up** direction (top of the gate). For an upright gate this
    quat encodes a pure Z-rotation of magnitude ``yaw``; for a Split-S
    inverted gate the quat additionally rolls 180° about its +x axis so
    the up vector points world-down (-Z). When ``quat`` is None, callers
    derive an upright quaternion from ``yaw``.
    """

    gate_id: str
    pos: np.ndarray
    yaw: float
    size: np.ndarray
    quat: np.ndarray | None = None  # (w, x, y, z), world-from-gate; None -> derived from yaw


class FlightmareExpertEnv:
    """Privileged-expert collection env: motor input, RGB image slots + full state out."""

    def __init__(
        self,
        image_size: int = 224,
        cameras: tuple[str, ...] = ("forward",),
        control_hz: float = 100.0,
        params: QuadParams = QuadParams(),
        scene: str = "industrial",
        render: bool = True,
        seed: int = 1,
        visual_backend: bool = False,
        backend: str = "auto",
    ):
        self.image_size = image_size
        self.cameras = tuple(cameras)
        self.dt = 1.0 / float(control_hz)
        self.params = params
        self.scene = scene
        self._render = render
        self.seed = int(seed)
        self.visual_backend = bool(visual_backend)
        self.backend = str(backend)
        self._impl = self._build_impl()

    def _build_impl(self):
        try:
            if self.backend == "numpy":
                # DEPRECATED: this is NOT a Flightmare backend. It is a hand-rolled
                # rigid-body integrator with no motor lag, no aero, and no Unity. It
                # exists only as a smoke-test fallback for environments where the
                # Flightmare bindings cannot be loaded. Do NOT use it for BC data
                # collection, PPO training, or evaluation that will appear in the
                # paper — switch to ``flightgym`` (headless Flightmare C++ dynamics)
                # or ``visual`` (Flightmare + Unity rendering).
                raise RuntimeError(
                    "backend='numpy' is deprecated and intentionally disabled. "
                    "It is not a Flightmare backend — it's a stripped-down rigid-body "
                    "fallback that bypasses motor dynamics, the thrust map, and aero. "
                    "Use backend='flightgym' (headless real dynamics) or 'visual' (with Unity). "
                    "If you are intentionally running a smoke test outside the Docker image "
                    "where flightgym is unavailable, set ALPHA_NEWTON_ALLOW_NUMPY_FALLBACK=1."
                )
            # Both 'flightgym' (headless) and 'visual' (with Unity rendering)
            # route to the same VisualRacingEnv_v0 binding (see
            # patch_flightmare_pybind.py). It exposes setState/stepCtbr/stepMotor
            # which we need; the bare QuadrotorEnv_v1 wrapper does not. The only
            # difference between the two backends is whether connectUnity() is
            # called and whether RGB frames are read out — both controlled by
            # the ``render`` flag passed below.
            if self.backend == "flightgym":
                impl_cls = _VisualFlightmareImpl
            elif self.backend == "visual":
                impl_cls = _VisualFlightmareImpl
            elif self.backend == "auto":
                impl_cls = _VisualFlightmareImpl
            else:
                raise ValueError(f"Unknown FlightmareExpertEnv backend={self.backend!r}")
            return impl_cls(
                image_size=self.image_size,
                cameras=self.cameras,
                dt=self.dt,
                params=self.params,
                scene=self.scene,
                render=self._render,
                seed=self.seed,
            )
        except Exception as e:  # pragma: no cover - depends on host install
            if os.environ.get("ALPHA_NEWTON_ALLOW_NUMPY_FALLBACK", "0") != "1":
                raise
            warnings.warn(
                f"Flightmare bindings unavailable ({e!r}); ALPHA_NEWTON_ALLOW_NUMPY_FALLBACK=1 "
                "is set so falling back to deprecated pure-numpy rigid-body sim with blank "
                "images. NOT VALID for any data that will be reported in the paper.",
                stacklevel=2,
            )
            return _NumpyFallbackImpl(
                image_size=self.image_size,
                cameras=self.cameras,
                dt=self.dt,
                params=self.params,
            )

    @property
    def using_fallback(self) -> bool:
        return isinstance(self._impl, _NumpyFallbackImpl)

    def reset(self, init_pos: np.ndarray, yaw: float = 0.0) -> StepResult:
        return self._impl.reset(np.asarray(init_pos, dtype=np.float64), float(yaw))

    def step_ctbr(self, thrust_newton: float, omega_des: np.ndarray) -> StepResult:
        return self._impl.step_ctbr(float(thrust_newton), np.asarray(omega_des, dtype=np.float64))

    def step_motor(self, motor_normalized: np.ndarray) -> StepResult:
        return self._impl.step_motor(np.asarray(motor_normalized, dtype=np.float64))

    def add_gates(self, gates: list[GateSpec]) -> None:
        if hasattr(self._impl, "add_gates"):
            self._impl.add_gates(gates)

    def close(self) -> None:
        self._impl.close()


# ---------------------------------------------------------------------------
# flightgym backed implementation
# ---------------------------------------------------------------------------
def _quat_from_euler_zyx(euler_zyx: np.ndarray) -> np.ndarray:
    """Flightmare obs uses Euler ZYX; return quaternion as w, x, y, z."""
    yaw, pitch, roll = euler_zyx
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


class _FlightgymImpl:
    def __init__(self, image_size, cameras, dt, params, scene, render, seed):
        import flightgym  # type: ignore

        self.dt = dt
        self.cameras = cameras
        self.image_size = image_size
        self.params = params
        self._zero_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        scene_id = _SCENE_IDS.get(str(scene).lower(), 0)
        self._runtime_root = Path(tempfile.mkdtemp(prefix="flightmare_cfg_"))
        cfg_dir = self._runtime_root / "flightlib" / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "quadrotor_env.yaml").write_text(f"""
quadrotor_env:
   camera: no
   sim_dt: {float(dt)}
   max_t: 5.0
   add_camera: no

quadrotor_dynamics:
  mass: {float(params.mass)}
  arm_l: {float(params.arm_length)}
  motor_omega_min: 150.0
  motor_omega_max: 3000.0
  motor_tau: 0.0001
  thrust_map: [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]
  kappa: 0.016
  omega_max: [6.0, 6.0, 6.0]

rl:
  pos_coeff: -0.002
  ori_coeff: -0.002
  lin_vel_coeff: -0.0002
  ang_vel_coeff: -0.0002
  act_coeff: -0.0002
""")
        self._old_flightmare_path = os.environ.get("FLIGHTMARE_PATH")
        os.environ["FLIGHTMARE_PATH"] = str(self._runtime_root)
        cfg = f"""
env:
  seed: {int(seed)}
  scene_id: {scene_id}
  num_envs: 1
  num_threads: 1
  render: {"yes" if render else "no"}
"""
        self.env = flightgym.QuadrotorEnv_v1(cfg, False)
        self.env.setSeed(int(seed))
        self.obs = np.zeros((1, self.env.getObsDim()), dtype=np.float32)
        self.act = np.zeros((1, self.env.getActDim()), dtype=np.float32)
        self.reward = np.zeros((1,), dtype=np.float32)
        self.done = np.zeros((1,), dtype=bool)
        self.extra = np.zeros((1, len(self.env.getExtraInfoNames())), dtype=np.float32)

        if render:
            connected = self.env.connectUnity()
            if not connected:
                warnings.warn(
                    "Flightmare Unity did not connect; collecting state/action data with blank images.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Flightmare's installed Python binding renders for visualization but does not "
                    "return RGB frames; HDF5 image datasets will be blank unless a custom binding is added.",
                    stacklevel=2,
                )

        self._render = render

    def reset(self, init_pos, yaw):
        # Flightgym's QuadrotorEnv_v1 reset() ignores init_pos/yaw and may spawn
        # the quad with a non-level attitude. Run a brief Python-side stabilize
        # loop (level-attitude PD on quat + body-rate damping) so the policy
        # always sees a settled hover state. Caller is expected to translate
        # the course origin to the resulting pos (flightgym never moves to the
        # requested init_pos — the installed binding doesn't expose setState).
        self.env.reset(self.obs)
        self._last_omega = self.obs[0, 9:12].astype(np.float64)
        hover_thrust = float(self.params.mass * 9.81)
        for _ in range(60):
            row = self.obs[0].astype(np.float64)
            quat = _quat_from_euler_zyx(row[3:6])
            omega = row[9:12]
            R = quat_to_R(quat)
            b3 = R[:, 2]
            # Drive thrust axis toward world +z. Cross product gives the body-rate
            # axis that rotates b3 onto [0,0,1]; magnitude is sin(angle).
            e_axis = np.cross(b3, np.array([0.0, 0.0, 1.0]))
            kp_att = 6.0
            kd_omega = 0.5
            omega_des = kp_att * e_axis - kd_omega * omega
            motor = self._ctbr_to_motor(hover_thrust, omega_des)
            self.act[0] = np.clip(2.0 * motor - 0.5, -0.5, 1.5).astype(np.float32)
            self.env.step(self.act, self.obs, self.reward, self.done, self.extra)
            self._last_omega = self.obs[0, 9:12].astype(np.float64)
        return self._observe(done=False)

    def step_ctbr(self, thrust_newton, omega_des):
        # Flightgym's QuadrotorEnv_v1 only accepts per-motor thrusts, so we run
        # the same X-mixer + body-rate inner loop the codebase uses for motor BC
        # (see controllers.py:GeometricSE3Controller._mix_to_motors). Inputs:
        #   thrust_newton: collective thrust in Newtons.
        #   omega_des:     desired body angular velocity (rad/s), 3-vector.
        motor_normalized = self._ctbr_to_motor(float(thrust_newton), np.asarray(omega_des, dtype=np.float64))
        return self.step_motor(motor_normalized)

    def _ctbr_to_motor(self, thrust_newton: float, omega_des: np.ndarray) -> np.ndarray:
        I = np.asarray(self.params.inertia, dtype=np.float64)
        omega = self._last_omega
        omega_err = omega_des - omega
        # Inner-loop body-rate gains match GeometricGains.kp_rate so the
        # flightgym dynamics see the same rate-loop characteristics as the
        # numpy/visual paths' motor labels were generated with.
        kp_rate = np.array([28.0, 28.0, 12.0], dtype=np.float64)
        alpha_des = kp_rate * omega_err
        gyro = np.cross(omega, I * omega)
        tau = I * alpha_des + gyro
        L = self.params.arm_length
        kT = self.params.k_thrust
        kQ = self.params.k_torque
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

    def step_motor(self, motor_normalized):
        # Flightgym's bundled wrapper has its own motor-action normalization.
        # The numpy/visual backends use motor_normalized as f_i / f_i,max.
        self.act[0] = np.clip(2.0 * motor_normalized - 0.5, -0.5, 1.5).astype(np.float32)
        self.env.step(self.act, self.obs, self.reward, self.done, self.extra)
        return self._observe(done=bool(self.done[0]))

    def _observe(self, done):
        row = self.obs[0].astype(np.float64)
        pos = row[0:3]
        quat = _quat_from_euler_zyx(row[3:6])
        vel = row[6:9]
        omega = row[9:12]
        self._last_omega = omega.copy()
        images = {cam_name: self._zero_img.copy() for cam_name in self.cameras}
        return StepResult(pos=pos, vel=vel, quat=quat, omega=omega, images=images, done=done)

    def close(self):
        try:
            self.env.disconnectUnity()
            self.env.close()
            if self._old_flightmare_path is not None:
                os.environ["FLIGHTMARE_PATH"] = self._old_flightmare_path
        except Exception:
            pass


class _VisualFlightmareImpl:
    def __init__(self, image_size, cameras, dt, params, scene, render, seed):
        import flightgym  # type: ignore

        if not hasattr(flightgym, "VisualRacingEnv_v0"):
            raise RuntimeError(
                "flightgym.VisualRacingEnv_v0 missing. Rebuild the Docker image "
                "after applying scripts/flightmare_bc/patch_flightmare_pybind.py."
            )
        self.dt = dt
        self.params = params
        self.cameras = cameras
        self.image_size = image_size
        self.env = flightgym.VisualRacingEnv_v0(
            int(_SCENE_IDS.get(str(scene).lower(), 1)),
            int(image_size),
            int(image_size),
            90.0,
        )
        self.render = bool(render)
        self.connected = False

    def add_gates(self, gates: list[GateSpec]) -> None:
        for gate in gates:
            self.env.addGate(
                gate.gate_id,
                np.asarray(gate.pos, dtype=np.float32),
                float(gate.yaw),
                np.asarray(gate.size, dtype=np.float32),
            )

    def reset(self, init_pos, yaw):
        if self.render and not self.connected:
            self.connected = bool(self.env.connectUnity())
            if not self.connected:
                warnings.warn(
                    "Flightmare Unity did not connect; real RGB frames will be blank.",
                    stacklevel=2,
                )
        quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)
        self.env.reset(np.asarray(init_pos, dtype=np.float32), quat)
        if self.render:
            self.env.render()
        return self._observe(done=False)

    def step_ctbr(self, thrust_newton, omega_des):
        self.env.stepCtbr(
            float(thrust_newton),
            np.asarray(omega_des, dtype=np.float32),
            float(self.dt),
        )
        if self.render:
            self.env.render()
        return self._observe(done=False)

    def step_motor(self, motor_normalized):
        per_motor_max = float(self.params.max_collective_thrust / 4.0)
        thrusts = np.asarray(motor_normalized, dtype=np.float32) * per_motor_max
        self.env.stepMotor(thrusts, float(self.dt))
        if self.render:
            self.env.render()
        return self._observe(done=False)

    def _observe(self, done):
        row = np.asarray(self.env.getState(), dtype=np.float64)
        pos = row[0:3]
        quat = row[3:7]
        vel = row[7:10]
        omega = row[10:13]
        images = {}
        if self.cameras:
            rgb = np.asarray(self.env.getRGB(), dtype=np.uint8)
            images = {cam_name: rgb.copy() for cam_name in self.cameras}
        return StepResult(pos=pos, vel=vel, quat=quat, omega=omega, images=images, done=done)

    def close(self):
        try:
            self.env.disconnectUnity()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pure-numpy rigid-body fallback (no images)
# ---------------------------------------------------------------------------
class _NumpyFallbackImpl:
    def __init__(self, image_size, cameras, dt, params):
        self.image_size = image_size
        self.cameras = cameras
        self.dt = dt
        self.params = params
        self._zero_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)

    def reset(self, init_pos, yaw):
        self.pos = init_pos.copy()
        self.vel = np.zeros(3)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        self.quat = np.array([cy, 0.0, 0.0, sy])
        self.omega = np.zeros(3)
        return self._observe()

    def step_ctbr(self, thrust_newton, omega_des):
        m, g = self.params.mass, 9.81
        R = quat_to_R(self.quat)
        accel = R @ np.array([0.0, 0.0, thrust_newton / m]) - np.array([0.0, 0.0, g])
        self.vel = self.vel + accel * self.dt
        self.pos = self.pos + self.vel * self.dt
        # First-order omega tracking (perfect inner-loop assumption).
        self.omega = omega_des
        # Quaternion integration: q_dot = 0.5 * Omega(omega) * q
        wx, wy, wz = omega_des
        Omega = 0.5 * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0,  wz, -wy],
            [wy, -wz, 0,  wx],
            [wz, wy, -wx, 0],
        ])
        self.quat = self.quat + Omega @ self.quat * self.dt
        self.quat /= np.linalg.norm(self.quat) + 1e-9
        return self._observe()

    def step_motor(self, motor_normalized):
        motors = np.clip(np.asarray(motor_normalized, dtype=np.float64), 0.0, 1.0)
        f = motors * float(self.params.max_collective_thrust / 4.0)
        thrust_newton = float(f.sum())

        L = float(self.params.arm_length)
        s = np.sqrt(0.5)
        kappa = float(self.params.k_torque / max(self.params.k_thrust, 1e-12))
        tau = np.array([
            L * s * (-f[0] + f[1] + f[2] - f[3]),
            L * s * (-f[0] + f[1] - f[2] + f[3]),
            kappa * (-f[0] - f[1] + f[2] + f[3]),
        ], dtype=np.float64)

        I = np.diag(np.asarray(self.params.inertia, dtype=np.float64))
        Iomega = I @ self.omega
        omega_dot = np.linalg.solve(I, tau - np.cross(self.omega, Iomega))
        self.omega = np.clip(self.omega + omega_dot * self.dt, -40.0, 40.0)

        m, g = self.params.mass, 9.81
        R = quat_to_R(self.quat)
        accel = R @ np.array([0.0, 0.0, thrust_newton / m]) - np.array([0.0, 0.0, g])
        self.vel = self.vel + accel * self.dt
        self.pos = self.pos + self.vel * self.dt

        wx, wy, wz = self.omega
        Omega = 0.5 * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0,  wz, -wy],
            [wy, -wz, 0,  wx],
            [wz, wy, -wx, 0],
        ])
        self.quat = self.quat + Omega @ self.quat * self.dt
        self.quat /= np.linalg.norm(self.quat) + 1e-9
        return self._observe()

    def _observe(self):
        return StepResult(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            quat=self.quat.copy(),
            omega=self.omega.copy(),
            images={c: self._zero_img.copy() for c in self.cameras},
            done=False,
        )

    def close(self):
        return

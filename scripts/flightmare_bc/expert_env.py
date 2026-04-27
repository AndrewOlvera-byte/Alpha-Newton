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
import warnings
from dataclasses import dataclass

import numpy as np

from scripts.flightmare_bc.controllers import QuadParams, quat_to_R


@dataclass
class StepResult:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray            # (w, x, y, z)
    omega: np.ndarray           # body angular velocity
    images: dict[str, np.ndarray]   # {cam_name: uint8 [H, W, 3]}
    done: bool


class FlightmareExpertEnv:
    """Privileged-expert collection env: CTBR input, RGB image + full state out."""

    def __init__(
        self,
        image_size: int = 224,
        cameras: tuple[str, ...] = ("forward",),
        control_hz: float = 100.0,
        params: QuadParams = QuadParams(),
        scene: str = "industrial",
        render: bool = True,
    ):
        self.image_size = image_size
        self.cameras = tuple(cameras)
        self.dt = 1.0 / float(control_hz)
        self.params = params
        self.scene = scene
        self._render = render
        self._impl = self._build_impl()

    def _build_impl(self):
        try:
            return _FlightlibImpl(
                image_size=self.image_size,
                cameras=self.cameras,
                dt=self.dt,
                params=self.params,
                scene=self.scene,
                render=self._render,
            )
        except Exception as e:  # pragma: no cover - depends on host install
            warnings.warn(
                f"Flightmare bindings unavailable ({e!r}); falling back to "
                "pure-numpy rigid-body sim with blank images. Useful only for "
                "pipeline tests, NOT for real BC training data.",
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

    def close(self) -> None:
        self._impl.close()


# ---------------------------------------------------------------------------
# flightlib / flightgym backed implementation
# ---------------------------------------------------------------------------
class _FlightlibImpl:
    def __init__(self, image_size, cameras, dt, params, scene, render):
        from flightgym import VisionEnv_v1  # type: ignore  # noqa: F401
        from flightlib import (  # type: ignore  # noqa: F401
            QuadState,
            Quadrotor,
            UnityBridge,
            Command,
            CommandMode,
        )

        self._QuadState = QuadState
        self._Command = Command
        self._CommandMode = CommandMode

        self.dt = dt
        self.cameras = cameras
        self.image_size = image_size

        self.quad = Quadrotor()
        self.quad.setMass(params.mass)
        self.state = QuadState()

        self.bridge = UnityBridge()
        if render:
            self.bridge.connectUnity(scene)
            self.bridge.addQuadrotor(self.quad)
            for cam_name in cameras:
                self.bridge.addCamera(self.quad, cam_name, image_size, image_size)

        self._render = render

    def reset(self, init_pos, yaw):
        self.state.setZero()
        self.state.x[0:3] = init_pos
        from scripts.flightmare_bc.controllers import quat_to_R  # noqa: F401
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        self.state.q = np.array([cy, 0.0, 0.0, sy])
        self.quad.setState(self.state)
        return self._observe(done=False)

    def step_ctbr(self, thrust_newton, omega_des):
        cmd = self._Command()
        cmd.t = 0.0
        cmd.collective_thrust = float(thrust_newton)
        cmd.omega = omega_des.astype(np.float32)
        cmd.cmd_mode = self._CommandMode.THRUSTRATE
        self.quad.setCommand(cmd)
        self.quad.run(self.dt)
        self.quad.getState(self.state)
        return self._observe(done=False)

    def _observe(self, done):
        pos = np.asarray(self.state.x[0:3], dtype=np.float64)
        vel = np.asarray(self.state.v, dtype=np.float64)
        quat = np.asarray(self.state.q, dtype=np.float64)
        omega = np.asarray(self.state.w, dtype=np.float64)

        images: dict[str, np.ndarray] = {}
        if self._render:
            self.bridge.getRender()
            self.bridge.handleOutput()
            for cam_name in self.cameras:
                img = self.bridge.getImage(cam_name)
                images[cam_name] = np.asarray(img, dtype=np.uint8).reshape(
                    self.image_size, self.image_size, 3
                )
        else:
            for cam_name in self.cameras:
                images[cam_name] = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return StepResult(pos=pos, vel=vel, quat=quat, omega=omega, images=images, done=done)

    def close(self):
        try:
            self.bridge.disconnectUnity()
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

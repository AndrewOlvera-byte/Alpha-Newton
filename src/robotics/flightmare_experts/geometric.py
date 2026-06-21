"""Geometric SE(3) expert as an ExpertBackend.

Thin wrapper around the existing ``GeometricSE3Controller`` so the geometric
bootstrap expert flows through the same backend interface as MPPI and tags its
labels ``geometric_minjerk``.
"""
from __future__ import annotations

import numpy as np

from scripts.flightmare_bc.controllers import GeometricGains, GeometricSE3Controller, QuadParams


class GeometricExpertBackend:
    source = "geometric_minjerk"

    def __init__(self, params: QuadParams | None = None, gains: GeometricGains | None = None):
        self.controller = GeometricSE3Controller(params=params or QuadParams(), gains=gains or GeometricGains())

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
        cmd = self.controller.compute(
            pos=pos, vel=vel, quat=quat,
            pos_des=pos_des, vel_des=vel_des, acc_des=acc_des,
            yaw_des=yaw_des, omega=omega,
        )
        cmd["source"] = self.source
        return cmd

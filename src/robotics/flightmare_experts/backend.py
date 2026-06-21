"""Expert backend protocol.

An expert backend produces a CTBR command (collective thrust + body rates) for
the current state given a reference. Both the geometric SE(3) tracker and the
MPPI controller satisfy this interface, so collection / eval can swap experts
without changing the rollout loop. The command dict matches
``GeometricSE3Controller.compute``::

    {
        "thrust_newton": float,
        "thrust_normalized": float in [0, 1],
        "body_rates": np.ndarray(3,),     # rad/s
        "motor_normalized": np.ndarray(4,),
        "source": str,                    # expert backend tag for the label schema
    }
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ExpertBackend(Protocol):
    source: str

    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        pos_des: np.ndarray,
        vel_des: np.ndarray,
        acc_des: np.ndarray,
        yaw_des: float,
        omega: np.ndarray | None = ...,
    ) -> dict:
        ...

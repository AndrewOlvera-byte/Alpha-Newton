"""Safety supervisor for expert labels.

Rejects (or flags for fallback) labels whose command is non-finite, saturates
the actuators beyond a tolerance, or exceeds the configured body-rate ceiling.
Used by MPPI collection so a bad solve never writes a label silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyVerdict:
    ok: bool
    reason: str = ""
    thrust_saturated: bool = False
    rate_saturated: bool = False


class SafetySupervisor:
    def __init__(
        self,
        max_body_rate: float = 18.0,
        thrust_sat_tol: float = 0.02,
        max_thrust_sat_frac: float = 1.0,
    ):
        self.max_body_rate = float(max_body_rate)
        self.thrust_sat_tol = float(thrust_sat_tol)
        self.max_thrust_sat_frac = float(max_thrust_sat_frac)

    def check(self, command: dict) -> SafetyVerdict:
        thrust = float(command.get("thrust_normalized", 0.0))
        rates = np.asarray(command.get("body_rates", np.zeros(3)), dtype=np.float64)
        motor = np.asarray(command.get("motor_normalized", np.zeros(4)), dtype=np.float64)
        if not (np.isfinite(thrust) and np.all(np.isfinite(rates)) and np.all(np.isfinite(motor))):
            return SafetyVerdict(False, "nonfinite_command")
        rate_sat = bool(np.any(np.abs(rates) > self.max_body_rate + 1e-6))
        if rate_sat:
            return SafetyVerdict(False, "body_rate_exceeded", rate_saturated=True)
        thrust_sat = bool(thrust <= self.thrust_sat_tol or thrust >= 1.0 - self.thrust_sat_tol)
        return SafetyVerdict(True, "", thrust_saturated=thrust_sat, rate_saturated=False)

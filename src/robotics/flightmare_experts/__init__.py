"""Expert backends for Flightmare BC label generation.

- ``GeometricExpertBackend``: the existing SE(3) min-jerk tracker (bootstrap).
- ``MPPIController``: a sampling-based receding-horizon CTBR expert for
  high-quality high-speed labels.
- ``SafetySupervisor``: rejects/flags unsafe expert commands.

``build_expert_controller(cfg, params)`` selects the backend from a collection
config (``expert.controller`` + ``mpcc`` block).
"""
from __future__ import annotations

from typing import Any

from scripts.flightmare_bc.controllers import GeometricGains, GeometricSE3Controller, QuadParams
from src.robotics.flightmare_experts.geometric import GeometricExpertBackend
from src.robotics.flightmare_experts.labels import ExpertLabel, HybridExpertLabeler
from src.robotics.flightmare_experts.mppi import MPPIConfig, MPPIController
from src.robotics.flightmare_experts.safety import SafetySupervisor, SafetyVerdict

__all__ = [
    "GeometricExpertBackend",
    "MPPIController",
    "MPPIConfig",
    "ExpertLabel",
    "HybridExpertLabeler",
    "SafetySupervisor",
    "SafetyVerdict",
    "build_expert_controller",
]


def build_expert_controller(cfg: dict, params: QuadParams):
    """Return an expert backend (compatible with GeometricSE3Controller.compute).

    ``expert.controller == 'geometric_minjerk'`` -> GeometricSE3Controller.
    ``expert.controller == 'mpcc'`` (with ``mpcc.enabled``) -> MPPIController.
    """
    expert = cfg.get("expert", {}) or {}
    controller = str(expert.get("controller", "geometric_minjerk"))
    if controller == "mpcc":
        mpcc = cfg.get("mpcc", {}) or {}
        if not bool(mpcc.get("enabled", False)):
            raise ValueError("expert.controller='mpcc' requires mpcc.enabled=true.")
        mppi_kwargs: dict[str, Any] = dict(mpcc.get("mppi", {}) or {})
        config = MPPIConfig(**mppi_kwargs) if mppi_kwargs else MPPIConfig()
        seed = int(cfg.get("seed", 0))
        return MPPIController(params=params, config=config, seed=seed)

    gains_cfg = expert.get("gains", {}) or {}
    if not gains_cfg:
        return GeometricSE3Controller(params=params)
    gains = GeometricGains(
        kp_pos=tuple(float(x) for x in gains_cfg.get("kp_pos", GeometricGains.kp_pos)),
        kd_pos=tuple(float(x) for x in gains_cfg.get("kd_pos", GeometricGains.kd_pos)),
        kp_att=tuple(float(x) for x in gains_cfg.get("kp_att", GeometricGains.kp_att)),
        kp_rate=tuple(float(x) for x in gains_cfg.get("kp_rate", GeometricGains.kp_rate)),
    )
    return GeometricSE3Controller(params=params, gains=gains)

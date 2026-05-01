"""Flightmare drone visuomotor policies.

Importing this package registers all Flightmare architectures with the
``architecture`` registry (``flightmare_waypoint_*``, ``flightmare_ctbr_*``,
``flightmare_motor_*``, ``flightmare_mlp_actor``).
"""
from src.robotics.models.flightmare import actors  # noqa: F401  (registers builders)
from src.robotics.models.flightmare.GaussianActionExpert import GaussianActionExpert
from src.robotics.models.flightmare.MLP import MLPFusion, TinyVisionEncoder
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import (
    MLPFusionGaussianExpertActor,
)

__all__ = [
    "GaussianActionExpert",
    "MLPFusion",
    "MLPFusionGaussianExpertActor",
    "TinyVisionEncoder",
    "actors",
]

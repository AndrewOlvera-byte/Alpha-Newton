"""Scenario + distribution dataclasses for procedural Flightmare courses.

A ``CourseDistributionConfig`` is a weighted list of ``ScenarioConfig`` plus
global geometry bounds. ``sample_course`` (see ``generator.py``) picks a
scenario family by weight, generates gates, validates the geometry, and returns
a ``GeneratedCourse`` carrying the gates, waypoints, and per-episode metadata.

The same distribution object is shared by BC collection, controller/policy
eval, and PPO env reset so every consumer samples from one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np


GENERATOR_VERSION = "courses_v1"


class GateRole(str, Enum):
    """Semantic role of a gate within a generated course."""

    ENTRY = "entry"
    FLOW = "flow"
    CHICANE = "chicane"
    SPLIT_S_TOP = "split_s_top"
    INVERTED = "inverted"
    FINISH = "finish"


# Scenario families understood by the generator.
SCENARIO_FAMILIES = (
    "swift_v4_canonical",
    "swift_v4_noisy",
    "procedural_flow",
    "procedural_chicane",
    "procedural_split_s",
    "mixed_inverted",
    "fixed_layout_bank",
)


@dataclass
class CourseBounds:
    """Global geometry validity envelope shared by every scenario.

    All distances are meters, angles radians. A course is rejected (and
    resampled) by ``validate_course`` if it violates any of these.
    """

    arena_radius: float = 60.0          # |gate_xy_center| must stay within this
    z_min: float = 1.0                  # gate altitude floor
    z_max: float = 8.0                  # gate altitude ceiling
    min_gate_clearance_m: float = 0.6   # extra margin beyond gate half-width between any two gates
    min_consecutive_m: float = 3.0      # min spacing between adjacent gates
    max_consecutive_m: float = 14.0     # max spacing between adjacent gates
    max_climb_angle_rad: float = 1.0472   # 60 deg: max |pitch| of a segment between gates
    max_heading_change_rad: float = 1.9199  # 110 deg: max turn between adjacent segments
    total_length_range: Sequence[float] = (25.0, 140.0)  # min/max path length through gate centers
    vehicle_radius: float = 0.15        # aperture clearance used in validation

    def as_dict(self) -> dict[str, Any]:
        return {
            "arena_radius": float(self.arena_radius),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "min_gate_clearance_m": float(self.min_gate_clearance_m),
            "min_consecutive_m": float(self.min_consecutive_m),
            "max_consecutive_m": float(self.max_consecutive_m),
            "max_climb_angle_rad": float(self.max_climb_angle_rad),
            "max_heading_change_rad": float(self.max_heading_change_rad),
            "total_length_range": [float(self.total_length_range[0]), float(self.total_length_range[1])],
            "vehicle_radius": float(self.vehicle_radius),
        }


@dataclass
class ScenarioConfig:
    """One scenario family with its sampling ranges.

    ``family`` selects the generator branch; ``weight`` is the (unnormalized)
    probability of picking this scenario. The remaining fields parameterize the
    procedural generators; not all fields apply to every family (e.g.
    ``gate_layouts`` is only used by ``fixed_layout_bank``).
    """

    name: str
    family: str
    weight: float = 1.0
    num_gates_range: Sequence[int] = (5, 9)
    spacing_range: Sequence[float] = (5.0, 10.0)
    z_range: Sequence[float] = (1.5, 4.5)
    gate_size: float = 1.6
    gate_approach_m: float = 3.0
    # Heading dynamics (procedural families).
    heading_step_range: Sequence[float] = (0.0, 0.7)   # |per-gate heading change|
    lateral_jitter_m: float = 1.5
    # Inverted gate control (split_s / mixed_inverted).
    num_inverted_range: Sequence[int] = (0, 0)
    inverted_roll_jitter_rad: float = 0.2618           # ~15 deg around 180
    # Per-gate noise applied after layout.
    pos_noise_xyz: Sequence[float] = (0.0, 0.0, 0.0)
    yaw_noise_rad: float = 0.0
    # Travel direction: if True, the start heading is randomized over [0, 2pi).
    random_direction: bool = True
    # Fixed-layout bank: list of JSON layout paths to sample from.
    gate_layouts: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.family not in SCENARIO_FAMILIES:
            raise ValueError(
                f"ScenarioConfig {self.name!r}: unknown family {self.family!r}; "
                f"expected one of {SCENARIO_FAMILIES}"
            )
        if float(self.weight) < 0.0:
            raise ValueError(f"ScenarioConfig {self.name!r}: weight must be >= 0")
        lo, hi = int(self.num_gates_range[0]), int(self.num_gates_range[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"ScenarioConfig {self.name!r}: invalid num_gates_range {self.num_gates_range}")
        ilo, ihi = int(self.num_inverted_range[0]), int(self.num_inverted_range[1])
        if ilo < 0 or ihi < ilo:
            raise ValueError(f"ScenarioConfig {self.name!r}: invalid num_inverted_range {self.num_inverted_range}")
        if self.family == "fixed_layout_bank" and not self.gate_layouts:
            raise ValueError(f"ScenarioConfig {self.name!r}: fixed_layout_bank requires gate_layouts")


@dataclass
class CourseDistributionConfig:
    """Weighted mixture of scenarios + shared geometry bounds."""

    scenarios: Sequence[ScenarioConfig]
    bounds: CourseBounds = field(default_factory=CourseBounds)
    max_resample_tries: int = 24

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("CourseDistributionConfig requires at least one scenario")
        total = sum(float(s.weight) for s in self.scenarios)
        if total <= 0.0:
            raise ValueError("CourseDistributionConfig: scenario weights sum to 0")

    @property
    def weights(self) -> np.ndarray:
        w = np.asarray([float(s.weight) for s in self.scenarios], dtype=np.float64)
        return w / w.sum()

    def pick_scenario(self, rng: np.random.Generator) -> ScenarioConfig:
        idx = int(rng.choice(len(self.scenarios), p=self.weights))
        return self.scenarios[idx]


@dataclass
class GeneratedCourse:
    """A fully-generated, validated course plus metadata for logging."""

    gates: list  # list[GateSpec] (scripts.flightmare_bc.expert_env.GateSpec)
    waypoints: np.ndarray
    scenario_name: str
    scenario_family: str
    scenario_seed: int
    difficulty: float
    path_length_m: float
    num_inverted_gates: int
    gate_roles: list[str]
    noise_params: dict[str, Any]
    course_bounds: dict[str, Any]
    generator_version: str = GENERATOR_VERSION

    def metadata(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_family": self.scenario_family,
            "scenario_seed": int(self.scenario_seed),
            "difficulty": float(self.difficulty),
            "path_length_m": float(self.path_length_m),
            "num_inverted_gates": int(self.num_inverted_gates),
            "gate_roles": list(self.gate_roles),
            "noise_params": dict(self.noise_params),
            "course_bounds": dict(self.course_bounds),
            "generator_version": self.generator_version,
        }


def default_course_distribution() -> CourseDistributionConfig:
    """A balanced default mixture covering all major scenario families.

    Used as the v7 collection / PPO default: a mix of upright procedural flow,
    chicanes, Split-S, mixed-inverted, and the canonical/noisy swift_v4 anchors.
    """
    # Wide swift-scale spacing (>~10 m) so the geometric bootstrap expert tracks
    # cleanly (~0.02 m) and clears gate apertures; tight spacing creates
    # untrackable curvature. Verified flyable in Flightmare at ~2 m/s.
    scenarios = [
        ScenarioConfig(
            name="flow_upright",
            family="procedural_flow",
            weight=0.30,
            num_gates_range=(5, 7),
            spacing_range=(10.0, 13.0),
            z_range=(1.8, 4.0),
            heading_step_range=(0.0, 0.3),
            lateral_jitter_m=1.0,
            pos_noise_xyz=(0.12, 0.12, 0.08),
            yaw_noise_rad=0.04,
        ),
        ScenarioConfig(
            name="chicane",
            family="procedural_chicane",
            weight=0.20,
            num_gates_range=(5, 7),
            spacing_range=(10.0, 13.0),
            z_range=(1.8, 3.5),
            heading_step_range=(0.25, 0.45),
            lateral_jitter_m=0.8,
            pos_noise_xyz=(0.12, 0.12, 0.08),
            yaw_noise_rad=0.04,
        ),
        ScenarioConfig(
            name="split_s",
            family="procedural_split_s",
            weight=0.20,
            num_gates_range=(6, 7),
            spacing_range=(10.0, 13.0),
            z_range=(2.0, 4.5),
            num_inverted_range=(1, 1),
            inverted_roll_jitter_rad=0.2618,
            pos_noise_xyz=(0.12, 0.12, 0.08),
            yaw_noise_rad=0.04,
        ),
        ScenarioConfig(
            name="mixed_inverted",
            family="mixed_inverted",
            weight=0.15,
            num_gates_range=(5, 7),
            spacing_range=(10.0, 13.0),
            z_range=(2.0, 4.5),
            num_inverted_range=(0, 2),
            inverted_roll_jitter_rad=0.2618,
            pos_noise_xyz=(0.12, 0.12, 0.08),
            yaw_noise_rad=0.04,
        ),
        ScenarioConfig(
            name="swift_v4_canonical",
            family="swift_v4_canonical",
            weight=0.07,
            num_gates_range=(7, 7),
            num_inverted_range=(1, 1),
            pos_noise_xyz=(0.0, 0.0, 0.0),
            yaw_noise_rad=0.0,
            random_direction=False,
        ),
        ScenarioConfig(
            name="swift_v4_noisy",
            family="swift_v4_noisy",
            weight=0.08,
            num_gates_range=(7, 7),
            num_inverted_range=(1, 1),
            inverted_roll_jitter_rad=0.2618,
            pos_noise_xyz=(0.20, 0.20, 0.15),
            yaw_noise_rad=0.05,
            random_direction=False,
        ),
    ]
    # Bounds are a safety net wide enough to admit the legitimate fixed
    # swift_v4 course (z up to 5.5, ~80 deg turn at the loop); the procedural
    # scenario PARAMETERS above (wide spacing, gentle heading, smooth z) keep
    # generated courses well inside these and reliably flyable.
    bounds = CourseBounds(
        arena_radius=80.0,
        z_max=6.0,
        min_consecutive_m=7.0,
        max_consecutive_m=16.0,
        max_climb_angle_rad=1.0472,
        max_heading_change_rad=1.7453,
        total_length_range=(40.0, 135.0),
    )
    return CourseDistributionConfig(scenarios=scenarios, bounds=bounds)

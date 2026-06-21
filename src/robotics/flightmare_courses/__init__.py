"""Shared procedural course generation for Flightmare BC + PPO.

One source of truth for course sampling used identically by BC collection,
controller/policy eval, and PPO env reset.

Public API:
    sample_course(rng, dist) -> GeneratedCourse
    validate_course(gates, bounds, waypoints=None) -> (ok, reasons)
    CourseDistributionConfig, ScenarioConfig, CourseBounds, GeneratedCourse, GateRole
    default_course_distribution()
    course_distribution_from_dict(d) -> CourseDistributionConfig
"""
from __future__ import annotations

from typing import Any

from src.robotics.flightmare_courses.generator import generate_scenario, sample_course
from src.robotics.flightmare_courses.scenarios import (
    GENERATOR_VERSION,
    SCENARIO_FAMILIES,
    CourseBounds,
    CourseDistributionConfig,
    GateRole,
    GeneratedCourse,
    ScenarioConfig,
    default_course_distribution,
)
from src.robotics.flightmare_courses.validator import path_length, validate_course

__all__ = [
    "sample_course",
    "generate_scenario",
    "validate_course",
    "path_length",
    "CourseBounds",
    "CourseDistributionConfig",
    "ScenarioConfig",
    "GeneratedCourse",
    "GateRole",
    "default_course_distribution",
    "course_distribution_from_dict",
    "GENERATOR_VERSION",
    "SCENARIO_FAMILIES",
]


def course_distribution_from_dict(d: dict[str, Any] | None) -> CourseDistributionConfig | None:
    """Build a CourseDistributionConfig from a plain config dict (YAML/manifest).

    Returns ``None`` when ``d`` is falsy so callers can fall back to the legacy
    single-course path. Schema::

        course_distribution:
          max_resample_tries: 24
          bounds: { arena_radius: 60.0, z_min: 1.0, ... }
          scenarios:
            - { name: flow, family: procedural_flow, weight: 0.3, ... }
    """
    if not d:
        return None
    if isinstance(d, CourseDistributionConfig):
        return d
    bounds_d = dict(d.get("bounds", {}) or {})
    bounds = CourseBounds(**bounds_d) if bounds_d else CourseBounds()
    scenarios_raw = d.get("scenarios")
    if not scenarios_raw:
        raise ValueError("course_distribution requires a non-empty 'scenarios' list")
    scenarios = [s if isinstance(s, ScenarioConfig) else ScenarioConfig(**dict(s)) for s in scenarios_raw]
    kwargs: dict[str, Any] = {"scenarios": scenarios, "bounds": bounds}
    if "max_resample_tries" in d:
        kwargs["max_resample_tries"] = int(d["max_resample_tries"])
    return CourseDistributionConfig(**kwargs)

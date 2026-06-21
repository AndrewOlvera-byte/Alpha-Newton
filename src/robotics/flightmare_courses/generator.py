"""Procedural course generators + the top-level ``sample_course`` entry point.

Each scenario family builds a ``list[GateSpec]`` (the canonical gate spec from
``scripts.flightmare_bc.expert_env``). ``sample_course`` picks a family by
weight, generates gates, validates the geometry (reject + resample with bounded
retries), builds approach/center/exit waypoints, and attaches metadata.

Small layout primitives are reused from ``scripts.flightmare_bc.collect``
(``_quat_from_yaw_pitch_roll``, ``_compute_path_yaws``, ``waypoints_from_gates``,
``_swift_v4_gate_course``, ``_layout_gate_course``) via lazy import to avoid an
import cycle (``collect`` dispatches into this module).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.robotics.flightmare_courses.scenarios import (
    CourseBounds,
    CourseDistributionConfig,
    GateRole,
    GeneratedCourse,
    ScenarioConfig,
)
from src.robotics.flightmare_courses.validator import path_length, validate_course


def _collect_helpers():
    """Lazy import of collect.py primitives (breaks the import cycle)."""
    from scripts.flightmare_bc import collect as _c  # noqa: PLC0415

    return _c


def _gate_spec_cls():
    from scripts.flightmare_bc.expert_env import GateSpec  # noqa: PLC0415

    return GateSpec


def _sample_int(rng: np.random.Generator, rng_pair) -> int:
    lo, hi = int(rng_pair[0]), int(rng_pair[1])
    return int(rng.integers(lo, hi + 1))


def _sample_float(rng: np.random.Generator, rng_pair) -> float:
    lo, hi = float(rng_pair[0]), float(rng_pair[1])
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _make_gate(prefix: str, i: int, center, yaw: float, size: float, roll: float = 0.0):
    helpers = _collect_helpers()
    GateSpec = _gate_spec_cls()
    quat = helpers._quat_from_yaw_pitch_roll(float(yaw), 0.0, float(roll))
    return GateSpec(
        gate_id=f"{prefix}_{i:03d}",
        pos=np.asarray(center, dtype=np.float64).copy(),
        yaw=float(yaw),
        size=np.array([size, size, size], dtype=np.float64),
        quat=quat,
    )


def _apply_pos_yaw_noise(rng, centers, yaws, sc: ScenarioConfig, bounds: CourseBounds):
    sigmas = np.asarray(sc.pos_noise_xyz, dtype=np.float64).reshape(-1)
    if sigmas.size == 3 and np.any(sigmas > 0.0):
        centers = centers + rng.normal(0.0, 1.0, size=centers.shape) * sigmas[None, :]
        centers[:, 2] = np.clip(centers[:, 2], bounds.z_min, bounds.z_max)
    if sc.yaw_noise_rad > 0.0:
        yaws = [y + float(rng.uniform(-sc.yaw_noise_rad, sc.yaw_noise_rad)) for y in yaws]
    return centers, yaws


def _march_centers(rng, sc: ScenarioConfig, bounds: CourseBounds, *, chicane: bool):
    """Forward-marching gate centers with bounded heading + altitude changes.

    Altitude follows a bounded random walk (``max_dz`` per gate, capped at the
    spacing-implied climb angle) rather than independent sampling, so adjacent
    gates never teleport in height — realistic flyable shapes.
    """
    n = _sample_int(rng, sc.num_gates_range)
    z0 = _sample_float(rng, sc.z_range)
    z = float(np.clip(z0, bounds.z_min, bounds.z_max))
    pos = np.array([0.0, 0.0, z], dtype=np.float64)
    heading = float(rng.uniform(0.0, 2.0 * np.pi)) if sc.random_direction else 0.0
    centers = [pos.copy()]
    turn_sign = 1.0
    z_lo, z_hi = float(sc.z_range[0]), float(sc.z_range[1])
    for i in range(1, n):
        spacing = _sample_float(rng, sc.spacing_range)
        step = _sample_float(rng, sc.heading_step_range)
        if chicane:
            heading += turn_sign * step
            turn_sign *= -1.0
        else:
            heading += float(rng.uniform(-step, step))
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        pos = pos + spacing * forward
        if sc.lateral_jitter_m > 0.0:
            pos = pos + float(rng.uniform(-sc.lateral_jitter_m, sc.lateral_jitter_m)) * lateral
        # Smooth altitude random walk, bounded by the climb-angle limit so the
        # min-jerk path between gates stays trackable.
        max_dz = min(0.5 * spacing * np.tan(bounds.max_climb_angle_rad), 0.25 * spacing + 0.5)
        z = float(np.clip(z + rng.uniform(-max_dz, max_dz), z_lo, z_hi))
        z = float(np.clip(z, bounds.z_min, bounds.z_max))
        pos = pos.copy()
        pos[2] = z
        centers.append(pos)
    return np.stack(centers, axis=0)


def gen_procedural_flow(rng, sc, bounds):
    centers = _march_centers(rng, sc, bounds, chicane=False)
    helpers = _collect_helpers()
    yaws = helpers._compute_path_yaws(centers)
    centers, yaws = _apply_pos_yaw_noise(rng, centers, yaws, sc, bounds)
    gates = [_make_gate("flow", i, centers[i], yaws[i], sc.gate_size) for i in range(len(centers))]
    roles = _roles_for(len(gates), inverted_idx=set())
    return gates, roles


def gen_procedural_chicane(rng, sc, bounds):
    centers = _march_centers(rng, sc, bounds, chicane=True)
    helpers = _collect_helpers()
    yaws = helpers._compute_path_yaws(centers)
    centers, yaws = _apply_pos_yaw_noise(rng, centers, yaws, sc, bounds)
    gates = [_make_gate("chicane", i, centers[i], yaws[i], sc.gate_size) for i in range(len(centers))]
    roles = _roles_for(len(gates), inverted_idx=set())
    for i in range(1, len(roles) - 1):
        roles[i] = GateRole.CHICANE.value
    return gates, roles


def gen_procedural_split_s(rng, sc, bounds):
    """Climb -> inverted/rolled top gate -> descent -> recovery finish."""
    centers = _march_centers(rng, sc, bounds, chicane=False)
    n = len(centers)
    # Force a climb into a mid-course top gate, then a descent.
    top = max(1, min(n - 2, n // 2))
    z_top = float(np.clip(sc.z_range[1], bounds.z_min, bounds.z_max))
    centers[top, 2] = z_top
    for i in range(top):
        frac = (i + 1) / (top + 1)
        centers[i, 2] = float(np.clip(bounds.z_min + frac * (z_top - bounds.z_min), bounds.z_min, bounds.z_max))
    for i in range(top + 1, n):
        frac = (i - top) / (n - top)
        centers[i, 2] = float(np.clip(z_top - frac * (z_top - bounds.z_min - 0.5), bounds.z_min, bounds.z_max))
    helpers = _collect_helpers()
    yaws = helpers._compute_path_yaws(centers)
    centers, yaws = _apply_pos_yaw_noise(rng, centers, yaws, sc, bounds)
    gates = []
    for i in range(n):
        roll = 0.0
        if i == top:
            roll = np.pi + float(rng.uniform(-sc.inverted_roll_jitter_rad, sc.inverted_roll_jitter_rad))
        gates.append(_make_gate("split_s", i, centers[i], yaws[i], sc.gate_size, roll=roll))
    roles = _roles_for(n, inverted_idx={top})
    roles[top] = GateRole.SPLIT_S_TOP.value
    return gates, roles


def gen_mixed_inverted(rng, sc, bounds):
    centers = _march_centers(rng, sc, bounds, chicane=False)
    n = len(centers)
    helpers = _collect_helpers()
    yaws = helpers._compute_path_yaws(centers)
    centers, yaws = _apply_pos_yaw_noise(rng, centers, yaws, sc, bounds)
    k = min(_sample_int(rng, sc.num_inverted_range), max(0, n - 2))
    # Invert interior gates only (never entry/finish) to keep the course flyable.
    interior = list(range(1, n - 1)) if n > 2 else []
    inverted_idx = set()
    if k > 0 and interior:
        inverted_idx = set(int(x) for x in rng.choice(interior, size=min(k, len(interior)), replace=False))
    gates = []
    for i in range(n):
        roll = 0.0
        if i in inverted_idx:
            roll = np.pi + float(rng.uniform(-sc.inverted_roll_jitter_rad, sc.inverted_roll_jitter_rad))
        gates.append(_make_gate("mixed", i, centers[i], yaws[i], sc.gate_size, roll=roll))
    roles = _roles_for(n, inverted_idx=inverted_idx)
    return gates, roles


def _swift_v4(rng, sc, bounds, noisy: bool):
    helpers = _collect_helpers()
    cfg = SimpleNamespace(
        num_gates=7,
        gate_size=sc.gate_size,
        z_min=bounds.z_min,
        fixed_gate_pos_noise=(float(max(sc.pos_noise_xyz)) if noisy else 0.0),
        fixed_gate_yaw_noise=(sc.yaw_noise_rad if noisy else 0.0),
        inverted_gate_index=2,
        inverted_roll_jitter_rad=(sc.inverted_roll_jitter_rad if noisy else 0.0),
        gate_clearance_m=bounds.min_gate_clearance_m,
    )
    gates = helpers._swift_v4_gate_course(rng, cfg)
    roles = _roles_for(len(gates), inverted_idx={2})
    roles[2] = GateRole.SPLIT_S_TOP.value
    return gates, roles


def gen_fixed_layout_bank(rng, sc, bounds):
    helpers = _collect_helpers()
    layout = str(rng.choice(list(sc.gate_layouts)))
    cfg = SimpleNamespace(
        course_mode="fixed_gates",
        gate_layout=layout,
        gate_size=sc.gate_size,
        z_min=bounds.z_min,
        random_start_gate=False,
        fixed_gate_pos_noise=float(max(sc.pos_noise_xyz)),
        fixed_gate_yaw_noise=sc.yaw_noise_rad,
    )
    gates = helpers._layout_gate_course(rng, cfg)
    inverted = {i for i, g in enumerate(gates) if _is_inverted(g)}
    roles = _roles_for(len(gates), inverted_idx=inverted)
    return gates, roles


def _is_inverted(gate) -> bool:
    from src.robotics.flightmare_autonomy_fsw.gates import gate_frame  # noqa: PLC0415

    _, _, _, up = gate_frame(gate)
    return bool(up[2] < 0.0)


def _roles_for(n: int, inverted_idx: set[int]) -> list[str]:
    roles = [GateRole.FLOW.value] * n
    if n > 0:
        roles[0] = GateRole.ENTRY.value
        roles[-1] = GateRole.FINISH.value
    for i in inverted_idx:
        if 0 <= i < n:
            roles[i] = GateRole.INVERTED.value
    return roles


_FAMILY_DISPATCH = {
    "procedural_flow": gen_procedural_flow,
    "procedural_chicane": gen_procedural_chicane,
    "procedural_split_s": gen_procedural_split_s,
    "mixed_inverted": gen_mixed_inverted,
    "swift_v4_canonical": lambda rng, sc, b: _swift_v4(rng, sc, b, noisy=False),
    "swift_v4_noisy": lambda rng, sc, b: _swift_v4(rng, sc, b, noisy=True),
    "fixed_layout_bank": gen_fixed_layout_bank,
}


def generate_scenario(rng, sc: ScenarioConfig, bounds: CourseBounds):
    """Generate raw (gates, roles) for one scenario family (no validation)."""
    return _FAMILY_DISPATCH[sc.family](rng, sc, bounds)


def _difficulty(gates, roles, bounds: CourseBounds) -> float:
    """Heuristic 0..1 difficulty: length, inverted count, mean heading change."""
    n = len(gates)
    inv = sum(1 for r in roles if r in (GateRole.INVERTED.value, GateRole.SPLIT_S_TOP.value))
    length = path_length(gates)
    length_term = float(np.clip(length / max(bounds.total_length_range[1], 1.0), 0.0, 1.0))
    inv_term = float(np.clip(inv / max(1, n), 0.0, 1.0))
    return float(np.clip(0.4 * length_term + 0.6 * inv_term, 0.0, 1.0))


def sample_course(rng: np.random.Generator, dist: CourseDistributionConfig) -> GeneratedCourse:
    """Pick a scenario, generate + validate a course, return a GeneratedCourse.

    On repeated validation failure, falls back to the canonical swift_v4 course
    (always flyable) so callers never receive an invalid course.
    """
    helpers = _collect_helpers()
    bounds = dist.bounds
    scenario = dist.pick_scenario(rng)
    seed = int(rng.integers(0, 2**31 - 1))

    gates = roles = None
    for _ in range(max(1, int(dist.max_resample_tries))):
        cand_gates, cand_roles = generate_scenario(rng, scenario, bounds)
        waypoints = helpers.waypoints_from_gates(
            cand_gates, bounds.z_min, d_approach=float(scenario.gate_approach_m)
        )
        ok, _reasons = validate_course(cand_gates, bounds, waypoints=waypoints)
        if ok:
            gates, roles = cand_gates, cand_roles
            break
    if gates is None:
        # Fallback: canonical swift_v4 (deterministic, always valid).
        gates, roles = _swift_v4(rng, scenario, bounds, noisy=False)
        scenario_name = f"{scenario.name}__fallback_swift_v4"
        waypoints = helpers.waypoints_from_gates(gates, bounds.z_min, d_approach=float(scenario.gate_approach_m))
    else:
        scenario_name = scenario.name

    num_inverted = sum(1 for r in roles if r in (GateRole.INVERTED.value, GateRole.SPLIT_S_TOP.value))
    return GeneratedCourse(
        gates=gates,
        waypoints=np.asarray(waypoints, dtype=np.float64),
        scenario_name=scenario_name,
        scenario_family=scenario.family,
        scenario_seed=seed,
        difficulty=_difficulty(gates, roles, bounds),
        path_length_m=path_length(gates),
        num_inverted_gates=num_inverted,
        gate_roles=list(roles),
        noise_params={
            "pos_noise_xyz": [float(x) for x in scenario.pos_noise_xyz],
            "yaw_noise_rad": float(scenario.yaw_noise_rad),
            "inverted_roll_jitter_rad": float(scenario.inverted_roll_jitter_rad),
        },
        course_bounds=bounds.as_dict(),
    )

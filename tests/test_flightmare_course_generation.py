"""Tests for the shared procedural course module + MPPI expert wiring."""
from collections import Counter

import numpy as np
import pytest

from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.expert_env import GateSpec
from src.robotics.flightmare_autonomy_fsw.gates import gate_frame, signed_gate_distance
from src.robotics.flightmare_courses import (
    CourseBounds,
    CourseDistributionConfig,
    ScenarioConfig,
    course_distribution_from_dict,
    default_course_distribution,
    sample_course,
    validate_course,
)


def test_default_distribution_samples_are_all_valid():
    rng = np.random.default_rng(0)
    dist = default_course_distribution()
    for _ in range(200):
        gc = sample_course(rng, dist)
        ok, reasons = validate_course(gc.gates, dist.bounds, waypoints=gc.waypoints)
        assert ok, reasons
        assert gc.metadata()["generator_version"]
        assert len(gc.gate_roles) == len(gc.gates)


def test_deterministic_for_fixed_seed():
    dist = default_course_distribution()
    a = sample_course(np.random.default_rng(7), dist)
    b = sample_course(np.random.default_rng(7), dist)
    assert a.scenario_family == b.scenario_family
    np.testing.assert_allclose(
        np.stack([g.pos for g in a.gates]), np.stack([g.pos for g in b.gates])
    )


def test_scenario_weights_drive_family_mix():
    dist = CourseDistributionConfig(
        scenarios=[
            ScenarioConfig(name="flow", family="procedural_flow", weight=0.9),
            ScenarioConfig(name="chic", family="procedural_chicane", weight=0.1),
        ]
    )
    rng = np.random.default_rng(0)
    fams = Counter(sample_course(rng, dist).scenario_family for _ in range(400))
    assert fams["procedural_flow"] > fams["procedural_chicane"]
    assert fams["procedural_chicane"] > 0


def test_split_s_has_inverted_gate_with_down_up_axis():
    dist = CourseDistributionConfig(
        scenarios=[ScenarioConfig(name="ss", family="procedural_split_s", weight=1.0, num_inverted_range=(1, 1))]
    )
    gc = sample_course(np.random.default_rng(1), dist)
    assert gc.num_inverted_gates >= 1
    inverted = [g for g, r in zip(gc.gates, gc.gate_roles) if r in ("inverted", "split_s_top")]
    assert inverted
    for g in inverted:
        _, _, _, up = gate_frame(g)
        assert up[2] < 0.0  # inverted: gate "up" points world-down


def test_waypoints_cross_correct_gate_plane():
    dist = default_course_distribution()
    gc = sample_course(np.random.default_rng(3), dist)
    for g in gc.gates:
        center, forward, _, _ = gate_frame(g)
        pre = center - 0.5 * forward
        post = center + 0.5 * forward
        assert signed_gate_distance(pre, g) < 0.0
        assert signed_gate_distance(post, g) > 0.0


def _bad_bounds():
    return CourseBounds(arena_radius=60.0, z_min=1.0, z_max=6.0)


def test_validator_rejects_underground_gate():
    bounds = _bad_bounds()
    gates = [
        GateSpec("g0", np.array([0.0, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
        GateSpec("g1", np.array([6.0, 0.0, -1.0]), 0.0, np.array([1.6, 1.6, 1.6])),
        GateSpec("g2", np.array([12.0, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
    ]
    ok, reasons = validate_course(gates, bounds)
    assert not ok
    assert any("z_min" in r for r in reasons)


def test_validator_rejects_overlapping_gates():
    bounds = _bad_bounds()
    gates = [
        GateSpec("g0", np.array([0.0, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
        GateSpec("g1", np.array([0.2, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
    ]
    ok, reasons = validate_course(gates, bounds)
    assert not ok
    assert any("too close" in r for r in reasons)


def test_validator_rejects_out_of_arena():
    bounds = _bad_bounds()
    gates = [
        GateSpec("g0", np.array([0.0, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
        GateSpec("g1", np.array([500.0, 0.0, 2.0]), 0.0, np.array([1.6, 1.6, 1.6])),
    ]
    ok, reasons = validate_course(gates, bounds)
    assert not ok
    assert any("arena_radius" in r for r in reasons)


def test_course_distribution_from_dict_roundtrip():
    d = {
        "max_resample_tries": 10,
        "bounds": {"arena_radius": 40.0, "z_min": 1.0, "z_max": 5.0},
        "scenarios": [{"name": "f", "family": "procedural_flow", "weight": 1.0}],
    }
    dist = course_distribution_from_dict(d)
    assert dist.bounds.arena_radius == 40.0
    assert dist.max_resample_tries == 10
    gc = sample_course(np.random.default_rng(0), dist)
    assert gc.scenario_family == "procedural_flow"


def test_bad_scenario_family_raises():
    with pytest.raises(ValueError):
        ScenarioConfig(name="x", family="not_a_family", weight=1.0)


# --- MPPI expert ---------------------------------------------------------


def test_mppi_produces_finite_safe_ctbr():
    from src.robotics.flightmare_experts import MPPIConfig, MPPIController, SafetySupervisor

    params = QuadParams()
    params.max_collective_thrust = 5.5 * params.mass * params.g
    mppi = MPPIController(params=params, config=MPPIConfig(num_samples=128, horizon=15), seed=0)
    sup = SafetySupervisor(max_body_rate=18.0)
    cmd = mppi.compute(
        pos=np.array([0.0, 0.0, 2.0]),
        vel=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        pos_des=np.array([2.0, 0.0, 2.0]),
        vel_des=np.array([4.0, 0.0, 0.0]),
        acc_des=np.zeros(3),
        yaw_des=0.0,
        omega=np.zeros(3),
    )
    assert cmd["source"] == "mppi"
    assert np.isfinite(cmd["thrust_normalized"])
    assert cmd["body_rates"].shape == (4,) or cmd["body_rates"].shape == (3,)
    assert sup.check(cmd).ok


def test_mppi_deterministic_for_fixed_seed():
    from src.robotics.flightmare_experts import MPPIConfig, MPPIController

    def first_action():
        m = MPPIController(config=MPPIConfig(num_samples=64, horizon=10), seed=42)
        return m.compute(
            np.array([0.0, 0.0, 2.0]), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 2.0]), np.zeros(3), np.zeros(3), 0.0, omega=np.zeros(3),
        )["thrust_normalized"]

    assert first_action() == first_action()


def test_collect_dataset_gate_frame_is_quaternion_aware():
    """Recovery sampling must use the real gate up-axis for inverted gates."""
    from scripts.flightmare_bc.collect_dataset import _gate_frame

    dist = CourseDistributionConfig(
        scenarios=[ScenarioConfig(name="ss", family="procedural_split_s", weight=1.0, num_inverted_range=(1, 1))]
    )
    gc = sample_course(np.random.default_rng(2), dist)
    inverted = [g for g, r in zip(gc.gates, gc.gate_roles) if r in ("inverted", "split_s_top")][0]
    fwd, lat, up = _gate_frame(inverted)
    _, gf_fwd, gf_lat, gf_up = gate_frame(inverted)
    np.testing.assert_allclose(up, gf_up, atol=1e-9)
    assert up[2] < 0.0  # world-up reconstruction would have given +z here


def test_build_env_config_carries_course_distribution():
    from src.robotics.flightmare_envs import build_flightmare_env_config

    dist = default_course_distribution()
    cfg = build_flightmare_env_config(course_distribution=dist, obs_schema="v3")
    assert cfg.course.course_distribution is dist


def test_mppi_tracks_moving_reference_closed_loop():
    from src.robotics.flightmare_experts import MPPIConfig, MPPIController
    from src.robotics.flightmare_experts.mppi import _quat_integrate_batch, _quat_to_R_batch

    params = QuadParams()
    params.max_collective_thrust = 5.5 * params.mass * params.g
    mppi = MPPIController(params=params, config=MPPIConfig(), seed=0)
    pos = np.array([0.0, 0.0, 2.0])
    vel = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    m, g = params.mass, params.g
    grav = np.array([0.0, 0.0, g])
    errs = []
    for k in range(300):
        t = k * 0.01
        target = np.array([4.0 * t, 0.0, 2.0])
        cmd = mppi.compute(pos, vel, quat, target, np.array([4.0, 0.0, 0.0]), np.zeros(3), 0.0, omega=omega)
        om = np.clip(cmd["body_rates"].astype(float), -18, 18)
        quat = _quat_integrate_batch(quat[None, :], om[None, :], 0.01)[0]
        R = _quat_to_R_batch(quat[None, :])[0]
        acc = (cmd["thrust_normalized"] * params.max_collective_thrust / m) * R[:, 2] - grav
        vel = vel + acc * 0.01
        pos = pos + vel * 0.01
        omega = om
        errs.append(float(np.linalg.norm(pos - target)))
    assert float(np.mean(errs[-150:])) < 0.5

import numpy as np

from src.robotics.rewards import flightmare_racing_v1, flightmare_racing_v2, flightmare_racing_v3


def _base_motor_info() -> dict:
    return {
        "distance_progress_m": 0.1,
        "segment_progress_m": 0.1,
        "gate_alignment": 1.0,
        "source_action_type": "motor",
        "motor_command": np.full(4, 0.2, dtype=np.float32),
        "omega": np.zeros(3, dtype=np.float32),
        "body_z_world_z": 1.0,
        "vertical_speed_mps": 0.0,
    }


def test_flightmare_motor_stability_terms_are_opt_in():
    info = _base_motor_info()
    unstable = dict(info)
    unstable.update(
        {
            "motor_command": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "omega": np.array([8.0, -6.0, 3.0], dtype=np.float32),
            "body_z_world_z": 0.2,
            "vertical_speed_mps": -8.0,
        }
    )

    stable_default = flightmare_racing_v1(
        info=info,
        action=np.full(4, 0.2, dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
        body_rate_penalty=0.0,
        action_smoothness_penalty=0.0,
    )
    unstable_default = flightmare_racing_v1(
        info=unstable,
        action=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
        body_rate_penalty=0.0,
        action_smoothness_penalty=0.0,
    )

    assert unstable_default == stable_default


def test_flightmare_motor_stability_terms_penalize_unstable_motor_states():
    stable = flightmare_racing_v1(
        info=_base_motor_info(),
        action=np.full(4, 0.2, dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
        angular_rate_penalty=0.001,
        tilt_penalty=0.15,
        vertical_speed_penalty=0.01,
        motor_spread_penalty=0.08,
        motor_saturation_penalty=0.02,
        collective_hover_penalty=0.02,
        hover_motor_value=0.1818,
    )

    unstable_info = _base_motor_info()
    unstable_info.update(
        {
            "motor_command": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "omega": np.array([8.0, -6.0, 3.0], dtype=np.float32),
            "body_z_world_z": 0.2,
            "vertical_speed_mps": -8.0,
        }
    )
    unstable = flightmare_racing_v1(
        info=unstable_info,
        action=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
        angular_rate_penalty=0.001,
        tilt_penalty=0.15,
        vertical_speed_penalty=0.01,
        motor_spread_penalty=0.08,
        motor_saturation_penalty=0.02,
        collective_hover_penalty=0.02,
        hover_motor_value=0.1818,
    )

    assert unstable < stable


def test_flightmare_racing_v2_logs_reward_terms():
    info = _base_motor_info()
    info.update(
        {
            "gate_normal_progress_m": 0.2,
            "gate_normal_velocity_mps": 3.0,
            "gate_lateral_norm": 0.1,
            "gate_vertical_norm": -0.2,
            "gate_signed_distance_m": -1.0,
            "gate_passed": True,
            "gate_margin_m": 0.3,
        }
    )
    reward = flightmare_racing_v2(
        info=info,
        action=np.full(4, 0.2, dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
    )

    assert reward == info["reward_terms"]["total"]
    assert info["reward_terms"]["progress_gate_normal"] > 0.0
    assert info["reward_terms"]["gate_pass"] > 0.0
    assert "reward/progress_gate_normal" in info


def test_flightmare_racing_v3_logs_transition_terms():
    info = _base_motor_info()
    info.update(
        {
            "step": 40,
            "gate_normal_progress_m": 0.05,
            "gate_normal_velocity_mps": 0.25,
            "gate_center_error_norm": 1.5,
            "gate_center_error_norm_next": 1.0,
            "gate_center_error_progress": 0.5,
            "gate_lateral_norm_next": 1.3,
            "gate_vertical_norm_next": 0.2,
            "gate_signed_distance_next_m": -1.0,
        }
    )
    reward = flightmare_racing_v3(
        info=info,
        action=np.full(4, 0.2, dtype=np.float32),
        prev_action=np.full(4, 0.2, dtype=np.float32),
    )

    assert reward == info["reward_terms"]["total"]
    assert info["reward_terms"]["centerline_progress"] > 0.0
    assert info["reward_terms"]["centerline_error"] < 0.0
    assert info["reward_terms"]["aperture_violation_near"] < 0.0
    assert info["reward_terms"]["no_progress"] < 0.0
    assert "reward/centerline_progress" in info

import numpy as np

from src.robotics.rewards import flightmare_racing_v1


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

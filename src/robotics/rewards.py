"""Reward functions for robotics RL fine-tuning.

Mirrors the registry-style lookup used in ``src/rlvr/math_verifier.py``:

    >>> from src.robotics.rewards import get_reward_function
    >>> fn = get_reward_function("dense_multitask_v1")
    >>> r = fn(task_name="lift", obs=obs_dict, base_reward=0.4,
    ...       terminated=False, truncated=False, info={"success": False})

The selected reward name is passed through from the PPO config to
``RobosuiteGymEnv``, which wraps each environment step and returns the
schema's scalar in place of robosuite's raw reward.

Function signature
------------------
All reward functions share this signature::

    def reward_fn(
        *,
        task_name: str,            # one of: lift | can | square | tool_hang | ...
        obs: dict,                 # raw robosuite observation dict (pre-normalization)
        base_reward: float,        # robosuite's native reward for this step
        terminated: bool,          # success-triggered termination
        truncated: bool,           # horizon-triggered termination
        info: dict,                # step info, including "success" (bool)
        step_in_episode: int,      # 0-indexed step count in the current episode
    ) -> float

Keyword-only by design so reward implementations can pick off just what they
need and ignore the rest.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Per-task normalization — rough upper bounds of robosuite's dense shaped
# reward at success. Used to bring all four tasks onto a comparable scale so
# PPO advantage magnitudes aren't dominated by whichever task naturally gives
# bigger numbers.  Measured from robomimic expert rollouts.
# ---------------------------------------------------------------------------
_TASK_REWARD_SCALE = {
    "lift": 2.25,
    "can": 2.25,
    "square": 5.00,
    "tool_hang": 1.00,  # ToolHang native reward caps ~1.0; don't shrink further
}
# Terminal success bonus — must dominate per-episode shaping variance, not
# mean. With shaping capped ~0.15/step and horizons up to 700, cumulative
# shaping is ~100; raw variance across failing trajectories is much smaller,
# so 50 cleanly separates successes.
_SUCCESS_BONUS = 50.0
# Per-step time penalty. Disabled for now: at 0.005 × 700 steps = 3.5 floor,
# and with 0% success the gradient is dominated by "fail faster" rather than
# "succeed." Re-enable (and shrink) once any task reaches >10% success.
_TIME_PENALTY = 0.0


# ===========================================================================
# Baseline reward functions
# ===========================================================================

def native_shaped(*, base_reward: float, **_) -> float:
    """Use robosuite's built-in shaped reward as-is (baseline, no transform)."""
    return float(base_reward)


def sparse_success(*, info: dict, **_) -> float:
    """Pure 0/1 on success — hard exploration baseline."""
    return 1.0 if info.get("success", False) else 0.0


# ===========================================================================
# Task-dispatched dense reward — per-task shaping for the 4 robomimic tasks
# ===========================================================================

def _eef_to_point_dist(obs: dict, point: np.ndarray) -> float:
    eef = obs.get("robot0_eef_pos")
    if eef is None or point is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(eef) - np.asarray(point)))


def _reward_lift(obs: dict, base_reward: float) -> float:
    """Lift: robosuite's shaped reward is already well-tuned (reach/grasp/lift).
    Add a small extra bonus proportional to cube height above the table."""
    shaped = base_reward / _TASK_REWARD_SCALE["lift"]
    cube_pos = obs.get("cube_pos")
    if cube_pos is not None:
        # Reward cube being lifted off the table (rough table height ~0.82m).
        height_bonus = max(0.0, float(cube_pos[2]) - 0.82) * 2.0
        shaped += height_bonus
    return shaped


def _reward_can(obs: dict, base_reward: float) -> float:
    """PickPlaceCan: reach can → grasp → transport → drop in bin."""
    return base_reward / _TASK_REWARD_SCALE["can"]


def _reward_square(obs: dict, base_reward: float) -> float:
    """NutAssemblySquare: reach nut → grasp → align with peg → lower onto peg.
    Robosuite doesn't expose `peg_pos`; the previous implementation silently
    fell through to base_reward. Use the nut-to-EEF vector robosuite *does*
    expose to reward pre-grasp approach — the signal most missing from the
    native shaped reward at the start of an episode.
    """
    shaped = base_reward / _TASK_REWARD_SCALE["square"]
    nut_to_eef = obs.get("SquareNut_to_robot0_eef_pos")
    if nut_to_eef is not None:
        d = float(np.linalg.norm(np.asarray(nut_to_eef)))
        shaped += 0.05 * (1.0 - np.tanh(3.0 * d))
    return shaped


def _reward_tool_hang(obs: dict, base_reward: float) -> float:
    """ToolHang: the hardest task. Layer extra shaping on the geometry
    robosuite actually exposes (tool_to_robot0_eef_pos, tool_pos, frame_pos)
    plus the two milestone booleans (tool_on_frame, frame_is_assembled).
    """
    shaped = base_reward / _TASK_REWARD_SCALE["tool_hang"]
    # Pre-grasp: EEF → tool approach.
    tool_to_eef = obs.get("tool_to_robot0_eef_pos")
    if tool_to_eef is not None:
        d = float(np.linalg.norm(np.asarray(tool_to_eef)))
        shaped += 0.03 * (1.0 - np.tanh(3.0 * d))
    # Transport: tool → frame proximity.
    tool_pos = obs.get("tool_pos")
    frame_pos = obs.get("frame_pos")
    if tool_pos is not None and frame_pos is not None:
        d = float(np.linalg.norm(np.asarray(tool_pos) - np.asarray(frame_pos)))
        shaped += 0.05 * (1.0 - np.tanh(2.0 * d))
    # Milestone bonuses — sparse, informative, unambiguous sub-goals.
    if bool(obs.get("tool_on_frame", False)):
        shaped += 0.2
    if bool(obs.get("frame_is_assembled", False)):
        shaped += 0.5
    return shaped


# Per-task shaping dispatch table. Easy to extend: add a new task by
# registering a new function here.
_DENSE_TASK_HANDLERS: Dict[str, Callable] = {
    "lift": _reward_lift,
    "can": _reward_can,
    "square": _reward_square,
    "tool_hang": _reward_tool_hang,
}


def dense_multitask_v1(
    *,
    task_name: str,
    obs: dict,
    base_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    step_in_episode: int,
    **_,
) -> float:
    """Task-dispatched dense reward with a unified success/time schema.

    Structure:
      1. Per-task dense shaping (see ``_DENSE_TASK_HANDLERS``) — normalized
         to a comparable scale across tasks.
      2. Flat ``+_SUCCESS_BONUS`` on success (terminal).
      3. Flat ``-_TIME_PENALTY`` per step to prefer shorter episodes.

    Unknown tasks fall back to the normalized robosuite shaped reward.
    """
    handler = _DENSE_TASK_HANDLERS.get(task_name)
    if handler is not None:
        r = handler(obs, float(base_reward))
    else:
        scale = _TASK_REWARD_SCALE.get(task_name, 1.0)
        r = float(base_reward) / scale

    r -= _TIME_PENALTY
    if info.get("success", False):
        r += _SUCCESS_BONUS
    return float(r)


# ===========================================================================
# Flightmare drone racing reward
# ===========================================================================

def flightmare_racing_v1(
    *,
    info: dict,
    action: np.ndarray | None = None,
    prev_action: np.ndarray | None = None,
    dt: float = 0.01,
    progress_scale: float = 8.0,
    segment_progress_scale: float = 1.0,
    gate_pass_bonus: float = 12.0,
    completion_bonus: float = 75.0,
    gate_miss_penalty: float = 25.0,
    crash_penalty: float = 25.0,
    time_penalty: float = 0.01,
    body_rate_penalty: float = 0.002,
    action_smoothness_penalty: float = 0.01,
    alignment_scale: float = 0.05,
    gate_centering_penalty: float = 0.0,
    gate_violation_penalty: float = 0.0,
    gate_centering_near_m: float = 4.0,
    max_progress_reward: float = 2.0,
    angular_rate_penalty: float = 0.0,
    tilt_penalty: float = 0.0,
    vertical_speed_penalty: float = 0.0,
    motor_spread_penalty: float = 0.0,
    motor_saturation_penalty: float = 0.0,
    collective_hover_penalty: float = 0.0,
    hover_motor_value: float | None = None,
    **_,
) -> float:
    """Dense gate-progress reward for state-only autonomous drone racing.

    The structure mirrors modern drone-racing RL rewards:
      progress toward the next gate + line-segment progress + action/body-rate
      smoothness + terminal miss/crash penalties. The visual perception term
      used by Swift-style policies is represented here as a lightweight
      body-forward alignment bonus because this stack is privileged-state only.
    """
    dist_progress = float(info.get("distance_progress_m", 0.0))
    segment_progress = float(info.get("segment_progress_m", 0.0))
    dist_progress = float(np.clip(dist_progress, -max_progress_reward, max_progress_reward))
    segment_progress = float(np.clip(segment_progress, -max_progress_reward, max_progress_reward))

    r = progress_scale * dist_progress
    r += segment_progress_scale * segment_progress
    r -= float(time_penalty)

    if action is not None:
        a = np.asarray(action, dtype=np.float32)
        if a.size >= 4:
            r -= float(body_rate_penalty) * float(np.sum(a[1:4] ** 2))
        if prev_action is not None:
            pa = np.asarray(prev_action, dtype=np.float32)
            r -= float(action_smoothness_penalty) * float(np.sum((a - pa) ** 2))

    if angular_rate_penalty > 0.0:
        omega = info.get("omega")
        if omega is not None:
            omega_sq = float(np.sum(np.asarray(omega, dtype=np.float32) ** 2))
        else:
            omega_sq = float(info.get("angular_rate_norm", 0.0)) ** 2
        r -= float(angular_rate_penalty) * omega_sq

    if tilt_penalty > 0.0:
        upright = float(np.clip(info.get("body_z_world_z", 1.0), -1.0, 1.0))
        r -= float(tilt_penalty) * max(0.0, 1.0 - upright)

    if vertical_speed_penalty > 0.0:
        vz = float(info.get("vertical_speed_mps", 0.0))
        r -= float(vertical_speed_penalty) * vz * vz

    if (
        info.get("source_action_type") == "motor"
        and (
            motor_spread_penalty > 0.0
            or motor_saturation_penalty > 0.0
            or collective_hover_penalty > 0.0
        )
    ):
        motors = np.asarray(info.get("motor_command", np.zeros(4, dtype=np.float32)), dtype=np.float32)
        if motors.size:
            mean_motor = float(np.mean(motors))
            if motor_spread_penalty > 0.0:
                r -= float(motor_spread_penalty) * float(np.sum((motors - mean_motor) ** 2))
            if motor_saturation_penalty > 0.0:
                low_sat = np.maximum(0.05 - motors, 0.0)
                high_sat = np.maximum(motors - 0.95, 0.0)
                r -= float(motor_saturation_penalty) * float(np.sum(low_sat ** 2 + high_sat ** 2))
            if collective_hover_penalty > 0.0 and hover_motor_value is not None:
                err = mean_motor - float(hover_motor_value)
                r -= float(collective_hover_penalty) * err * err

    # Alignment is in [0, 1] where 1 means body/camera forward points at next gate.
    r += float(alignment_scale) * float(info.get("gate_alignment", 0.0))

    if gate_centering_penalty > 0.0 or gate_violation_penalty > 0.0:
        lat_norm = float(info.get("gate_lateral_norm", 0.0))
        vert_norm = float(info.get("gate_vertical_norm", 0.0))
        signed_dist = abs(float(info.get("gate_signed_distance_m", 0.0)))
        near = max(0.0, 1.0 - signed_dist / max(float(gate_centering_near_m), 1e-6))
        center_err = min(lat_norm ** 2 + vert_norm ** 2, 4.0)
        aperture_violation = min(max(abs(lat_norm) - 1.0, 0.0) + max(abs(vert_norm) - 1.0, 0.0), 4.0)
        r -= float(gate_centering_penalty) * near * center_err
        r -= float(gate_violation_penalty) * near * aperture_violation

    if info.get("gate_passed", False):
        margin = max(0.0, float(info.get("gate_margin_m", 0.0)))
        r += float(gate_pass_bonus) + 2.0 * margin
    if info.get("success", False):
        r += float(completion_bonus)
    if info.get("gate_missed", False):
        r -= float(gate_miss_penalty)
    if info.get("crash", False):
        r -= float(crash_penalty)
    return float(r)


# ===========================================================================
# Registry + lookup
# ===========================================================================

REWARD_FUNCTIONS: Dict[str, Callable] = {
    "native_shaped": native_shaped,
    "sparse_success": sparse_success,
    "dense_multitask_v1": dense_multitask_v1,
    "flightmare_racing_v1": flightmare_racing_v1,
}


def get_reward_function(name: str) -> Callable:
    """Look up a reward function by name, as in ``math_verifier.get_reward_function``."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name!r}. "
            f"Available: {sorted(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]

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
    "tool_hang": 5.00,
}
# Terminal success bonus (common across tasks) — large enough to dominate
# shaping noise, small enough not to create runaway advantage spikes.
_SUCCESS_BONUS = 10.0
# Small time penalty to encourage fast completion without overwhelming the
# shaping signal.
_TIME_PENALTY = 0.005


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
    Add a small extra shaping for EEF proximity to the peg after grasping."""
    shaped = base_reward / _TASK_REWARD_SCALE["square"]
    # If robosuite exposes peg_pos, reward getting the EEF near it.
    peg_pos = obs.get("peg_pos") if "peg_pos" in obs else None
    if peg_pos is not None:
        d = _eef_to_point_dist(obs, np.asarray(peg_pos))
        # Tanh bounded proximity bonus: max 0.1 when EEF touches the peg.
        shaped += 0.1 * (1.0 - np.tanh(3.0 * d))
    return shaped


def _reward_tool_hang(obs: dict, base_reward: float) -> float:
    """ToolHang: the hardest task. Robosuite's shaped reward is relatively
    weak here, so we layer extra shaping based on tool↔frame geometry when
    those keys are present in the obs."""
    shaped = base_reward / _TASK_REWARD_SCALE["tool_hang"]
    # Tool-to-frame proximity bonus (hang target).
    tool_pos = obs.get("tool_pos")
    frame_pos = obs.get("frame_pos")
    if tool_pos is not None and frame_pos is not None:
        d = float(np.linalg.norm(np.asarray(tool_pos) - np.asarray(frame_pos)))
        shaped += 0.2 * (1.0 - np.tanh(2.0 * d))
    # Additional gripper-to-tool bonus before grasping.
    if tool_pos is not None:
        d_eef = _eef_to_point_dist(obs, np.asarray(tool_pos))
        shaped += 0.05 * (1.0 - np.tanh(3.0 * d_eef))
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
# Registry + lookup
# ===========================================================================

REWARD_FUNCTIONS: Dict[str, Callable] = {
    "native_shaped": native_shaped,
    "sparse_success": sparse_success,
    "dense_multitask_v1": dense_multitask_v1,
}


def get_reward_function(name: str) -> Callable:
    """Look up a reward function by name, as in ``math_verifier.get_reward_function``."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name!r}. "
            f"Available: {sorted(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]

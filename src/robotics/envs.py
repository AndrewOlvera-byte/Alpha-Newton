"""Gymnasium-wrapped robosuite environments for RL training.

Wraps robosuite envs as gymnasium.Env instances with observation/action
formats matching the BC pipeline (HistoryBuffer expects raw uint8 images
and float32 state; actions are in normalized space).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from src.robotics.normalization import NormStats
from src.robotics.rewards import get_reward_function


# Robomimic config task names → robosuite environment names
_ROBOSUITE_ENV_MAP = {
    "lift": "Lift",
    "can": "PickPlaceCan",
    "square": "NutAssemblySquare",
    "stack": "Stack",
    "door": "Door",
    "wipe": "Wipe",
    "tool_hang": "ToolHang",
    "transport": "TwoArmTransport",
}


class RobosuiteGymEnv(gymnasium.Env):
    """Gymnasium wrapper around a robosuite environment.

    Observations are returned as a dict with:
      - "state": float32 array of concatenated low-dim obs
      - "images": dict of {cam_key: uint8 HWC array}

    Actions are accepted in **normalized** space (matching BC model output).
    The wrapper denormalizes before stepping robosuite.
    """

    def __init__(
        self,
        task_name: str,
        norm_stats: NormStats,
        camera_names: List[str],
        camera_size: int = 160,
        obs_keys_low_dim: List[str] = None,
        obs_keys_image: List[str] = None,
        horizon: int = 400,
        reward_shaping: bool = True,
        seed: int = 0,
        reward_fn: Optional[str] = None,
    ):
        super().__init__()
        import robosuite as suite

        self.norm_stats = norm_stats
        self.obs_keys_low_dim = obs_keys_low_dim or []
        self.obs_keys_image = obs_keys_image or []
        self.horizon = horizon
        self._step_count = 0
        self._task_name = task_name
        # Look up the reward-shaping callable once — None means "use robosuite's
        # native reward as-is" (backwards compatible with earlier PPO runs).
        self._reward_fn = get_reward_function(reward_fn) if reward_fn else None

        robosuite_name = _ROBOSUITE_ENV_MAP.get(task_name.lower(), task_name.capitalize())
        self._env = suite.make(
            robosuite_name,
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=camera_size,
            camera_widths=camera_size,
            camera_depths=False,
            use_object_obs=True,
            ignore_done=False,
            reward_shaping=reward_shaping,
            control_freq=20,
            horizon=horizon,
        )

        # Determine dims from a probe reset
        probe_obs = self._env.reset()
        state = self._extract_state(probe_obs)
        state_dim = state.shape[0]
        action_dim = len(self.norm_stats.action_mean)

        # Observation space
        img_spaces = {}
        for key in self.obs_keys_image:
            img_spaces[key] = spaces.Box(0, 255, shape=(camera_size, camera_size, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
            "images": spaces.Dict(img_spaces),
        })

        # Action space in normalized space (model outputs clamped to [-5, 5])
        self.action_space = spaces.Box(-5.0, 5.0, shape=(action_dim,), dtype=np.float32)

        # Seed
        self._seed = seed

    def _extract_state(self, obs: dict) -> np.ndarray:
        parts = []
        for key in self.obs_keys_low_dim:
            if key in obs:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    parts.append(val.flatten().astype(np.float32))
                else:
                    parts.append(np.array([val], dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _extract_images(self, obs: dict) -> Dict[str, np.ndarray]:
        images = {}
        for key in self.obs_keys_image:
            if key in obs:
                images[key] = obs[key].astype(np.uint8)
            else:
                cam = key.replace("_image", "")
                img_key = f"{cam}_image"
                if img_key in obs:
                    images[key] = obs[img_key].astype(np.uint8)
        return images

    def _make_obs(self, raw_obs: dict) -> Dict[str, Any]:
        return {
            "state": self._extract_state(raw_obs),
            "images": self._extract_images(raw_obs),
        }

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        if seed is not None:
            self._seed = seed
        raw_obs = self._env.reset()
        self._step_count = 0
        return self._make_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        # Denormalize action from model's normalized space to world space
        raw_action = self.norm_stats.denormalize_action(action.astype(np.float32))
        raw_action = np.clip(raw_action, -1.0, 1.0)

        raw_obs, reward, done, info = self._env.step(raw_action)
        self._step_count += 1

        success = bool(self._env._check_success())
        info["success"] = success
        info["raw_action"] = raw_action

        terminated = success
        truncated = self._step_count >= self.horizon and not success

        # Apply custom reward shaping if one was requested. The function sees
        # the raw robosuite obs (has task-specific keys like cube_pos, tool_pos,
        # frame_pos) plus robosuite's native reward as its base.
        if self._reward_fn is not None:
            info["base_reward"] = float(reward)
            reward = self._reward_fn(
                task_name=self._task_name,
                obs=raw_obs,
                base_reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                step_in_episode=self._step_count - 1,
            )

        return self._make_obs(raw_obs), float(reward), terminated, truncated, info

    def close(self):
        self._env.close()


@dataclass
class EnvSpec:
    """Per-env specification for building a heterogeneous vector env."""
    task_name: str
    task_id: int
    norm_stats: NormStats
    horizon: int = 400
    reward_shaping: bool = True
    reward_fn: Optional[str] = None  # name from REWARD_FUNCTIONS registry


class MultiTaskVecEnv:
    """Vectorized robosuite env supporting heterogeneous tasks.

    Unlike ``gymnasium.vector.SyncVectorEnv``, this wrapper does NOT stack
    per-env observations — different tasks have different state/object
    dimensions, so observations are returned as a per-env list.  Physics
    steps are parallelized via a ``ThreadPoolExecutor`` (robosuite /
    MuJoCo release the GIL during ``sim.step``).

    API (mirrors SyncVectorEnv but list-valued where stacking is impossible):
        reset()  → (List[obs_dict], List[info_dict])
        step(actions: np.ndarray[N, action_dim])
                 → (List[obs_dict], rewards[N], terms[N], truncs[N], List[info_dict])
        close(), n_envs
    """

    def __init__(self, env_fns: List[callable], max_workers: Optional[int] = None):
        self.envs: List[RobosuiteGymEnv] = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        # One worker per env — robosuite steps are CPU-bound but GIL-releasing.
        self._pool = ThreadPoolExecutor(max_workers=max_workers or self.n_envs)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
        def _reset(i_env):
            i, env = i_env
            return env.reset(seed=None if seed is None else seed + i)
        results = list(self._pool.map(_reset, enumerate(self.envs)))
        return [r[0] for r in results], [r[1] for r in results]

    def step(self, actions: np.ndarray):
        def _step(args):
            env, action = args
            return env.step(action)
        results = list(self._pool.map(_step, zip(self.envs, actions)))
        obs_list = [r[0] for r in results]
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        terms = np.asarray([r[2] for r in results], dtype=bool)
        truncs = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs_list, rewards, terms, truncs, infos

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self._pool.shutdown(wait=False)


def allocate_envs_by_weights(n_envs: int, weights: List[float]) -> List[int]:
    """Largest-remainder (Hamilton/Hare) allocation of ``n_envs`` across tasks.

    Every task with a strictly-positive weight is guaranteed at least one
    env so that harder-but-downweighted tasks still get gradient signal.
    The remaining envs are distributed by fractional parts of ``w * n_envs``.

    Example:
        n_envs=20, weights=[0.05, 0.05, 0.10, 0.80] → [1, 1, 2, 16]
    """
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any():
        raise ValueError(f"Negative weights not allowed: {weights}")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must have positive sum")
    w = w / total
    n_tasks = len(w)
    if n_envs < (w > 0).sum():
        raise ValueError(
            f"n_envs={n_envs} is too small — need at least one env per "
            f"positive-weight task ({(w > 0).sum()})"
        )

    raw = w * n_envs
    counts = np.floor(raw).astype(int)
    # Min-1 guarantee for every positive-weight task.
    counts = np.where((w > 0) & (counts == 0), 1, counts)
    deficit = n_envs - counts.sum()

    if deficit > 0:
        # Distribute remaining envs to largest fractional parts.
        frac = raw - np.floor(raw)
        # Tasks already bumped to 1 should not get another bonus from their frac
        # unless they had a natural remainder (w*n_envs was between 0 and 1).
        order = np.argsort(-frac)
        for i in order:
            if deficit == 0:
                break
            counts[i] += 1
            deficit -= 1
    elif deficit < 0:
        # Overshoot from min-1 clamping — strip from most over-allocated tasks
        # relative to their fair share, never going below 1 for positive tasks.
        over = -deficit
        excess = counts - raw
        order = np.argsort(-excess)
        while over > 0:
            progressed = False
            for i in order:
                floor_min = 1 if w[i] > 0 else 0
                if counts[i] > floor_min:
                    counts[i] -= 1
                    over -= 1
                    progressed = True
                    if over == 0:
                        break
            if not progressed:
                raise RuntimeError("Cannot satisfy env allocation constraints")

    return counts.tolist()


def make_robosuite_multitask_vec_env(
    env_specs: List[EnvSpec],
    camera_names: List[str],
    camera_size: int = 160,
    obs_keys_low_dim: List[str] = None,
    obs_keys_image: List[str] = None,
    seed: int = 0,
) -> MultiTaskVecEnv:
    """Build a ``MultiTaskVecEnv`` from a per-env list of ``EnvSpec``s.

    The length of ``env_specs`` determines ``n_envs``.  Each env uses its
    spec's ``task_name``, ``norm_stats``, and ``horizon`` independently.
    """
    def make_env(i: int, spec: EnvSpec):
        def _init():
            return RobosuiteGymEnv(
                task_name=spec.task_name,
                norm_stats=spec.norm_stats,
                camera_names=camera_names,
                camera_size=camera_size,
                obs_keys_low_dim=obs_keys_low_dim,
                obs_keys_image=obs_keys_image,
                horizon=spec.horizon,
                reward_shaping=spec.reward_shaping,
                seed=seed + i,
                reward_fn=spec.reward_fn,
            )
        return _init

    return MultiTaskVecEnv([make_env(i, spec) for i, spec in enumerate(env_specs)])


def make_robosuite_vec_env(
    task_name: str,
    n_envs: int,
    norm_stats: NormStats,
    camera_names: List[str],
    camera_size: int = 160,
    obs_keys_low_dim: List[str] = None,
    obs_keys_image: List[str] = None,
    horizon: int = 400,
    reward_shaping: bool = True,
    seed: int = 0,
) -> MultiTaskVecEnv:
    """Single-task vec env (thin wrapper around ``MultiTaskVecEnv``)."""
    specs = [
        EnvSpec(
            task_name=task_name, task_id=0, norm_stats=norm_stats,
            horizon=horizon, reward_shaping=reward_shaping,
        )
        for _ in range(n_envs)
    ]
    return make_robosuite_multitask_vec_env(
        specs, camera_names=camera_names, camera_size=camera_size,
        obs_keys_low_dim=obs_keys_low_dim, obs_keys_image=obs_keys_image,
        seed=seed,
    )

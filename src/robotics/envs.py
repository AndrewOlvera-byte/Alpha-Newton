"""Gymnasium-wrapped robosuite environments for RL training.

Wraps robosuite envs as gymnasium.Env instances with observation/action
formats matching the BC pipeline (HistoryBuffer expects raw uint8 images
and float32 state; actions are in normalized space).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from src.robotics.normalization import NormStats


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
    ):
        super().__init__()
        import robosuite as suite

        self.norm_stats = norm_stats
        self.obs_keys_low_dim = obs_keys_low_dim or []
        self.obs_keys_image = obs_keys_image or []
        self.horizon = horizon
        self._step_count = 0

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

        return self._make_obs(raw_obs), float(reward), terminated, truncated, info

    def close(self):
        self._env.close()


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
) -> gymnasium.vector.SyncVectorEnv:
    """Create a vectorized robosuite environment."""
    def make_env(env_seed):
        def _init():
            return RobosuiteGymEnv(
                task_name=task_name,
                norm_stats=norm_stats,
                camera_names=camera_names,
                camera_size=camera_size,
                obs_keys_low_dim=obs_keys_low_dim,
                obs_keys_image=obs_keys_image,
                horizon=horizon,
                reward_shaping=reward_shaping,
                seed=env_seed,
            )
        return _init

    return gymnasium.vector.SyncVectorEnv(
        [make_env(seed + i) for i in range(n_envs)]
    )

"""Robosuite environment backend for robotics reinforcement learning."""
from __future__ import annotations

import os

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from src.robotics.normalization import NormStats
from src.robotics.rewards import get_reward_function

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
    """Gymnasium wrapper with BC-compatible observations and actions."""

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

        probe_obs = self._env.reset()
        state = self._extract_state(probe_obs)
        state_dim = state.shape[0]
        action_dim = len(self.norm_stats.action_mean)

        img_spaces = {}
        for key in self.obs_keys_image:
            img_spaces[key] = spaces.Box(0, 255, shape=(camera_size, camera_size, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
            "images": spaces.Dict(img_spaces),
        })

        self.action_space = spaces.Box(-5.0, 5.0, shape=(action_dim,), dtype=np.float32)
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
        raw_action = self.norm_stats.denormalize_action(action.astype(np.float32))
        raw_action = np.clip(raw_action, -1.0, 1.0)

        raw_obs, reward, done, info = self._env.step(raw_action)
        self._step_count += 1

        success = bool(self._env._check_success())
        info["success"] = success
        info["raw_action"] = raw_action

        terminated = success
        truncated = self._step_count >= self.horizon and not success

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
    reward_fn: Optional[str] = None


class MultiTaskVecEnv:
    """Threaded vector environment for heterogeneous robosuite tasks."""

    def __init__(self, env_fns: List[callable], max_workers: Optional[int] = None):
        self.envs: List[RobosuiteGymEnv] = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self._pool = ThreadPoolExecutor(max_workers=max_workers or self.n_envs)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
        def _reset(i_env):
            i, env = i_env
            return env.reset(seed=None if seed is None else seed + i)
        results = list(self._pool.map(_reset, enumerate(self.envs)))
        return [r[0] for r in results], [r[1] for r in results]

    def reset_at(self, indices: List[int], *, seed: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
        def _reset(i):
            return self.envs[i].reset(seed=None if seed is None else seed + i)
        results = list(self._pool.map(_reset, indices))
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


def _subproc_worker(conn, env_fn: Callable):
    """Build one environment and serve reset/step commands over a pipe."""
    try:
        env = env_fn()
        conn.send(("ready", None))
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                conn.send((obs, info))
            elif cmd == "step":
                conn.send(env.step(data))
            elif cmd == "close":
                env.close()
                conn.send(("closed", None))
                break
            else:
                conn.send(("error", f"unknown cmd {cmd}"))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        conn.send(("error", f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


class SubprocMultiTaskVecEnv:
    """Process-per-env vector environment for heterogeneous robosuite tasks."""

    def __init__(self, env_fns: List[Callable], start_method: Optional[str] = None,
                 startup_parallelism: Optional[int] = None):
        """Create subprocess workers with bounded startup concurrency."""
        if start_method is None:
            start_method = os.environ.get("PPO_START_METHOD")
        if start_method is None:
            start_method = "fork" if os.name == "posix" else "spawn"
        if startup_parallelism is None:
            startup_parallelism = int(os.environ.get("PPO_STARTUP_PARALLELISM", "2"))
        self._timeout_s = float(os.environ.get("PPO_ENV_TIMEOUT_S", "120"))
        self.start_method = start_method
        self.n_envs = len(env_fns)
        ctx = mp.get_context(start_method)
        self.conns: List = []
        self.procs: List = []

        pending: List = []
        for i, fn in enumerate(env_fns):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_subproc_worker, args=(child, fn), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
            pending.append((i, parent, p))

            if len(pending) >= startup_parallelism:
                self._drain_ready(pending.pop(0), self._timeout_s)

        for item in pending:
            self._drain_ready(item, self._timeout_s)

    @staticmethod
    def _drain_ready(item, timeout_s: float):
        i, conn, proc = item
        if not conn.poll(timeout_s):
            alive = proc.is_alive()
            raise TimeoutError(
                f"Env {i} did not signal ready within {timeout_s:.0f}s "
                f"(worker alive={alive})"
            )
        msg = conn.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"Env {i} failed to start: {msg}")

    def _recv_one(self, i: int):
        conn = self.conns[i]
        if not conn.poll(self._timeout_s):
            alive = self.procs[i].is_alive()
            raise TimeoutError(
                f"Env {i} did not respond within {self._timeout_s:.0f}s "
                f"(worker alive={alive})"
            )
        result = conn.recv()
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            raise RuntimeError(f"Env {i} worker error:\n{result[1]}")
        return result

    def reset(self, *, seed: Optional[int] = None):
        for i, c in enumerate(self.conns):
            c.send(("reset", None if seed is None else seed + i))
        results = [self._recv_one(i) for i in range(self.n_envs)]
        return [r[0] for r in results], [r[1] for r in results]

    def reset_at(self, indices: List[int], *, seed: Optional[int] = None):
        indices = list(indices)
        for i in indices:
            self.conns[i].send(("reset", None if seed is None else seed + i))
        results = [self._recv_one(i) for i in indices]
        return [r[0] for r in results], [r[1] for r in results]

    def step(self, actions: np.ndarray):
        import time as _t
        _s0 = _t.time()
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        self._last_send_s = _t.time() - _s0
        _s0 = _t.time()
        results = [self._recv_one(i) for i in range(self.n_envs)]
        self._last_recv_s = _t.time() - _s0
        obs_list = [r[0] for r in results]
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        terms = np.asarray([r[2] for r in results], dtype=bool)
        truncs = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs_list, rewards, terms, truncs, infos

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for c in self.conns:
            try:
                if c.poll(5):
                    c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def allocate_envs_by_weights(n_envs: int, weights: List[float]) -> List[int]:
    """Allocate vector env slots by normalized task weights."""
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
            f"n_envs={n_envs} is too small - need at least one env per "
            f"positive-weight task ({(w > 0).sum()})"
        )

    raw = w * n_envs
    counts = np.floor(raw).astype(int)
    counts = np.where((w > 0) & (counts == 0), 1, counts)
    deficit = n_envs - counts.sum()

    if deficit > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in order:
            if deficit == 0:
                break
            counts[i] += 1
            deficit -= 1
    elif deficit < 0:
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


@dataclass
class _EnvBuilder:
    """Picklable robosuite environment factory."""
    spec: EnvSpec
    camera_names: List[str]
    camera_size: int
    obs_keys_low_dim: List[str]
    obs_keys_image: List[str]
    seed: int

    def __call__(self) -> RobosuiteGymEnv:
        return RobosuiteGymEnv(
            task_name=self.spec.task_name,
            norm_stats=self.spec.norm_stats,
            camera_names=self.camera_names,
            camera_size=self.camera_size,
            obs_keys_low_dim=self.obs_keys_low_dim,
            obs_keys_image=self.obs_keys_image,
            horizon=self.spec.horizon,
            reward_shaping=self.spec.reward_shaping,
            seed=self.seed,
            reward_fn=self.spec.reward_fn,
        )


def make_robosuite_multitask_vec_env(
    env_specs: List[EnvSpec],
    camera_names: List[str],
    camera_size: int = 160,
    obs_keys_low_dim: List[str] = None,
    obs_keys_image: List[str] = None,
    seed: int = 0,
) -> "SubprocMultiTaskVecEnv":
    """Build a subprocess-parallel vec env from a per-env list of ``EnvSpec``s."""
    fns = [
        _EnvBuilder(
            spec=spec,
            camera_names=camera_names or [],
            camera_size=camera_size,
            obs_keys_low_dim=obs_keys_low_dim or [],
            obs_keys_image=obs_keys_image or [],
            seed=seed + i,
        )
        for i, spec in enumerate(env_specs)
    ]
    return SubprocMultiTaskVecEnv(fns)


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

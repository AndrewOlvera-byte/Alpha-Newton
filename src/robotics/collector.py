"""Rollout buffer and collection for on-policy PPO training.

Stores transitions from parallel environments and provides minibatch
iteration for PPO updates. Images are kept as uint8 on CPU to save
GPU memory; conversion to float32 happens lazily per minibatch.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from src.robotics.models.HistoryBuffer import HistoryBuffer
from src.robotics.loss import compute_gae


class RolloutBuffer:
    """Fixed-size buffer for N steps across K parallel environments."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        action_dim: int,
        state_dim: int,
        image_specs: Dict[str, tuple],  # {cam_key: (H, W, C)}
        history_length: int,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.action_dim = action_dim
        self.history_length = history_length
        self.device = device

        # Per-step storage: images on CPU as uint8, everything else on device
        self.images: Dict[str, np.ndarray] = {}
        for cam, (h, w, c) in image_specs.items():
            # [T, N, H, C, h, w] -- stored per history frame
            self.images[cam] = np.zeros(
                (n_steps, n_envs, history_length, c, h, w), dtype=np.uint8
            )

        self.states = torch.zeros(n_steps, n_envs, history_length, state_dim, device=device)
        self.prev_actions = torch.zeros(n_steps, n_envs, history_length, action_dim, device=device)
        self.task_ids = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.actions = torch.zeros(n_steps, n_envs, action_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)

        self._ptr = 0

    def insert(
        self,
        step: int,
        batch: dict,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """Store one transition for all envs."""
        # batch comes from HistoryBuffer.get_batch() calls stacked across envs
        # images: {cam: [N, H, C, h, w]} as float tensors -- store as uint8
        for cam in self.images:
            imgs = batch["images"][cam]  # [N, H, C, h, w] float [0,1]
            self.images[cam][step] = (imgs * 255).byte().cpu().numpy()

        self.states[step] = batch["state"]  # [N, H, state_dim]
        self.prev_actions[step] = batch["prev_actions"]  # [N, H, action_dim]
        self.task_ids[step] = batch["task_id"]  # [N]
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.rewards[step] = torch.from_numpy(rewards).to(self.device)
        self.dones[step] = torch.from_numpy(dones.astype(np.float32)).to(self.device)

    def compute_advantages(
        self, last_values: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        """Compute GAE advantages and returns in-place."""
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, gae_lambda
        )

    def get_minibatches(
        self, batch_size: int, n_epochs: int
    ) -> Iterator[dict]:
        """Yield shuffled minibatches for PPO update epochs."""
        T, N = self.n_steps, self.n_envs
        total = T * N
        indices = np.arange(total)

        # Flatten [T, N] -> [T*N] for all tensors
        flat_states = self.states.view(total, self.history_length, -1)
        flat_prev_actions = self.prev_actions.view(total, self.history_length, -1)
        flat_task_ids = self.task_ids.view(total)
        flat_actions = self.actions.view(total, -1)
        flat_log_probs = self.log_probs.view(total)
        flat_values = self.values.view(total)
        flat_advantages = self.advantages.view(total)
        flat_returns = self.returns.view(total)

        # Flatten images: {cam: [T*N, H, C, h, w]} numpy
        flat_images = {}
        for cam, arr in self.images.items():
            flat_images[cam] = arr.reshape(total, *arr.shape[2:])

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                idx = indices[start : start + batch_size]
                idx_t = torch.from_numpy(idx).long().to(self.device)

                # Convert images to float32 GPU tensors for this minibatch only
                mb_images = {}
                for cam, flat in flat_images.items():
                    img_np = flat[idx]  # [B, H, C, h, w] uint8
                    mb_images[cam] = torch.from_numpy(img_np).float().to(self.device) / 255.0

                yield {
                    "batch": {
                        "images": mb_images,
                        "state": flat_states[idx_t],
                        "prev_actions": flat_prev_actions[idx_t],
                        "task_id": flat_task_ids[idx_t],
                    },
                    "actions": flat_actions[idx_t],
                    "old_log_probs": flat_log_probs[idx_t],
                    "old_values": flat_values[idx_t],
                    "advantages": flat_advantages[idx_t],
                    "returns": flat_returns[idx_t],
                }


def collect_rollouts(
    vec_env,
    actor_critic,
    history_buffers: List[HistoryBuffer],
    rollout_buffer: RolloutBuffer,
    n_steps: int,
    device: torch.device,
    task_id_per_env: Optional[List[int]] = None,
    task_name_per_env: Optional[List[str]] = None,
) -> dict:
    """Collect n_steps of experience from all parallel environments.

    Expects a ``MultiTaskVecEnv``-style API: step returns
    ``(List[obs], rewards, terms, truncs, List[info])`` with per-env
    observation dicts (so tasks can have differing native state dims).

    When ``task_id_per_env`` / ``task_name_per_env`` are provided, per-task
    episode stats are returned under ``per_task`` (keyed by task name if
    given, else by task id).
    """
    n_envs = len(history_buffers)
    current_rewards = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=int)

    global_rewards: List[float] = []
    global_lengths: List[int] = []
    global_successes: List[bool] = []
    per_task: Dict[object, Dict[str, list]] = {}

    def _task_key(i: int):
        if task_name_per_env is not None:
            return task_name_per_env[i]
        if task_id_per_env is not None:
            return task_id_per_env[i]
        return None

    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    for step in range(n_steps):
        # Build batched input from all history buffers (each carries its own
        # task_id + norm_stats, so the stacked batch already respects tasks).
        batches = [hb.get_batch(device=device) for hb in history_buffers]
        batch = _stack_batches(batches, device)

        with amp_ctx, torch.no_grad():
            actions, log_probs, values = actor_critic.act(batch)

        actions_np = actions.float().cpu().numpy()  # [N, action_dim]

        obs_list, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
        dones = np.logical_or(terminateds, truncateds)

        rollout_buffer.insert(step, batch, actions, log_probs, values, rewards, dones)

        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            info_i = infos[i] if isinstance(infos, list) else {}
            if dones[i]:
                succ = bool(info_i.get("success", False))
                global_rewards.append(float(current_rewards[i]))
                global_lengths.append(int(current_lengths[i]))
                global_successes.append(succ)
                key = _task_key(i)
                if key is not None:
                    stats = per_task.setdefault(
                        key, {"rewards": [], "lengths": [], "successes": []}
                    )
                    stats["rewards"].append(float(current_rewards[i]))
                    stats["lengths"].append(int(current_lengths[i]))
                    stats["successes"].append(succ)
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                history_buffers[i].reset()

            # Push next observation (per-env dict from MultiTaskVecEnv)
            o = obs_list[i]
            history_buffers[i].push(o["images"], o["state"])
            raw_action = info_i.get("raw_action", actions_np[i])
            history_buffers[i].record_action(raw_action)

    # Bootstrap value for GAE
    batches = [hb.get_batch(device=device) for hb in history_buffers]
    batch = _stack_batches(batches, device)
    with amp_ctx, torch.no_grad():
        _, _, last_values = actor_critic.act(batch)

    per_task_summary = {
        k: {
            "mean_reward": float(np.mean(v["rewards"])) if v["rewards"] else 0.0,
            "mean_length": float(np.mean(v["lengths"])) if v["lengths"] else 0.0,
            "success_rate": float(np.mean(v["successes"])) if v["successes"] else 0.0,
            "n_episodes": len(v["rewards"]),
        }
        for k, v in per_task.items()
    }

    return {
        "last_values": last_values,
        "mean_reward": float(np.mean(global_rewards)) if global_rewards else 0.0,
        "mean_length": float(np.mean(global_lengths)) if global_lengths else 0.0,
        "success_rate": float(np.mean(global_successes)) if global_successes else 0.0,
        "n_episodes": len(global_rewards),
        "per_task": per_task_summary,
    }


def _stack_batches(batches: List[dict], device: torch.device) -> dict:
    """Stack B=1 batches from individual HistoryBuffers into a single batch."""
    images = {}
    cam_keys = batches[0]["images"].keys()
    for cam in cam_keys:
        images[cam] = torch.cat([b["images"][cam] for b in batches], dim=0)

    return {
        "images": images,
        "state": torch.cat([b["state"] for b in batches], dim=0),
        "prev_actions": torch.cat([b["prev_actions"] for b in batches], dim=0),
        "task_id": torch.cat([b["task_id"] for b in batches], dim=0),
    }

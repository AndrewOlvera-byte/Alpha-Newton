"""Rollout buffer and collection for on-policy PPO training.

Stores transitions from parallel environments and provides minibatch
iteration for PPO updates. Images are kept as uint8 on CPU to save
GPU memory; conversion to float32 happens lazily per minibatch.
"""
from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

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

        # Per-step storage: images on GPU as uint8. Previously stored on CPU
        # as a numpy array and copied float32 per-minibatch — that was the
        # single biggest source of both host-RAM pressure and PCIe traffic
        # during the update phase. uint8 on GPU costs ~0.9 GB at
        # T=192, N=16, hist=3, 2 cams, 128² (~1.4 GB at 160²) and fits
        # comfortably alongside the ViT on a 16 GB card.
        self.images: Dict[str, torch.Tensor] = {}
        for cam, (h, w, c) in image_specs.items():
            # [T, N, H, C, h, w] -- stored per history frame
            self.images[cam] = torch.zeros(
                (n_steps, n_envs, history_length, c, h, w),
                dtype=torch.uint8, device=device,
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
        # images: {cam: [N, H, C, h, w]} as float tensors -- store as uint8 on GPU
        for cam in self.images:
            imgs = batch["images"][cam]  # [N, H, C, h, w] float [0,1] on device
            self.images[cam][step].copy_(
                imgs.mul(255).clamp_(0, 255).to(torch.uint8), non_blocking=True
            )

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

    def normalize_advantages_per_task(self) -> None:
        """Z-score advantages within each task before minibatching.

        A medium model learning 4 tasks with different horizons and reward
        scales gets dominated by whichever task has the biggest advantage
        variance. Normalizing per-task equalizes gradient magnitude across
        tasks — important when downweighted tasks (e.g. lift at n_env=1)
        would otherwise produce advantages with different characteristic
        scale than tool_hang at n_env=8.
        """
        adv = self.advantages          # [T, N]
        tids = self.task_ids           # [T, N]
        flat_adv = adv.reshape(-1)
        flat_tids = tids.reshape(-1)
        out = flat_adv.clone()
        for tid in torch.unique(flat_tids):
            mask = flat_tids == tid
            a = flat_adv[mask]
            if a.numel() > 1:
                out[mask] = (a - a.mean()) / (a.std() + 1e-8)
            else:
                out[mask] = a - a.mean()
        self.advantages = out.view_as(adv)

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

        # Flatten images on-device: {cam: [T*N, H, C, h, w]} uint8 GPU tensor
        flat_images = {}
        for cam, arr in self.images.items():
            flat_images[cam] = arr.view(total, *arr.shape[2:])

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                idx = indices[start : start + batch_size]
                idx_t = torch.from_numpy(idx).long().to(self.device)

                # No CPU→GPU hop — index the uint8 GPU buffer directly and
                # cast to float inside the minibatch. This used to be a 1+ GB
                # pinned copy per epoch for nothing.
                mb_images = {}
                for cam, flat in flat_images.items():
                    mb_images[cam] = flat.index_select(0, idx_t).float().mul_(1.0 / 255.0)

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
    profile: bool = False,
    show_progress: bool = False,
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

    t0 = time.time()
    # Fine-grained wall-time breakdown per phase — lets us pick the right
    # optimization lever (render vs IPC vs transfer vs forward).
    t_batch_build = 0.0
    t_forward = 0.0
    t_vec_step = 0.0
    t_insert = 0.0
    t_hb_push = 0.0
    pbar = None
    iterator = range(n_steps)
    if show_progress:
        pbar = tqdm(
            iterator,
            desc="rollout",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        iterator = pbar

    for step in iterator:
        _t = time.time()
        # Build batched input from all history buffers (each carries its own
        # task_id + norm_stats, so the stacked batch already respects tasks).
        batches = [hb.get_batch(device=device) for hb in history_buffers]
        batch = _stack_batches(batches, device)
        t_batch_build += time.time() - _t

        _t = time.time()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
            actions, log_probs, values = actor_critic.act(batch)
        actions_np = actions.float().cpu().numpy()  # [N, action_dim]
        # Only synchronize for diagnostics. A device-wide synchronize inside
        # background rollouts makes async PPO wait on unrelated update kernels.
        if profile and device.type == "cuda":
            torch.cuda.synchronize()
        t_forward += time.time() - _t

        _t = time.time()
        obs_list, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)
        t_vec_step += time.time() - _t
        dones = np.logical_or(terminateds, truncateds)

        _t = time.time()
        rollout_buffer.insert(step, batch, actions, log_probs, values, rewards, dones)
        t_insert += time.time() - _t

        current_rewards += rewards
        current_lengths += 1

        done_indices = np.flatnonzero(dones).tolist()
        reset_obs_by_env = {}
        if done_indices:
            if not hasattr(vec_env, "reset_at"):
                raise RuntimeError(
                    "vec_env must implement reset_at(indices) so PPO can autoreset "
                    "only terminated/truncated robosuite envs"
                )
            reset_obs, _reset_infos = vec_env.reset_at(done_indices)
            reset_obs_by_env = dict(zip(done_indices, reset_obs))

        _t = time.time()
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
                o = reset_obs_by_env[i]
            else:
                raw_action = info_i.get("raw_action", actions_np[i])
                history_buffers[i].record_action(raw_action)
                o = obs_list[i]

            # Push next observation (per-env dict from MultiTaskVecEnv)
            history_buffers[i].push(o["images"], o["state"])
        t_hb_push += time.time() - _t

        # Live FPS — env-steps/sec across all envs.
        if pbar is not None and ((step + 1) % 8 == 0 or step == n_steps - 1):
            elapsed = max(time.time() - t0, 1e-6)
            fps = (step + 1) * n_envs / elapsed
            pbar.set_postfix(fps=f"{fps:.0f}", eps=len(global_rewards))

    if pbar is not None:
        pbar.close()
    rollout_time = time.time() - t0

    # Bootstrap value for GAE
    batches = [hb.get_batch(device=device) for hb in history_buffers]
    batch = _stack_batches(batches, device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
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

    lengths_arr = np.asarray(global_lengths) if global_lengths else np.zeros(1)
    rewards_arr = np.asarray(global_rewards) if global_rewards else np.zeros(1)
    return {
        "last_values": last_values,
        "rollout_time": rollout_time,
        "rollout_fps": float(n_steps * n_envs / max(rollout_time, 1e-6)),
        "mean_reward": float(rewards_arr.mean()) if global_rewards else 0.0,
        "reward_std": float(rewards_arr.std()) if global_rewards else 0.0,
        "reward_min": float(rewards_arr.min()) if global_rewards else 0.0,
        "reward_max": float(rewards_arr.max()) if global_rewards else 0.0,
        "mean_length": float(lengths_arr.mean()) if global_lengths else 0.0,
        "length_p50": float(np.median(lengths_arr)) if global_lengths else 0.0,
        "length_p95": float(np.quantile(lengths_arr, 0.95)) if global_lengths else 0.0,
        "length_min": int(lengths_arr.min()) if global_lengths else 0,
        "length_max": int(lengths_arr.max()) if global_lengths else 0,
        "success_rate": float(np.mean(global_successes)) if global_successes else 0.0,
        "n_episodes": len(global_rewards),
        "per_task": per_task_summary,
        "timing": {
            "total_s": rollout_time,
            "batch_build_s": t_batch_build,
            "forward_s": t_forward,
            "vec_step_s": t_vec_step,
            "insert_s": t_insert,
            "hb_push_s": t_hb_push,
        },
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

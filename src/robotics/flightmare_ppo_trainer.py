"""PPO trainer for state-only Flightmare racing policies."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.core.registry import register
from src.robotics.curriculum import Curriculum
from src.robotics.flightmare_envs import make_flightmare_vec_env
from src.robotics.loss import compute_gae, ppo_clip_loss, value_loss


class FlightmareRolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

        self.states = torch.zeros(n_steps, n_envs, state_dim, device=device)
        self.prev_actions = torch.zeros(n_steps, n_envs, action_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, action_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)

    def insert(
        self,
        step: int,
        batch: dict,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.states[step] = batch["state"]
        self.prev_actions[step] = batch["prev_actions"]
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.rewards[step] = torch.from_numpy(rewards).to(self.device)
        self.dones[step] = torch.from_numpy(dones.astype(np.float32)).to(self.device)

    def compute_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, gae_lambda
        )

    def normalize_advantages(self) -> None:
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_minibatches(self, batch_size: int, n_epochs: int):
        total = self.n_steps * self.n_envs
        indices = np.arange(total)
        flat_states = self.states.view(total, self.state_dim)
        flat_prev_actions = self.prev_actions.view(total, self.action_dim)
        flat_actions = self.actions.view(total, self.action_dim)
        flat_log_probs = self.log_probs.view(total)
        flat_values = self.values.view(total)
        flat_advantages = self.advantages.view(total)
        flat_returns = self.returns.view(total)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                idx = indices[start:start + batch_size]
                idx_t = torch.from_numpy(idx).long().to(self.device)
                yield {
                    "batch": {
                        "images": {},
                        "state": flat_states.index_select(0, idx_t),
                        "prev_actions": flat_prev_actions.index_select(0, idx_t),
                    },
                    "actions": flat_actions.index_select(0, idx_t),
                    "old_log_probs": flat_log_probs.index_select(0, idx_t),
                    "old_values": flat_values.index_select(0, idx_t),
                    "advantages": flat_advantages.index_select(0, idx_t),
                    "returns": flat_returns.index_select(0, idx_t),
                }


def _stack_obs(obs_list: list[dict], prev_actions: np.ndarray, device: torch.device) -> dict:
    states = np.stack([obs["state"] for obs in obs_list], axis=0).astype(np.float32)
    return {
        "images": {},
        "state": torch.from_numpy(states).to(device),
        "prev_actions": torch.from_numpy(prev_actions.astype(np.float32)).to(device),
    }


def _initial_prev_actions(infos: list[dict], n_envs: int, action_dim: int) -> np.ndarray:
    prev_actions = np.zeros((n_envs, action_dim), dtype=np.float32)
    for i, info in enumerate(infos):
        initial = info.get("initial_prev_action_norm")
        if initial is None:
            continue
        initial = np.asarray(initial, dtype=np.float32)
        if initial.shape == (action_dim,):
            prev_actions[i] = initial
    return prev_actions


def collect_flightmare_rollouts(
    vec_env,
    model: nn.Module,
    obs_list: list[dict],
    prev_actions: np.ndarray,
    buffer: FlightmareRolloutBuffer,
    n_steps: int,
    device: torch.device,
    use_bf16: bool = False,
    action_clip: float | None = None,
) -> tuple[list[dict], np.ndarray, dict]:
    n_envs = len(obs_list)
    current_rewards = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    successes: list[bool] = []
    gate_completion: list[float] = []
    gate_passes = 0
    gate_misses = 0
    crash_count = 0
    max_speed = 0.0
    speed_sum = 0.0
    speed_count = 0
    action_clip_sum = 0.0
    action_clip_count = 0
    reward_sum = 0.0
    reward_sq_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")
    reward_term_sums: dict[str, float] = {}
    reward_term_count = 0

    t0 = time.time()
    amp = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16 and device.type == "cuda",
    )

    for step in range(n_steps):
        batch = _stack_obs(obs_list, prev_actions, device)
        with torch.no_grad(), amp:
            sampled_actions, _, values = model.act(batch)
            if action_clip is not None:
                clip = float(action_clip)
                actions = sampled_actions.clamp(-clip, clip)
                log_probs, _, eval_values = model.evaluate_actions(batch, actions)
                if eval_values is not None:
                    values = eval_values
                action_clip_sum += float((sampled_actions.abs() > clip).float().mean().item())
                action_clip_count += 1
            else:
                actions = sampled_actions
                log_probs, _, eval_values = model.evaluate_actions(batch, actions)
                if eval_values is not None:
                    values = eval_values
        actions_np = actions.float().cpu().numpy()

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions_np)
        dones = np.logical_or(terms, truncs)
        buffer.insert(step, batch, actions, log_probs, values, rewards, dones)

        reward_sum += float(np.sum(rewards))
        reward_sq_sum += float(np.sum(np.square(rewards)))
        reward_min = min(reward_min, float(np.min(rewards)))
        reward_max = max(reward_max, float(np.max(rewards)))
        current_rewards += rewards
        current_lengths += 1

        done_indices = np.flatnonzero(dones).tolist()
        reset_obs_by_env = {}
        reset_prev_by_env = {}
        if done_indices:
            reset_obs, reset_infos = vec_env.reset_at(done_indices)
            reset_obs_by_env = dict(zip(done_indices, reset_obs))
            reset_prev_by_env = {
                env_i: np.asarray(info.get("initial_prev_action_norm", np.zeros(buffer.action_dim)), dtype=np.float32)
                for env_i, info in zip(done_indices, reset_infos)
            }

        for i, info in enumerate(infos):
            speed = float(info.get("speed_mps", 0.0))
            max_speed = max(max_speed, speed)
            speed_sum += speed
            speed_count += 1
            if info.get("gate_passed", False):
                gate_passes += 1
            if info.get("gate_missed", False):
                gate_misses += 1
            if info.get("crash", False):
                crash_count += 1
            terms = info.get("reward_terms")
            if isinstance(terms, dict):
                reward_term_count += 1
                for key, value in terms.items():
                    reward_term_sums[key] = reward_term_sums.get(key, 0.0) + float(value)

            if dones[i]:
                ep_rewards.append(float(current_rewards[i]))
                ep_lengths.append(int(current_lengths[i]))
                successes.append(bool(info.get("success", False)))
                gate_completion.append(float(info.get("gate_completion", 0.0)))
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                next_obs[i] = reset_obs_by_env[i]
                reset_prev = reset_prev_by_env.get(i)
                if reset_prev is not None and reset_prev.shape == prev_actions[i].shape:
                    prev_actions[i] = reset_prev
                else:
                    prev_actions[i] = 0.0
            else:
                applied_action = info.get("action_norm")
                if applied_action is not None:
                    applied_action = np.asarray(applied_action, dtype=np.float32)
                    if applied_action.shape == prev_actions[i].shape:
                        prev_actions[i] = applied_action
                    else:
                        prev_actions[i] = actions_np[i]
                else:
                    prev_actions[i] = actions_np[i]

        obs_list = next_obs

    batch = _stack_obs(obs_list, prev_actions, device)
    with torch.no_grad(), amp:
        _, _, last_values = model.act(batch)

    elapsed = time.time() - t0
    rewards_arr = np.asarray(ep_rewards, dtype=np.float32) if ep_rewards else np.zeros(1, dtype=np.float32)
    lengths_arr = np.asarray(ep_lengths, dtype=np.float32) if ep_lengths else np.zeros(1, dtype=np.float32)
    n_samples = max(1, n_steps * n_envs)
    step_reward_mean = reward_sum / n_samples
    step_reward_var = max(0.0, reward_sq_sum / n_samples - step_reward_mean ** 2)
    stats = {
        "last_values": last_values,
        "rollout_time": elapsed,
        "rollout_fps": float(n_steps * n_envs / max(elapsed, 1e-6)),
        "step_reward_mean": float(step_reward_mean),
        "step_reward_std": float(np.sqrt(step_reward_var)),
        "step_reward_min": float(reward_min if np.isfinite(reward_min) else 0.0),
        "step_reward_max": float(reward_max if np.isfinite(reward_max) else 0.0),
        "mean_reward": float(rewards_arr.mean()) if ep_rewards else 0.0,
        "reward_std": float(rewards_arr.std()) if ep_rewards else 0.0,
        "reward_min": float(rewards_arr.min()) if ep_rewards else 0.0,
        "reward_max": float(rewards_arr.max()) if ep_rewards else 0.0,
        "mean_length": float(lengths_arr.mean()) if ep_lengths else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_gate_completion": float(np.mean(gate_completion)) if gate_completion else 0.0,
        "n_episodes": len(ep_rewards),
        "gate_passes": int(gate_passes),
        "gate_misses": int(gate_misses),
        "crashes": int(crash_count),
        "mean_speed_mps": float(speed_sum / max(1, speed_count)),
        "max_speed_mps": float(max_speed),
        "action_clip_fraction": float(action_clip_sum / max(1, action_clip_count)),
    }
    if reward_term_sums:
        denom = max(1, reward_term_count)
        stats["reward_terms"] = {k: float(v / denom) for k, v in sorted(reward_term_sums.items())}
    return obs_list, prev_actions, stats


@register("trainer", "ppo_flightmare")
class FlightmarePPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        training_cfg: dict,
        robotics_cfg: dict,
        wandb_cfg: Optional[dict] = None,
        data_cfg: Optional[dict] = None,
    ):
        self.model = model
        self.training_cfg = training_cfg
        self.robotics_cfg = robotics_cfg
        self.wandb_cfg = wandb_cfg or {}
        self.data_cfg = data_cfg or {}

        ppo = robotics_cfg.get("ppo", {})
        self.ppo_cfg = ppo
        self.n_envs = int(ppo.get("n_envs", 16))
        self.n_steps = int(ppo.get("n_steps", 256))
        self.ppo_epochs = int(ppo.get("ppo_epochs", 4))
        self.minibatch_size = int(ppo.get("minibatch_size", 1024))
        self.clip_eps = float(ppo.get("clip_eps", 0.2))
        self.vf_coeff = float(ppo.get("vf_coeff", 0.5))
        self.ent_coeff = float(ppo.get("ent_coeff", 0.005))
        self.gamma = float(ppo.get("gamma", 0.995))
        self.gae_lambda = float(ppo.get("gae_lambda", 0.95))
        self.max_iterations = int(ppo.get("max_iterations", 1000))
        self.actor_lr = float(ppo.get("actor_lr", 3e-5))
        self.critic_lr = float(ppo.get("critic_lr", 1e-4))
        target_kl = ppo.get("target_kl")
        self.target_kl = float(target_kl) if target_kl is not None else None
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 0.5))
        self.use_bf16 = bool(training_cfg.get("bf16", True))
        self.output_dir = training_cfg.get("output_dir", "outputs/flightmare_ppo")
        self.logging_steps = int(training_cfg.get("logging_steps", 1))
        self.save_steps = int(training_cfg.get("save_steps", 25) or 0)
        self.best_steps = int(training_cfg.get("best_steps", self.save_steps) or 0)
        self.selection_metric = str(training_cfg.get("selection_metric", "rollout_score"))
        default_score_weights = {
            "success_rate": 1.0,
            "mean_gate_completion": 0.1,
            "mean_reward": 0.0,
            "mean_speed_mps": 0.0,
            "mean_length": 0.0,
            # Stage-aware best: each curriculum stage adds a flat bonus so
            # progress (e.g. solid success on a later stage) outranks an
            # already-saturated win on an earlier one. Default chosen so one
            # full stage advance dominates ~2x any within-stage success swing.
            "stage_index": 2.0,
        }
        configured_weights = training_cfg.get("rollout_score_weights", {}) or {}
        self.rollout_score_weights = {
            key: float(configured_weights.get(key, value))
            for key, value in default_score_weights.items()
        }
        self.best_score = float("-inf")
        self.best_iteration = 0
        guard_cfg = ppo.get("collapse_guard", {}) or {}
        self.collapse_guard_enabled = bool(guard_cfg.get("enabled", False))
        self.collapse_guard_after_iter = int(guard_cfg.get("after_iter", 25))
        self.collapse_guard_patience = int(guard_cfg.get("patience", 3))
        self.collapse_guard_min_success = float(guard_cfg.get("min_success_rate", 0.05))
        self.collapse_guard_max_gate_completion = float(guard_cfg.get("max_gate_completion", 0.25))
        self.collapse_guard_skip_update = bool(guard_cfg.get("skip_update", True))
        self._collapse_guard_count = 0
        # When set, the trainer restores actor+critic weights, the AdamW
        # optimizer state, and the rolling best-score from this directory at
        # the start of train(). Falls back gracefully when optimizer.pt or
        # training_state.json are missing — e.g. when the source checkpoint
        # was saved before save_optimizer_state was enabled, in which case
        # only the model weights are restored and the optimizer starts fresh.
        self.resume_from = ppo.get("resume_from") or training_cfg.get("resume_from")
        self.start_iteration = 1
        self.save_total_limit = training_cfg.get("save_total_limit")
        self.save_total_limit = int(self.save_total_limit) if self.save_total_limit else None
        self.save_final = bool(training_cfg.get("save_final", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optional course curriculum. When enabled, the trainer rebuilds the
        # vec env at stage transitions to update num_gates / start randomness /
        # gate noise. Stage 0 is applied at training start.
        cur_cfg = ppo.get("curriculum", {}) or {}
        if cur_cfg.get("enabled", False):
            self.curriculum = Curriculum.from_config(cur_cfg)
            self._base_ent_coeff = self.ent_coeff
        else:
            self.curriculum = None
            self._base_ent_coeff = self.ent_coeff

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    def _disable_dropout(self) -> int:
        """PPO ratios assume the policy evaluated during updates matches rollout collection."""
        disabled = 0
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) and module.p != 0.0:
                module.p = 0.0
                disabled += 1
        return disabled

    def _env_kwargs(self) -> dict:
        ppo = self.ppo_cfg
        return {
            "n_workers": int(ppo.get("n_workers", 0)),
            "data_dir": self.data_cfg.get("data_dir", "data/flightmare/bc_v1"),
            "action_type": self.data_cfg.get("action_type", ppo.get("action_type", "ctbr")),
            "obs_schema": self.data_cfg.get("obs_schema", ppo.get("obs_schema", "v2")),
            "include_mission": self.data_cfg.get("include_mission", True),
            "normalize_state": self.data_cfg.get("normalize_state", True),
            "normalize_mission": self.data_cfg.get("normalize_mission", True),
            "normalize_action": self.data_cfg.get("normalize_action", True),
            "action_normalization": self.data_cfg.get("action_normalization", "auto"),
            "control_hz": ppo.get("control_hz", 100.0),
            "horizon": ppo.get("horizon", 1500),
            "scene": ppo.get("scene", "industrial"),
            "render": ppo.get("render", False),
            "backend": ppo.get("backend", "auto"),
            "course_mode": ppo.get("course_mode", "gates"),
            "gate_layout": ppo.get("gate_layout"),
            "random_start_gate": ppo.get("random_start_gate", False),
            "fixed_gate_pos_noise": ppo.get("fixed_gate_pos_noise", 0.0),
            "fixed_gate_pos_noise_xyz": ppo.get("fixed_gate_pos_noise_xyz"),
            "fixed_gate_yaw_noise": ppo.get("fixed_gate_yaw_noise", 0.0),
            "num_gates": ppo.get("num_gates", 8),
            "gate_spacing_range": ppo.get("gate_spacing_range", [4.0, 9.0]),
            "gate_lateral_jitter": ppo.get("gate_lateral_jitter", 2.0),
            "gate_z_range": ppo.get("gate_z_range", [1.5, 4.0]),
            "gate_yaw_step": ppo.get("gate_yaw_step", 0.7),
            "gate_yaw_noise": ppo.get("gate_yaw_noise", 0.25),
            "gate_size": ppo.get("gate_size", 1.0),
            "gate_approach_m": ppo.get("gate_approach_m", 1.2),
            "z_min": ppo.get("z_min", 1.0),
            "gate_vehicle_radius": ppo.get("gate_vehicle_radius", 0.15),
            "max_world_radius": ppo.get("max_world_radius", 120.0),
            "min_z": ppo.get("min_z", -0.25),
            "terminate_on_gate_miss": ppo.get("terminate_on_gate_miss", True),
            "terminate_on_crash": ppo.get("terminate_on_crash", True),
            "reward_fn": ppo.get("reward_fn", "flightmare_racing_v1"),
            "reward_kwargs": ppo.get("reward_kwargs", {}),
            "max_body_rate": ppo.get("max_body_rate", 8.0),
            "max_waypoint_speed": ppo.get("max_waypoint_speed", 15.0),
            "max_collective_thrust_g": ppo.get("max_collective_thrust_g", 4.0),
            "plant": ppo.get("plant"),
            "action_bounds_override": ppo.get("action_bounds_override"),
            "reset_mode": ppo.get("reset_mode", "course_start"),
            "start_gate_index": ppo.get("start_gate_index"),
            "start_gate_choices": ppo.get("start_gate_choices"),
            "start_offset_m": ppo.get("start_offset_m", 3.0),
            "start_offset_range": ppo.get("start_offset_range"),
            "reference_speed_mps": ppo.get("reference_speed_mps", 4.0),
            "goal_mode": ppo.get("goal_mode", "finish_remaining_course"),
            "goal_gate_span": ppo.get("goal_gate_span"),
            "action_clip": ppo.get(
                "action_clip",
                (self.robotics_cfg.get("architecture", {}) or {}).get("action_clip", 5.0),
            ),
        }

    def _apply_curriculum_overrides(self, env_kwargs: dict) -> dict:
        """Layer the active curriculum stage onto a base env_kwargs dict."""
        if self.curriculum is None:
            return env_kwargs
        env_overrides = dict(self.curriculum.env_overrides())
        merged = dict(env_kwargs)
        reward_overrides = env_overrides.pop("reward_kwargs", None)
        plant_overrides = env_overrides.pop("plant", None)
        bounds_overrides = env_overrides.pop("action_bounds_override", None)
        merged.update(env_overrides)
        if reward_overrides is not None:
            merged["reward_kwargs"] = {
                **(env_kwargs.get("reward_kwargs", {}) or {}),
                **(reward_overrides or {}),
            }
        if plant_overrides is not None:
            merged["plant"] = {
                **(env_kwargs.get("plant") or {}),
                **(plant_overrides or {}),
            }
        if bounds_overrides is not None:
            merged["action_bounds_override"] = {
                **(env_kwargs.get("action_bounds_override") or {}),
                **(bounds_overrides or {}),
            }
        # Trainer-side overrides (entropy coefficient).
        trainer_overrides = self.curriculum.trainer_overrides()
        if "ent_coeff_override" in trainer_overrides:
            self.ent_coeff = float(trainer_overrides["ent_coeff_override"])
        else:
            self.ent_coeff = float(self._base_ent_coeff)
        return merged

    def _split_params(self):
        critic_params = []
        if getattr(self.model, "head", None) is not None and getattr(self.model.head, "critic", None) is not None:
            critic_params = [p for p in self.model.head.critic.parameters() if p.requires_grad]
        critic_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in critic_ids]
        return actor_params, critic_params

    def _init_wandb(self) -> None:
        """Initialize WandB with the same interactive fallback as the BC trainer."""
        project = self.wandb_cfg.get("project")
        if not project:
            print("[Flightmare PPO] No wandb project set - skipping WandB logging.")
            self._wandb = None
            return

        mode = self.wandb_cfg.get("mode") or os.environ.get("WANDB_MODE", "").strip()
        if mode == "disabled":
            print("[Flightmare PPO] WandB disabled by config/env.")
            self._wandb = None
            return

        try:
            import wandb

            api_key = (
                os.environ.get("WANDB_API_KEY", "").strip()
                or (wandb.api.api_key or "")
            )
            if not api_key and mode not in {"offline", "dryrun"}:
                if sys.stdin.isatty():
                    print("\n[Wandb] No API key found.")
                    api_key = input(
                        "[Wandb] Enter WANDB_API_KEY (or press Enter to skip): "
                    ).strip()
                    if api_key:
                        os.environ["WANDB_API_KEY"] = api_key
                        wandb.login(key=api_key, relogin=True)
                    else:
                        print("[Wandb] Skipping wandb logging.\n")
                        self._wandb = None
                        return
                else:
                    print("[Wandb] No API key and non-interactive session - skipping WandB.")
                    self._wandb = None
                    return

            init_kwargs = {
                "project": project,
                "entity": self.wandb_cfg.get("entity"),
                "name": self.wandb_cfg.get("run_name", "flightmare_ppo"),
                "tags": self.wandb_cfg.get("tags", []),
                "config": {
                    "ppo": self.ppo_cfg,
                    "training": self.training_cfg,
                    "robotics": self.robotics_cfg,
                    "data": self.data_cfg,
                },
            }
            if mode:
                init_kwargs["mode"] = mode
            wandb.init(**init_kwargs)
            self._wandb = wandb
            print(f"[Wandb] Logging to {project}/{wandb.run.name}")
        except Exception as e:
            print(f"[Wandb] Init failed: {e}. Continuing without WandB.")
            self._wandb = None

    def _log(self, metrics: dict[str, Any], step: int) -> None:
        if getattr(self, "_wandb", None) is not None and self._wandb.run:
            self._wandb.log(metrics, step=step)

    def _selection_score(self, stats: dict) -> float:
        if self.selection_metric == "success_rate":
            return float(stats["success_rate"])
        if self.selection_metric == "mean_reward":
            return float(stats["mean_reward"])
        if self.selection_metric == "gate_completion":
            return float(stats["mean_gate_completion"])
        if self.selection_metric != "rollout_score":
            raise ValueError(
                "training.selection_metric must be one of "
                "'rollout_score', 'success_rate', 'mean_reward', or 'gate_completion'."
            )
        stage_idx = int(self.curriculum.active_index) if self.curriculum is not None else 0
        return float(
            self.rollout_score_weights["success_rate"] * stats["success_rate"]
            + self.rollout_score_weights["mean_gate_completion"] * stats["mean_gate_completion"]
            + self.rollout_score_weights["mean_reward"] * stats["mean_reward"]
            + self.rollout_score_weights["mean_speed_mps"] * stats["mean_speed_mps"]
            + self.rollout_score_weights["mean_length"] * stats["mean_length"]
            + self.rollout_score_weights["stage_index"] * stage_idx
        )

    def _checkpoint_metrics(self, stats: dict, score: float) -> dict[str, float | int]:
        return {
            "selection_score": float(score),
            "stage_index": int(self.curriculum.active_index) if self.curriculum is not None else 0,
            "success_rate": float(stats["success_rate"]),
            "mean_gate_completion": float(stats["mean_gate_completion"]),
            "mean_reward": float(stats["mean_reward"]),
            "step_reward_mean": float(stats["step_reward_mean"]),
            "mean_length": float(stats["mean_length"]),
            "n_episodes": int(stats["n_episodes"]),
            "gate_passes": int(stats["gate_passes"]),
            "gate_misses": int(stats["gate_misses"]),
            "crashes": int(stats["crashes"]),
            "mean_speed_mps": float(stats["mean_speed_mps"]),
            "max_speed_mps": float(stats["max_speed_mps"]),
            "action_clip_fraction": float(stats["action_clip_fraction"]),
        }

    def _collapse_guard_bad_rollout(self, iteration: int, stats: dict) -> bool:
        if not self.collapse_guard_enabled:
            return False
        if iteration < self.collapse_guard_after_iter:
            return False
        return (
            float(stats["success_rate"]) <= self.collapse_guard_min_success
            and float(stats["mean_gate_completion"]) <= self.collapse_guard_max_gate_completion
        )

    def _restore_best_checkpoint(self, model: nn.Module, optimizer) -> bool:
        best_dir = os.path.join(self.output_dir, "best")
        if not os.path.isdir(best_dir):
            return False
        self._restore_from_checkpoint(best_dir, model, optimizer)
        model.to(self.device)
        return True

    def smoke_test(self) -> None:
        disabled_dropout = self._disable_dropout()
        if disabled_dropout:
            print(f"[Flightmare PPO] Disabled {disabled_dropout} dropout layer(s) for PPO.")
        vec_env = make_flightmare_vec_env(n_envs=min(2, self.n_envs), seed=0, **self._env_kwargs())
        try:
            obs_list, infos = vec_env.reset(seed=0)
            print(f"[Flightmare PPO Test] envs={len(obs_list)} fallback={infos[0].get('using_fallback')}")
            print(f"[Flightmare PPO Test] state shape={obs_list[0]['state'].shape}")
            model = self.model.to(self.device).eval()
            prev_actions = _initial_prev_actions(infos, len(obs_list), int(self.model.action_dim))
            buffer = FlightmareRolloutBuffer(
                n_steps=8,
                n_envs=len(obs_list),
                state_dim=int(self.robotics_cfg.get("architecture", {}).get("state_dim", obs_list[0]["state"].shape[0])),
                action_dim=int(self.model.action_dim),
                device=self.device,
            )
            obs_list, prev_actions, stats = collect_flightmare_rollouts(
                vec_env,
                model,
                obs_list,
                prev_actions,
                buffer,
                8,
                self.device,
                self.use_bf16,
                action_clip=float(self._env_kwargs().get("action_clip", 1.0)),
            )
            buffer.compute_advantages(stats["last_values"], self.gamma, self.gae_lambda)
            buffer.normalize_advantages()
            mb = next(buffer.get_minibatches(batch_size=min(8, 8 * len(prev_actions)), n_epochs=1))
            with torch.no_grad():
                log_prob, entropy, value = model.evaluate_actions(mb["batch"], mb["actions"])
            print(
                "[Flightmare PPO Test] "
                f"rollout_fps={stats['rollout_fps']:.0f} "
                f"reward_step_mean={buffer.rewards.mean().item():.3f} "
                f"log_prob={log_prob.mean().item():.3f} "
                f"entropy={entropy.mean().item():.3f} "
                f"value={value.mean().item():.3f}"
            )
        finally:
            vec_env.close()

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        disabled_dropout = self._disable_dropout()
        if disabled_dropout:
            print(f"[Flightmare PPO] Disabled {disabled_dropout} dropout layer(s) for PPO.")
        self._init_wandb()
        model = self.model.to(self.device)
        model.train()

        actor_params, critic_params = self._split_params()
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": self.actor_lr, "weight_decay": 1e-4},
            {"params": critic_params, "lr": self.critic_lr, "weight_decay": 1e-4},
        ])

        if self.resume_from:
            self._restore_from_checkpoint(self.resume_from, model, optimizer)

        base_env_kwargs = self._env_kwargs()
        env_seed = int(self.ppo_cfg.get("seed", 0))
        if self.curriculum is not None:
            # Align the active curriculum stage before constructing the first
            # resumed environment.
            self.curriculum.update(self.start_iteration, None)
        if self.curriculum is not None:
            print(f"[Flightmare PPO] Curriculum enabled: {self.curriculum.describe()}")
        env_kwargs = self._apply_curriculum_overrides(base_env_kwargs)
        vec_env = make_flightmare_vec_env(n_envs=self.n_envs, seed=env_seed + self.start_iteration - 1, **env_kwargs)
        obs_list, infos = vec_env.reset(seed=env_seed + self.start_iteration - 1)
        if infos and infos[0].get("using_fallback"):
            print("[Flightmare PPO] WARNING: using numpy fallback; metrics are not authoritative.")

        state_dim = int(self.robotics_cfg.get("architecture", {}).get("state_dim", obs_list[0]["state"].shape[0]))
        action_dim = int(self.model.action_dim)
        prev_actions = _initial_prev_actions(infos, self.n_envs, action_dim)
        buffer = FlightmareRolloutBuffer(self.n_steps, self.n_envs, state_dim, action_dim, self.device)

        t_start = time.time()
        print("\n[Flightmare PPO] Starting training")
        print(f"  Envs: {self.n_envs} | Steps/rollout: {self.n_steps} | Epochs: {self.ppo_epochs}")
        print(f"  Action: {env_kwargs['action_type']} | Reward: {env_kwargs['reward_fn']}")
        print(f"  Actor LR: {self.actor_lr} | Critic LR: {self.critic_lr}")
        print(
            "  Best selection: "
            f"{self.selection_metric} every {self.best_steps or 'disabled'} iteration(s)"
        )
        print(f"  Iterations: {self.start_iteration} -> {self.max_iterations} | Device: {self.device}\n")

        amp = torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16 and self.device.type == "cuda",
        )

        last_rollout_stats: dict | None = None
        try:
            for iteration in range(self.start_iteration, self.max_iterations + 1):
                # Curriculum: advance stage if the iteration window expired or
                # the early-advance metric crossed its threshold last rollout.
                if self.curriculum is not None and self.curriculum.update(
                    iteration, last_rollout_stats
                ):
                    new_kwargs = self._apply_curriculum_overrides(base_env_kwargs)
                    env_kwargs = new_kwargs
                    print(
                        f"[Flightmare PPO] Curriculum transition at iter {iteration}: "
                        f"{self.curriculum.describe()} (ent_coeff={self.ent_coeff})"
                    )
                    try:
                        vec_env.close()
                    except Exception:
                        pass
                    vec_env = make_flightmare_vec_env(n_envs=self.n_envs, seed=env_seed + iteration, **new_kwargs)
                    obs_list, infos = vec_env.reset(seed=env_seed + iteration)
                    prev_actions = _initial_prev_actions(infos, self.n_envs, action_dim)
                iter_t0 = time.time()
                model.eval()
                obs_list, prev_actions, stats = collect_flightmare_rollouts(
                    vec_env,
                    model,
                    obs_list,
                    prev_actions,
                    buffer,
                    self.n_steps,
                    self.device,
                    self.use_bf16,
                    action_clip=float(env_kwargs.get("action_clip", 1.0)),
                )
                last_rollout_stats = stats
                buffer.compute_advantages(stats["last_values"], self.gamma, self.gae_lambda)
                buffer.normalize_advantages()

                selection_score = self._selection_score(stats)
                is_new_best = False
                if self.best_steps > 0 and iteration % self.best_steps == 0:
                    if selection_score > self.best_score:
                        self.best_score = selection_score
                        self.best_iteration = iteration
                        is_new_best = True
                        self._save_checkpoint(
                            model,
                            optimizer,
                            iteration,
                            is_best=True,
                            metrics=self._checkpoint_metrics(stats, selection_score),
                            best_score=self.best_score,
                        )

                guard_bad_rollout = self._collapse_guard_bad_rollout(iteration, stats)
                guard_restored = False
                skip_update = False
                if guard_bad_rollout:
                    self._collapse_guard_count += 1
                    skip_update = self.collapse_guard_skip_update
                    if self._collapse_guard_count >= self.collapse_guard_patience:
                        guard_restored = self._restore_best_checkpoint(model, optimizer)
                        if guard_restored:
                            self._collapse_guard_count = 0
                else:
                    self._collapse_guard_count = 0

                model.train()
                pol_loss_sum = val_loss_sum = ent_sum = kl_sum = clip_sum = grad_sum = 0.0
                n_updates = 0
                early_stop = False
                update_t0 = time.time()
                if not skip_update:
                    for mb in buffer.get_minibatches(self.minibatch_size, self.ppo_epochs):
                        adv = mb["advantages"]
                        with amp:
                            log_prob, entropy, value = model.evaluate_actions(mb["batch"], mb["actions"])
                            p_loss = ppo_clip_loss(log_prob, mb["old_log_probs"], adv, self.clip_eps)
                            v_loss = value_loss(value, mb["returns"], mb["old_values"], self.clip_eps)
                            ent = entropy.mean()
                            loss = p_loss + self.vf_coeff * v_loss - self.ent_coeff * ent

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        optimizer.step()

                        with torch.no_grad():
                            log_ratio = log_prob - mb["old_log_probs"]
                            ratio = log_ratio.exp()
                            approx_kl = ((ratio - 1.0) - log_ratio).mean()
                            clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                        pol_loss_sum += float(p_loss.item())
                        val_loss_sum += float(v_loss.item())
                        ent_sum += float(ent.item())
                        kl_sum += float(approx_kl.item())
                        clip_sum += float(clip_frac.item())
                        grad_sum += float(grad_norm)
                        n_updates += 1
                        if self.target_kl is not None and float(approx_kl.item()) > 1.5 * self.target_kl:
                            early_stop = True
                            break
                update_time = time.time() - update_t0

                if iteration % self.logging_steps == 0:
                    elapsed = time.time() - t_start
                    completed_iterations = iteration - self.start_iteration + 1
                    fps = completed_iterations * self.n_steps * self.n_envs / max(elapsed, 1e-6)
                    values_flat = buffer.values.flatten()
                    returns_flat = buffer.returns.flatten()
                    var_returns = returns_flat.var()
                    explained_var = (
                        1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)
                    ).item() if var_returns > 0 else 0.0
                    log_dict = {
                        "iteration": iteration,
                        "ppo/policy_loss": pol_loss_sum / max(1, n_updates),
                        "ppo/value_loss": val_loss_sum / max(1, n_updates),
                        "ppo/entropy": ent_sum / max(1, n_updates),
                        "ppo/approx_kl": kl_sum / max(1, n_updates),
                        "ppo/clip_fraction": clip_sum / max(1, n_updates),
                        "ppo/grad_norm": grad_sum / max(1, n_updates),
                        "ppo/explained_variance": explained_var,
                        "ppo/early_stop": float(early_stop),
                        "rollout/mean_reward": stats["mean_reward"],
                        "rollout/success_rate": stats["success_rate"],
                        "rollout/mean_gate_completion": stats["mean_gate_completion"],
                        "rollout/n_episodes": stats["n_episodes"],
                        "rollout/mean_length": stats["mean_length"],
                        "rollout/gate_passes": stats["gate_passes"],
                        "rollout/gate_misses": stats["gate_misses"],
                        "rollout/crashes": stats["crashes"],
                        "rollout/mean_speed_mps": stats["mean_speed_mps"],
                        "rollout/max_speed_mps": stats["max_speed_mps"],
                        "rollout/action_clip_fraction": stats["action_clip_fraction"],
                        "rollout/reward_step_mean": stats["step_reward_mean"],
                        "rollout/reward_step_std": stats["step_reward_std"],
                        "rollout/reward_step_min": stats["step_reward_min"],
                        "rollout/reward_step_max": stats["step_reward_max"],
                        "time/fps": fps,
                        "time/rollout_fps": stats["rollout_fps"],
                        "time/rollout_time": stats["rollout_time"],
                        "time/update_time": update_time,
                        "checkpoint/selection_score": selection_score,
                        "checkpoint/best_score": self.best_score if np.isfinite(self.best_score) else -1.0,
                        "checkpoint/best_iteration": self.best_iteration,
                        "checkpoint/is_new_best": float(is_new_best),
                        "collapse_guard/bad_rollout": float(guard_bad_rollout),
                        "collapse_guard/count": float(self._collapse_guard_count),
                        "collapse_guard/skipped_update": float(skip_update),
                        "collapse_guard/restored_best": float(guard_restored),
                        **(
                            {
                                "curriculum/stage_idx": float(self.curriculum.active_index),
                                "curriculum/num_stages": float(self.curriculum.num_stages),
                                "curriculum/ent_coeff": float(self.ent_coeff),
                            }
                            if self.curriculum is not None
                            else {}
                        ),
                        **{
                            f"reward_terms/{key}": float(value)
                            for key, value in stats.get("reward_terms", {}).items()
                        },
                    }
                    print(
                        f"[{iteration}/{self.max_iterations}] "
                        f"ep_r={stats['mean_reward']:.2f} "
                        f"step_r={stats['step_reward_mean']:+.3f} "
                        f"eps={stats['n_episodes']} len={stats['mean_length']:.0f} "
                        f"sr={stats['success_rate']:.1%} "
                        f"gc={stats['mean_gate_completion']:.1%} "
                        f"pass={stats['gate_passes']} miss={stats['gate_misses']} crash={stats['crashes']} "
                        f"v={stats['mean_speed_mps']:.1f}/{stats['max_speed_mps']:.1f} "
                        f"aclip={stats['action_clip_fraction']:.2f} "
                        f"pl={pol_loss_sum/max(1,n_updates):.4f} "
                        f"vl={val_loss_sum/max(1,n_updates):.4f} "
                        f"ent={ent_sum/max(1,n_updates):.3f} "
                        f"kl={kl_sum/max(1,n_updates):.4f} "
                        f"stop={int(early_stop)} "
                        f"guard={int(guard_bad_rollout)}/{self._collapse_guard_count} "
                        f"ev={explained_var:+.2f} "
                        f"fps={fps:.0f} rfps={stats['rollout_fps']:.0f} "
                        f"[roll={stats['rollout_time']:.1f}s upd={update_time:.1f}s]"
                    )
                    self._log(log_dict, step=iteration)
                    if is_new_best:
                        print(
                            "  [best] "
                            f"score={selection_score:.3f} "
                            f"sr={stats['success_rate']:.1%} "
                            f"gc={stats['mean_gate_completion']:.1%}"
                        )

                if self.save_steps > 0 and iteration % self.save_steps == 0:
                    self._save_checkpoint(
                        model,
                        optimizer,
                        iteration,
                        metrics=self._checkpoint_metrics(stats, selection_score),
                        best_score=self.best_score,
                    )
                    self._prune_checkpoints()
        finally:
            vec_env.close()

        if self.save_final:
            self._save_checkpoint(
                model,
                optimizer,
                self.max_iterations,
                name="final",
                best_score=self.best_score,
            )
        if getattr(self, "_wandb", None) is not None and self._wandb.run:
            self._wandb.finish()

    def _restore_from_checkpoint(self, path: str, model: nn.Module, optimizer) -> None:
        """Restore actor+critic weights, optimizer state, and best-score
        bookkeeping from a previous PPO checkpoint directory.

        The BC entrypoint's ``from_bc_checkpoint`` already preloaded matching
        weights when the same path was passed as ``bc_checkpoint``; this is a
        belt-and-suspenders re-load so configs that only set ``resume_from``
        (no ``bc_checkpoint``) still pick up actor+critic. ``strict=False`` so
        a critic added after the source checkpoint was written stays at its
        freshly-initialized values.
        """
        ckpt_dir = path
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"resume_from directory not found: {ckpt_dir}")

        model_path = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(ckpt_dir, "actor_critic.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Neither model.pt nor actor_critic.pt found under {ckpt_dir}."
            )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Flightmare PPO] Resumed model weights from {model_path}")
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt_state = torch.load(opt_path, map_location=self.device, weights_only=True)
                optimizer.load_state_dict(opt_state.get("optimizer", opt_state))
                print(f"[Flightmare PPO] Resumed optimizer state from {opt_path}")
            except Exception as exc:  # noqa: BLE001 — broad on purpose; downgrade to warn
                print(
                    f"[Flightmare PPO] WARNING: failed to load optimizer state from {opt_path}: "
                    f"{type(exc).__name__}: {exc}. Continuing with fresh optimizer."
                )
        else:
            print(
                "[Flightmare PPO] No optimizer.pt found alongside the resumed model "
                "(was save_optimizer_state disabled when this checkpoint was written?). "
                "Optimizer starts fresh — fine for low-LR PPO; momentum/variance estimates rebuild quickly."
            )

        ts_path = os.path.join(ckpt_dir, "training_state.json")
        if os.path.exists(ts_path):
            try:
                with open(ts_path) as f:
                    ts = json.load(f)
                resume_iteration = int(ts.get("iteration") or 0)
                if resume_iteration > 0:
                    self.start_iteration = resume_iteration + 1
                bs = ts.get("best_score")
                if bs is not None and np.isfinite(bs):
                    self.best_score = float(bs)
                self.best_iteration = int(ts.get("best_iteration") or 0)
                print(
                    f"[Flightmare PPO] Restored training state: "
                    f"next_iteration={self.start_iteration} "
                    f"best_score={self.best_score:.4f} best_iteration={self.best_iteration}"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[Flightmare PPO] WARNING: failed to parse {ts_path}: "
                    f"{type(exc).__name__}: {exc}. best-score starts at -inf."
                )
        else:
            match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(ckpt_dir)))
            if match:
                self.start_iteration = int(match.group(1)) + 1
                print(
                    f"[Flightmare PPO] Inferred next_iteration={self.start_iteration} "
                    f"from checkpoint directory name."
                )

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer,
        iteration: int,
        is_best: bool = False,
        name: str | None = None,
        metrics: dict[str, float | int] | None = None,
        best_score: float | None = None,
    ):
        if is_best:
            ckpt_dir = os.path.join(self.output_dir, "best")
        elif name is not None:
            ckpt_dir = os.path.join(self.output_dir, name)
        else:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{iteration}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "actor_critic.pt"))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        safe_best_score = (
            float(best_score)
            if best_score is not None and np.isfinite(best_score)
            else None
        )
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump(
                {
                    "iteration": iteration,
                    "is_best": is_best,
                    "selection_metric": self.selection_metric,
                    "best_score": safe_best_score,
                    "best_iteration": self.best_iteration,
                    "metrics": metrics or {},
                },
                f,
                indent=2,
            )
        if self.training_cfg.get("save_optimizer_state", False):
            torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, os.path.join(ckpt_dir, "optimizer.pt"))
        return ckpt_dir

    def _prune_checkpoints(self) -> None:
        if self.save_total_limit is None or self.save_total_limit <= 0:
            return
        checkpoints = []
        if not os.path.isdir(self.output_dir):
            return
        for name in os.listdir(self.output_dir):
            if not name.startswith("checkpoint-"):
                continue
            try:
                checkpoints.append((int(name.split("-", 1)[1]), name))
            except ValueError:
                continue
        checkpoints.sort()
        excess = len(checkpoints) - self.save_total_limit
        for _, name in checkpoints[:max(0, excess)]:
            shutil.rmtree(os.path.join(self.output_dir, name), ignore_errors=True)

"""PPO Trainer for fine-tuning BC-pretrained visuomotor policies.

Registered as trainer type "ppo". Loads a BC checkpoint, wraps it
as an actor-critic, and runs on-policy PPO in robosuite environments.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.core.registry import register
from src.robotics.collector import RolloutBuffer, collect_rollouts
from src.robotics.envs import (
    EnvSpec,
    allocate_envs_by_weights,
    make_robosuite_multitask_vec_env,
)
from src.robotics.loss import ppo_clip_loss, value_loss
from src.robotics.models.HistoryBuffer import HistoryBuffer
from src.robotics.normalization import NormStats


@register("trainer", "ppo")
class PPOTrainer:
    """On-policy PPO trainer for robosuite environments."""

    def __init__(
        self,
        model: nn.Module,
        training_cfg: dict,
        robotics_cfg: dict,
        wandb_cfg: dict,
        # Single-task interface (backward-compat)
        norm_stats: Optional[NormStats] = None,
        task_name: Optional[str] = None,
        task_id: int = 0,
        # Multi-task interface — all four may be provided together
        task_names: Optional[list] = None,
        task_ids: Optional[list] = None,
        task_weights: Optional[list] = None,
        norm_stats_by_task: Optional[Dict[str, NormStats]] = None,
        task_horizons: Optional[Dict[str, int]] = None,
        obs_keys_low_dim: list = None,
        obs_keys_image: list = None,
        camera_size: int = 160,
    ):
        self.model = model
        self.training_cfg = training_cfg
        self.robotics_cfg = robotics_cfg
        self.wandb_cfg = wandb_cfg
        self.obs_keys_low_dim = obs_keys_low_dim or []
        self.obs_keys_image = obs_keys_image or []
        self.camera_size = camera_size

        # --- Resolve task list (single-task → wrap in 1-element lists) ---
        if task_names is None:
            if task_name is None:
                raise ValueError("Must provide either task_name or task_names")
            self.task_names = [task_name]
            self.task_ids = [task_id]
            self.norm_stats_by_task = {task_name: norm_stats}
            self.task_weights = [1.0]
        else:
            self.task_names = list(task_names)
            self.task_ids = (
                list(task_ids) if task_ids is not None else list(range(len(self.task_names)))
            )
            if norm_stats_by_task is None:
                raise ValueError("norm_stats_by_task is required for multi-task PPO")
            self.norm_stats_by_task = norm_stats_by_task
            if task_weights is None:
                # Uniform
                self.task_weights = [1.0 / len(self.task_names)] * len(self.task_names)
            else:
                self.task_weights = list(task_weights)
            if len(self.task_weights) != len(self.task_names):
                raise ValueError(
                    f"task_weights length {len(self.task_weights)} != "
                    f"task_names length {len(self.task_names)}"
                )
            missing = [t for t in self.task_names if t not in self.norm_stats_by_task]
            if missing:
                raise ValueError(f"Missing norm_stats for tasks: {missing}")

        # PPO hyperparams
        ppo_cfg = robotics_cfg.get("ppo", {})
        self.n_envs = ppo_cfg.get("n_envs", 8)
        self.n_steps = ppo_cfg.get("n_steps", 256)
        self.ppo_epochs = ppo_cfg.get("ppo_epochs", 4)
        self.minibatch_size = ppo_cfg.get("minibatch_size", 64)
        self.clip_eps = ppo_cfg.get("clip_eps", 0.2)
        self.vf_coeff = ppo_cfg.get("vf_coeff", 0.5)
        self.ent_coeff = ppo_cfg.get("ent_coeff", 0.01)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.max_iterations = ppo_cfg.get("max_iterations", 500)
        self.horizon = ppo_cfg.get("horizon", 400)
        self.reward_shaping = ppo_cfg.get("reward_shaping", True)
        # Custom reward function name (looked up in src.robotics.rewards
        # registry inside RobosuiteGymEnv). None = use robosuite native.
        self.reward_fn_name = ppo_cfg.get("reward_fn")
        # Per-task horizon override (tool_hang needs longer episodes)
        self.task_horizons = dict(task_horizons) if task_horizons else {}

        # Training config
        self.actor_lr = ppo_cfg.get("actor_lr", 1e-5)
        self.critic_lr = ppo_cfg.get("critic_lr", 1e-4)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 0.5)
        self.use_bf16 = training_cfg.get("bf16", True)
        self.output_dir = training_cfg.get("output_dir", "outputs/ppo")
        self.logging_steps = training_cfg.get("logging_steps", 1)
        self.save_steps = training_cfg.get("save_steps", 50)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        """Main PPO training loop."""
        os.makedirs(self.output_dir, exist_ok=True)
        model = self.model.to(self.device)
        model.train()

        # Separate param groups: lower LR for pretrained actor, higher for fresh critic
        actor_params = list(model.backbone.parameters()) + list(model.actor_head.parameters()) + [model.log_std]
        critic_params = list(model.value_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": self.actor_lr, "weight_decay": 1e-4},
            {"params": critic_params, "lr": self.critic_lr, "weight_decay": 1e-4},
        ])

        # WandB
        run = None
        if self.wandb_cfg:
            try:
                import wandb
                run = wandb.init(
                    project=self.wandb_cfg.get("project", "robotics-ppo"),
                    entity=self.wandb_cfg.get("entity"),
                    name=self.wandb_cfg.get("run_name", "ppo"),
                    tags=self.wandb_cfg.get("tags", []),
                    config={
                        "n_envs": self.n_envs,
                        "n_steps": self.n_steps,
                        "ppo_epochs": self.ppo_epochs,
                        "clip_eps": self.clip_eps,
                        "actor_lr": self.actor_lr,
                        "critic_lr": self.critic_lr,
                        "gamma": self.gamma,
                        "gae_lambda": self.gae_lambda,
                    },
                )
            except Exception as e:
                print(f"[PPO] WandB init failed: {e}")

        # --- Mixture allocation ---
        # Largest-remainder allocates ``n_envs`` across tasks proportional to
        # ``task_weights``; harder-but-downweighted tasks still get ≥1 env.
        counts = allocate_envs_by_weights(self.n_envs, self.task_weights)
        env_specs: List[EnvSpec] = []
        task_id_per_env: List[int] = []
        task_name_per_env: List[str] = []
        for task_name, task_id, count in zip(self.task_names, self.task_ids, counts):
            task_horizon = self.task_horizons.get(task_name, self.horizon)
            for _ in range(count):
                env_specs.append(EnvSpec(
                    task_name=task_name, task_id=task_id,
                    norm_stats=self.norm_stats_by_task[task_name],
                    horizon=task_horizon,
                    reward_shaping=self.reward_shaping,
                    reward_fn=self.reward_fn_name,
                ))
                task_id_per_env.append(task_id)
                task_name_per_env.append(task_name)

        mixture_summary = {
            t: f"{c}/{self.n_envs} ({c / self.n_envs:.0%})"
            for t, c in zip(self.task_names, counts)
        }
        print(f"[PPO] Env mixture: {mixture_summary}")

        camera_names = [k.replace("_image", "") for k in self.obs_keys_image]
        vec_env = make_robosuite_multitask_vec_env(
            env_specs=env_specs,
            camera_names=camera_names,
            camera_size=self.camera_size,
            obs_keys_low_dim=self.obs_keys_low_dim,
            obs_keys_image=self.obs_keys_image,
        )

        # Create history buffers (one per env), each carrying ITS task's
        # task_id and norm_stats — so per-env normalization is correct and
        # the stacked model batch has the right task embeddings.
        arch_cfg = self.robotics_cfg.get("architecture", {})
        history_length = arch_cfg.get("history_length", 3)
        action_dim = arch_cfg.get("action_dim", 7)
        obs_dim = arch_cfg.get("obs_dim")

        history_buffers = [
            HistoryBuffer(
                history_length=history_length,
                action_dim=action_dim,
                task_id=spec.task_id,
                norm_stats=spec.norm_stats,
                target_state_dim=obs_dim,  # pad to unified arch input dim
            )
            for spec in env_specs
        ]

        # Determine image/state dims for rollout buffer
        image_specs = {
            cam: (self.camera_size, self.camera_size, 3)
            for cam in self.obs_keys_image
        }
        # In multi-task, tasks have different native state dims — arch_cfg.obs_dim
        # is the padded width emitted by every HistoryBuffer.
        if obs_dim is not None:
            state_dim = obs_dim
        else:
            first_ns = next(iter(self.norm_stats_by_task.values()))
            state_dim = len(first_ns.state_mean)

        rollout_buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            action_dim=action_dim,
            state_dim=state_dim,
            image_specs=image_specs,
            history_length=history_length,
            device=self.device,
        )

        # Initial env reset + push first obs to history buffers.
        # MultiTaskVecEnv returns per-env observation dicts (list).
        obs_list, _ = vec_env.reset()
        for i in range(self.n_envs):
            history_buffers[i].reset()
            o = obs_list[i]
            history_buffers[i].push(o["images"], o["state"])

        amp_ctx = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16) if self.use_bf16 else torch.amp.autocast(device_type=self.device.type, enabled=False)

        best_success_rate = -1.0
        t_start = time.time()

        print(f"\n[PPO] Starting training")
        print(f"  Envs: {self.n_envs} | Steps/rollout: {self.n_steps} | Epochs: {self.ppo_epochs}")
        print(f"  Minibatch: {self.minibatch_size} | Clip: {self.clip_eps}")
        print(f"  Actor LR: {self.actor_lr} | Critic LR: {self.critic_lr}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Device: {self.device}\n")

        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()

            # --- Collect rollouts ---
            model.eval()
            rollout_stats = collect_rollouts(
                vec_env, model, history_buffers, rollout_buffer,
                self.n_steps, self.device,
                task_id_per_env=task_id_per_env,
                task_name_per_env=task_name_per_env,
            )

            # --- Compute GAE ---
            rollout_buffer.compute_advantages(
                rollout_stats["last_values"], self.gamma, self.gae_lambda
            )

            # --- PPO update ---
            model.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_clip_fraction = 0.0
            n_updates = 0

            for mb in rollout_buffer.get_minibatches(self.minibatch_size, self.ppo_epochs):
                # Normalize advantages
                adv = mb["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                with amp_ctx:
                    log_prob, entropy, value = model.evaluate_actions(mb["batch"], mb["actions"])

                    p_loss = ppo_clip_loss(log_prob, mb["old_log_probs"], adv, self.clip_eps)
                    v_loss = value_loss(value, mb["returns"], mb["old_values"], self.clip_eps)
                    ent = entropy.mean()

                    loss = p_loss + self.vf_coeff * v_loss - self.ent_coeff * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                # Track stats
                with torch.no_grad():
                    ratio = (log_prob - mb["old_log_probs"]).exp()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                total_policy_loss += p_loss.item()
                total_value_loss += v_loss.item()
                total_entropy += ent.item()
                total_clip_fraction += clip_frac.item()
                n_updates += 1

            # --- Logging ---
            if iteration % self.logging_steps == 0:
                elapsed = time.time() - t_start
                fps = (iteration * self.n_steps * self.n_envs) / elapsed
                iter_time = time.time() - iter_start

                log_dict = {
                    "iteration": iteration,
                    "policy_loss": total_policy_loss / n_updates,
                    "value_loss": total_value_loss / n_updates,
                    "entropy": total_entropy / n_updates,
                    "clip_fraction": total_clip_fraction / n_updates,
                    "mean_reward": rollout_stats["mean_reward"],
                    "success_rate": rollout_stats["success_rate"],
                    "mean_ep_length": rollout_stats["mean_length"],
                    "n_episodes": rollout_stats["n_episodes"],
                    "log_std_mean": model.log_std.data.mean().item(),
                    "fps": fps,
                    "iter_time": iter_time,
                }
                # Per-task breakdown — essential for spotting catastrophic
                # forgetting on downweighted tasks.
                for task, stats in rollout_stats.get("per_task", {}).items():
                    log_dict[f"per_task/{task}/success_rate"] = stats["success_rate"]
                    log_dict[f"per_task/{task}/mean_reward"] = stats["mean_reward"]
                    log_dict[f"per_task/{task}/n_episodes"] = stats["n_episodes"]

                per_task_str = " ".join(
                    f"{t}:{s['success_rate']:.0%}({s['n_episodes']})"
                    for t, s in rollout_stats.get("per_task", {}).items()
                )
                print(
                    f"[{iteration}/{self.max_iterations}] "
                    f"r={rollout_stats['mean_reward']:.2f} "
                    f"sr={rollout_stats['success_rate']:.1%} "
                    f"pl={total_policy_loss/n_updates:.4f} "
                    f"vl={total_value_loss/n_updates:.4f} "
                    f"ent={total_entropy/n_updates:.3f} "
                    f"clip={total_clip_fraction/n_updates:.3f} "
                    f"fps={fps:.0f}"
                    + (f" | {per_task_str}" if per_task_str else "")
                )

                if run is not None:
                    run.log(log_dict, step=iteration)

            # --- Checkpointing ---
            if iteration % self.save_steps == 0:
                self._save_checkpoint(model, optimizer, iteration)

                # Track best by success rate
                sr = rollout_stats["success_rate"]
                if sr > best_success_rate:
                    best_success_rate = sr
                    self._save_checkpoint(model, optimizer, iteration, is_best=True)
                    print(f"  [best] New best success rate: {sr:.1%}")

        vec_env.close()
        self._save_checkpoint(model, optimizer, self.max_iterations)
        print(f"\n[PPO] Training complete. Best success rate: {best_success_rate:.1%}")

        if run is not None:
            run.finish()

    def _save_checkpoint(self, model, optimizer, iteration, is_best=False):
        """Save model checkpoint in both PPO and BC-compatible formats."""
        if is_best:
            ckpt_dir = os.path.join(self.output_dir, "best")
        else:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{iteration}")

        os.makedirs(ckpt_dir, exist_ok=True)

        # Full actor-critic state
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "actor_critic.pt"))

        # BC-compatible actor-only state (for eval_robomimic.py)
        bc_state = {}
        for k, v in model.state_dict().items():
            if not k.startswith("value_head.") and k != "log_std":
                bc_state[k.replace("actor_head.", "controller.")] = v
        torch.save(bc_state, os.path.join(ckpt_dir, "model.pt"))

        # Optimizer + iteration
        torch.save({
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        }, os.path.join(ckpt_dir, "training_state.pt"))

        # Norm stats — save one file per task so downstream eval / re-init
        # can pick the right stats regardless of which task is being rolled out.
        for task_name, ns in self.norm_stats_by_task.items():
            ns.save(os.path.join(ckpt_dir, f"norm_stats_{task_name}.json"))

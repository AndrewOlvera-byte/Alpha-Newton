"""PPO Trainer for fine-tuning BC-pretrained visuomotor policies.

Registered as trainer type "ppo". Loads a BC checkpoint, wraps it
as an actor-critic, and runs on-policy PPO in robosuite environments.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.core.registry import register
from src.robotics.collector import RolloutBuffer, collect_rollouts
from src.robotics.envs import make_robosuite_vec_env
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
        norm_stats: NormStats,
        task_name: str,
        task_id: int = 0,
        obs_keys_low_dim: list = None,
        obs_keys_image: list = None,
        camera_size: int = 160,
    ):
        self.model = model
        self.training_cfg = training_cfg
        self.robotics_cfg = robotics_cfg
        self.wandb_cfg = wandb_cfg
        self.norm_stats = norm_stats
        self.task_name = task_name
        self.task_id = task_id
        self.obs_keys_low_dim = obs_keys_low_dim or []
        self.obs_keys_image = obs_keys_image or []
        self.camera_size = camera_size

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

        # Create vectorized environment
        camera_names = [k.replace("_image", "") for k in self.obs_keys_image]
        vec_env = make_robosuite_vec_env(
            task_name=self.task_name,
            n_envs=self.n_envs,
            norm_stats=self.norm_stats,
            camera_names=camera_names,
            camera_size=self.camera_size,
            obs_keys_low_dim=self.obs_keys_low_dim,
            obs_keys_image=self.obs_keys_image,
            horizon=self.horizon,
            reward_shaping=self.reward_shaping,
        )

        # Create history buffers (one per env)
        arch_cfg = self.robotics_cfg.get("architecture", {})
        history_length = arch_cfg.get("history_length", 3)
        action_dim = arch_cfg.get("action_dim", 7)
        obs_dim = arch_cfg.get("obs_dim")

        history_buffers = [
            HistoryBuffer(
                history_length=history_length,
                action_dim=action_dim,
                task_id=self.task_id,
                norm_stats=self.norm_stats,
                target_state_dim=obs_dim,
            )
            for _ in range(self.n_envs)
        ]

        # Determine image/state dims for rollout buffer
        image_specs = {
            cam: (self.camera_size, self.camera_size, 3)
            for cam in self.obs_keys_image
        }
        state_dim = obs_dim or len(self.norm_stats.state_mean)

        rollout_buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            action_dim=action_dim,
            state_dim=state_dim,
            image_specs=image_specs,
            history_length=history_length,
            device=self.device,
        )

        # Initial env reset + push first obs to history buffers
        obs_list, _ = vec_env.reset()
        for i in range(self.n_envs):
            history_buffers[i].reset()
            state_i = obs_list["state"][i]
            images_i = {cam: obs_list["images"][cam][i] for cam in obs_list["images"]}
            history_buffers[i].push(images_i, state_i)

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

                print(
                    f"[{iteration}/{self.max_iterations}] "
                    f"r={rollout_stats['mean_reward']:.2f} "
                    f"sr={rollout_stats['success_rate']:.1%} "
                    f"pl={total_policy_loss/n_updates:.4f} "
                    f"vl={total_value_loss/n_updates:.4f} "
                    f"ent={total_entropy/n_updates:.3f} "
                    f"clip={total_clip_fraction/n_updates:.3f} "
                    f"fps={fps:.0f}"
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

        # Norm stats
        self.norm_stats.save(os.path.join(ckpt_dir, f"norm_stats_{self.task_name}.json"))

"""Robosuite PPO trainer for BC-pretrained visuomotor policies."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

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


@register("trainer", "ppo_robosuite")
class PPOTrainer:
    """On-policy PPO trainer backed by robosuite vector environments."""

    def __init__(
        self,
        model: nn.Module,
        training_cfg: dict,
        robotics_cfg: dict,
        wandb_cfg: dict,

        norm_stats: Optional[NormStats] = None,
        task_name: Optional[str] = None,
        task_id: int = 0,

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

        self.reward_fn_name = ppo_cfg.get("reward_fn")

        self.task_horizons = dict(task_horizons) if task_horizons else {}

        self.actor_lr = ppo_cfg.get("actor_lr", 1e-5)
        self.critic_lr = ppo_cfg.get("critic_lr", 1e-4)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 0.5)
        self.use_bf16 = training_cfg.get("bf16", True)
        self.compile = ppo_cfg.get("compile", False)
        self.profile_rollouts = bool(ppo_cfg.get("profile_rollouts", False))
        self.show_progress = bool(ppo_cfg.get("show_progress", False))

        self.async_rollout = ppo_cfg.get("async_rollout", False)
        self.output_dir = training_cfg.get("output_dir", "outputs/ppo")
        self.logging_steps = training_cfg.get("logging_steps", 1)
        self.save_steps = int(training_cfg.get("save_steps", 50) or 0)
        self.save_final = bool(training_cfg.get("save_final", True))

        self.save_optimizer_state = bool(training_cfg.get("save_optimizer_state", False))
        self.save_bc_compatible_every_checkpoint = bool(
            training_cfg.get("save_bc_compatible_every_checkpoint", False)
        )
        self.save_total_limit = training_cfg.get("save_total_limit")
        if self.save_total_limit is not None:
            self.save_total_limit = int(self.save_total_limit)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

    def train(self):
        """Main PPO training loop."""
        os.makedirs(self.output_dir, exist_ok=True)

        model = self.model
        model.train()

        raw_model = model

        actor_params = [
            p for p in (
                list(model.backbone.parameters())
                + list(model.actor_head.parameters())
                + [model.log_std]
            )
            if p.requires_grad
        ]
        critic_params = [p for p in model.value_head.parameters() if p.requires_grad]
        trainable_params = actor_params + critic_params
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": self.actor_lr, "weight_decay": 1e-4},
            {"params": critic_params, "lr": self.critic_lr, "weight_decay": 1e-4},
        ])

        run = None
        if self.wandb_cfg:
            try:
                import wandb
                project = self.wandb_cfg.get("project", "robotics-ppo")
                mode = self.wandb_cfg.get("mode") or os.environ.get("WANDB_MODE", "").strip()
                api_key = os.environ.get("WANDB_API_KEY", "").strip()
                if mode == "disabled":
                    print("[PPO] WandB disabled by config/env.")
                elif not api_key and mode not in {"offline", "dryrun"}:
                    print("[PPO] No WANDB_API_KEY; skipping WandB to avoid network/disk stalls.")
                else:
                    init_kwargs = {
                        "project": project,
                        "entity": self.wandb_cfg.get("entity"),
                        "name": self.wandb_cfg.get("run_name", "ppo"),
                        "tags": self.wandb_cfg.get("tags", []),
                        "config": {
                            "n_envs": self.n_envs,
                            "n_steps": self.n_steps,
                            "ppo_epochs": self.ppo_epochs,
                            "minibatch_size": self.minibatch_size,
                            "clip_eps": self.clip_eps,
                            "vf_coeff": self.vf_coeff,
                            "ent_coeff": self.ent_coeff,
                            "actor_lr": self.actor_lr,
                            "critic_lr": self.critic_lr,
                            "max_grad_norm": self.max_grad_norm,
                            "gamma": self.gamma,
                            "gae_lambda": self.gae_lambda,
                            "horizon": self.horizon,
                            "task_horizons": self.task_horizons,
                            "task_names": self.task_names,
                            "task_weights": self.task_weights,
                            "reward_fn": self.reward_fn_name,
                            "bf16": self.use_bf16,
                            "compile": self.compile,
                            "async_rollout": self.async_rollout,
                            "gradient_checkpointing": self.robotics_cfg.get(
                                "architecture", {}
                            ).get("gradient_checkpointing", False),
                        },
                    }
                    if mode:
                        init_kwargs["mode"] = mode
                    run = wandb.init(**init_kwargs)
            except Exception as e:
                print(f"[PPO] WandB init failed: {e}")

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

        model = model.to(self.device)
        model.train()
        raw_model = model

        if self.compile and self.device.type == "cuda":
            print("[PPO] torch.compile(model, dynamic=True, mode='reduce-overhead')")
            try:
                model = torch.compile(model, dynamic=True, mode="reduce-overhead")
            except Exception as e:
                print(f"[PPO] compile failed, falling back to eager: {e}")

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
                target_state_dim=obs_dim,
            )
            for spec in env_specs
        ]

        image_specs = {
            cam: (self.camera_size, self.camera_size, 3)
            for cam in self.obs_keys_image
        }

        if obs_dim is not None:
            state_dim = obs_dim
        else:
            first_ns = next(iter(self.norm_stats_by_task.values()))
            state_dim = len(first_ns.state_mean)

        def _make_buffer():
            return RolloutBuffer(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                action_dim=action_dim,
                state_dim=state_dim,
                image_specs=image_specs,
                history_length=history_length,
                device=self.device,
            )

        rollout_buffer = _make_buffer()

        rollout_buffer_next = _make_buffer() if self.async_rollout else None
        if self.async_rollout:
            rollout_model = copy.deepcopy(raw_model).to(self.device).eval()
            for p in rollout_model.parameters():
                p.requires_grad_(False)
            async_pool = ThreadPoolExecutor(max_workers=1)
            print("[PPO] async_rollout=True - double-buffered collect/update")
        else:
            rollout_model = None
            async_pool = None
        pending_future: Optional[Future] = None

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

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            if self.async_rollout and pending_future is not None:
                rollout_stats = pending_future.result()
                rollout_buffer, rollout_buffer_next = rollout_buffer_next, rollout_buffer
                pending_future = None
            else:
                model.eval()
                rollout_stats = collect_rollouts(
                    vec_env, model, history_buffers, rollout_buffer,
                    self.n_steps, self.device,
                    task_id_per_env=task_id_per_env,
                    task_name_per_env=task_name_per_env,
                    profile=self.profile_rollouts,
                    show_progress=self.show_progress,
                )

            if self.async_rollout and iteration < self.max_iterations:
                rollout_model.load_state_dict(raw_model.state_dict())
                rollout_model.eval()
                pending_future = async_pool.submit(
                    collect_rollouts,
                    vec_env, rollout_model, history_buffers,
                    rollout_buffer_next, self.n_steps, self.device,
                    task_id_per_env, task_name_per_env,
                    self.profile_rollouts, self.show_progress,
                )

            rollout_buffer.compute_advantages(
                rollout_stats["last_values"], self.gamma, self.gae_lambda
            )
            rollout_buffer.normalize_advantages_per_task()

            model.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_clip_fraction = 0.0
            total_approx_kl = 0.0
            total_grad_norm = 0.0
            total_ratio_mean = 0.0
            total_ratio_std = 0.0
            n_updates = 0

            update_t0 = time.time()
            total_mb = self.ppo_epochs * ((self.n_steps * self.n_envs + self.minibatch_size - 1) // self.minibatch_size)
            mb_iter = rollout_buffer.get_minibatches(self.minibatch_size, self.ppo_epochs)
            mb_pbar = None
            if self.show_progress:
                mb_pbar = tqdm(
                    mb_iter,
                    total=total_mb, desc="update", leave=False,
                    dynamic_ncols=True, mininterval=0.5,
                )
                mb_iter = mb_pbar
            for mb in mb_iter:
                adv = mb["advantages"]

                with amp_ctx:
                    log_prob, entropy, value = model.evaluate_actions(mb["batch"], mb["actions"])

                    p_loss = ppo_clip_loss(log_prob, mb["old_log_probs"], adv, self.clip_eps)
                    v_loss = value_loss(value, mb["returns"], mb["old_values"], self.clip_eps)
                    ent = entropy.mean()

                    loss = p_loss + self.vf_coeff * v_loss - self.ent_coeff * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - mb["old_log_probs"]
                    ratio = log_ratio.exp()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                total_policy_loss += p_loss.item()
                total_value_loss += v_loss.item()
                total_entropy += ent.item()
                total_clip_fraction += clip_frac.item()
                total_approx_kl += approx_kl.item()
                total_grad_norm += float(grad_norm)
                total_ratio_mean += ratio.mean().item()
                total_ratio_std += ratio.std().item()
                n_updates += 1

            if mb_pbar is not None:
                mb_pbar.close()
            update_time = time.time() - update_t0

            if iteration % self.logging_steps == 0:
                elapsed = time.time() - t_start
                fps = (iteration * self.n_steps * self.n_envs) / elapsed
                iter_time = time.time() - iter_start

                with torch.no_grad():
                    returns_flat = rollout_buffer.returns.flatten()
                    values_flat = rollout_buffer.values.flatten()
                    var_returns = returns_flat.var()
                    explained_var = (
                        1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)
                    ).item() if var_returns > 0 else 0.0

                    adv_flat = rollout_buffer.advantages.flatten()
                    actions_flat = rollout_buffer.actions.view(-1, rollout_buffer.actions.shape[-1])
                    rewards_flat = rollout_buffer.rewards.flatten()
                    log_std = raw_model.log_std.detach()

                if self.device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                else:
                    mem_alloc = mem_reserved = mem_peak = 0.0

                log_dict = {

                    "iteration": iteration,
                    "ppo/policy_loss": total_policy_loss / n_updates,
                    "ppo/value_loss": total_value_loss / n_updates,
                    "ppo/entropy": total_entropy / n_updates,
                    "ppo/clip_fraction": total_clip_fraction / n_updates,
                    "ppo/approx_kl": total_approx_kl / n_updates,
                    "ppo/grad_norm": total_grad_norm / n_updates,
                    "ppo/ratio_mean": total_ratio_mean / n_updates,
                    "ppo/ratio_std": total_ratio_std / n_updates,
                    "ppo/explained_variance": explained_var,

                    "optim/actor_lr": optimizer.param_groups[0]["lr"],
                    "optim/critic_lr": optimizer.param_groups[1]["lr"],

                    "rollout/mean_reward": rollout_stats["mean_reward"],
                    "rollout/success_rate": rollout_stats["success_rate"],
                    "rollout/mean_ep_length": rollout_stats["mean_length"],
                    "rollout/ep_length_p50": rollout_stats.get("length_p50", 0.0),
                    "rollout/ep_length_p95": rollout_stats.get("length_p95", 0.0),
                    "rollout/ep_length_min": rollout_stats.get("length_min", 0),
                    "rollout/ep_length_max": rollout_stats.get("length_max", 0),
                    "rollout/ep_reward_std": rollout_stats.get("reward_std", 0.0),
                    "rollout/ep_reward_min": rollout_stats.get("reward_min", 0.0),
                    "rollout/ep_reward_max": rollout_stats.get("reward_max", 0.0),
                    "rollout/n_episodes": rollout_stats["n_episodes"],

                    "rollout/reward_step_mean": rewards_flat.mean().item(),
                    "rollout/reward_step_std": rewards_flat.std().item(),
                    "rollout/reward_step_min": rewards_flat.min().item(),
                    "rollout/reward_step_max": rewards_flat.max().item(),

                    "adv/mean": adv_flat.mean().item(),
                    "adv/std": adv_flat.std().item(),
                    "adv/min": adv_flat.min().item(),
                    "adv/max": adv_flat.max().item(),

                    "value/mean": values_flat.mean().item(),
                    "value/std": values_flat.std().item(),
                    "return/mean": returns_flat.mean().item(),
                    "return/std": returns_flat.std().item(),

                    "policy/log_std_mean": log_std.mean().item(),
                    "policy/log_std_max": log_std.max().item(),
                    "policy/log_std_min": log_std.min().item(),
                    "policy/action_mean_abs": actions_flat.abs().mean().item(),
                    "policy/action_std": actions_flat.std().item(),
                    "policy/action_max_abs": actions_flat.abs().max().item(),

                    "time/fps": fps,
                    "time/iter_time": iter_time,
                    "time/rollout_time": rollout_stats.get("rollout_time", 0.0),
                    "time/update_time": update_time,
                    "time/rollout_fps": rollout_stats.get("rollout_fps", 0.0),
                    "gpu/mem_alloc_mib": mem_alloc,
                    "gpu/mem_reserved_mib": mem_reserved,
                    "gpu/mem_peak_mib": mem_peak,
                }

                for d in range(log_std.numel()):
                    log_dict[f"policy/log_std_d{d}"] = log_std[d].item()
                    log_dict[f"policy/action_mean_d{d}"] = actions_flat[:, d].mean().item()
                    log_dict[f"policy/action_std_d{d}"] = actions_flat[:, d].std().item()

                for task, stats in rollout_stats.get("per_task", {}).items():
                    log_dict[f"per_task/{task}/success_rate"] = stats["success_rate"]
                    log_dict[f"per_task/{task}/mean_reward"] = stats["mean_reward"]
                    log_dict[f"per_task/{task}/mean_length"] = stats["mean_length"]
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
                    f"kl={total_approx_kl/n_updates:.4f} "
                    f"clip={total_clip_fraction/n_updates:.3f} "
                    f"gn={total_grad_norm/n_updates:.2f} "
                    f"ev={explained_var:+.2f} "
                    f"mem={mem_peak/1024:.1f}G "
                    f"fps={fps:.0f} "
                    f"[roll={rollout_stats.get('rollout_time', 0):.1f}s "
                    f"upd={update_time:.1f}s "
                    f"rfps={rollout_stats.get('rollout_fps', 0):.0f}]"
                    + (f" | {per_task_str}" if per_task_str else "")
                )

                if run is not None:
                    run.log(log_dict, step=iteration)

            if self.save_steps > 0 and iteration % self.save_steps == 0:
                self._save_checkpoint(raw_model, optimizer, iteration)
                self._prune_checkpoints()

                sr = rollout_stats["success_rate"]
                if sr > best_success_rate:
                    best_success_rate = sr
                    self._save_checkpoint(raw_model, optimizer, iteration, is_best=True)
                    print(f"  [best] New best success rate: {sr:.1%}")

        if pending_future is not None:
            try:
                pending_future.result(timeout=60)
            except Exception as e:
                print(f"[PPO] pending async rollout errored on shutdown: {e}")
        if async_pool is not None:
            async_pool.shutdown(wait=False)
        vec_env.close()
        if self.save_final:
            self._save_checkpoint(raw_model, optimizer, self.max_iterations, name="final")
        print(f"\n[PPO] Training complete. Best success rate: {best_success_rate:.1%}")

        if run is not None:
            run.finish()

    def _save_checkpoint(self, model, optimizer, iteration, is_best=False, name: str = None):
        """Save model checkpoint in both PPO and BC-compatible formats."""
        if is_best:
            ckpt_dir = os.path.join(self.output_dir, "best")
        elif name is not None:
            ckpt_dir = os.path.join(self.output_dir, name)
        else:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{iteration}")

        os.makedirs(ckpt_dir, exist_ok=True)

        model_state = model.state_dict()
        torch.save(model_state, os.path.join(ckpt_dir, "actor_critic.pt"))

        include_bc = (
            is_best
            or name == "final"
            or self.save_bc_compatible_every_checkpoint
        )
        if include_bc:
            bc_state = {}
            for k, v in model_state.items():
                if not k.startswith("value_head.") and k != "log_std":
                    bc_state[k.replace("actor_head.", "controller.")] = v
            torch.save(bc_state, os.path.join(ckpt_dir, "model.pt"))

        metadata = {"iteration": iteration, "optimizer_state_saved": self.save_optimizer_state}
        if self.save_optimizer_state:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            }, os.path.join(ckpt_dir, "training_state.pt"))
        else:
            with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        for task_name, ns in self.norm_stats_by_task.items():
            ns.save(os.path.join(ckpt_dir, f"norm_stats_{task_name}.json"))

        return ckpt_dir

    def _prune_checkpoints(self):
        """Keep only the newest N regular checkpoints when configured."""
        if self.save_total_limit is None or self.save_total_limit <= 0:
            return
        if not os.path.isdir(self.output_dir):
            return

        checkpoints = []
        for name in os.listdir(self.output_dir):
            if not name.startswith("checkpoint-"):
                continue
            try:
                iteration = int(name.split("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((iteration, name))

        checkpoints.sort()
        excess = len(checkpoints) - self.save_total_limit
        if excess <= 0:
            return
        for _, name in checkpoints[:excess]:
            shutil.rmtree(os.path.join(self.output_dir, name), ignore_errors=True)

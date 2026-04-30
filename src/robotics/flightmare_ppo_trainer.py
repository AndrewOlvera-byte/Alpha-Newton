"""PPO trainer for state-only Flightmare racing policies."""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.core.registry import register
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


def collect_flightmare_rollouts(
    vec_env,
    model: nn.Module,
    obs_list: list[dict],
    prev_actions: np.ndarray,
    buffer: FlightmareRolloutBuffer,
    n_steps: int,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[list[dict], np.ndarray, dict]:
    n_envs = len(obs_list)
    current_rewards = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    successes: list[bool] = []
    gate_completion: list[float] = []
    gate_misses = 0
    crash_count = 0
    max_speed = 0.0

    t0 = time.time()
    amp = torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16 and device.type == "cuda",
    )

    for step in range(n_steps):
        batch = _stack_obs(obs_list, prev_actions, device)
        with torch.no_grad(), amp:
            actions, log_probs, values = model.act(batch)
        actions_np = actions.float().cpu().numpy()

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions_np)
        dones = np.logical_or(terms, truncs)
        buffer.insert(step, batch, actions, log_probs, values, rewards, dones)

        current_rewards += rewards
        current_lengths += 1

        done_indices = np.flatnonzero(dones).tolist()
        reset_obs_by_env = {}
        if done_indices:
            reset_obs, _ = vec_env.reset_at(done_indices)
            reset_obs_by_env = dict(zip(done_indices, reset_obs))

        for i, info in enumerate(infos):
            max_speed = max(max_speed, float(info.get("speed_mps", 0.0)))
            if info.get("gate_missed", False):
                gate_misses += 1
            if info.get("crash", False):
                crash_count += 1

            if dones[i]:
                ep_rewards.append(float(current_rewards[i]))
                ep_lengths.append(int(current_lengths[i]))
                successes.append(bool(info.get("success", False)))
                gate_completion.append(float(info.get("gate_completion", 0.0)))
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                next_obs[i] = reset_obs_by_env[i]
                prev_actions[i] = 0.0
            else:
                prev_actions[i] = actions_np[i]

        obs_list = next_obs

    batch = _stack_obs(obs_list, prev_actions, device)
    with torch.no_grad(), amp:
        _, _, last_values = model.act(batch)
    buffer.compute_advantages(last_values, gamma=1.0, gae_lambda=1.0)  # overwritten by trainer

    elapsed = time.time() - t0
    rewards_arr = np.asarray(ep_rewards, dtype=np.float32) if ep_rewards else np.zeros(1, dtype=np.float32)
    lengths_arr = np.asarray(ep_lengths, dtype=np.float32) if ep_lengths else np.zeros(1, dtype=np.float32)
    stats = {
        "last_values": last_values,
        "rollout_time": elapsed,
        "rollout_fps": float(n_steps * n_envs / max(elapsed, 1e-6)),
        "mean_reward": float(rewards_arr.mean()) if ep_rewards else 0.0,
        "reward_std": float(rewards_arr.std()) if ep_rewards else 0.0,
        "reward_min": float(rewards_arr.min()) if ep_rewards else 0.0,
        "reward_max": float(rewards_arr.max()) if ep_rewards else 0.0,
        "mean_length": float(lengths_arr.mean()) if ep_lengths else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_gate_completion": float(np.mean(gate_completion)) if gate_completion else 0.0,
        "n_episodes": len(ep_rewards),
        "gate_misses": int(gate_misses),
        "crashes": int(crash_count),
        "max_speed_mps": float(max_speed),
    }
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
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 0.5))
        self.use_bf16 = bool(training_cfg.get("bf16", True))
        self.output_dir = training_cfg.get("output_dir", "outputs/flightmare_ppo")
        self.logging_steps = int(training_cfg.get("logging_steps", 1))
        self.save_steps = int(training_cfg.get("save_steps", 25) or 0)
        self.save_total_limit = training_cfg.get("save_total_limit")
        self.save_total_limit = int(self.save_total_limit) if self.save_total_limit else None
        self.save_final = bool(training_cfg.get("save_final", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    def _env_kwargs(self) -> dict:
        ppo = self.ppo_cfg
        return {
            "data_dir": self.data_cfg.get("data_dir", "data/flightmare/bc_v1"),
            "action_type": self.data_cfg.get("action_type", ppo.get("action_type", "ctbr")),
            "include_mission": self.data_cfg.get("include_mission", True),
            "normalize_state": self.data_cfg.get("normalize_state", True),
            "normalize_mission": self.data_cfg.get("normalize_mission", True),
            "normalize_action": self.data_cfg.get("normalize_action", True),
            "control_hz": ppo.get("control_hz", 100.0),
            "horizon": ppo.get("horizon", 1500),
            "scene": ppo.get("scene", "industrial"),
            "render": ppo.get("render", False),
            "backend": ppo.get("backend", "auto"),
            "num_gates": ppo.get("num_gates", 8),
            "gate_spacing_range": ppo.get("gate_spacing_range", [4.0, 9.0]),
            "gate_lateral_jitter": ppo.get("gate_lateral_jitter", 2.0),
            "gate_z_range": ppo.get("gate_z_range", [1.5, 4.0]),
            "gate_yaw_step": ppo.get("gate_yaw_step", 0.7),
            "gate_yaw_noise": ppo.get("gate_yaw_noise", 0.25),
            "gate_size": ppo.get("gate_size", 1.0),
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
        }

    def _split_params(self):
        critic_params = []
        if getattr(self.model, "head", None) is not None and getattr(self.model.head, "critic", None) is not None:
            critic_params = [p for p in self.model.head.critic.parameters() if p.requires_grad]
        critic_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in critic_ids]
        return actor_params, critic_params

    def smoke_test(self) -> None:
        vec_env = make_flightmare_vec_env(n_envs=min(2, self.n_envs), seed=0, **self._env_kwargs())
        try:
            obs_list, infos = vec_env.reset(seed=0)
            print(f"[Flightmare PPO Test] envs={len(obs_list)} fallback={infos[0].get('using_fallback')}")
            print(f"[Flightmare PPO Test] state shape={obs_list[0]['state'].shape}")
            model = self.model.to(self.device).eval()
            prev_actions = np.zeros((len(obs_list), int(self.model.action_dim)), dtype=np.float32)
            buffer = FlightmareRolloutBuffer(
                n_steps=8,
                n_envs=len(obs_list),
                state_dim=int(self.robotics_cfg.get("architecture", {}).get("state_dim", obs_list[0]["state"].shape[0])),
                action_dim=int(self.model.action_dim),
                device=self.device,
            )
            obs_list, prev_actions, stats = collect_flightmare_rollouts(
                vec_env, model, obs_list, prev_actions, buffer, 8, self.device, self.use_bf16
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
        model = self.model.to(self.device)
        model.train()

        actor_params, critic_params = self._split_params()
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": self.actor_lr, "weight_decay": 1e-4},
            {"params": critic_params, "lr": self.critic_lr, "weight_decay": 1e-4},
        ])

        run = None
        if self.wandb_cfg:
            try:
                import wandb
                mode = self.wandb_cfg.get("mode") or os.environ.get("WANDB_MODE", "").strip()
                api_key = os.environ.get("WANDB_API_KEY", "").strip()
                if mode == "disabled" or (not api_key and mode not in {"offline", "dryrun"}):
                    print("[Flightmare PPO] WandB disabled/skipped.")
                else:
                    init_kwargs = {
                        "project": self.wandb_cfg.get("project", "flightmare-ppo"),
                        "entity": self.wandb_cfg.get("entity"),
                        "name": self.wandb_cfg.get("run_name", "flightmare_ppo"),
                        "tags": self.wandb_cfg.get("tags", []),
                        "config": {"ppo": self.ppo_cfg, "training": self.training_cfg},
                    }
                    if mode:
                        init_kwargs["mode"] = mode
                    run = wandb.init(**init_kwargs)
            except Exception as e:
                print(f"[Flightmare PPO] WandB init failed: {e}")

        env_kwargs = self._env_kwargs()
        vec_env = make_flightmare_vec_env(n_envs=self.n_envs, seed=int(self.ppo_cfg.get("seed", 0)), **env_kwargs)
        obs_list, infos = vec_env.reset(seed=int(self.ppo_cfg.get("seed", 0)))
        if infos and infos[0].get("using_fallback"):
            print("[Flightmare PPO] WARNING: using numpy fallback; metrics are not authoritative.")

        state_dim = int(self.robotics_cfg.get("architecture", {}).get("state_dim", obs_list[0]["state"].shape[0]))
        action_dim = int(self.model.action_dim)
        prev_actions = np.zeros((self.n_envs, action_dim), dtype=np.float32)
        buffer = FlightmareRolloutBuffer(self.n_steps, self.n_envs, state_dim, action_dim, self.device)

        best_score = -1.0
        t_start = time.time()
        print("\n[Flightmare PPO] Starting training")
        print(f"  Envs: {self.n_envs} | Steps/rollout: {self.n_steps} | Epochs: {self.ppo_epochs}")
        print(f"  Action: {env_kwargs['action_type']} | Reward: {env_kwargs['reward_fn']}")
        print(f"  Actor LR: {self.actor_lr} | Critic LR: {self.critic_lr}")
        print(f"  Max iterations: {self.max_iterations} | Device: {self.device}\n")

        amp = torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16 and self.device.type == "cuda",
        )

        try:
            for iteration in range(1, self.max_iterations + 1):
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
                )
                buffer.compute_advantages(stats["last_values"], self.gamma, self.gae_lambda)
                buffer.normalize_advantages()

                model.train()
                pol_loss_sum = val_loss_sum = ent_sum = kl_sum = clip_sum = grad_sum = 0.0
                n_updates = 0
                update_t0 = time.time()
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
                update_time = time.time() - update_t0

                if iteration % self.logging_steps == 0:
                    elapsed = time.time() - t_start
                    fps = iteration * self.n_steps * self.n_envs / max(elapsed, 1e-6)
                    rewards_flat = buffer.rewards.flatten()
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
                        "rollout/mean_reward": stats["mean_reward"],
                        "rollout/success_rate": stats["success_rate"],
                        "rollout/mean_gate_completion": stats["mean_gate_completion"],
                        "rollout/n_episodes": stats["n_episodes"],
                        "rollout/gate_misses": stats["gate_misses"],
                        "rollout/crashes": stats["crashes"],
                        "rollout/max_speed_mps": stats["max_speed_mps"],
                        "rollout/reward_step_mean": rewards_flat.mean().item(),
                        "rollout/reward_step_std": rewards_flat.std().item(),
                        "time/fps": fps,
                        "time/rollout_fps": stats["rollout_fps"],
                        "time/rollout_time": stats["rollout_time"],
                        "time/update_time": update_time,
                    }
                    print(
                        f"[{iteration}/{self.max_iterations}] "
                        f"r={stats['mean_reward']:.2f} "
                        f"sr={stats['success_rate']:.1%} "
                        f"gc={stats['mean_gate_completion']:.1%} "
                        f"miss={stats['gate_misses']} crash={stats['crashes']} "
                        f"pl={pol_loss_sum/max(1,n_updates):.4f} "
                        f"vl={val_loss_sum/max(1,n_updates):.4f} "
                        f"ent={ent_sum/max(1,n_updates):.3f} "
                        f"kl={kl_sum/max(1,n_updates):.4f} "
                        f"ev={explained_var:+.2f} "
                        f"fps={fps:.0f} rfps={stats['rollout_fps']:.0f} "
                        f"[roll={stats['rollout_time']:.1f}s upd={update_time:.1f}s]"
                    )
                    if run is not None:
                        run.log(log_dict, step=iteration)

                if self.save_steps > 0 and iteration % self.save_steps == 0:
                    self._save_checkpoint(model, optimizer, iteration)
                    self._prune_checkpoints()
                    score = stats["success_rate"] + 0.1 * stats["mean_gate_completion"]
                    if score > best_score:
                        best_score = score
                        self._save_checkpoint(model, optimizer, iteration, is_best=True)
                        print(f"  [best] score={score:.3f} sr={stats['success_rate']:.1%}")
        finally:
            vec_env.close()

        if self.save_final:
            self._save_checkpoint(model, optimizer, self.max_iterations, name="final")
        if run is not None:
            run.finish()

    def _save_checkpoint(self, model: nn.Module, optimizer, iteration: int, is_best: bool = False, name: str | None = None):
        if is_best:
            ckpt_dir = os.path.join(self.output_dir, "best")
        elif name is not None:
            ckpt_dir = os.path.join(self.output_dir, name)
        else:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{iteration}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "actor_critic.pt"))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"iteration": iteration, "is_best": is_best}, f, indent=2)
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

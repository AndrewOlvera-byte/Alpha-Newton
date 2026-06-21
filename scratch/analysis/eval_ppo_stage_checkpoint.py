"""Evaluate a Flightmare PPO checkpoint in the exact trainer stage env.

This is a diagnostic helper for stage-transition failures.  It deliberately
uses the PPO trainer's env kwargs and curriculum application path instead of
the autonomy-stack evaluator, whose reset/goal semantics are different.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.core.config import Config
from src.core.registry import build
from src.entrypoints.train_ppo import _build_flightmare_ppo_model, _resolve_bc_checkpoint
from src.robotics.flightmare_envs import make_flightmare_vec_env
from src.robotics.flightmare_ppo_trainer import _initial_prev_actions, _stack_obs

# Register builders.
import src.robotics.flightmare_envs  # noqa: F401
import src.robotics.flightmare_ppo_trainer  # noqa: F401
import src.robotics.models.flightmare  # noqa: F401


def _load_model(cfg: Config, checkpoint: str, device: torch.device):
    bc_checkpoint = _resolve_bc_checkpoint(cfg, None)
    model = _build_flightmare_ppo_model(cfg, bc_checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[Warn] unexpected checkpoint keys: {unexpected}")
    if missing:
        print(f"[Warn] missing checkpoint keys: {missing}")
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iteration", type=int, required=True, help="Trainer iteration used to select curriculum stage.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = Config.from_experiment(args.exp)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(cfg, args.checkpoint, device)

    trainer = build(
        "trainer",
        type=cfg.training.get("trainer_type", "ppo_flightmare"),
        model=model,
        training_cfg=cfg.training,
        robotics_cfg=cfg.robotics,
        wandb_cfg={"mode": "disabled"},
        data_cfg=cfg.data,
    )
    if trainer.curriculum is not None:
        trainer.curriculum.update(int(args.iteration), None)
        print(f"[Stage] {trainer.curriculum.describe()}")
    env_kwargs = trainer._apply_curriculum_overrides(trainer._env_kwargs())
    env_kwargs["n_workers"] = 0
    action_clip = env_kwargs.get("action_clip")

    vec_env = make_flightmare_vec_env(n_envs=1, seed=int(args.seed), **env_kwargs)
    obs_list, infos = vec_env.reset(seed=int(args.seed))
    prev_actions = _initial_prev_actions(infos, 1, int((cfg.robotics or {}).get("architecture", {}).get("action_dim", 4)))

    episode_count = 0
    current_reward = np.zeros(1, dtype=np.float32)
    current_length = np.zeros(1, dtype=np.int32)
    episodes: list[dict] = []
    counters = Counter()
    crash_reasons = Counter()
    max_speed = 0.0
    speed_sum = 0.0
    speed_count = 0

    try:
        for _ in range(int(args.max_steps)):
            batch = _stack_obs(obs_list, prev_actions, device)
            with torch.no_grad():
                if args.deterministic and hasattr(model, "predict"):
                    actions = model.predict(batch)
                else:
                    actions, _, _ = model.act(batch)
                if action_clip is not None:
                    actions = actions.clamp(-float(action_clip), float(action_clip))
            actions_np = actions.float().cpu().numpy()
            next_obs, rewards, terms, truncs, step_infos = vec_env.step(actions_np)
            done = np.logical_or(terms, truncs)
            current_reward += rewards
            current_length += 1

            for i, info in enumerate(step_infos):
                speed = float(info.get("speed_mps", 0.0))
                max_speed = max(max_speed, speed)
                speed_sum += speed
                speed_count += 1
                if info.get("gate_passed", False):
                    counters["gate_passes"] += 1
                if info.get("gate_missed", False):
                    counters["gate_misses"] += 1
                if info.get("crash", False):
                    counters["crashes"] += 1
                    crash_reasons[str(info.get("crash_reason"))] += 1

            if done[0]:
                info = step_infos[0]
                episode = {
                    "success": bool(info.get("success", False)),
                    "length": int(current_length[0]),
                    "reward": float(current_reward[0]),
                    "gates_completed": int(info.get("gates_completed", 0)),
                    "gate_completion": float(info.get("gate_completion", 0.0)),
                    "goal_completion": float(info.get("goal_completion", 0.0)),
                    "gate_missed": bool(info.get("gate_missed", False)),
                    "crash": bool(info.get("crash", False)),
                    "crash_reason": info.get("crash_reason"),
                    "speed_mps": float(info.get("speed_mps", 0.0)),
                    "pos": np.asarray(info.get("pos", np.zeros(3)), dtype=float).tolist()
                    if info.get("pos") is not None
                    else None,
                }
                episodes.append(episode)
                episode_count += 1
                reset_obs, reset_infos = vec_env.reset_at([0])
                next_obs[0] = reset_obs[0]
                reset_prev = np.asarray(reset_infos[0].get("initial_prev_action_norm", np.zeros(prev_actions.shape[1])), dtype=np.float32)
                prev_actions[0] = reset_prev if reset_prev.shape == prev_actions[0].shape else 0.0
                current_reward[0] = 0.0
                current_length[0] = 0
                if episode_count >= int(args.episodes):
                    break
            else:
                applied = step_infos[0].get("action_norm")
                if applied is not None:
                    applied = np.asarray(applied, dtype=np.float32)
                    prev_actions[0] = applied if applied.shape == prev_actions[0].shape else actions_np[0]
                else:
                    prev_actions[0] = actions_np[0]
            obs_list = next_obs
    finally:
        vec_env.close()

    successes = [ep["success"] for ep in episodes]
    result = {
        "exp": args.exp,
        "checkpoint": args.checkpoint,
        "iteration": int(args.iteration),
        "deterministic": bool(args.deterministic),
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_length": float(np.mean([ep["length"] for ep in episodes])) if episodes else 0.0,
        "mean_reward": float(np.mean([ep["reward"] for ep in episodes])) if episodes else 0.0,
        "mean_goal_completion": float(np.mean([ep["goal_completion"] for ep in episodes])) if episodes else 0.0,
        "mean_gate_completion": float(np.mean([ep["gate_completion"] for ep in episodes])) if episodes else 0.0,
        "mean_speed_mps": float(speed_sum / max(1, speed_count)),
        "max_speed_mps": float(max_speed),
        "counts": dict(counters),
        "crash_reasons": dict(crash_reasons),
        "first_episodes": episodes[:10],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

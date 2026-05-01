"""PPO diagnostics: rollout FPS, reward sanity, per-task advantage norm.

Runs inside the alpha-newton docker container. Benchmarks with the current
EGL path, inspects per-task reward magnitudes over a few hundred steps with a
BC-loaded actor-critic, and verifies the new per-task advantage normalization.
"""
from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys, time, json
from collections import defaultdict

import numpy as np
import torch

from src.core.config import Config
from src.robotics.models.ActorCritic import VLAMLPActorCritic
from src.robotics.models.HistoryBuffer import HistoryBuffer
from src.robotics.normalization import NormStats
from src.robotics.envs import (
    EnvSpec, allocate_envs_by_weights, make_robosuite_multitask_vec_env,
)
from src.robotics.collector import RolloutBuffer, collect_rollouts
import src.robotics.models.VLA_MLP  # noqa
import src.robotics.models.VLA_Gaussian  # noqa

CKPT = "outputs/bc_multitask_mlp_rl_ready_medium_gnll_all/best/model.pt"
EXP = "ppo_all"

# Overrides via env vars so we can sweep without editing the config.
_ENV_N_ENVS = int(os.environ.get("DIAG_N_ENVS", "8"))
_ENV_CAM_SIZE = int(os.environ.get("DIAG_CAM_SIZE", "0"))  # 0 = use cfg
_ENV_N_STEPS = int(os.environ.get("DIAG_N_STEPS", "64"))
_ENV_ALLOW_MANY = os.environ.get("DIAG_ALLOW_MANY_ENVS", "0") == "1"


def load_norm(task):
    base = os.path.dirname(CKPT)
    path = os.path.join(base, f"norm_stats_{task}.json")
    return NormStats.load(path)


def main():
    cfg = Config.from_experiment(EXP)
    arch_cfg = cfg.robotics["architecture"]
    ppo_cfg = cfg.robotics["ppo"]
    obs_keys = cfg.data["obs_keys"]
    camera_names = [k.replace("_image", "") for k in obs_keys["image"]]
    camera_size = _ENV_CAM_SIZE or cfg.data["image_size"]

    tasks = [e["name"] for e in cfg.data["tasks"]]
    weights = [float(e["weight"]) for e in cfg.data["tasks"]]
    task_ids = list(range(len(tasks)))
    norm_by_task = {t: load_norm(t) for t in tasks}

    # --- Build a smaller vec env for quick benchmark ---
    n_envs = _ENV_N_ENVS
    if n_envs > 8 and not _ENV_ALLOW_MANY:
        raise SystemExit(
            f"DIAG_N_ENVS={n_envs} is blocked by default because env startup can "
            "saturate disk/RAM. Set DIAG_ALLOW_MANY_ENVS=1 only for an intentional "
            "16-env benchmark."
        )
    counts = allocate_envs_by_weights(n_envs, weights)
    print(f"[DIAG] n_envs={n_envs} camera_size={camera_size} allocation: "
          f"{dict(zip(tasks, counts))}")

    task_horizons = ppo_cfg.get("task_horizons", {})
    specs, task_id_per_env, task_name_per_env = [], [], []
    for t, tid, c in zip(tasks, task_ids, counts):
        h = int(task_horizons.get(t, ppo_cfg.get("horizon", 400)))
        for _ in range(c):
            specs.append(EnvSpec(
                task_name=t, task_id=tid, norm_stats=norm_by_task[t],
                horizon=h, reward_shaping=True,
                reward_fn=ppo_cfg.get("reward_fn"),
            ))
            task_id_per_env.append(tid); task_name_per_env.append(t)

    t_build = time.time()
    vec = make_robosuite_multitask_vec_env(
        env_specs=specs, camera_names=camera_names, camera_size=camera_size,
        obs_keys_low_dim=obs_keys["low_dim"], obs_keys_image=obs_keys["image"],
    )
    print(f"[DIAG] vec_env build: {time.time()-t_build:.1f}s")

    # Move the policy to CUDA only after env workers are launched. This keeps
    # Linux fork workers from inheriting a live CUDA context and avoids spawn's
    # repeated heavy import storm for large env counts.
    device = torch.device("cuda")
    actor_critic = VLAMLPActorCritic.from_bc_checkpoint(
        CKPT, arch_cfg, value_hidden_dim=ppo_cfg.get("value_hidden_dim", 512)
    ).to(device).eval()

    H = arch_cfg["history_length"]; A = arch_cfg["action_dim"]; O = arch_cfg["obs_dim"]
    hbs = [HistoryBuffer(history_length=H, action_dim=A, task_id=s.task_id,
                         norm_stats=s.norm_stats, target_state_dim=O) for s in specs]

    image_specs = {c: (camera_size, camera_size, 3) for c in obs_keys["image"]}
    n_steps = _ENV_N_STEPS
    buf = RolloutBuffer(n_steps=n_steps, n_envs=n_envs, action_dim=A,
                        state_dim=O, image_specs=image_specs,
                        history_length=H, device=device)

    obs_list, _ = vec.reset()
    for i in range(n_envs):
        hbs[i].reset()
        hbs[i].push(obs_list[i]["images"], obs_list[i]["state"])

    # --- Rollout benchmark (measures EGL throughput) ---
    print(f"[DIAG] warming up 8 steps...")
    _ = collect_rollouts(vec, actor_critic, hbs, buf, 8, device,
                         task_id_per_env=task_id_per_env,
                         task_name_per_env=task_name_per_env,
                         profile=True, show_progress=True)

    print(f"[DIAG] benchmark rollout: n_steps={n_steps} n_envs={n_envs}")
    t0 = time.time()
    stats = collect_rollouts(vec, actor_critic, hbs, buf, n_steps, device,
                             task_id_per_env=task_id_per_env,
                             task_name_per_env=task_name_per_env,
                             profile=True, show_progress=True)
    dt = time.time() - t0
    total_steps = n_steps * n_envs
    print(f"[DIAG] rollout: {dt:.2f}s → {total_steps/dt:.1f} env-steps/s "
          f"({total_steps/dt/n_envs:.2f} FPS/env)")
    print(f"[DIAG] rollout reported rfps={stats['rollout_fps']:.1f}")
    tm = stats.get("timing", {})
    if tm:
        total = tm["total_s"]
        print(f"[DIAG] phase breakdown (of {total:.2f}s, n_steps={n_steps}):")
        for k in ("batch_build_s", "forward_s", "vec_step_s", "insert_s", "hb_push_s"):
            v = tm.get(k, 0.0)
            print(f"   {k:16s} {v:6.2f}s  ({v/total*100:5.1f}%)  "
                  f"{v/n_steps*1000:6.1f} ms/step")

    # --- Reward distribution per task (from the rollout just taken) ---
    rewards = buf.rewards.cpu().numpy()      # [T, N]
    tids = buf.task_ids.cpu().numpy()        # [T, N]
    print("\n[DIAG] per-task per-step reward stats (from 64 steps):")
    for tid, tname in enumerate(tasks):
        mask = (tids == tid)
        if not mask.any():
            continue
        r = rewards[mask]
        print(f"  {tname:10s}  n={mask.sum():4d}  mean={r.mean():+.4f}  "
              f"std={r.std():.4f}  min={r.min():+.4f}  max={r.max():+.4f}")

    # --- GAE + per-task advantage normalization check ---
    stats_last = stats["last_values"]
    buf.compute_advantages(stats_last, ppo_cfg["gamma"], ppo_cfg["gae_lambda"])
    adv_before = buf.advantages.clone().cpu().numpy()
    buf.normalize_advantages_per_task()
    adv_after = buf.advantages.cpu().numpy()

    print("\n[DIAG] per-task advantage stats (pre → post normalize_per_task):")
    for tid, tname in enumerate(tasks):
        mask = (tids == tid)
        if not mask.any():
            continue
        b = adv_before[mask]; a = adv_after[mask]
        print(f"  {tname:10s}  pre  μ={b.mean():+.3f} σ={b.std():.3f}  "
              f"| post μ={a.mean():+.3f} σ={a.std():.3f}")

    # --- Verify no stray obs-key fallbacks in rewards (square/tool_hang) ---
    from src.robotics.rewards import (_reward_square, _reward_tool_hang,
                                      dense_multitask_v1)
    fake_info = {"success": False}
    # Square should USE SquareNut_to_robot0_eef_pos now
    sq_obs_ok = {"SquareNut_to_robot0_eef_pos": np.array([0.05, 0.0, 0.0]),
                 "robot0_eef_pos": np.array([0.0, 0.0, 1.0])}
    sq_obs_empty = {"robot0_eef_pos": np.array([0.0, 0.0, 1.0])}
    r1 = _reward_square(sq_obs_ok, base_reward=0.0)
    r2 = _reward_square(sq_obs_empty, base_reward=0.0)
    print(f"\n[DIAG] square reward shaping delta (with key vs without): "
          f"{r1:+.4f} vs {r2:+.4f}  → delta={r1-r2:+.4f}")
    assert r1 != r2, "square reward not using SquareNut_to_robot0_eef_pos!"

    th_obs = {"tool_to_robot0_eef_pos": np.array([0.1, 0.0, 0.0]),
              "tool_pos": np.array([0.0, 0.0, 1.0]),
              "frame_pos": np.array([0.1, 0.0, 1.0]),
              "tool_on_frame": True, "frame_is_assembled": False,
              "robot0_eef_pos": np.array([0.0, 0.0, 1.0])}
    r_th = _reward_tool_hang(th_obs, base_reward=0.5)
    print(f"[DIAG] tool_hang sample reward (with milestones): {r_th:+.4f}")

    # success bonus path
    r_succ = dense_multitask_v1(task_name="lift", obs={"cube_pos":np.array([0,0,1.0])},
                                base_reward=1.0, terminated=True, truncated=False,
                                info={"success": True}, step_in_episode=100)
    r_fail = dense_multitask_v1(task_name="lift", obs={"cube_pos":np.array([0,0,1.0])},
                                base_reward=1.0, terminated=False, truncated=False,
                                info={"success": False}, step_in_episode=100)
    print(f"[DIAG] success-bonus delta (lift terminal): {r_succ-r_fail:+.2f} "
          f"(expect ~50.0)")

    # --- GPU use during rollout (heuristic: EGL renders allocate GPU memory) ---
    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n[DIAG] GPU memory: alloc={mem_alloc:.0f} MiB peak={mem_peak:.0f} MiB")

    vec.close()
    print("\n[DIAG] all checks passed")


if __name__ == "__main__":
    main()

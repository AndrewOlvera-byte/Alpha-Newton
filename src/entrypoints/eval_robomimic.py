"""
Evaluate a trained BC policy on robomimic tasks via robosuite rollouts.

Usage:
    python -m src.entrypoints.eval_robomimic \
        --exp bc_multitask_ph \
        --checkpoint outputs/bc_multitask_ph/best/model.pt \
        --episodes 50 \
        --save-video
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from src.core.config import Config
from src.core.registry import build
from src.robotics.models.HistoryBuffer import HistoryBuffer

import src.robotics.data
import src.robotics.models.VLA_like
import src.robotics.models.VLA_MLP
import src.robotics.models.VLA_Gaussian
from src.robotics.normalization import NormStats


# Robomimic config task names → robosuite environment names
_ROBOSUITE_ENV_MAP = {
    "lift":      "Lift",
    "can":       "PickPlaceCan",
    "square":    "NutAssemblySquare",
    "stack":     "Stack",
    "door":      "Door",
    "wipe":      "Wipe",
    "tool_hang": "ToolHang",
    "transport": "TwoArmTransport",
}


def create_eval_env(task_name: str, camera_names: list, camera_size: int = 84, render_video: bool = False):
    """Create robosuite env for policy evaluation."""
    import robosuite as suite

    robosuite_name = _ROBOSUITE_ENV_MAP.get(task_name.lower(), task_name.capitalize())
    env = suite.make(
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
        ignore_done=True,
        reward_shaping=False,
        control_freq=20,
    )
    return env


def extract_obs(env_obs: dict, obs_keys_low_dim: list, obs_keys_image: list):
    """Extract structured obs from robosuite env observation."""
    # Low-dim state
    state_parts = []
    for key in obs_keys_low_dim:
        if key in env_obs:
            val = env_obs[key]
            if isinstance(val, np.ndarray):
                state_parts.append(val.flatten().astype(np.float32))
            else:
                state_parts.append(np.array([val], dtype=np.float32))
    state = np.concatenate(state_parts) if state_parts else np.zeros(0, dtype=np.float32)

    # Images: {cam_name: [H, W, C] uint8}
    images = {}
    for key in obs_keys_image:
        if key in env_obs:
            images[key] = env_obs[key]
        else:
            cam = key.replace("_image", "")
            img_key = f"{cam}_image"
            if img_key in env_obs:
                images[key] = env_obs[img_key]

    return state, images


def run_episode(
    env,
    model,
    history_buffer: HistoryBuffer,
    obs_keys_low_dim: list,
    obs_keys_image: list,
    max_steps: int,
    device: torch.device,
    num_flow_steps: int = 10,
    norm_stats: NormStats = None,
    ema_alpha: float = 0.6,
    obs_dim: int = None,
):
    """Run single evaluation episode. Returns (success, total_reward, frames, ep_len)."""
    obs = env.reset()
    history_buffer.reset()

    total_reward = 0.0
    frames = []
    ema_action_norm = None  # EMA-smoothed action in normalized space

    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    for step in range(max_steps):
        state, images = extract_obs(obs, obs_keys_low_dim, obs_keys_image)

        # Store frame for video
        if "agentview_image" in images:
            frames.append(images["agentview_image"].copy())

        # Push raw state — buffer normalizes then pads to obs_dim internally
        history_buffer.push(images, state)

        # Get model prediction (outputs normalized action) — match training bf16 precision
        batch = history_buffer.get_batch(device=device)
        with amp_ctx:
            action_norm = model.predict(batch, num_steps=num_flow_steps)
        action_norm_np = action_norm.float().cpu().numpy().squeeze(0)

        # EMA temporal smoothing in normalized space: reduces jitter on single-step execution
        if ema_action_norm is None or ema_alpha <= 0.0:
            ema_action_norm = action_norm_np.copy()
        else:
            ema_action_norm = ema_alpha * action_norm_np + (1.0 - ema_alpha) * ema_action_norm

        # Denormalize → raw action space
        action_np = norm_stats.denormalize_action(ema_action_norm) if norm_stats else ema_action_norm

        # Clip to valid env range
        action_np = np.clip(action_np, -1.0, 1.0)

        # Step environment
        obs, reward, done, info = env.step(action_np)
        total_reward += reward

        # Record executed raw action so buffer uses it as prev_action next step
        history_buffer.record_action(action_np)

        if env._check_success():
            return True, total_reward, frames, step + 1

    return False, total_reward, frames, max_steps


def load_task_norm_stats(ckpt_dir: str, task_name: str) -> NormStats | None:
    """Load per-task norm stats, checking checkpoint dir for norm_stats_{task}.json."""
    candidates = [
        os.path.join(ckpt_dir, f"norm_stats_{task_name}.json"),
        os.path.join(os.path.dirname(ckpt_dir), f"norm_stats_{task_name}.json"),
        os.path.join(ckpt_dir, "norm_stats.json"),
        os.path.join(os.path.dirname(ckpt_dir), "norm_stats.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            stats = NormStats.load(path)
            print(f"[Norm] {task_name}: loaded from {path}")
            return stats
    print(f"[Norm] WARNING: No norm_stats found for task '{task_name}' — actions will NOT be denormalized.")
    return None


def evaluate(
    model: torch.nn.Module,
    task_name: str,
    task_id: int,
    obs_keys_low_dim: list,
    obs_keys_image: list,
    num_episodes: int = 50,
    max_steps: int = 400,
    history_length: int = 3,
    action_dim: int = 7,
    device: torch.device = None,
    num_flow_steps: int = 10,
    camera_size: int = 84,
    save_video: bool = False,
    output_dir: str = "outputs/eval",
    wandb_cfg: dict = None,
    norm_stats: NormStats = None,
    ema_alpha: float = 0.6,
    obs_dim: int = None,
    seed: int | None = None,
):
    """Run full evaluation for one task."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()

    # Camera names from image obs keys (strip _image suffix for robosuite env)
    camera_names = [k.replace("_image", "") for k in obs_keys_image if k]

    print(f"\n[Eval] Task: {task_name} (task_id={task_id})")
    print(f"[Eval] Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"[Eval] Cameras: {camera_names} | History: {history_length} | Camera size: {camera_size}")
    print(f"[Eval] Device: {device} | Flow steps: {num_flow_steps}")

    env = create_eval_env(task_name, camera_names, camera_size)
    # Pass norm_stats + task_id to buffer — it normalizes then pads state internally
    history_buffer = HistoryBuffer(
        history_length, action_dim,
        task_id=task_id,
        norm_stats=norm_stats,
        target_state_dim=obs_dim,
    )

    successes = []
    rewards = []
    episode_lengths = []
    all_frames = [] if save_video else None

    for ep in range(num_episodes):
        success, ep_reward, frames, ep_len = run_episode(
            env, model, history_buffer,
            obs_keys_low_dim, obs_keys_image,
            max_steps, device, num_flow_steps,
            norm_stats=norm_stats,
            ema_alpha=ema_alpha,
            obs_dim=obs_dim,
        )
        successes.append(success)
        rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if save_video and ep < 5 and frames:
            all_frames.append(frames)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            sr = sum(successes) / len(successes) * 100
            print(f"  [{ep+1}/{num_episodes}] SR={sr:.1f}% reward={np.mean(rewards):.2f} "
                  f"len={np.mean(episode_lengths):.0f}")

    env.close()

    success_rate = sum(successes) / len(successes) * 100
    avg_reward = np.mean(rewards)
    avg_len = np.mean(episode_lengths)

    print(f"\n[Eval Results] {task_name}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Reward:   {avg_reward:.3f}")
    print(f"  Avg Length:   {avg_len:.0f}")

    # Save video
    if save_video and all_frames:
        os.makedirs(output_dir, exist_ok=True)
        try:
            import imageio
            for i, ep_frames in enumerate(all_frames):
                video_path = os.path.join(output_dir, f"{task_name}_ep{i}.mp4")
                imageio.mimwrite(video_path, ep_frames, fps=20)
                print(f"  Video saved: {video_path}")
        except ImportError:
            print("  [warn] imageio not installed — skipping video save")

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
    }


def _ensure_display():
    """Start a virtual framebuffer if no display is available (headless Docker)."""
    if os.environ.get("DISPLAY"):
        return
    try:
        import subprocess
        disp = ":99"
        subprocess.Popen(
            ["Xvfb", disp, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = disp
        time.sleep(0.5)
    except FileNotFoundError:
        print("[Eval] WARNING: Xvfb not found — rendering may fail. "
              "Install with: apt-get install -y xvfb")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy on robomimic tasks")
    parser.add_argument("--exp", type=str, required=True, help="Experiment config name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--ema-alpha", type=float, default=0.6,
                        help="EMA current-action coefficient for temporal filtering (0 disables smoothing)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Deterministic evaluation seed")
    args = parser.parse_args()

    _ensure_display()

    cfg = Config.from_experiment(args.exp)
    robotics_cfg = cfg.robotics or {}
    arch_cfg = robotics_cfg.get("architecture", {})

    print(f"[Config] Loading model: {arch_cfg.get('type')}")
    model = build("architecture", **arch_cfg)

    # Load checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[Checkpoint] Loaded: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_keys = cfg.data.get("obs_keys", {})
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])

    tasks = cfg.data.get("tasks", ["lift"])

    all_results = {}
    for task_id, task in enumerate(tasks):
        norm_stats = load_task_norm_stats(ckpt_dir, task)
        result = evaluate(
            model=model,
            task_name=task,
            task_id=task_id,
            obs_keys_low_dim=obs_keys_low_dim,
            obs_keys_image=obs_keys_image,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            history_length=arch_cfg.get("history_length", 3),
            action_dim=arch_cfg.get("action_dim", 7),
            device=device,
            num_flow_steps=args.flow_steps,
            camera_size=cfg.data.get("image_size", 84),
            save_video=args.save_video,
            output_dir=args.output_dir,
            wandb_cfg=cfg.wandb,
            norm_stats=norm_stats,
            ema_alpha=args.ema_alpha,
            obs_dim=arch_cfg.get("obs_dim"),
            seed=args.seed,
        )
        all_results[task] = result

    # Summary across all tasks
    if len(all_results) > 1:
        mean_sr = np.mean([r["success_rate"] for r in all_results.values()])
        print(f"\n[Summary] Mean success rate across {len(tasks)} tasks: {mean_sr:.1f}%")
        for task, r in all_results.items():
            print(f"  {task}: {r['success_rate']:.1f}%")


if __name__ == "__main__":
    main()

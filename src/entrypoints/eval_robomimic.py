"""
Evaluate a trained BC policy on robomimic tasks via robosuite rollouts.

Usage:
    python -m src.entrypoints.eval_robomimic \
        --exp bc_lift_ph \
        --checkpoint outputs/bc_lift_ph/best/model.pt \
        --episodes 50 \
        --save-video
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.core.config import Config
from src.core.registry import build
from src.robotics.models.HistoryBuffer import HistoryBuffer

import src.robotics.data
import src.robotics.models.VLA_like
from src.robotics.normalization import NormStats


def create_eval_env(task_name: str, camera_names: list, camera_size: int = 84, render_video: bool = False):
    """Create robosuite env for policy evaluation."""
    import robosuite as suite

    env = suite.make(
        task_name.capitalize(),
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
        # robosuite uses different key names than HDF5
        # Map common names
        if key == "object-state" and key not in env_obs:
            key = "object-state"
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
        # HDF5 key "agentview_image" corresponds to env obs "agentview_image"
        if key in env_obs:
            images[key] = env_obs[key]
        else:
            # Try without _image suffix
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
):
    """Run single evaluation episode. Returns (success, total_reward, frames, ep_len)."""
    obs = env.reset()
    history_buffer.reset()

    total_reward = 0.0
    frames = []
    prev_action_norm_np = np.zeros(history_buffer.action_dim, dtype=np.float32)
    ema_action_np = None  # EMA-smoothed action in normalized space

    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    for step in range(max_steps):
        state, images = extract_obs(obs, obs_keys_low_dim, obs_keys_image)

        # Store frame for video
        if "agentview_image" in images:
            frames.append(images["agentview_image"].copy())

        # Normalize state before pushing to buffer
        state_norm = norm_stats.normalize_state(state) if norm_stats else state
        prev_action_in = prev_action_norm_np if step > 0 else None

        history_buffer.push(images, state_norm, prev_action_in)

        # Get model prediction (outputs normalized action) — match training bf16 precision
        batch = history_buffer.get_batch(device=device)
        with amp_ctx:
            action_norm = model.predict(batch, num_steps=num_flow_steps)
        action_norm_np = action_norm.float().cpu().numpy().squeeze(0)

        # EMA temporal smoothing: reduces jitter on single-step execution
        if ema_action_np is None:
            ema_action_np = action_norm_np.copy()
        else:
            ema_action_np = ema_alpha * action_norm_np + (1.0 - ema_alpha) * ema_action_np
        action_norm_np = ema_action_np

        # Keep normalized output for next step's prev_action (no round-trip)
        prev_action_norm_np = action_norm_np.copy()

        # Denormalize → raw action space
        action_np = norm_stats.denormalize_action(action_norm_np) if norm_stats else action_norm_np

        # Clip to valid env range
        action_np = np.clip(action_np, -1.0, 1.0)

        # Step environment
        obs, reward, done, info = env.step(action_np)
        total_reward += reward

        if env._check_success():
            return True, total_reward, frames, step + 1

    return False, total_reward, frames, max_steps


def evaluate(
    model: torch.nn.Module,
    task_name: str,
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
):
    """Run full evaluation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Camera names from image obs keys (strip _image suffix for robosuite env)
    camera_names = [k.replace("_image", "") for k in obs_keys_image if k]

    print(f"[Eval] Task: {task_name}")
    print(f"[Eval] Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"[Eval] Cameras: {camera_names} | History: {history_length}")
    print(f"[Eval] Device: {device} | Flow steps: {num_flow_steps}")

    env = create_eval_env(task_name, camera_names, camera_size)
    history_buffer = HistoryBuffer(history_length, action_dim)

    # Wandb
    wandb_run = None
    if wandb_cfg and wandb_cfg.get("project"):
        try:
            import wandb
            if wandb.api.api_key:
                wandb_run = wandb.init(
                    project=wandb_cfg["project"],
                    entity=wandb_cfg.get("entity"),
                    name=f"eval_{task_name}",
                    tags=wandb_cfg.get("tags", []) + ["eval"],
                )
        except Exception:
            pass

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

    # Final metrics
    success_rate = sum(successes) / len(successes) * 100
    avg_reward = np.mean(rewards)
    avg_len = np.mean(episode_lengths)

    print(f"\n[Eval Results] {task_name}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Reward:   {avg_reward:.3f}")
    print(f"  Avg Length:   {avg_len:.0f}")

    if wandb_run:
        import wandb
        wandb.log({
            f"eval/{task_name}/success_rate": success_rate,
            f"eval/{task_name}/avg_reward": avg_reward,
            f"eval/{task_name}/avg_length": avg_len,
        })

    # Save video
    if save_video and all_frames:
        os.makedirs(output_dir, exist_ok=True)
        try:
            import imageio
            for i, frames in enumerate(all_frames):
                video_path = os.path.join(output_dir, f"{task_name}_ep{i}.mp4")
                imageio.mimwrite(video_path, frames, fps=20)
                print(f"  Video saved: {video_path}")
                if wandb_run:
                    wandb.log({f"eval/{task_name}/video_{i}": wandb.Video(video_path, fps=20)})
        except ImportError:
            print("  [warn] imageio not installed — skipping video save")

    if wandb_run:
        wandb.finish()

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
                        help="EMA smoothing coefficient for action temporal filtering (0=no smoothing)")
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

    # Load norm stats — check checkpoint dir, then output_dir root
    norm_stats = None
    for stats_path in [
        os.path.join(ckpt_dir, "norm_stats.json"),
        os.path.join(os.path.dirname(ckpt_dir), "norm_stats.json"),
    ]:
        if os.path.exists(stats_path):
            norm_stats = NormStats.load(stats_path)
            print(f"[Norm] Loaded stats from {stats_path}")
            break
    if norm_stats is None:
        print("[Norm] WARNING: No norm_stats.json found — actions will NOT be denormalized.")

    obs_keys = cfg.data.get("obs_keys", {})
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    obs_keys_image = obs_keys.get("image", [])

    tasks = cfg.data.get("tasks", ["lift"])

    for task in tasks:
        evaluate(
            model=model,
            task_name=task,
            obs_keys_low_dim=obs_keys_low_dim,
            obs_keys_image=obs_keys_image,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            history_length=arch_cfg.get("history_length", 3),
            action_dim=arch_cfg.get("action_dim", 7),
            num_flow_steps=args.flow_steps,
            camera_size=84,  # match training: features extracted from 84px → ViT bilinear upsample to 224
            save_video=args.save_video,
            output_dir=args.output_dir,
            wandb_cfg=cfg.wandb,
            norm_stats=norm_stats,
            ema_alpha=args.ema_alpha,
        )


if __name__ == "__main__":
    main()

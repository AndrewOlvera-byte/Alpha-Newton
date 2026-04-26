"""
PPO fine-tuning entrypoint for BC-pretrained robosuite policies.

Usage:
    python -m src.entrypoints.train_ppo --exp ppo_lift --bc-checkpoint outputs/bc_lift_mlp_rl_ready/best/model.pt
    python -m src.entrypoints.train_ppo --exp ppo_lift --bc-checkpoint outputs/bc_lift_mlp_rl_ready/best/model.pt --test
"""
import os
# Force MuJoCo onto GPU (EGL) before robosuite/mujoco import — 5-10× faster
# offscreen rendering than osmesa, and rendering actually releases the GIL so
# ThreadPoolExecutor env-stepping parallelizes.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL_OVERRIDE", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("MUJOCO_GL_OVERRIDE", "egl")

import argparse

import torch

from src.core.config import Config
from src.core.registry import build
from src.robotics.models.ActorCritic import VLAMLPActorCritic
from src.robotics.normalization import NormStats

# Register builders
import src.robotics.models.VLA_MLP
import src.robotics.models.VLA_Gaussian
import src.robotics.models.ActorCritic
import src.robotics.ppo_trainer


def _load_norm_stats(ckpt_dir: str, task_name: str) -> NormStats:
    """Load norm stats from BC checkpoint directory."""
    candidates = [
        os.path.join(ckpt_dir, f"norm_stats_{task_name}.json"),
        os.path.join(os.path.dirname(ckpt_dir), f"norm_stats_{task_name}.json"),
        os.path.join(ckpt_dir, "norm_stats.json"),
        os.path.join(os.path.dirname(ckpt_dir), "norm_stats.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[Norm] Loaded: {path}")
            return NormStats.load(path)

    # Fallback: load from HDF5
    data_path = f"data/robomimic/{task_name}/ph/image.hdf5"
    if os.path.exists(data_path) and NormStats.exists_in_hdf5(data_path):
        print(f"[Norm] Loaded from HDF5: {data_path}")
        return NormStats.load_from_hdf5(data_path)

    raise FileNotFoundError(f"No norm_stats found for task '{task_name}' in {ckpt_dir}")


def test(cfg: Config, bc_checkpoint: str):
    """Dry-run: build env, model, collect 1 step, verify shapes."""
    from src.robotics.envs import RobosuiteGymEnv
    from src.robotics.models.HistoryBuffer import HistoryBuffer

    arch_cfg = cfg.robotics.get("architecture", {})
    raw_tasks = cfg.data.get("tasks", ["lift"])
    # Normalize string|dict task entries (see main() for the full parser).
    first = raw_tasks[0]
    task_name = first if isinstance(first, str) else first["name"]

    ckpt_dir = os.path.dirname(bc_checkpoint)
    norm_stats = _load_norm_stats(ckpt_dir, task_name)

    obs_keys = cfg.data.get("obs_keys", {})
    obs_keys_image = obs_keys.get("image", [])
    obs_keys_low_dim = obs_keys.get("low_dim", [])
    camera_names = [k.replace("_image", "") for k in obs_keys_image]
    camera_size = cfg.data.get("image_size", 160)

    print("=" * 60)
    print("[TEST] PPO Pipeline Validation")
    print("=" * 60)

    # 1. Build actor-critic from BC checkpoint
    print("\n--- Actor-Critic ---")
    ppo_cfg = cfg.robotics.get("ppo", {})
    actor_critic = VLAMLPActorCritic.from_bc_checkpoint(
        bc_checkpoint, arch_cfg, value_hidden_dim=ppo_cfg.get("value_hidden_dim", 512)
    )
    trainable = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in actor_critic.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")
    print(f"log_std init: {actor_critic.log_std.data.tolist()}")

    # 2. Test environment
    print("\n--- Environment ---")
    env = RobosuiteGymEnv(
        task_name=task_name, norm_stats=norm_stats,
        camera_names=camera_names, camera_size=camera_size,
        obs_keys_low_dim=obs_keys_low_dim, obs_keys_image=obs_keys_image,
    )
    obs, info = env.reset()
    print(f"State: {obs['state'].shape}")
    for cam, img in obs["images"].items():
        print(f"Image/{cam}: {img.shape} dtype={img.dtype}")
    print(f"Action space: {env.action_space}")

    # 3. Test forward pass
    print("\n--- Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = actor_critic.to(device).eval()

    hb = HistoryBuffer(
        history_length=arch_cfg.get("history_length", 3),
        action_dim=arch_cfg.get("action_dim", 7),
        task_id=0, norm_stats=norm_stats,
        target_state_dim=arch_cfg.get("obs_dim"),
    )
    hb.push(obs["images"], obs["state"])
    batch = hb.get_batch(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        action, log_prob, value = actor_critic.act(batch)
    print(f"Action: {action.shape} range=[{action.min():.3f}, {action.max():.3f}]")
    print(f"Log prob: {log_prob.item():.4f}")
    print(f"Value: {value.item():.4f}")

    # 4. Test env step
    obs2, reward, term, trunc, info = env.step(action.float().cpu().numpy()[0])
    print(f"\nStep reward: {reward:.4f} | terminated: {term} | truncated: {trunc}")

    env.close()
    print(f"\n{'=' * 60}")
    print("[TEST] All checks passed!")
    print(f"{'=' * 60}")


def main(exp_name: str, bc_checkpoint: str, test_only: bool = False):
    cfg = Config.from_experiment(exp_name)
    if cfg.run.get("mode") not in {"ppo", "bc"}:
        raise ValueError(
            f"train_ppo.py expects a PPO robotics config. "
            f"Got mode={cfg.run.get('mode')!r} for exp={exp_name!r}."
        )
    if not (cfg.robotics or {}).get("ppo"):
        raise ValueError(
            f"train_ppo.py requires robotics.ppo in exp={exp_name!r}."
        )

    print(f"[Config] Run:  {cfg.run['name']}")
    print(f"[Config] Task: {cfg.data.get('tasks', [])}")
    print(f"[Config] Output: {cfg.training['output_dir']}")
    print()

    if test_only:
        test(cfg, bc_checkpoint)
        return

    # Load components
    arch_cfg = cfg.robotics.get("architecture", {})
    ppo_cfg = cfg.robotics.get("ppo", {})
    raw_tasks = cfg.data.get("tasks", ["lift"])

    # `tasks` can be either:
    #   - a list of strings (BC-style): ["lift", "can", ...]
    #   - a list of dicts with name + optional weight (PPO-style):
    #       [{name: lift, weight: 0.1}, ...]
    # The task order defines task_id — must match the BC training order so
    # task embeddings stay aligned with the pretrained checkpoint.
    task_names: list = []
    task_weights: list = []
    for entry in raw_tasks:
        if isinstance(entry, str):
            task_names.append(entry)
            task_weights.append(None)
        elif isinstance(entry, dict):
            task_names.append(entry["name"])
            task_weights.append(entry.get("weight"))
        else:
            raise ValueError(f"Unrecognized task entry: {entry!r}")
    if all(w is None for w in task_weights):
        task_weights = None  # trainer falls back to uniform
    else:
        task_weights = [float(w) if w is not None else 0.0 for w in task_weights]
    tasks = task_names

    ckpt_dir = os.path.dirname(bc_checkpoint)

    # Load norm_stats for every task in the mixture.
    norm_stats_by_task = {t: _load_norm_stats(ckpt_dir, t) for t in tasks}
    task_ids = list(range(len(tasks)))

    actor_critic = VLAMLPActorCritic.from_bc_checkpoint(
        bc_checkpoint, arch_cfg, value_hidden_dim=ppo_cfg.get("value_hidden_dim", 512)
    )

    obs_keys = cfg.data.get("obs_keys", {})

    raw_horizons = ppo_cfg.get("task_horizons") or {}
    task_horizons = {str(k): int(v) for k, v in raw_horizons.items()}

    trainer = build(
        "trainer",
        type="ppo",
        model=actor_critic,
        training_cfg=cfg.training,
        robotics_cfg=cfg.robotics,
        wandb_cfg=cfg.wandb,
        task_names=tasks,
        task_ids=task_ids,
        task_weights=task_weights,
        norm_stats_by_task=norm_stats_by_task,
        task_horizons=task_horizons,
        obs_keys_low_dim=obs_keys.get("low_dim", []),
        obs_keys_image=obs_keys.get("image", []),
        camera_size=cfg.data.get("image_size", 160),
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO fine-tuning of BC policy")
    parser.add_argument("--exp", type=str, required=True, help="Experiment config name")
    parser.add_argument("--bc-checkpoint", type=str, required=True, help="Path to BC model.pt")
    parser.add_argument("--test", action="store_true", help="Test pipeline only")
    args = parser.parse_args()
    main(exp_name=args.exp, bc_checkpoint=args.bc_checkpoint, test_only=args.test)

#!/usr/bin/env python3
"""Debug robomimic Lift BC evaluation rollouts."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import h5py
import numpy as np
import torch

from src.core.config import Config
from src.core.registry import build
from src.entrypoints.eval_robomimic import _ensure_display, create_eval_env, extract_obs
from src.robotics.models.HistoryBuffer import HistoryBuffer
from src.robotics.normalization import NormStats

import src.robotics.data
import src.robotics.models.VLA_MLP
import src.robotics.models.VLA_Gaussian
import src.robotics.models.VLA_like


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _hdf5_path(cfg: Config, task: str = "lift") -> str:
    use_images = bool(cfg.data.get("obs_keys", {}).get("image", []))
    hdf5_name = "image.hdf5" if use_images else "low_dim.hdf5"
    return os.path.join(cfg.data["data_dir"], task, "ph", hdf5_name)


def inspect_hdf5(cfg: Config):
    path = _hdf5_path(cfg)
    obs_keys = cfg.data["obs_keys"]["low_dim"]
    with h5py.File(path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demos = sorted(f["data"].keys())
        n_eval = max(1, int(len(demos) * cfg.data.get("eval_ratio", 0.1)))
        train_demos = demos[: len(demos) - n_eval]
        actions = np.concatenate([f[f"data/{dk}/actions"][:] for dk in train_demos], axis=0)
        state = np.concatenate(
            [
                np.concatenate(
                    [
                        f[f"data/{dk}/obs/{k}"][:]
                        if f[f"data/{dk}/obs/{k}"][:].ndim > 1
                        else f[f"data/{dk}/obs/{k}"][:][:, None]
                        for k in obs_keys
                    ],
                    axis=-1,
                )
                for dk in train_demos
            ],
            axis=0,
        )
        first = f[f"data/{demos[0]}"]
        print("[HDF5]", path)
        print("[HDF5] env_args controller:", env_args["env_kwargs"].get("controller_configs"))
        print("[HDF5] env_args cameras:", {
            k: env_args["env_kwargs"].get(k)
            for k in ["use_camera_obs", "camera_heights", "camera_widths"]
        })
        print("[HDF5] first obs keys:", sorted(first["obs"].keys()))
        for key in cfg.data["obs_keys"].get("image", []):
            feat_key = key.replace("_image", "_features")
            if f"data/{demos[0]}/obs/{feat_key}" in f:
                print("[HDF5] feature", feat_key, first[f"obs/{feat_key}"].shape, first[f"obs/{feat_key}"].dtype)
            print("[HDF5] image", key, first[f"obs/{key}"].shape, first[f"obs/{key}"].dtype)
        print("[HDF5] action mean/std:", actions.mean(0), actions.std(0))
        print("[HDF5] action p01/p99:", np.percentile(actions, 1, axis=0), np.percentile(actions, 99, axis=0))
        print("[HDF5] state mean/std:", state.mean(0), state.std(0))


def make_probe_env(cfg: Config, controller_mode: str):
    import robosuite as suite

    camera_names = [k.replace("_image", "") for k in cfg.data["obs_keys"].get("image", []) if k]
    common = dict(
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=cfg.data.get("image_size", 84),
        camera_widths=cfg.data.get("image_size", 84),
        camera_depths=False,
        use_object_obs=True,
        ignore_done=True,
        reward_shaping=False,
        control_freq=20,
    )
    if controller_mode == "default":
        return suite.make("Lift", **common)
    if controller_mode == "osc_pose":
        common["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSE")
        return suite.make("Lift", **common)
    if controller_mode == "stored":
        with h5py.File(_hdf5_path(cfg), "r") as f:
            env_args = json.loads(f["data"].attrs["env_args"])
        common["controller_configs"] = env_args["env_kwargs"]["controller_configs"]
        return suite.make("Lift", **common)
    raise ValueError(f"Unknown controller_mode={controller_mode}")


def inspect_controller(cfg: Config):
    import robosuite as suite

    print("[Env] robosuite version:", getattr(suite, "__version__", None))
    for mode in ["default", "osc_pose", "stored"]:
        print(f"\n[Env] controller_mode={mode}")
        try:
            env = make_probe_env(cfg, mode)
        except Exception as exc:
            print("[Env] create failed:", repr(exc))
            continue
        obs = env.reset()
        print("[Env] action_spec:", env.action_spec, "action_dim:", getattr(env, "action_dim", None))
        print("[Env] reset eef/cube/gripper:", obs["robot0_eef_pos"], obs["cube_pos"], obs["robot0_gripper_qpos"])
        for i, robot in enumerate(getattr(env, "robots", [])):
            print("[Env] robot", i, "class:", type(robot))
            for attr in ["controller", "composite_controller", "part_controllers", "arms"]:
                value = getattr(robot, attr, None)
                print("[Env] ", attr, type(value), value)
        probes = {
            "zero": np.zeros(7, dtype=np.float32),
            "xpos": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "zneg": np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "zpos": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            "grip_neg": np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            "grip_pos": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        }
        for name, action in probes.items():
            obs = env.reset()
            eef0 = obs["robot0_eef_pos"].copy()
            grip0 = obs["robot0_gripper_qpos"].copy()
            for _ in range(5):
                obs, _, _, _ = env.step(action)
            print(
                "[Probe]",
                name,
                "eef_delta5=",
                np.round(obs["robot0_eef_pos"] - eef0, 4),
                "grip_delta5=",
                np.round(obs["robot0_gripper_qpos"] - grip0, 4),
            )
        env.close()


def load_policy(cfg: Config, checkpoint: str, device: torch.device):
    arch_cfg = cfg.robotics["architecture"]
    model = build("architecture", **arch_cfg)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def rollout_trace(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_policy(cfg, args.checkpoint, device)
    ckpt_dir = os.path.dirname(args.checkpoint)
    stats_path = os.path.join(ckpt_dir, "norm_stats.json")
    norm_stats = NormStats.load(stats_path)
    arch_cfg = cfg.robotics["architecture"]
    obs_keys = cfg.data["obs_keys"]

    _ensure_display()
    if args.controller_mode == "eval_default":
        env = create_eval_env("lift", [k.replace("_image", "") for k in obs_keys["image"]], cfg.data.get("image_size", 84))
    else:
        env = make_probe_env(cfg, args.controller_mode)

    history = HistoryBuffer(
        arch_cfg.get("history_length", 3),
        arch_cfg.get("action_dim", 7),
        task_id=0,
        norm_stats=norm_stats,
        target_state_dim=arch_cfg.get("obs_dim"),
    )
    obs = env.reset()
    history.reset()
    ema_action_norm = None
    rows = []

    for step in range(args.steps):
        state, images = extract_obs(obs, obs_keys["low_dim"], obs_keys["image"])
        state_z = norm_stats.normalize_state(state)
        history.push(images, state)
        batch = history.get_batch(device=device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            action_norm = model.predict(batch).float().cpu().numpy().squeeze(0)
        if ema_action_norm is None or args.ema_alpha <= 0.0:
            ema_action_norm = action_norm.copy()
        else:
            ema_action_norm = args.ema_alpha * action_norm + (1.0 - args.ema_alpha) * ema_action_norm
        raw_action = np.clip(norm_stats.denormalize_action(ema_action_norm), -1.0, 1.0)
        eef = obs["robot0_eef_pos"].copy()
        cube = obs["cube_pos"].copy()
        grip = obs["robot0_gripper_qpos"].copy()
        obs, reward, done, info = env.step(raw_action)
        history.record_action(raw_action)
        success = bool(env._check_success())
        row = {
            "step": step,
            "eef": eef,
            "cube": cube,
            "gripper": grip,
            "gripper_to_cube": cube - eef,
            "state_z_abs_mean": float(np.abs(state_z).mean()),
            "state_z_abs_max": float(np.abs(state_z).max()),
            "action_norm": action_norm,
            "raw_action": raw_action,
            "reward": float(reward),
            "success": success,
        }
        rows.append(row)
        if step < args.print_steps or step % args.print_every == 0 or success:
            print(
                f"[Rollout] t={step:03d} reward={reward:.3f} success={success} "
                f"eef={np.round(eef, 3)} cube={np.round(cube, 3)} "
                f"g2c={np.round(cube - eef, 3)} zmean={row['state_z_abs_mean']:.2f} "
                f"an={np.round(action_norm, 3)} raw={np.round(raw_action, 3)}"
            )
        if success:
            break

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for row in rows:
                f.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")
        print("[Rollout] wrote", args.jsonl)
    env.close()


def offline_policy_stats(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = build("data", **cfg.data)
    norm_stats = NormStats.load(os.path.join(os.path.dirname(args.checkpoint), "norm_stats.json"))
    for split in ["train", "eval"]:
        ds = dataset[split]
        for d in (ds.datasets if hasattr(ds, "datasets") else [ds]):
            d.norm_stats = norm_stats
    model = load_policy(cfg, args.checkpoint, device)
    loader = torch.utils.data.DataLoader(dataset["eval"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}
            out = model(batch)
            pred = model.predict(batch).float().cpu()
            preds.append(pred)
            targets.append(batch["action"].float().cpu())
            losses.append(out["loss"].item())
            if i + 1 >= args.batches:
                break
    pred = torch.cat(preds)
    target = torch.cat(targets)
    print("[Offline] batches:", len(losses), "loss:", float(np.mean(losses)), "mse:", torch.mean((pred - target) ** 2).item())
    print("[Offline] pred mean/std:", pred.mean(0).numpy(), pred.std(0).numpy())
    print("[Offline] target mean/std:", target.mean(0).numpy(), target.std(0).numpy())
    print("[Offline] first pred/target:", pred[:5].numpy(), target[:5].numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="bc_lift_mlp_rl_ready")
    parser.add_argument("--checkpoint", default="outputs/bc_lift_mlp_rl_ready/best/model.pt")
    parser.add_argument("--mode", choices=["all", "hdf5", "controller", "rollout", "offline"], default="all")
    parser.add_argument("--controller-mode", choices=["eval_default", "default", "osc_pose", "stored"], default="eval_default")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--print-steps", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--ema-alpha", type=float, default=0.0)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    cfg = Config.from_experiment(args.exp)
    if args.mode in ("all", "hdf5"):
        inspect_hdf5(cfg)
    if args.mode in ("all", "controller"):
        inspect_controller(cfg)
    if args.mode in ("all", "offline"):
        offline_policy_stats(cfg, args)
    if args.mode in ("all", "rollout"):
        rollout_trace(cfg, args)


if __name__ == "__main__":
    main()

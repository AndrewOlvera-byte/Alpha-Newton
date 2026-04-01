"""
BC Trainer for robotics visuomotor policies.

Handles:
- Flow matching training loop
- Mixed precision (bf16, FP8 stub)
- Gradient clipping & accumulation
- Cosine LR schedule with warmup
- Wandb logging
- Checkpointing with resume
- Evaluation
"""
from __future__ import annotations

import os
import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.registry import register


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move batch tensors to device with non_blocking."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = _move_batch_to_device(v, device)
        else:
            out[k] = v
    return out


class BCTrainer:
    """Behavioral Cloning trainer with flow matching loss."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset,
        training_cfg: Dict[str, Any],
        robotics_cfg: Dict[str, Any],
        wandb_cfg: Dict[str, Any],
        norm_stats=None,  # NormStats — stored in checkpoint, used by eval
        dataset_meta: dict = None,  # _train_demo_info, _obs_keys_low_dim from data builder
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_cfg = training_cfg
        self.robotics_cfg = robotics_cfg or {}
        self.wandb_cfg = wandb_cfg or {}
        self.norm_stats = norm_stats
        self._dataset_meta = dataset_meta or {}

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Training params
        self.batch_size = training_cfg.get("per_device_train_batch_size", 64)
        self.eval_batch_size = training_cfg.get("per_device_eval_batch_size", self.batch_size)
        self.grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
        self.max_steps = training_cfg.get("max_steps", 50000)
        self.lr = training_cfg.get("learning_rate", 1e-4)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.warmup_steps = training_cfg.get("warmup_steps", 500)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
        self.logging_steps = training_cfg.get("logging_steps", 50)
        self.save_steps = training_cfg.get("save_steps", 5000)
        self.eval_steps = training_cfg.get("eval_steps", 5000)
        self.output_dir = training_cfg.get("output_dir", "outputs/bc")
        self.num_workers = training_cfg.get("dataloader_num_workers", 4)
        self.pin_memory = training_cfg.get("dataloader_pin_memory", True)
        self.prefetch_factor = training_cfg.get("dataloader_prefetch_factor", 2)

        # Mixed precision
        self.use_bf16 = training_cfg.get("bf16", True)
        self.use_fp8 = training_cfg.get("fp8", False)
        # bf16 autocast always runs — FP8 operates below autocast at the Linear level
        self.amp_dtype = torch.bfloat16 if (self.use_bf16 or self.use_fp8) else torch.float32
        self._fp8_enabled = False

        if self.use_fp8:
            try:
                from torchao.float8 import (
                    convert_to_float8_training,
                    Float8LinearConfig,
                    Float8LinearRecipeName,
                )

                # Filter: only convert aligned Linear layers (dims divisible by 16).
                # - Exclude frozen ViT entirely (not in training graph anyway).
                # - Exclude small boundary layers (action_dim=7, obs_dim=19) that
                #   would fail torch._scaled_mm alignment checks.
                def _fp8_filter(mod, fqn):
                    if "vit" in fqn:
                        return False
                    if isinstance(mod, nn.Linear):
                        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
                    return False

                # ROWWISE_WITH_GW_HP: FP8 for forward + dX, bf16 for dW.
                # Compatible with PyTorch 2.9 (no batch-dim constraint in backward).
                config = Float8LinearConfig.from_recipe_name(
                    Float8LinearRecipeName.ROWWISE_WITH_GW_HP
                )
                convert_to_float8_training(
                    self.model, config=config, module_filter_fn=_fp8_filter
                )
                self._fp8_enabled = True
                print("[Trainer] FP8 training enabled via torchao (ROWWISE_WITH_GW_HP)")
                print("[Trainer] FP8 covers transformer QKV/FFN and DiT blocks; "
                      "boundary layers (state/action encoders, output proj) stay in bf16")
            except ImportError:
                print("[Trainer] torchao not found — falling back to bf16. "
                      "Install: pip install torchao")
                self.use_fp8 = False
            except Exception as e:
                print(f"[Trainer] FP8 init failed ({e}) — falling back to bf16")
                self.use_fp8 = False

        # torch.compile
        self.use_compile = training_cfg.get("torch_compile", False)
        if self.use_compile:
            print("[Trainer] Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay, fused=True
        )

        # Cosine schedule with linear warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # DataLoaders
        worker_kwargs = {}
        if self.num_workers > 0:
            worker_kwargs["prefetch_factor"] = self.prefetch_factor
            worker_kwargs["persistent_workers"] = True

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            **worker_kwargs,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=min(self.num_workers, 2),
            pin_memory=self.pin_memory,
            **({"prefetch_factor": self.prefetch_factor, "persistent_workers": True}
               if min(self.num_workers, 2) > 0 else {}),
        )

        # State
        self.global_step = 0
        self.best_eval_loss = float("inf")

        os.makedirs(self.output_dir, exist_ok=True)

        # Norm stats: compute from training data if not provided
        if self.norm_stats is None:
            self._maybe_compute_norm_stats()
        else:
            print("[Trainer] Using pre-computed norm stats.")
            self._inject_norm_stats()

    def _maybe_compute_norm_stats(self):
        """Compute or load normalization statistics from training data."""
        from src.robotics.normalization import NormStats

        stats_path = os.path.join(self.output_dir, "norm_stats.json")

        # Load from existing file if present
        if os.path.exists(stats_path):
            self.norm_stats = NormStats.load(stats_path)
            print(f"[Trainer] Loaded norm stats from {stats_path}")
            self._inject_norm_stats()
            return

        # Compute from training data
        demo_info = self._dataset_meta.get("_train_demo_info")
        obs_keys_low_dim = self._dataset_meta.get("_obs_keys_low_dim", [])

        if not demo_info or not obs_keys_low_dim:
            print("[Trainer] No dataset metadata — skipping norm stats (set norm_stats manually).")
            return

        print("[Trainer] Computing normalization statistics from training data...")
        # Compute per-file and merge (weighted by demo count)
        import numpy as np
        all_actions = []
        all_states = []
        import h5py
        for hdf5_path, demo_keys in demo_info:
            with h5py.File(hdf5_path, "r") as f:
                for dk in demo_keys:
                    all_actions.append(f[f"data/{dk}/actions"][:])
                    parts = []
                    for key in obs_keys_low_dim:
                        obs = f[f"data/{dk}/obs/{key}"][:]
                        if obs.ndim == 1:
                            obs = obs[:, None]
                        parts.append(obs)
                    all_states.append(np.concatenate(parts, axis=-1))

        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)

        min_std = 1e-3
        self.norm_stats = NormStats(
            action_mean=all_actions.mean(0).tolist(),
            action_std=np.maximum(all_actions.std(0), min_std).tolist(),
            state_mean=all_states.mean(0).tolist(),
            state_std=np.maximum(all_states.std(0), min_std).tolist(),
        )
        self.norm_stats.save(stats_path)
        print(f"[Trainer] Norm stats computed and saved to {stats_path}")
        self._inject_norm_stats()

    def _inject_norm_stats(self):
        """Propagate norm stats to datasets so workers see them."""
        if self.norm_stats is None:
            return
        # Inject into dataset instances (handles ConcatDataset too)
        for ds in [self.train_dataset, self.eval_dataset]:
            datasets = ds.datasets if hasattr(ds, "datasets") else [ds]
            for d in datasets:
                d.norm_stats = self.norm_stats

    def _init_wandb(self):
        """Initialize wandb, prompting for API key if not set."""
        project = self.wandb_cfg.get("project")
        if not project:
            print("[Trainer] No wandb project set — skipping wandb logging")
            self._wandb = None
            return

        try:
            import os
            import sys
            import wandb

            # Resolve API key: env var → wandb netrc → interactive prompt
            api_key = (
                os.environ.get("WANDB_API_KEY", "").strip()
                or (wandb.api.api_key or "")
            )

            if not api_key:
                if sys.stdin.isatty():
                    print("\n[Wandb] No API key found.")
                    api_key = input(
                        "[Wandb] Enter WANDB_API_KEY (or press Enter to skip): "
                    ).strip()
                    if api_key:
                        os.environ["WANDB_API_KEY"] = api_key
                        wandb.login(key=api_key, relogin=True)
                    else:
                        print("[Wandb] Skipping wandb logging.\n")
                        self._wandb = None
                        return
                else:
                    print("[Wandb] No API key and non-interactive session — skipping wandb.")
                    self._wandb = None
                    return

            wandb.init(
                project=project,
                entity=self.wandb_cfg.get("entity"),
                name=self.wandb_cfg.get("run_name"),
                tags=self.wandb_cfg.get("tags", []),
                config={
                    "training": self.training_cfg,
                    "robotics": self.robotics_cfg,
                },
            )
            self._wandb = wandb
            print(f"[Wandb] Logging to {project}/{wandb.run.name}")
        except Exception as e:
            print(f"[Wandb] Init failed: {e}. Continuing without wandb.")
            self._wandb = None

    def _log(self, metrics: dict, step: int):
        if self._wandb and self._wandb.run:
            self._wandb.log(metrics, step=step)

    def train(self):
        """Main training loop."""
        self._init_wandb()

        # Print model stats
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"[Trainer] Trainable: {trainable:,} | Frozen: {frozen:,} | Total: {trainable+frozen:,}")
        print(f"[Trainer] Device: {self.device} | bf16: {self.use_bf16} | fp8: {self.use_fp8}")
        print(f"[Trainer] Batch: {self.batch_size} × {self.grad_accum_steps} accum = {self.batch_size * self.grad_accum_steps} effective")
        print(f"[Trainer] LR: {self.lr} | Warmup: {self.warmup_steps} | Max steps: {self.max_steps}")
        print(f"[Trainer] Train: {len(self.train_dataset):,} | Eval: {len(self.eval_dataset):,} samples")
        print()

        # Resume from checkpoint if exists
        self._try_resume()

        self.model.train()
        train_iter = iter(self.train_loader)
        step_losses = []
        t0 = time.time()

        while self.global_step < self.max_steps:
            # Get batch (cycle through dataloader)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = _move_batch_to_device(batch, self.device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_bf16):
                output = self.model(batch)
                loss = output["loss"] / self.grad_accum_steps

            loss.backward()
            step_losses.append(output["loss"].item())

            # Gradient step (with accumulation)
            if (self.global_step + 1) % self.grad_accum_steps == 0 or self.global_step == self.max_steps - 1:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)


            self.global_step += 1

            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_loss = sum(step_losses[-self.logging_steps:]) / min(len(step_losses), self.logging_steps)
                elapsed = time.time() - t0
                steps_per_sec = self.global_step / elapsed
                lr_now = self.scheduler.get_last_lr()[0]
                print(f"[Step {self.global_step}/{self.max_steps}] "
                      f"loss={avg_loss:.4f} lr={lr_now:.2e} "
                      f"steps/s={steps_per_sec:.1f}")
                self._log({
                    "train/loss": avg_loss,
                    "train/lr": lr_now,
                    "train/steps_per_sec": steps_per_sec,
                }, self.global_step)

            # Evaluation
            if self.global_step % self.eval_steps == 0:
                eval_loss = self.evaluate()
                self._log({"eval/loss": eval_loss}, self.global_step)

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")
                    print(f"[Eval] New best: {eval_loss:.4f}")

                self.model.train()

            # Checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(f"checkpoint-{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        print(f"\n[Trainer] Training complete. Output: {self.output_dir}")

        if self._wandb and self._wandb.run:
            self._wandb.finish()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation, return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            batch = _move_batch_to_device(batch, self.device)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_bf16):
                output = self.model(batch)
            total_loss += output["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Eval @ {self.global_step}] loss={avg_loss:.4f} ({n_batches} batches)")
        return avg_loss

    def _save_checkpoint(self, name: str):
        """Save model + optimizer + scheduler state."""
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)

        # Save model weights (handle torch.compile wrapper)
        model_to_save = self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod

        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(path, "training_state.pt"))

        # Save norm stats alongside checkpoint
        if self.norm_stats is not None:
            self.norm_stats.save(os.path.join(path, "norm_stats.json"))

        print(f"[Save] {path}")

    def _try_resume(self):
        """Resume from latest checkpoint if available."""
        # Check for explicit resume path
        resume_path = self.training_cfg.get("resume_from")
        if resume_path and os.path.isdir(resume_path):
            self._load_checkpoint(resume_path)
            return

        # Auto-resume from latest checkpoint-N
        checkpoints = []
        if os.path.exists(self.output_dir):
            for name in os.listdir(self.output_dir):
                if name.startswith("checkpoint-"):
                    try:
                        step = int(name.split("-")[1])
                        checkpoints.append((step, name))
                    except ValueError:
                        pass

        if checkpoints:
            checkpoints.sort()
            latest = checkpoints[-1][1]
            self._load_checkpoint(os.path.join(self.output_dir, latest))

    def _load_checkpoint(self, path: str):
        """Load checkpoint state."""
        model_path = os.path.join(path, "model.pt")
        state_path = os.path.join(path, "training_state.pt")

        if not os.path.exists(model_path):
            return

        model_to_load = self.model
        if hasattr(model_to_load, "_orig_mod"):
            model_to_load = model_to_load._orig_mod

        model_to_load.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["step"]
            self.best_eval_loss = state.get("best_eval_loss", float("inf"))

        print(f"[Resume] Loaded checkpoint from {path} (step {self.global_step})")


@register("trainer", "bc_flow_matching")
def build_bc_trainer(
    model: nn.Module,
    dataset: dict,
    training_cfg: dict,
    robotics_cfg: dict = None,
    wandb_cfg: dict = None,
    **kwargs,
) -> BCTrainer:
    # Extract metadata passed through from data builder (not actual dataset splits)
    dataset_meta = {
        "_train_demo_info": dataset.get("_train_demo_info"),
        "_obs_keys_low_dim": dataset.get("_obs_keys_low_dim"),
    }
    return BCTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        training_cfg=training_cfg,
        robotics_cfg=robotics_cfg,
        wandb_cfg=wandb_cfg,
        dataset_meta=dataset_meta,
    )

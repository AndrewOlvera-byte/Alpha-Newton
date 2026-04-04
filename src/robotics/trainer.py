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
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from src.core.registry import register


class TaskBalancedBatchSampler(Sampler):
    """Yields complete batches with exactly spt = batch_size // n_tasks samples
    from each task.  Implemented as a BatchSampler (yields lists of indices) so
    the DataLoader never crosses a round boundary — the guarantee is exact.

    Works with ConcatDataset where each component dataset corresponds to one task.
    Each epoch independently shuffles within each task before interleaving.
    Falls back cleanly for single-task datasets (equivalent to random batching).

    Note: effective batch size = spt * n_tasks which may be slightly less than
    the requested batch_size when batch_size is not divisible by n_tasks.
    e.g. batch_size=128, n_tasks=3 → spt=42, effective_bs=126.  The <2%
    difference is negligible for LR scaling purposes.
    """

    def __init__(self, dataset, batch_size: int):
        datasets = list(getattr(dataset, "datasets", [dataset]))
        self.n_tasks = len(datasets)
        self.spt = batch_size // self.n_tasks  # samples per task per batch

        # Absolute index range for each task within the ConcatDataset
        self._task_ranges: List[tuple] = []
        offset = 0
        for ds in datasets:
            n = len(ds)
            self._task_ranges.append((offset, offset + n))
            offset += n

        # Limit to the smallest task so all tasks contribute equally.
        min_task_n = min(hi - lo for lo, hi in self._task_ranges)
        self._n_batches = min_task_n // self.spt
        self.effective_batch_size = self.spt * self.n_tasks

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        # Independently shuffle each task's index pool for this epoch
        per_task = []
        for lo, hi in self._task_ranges:
            pool = list(range(lo, hi))
            random.shuffle(pool)
            per_task.append(pool)

        # Each iteration yields one complete batch list.
        # Shuffling within the batch mixes tasks so they're not contiguous.
        for b in range(self._n_batches):
            batch = []
            start = b * self.spt
            for task_pool in per_task:
                batch.extend(task_pool[start: start + self.spt])
            random.shuffle(batch)
            yield batch


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
        # {0: "lift", 1: "can", ...} — used to label per-task wandb metrics readably
        self._task_names: Dict[int, str] = {
            i: name for i, name in enumerate(self._dataset_meta.get("_task_names", []))
        }

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

        # Use TaskBalancedBatchSampler for multi-task (ConcatDataset) to guarantee
        # exactly spt = batch_size // n_tasks samples from each task per batch.
        # Implemented as a batch_sampler so the DataLoader never crosses a round
        # boundary — the guarantee is exact, not approximate.
        _n_task_components = len(getattr(self.train_dataset, "datasets", [None]))
        if _n_task_components > 1:
            _batch_sampler = TaskBalancedBatchSampler(self.train_dataset, batch_size=self.batch_size)
            print(f"[Trainer] TaskBalancedBatchSampler: {_n_task_components} tasks, "
                  f"{_batch_sampler.spt} samples/task/batch "
                  f"(effective bs={_batch_sampler.effective_batch_size}), "
                  f"{_batch_sampler._n_batches} batches/epoch")
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                **worker_kwargs,
            )
        else:
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

        # Early stopping: stop when eval loss fails to improve for `patience` consecutive
        # evaluations. patience=0 disables early stopping.
        self.patience = training_cfg.get("early_stopping_patience", 0)
        self._es_counter = 0  # consecutive evals without improvement

        os.makedirs(self.output_dir, exist_ok=True)

        # Norm stats: compute from training data if not provided
        if self.norm_stats is None:
            self._maybe_compute_norm_stats()
        else:
            print("[Trainer] Using pre-computed norm stats.")
            self._inject_norm_stats()

    def _maybe_compute_norm_stats(self):
        """Compute or load per-task normalization statistics.

        Priority for each task's stats:
          1. norm_stats/ group in the task's own HDF5  (primary — survives across runs)
          2. norm_stats_{task}.json in output_dir       (fallback if HDF5 was regenerated)
          3. Compute from scratch, then write both

        Per-task (not global) normalization is critical for multi-task BC:
        each task has a different action distribution and needs its own reference
        frame.  The policy's task embedding disambiguates, so the model can learn
        separate normalized output spaces per task without confusion.

        Storing stats in the HDF5 collocates normalization with data, making the
        RL handoff clean — the RL script only needs the HDF5 path, not the BC
        output directory.
        """
        from src.robotics.normalization import NormStats

        demo_info = self._dataset_meta.get("_train_demo_info")
        obs_keys_low_dim = self._dataset_meta.get("_obs_keys_low_dim", [])

        if not demo_info or not obs_keys_low_dim:
            print("[Trainer] No dataset metadata — skipping norm stats.")
            return

        is_multitask = len(demo_info) > 1

        if not is_multitask:
            # ── Single-task (backward-compatible path) ──────────────────
            hdf5_path, demo_keys = demo_info[0]
            json_path = os.path.join(self.output_dir, "norm_stats.json")

            if NormStats.exists_in_hdf5(hdf5_path):
                self.norm_stats = NormStats.load_from_hdf5(hdf5_path)
                print(f"[NormStats] Loaded from {hdf5_path} (HDF5 cache)")
            elif os.path.exists(json_path):
                self.norm_stats = NormStats.load(json_path)
                print(f"[NormStats] Loaded from {json_path}")
            else:
                print("[NormStats] Computing for single task...")
                self.norm_stats = NormStats.compute(hdf5_path, demo_keys, obs_keys_low_dim)
                self.norm_stats.save_to_hdf5(hdf5_path, n_train_demos=len(demo_keys))
                self.norm_stats.save(json_path)
                print(f"[NormStats] Saved to HDF5 and {json_path}")

            self._inject_norm_stats()
            return

        # ── Multi-task: compute independently per task ───────────────────
        # Global stats would blend task distributions (lift=vertical,
        # can=horizontal, square=rotational), producing a biased reference.
        # Each task normalizes in its own action/state space.
        print("[NormStats] Multi-task mode — computing per-task statistics:")
        task_stats: Dict[int, NormStats] = {}

        for task_id, (hdf5_path, demo_keys) in enumerate(demo_info):
            task_name = self._task_names.get(task_id, str(task_id))
            json_path = os.path.join(self.output_dir, f"norm_stats_{task_name}.json")

            if NormStats.exists_in_hdf5(hdf5_path):
                ns = NormStats.load_from_hdf5(hdf5_path)
                print(f"  [{task_name}] Loaded from HDF5 cache")
            elif os.path.exists(json_path):
                ns = NormStats.load(json_path)
                print(f"  [{task_name}] Loaded from {json_path}")
            else:
                print(f"  [{task_name}] Computing from {len(demo_keys)} train demos...")
                ns = NormStats.compute(hdf5_path, demo_keys, obs_keys_low_dim)
                ns.save_to_hdf5(hdf5_path, n_train_demos=len(demo_keys))
                ns.save(json_path)
                print(f"  [{task_name}] Saved to HDF5 and {json_path}")

            task_stats[task_id] = ns

        self.norm_stats = task_stats
        self._inject_norm_stats()

    def _inject_norm_stats(self):
        """Propagate norm stats into each dataset so DataLoader workers see them.

        For multi-task (self.norm_stats is a dict), each task dataset gets its
        own NormStats keyed by task_id.  For single-task, all datasets receive
        the same NormStats (backward-compatible).

        Workers are forked on the first iter() call in train(), which happens
        after this method completes — so the injected stats are visible to workers.
        """
        if self.norm_stats is None:
            return
        for ds in [self.train_dataset, self.eval_dataset]:
            sub_datasets = ds.datasets if hasattr(ds, "datasets") else [ds]
            for d in sub_datasets:
                if isinstance(self.norm_stats, dict):
                    # Per-task: look up by the dataset's own task_id
                    d.norm_stats = self.norm_stats.get(d.task_id)
                else:
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

    # ------------------------------------------------------------------
    # Optimization dynamics helpers
    # ------------------------------------------------------------------

    def _grad_norm(self, params) -> float:
        """Total L2 grad norm across a param iterable (pre-clip)."""
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def _param_norm(self, params) -> float:
        """Total L2 weight norm across a param iterable."""
        total = 0.0
        for p in params:
            total += p.data.float().norm(2).item() ** 2
        return total ** 0.5

    def _named_module_params(self):
        """Return {group_name: [params]} for the key trainable sub-modules."""
        m = self.model if not hasattr(self.model, "_orig_mod") else self.model._orig_mod
        groups = {
            "transformer": list(m.transformer.parameters()),
            "dit_head": list(m.dit_head.parameters()),
            "readout_proj": list(m.readout_proj.parameters()),
        }
        if hasattr(m, "vis_pool") and m.vis_pool is not None:
            groups["vis_pool"] = list(m.vis_pool.parameters())
        if hasattr(m, "task_emb"):
            groups["task_emb"] = list(m.task_emb.parameters()) + list(m.task_emb_dit.parameters())
        return groups

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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

        # Rolling buffers for the current logging window
        step_losses: list = []
        step_grad_norms: list = []
        # Per-sample (loss, t) for flow-timestep bucketing
        _window_t: list = []          # list of [B] tensors
        _window_lps: list = []        # list of [B] tensors (loss_per_sample)

        t0 = time.time()
        t_last_log = t0
        _stop_early = False

        while self.global_step < self.max_steps:
            # ── Get batch ──────────────────────────────────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = _move_batch_to_device(batch, self.device)

            # ── Forward ────────────────────────────────────────────────
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_bf16):
                output = self.model(batch)
                loss = output["loss"] / self.grad_accum_steps

            loss.backward()
            step_losses.append(output["loss"].item())

            # Accumulate per-sample diagnostics for t-bucketing
            if "_t" in output:
                _window_t.append(output["_t"].cpu())
                _window_lps.append(output["_loss_per_sample"].cpu())

            # ── Optimizer step ─────────────────────────────────────────
            if (self.global_step + 1) % self.grad_accum_steps == 0 or self.global_step == self.max_steps - 1:
                # Capture grad norm BEFORE clipping
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                grad_norm = self._grad_norm(trainable_params)
                step_grad_norms.append(grad_norm)

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

            # ── Logging ────────────────────────────────────────────────
            if self.global_step % self.logging_steps == 0:
                window = min(len(step_losses), self.logging_steps)
                avg_loss = sum(step_losses[-window:]) / window
                loss_std = float(torch.tensor(step_losses[-window:]).std()) if window > 1 else 0.0
                avg_gnorm = sum(step_grad_norms[-window:]) / max(1, len(step_grad_norms[-window:]))
                grad_clipped_frac = sum(
                    1 for g in step_grad_norms[-window:] if g > self.max_grad_norm
                ) / max(1, len(step_grad_norms[-window:]))

                elapsed = time.time() - t0
                steps_per_sec = self.global_step / elapsed
                remaining_steps = self.max_steps - self.global_step
                eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                lr_now = self.scheduler.get_last_lr()[0]

                # Flow-timestep loss bucketing [0,.25), [.25,.5), [.5,.75), [.75,1]
                t_bucket_metrics = {}
                if _window_t:
                    all_t = torch.cat(_window_t)          # [N]
                    all_lps = torch.cat(_window_lps)      # [N]
                    bucket_edges = [0.0, 0.25, 0.5, 0.75, 1.01]
                    for i in range(4):
                        lo, hi = bucket_edges[i], bucket_edges[i + 1]
                        mask = (all_t >= lo) & (all_t < hi)
                        if mask.any():
                            t_bucket_metrics[f"train/loss_t{int(lo*100):02d}_{int(bucket_edges[i+1]*100):02d}"] = \
                                all_lps[mask].mean().item()
                    _window_t.clear()
                    _window_lps.clear()

                print(
                    f"[{self.global_step:>6}/{self.max_steps}] "
                    f"loss={avg_loss:.4f}±{loss_std:.4f} "
                    f"gnorm={avg_gnorm:.3f} "
                    f"clip={grad_clipped_frac*100:.0f}% "
                    f"lr={lr_now:.2e} "
                    f"{steps_per_sec:.1f}it/s "
                    f"ETA={self._fmt_eta(eta_sec)}"
                )

                self._log({
                    "train/loss": avg_loss,
                    "train/loss_std": loss_std,
                    "train/grad_norm": avg_gnorm,
                    "train/grad_clip_frac": grad_clipped_frac,
                    "train/lr": lr_now,
                    "train/steps_per_sec": steps_per_sec,
                    **t_bucket_metrics,
                }, self.global_step)

                # ── Per-module grad & param norms (every 5× logging interval) ──
                if self.global_step % (self.logging_steps * 5) == 0:
                    module_metrics = {}
                    for name, params in self._named_module_params().items():
                        gnorm = self._grad_norm(params)
                        pnorm = self._param_norm(params)
                        module_metrics[f"grad_norm/{name}"] = gnorm
                        module_metrics[f"param_norm/{name}"] = pnorm
                    self._log(module_metrics, self.global_step)

            # ── Evaluation ─────────────────────────────────────────────
            if self.global_step % self.eval_steps == 0:
                # Use recent training loss for gap metric
                recent_train_loss = sum(step_losses[-self.logging_steps:]) / min(len(step_losses), self.logging_steps)
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics["loss"]

                # Remap task_{id}_* keys to task_{name}_* for readability in wandb
                def _remap(k: str, v: float):
                    for tid, tname in self._task_names.items():
                        k = k.replace(f"task_{tid}_", f"task_{tname}_")
                    return k, v

                remapped = dict(_remap(k, v) for k, v in eval_metrics.items() if k != "loss")
                self._log({
                    "eval/loss": eval_loss,
                    "eval/train_gap": eval_loss - recent_train_loss,
                    **{f"eval/{k}": v for k, v in remapped.items()},
                }, self.global_step)

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._es_counter = 0
                    self._save_checkpoint("best")
                    print(f"[Eval] New best: {eval_loss:.4f}")
                elif self.patience > 0:
                    self._es_counter += 1
                    print(f"[EarlyStopping] No improvement {self._es_counter}/{self.patience}")
                    if self._es_counter >= self.patience:
                        print(f"[EarlyStopping] Patience exhausted — stopping training.")
                        _stop_early = True

                self._log({"eval/es_counter": self._es_counter}, self.global_step)
                self.model.train()
                if _stop_early:
                    self._save_checkpoint("early_stop")
                    break

            # ── Checkpoint ─────────────────────────────────────────────
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(f"checkpoint-{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        elapsed_total = time.time() - t0
        print(f"\n[Trainer] Training complete in {self._fmt_eta(elapsed_total)}. Output: {self.output_dir}")

        if self._wandb and self._wandb.run:
            self._wandb.finish()

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate over the full eval set.

        Returns metrics including:
          - loss: mean flow-matching loss across all tasks
          - task_{id}_loss: per-task mean loss (key diagnostic for multi-task overfit)
          - action_sample_std / task_{id}_action_std: output diversity per task
          - loss_t{lo}_{hi}: t-bucket losses
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        all_t: list = []
        all_lps: list = []

        # Per-task loss accumulation: {task_id: [per-sample losses]}
        task_lps: Dict[int, list] = {}
        # One representative batch per task for diversity sampling.
        # eval_loader is unshuffled (tasks arrive in blocks) so the first batch
        # seen for each task_id is a clean representative.
        task_batches: Dict[int, dict] = {}

        for batch in self.eval_loader:
            batch = _move_batch_to_device(batch, self.device)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_bf16):
                output = self.model(batch)

            total_loss += output["loss"].item()
            n_batches += 1

            if "_t" in output:
                all_t.append(output["_t"].cpu())
                all_lps.append(output["_loss_per_sample"].cpu())

            # Accumulate per-task losses and capture representative batches
            if "task_id" in batch and "_loss_per_sample" in output:
                lps = output["_loss_per_sample"].cpu()
                for i, tid in enumerate(batch["task_id"].cpu().tolist()):
                    tid = int(tid)
                    task_lps.setdefault(tid, []).append(lps[i].item())
                    if tid not in task_batches:
                        task_batches[tid] = batch

        avg_loss = total_loss / max(1, n_batches)
        metrics: Dict[str, float] = {"loss": avg_loss}

        # ── Per-task mean loss ──────────────────────────────────────────
        for tid, losses in sorted(task_lps.items()):
            metrics[f"task_{tid}_loss"] = sum(losses) / len(losses)

        # ── Flow-timestep loss buckets ──────────────────────────────────
        if all_t:
            cat_t = torch.cat(all_t)
            cat_lps = torch.cat(all_lps)
            bucket_edges = [0.0, 0.25, 0.5, 0.75, 1.01]
            for i in range(4):
                lo, hi = bucket_edges[i], bucket_edges[i + 1]
                mask = (cat_t >= lo) & (cat_t < hi)
                if mask.any():
                    metrics[f"loss_t{int(lo*100):02d}_{int(bucket_edges[i+1]*100):02d}"] = \
                        cat_lps[mask].mean().item()

        # ── Action diversity per task ───────────────────────────────────
        # Sample the policy n_samples times on the same obs with different noise.
        # High std = stochastic/diverse policy. Near-zero = mode-collapsed.
        diversity_batches = task_batches if task_batches else {"0": next(iter(self.eval_loader))}
        all_stds = []
        n_samples = 8
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_bf16):
            for tid, tb in diversity_batches.items():
                if not isinstance(tb, dict):
                    tb = _move_batch_to_device(tb, self.device)
                samples = torch.stack([
                    self.model.predict(tb).float() for _ in range(n_samples)
                ], dim=0)  # [n_samples, B, action_dim]
                std = samples.std(dim=0).mean().item()
                all_stds.append(std)
                metrics[f"task_{tid}_action_std"] = std
        metrics["action_sample_std"] = sum(all_stds) / max(1, len(all_stds))

        task_loss_str = "  ".join(
            f"t{k}={v:.4f}" for k, v in metrics.items()
            if k.startswith("task_") and k.endswith("_loss")
        )
        print(
            f"[Eval @ {self.global_step}] "
            f"loss={avg_loss:.4f}  "
            + (f"{task_loss_str}  " if task_loss_str else "")
            + f"action_std={metrics['action_sample_std']:.4f}  "
            f"({n_batches} batches)"
        )
        return metrics

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

        # Save norm stats alongside checkpoint (human-readable backup).
        # Primary cache is already in the HDF5 files; these JSONs exist for
        # quick inspection and as a fallback if HDF5s are regenerated.
        if isinstance(self.norm_stats, dict):
            for task_id, ns in self.norm_stats.items():
                task_name = self._task_names.get(task_id, str(task_id))
                ns.save(os.path.join(path, f"norm_stats_{task_name}.json"))
        elif self.norm_stats is not None:
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
        "_task_names": dataset.get("_task_names", []),
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

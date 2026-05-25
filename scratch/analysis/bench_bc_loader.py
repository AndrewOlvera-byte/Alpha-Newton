"""Micro-benchmark: BC dataset throughput before/after preload."""
import time

import torch
from torch.utils.data import DataLoader

from src.robotics.data import FlightmareBCStateV3Dataset

DATA = "/workspace/data/flightmare/bc_v5"
BS = 16384


def bench(preload: bool, num_workers: int, n_steps: int = 20):
    ds = FlightmareBCStateV3Dataset(
        data_dir=DATA, action_type="ctbr", split="train",
        normalize_obs=True, normalize_action=True,
        action_normalization="bounds", preload=preload,
    )
    dl = DataLoader(
        ds, batch_size=BS, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    it = iter(dl)
    # Warm up
    next(it)
    t0 = time.time()
    n = 0
    for _ in range(n_steps):
        b = next(it)
        # Move to GPU like real training would
        _ = b["state"].cuda(non_blocking=True)
        _ = b["action"].cuda(non_blocking=True)
        n += b["state"].shape[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"preload={preload} workers={num_workers}: "
          f"{n_steps} steps in {dt:.2f}s -> {n_steps/dt:.1f} steps/s, "
          f"{n/dt:.0f} samples/s")


bench(preload=False, num_workers=8)
bench(preload=True, num_workers=0)
bench(preload=True, num_workers=2)

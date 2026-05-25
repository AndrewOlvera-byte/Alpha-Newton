"""Quick health check on data/flightmare/bc_v5 vs bc_v4.

Reports:
  - episode-level: completion rate, mean speed, length
  - per-axis CTBR action distribution + saturation against norm_stats bounds
  - body-rate envelope (recorded vs racing-plant integrator cap of 18 rad/s)
  - manifest plant params to confirm omega_max_body=18 was actually used
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOTS = {
    "bc_v4 (tame plant)": Path("/workspace/data/flightmare/bc_v4"),
    "bc_v5 (racing plant)": Path("/workspace/data/flightmare/bc_v5"),
}


def summarize(root: Path) -> None:
    idx = json.loads((root / "index.json").read_text())
    eps = idx.get("episodes", [])
    print(f"\n=== {root} ===")
    print(f"episodes in manifest: {len(eps)}")

    plant = idx.get("params", {})
    print("plant.omega_max_body =", plant.get("omega_max_body"))
    print("plant.mass =", plant.get("mass"),
          " inertia =", plant.get("inertia"),
          " motor_omega_max =", plant.get("motor_omega_max"))

    splits = {}
    completions = []
    speeds = []
    lengths = []
    for e in eps:
        splits[e.get("split", "?")] = splits.get(e.get("split", "?"), 0) + 1
        if "gate_completion" in e:
            completions.append(float(e["gate_completion"]))
        if "mean_speed" in e:
            speeds.append(float(e["mean_speed"]))
        if "num_steps" in e:
            lengths.append(int(e["num_steps"]))
    print("splits:", splits)
    if completions:
        c = np.array(completions)
        print(f"gate_completion: mean={c.mean():.3f} p50={np.median(c):.3f} >=0.99 frac={(c>=0.99).mean():.3f}")

    ns = np.load(root / "norm_stats.npz")
    low = ns["ctbr_low"]; high = ns["ctbr_high"]
    print(f"ctbr_low  = {low}")
    print(f"ctbr_high = {high}")

    # Sample first N kept episodes to compute action + body-rate stats.
    kept = [e for e in eps if e.get("split") in ("train", "val")][:200]
    acts = []
    omegas = []
    for e in kept:
        path = root / e["path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            if "action/ctbr" in f:
                acts.append(f["action/ctbr"][:])
            if "obs/state" in f:
                s = f["obs/state"][:]
                if s.shape[1] >= 13:
                    omegas.append(s[:, 10:13])
    if not acts:
        print("no actions found")
        return
    A = np.concatenate(acts, axis=0)
    print(f"ctbr action samples: {A.shape[0]}")
    for i, name in enumerate(["thrust_norm", "wx_cmd", "wy_cmd", "wz_cmd"]):
        col = A[:, i]
        sat_lo = (col <= low[i] + 1e-6).mean()
        sat_hi = (col >= high[i] - 1e-6).mean()
        print(f"  {name:11s} min={col.min():.3f} max={col.max():.3f} "
              f"mean={col.mean():.3f} std={col.std():.3f} "
              f"sat_lo={sat_lo:.3f} sat_hi={sat_hi:.3f}")
    if omegas:
        W = np.concatenate(omegas, axis=0)
        for i, ax in enumerate("xyz"):
            w = W[:, i]
            print(f"  omega_body_{ax}: min={w.min():.3f} max={w.max():.3f} "
                  f"|w|>6 frac={(np.abs(w)>6).mean():.4f}  "
                  f"|w|>10 frac={(np.abs(w)>10).mean():.4f}  "
                  f"|w|>15 frac={(np.abs(w)>15).mean():.4f}")


for label, r in ROOTS.items():
    if not r.exists():
        print(f"skip {label}: {r} missing")
        continue
    print(f"\n##### {label} #####")
    summarize(r)

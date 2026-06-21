"""Per-stage summary of the teacher gate-chaining runs."""
from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/scratch/analysis/csv_diag")
RUNS = {
    "v5_stage12": "run-20260525_051645-b53kjvg8.csv",
    "v6_stage12": "run-20260526_043731-a49gbbny.csv",
    "v7_stage12": "run-20260528_180825-9wsuub8m.csv",
    "v7_stage2":  "run-20260528_210603-8v34uosm.csv",
}
METRICS = [
    "rollout/success_rate",
    "rollout/mean_goal_completion",
    "rollout/mean_gate_completion",
    "rollout/mean_reward",
    "rollout/mean_speed_mps",
    "rollout/mean_length",
    "rollout/gate_passes",
    "rollout/gate_misses",
    "rollout/crashes",
    "rollout/action_clip_fraction",
    "ppo/approx_kl",
    "ppo/grad_norm",
    "ppo/explained_variance",
    "ppo/entropy",
]

def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")

for label, fn in RUNS.items():
    p = ROOT / fn
    if not p.exists():
        print(f"\n##### {label}: MISSING {p}")
        continue
    df = pd.read_csv(p)
    print(f"\n############ {label}  ({fn})  rows={len(df)}")
    if "curriculum/stage_idx" not in df.columns:
        print("  no stage_idx; cols:", [c for c in df.columns if 'rollout' in c][:8])
        continue
    stage = num(df, "curriculum/stage_idx").ffill().fillna(-1).astype(int)
    it = num(df, "iteration")
    for st in sorted(stage.dropna().unique()):
        seg = df.loc[stage == st]
        i0 = num(seg, "iteration").min()
        i1 = num(seg, "iteration").max()
        print(f"\n  --- stage {st}  iters {i0:.0f}-{i1:.0f}  rows={len(seg)}")
        for c in METRICS:
            if c not in seg.columns:
                continue
            s = num(seg, c).dropna()
            if s.empty:
                continue
            # last-quarter mean shows converged behavior in the stage
            tail = s.tail(max(1, len(s)//4))
            print(f"    {c:32s} mean={s.mean():9.3f} last_q={tail.mean():9.3f} max={s.max():9.3f}")

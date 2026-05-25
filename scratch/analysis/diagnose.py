"""Stage-by-stage diagnostic comparison of selected runs."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/workspace/scratch/analysis/csv")
RUNS = {
    "bridge_teacher (FAIL)": "run-20260524_070356-6l8f21c1.csv",
    "full_curr_teacher (FAIL)": "run-20260522_043706-jfdtajtv.csv",
    "exp_stage12_v4 (OK)": "run-20260512_045616-14430cgd.csv",
    "exp_stage12_v4_short (OK)": "run-20260512_035348-er7ns9o3.csv",
    "exp_stage23 (OK)": "run-20260512_213008-ckqir51h.csv",
}

KEY_METRICS = [
    "rollout/success_rate",
    "rollout/mean_gate_completion",
    "rollout/mean_reward",
    "rollout/mean_speed_mps",
    "rollout/action_clip_fraction",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "ppo/grad_norm",
    "ppo/value_loss",
    "ppo/explained_variance",
    "ppo/entropy",
    "ppo/early_stop",
    "curriculum/stage_idx",
    "curriculum/ent_coeff",
    "reward_terms/total",
    "reward_terms/gate_pass",
    "reward_terms/gate_miss",
    "reward_terms/crash",
    "reward_terms/body_rate",
    "reward_terms/action_smoothness",
    "reward_terms/centerline_error",
]


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def stage_segments(df):
    if "curriculum/stage_idx" not in df:
        return [(0, 0, len(df) - 1)]
    s = df["curriculum/stage_idx"].ffill().fillna(0).astype(int).to_numpy()
    segs = []
    cur = s[0]
    start = 0
    for i in range(1, len(s)):
        if s[i] != cur:
            segs.append((int(cur), start, i - 1))
            cur = s[i]
            start = i
    segs.append((int(cur), start, len(s) - 1))
    return segs


def quantile(s, q):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def first_crossing(df, col, thresh, op=">"):
    if col not in df:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    mask = (s > thresh) if op == ">" else (s < thresh)
    idx = np.where(mask.fillna(False))[0]
    return int(idx[0]) if len(idx) else None


print("=" * 100)
print("STAGE-BY-STAGE SUMMARY (mean per stage)")
print("=" * 100)
for label, fn in RUNS.items():
    p = ROOT / fn
    if not p.exists():
        print(f"missing {fn}")
        continue
    df = pd.read_csv(p)
    print(f"\n### {label}   rows={len(df)}")
    for stage, a, b in stage_segments(df):
        seg = df.iloc[a:b + 1]
        print(f"  stage={stage}  iters {a}-{b}  (n={len(seg)})")
        for m in KEY_METRICS:
            if m in seg.columns:
                v = safe_mean(seg[m])
                p99 = quantile(seg[m], 0.99)
                print(f"    {m:42s} mean={v:11.4f}  p99={p99:11.4f}")

print("\n" + "=" * 100)
print("KL / GRADIENT BLOWUP DETECTION")
print("=" * 100)
for label, fn in RUNS.items():
    df = pd.read_csv(ROOT / fn)
    kl_blowup = first_crossing(df, "ppo/approx_kl", 1.0)
    kl_huge = first_crossing(df, "ppo/approx_kl", 1000.0)
    grad_huge = first_crossing(df, "ppo/grad_norm", 5000.0)
    print(f"{label:35s}  first kl>1: {kl_blowup}  first kl>1000: {kl_huge}  first grad>5000: {grad_huge}")
    if "ppo/approx_kl" in df:
        kl = pd.to_numeric(df["ppo/approx_kl"], errors="coerce")
        print(f"    kl quantiles: p50={kl.quantile(.5):.4f}  p90={kl.quantile(.9):.4f}  p99={kl.quantile(.99):.4f}  max={kl.max():.4g}")
    if "ppo/grad_norm" in df:
        gn = pd.to_numeric(df["ppo/grad_norm"], errors="coerce")
        print(f"    grad quantiles: p50={gn.quantile(.5):.2f}  p90={gn.quantile(.9):.2f}  p99={gn.quantile(.99):.2f}  max={gn.max():.4g}")

print("\n" + "=" * 100)
print("ACTION CLIP SATURATION (network output clipping to action_clip=1.0)")
print("=" * 100)
for label, fn in RUNS.items():
    df = pd.read_csv(ROOT / fn)
    c = df.get("rollout/action_clip_fraction")
    if c is None:
        print(f"{label}: no clip metric"); continue
    c = pd.to_numeric(c, errors="coerce").dropna()
    print(f"{label:35s}  mean={c.mean():.3f}  p50={c.quantile(.5):.3f}  p90={c.quantile(.9):.3f}  max={c.max():.3f}")

print("\n" + "=" * 100)
print("BRIDGE TEACHER: ITERATION-LEVEL TRAJECTORY AROUND STAGE TRANSITIONS")
print("=" * 100)
df = pd.read_csv(ROOT / RUNS["bridge_teacher (FAIL)"])
df["it"] = np.arange(len(df))
# stage transitions
segs = stage_segments(df)
print("segments:", segs)
# Look at windows around iter 1200 (stage 1->2a) and around best iter 404
for center in [400, 800, 1200, 1500, 2000, 3000, 4500, 6000, 8000]:
    if center >= len(df):
        continue
    w = df.iloc[max(0, center - 5):min(len(df), center + 5)]
    row = w.iloc[len(w) // 2]
    print(f"iter~{center:5d} stage={int(row.get('curriculum/stage_idx', -1))}  "
          f"succ={row.get('rollout/success_rate', float('nan')):.2f}  "
          f"gc={row.get('rollout/mean_gate_completion', float('nan')):.2f}  "
          f"R={row.get('rollout/mean_reward', float('nan')):8.2f}  "
          f"kl={row.get('ppo/approx_kl', float('nan')):.4g}  "
          f"gn={row.get('ppo/grad_norm', float('nan')):.1f}  "
          f"vL={row.get('ppo/value_loss', float('nan')):.2f}  "
          f"ent={row.get('ppo/entropy', float('nan')):.3f}  "
          f"clipF={row.get('rollout/action_clip_fraction', float('nan')):.3f}")

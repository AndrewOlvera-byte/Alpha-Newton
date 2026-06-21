"""Summarize stage metrics for the v5 stage12 PPO investigation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/workspace/scratch/analysis/csv_v5")
RUNS = {
    "v5_latest": "run-20260525_051645-b53kjvg8.csv",
    "v5_quickfail": "run-20260525_050924-7m2ohheg.csv",
    "bridge_fail": "run-20260524_070356-6l8f21c1.csv",
    "ok_v4": "run-20260512_045616-14430cgd.csv",
}
METRICS = [
    "rollout/success_rate",
    "rollout/mean_gate_completion",
    "rollout/mean_reward",
    "rollout/mean_speed_mps",
    "rollout/max_speed_mps",
    "rollout/mean_length",
    "rollout/crashes",
    "rollout/gate_misses",
    "rollout/action_clip_fraction",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "ppo/grad_norm",
    "ppo/value_loss",
    "ppo/explained_variance",
    "collapse_guard/skipped_update",
    "checkpoint/best_iteration",
]


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()


def show_run(label: str, filename: str) -> None:
    path = ROOT / filename
    if not path.exists():
        print(f"\n### {label}: missing {path}")
        return
    df = pd.read_csv(path)
    print(f"\n### {label}  {filename}  rows={len(df)} cols={len(df.columns)}")
    if "curriculum/stage_idx" not in df.columns:
        print("no curriculum/stage_idx column")
        print(df.tail(5).to_string(index=False))
        return

    stage = df["curriculum/stage_idx"].ffill().fillna(-1).astype(int)
    print("stage counts:", stage.value_counts().sort_index().to_dict())
    for st in sorted(stage.unique()):
        seg = df.loc[stage == st]
        print(f" stage {st}: rows={len(seg)} rows {seg.index.min() + 1}-{seg.index.max() + 1}")
        for col in METRICS:
            if col not in seg.columns:
                continue
            s = numeric(seg, col)
            if s.empty:
                continue
            print(
                f"  {col:34s} mean={s.mean():10.4f} "
                f"p50={s.quantile(0.50):10.4f} "
                f"p90={s.quantile(0.90):10.4f} "
                f"max={s.max():10.4f}"
            )

    keep = [
        "iteration",
        "_step",
        "curriculum/stage_idx",
        "rollout/success_rate",
        "rollout/mean_gate_completion",
        "rollout/mean_reward",
        "rollout/mean_speed_mps",
        "rollout/max_speed_mps",
        "rollout/mean_length",
        "rollout/gate_passes",
        "rollout/gate_misses",
        "rollout/crashes",
        "rollout/action_clip_fraction",
        "ppo/approx_kl",
        "ppo/grad_norm",
        "collapse_guard/skipped_update",
        "collapse_guard/restored_best",
        "checkpoint/best_iteration",
    ]
    keep = [c for c in keep if c in df.columns]
    print(" tail:")
    print(df[keep].tail(10).to_string(index=False))

    if label == "v5_latest":
        print(" stage transition windows:")
        for center in [1995, 2000, 2001, 2016, 3998, 4000, 4001, 4010, 4500, 5200]:
            lo = max(0, center - 3)
            hi = min(len(df), center + 4)
            window = df.iloc[lo:hi][keep]
            print(f"\n window around row {center}:")
            print(window.to_string(index=False))


def main() -> None:
    for label, filename in RUNS.items():
        show_run(label, filename)

    keep = [
        "iteration",
        "curriculum/stage_idx",
        "rollout/success_rate",
        "rollout/mean_gate_completion",
        "rollout/mean_reward",
        "rollout/mean_speed_mps",
        "rollout/max_speed_mps",
        "rollout/mean_length",
        "rollout/gate_passes",
        "rollout/gate_misses",
        "rollout/crashes",
        "rollout/action_clip_fraction",
        "ppo/approx_kl",
        "ppo/grad_norm",
        "collapse_guard/skipped_update",
        "collapse_guard/restored_best",
        "checkpoint/best_iteration",
    ]
    probes = {
        "v5_latest": (
            "run-20260525_051645-b53kjvg8.csv",
            [3998, 4000, 4001, 4010, 4050, 4200, 4500, 5000, 5210],
        ),
        "ok_v4": (
            "run-20260512_045616-14430cgd.csv",
            [1198, 1200, 1201, 1210, 1250, 1500, 2000, 3000, 3500],
        ),
    }
    print("\n### probe rows")
    for label, (filename, rows) in probes.items():
        df = pd.read_csv(ROOT / filename)
        cols = [c for c in keep if c in df.columns]
        print(f"\n## {label}")
        for row_num in rows:
            if row_num < 1 or row_num > len(df):
                continue
            row = df.iloc[row_num - 1]
            values = {c: row[c] for c in cols}
            print(row_num, values)


if __name__ == "__main__":
    main()

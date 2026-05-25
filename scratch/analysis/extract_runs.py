"""Extract per-iteration time series from local offline/online wandb run dirs.

Reads the .wandb event file via wandb's stream protocol and dumps a parquet/csv
of selected metrics for each requested run.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Use wandb's internal reader for .wandb files
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


METRICS = [
    "iteration",
    "_step",
    "rollout/success_rate",
    "rollout/mean_gate_completion",
    "rollout/mean_reward",
    "rollout/mean_speed_mps",
    "rollout/max_speed_mps",
    "rollout/mean_length",
    "rollout/crashes",
    "rollout/gate_passes",
    "rollout/gate_misses",
    "rollout/action_clip_fraction",
    "rollout/reward_step_min",
    "rollout/reward_step_max",
    "rollout/reward_step_std",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "ppo/early_stop",
    "ppo/entropy",
    "ppo/explained_variance",
    "ppo/grad_norm",
    "ppo/policy_loss",
    "ppo/value_loss",
    "curriculum/stage_idx",
    "curriculum/ent_coeff",
    "reward_terms/total",
    "reward_terms/gate_pass",
    "reward_terms/gate_miss",
    "reward_terms/crash",
    "reward_terms/centerline_error",
    "reward_terms/centerline_progress",
    "reward_terms/progress_gate_normal",
    "reward_terms/progress_segment",
    "reward_terms/body_rate",
    "reward_terms/action_smoothness",
    "reward_terms/velocity_alignment",
    "reward_terms/aperture_violation_near",
    "checkpoint/best_score",
    "checkpoint/best_iteration",
    "checkpoint/selection_score",
    "collapse_guard/skipped_update",
]


def read_run(run_dir: Path) -> pd.DataFrame:
    wandb_files = list(run_dir.glob("*.wandb"))
    if not wandb_files:
        raise FileNotFoundError(f"no .wandb file in {run_dir}")
    path = wandb_files[0]
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            try:
                v = json.loads(item.value_json)
            except Exception:
                continue
            key = item.key if item.key else "/".join(item.nested_key)
            row[key] = v
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    keep = [c for c in METRICS if c in df.columns]
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-root", default="/workspace/wandb")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in args.runs:
        rd = Path(args.wandb_root) / r
        print(f"reading {rd}")
        df = read_run(rd)
        outp = out / f"{r}.csv"
        df.to_csv(outp, index=False)
        print(f"  -> {outp}  rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()

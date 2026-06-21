#!/usr/bin/env python3
"""Summarize Flightmare PPO WandB output logs by curriculum stage.

This is intentionally log-only: it does not create envs or touch Flightmare
workers, so it is safe to run while another training job is active.
"""

from __future__ import annotations

import argparse
import re
import statistics as stats
from dataclasses import dataclass
from pathlib import Path


ITER_RE = re.compile(r"^\[(\d+)/(\d+)\]")
STAGE_ENABLED_RE = re.compile(
    r"Curriculum enabled: stage (?P<idx>\d+)/(?P<num>\d+) '(?P<name>[^']+)' until_iter=(?P<until>\d+)"
)
STAGE_TRANSITION_RE = re.compile(
    r"Curriculum transition at iter (?P<iter>\d+): stage (?P<idx>\d+)/(?P<num>\d+) "
    r"'(?P<name>[^']+)' until_iter=(?P<until>\d+)"
)
PASS_RE = re.compile(r"pass=(\d+) miss=(\d+) crash=(\d+)")
SPEED_RE = re.compile(r"v=([-+0-9.]+)/([-+0-9.]+)")
START_GATE_RE = re.compile(r"sg=([^ ]+)")


@dataclass
class Stage:
    idx: int
    name: str
    start_iter: int
    until_iter: int
    end_iter: int | None = None


def _field(line: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([-+0-9.]+)", line)
    return float(match.group(1)) if match else None


def _parse_start_gates(line: str) -> dict[int, int]:
    match = START_GATE_RE.search(line)
    if not match:
        return {}
    out: dict[int, int] = {}
    for item in match.group(1).split(","):
        if ":" not in item:
            continue
        gate, count = item.split(":", 1)
        try:
            out[int(gate)] = int(count)
        except ValueError:
            continue
    return out


def parse_log(path: Path) -> tuple[list[Stage], list[dict[str, float | int | dict[int, int]]], int]:
    stages: list[Stage] = []
    rows: list[dict[str, float | int | dict[int, int]]] = []
    max_iter = 0

    for line in path.read_text(errors="replace").splitlines():
        enabled = STAGE_ENABLED_RE.search(line)
        if enabled:
            stages.append(
                Stage(
                    idx=int(enabled.group("idx")),
                    name=enabled.group("name"),
                    start_iter=1,
                    until_iter=int(enabled.group("until")),
                )
            )
            continue

        transition = STAGE_TRANSITION_RE.search(line)
        if transition:
            stages.append(
                Stage(
                    idx=int(transition.group("idx")),
                    name=transition.group("name"),
                    start_iter=int(transition.group("iter")),
                    until_iter=int(transition.group("until")),
                )
            )
            continue

        iter_match = ITER_RE.search(line)
        if not iter_match:
            continue

        pass_match = PASS_RE.search(line)
        speed_match = SPEED_RE.search(line)
        if not pass_match or not speed_match:
            continue

        iteration = int(iter_match.group(1))
        max_iter = max(max_iter, iteration)
        rows.append(
            {
                "iter": iteration,
                "success_rate": _field(line, "sr") or 0.0,
                "goal_completion": _field(line, "goal") or 0.0,
                "gate_completion": _field(line, "gc") or 0.0,
                "episode_reward": _field(line, "ep_r") or 0.0,
                "step_reward": _field(line, "step_r") or 0.0,
                "episodes": int(_field(line, "eps") or 0),
                "mean_length": int(_field(line, "len") or 0),
                "gate_passes": int(pass_match.group(1)),
                "gate_misses": int(pass_match.group(2)),
                "crashes": int(pass_match.group(3)),
                "mean_speed": float(speed_match.group(1)),
                "max_speed": float(speed_match.group(2)),
                "action_clip": _field(line, "aclip") or 0.0,
                "value_loss": _field(line, "vl") or 0.0,
                "kl": _field(line, "kl") or 0.0,
                "start_gates": _parse_start_gates(line),
            }
        )

    stages.sort(key=lambda stage: stage.start_iter)
    for i, stage in enumerate(stages):
        stage.end_iter = stages[i + 1].start_iter - 1 if i + 1 < len(stages) else max_iter
    return stages, rows, max_iter


def _mean(values: list[float | int]) -> float:
    return float(stats.mean(values)) if values else 0.0


def summarize_stage(stage: Stage, rows: list[dict[str, float | int | dict[int, int]]]) -> dict[str, str]:
    stage_rows = [
        row
        for row in rows
        if stage.start_iter <= int(row["iter"]) <= int(stage.end_iter or stage.until_iter)
    ]
    if not stage_rows:
        return {
            "stage": f"{stage.idx}:{stage.name}",
            "iters": f"{stage.start_iter}-{stage.end_iter or stage.until_iter}",
            "n": "0",
        }

    tail = stage_rows[-min(100, len(stage_rows)) :]
    final = stage_rows[-1]
    best_sr = max(stage_rows, key=lambda row: float(row["success_rate"]))
    best_goal = max(stage_rows, key=lambda row: float(row["goal_completion"]))
    start_gate_union = sorted(
        {
            gate
            for row in stage_rows
            for gate in (row.get("start_gates") or {}).keys()  # type: ignore[union-attr]
        }
    )

    return {
        "stage": f"{stage.idx}:{stage.name}",
        "iters": f"{stage.start_iter}-{stage.end_iter or stage.until_iter}",
        "n": str(len(stage_rows)),
        "best_sr": f"{float(best_sr['success_rate']):.1f}@{int(best_sr['iter'])}",
        "best_goal": f"{float(best_goal['goal_completion']):.1f}@{int(best_goal['iter'])}",
        "tail100_sr": f"{_mean([float(row['success_rate']) for row in tail]):.1f}",
        "final_sr": f"{float(final['success_rate']):.1f}",
        "final_goal": f"{float(final['goal_completion']):.1f}",
        "final_gc": f"{float(final['gate_completion']):.1f}",
        "final_pass": str(int(final["gate_passes"])),
        "final_miss": str(int(final["gate_misses"])),
        "final_crash": str(int(final["crashes"])),
        "final_aclip": f"{float(final['action_clip']):.2f}",
        "final_speed": f"{float(final['mean_speed']):.1f}/{float(final['max_speed']):.1f}",
        "start_gates": ",".join(str(gate) for gate in start_gate_union),
    }


def print_table(label: str, path: Path, stages: list[Stage], rows: list[dict[str, float | int | dict[int, int]]]) -> None:
    print(f"\n## {label}")
    print(f"log: {path}")
    if rows:
        latest = rows[-1]
        print(
            "latest: "
            f"iter={int(latest['iter'])} sr={float(latest['success_rate']):.1f}% "
            f"goal={float(latest['goal_completion']):.1f}% gc={float(latest['gate_completion']):.1f}% "
            f"pass={int(latest['gate_passes'])} miss={int(latest['gate_misses'])} crash={int(latest['crashes'])} "
            f"aclip={float(latest['action_clip']):.2f}"
        )

    columns = [
        "stage",
        "iters",
        "n",
        "best_sr",
        "best_goal",
        "tail100_sr",
        "final_sr",
        "final_goal",
        "final_gc",
        "final_pass",
        "final_miss",
        "final_crash",
        "final_aclip",
        "final_speed",
        "start_gates",
    ]
    print(",".join(columns))
    for stage in stages:
        summary = summarize_stage(stage, rows)
        print(",".join(summary.get(column, "") for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="WandB output.log path(s)")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each log, in the same order as logs.",
    )
    args = parser.parse_args()

    for idx, path in enumerate(args.logs):
        label = args.label[idx] if idx < len(args.label) else path.parent.parent.name
        stages, rows, _ = parse_log(path)
        print_table(label, path, stages, rows)


if __name__ == "__main__":
    main()

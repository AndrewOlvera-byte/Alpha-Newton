from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.robotics.flightmare_autonomy_fsw.graph import EpisodeResult


def summarize_results(results: list[EpisodeResult]) -> dict:
    if not results:
        return {}

    completed = [r for r in results if r.completed]
    completion_times = [r.completion_time_s for r in completed if r.completion_time_s is not None]
    return {
        "episodes": len(results),
        "success_rate": float(len(completed) / len(results)),
        "mean_gates_completed": float(np.mean([r.gates_completed for r in results])),
        "mean_gate_completion": float(np.mean([r.gate_completion for r in results])),
        "mean_completion_time_s": float(np.mean(completion_times)) if completion_times else None,
        "median_completion_time_s": float(np.median(completion_times)) if completion_times else None,
        "mean_elapsed_time_s": float(np.mean([r.elapsed_time_s for r in results])),
        "mean_speed_mps": float(np.mean([r.mean_speed_mps for r in results])),
        "max_speed_mps": float(np.max([r.max_speed_mps for r in results])),
        "mean_path_length_m": float(np.mean([r.path_length_m for r in results])),
        "total_gate_misses": int(sum(getattr(r, "gate_misses", 0) for r in results)),
        "gate_miss_rate": float(np.mean([getattr(r, "gate_misses", 0) > 0 for r in results])),
        "fallback_episodes": int(sum(r.using_fallback for r in results)),
    }


def write_stats_json(results: list[EpisodeResult], path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_results(results),
        "episodes": [r.metrics_dict() for r in results],
    }
    if extra:
        payload["config"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_trajectory_plot(
    results: list[EpisodeResult],
    path: str | Path,
    title: str = "Flightmare State-Only Evaluation",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    gate_positions = None
    gate_yaws = None
    for idx, result in enumerate(results):
        traj = result.trajectory
        if traj.size:
            color = cmap(idx % 20)
            label = f"ep {result.episode_id}: {result.gates_completed}/{result.num_gates}"
            if result.completed and result.completion_time_s is not None:
                label += f", {result.completion_time_s:.2f}s"
            ax_xy.plot(traj[:, 0], traj[:, 1], lw=1.5, color=color, alpha=0.9, label=label)
            ax_xy.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=18)
            ax_xy.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="x", s=24)
            ax_z.plot(np.arange(len(traj)), traj[:, 2], lw=1.2, color=color, alpha=0.9)
        if gate_positions is None and result.gate_positions.size:
            gate_positions = result.gate_positions
            gate_yaws = result.gate_yaws

    if gate_positions is not None:
        ax_xy.scatter(gate_positions[:, 0], gate_positions[:, 1], c="black", marker="s", s=34, label="gates")
        for i, gate in enumerate(gate_positions):
            ax_xy.text(gate[0], gate[1], str(i), fontsize=8, ha="center", va="bottom")
        if gate_yaws is not None and len(gate_yaws) == len(gate_positions):
            ax_xy.quiver(
                gate_positions[:, 0],
                gate_positions[:, 1],
                np.cos(gate_yaws),
                np.sin(gate_yaws),
                angles="xy",
                scale_units="xy",
                scale=1.8,
                width=0.004,
                color="black",
                alpha=0.7,
            )

    stats_lines = [
        f"episodes: {summary.get('episodes', 0)}",
        f"success: {100.0 * summary.get('success_rate', 0.0):.1f}%",
        f"mean gates: {summary.get('mean_gates_completed', 0.0):.2f}",
        f"mean completion: {summary.get('mean_completion_time_s') or 0.0:.2f}s",
        f"mean speed: {summary.get('mean_speed_mps', 0.0):.2f} m/s",
    ]
    ax_xy.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax_xy.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    ax_xy.set_title(title)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    if len(results) <= 12:
        ax_xy.legend(loc="best", fontsize=8)

    ax_z.set_title("Altitude")
    ax_z.set_xlabel("step")
    ax_z.set_ylabel("z [m]")
    ax_z.grid(True, alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)

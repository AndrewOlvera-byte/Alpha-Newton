"""Action-space bounds and normalization helpers for Flightmare labels."""
from __future__ import annotations

from typing import Mapping

import numpy as np


ACTION_TYPES = ("waypoint", "ctbr", "motor")

# Bounds are intentionally policy-interface bounds, not empirical dataset
# extrema. They keep normalized actions comparable when we recollect at higher
# speed and prevent the old low-speed BC statistics from becoming a speed cap.
DEFAULT_ACTION_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "waypoint": (
        np.array([-50.0, -50.0, -50.0, 0.0], dtype=np.float32),
        np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32),
    ),
    "ctbr": (
        np.array([0.0, -12.0, -12.0, -12.0], dtype=np.float32),
        np.array([1.0, 12.0, 12.0, 12.0], dtype=np.float32),
    ),
    "motor": (
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    ),
}


def action_bounds(action_type: str, stats: Mapping | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return low/high raw-action bounds for an action type."""
    low_key = f"{action_type}_low"
    high_key = f"{action_type}_high"
    if stats is not None and low_key in stats and high_key in stats:
        return (
            np.asarray(stats[low_key], dtype=np.float32),
            np.asarray(stats[high_key], dtype=np.float32),
        )
    if action_type not in DEFAULT_ACTION_BOUNDS:
        raise ValueError(f"Unsupported Flightmare action_type={action_type!r}")
    low, high = DEFAULT_ACTION_BOUNDS[action_type]
    return low.copy(), high.copy()


def stats_mode(stats: Mapping | None, default: str = "standard") -> str:
    if stats is None or "action_normalization" not in stats:
        return default
    raw = stats["action_normalization"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def normalize_action(
    action: np.ndarray,
    *,
    action_type: str,
    mean: np.ndarray,
    std: np.ndarray,
    mode: str,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if mode == "standard":
        return ((action - mean) / std).astype(np.float32)
    if mode == "bounds":
        if low is None or high is None:
            low, high = action_bounds(action_type)
        scale = np.maximum(high - low, 1e-6)
        return (2.0 * (action - low) / scale - 1.0).astype(np.float32)
    raise ValueError(f"Unknown action normalization mode={mode!r}")


def denormalize_action(
    action: np.ndarray,
    *,
    action_type: str,
    mean: np.ndarray,
    std: np.ndarray,
    mode: str,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if mode == "standard":
        return (action * std + mean).astype(np.float32)
    if mode == "bounds":
        if low is None or high is None:
            low, high = action_bounds(action_type)
        scale = np.maximum(high - low, 1e-6)
        return (low + 0.5 * (action + 1.0) * scale).astype(np.float32)
    raise ValueError(f"Unknown action normalization mode={mode!r}")

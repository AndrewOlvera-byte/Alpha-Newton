"""Action-space bounds and normalization helpers for Flightmare labels."""
from __future__ import annotations

from typing import Mapping

import numpy as np


ACTION_TYPES = ("waypoint", "ctbr", "motor")

# Fallback policy-interface bounds, used only when norm_stats.npz lacks
# per-channel low/high. Prefer empirical bounds (see ``empirical_action_bounds``)
# computed at norm_stats time, since these wide defaults under-utilize the
# normalized [-1, 1] range and starve bounds-mode BC of resolution.
DEFAULT_ACTION_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "waypoint": (
        np.array([-10.0, -10.0, -10.0, 0.0], dtype=np.float32),
        np.array([ 10.0,  10.0,  10.0, 35.0], dtype=np.float32),
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


# Hard physical bounds. Empirical limits are clamped to these so motor PWM
# stays in [0, 1] and waypoint speed stays non-negative regardless of data.
PHYSICAL_ACTION_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "waypoint": (
        np.array([-np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32),
        np.array([ np.inf,  np.inf,  np.inf, np.inf], dtype=np.float32),
    ),
    "ctbr": (
        np.array([0.0, -np.inf, -np.inf, -np.inf], dtype=np.float32),
        np.array([1.0,  np.inf,  np.inf,  np.inf], dtype=np.float32),
    ),
    "motor": (
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    ),
}


def empirical_action_bounds(
    action_type: str,
    actions: np.ndarray,
    margin: float = 0.10,
    min_span: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel low/high from observed actions, expanded by ``margin``.

    Bounds are clamped to ``PHYSICAL_ACTION_BOUNDS``. Use this at norm_stats
    write time so that bounds-mode normalization actually fills [-1, 1].
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [N, dim], got shape={actions.shape}")
    lo = actions.min(axis=0)
    hi = actions.max(axis=0)
    span = np.maximum(hi - lo, min_span)
    lo = lo - margin * span
    hi = hi + margin * span
    plo, phi = PHYSICAL_ACTION_BOUNDS.get(
        action_type,
        (np.full_like(lo, -np.inf), np.full_like(hi, np.inf)),
    )
    lo = np.maximum(lo, plo)
    hi = np.minimum(hi, phi)
    return lo.astype(np.float32), hi.astype(np.float32)


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

"""Distillation label helpers for Flightmare CTBR experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robotics.flightmare_experts.safety import SafetySupervisor


@dataclass
class ExpertLabel:
    """Normalized record written for one distillation target."""

    action_ctbr_raw: np.ndarray
    source: str
    valid: bool = True
    weight: float = 1.0
    confidence: float = 1.0
    action_log_std_norm: np.ndarray | None = None
    expert_cost: float | None = None
    cost_margin: float | None = None
    saturation_frac: float = 0.0
    safety_status: str = "ok"
    command: dict | None = None

    def as_hdf5_expert(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "weight": float(self.weight),
            "confidence": float(self.confidence),
            "source": str(self.source),
            "valid": bool(self.valid),
            "cost": np.nan if self.expert_cost is None else float(self.expert_cost),
            "cost_margin": np.nan if self.cost_margin is None else float(self.cost_margin),
            "saturation_frac": float(self.saturation_frac),
            "safety_status": str(self.safety_status),
        }
        if self.action_log_std_norm is not None:
            out["action_log_std"] = np.asarray(self.action_log_std_norm, dtype=np.float32)
        return out


def ctbr_command_to_raw(command: dict) -> np.ndarray:
    rates = np.asarray(command.get("body_rates", np.zeros(3)), dtype=np.float32)
    return np.asarray(
        [float(command.get("thrust_normalized", 0.0)), rates[0], rates[1], rates[2]],
        dtype=np.float32,
    )


def normalize_action_bounds(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    scale = np.maximum(high - low, 1e-6)
    return (2.0 * (action - low) / scale - 1.0).astype(np.float32)


def normalize_std_bounds(std_raw: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    std_raw = np.asarray(std_raw, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    scale = np.maximum(high - low, 1e-6)
    return (2.0 * std_raw / scale).astype(np.float32)


def command_saturation_frac(command: dict, max_body_rate: float, thrust_eps: float = 0.02) -> float:
    raw = ctbr_command_to_raw(command)
    thrust_sat = raw[0] <= thrust_eps or raw[0] >= 1.0 - thrust_eps
    rate_sat = np.abs(raw[1:4]) >= float(max_body_rate) * 0.98
    return float((float(thrust_sat) + float(np.mean(rate_sat))) / 2.0)


class HybridExpertLabeler:
    """Primary expert with geometric-style fallback and uniform label contract."""

    def __init__(
        self,
        primary,
        fallback=None,
        *,
        supervisor: SafetySupervisor | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        max_body_rate: float = 18.0,
        min_confidence: float = 0.05,
        default_log_std_norm: tuple[float, float, float, float] = (-2.0, -2.0, -2.0, -2.0),
    ):
        self.primary = primary
        self.fallback = fallback
        self.supervisor = supervisor or SafetySupervisor(max_body_rate=max_body_rate)
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32)
        self.max_body_rate = float(max_body_rate)
        self.min_confidence = float(min_confidence)
        self.default_log_std_norm = np.asarray(default_log_std_norm, dtype=np.float32)

    def compute(self, *args, **kwargs) -> ExpertLabel:
        primary_label = self._try_source(self.primary, *args, **kwargs)
        if primary_label.valid and primary_label.confidence >= self.min_confidence:
            return primary_label
        if self.fallback is None:
            return primary_label
        fallback_label = self._try_source(self.fallback, *args, **kwargs)
        if not fallback_label.valid:
            return primary_label
        fallback_label.source = f"{fallback_label.source}_fallback"
        fallback_label.weight = min(fallback_label.weight, max(0.25, primary_label.weight))
        return fallback_label

    def _try_source(self, expert, *args, **kwargs) -> ExpertLabel:
        source = str(getattr(expert, "source", expert.__class__.__name__))
        try:
            command = expert.compute(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive path
            return ExpertLabel(
                action_ctbr_raw=np.zeros(4, dtype=np.float32),
                source=source,
                valid=False,
                weight=0.0,
                confidence=0.0,
                safety_status=f"exception:{type(exc).__name__}",
            )

        verdict = self.supervisor.check(command)
        raw = ctbr_command_to_raw(command)
        finite = bool(np.all(np.isfinite(raw)))
        valid = bool(verdict.ok and finite)
        safety_status = "ok" if valid else (verdict.reason or "nonfinite_action")
        sat_frac = command_saturation_frac(command, self.max_body_rate)
        confidence = self._confidence(command, valid, sat_frac)
        weight = confidence if valid else 0.0
        return ExpertLabel(
            action_ctbr_raw=raw,
            source=str(command.get("source", source)),
            valid=valid,
            weight=weight,
            confidence=confidence,
            action_log_std_norm=self._log_std_norm(command),
            expert_cost=self._maybe_float(command.get("mppi_cost")),
            cost_margin=self._maybe_float(command.get("cost_margin")),
            saturation_frac=sat_frac,
            safety_status=safety_status,
            command=command,
        )

    def _confidence(self, command: dict, valid: bool, saturation_frac: float) -> float:
        if not valid:
            return 0.0
        conf = 1.0 - 0.5 * float(np.clip(saturation_frac, 0.0, 1.0))
        margin = self._maybe_float(command.get("cost_margin"))
        if margin is not None and np.isfinite(margin):
            conf *= float(np.clip(margin / (margin + 1.0), 0.05, 1.0))
        return float(np.clip(conf, 0.0, 1.0))

    def _log_std_norm(self, command: dict) -> np.ndarray:
        cov = command.get("action_cov_diag")
        if cov is None:
            return self.default_log_std_norm.copy()
        std_raw = np.sqrt(np.maximum(np.asarray(cov, dtype=np.float32), 1e-8))
        if self.action_low is not None and self.action_high is not None:
            std = normalize_std_bounds(std_raw, self.action_low, self.action_high)
        else:
            std = std_raw
        return np.log(np.maximum(std, 1e-4)).astype(np.float32)

    @staticmethod
    def _maybe_float(value) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

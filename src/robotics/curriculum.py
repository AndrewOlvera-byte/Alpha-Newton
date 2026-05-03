"""Stage-based course curriculum for Flightmare PPO.

Curriculum design follows the standard sparse-reward racing recipe: start with
a short course / fixed start gate / low gate noise so reward signal is dense
and BC-warm-started actors get immediate progress. Ramp `num_gates`, start-gate
randomization, and per-gate noise as the policy succeeds.

Configured under ``robotics.ppo.curriculum`` in the training YAML::

    curriculum:
      enabled: true
      advance_metric: success_rate     # success_rate|mean_gate_completion (or null)
      advance_threshold: 0.5           # advance early when this is reached
      stages:
        - until_iter: 400
          num_gates: 2
          random_start_gate: false
          fixed_gate_pos_noise: 0.05
          fixed_gate_yaw_noise: 0.0
        - until_iter: 1200
          num_gates: 4
          random_start_gate: true
          fixed_gate_pos_noise: 0.10
          fixed_gate_yaw_noise: 0.03
        - until_iter: 99999
          num_gates: 7
          random_start_gate: true
          fixed_gate_pos_noise: 0.15
          fixed_gate_yaw_noise: 0.05

Each stage may override any key passed to ``build_flightmare_env_config``
(``num_gates``, ``random_start_gate``, ``fixed_gate_pos_noise``,
``fixed_gate_yaw_noise``, ``gate_size``, ``gate_spacing_range``,
``gate_lateral_jitter``, ``gate_z_range``, ``gate_yaw_step``,
``gate_yaw_noise``, ``horizon``, ``terminate_on_gate_miss``,
``ent_coeff_override``, ...). Unknown keys are passed through.

A scalar ``ent_coeff_override`` is supported as a stage-local override of
``ppo.ent_coeff`` so early stages can use higher exploration without editing
the global value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Keys consumed by the trainer itself rather than the env builder.
_TRAINER_OVERRIDE_KEYS = {"ent_coeff_override"}


@dataclass
class CurriculumStage:
    until_iter: int
    overrides: dict[str, Any] = field(default_factory=dict)
    name: str = ""


class Curriculum:
    """Resolves which curriculum stage is active for a given iteration."""

    def __init__(
        self,
        stages: list[CurriculumStage],
        advance_metric: Optional[str] = None,
        advance_threshold: float = 1.0,
    ):
        if not stages:
            raise ValueError("Curriculum requires at least one stage.")
        # Sort defensively by until_iter.
        self.stages = sorted(stages, key=lambda s: s.until_iter)
        self.advance_metric = advance_metric
        self.advance_threshold = float(advance_threshold)
        self._active_idx = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "Curriculum":
        raw_stages = cfg.get("stages", []) or []
        if not raw_stages:
            raise ValueError("curriculum.stages must be a non-empty list when curriculum.enabled is true.")
        stages: list[CurriculumStage] = []
        for i, raw in enumerate(raw_stages):
            raw = dict(raw)
            until_iter = int(raw.pop("until_iter", 10**9))
            name = str(raw.pop("name", f"stage_{i}"))
            stages.append(CurriculumStage(until_iter=until_iter, overrides=raw, name=name))
        return cls(
            stages=stages,
            advance_metric=cfg.get("advance_metric"),
            advance_threshold=float(cfg.get("advance_threshold", 1.0)),
        )

    @property
    def active(self) -> CurriculumStage:
        return self.stages[self._active_idx]

    @property
    def active_index(self) -> int:
        return self._active_idx

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def update(self, iteration: int, last_stats: Optional[dict] = None) -> bool:
        """Advance to the next stage if the current one is exhausted.

        A stage is exhausted when ``iteration > until_iter`` OR (if
        ``advance_metric`` is set) the metric crosses ``advance_threshold``.
        Returns True iff the active stage changed (caller should rebuild env).
        """
        prev_idx = self._active_idx
        # Bump past expired stages.
        while (
            self._active_idx < len(self.stages) - 1
            and iteration > self.stages[self._active_idx].until_iter
        ):
            self._active_idx += 1
        # Optional early-advance based on rollout metric.
        if (
            self.advance_metric
            and last_stats is not None
            and self._active_idx < len(self.stages) - 1
        ):
            value = float(last_stats.get(self.advance_metric, 0.0))
            if value >= self.advance_threshold:
                self._active_idx += 1
        return self._active_idx != prev_idx

    def env_overrides(self) -> dict[str, Any]:
        """Stage overrides destined for the env builder (trainer keys removed)."""
        return {k: v for k, v in self.active.overrides.items() if k not in _TRAINER_OVERRIDE_KEYS}

    def trainer_overrides(self) -> dict[str, Any]:
        """Stage overrides consumed by the trainer (e.g. ent_coeff_override)."""
        return {k: v for k, v in self.active.overrides.items() if k in _TRAINER_OVERRIDE_KEYS}

    def describe(self) -> str:
        s = self.active
        body = ", ".join(f"{k}={v}" for k, v in s.overrides.items()) or "(no overrides)"
        return f"stage {self._active_idx + 1}/{len(self.stages)} '{s.name}' until_iter={s.until_iter}: {body}"

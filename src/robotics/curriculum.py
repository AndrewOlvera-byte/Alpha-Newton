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
          advance_after_iter: 200
          advance_patience: 3
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
#   ent_coeff_override  -> stage-local entropy coefficient
#   actor_lr_override   -> stage-local actor learning rate. Lets a stage run a
#                          critic-dominant "warmup" (tiny actor LR) right after a
#                          goal_mode change, so the value function can refit the
#                          new bootstrap targets before the policy is allowed to
#                          move on stale/garbage advantages.
_TRAINER_OVERRIDE_KEYS = {"ent_coeff_override", "actor_lr_override"}


@dataclass
class CurriculumStage:
    until_iter: int
    overrides: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    advance_metric: Optional[str] = None
    advance_threshold: Optional[float] = None
    advance_after_iter: int = 0
    advance_after_stage_iters: int = 0
    advance_patience: int = 1
    advance_on_timeout: bool = True
    duration_iters: Optional[int] = None


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
        self._advance_hits = 0
        self._stage_start_iter = 1

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
            stage_advance_metric = raw.pop("advance_metric", None)
            stage_advance_threshold = raw.pop("advance_threshold", None)
            advance_after_iter = int(raw.pop("advance_after_iter", 0) or 0)
            advance_after_stage_iters = int(raw.pop("advance_after_stage_iters", 0) or 0)
            advance_patience = max(1, int(raw.pop("advance_patience", 1) or 1))
            advance_on_timeout = bool(raw.pop("advance_on_timeout", True))
            duration_raw = raw.pop("duration_iters", None)
            duration_iters = None if duration_raw is None else max(1, int(duration_raw))
            stages.append(
                CurriculumStage(
                    until_iter=until_iter,
                    overrides=raw,
                    name=name,
                    advance_metric=stage_advance_metric,
                    advance_threshold=(
                        None if stage_advance_threshold is None else float(stage_advance_threshold)
                    ),
                    advance_after_iter=advance_after_iter,
                    advance_after_stage_iters=advance_after_stage_iters,
                    advance_patience=advance_patience,
                    advance_on_timeout=advance_on_timeout,
                    duration_iters=duration_iters,
                )
            )
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

        A stage is exhausted when its timeout is reached, unless
        ``advance_on_timeout`` is false. If ``advance_metric`` is set, a stage
        can also advance when the metric crosses ``advance_threshold`` for the
        configured patience window.
        Returns True iff the active stage changed (caller should rebuild env).
        """
        prev_idx = self._active_idx
        # Bump past expired stages. ``advance_on_timeout=False`` turns
        # ``until_iter`` into an advisory/logging boundary: the stage must then
        # advance via its metric gate instead of a wall-clock timeout.
        while (
            self._active_idx < len(self.stages) - 1
            and self._stage_timed_out(iteration, self.stages[self._active_idx])
        ):
            self._active_idx += 1
            self._advance_hits = 0
            self._stage_start_iter = int(iteration)
        # Optional early-advance based on rollout metric.
        stage = self.stages[self._active_idx]
        advance_metric = stage.advance_metric or self.advance_metric
        advance_threshold = (
            stage.advance_threshold if stage.advance_threshold is not None else self.advance_threshold
        )
        if advance_metric and last_stats is not None and self._active_idx < len(self.stages) - 1:
            value = float(last_stats.get(advance_metric, 0.0))
            elapsed = int(iteration) - int(self._stage_start_iter)
            absolute_ok = iteration >= int(stage.advance_after_iter)
            relative_ok = elapsed >= int(stage.advance_after_stage_iters)
            if absolute_ok and relative_ok and value >= float(advance_threshold):
                self._advance_hits += 1
            else:
                self._advance_hits = 0
            if self._advance_hits >= int(stage.advance_patience):
                self._active_idx += 1
                self._advance_hits = 0
                self._stage_start_iter = int(iteration)
        return self._active_idx != prev_idx

    def _stage_timed_out(self, iteration: int, stage: CurriculumStage) -> bool:
        if not bool(stage.advance_on_timeout):
            return False
        if stage.duration_iters is not None:
            return int(iteration) >= int(self._stage_start_iter) + int(stage.duration_iters)
        return int(iteration) > int(stage.until_iter)

    def env_overrides(self) -> dict[str, Any]:
        """Stage overrides destined for the env builder (trainer keys removed)."""
        return {k: v for k, v in self.active.overrides.items() if k not in _TRAINER_OVERRIDE_KEYS}

    def trainer_overrides(self) -> dict[str, Any]:
        """Stage overrides consumed by the trainer (e.g. ent_coeff_override)."""
        return {k: v for k, v in self.active.overrides.items() if k in _TRAINER_OVERRIDE_KEYS}

    def describe(self) -> str:
        s = self.active
        body = ", ".join(f"{k}={v}" for k, v in s.overrides.items()) or "(no overrides)"
        gate = ""
        metric = s.advance_metric or self.advance_metric
        if metric:
            threshold = s.advance_threshold if s.advance_threshold is not None else self.advance_threshold
            gate = (
                f", advance={metric}>={threshold} after_iter={s.advance_after_iter} "
                f"after_stage_iters={s.advance_after_stage_iters} "
                f"patience={s.advance_patience}"
            )
        timeout = ""
        if not s.advance_on_timeout:
            timeout = ", no_timeout_advance"
        elif s.duration_iters is not None:
            timeout = f", duration_iters={s.duration_iters}"
        return (
            f"stage {self._active_idx + 1}/{len(self.stages)} '{s.name}' "
            f"until_iter={s.until_iter}{timeout}{gate}: {body}"
        )

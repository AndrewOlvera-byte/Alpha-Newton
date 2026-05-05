"""Unified Flightmare actor registration.

A single ``flightmare_actor`` builder configurable via YAML for all action
spaces (waypoint, ctbr, motor) and both BC and PPO defaults. Earlier versions
shipped 12 near-identical builders (``flightmare_waypoint_bc``,
``flightmare_motor_ppo_state``, ...) which are now consolidated.

Required config keys
--------------------
``type: flightmare_actor``
``mode: bc | ppo``
``action_type: waypoint | ctbr | motor``

Mode defaults
-------------
``bc``  -> with_critic=False, state_dependent_std=True,  bc_loss_type=gnll
``ppo`` -> with_critic=True,  state_dependent_std=False, bc_loss_type=gnll

PPO builds optionally take ``bc_checkpoint`` to warm-start from a BC run.

Action spaces
-------------
``waypoint`` (Loquercio Sci. Robotics 2021, Swift Nature 2023):
    4-dim (Δx_body, Δy_body, Δz_body, v_norm). Decoded by
    ``WaypointLQRController`` -> SE3 inner loop -> CTBR -> Flightmare's C++
    motor mixer.

``ctbr`` (Foehn et al., Kaufmann et al.):
    4-dim (T_norm, ω_x, ω_y, ω_z). The de-facto low-level interface for
    real autonomous drone racing. Sent directly to ``env.stepCtbr``.

``motor``:
    4-dim per-rotor normalized thrust. Maximum actuator authority, hardest
    credit assignment. Sent directly to ``env.stepMotor``.

Backward-compatibility shim
---------------------------
The old per-action / per-mode names are kept as registry aliases so existing
configs that have not been migrated still build. New work should use
``flightmare_actor`` exclusively.
"""
from __future__ import annotations

import inspect

from src.core.registry import register
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import (
    MLPFusionGaussianExpertActor,
)


_ACTOR_PARAMS = set(inspect.signature(MLPFusionGaussianExpertActor.__init__).parameters)
_ACTOR_PARAMS.discard("self")
_ACTOR_PARAMS.update({"bc_checkpoint", "mode", "action_type"})


def _strip(kwargs: dict) -> dict:
    """Drop keys the actor doesn't accept (e.g. shared base-config fields like
    d_model that target other architectures), keeping the builder robust to
    cross-arch config merges."""
    cfg = {k: v for k, v in kwargs.items() if k in _ACTOR_PARAMS}
    cfg.pop("type", None)
    return cfg


def _build(mode: str, action_type: str, kwargs: dict):
    cfg = _strip(kwargs)
    cfg.pop("mode", None)
    cfg.pop("action_type", None)
    bc_ckpt = cfg.pop("bc_checkpoint", None)

    if action_type not in {"waypoint", "ctbr", "motor"}:
        raise ValueError(
            f"flightmare_actor.action_type must be one of 'waypoint', 'ctbr', 'motor', "
            f"got {action_type!r}."
        )
    cfg.setdefault("action_dim", 4)
    cfg.setdefault("use_vision", False)

    if mode == "bc":
        cfg.setdefault("with_critic", False)
        cfg.setdefault("state_dependent_std", True)
        cfg.setdefault("bc_loss_type", "gnll")
    elif mode == "ppo":
        cfg.setdefault("with_critic", True)
        cfg.setdefault("state_dependent_std", False)
        cfg.setdefault("bc_loss_type", "gnll")
    else:
        raise ValueError(f"flightmare_actor.mode must be 'bc' or 'ppo', got {mode!r}.")

    if bc_ckpt is not None:
        if mode != "ppo":
            raise ValueError("bc_checkpoint is only valid for mode='ppo'.")
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_actor")
def build_flightmare_actor(**kwargs):
    mode = kwargs.get("mode")
    action_type = kwargs.get("action_type")
    if mode is None or action_type is None:
        raise ValueError(
            "flightmare_actor requires 'mode' (bc|ppo) and 'action_type' "
            "(waypoint|ctbr|motor) in the architecture config."
        )
    return _build(str(mode), str(action_type), kwargs)


# ---------------------------------------------------------------------------
# Deprecated aliases. Existing configs that still reference the long names are
# routed through the unified builder. New work should use ``flightmare_actor``.
# ---------------------------------------------------------------------------
def _alias(name: str, mode: str, action_type: str):
    @register("architecture", name)
    def _builder(**kwargs):
        return _build(mode, action_type, kwargs)
    _builder.__name__ = f"build_{name}"
    return _builder


for _action in ("waypoint", "ctbr", "motor"):
    for _mode in ("bc", "ppo"):
        _alias(f"flightmare_{_action}_{_mode}", _mode, _action)
        _alias(f"flightmare_{_action}_{_mode}_state", _mode, _action)


# Generic MLP actor registration kept for arbitrary action_dim experiments.
@register("architecture", "flightmare_mlp_actor")
def build_flightmare_mlp_actor(**kwargs):
    return MLPFusionGaussianExpertActor(**_strip(kwargs))

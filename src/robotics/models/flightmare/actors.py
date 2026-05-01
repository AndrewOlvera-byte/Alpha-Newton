"""Registered Flightmare drone actor architectures.

Two action spaces are wired up separately so ablations can swap them via the
config ``architecture.type`` field without code changes:

  * ``flightmare_waypoint_bc`` / ``flightmare_waypoint_ppo``
        Action = (Δx_body, Δy_body, Δz_body, v_norm), 4-dim. High-level
        waypoint+speed reference, ~10-30 Hz, easy to learn from MPC traces.
        Used in "High-Speed Flight in the Wild" (Loquercio et al., Sci.
        Robotics 2021) and Swift (Kaufmann et al., Nature 2023).

  * ``flightmare_ctbr_bc`` / ``flightmare_ctbr_ppo``
        Action = (T_norm, ω_x, ω_y, ω_z), 4-dim. Collective thrust + body
        rates, the de-facto low-level interface for real autonomous drone
        racing (50-100 Hz). Harder to BC-fit but enables tighter trajectories
        and is what onboard inner-loop controllers consume directly.

  * ``flightmare_motor_bc`` / ``flightmare_motor_ppo``
        Action = four normalized per-rotor thrust commands, 4-dim. This is
        the lowest-level ablation: full actuator authority, but harder credit
        assignment and more reliance on accurate motor/dynamics modeling.

A generic builder ``flightmare_mlp_actor`` is also registered for arbitrary
``action_dim``, e.g. 4-rotor individual-motor control (4-dim) or higher-level
formulations (e.g. 6-dim with attitude).

BC builds default to: no critic, state-dependent log_std (for proper Gaussian
NLL), gnll loss. PPO builds default to: critic enabled, state-independent
learnable log_std parameter (PPO-stable per SB3/OpenAI Baselines convention).
PPO builds optionally take ``bc_checkpoint`` to warm-start from a BC run.
"""
from __future__ import annotations

import inspect

from src.core.registry import register
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import (
    MLPFusionGaussianExpertActor,
)


_ACTOR_PARAMS = set(inspect.signature(MLPFusionGaussianExpertActor.__init__).parameters)
_ACTOR_PARAMS.discard("self")
_ACTOR_PARAMS.add("bc_checkpoint")  # consumed by PPO builders before construction


def _strip(kwargs: dict) -> dict:
    """Drop keys the actor doesn't accept (e.g. shared base-config fields like
    d_model that target other architectures), keeping the builder robust to
    cross-arch config merges."""
    cfg = {k: v for k, v in kwargs.items() if k in _ACTOR_PARAMS}
    cfg.pop("type", None)
    return cfg


def _bc_defaults(cfg: dict, action_dim: int) -> dict:
    cfg.setdefault("action_dim", action_dim)
    cfg.setdefault("with_critic", False)
    cfg.setdefault("state_dependent_std", True)
    cfg.setdefault("bc_loss_type", "gnll")
    return cfg


def _ppo_defaults(cfg: dict, action_dim: int) -> dict:
    cfg.setdefault("action_dim", action_dim)
    cfg.setdefault("with_critic", True)
    cfg.setdefault("state_dependent_std", False)
    cfg.setdefault("bc_loss_type", "gnll")
    return cfg


@register("architecture", "flightmare_mlp_actor")
def build_flightmare_mlp_actor(**kwargs):
    return MLPFusionGaussianExpertActor(**_strip(kwargs))


@register("architecture", "flightmare_waypoint_bc")
def build_flightmare_waypoint_bc(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_waypoint_ppo")
def build_flightmare_waypoint_ppo(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_ctbr_bc")
def build_flightmare_ctbr_bc(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_ctbr_ppo")
def build_flightmare_ctbr_ppo(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_motor_bc")
def build_flightmare_motor_bc(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_motor_ppo")
def build_flightmare_motor_ppo(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


# ---------------------------------------------------------------------------
# State-only variants (no vision encoder). Used for vectorized RL at scale
# where running many Unity instances is infeasible. Inputs: proprio state +
# privileged mission vector (concatenated into ``state`` upstream).
# ---------------------------------------------------------------------------
@register("architecture", "flightmare_waypoint_bc_state")
def build_flightmare_waypoint_bc_state(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    cfg.setdefault("use_vision", False)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_waypoint_ppo_state")
def build_flightmare_waypoint_ppo_state(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    cfg.setdefault("use_vision", False)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_ctbr_bc_state")
def build_flightmare_ctbr_bc_state(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    cfg.setdefault("use_vision", False)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_ctbr_ppo_state")
def build_flightmare_ctbr_ppo_state(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    cfg.setdefault("use_vision", False)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_motor_bc_state")
def build_flightmare_motor_bc_state(**kwargs):
    cfg = _bc_defaults(_strip(kwargs), action_dim=4)
    cfg.setdefault("use_vision", False)
    return MLPFusionGaussianExpertActor(**cfg)


@register("architecture", "flightmare_motor_ppo_state")
def build_flightmare_motor_ppo_state(**kwargs):
    cfg = _strip(kwargs)
    bc_ckpt = cfg.pop("bc_checkpoint", None)
    cfg = _ppo_defaults(cfg, action_dim=4)
    cfg.setdefault("use_vision", False)
    if bc_ckpt is not None:
        return MLPFusionGaussianExpertActor.from_bc_checkpoint(bc_ckpt, cfg)
    return MLPFusionGaussianExpertActor(**cfg)

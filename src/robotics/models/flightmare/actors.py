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

A generic builder ``flightmare_mlp_actor`` is also registered for arbitrary
``action_dim``, e.g. 4-rotor individual-motor control (4-dim) or higher-level
formulations (e.g. 6-dim with attitude).

BC builds default to: no critic, state-dependent log_std (for proper Gaussian
NLL), gnll loss. PPO builds default to: critic enabled, state-independent
learnable log_std parameter (PPO-stable per SB3/OpenAI Baselines convention).
PPO builds optionally take ``bc_checkpoint`` to warm-start from a BC run.
"""
from __future__ import annotations

from src.core.registry import register
from src.robotics.models.flightmare.MLPFusionGaussianExpertActor import (
    MLPFusionGaussianExpertActor,
)


def _strip(kwargs: dict) -> dict:
    cfg = dict(kwargs)
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

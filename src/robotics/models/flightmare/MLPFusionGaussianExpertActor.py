"""End-to-end visuomotor actor for Flightmare drone BC and PPO.

Combines:
  * ``MLPFusion``: frozen pretrained vision encoder + state/action embed +
    fused MLP trunk producing a compact feature vector.
  * ``GaussianActionExpert``: Gaussian mean/log-std action head plus an
    optional value head for PPO.

Two output spaces are wired up via ``actors.py``:
  - waypoint+speed (4-dim): high-level, ~10-30 Hz; matches "Learning
    High-Speed Flight in the Wild" (Loquercio et al. 2021) and Swift
    (Kaufmann et al. 2023).
  - collective-thrust + body-rates (4-dim): low-level, 50-100 Hz; the
    standard real-drone control interface used in autonomous racing.

The same module is used for BC (no critic, NLL or MSE) and for PPO (critic
enabled, identical actor weights). ``from_bc_checkpoint`` instantiates a
PPO-ready clone from a BC weight file; the freshly added critic and
state-independent log_std are loaded with strict=False.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.robotics.models.flightmare.GaussianActionExpert import GaussianActionExpert
from src.robotics.models.flightmare.MLP import MLPFusion


class MLPFusionGaussianExpertActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        backbone: str = "resnet18",
        img_size: int = 224,
        pretrained: bool = True,
        freeze_vision: bool = True,
        n_cameras: int = 1,
        include_prev_action: bool = True,
        state_hidden_dim: int = 128,
        state_embed_dim: int = 128,
        trunk_hidden_dim: int = 256,
        trunk_depth: int = 2,
        dropout: float = 0.0,
        channels_last: bool = True,
        head_hidden_dim: int = 256,
        head_depth: int = 1,
        state_dependent_std: bool = True,
        init_log_std: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        with_critic: bool = False,
        critic_hidden_dim: int = 256,
        critic_depth: int = 2,
        action_clip: float = 5.0,
        bc_loss_type: str = "gnll",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bc_loss_type = bc_loss_type
        self.action_clip = action_clip

        self.fusion = MLPFusion(
            state_dim=state_dim,
            action_dim=action_dim,
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            freeze_vision=freeze_vision,
            n_cameras=n_cameras,
            include_prev_action=include_prev_action,
            state_hidden_dim=state_hidden_dim,
            state_embed_dim=state_embed_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            trunk_depth=trunk_depth,
            dropout=dropout,
            channels_last=channels_last,
        )
        self.head = GaussianActionExpert(
            feature_dim=trunk_hidden_dim,
            action_dim=action_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            state_dependent_std=state_dependent_std,
            init_log_std=init_log_std,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            with_critic=with_critic,
            critic_hidden_dim=critic_hidden_dim,
            critic_depth=critic_depth,
            action_clip=action_clip,
        )

    @property
    def has_critic(self) -> bool:
        return self.head.has_critic

    @staticmethod
    def _squeeze_chunk(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3 and target.shape[1] == 1:
            return target.squeeze(1)
        return target

    def forward(self, batch: dict) -> dict:
        feature = self.fusion(batch)
        mu, log_std = self.head(feature)

        if "action" not in batch:
            return {"mu": mu, "log_std": log_std}

        target = self._squeeze_chunk(batch["action"])
        if self.bc_loss_type == "gnll":
            per_elem = self.head.nll(mu, log_std, target)
            per_sample = per_elem.mean(dim=-1)
        elif self.bc_loss_type == "mse":
            per_sample = F.mse_loss(mu, target, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"unknown bc_loss_type {self.bc_loss_type!r}")

        return {"loss": per_sample.mean(), "_loss_per_sample": per_sample.detach()}

    @torch.no_grad()
    def predict(self, batch: dict, **_) -> torch.Tensor:
        feature = self.fusion(batch)
        mu, _ = self.head(feature)
        return mu.clamp(-self.action_clip, self.action_clip)

    def act(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feature = self.fusion(batch)
        action, log_prob, _, _ = self.head.sample(feature)
        value = self.head.value(feature) if self.head.has_critic else None
        return action, log_prob, value

    def evaluate_actions(
        self, batch: dict, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feature = self.fusion(batch)
        log_prob, entropy, _, _ = self.head.evaluate(feature, actions)
        value = self.head.value(feature) if self.head.has_critic else None
        return log_prob, entropy, value

    @classmethod
    def from_bc_checkpoint(
        cls,
        checkpoint_path: str,
        arch_cfg: dict,
        with_critic: bool = True,
        **overrides,
    ) -> "MLPFusionGaussianExpertActor":
        cfg = {**arch_cfg, **overrides}
        cfg.pop("type", None)
        cfg.pop("bc_checkpoint", None)
        cfg["with_critic"] = with_critic
        model = cls(**cfg)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        return model

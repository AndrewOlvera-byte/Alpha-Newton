"""Actor-Critic wrapper for PPO fine-tuning of BC-pretrained VLA policies.

Shares the VisionIntentBCBackbone between actor and critic to halve
the forward pass cost. The actor head is the BC-pretrained MLPController;
the critic is a fresh ValueHead reading from the intent token.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.core.registry import register, build
from src.robotics.models.robosuite.VLA_MLP import VLAMLPPolicy


class ValueHead(nn.Module):
    """Scalar value function from the backbone's intent token."""

    def __init__(self, d_model: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, intent_token: torch.Tensor) -> torch.Tensor:
        return self.net(intent_token).squeeze(-1)  # [B]


class VLAMLPActorCritic(nn.Module):
    """Actor-Critic wrapping a BC-pretrained VLAMLPPolicy for PPO.

    The backbone and actor head come from the BC checkpoint.
    A fresh ValueHead and learnable log_std are added for PPO.
    """

    def __init__(self, actor: VLAMLPPolicy, value_hidden_dim: int = 512):
        super().__init__()
        self.backbone = actor.backbone
        self.actor_head = actor.controller
        self.action_dim = actor.controller.action_dim

        d_model = actor.backbone.d_model
        self.value_head = ValueHead(d_model, value_hidden_dim)

        # Learnable per-dim exploration std.
        # If BC model has gnll log_var head, initialize from its bias (learned variance).
        init_log_std = -1.0  # conservative default
        if hasattr(actor.controller, 'log_var_head'):
            init_log_std = 0.5 * actor.controller.log_var_head.bias.data.mean().item()
        self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std))

    def _get_mu(self, encoded: dict) -> torch.Tensor:
        """Extract mean action [B, action_dim] from controller output."""
        ctrl_out = self.actor_head(encoded["decoded_sequence"], encoded["intent_token"])
        if isinstance(ctrl_out, tuple):
            mu = ctrl_out[0]
        else:
            mu = ctrl_out
        return mu[:, 0]  # [B, action_dim] (first action in chunk)

    def act(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For rollout collection. Returns (action, log_prob, value)."""
        encoded = self.backbone(batch)
        mu = self._get_mu(encoded)
        dist = Normal(mu, self.log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)  # [B]
        value = self.value_head(encoded["intent_token"])  # [B]
        return action.clamp(-5.0, 5.0), log_prob, value

    def evaluate_actions(
        self, batch: dict, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update. Returns (log_prob, entropy, value)."""
        encoded = self.backbone(batch)
        mu = self._get_mu(encoded)
        dist = Normal(mu, self.log_std.exp())
        log_prob = dist.log_prob(actions).sum(-1)  # [B]
        entropy = dist.entropy().sum(-1)  # [B]
        value = self.value_head(encoded["intent_token"])  # [B]
        return log_prob, entropy, value

    @torch.no_grad()
    def predict(self, batch: dict, **kwargs) -> torch.Tensor:
        """Deterministic inference (mean action). BC-compatible."""
        encoded = self.backbone(batch)
        mu = self._get_mu(encoded)
        return mu.clamp(-5.0, 5.0)

    @classmethod
    def from_bc_checkpoint(
        cls,
        checkpoint_path: str,
        arch_cfg: dict,
        value_hidden_dim: int = 512,
    ) -> "VLAMLPActorCritic":
        """Load a BC checkpoint and wrap it as an actor-critic."""
        bc_model = build("architecture", **arch_cfg)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        bc_model.load_state_dict(state_dict)
        return cls(actor=bc_model, value_hidden_dim=value_hidden_dim)


@register("architecture", "vla_mlp_ppo")
def build_vla_mlp_actor_critic(**kwargs):
    kwargs.pop("type", None)
    checkpoint_path = kwargs.pop("bc_checkpoint")
    value_hidden_dim = kwargs.pop("value_hidden_dim", 512)
    return VLAMLPActorCritic.from_bc_checkpoint(checkpoint_path, kwargs, value_hidden_dim)

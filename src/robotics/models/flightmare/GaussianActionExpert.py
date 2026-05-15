"""Gaussian MLP action head + optional value head for Flightmare drone control.

Design (PPO/SAC standard practice; SB3 default):
  * Mean head: small MLP over the trunk feature, orthogonal init with gain
    0.01 on the final linear (Mnih et al. 2015) so initial actions stay tiny
    and downstream RL/BC training is stable from random init.
  * Log-std options:
      - state-dependent ("heteroscedastic"): a Linear(feature -> action_dim)
        producing per-sample log_std. Required to fit a true Gaussian-NLL
        BC objective (otherwise NLL collapses to MSE up to a constant).
      - state-independent: a learnable nn.Parameter shared across states.
        More stable for downstream PPO (matches SB3 / OpenAI Baselines).
    Both modes clamp log_std to a sensible range.
  * Critic: optional value head reading the *same* trunk feature, so
    actor-critic forward = single backbone pass. Orthogonal init with
    gain 1.0 on the value linear (standard).

Used by ``MLPFusionGaussianExpertActor`` for both BC NLL and PPO.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


_LOG_2PI = math.log(2.0 * math.pi)


def _orthogonal_(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class GaussianActionExpert(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 1,
        state_dependent_std: bool = True,
        init_log_std: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        with_critic: bool = False,
        critic_hidden_dim: int = 256,
        critic_depth: int = 2,
        action_clip: float = 5.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_clip = action_clip

        head_layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(depth):
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            head_layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.head_trunk = nn.Sequential(*head_layers) if depth > 0 else nn.Identity()
        self._head_out_dim = in_dim

        self.mean = nn.Linear(in_dim, action_dim)
        _orthogonal_(self.mean, gain=0.01)

        if state_dependent_std:
            self.log_std_head = nn.Linear(in_dim, action_dim)
            _orthogonal_(self.log_std_head, gain=0.01)
            nn.init.constant_(self.log_std_head.bias, init_log_std)
            self.log_std_param = None
        else:
            self.log_std_head = None
            self.log_std_param = nn.Parameter(torch.full((action_dim,), init_log_std))

        if with_critic:
            critic_layers: list[nn.Module] = []
            cin = feature_dim
            for _ in range(critic_depth):
                critic_layers.append(nn.Linear(cin, critic_hidden_dim))
                critic_layers.append(nn.SiLU())
                cin = critic_hidden_dim
            value_out = nn.Linear(cin, 1)
            _orthogonal_(value_out, gain=1.0)
            critic_layers.append(value_out)
            self.critic = nn.Sequential(*critic_layers)
        else:
            self.critic = None

    @property
    def has_critic(self) -> bool:
        return self.critic is not None

    def _mu_log_std(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.head_trunk(feature)
        raw_mu = self.mean(h)
        if self.action_clip > 0:
            bounded_mu = raw_mu.clamp(-float(self.action_clip), float(self.action_clip))
            mu = raw_mu + (bounded_mu - raw_mu).detach()
        else:
            mu = raw_mu
        if self.state_dependent_std:
            log_std = self.log_std_head(h)
        else:
            log_std = self.log_std_param.expand_as(mu)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mu_log_std(feature)

    def value(self, feature: torch.Tensor) -> torch.Tensor:
        if self.critic is None:
            raise RuntimeError(
                "Critic disabled; instantiate with with_critic=True for PPO."
            )
        return self.critic(feature).squeeze(-1)

    def nll(
        self, mu: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-element diagonal-Gaussian NLL (sum over action dim afterwards)."""
        inv_var = torch.exp(-2.0 * log_std)
        return 0.5 * (((target - mu) ** 2) * inv_var + 2.0 * log_std + _LOG_2PI)

    def sample(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._mu_log_std(feature)
        dist = Normal(mu, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mu, log_std

    def evaluate(
        self, feature: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._mu_log_std(feature)
        dist = Normal(mu, log_std.exp())
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, mu, log_std

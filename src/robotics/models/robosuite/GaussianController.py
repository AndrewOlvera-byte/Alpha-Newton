import math

import torch
import torch.nn as nn

from src.robotics.models.MLPController import IntentCrossAttention


class GaussianController(nn.Module):
    """Diagonal-Gaussian controller that predicts mean actions deterministically at inference."""

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        hidden_dim: int = 512,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        action_chunk_size: int = 1,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.cross_attention = IntentCrossAttention(d_model, n_heads, d_ff, dropout=dropout)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        out_dim = action_chunk_size * action_dim
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.log_std_head = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -1.5)

    def forward(self, sequence_tokens: torch.Tensor, intent_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.cross_attention(sequence_tokens, intent_token)
        hidden = self.trunk(fused)
        mean = self.mean_head(hidden).view(hidden.shape[0], self.action_chunk_size, self.action_dim)
        log_std = self.log_std_head(hidden).view(hidden.shape[0], self.action_chunk_size, self.action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def nll(self, target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-2.0 * log_std)
        return 0.5 * (((target - mean) ** 2) * inv_var + 2.0 * log_std + math.log(2.0 * math.pi))

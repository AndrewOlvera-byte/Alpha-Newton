import torch.nn as nn


class StateEncoder(nn.Module):
    """MLP: state_dim → d_model. Processes [B, H, state_dim] → [B, H, d_model]."""

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state):
        return self.net(state)

import torch.nn as nn


class ActionEncoder(nn.Module):
    """MLP: action_dim → d_model. Processes [B, H, action_dim] → [B, H, d_model]."""

    def __init__(self, action_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, actions):
        return self.net(actions)

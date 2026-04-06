import torch
import torch.nn as nn


class IntentCrossAttention(nn.Module):
    """Fuse a latent intent token with the decoded history sequence."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.intent_norm = nn.LayerNorm(d_model)
        self.sequence_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, sequence_tokens: torch.Tensor, intent_token: torch.Tensor) -> torch.Tensor:
        query = self.intent_norm(intent_token).unsqueeze(1)
        source = self.sequence_norm(sequence_tokens)
        attended, _ = self.cross_attn(query=query, key=source, value=source, need_weights=False)
        fused = intent_token.unsqueeze(1) + self.dropout(attended)
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused.squeeze(1)


class MLPController(nn.Module):
    """Deterministic controller head for BC warm starts."""

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        hidden_dim: int = 512,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        action_chunk_size: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
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
        self.output = nn.Linear(hidden_dim, action_chunk_size * action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequence_tokens: torch.Tensor, intent_token: torch.Tensor) -> torch.Tensor:
        fused = self.cross_attention(sequence_tokens, intent_token)
        hidden = self.trunk(fused)
        action = self.output(hidden)
        return action.view(action.shape[0], self.action_chunk_size, self.action_dim)

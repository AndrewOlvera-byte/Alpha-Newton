import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.robotics.models.RMSNorm import RMSNorm


class MHSA(nn.Module):
    """Multi-Head Self-Attention using F.scaled_dot_product_attention (SDPA).

    Automatically dispatches to FlashAttention / memory-efficient attention
    when available.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, H, N, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → MHSA → residual → RMSNorm → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformerDecoder(nn.Module):
    """Stack of pre-norm transformer blocks with optional gradient checkpointing."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x)

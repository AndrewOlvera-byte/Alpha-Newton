"""Attention pooling for vision tokens (Perceiver/Flamingo style).

Learnable query tokens cross-attend to ViT patch tokens, compressing
256 patches down to a fixed number of tokens while preserving spatial
information better than average pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Cross-attention pooling: n_queries learnable tokens attend to input patches.

    Input:  [B, n_patches, d_in]
    Output: [B, n_queries, d_in]
    """

    def __init__(self, d_model: int, n_queries: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_patches, d_model] → [B, n_queries, d_model]."""
        B = x.shape[0]

        q = self.q_proj(self.queries.expand(B, -1, -1))
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, self.n_queries, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, x.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, self.n_queries, -1)
        return self.norm(self.out_proj(out))

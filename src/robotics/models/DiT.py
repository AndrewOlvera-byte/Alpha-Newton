import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar timesteps → [B, dim]."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: generates scale/shift from conditioning."""

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        # Initialize near-identity: gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:d_model] = 1.0  # gamma init

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D], cond: [B, cond_dim] → [B, N, D]."""
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return gamma * self.norm(x) + beta


class DiTBlock(nn.Module):
    """DiT block: AdaLN → MHSA → residual → AdaLN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.adaln1 = AdaLN(d_model, cond_dim)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # MHSA with AdaLN pre-norm
        h = self.adaln1(x, cond)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(attn_out)

        # FFN with AdaLN pre-norm
        x = x + self.ffn(self.adaln2(x, cond))
        return x


class DiTActionHead(nn.Module):
    """DiT-based action head with flow matching.

    Takes a noisy action + conditioning → predicts velocity field.
    """

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model

        # Project noisy action to token
        self.action_proj = nn.Linear(action_dim, d_model)

        # Timestep embedding
        self.t_emb = SinusoidalTimestepEmbedding(cond_dim)

        # Conditioning fusion: c + t_emb → combined conditioning
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_ff, cond_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final projection: token → velocity
        self.final_adaln = AdaLN(d_model, cond_dim)
        self.out_proj = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: [B, action_dim]
            timestep: [B] in [0, 1]
            conditioning: [B, cond_dim] from transformer readout
        Returns:
            velocity: [B, action_dim]
        """
        # Fuse conditioning with timestep embedding
        t_emb = self.t_emb(timestep)
        cond = self.cond_fuse(torch.cat([conditioning, t_emb], dim=-1))

        # Project action to token: [B, 1, d_model]
        x = self.action_proj(noisy_action).unsqueeze(1)

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final projection to velocity
        x = self.final_adaln(x, cond)
        velocity = self.out_proj(x.squeeze(1))
        return velocity

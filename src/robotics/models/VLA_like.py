"""
Visuomotor Policy — DiT Flow-Matching Architecture.

DINOv2 ViT-S/14 (frozen) → Causal Transformer Decoder → DiT + Flow Matching → 7-DoF Actions.

Registered as architecture type "vla_flow_matching".
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import register
from src.robotics.models.ViT import DINOv2Encoder
from src.robotics.models.StateEncoder import StateEncoder
from src.robotics.models.ActionEncoder import ActionEncoder
from src.robotics.models.DecoderTransformer import CausalTransformerDecoder
from src.robotics.models.DiT import DiTActionHead


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Fixed sinusoidal positional encoding [seq_len, d_model]."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    angles = pos / (10000.0 ** (dim / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return pe


def build_block_causal_mask(group_sizes: list, device: torch.device) -> torch.Tensor:
    """Block-causal attention mask.

    Tokens in group i attend to groups 0..i. Returns a **bool** mask
    [1, 1, total, total] for broadcasting over B and H dims.
    Convention: True = can attend, False = blocked.

    Using bool mask allows PyTorch SDPA to dispatch to FlashAttention
    (float -inf masks force fallback to efficient_attention/math backend).
    """
    total = sum(group_sizes)
    # Start fully blocked, then open attend regions
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)

    row_start = 0
    for i, size_i in enumerate(group_sizes):
        col_start = 0
        for j in range(i + 1):
            size_j = group_sizes[j]
            mask[row_start : row_start + size_i, col_start : col_start + size_j] = True
            col_start += size_j
        row_start += size_i

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


class VLAPolicy(nn.Module):
    """Full visuomotor policy with flow-matching action head.

    forward()  → training: returns {"loss": flow_matching_loss}
    predict()  → inference: Euler ODE sampling from noise to action
    encode()   → observation history to conditioning vector
    """

    def __init__(
        self,
        # Architecture dims
        d_model: int = 384,
        n_transformer_layers: int = 6,
        n_transformer_heads: int = 6,
        d_ff: int = 1536,
        d_dit: int = 256,
        n_dit_layers: int = 4,
        n_dit_heads: int = 4,
        d_dit_ff: int = 1024,
        dropout: float = 0.1,
        # Task params
        action_dim: int = 7,
        obs_dim: int = 19,
        n_cameras: int = 2,
        history_length: int = 3,
        # ViT params
        vit_model: str = "facebook/dinov2-small",
        vit_img_size: int = 224,
        freeze_vit: bool = True,
        # Vision token pooling: pool ViT patch tokens down to this many per camera.
        # DINOv2 ViT-S/14 @ 224px → 256 patches. Pooling to 64 reduces sequence
        # length from 1543 to 391 tokens (4× speedup in attention).
        # Set to 0 to disable pooling and use all patches.
        vis_tokens_per_camera: int = 64,
        # Flow matching
        num_flow_steps: int = 10,
        # Training
        gradient_checkpointing: bool = False,
        encoder_hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.history_length = history_length
        self.n_cameras = n_cameras
        self.num_flow_steps = num_flow_steps

        # === Vision encoder (frozen DINOv2) ===
        self.vit = DINOv2Encoder(model_name=vit_model, img_size=vit_img_size, freeze=freeze_vit)
        vit_dim = self.vit.d_model  # 384 for ViT-S

        # === Vision token pooling (AdaptiveAvgPool over patch sequence) ===
        # Pools raw_patches → vis_tokens_per_camera tokens, then projects dim if needed.
        raw_patches = self.vit.n_patches  # 256 for 224×224 with patch 14
        if vis_tokens_per_camera > 0 and vis_tokens_per_camera < raw_patches:
            self.vis_pool = nn.AdaptiveAvgPool1d(vis_tokens_per_camera)
            self.n_patches = vis_tokens_per_camera
        else:
            self.vis_pool = None
            self.n_patches = raw_patches

        # Project ViT dim → d_model (identity if same dim)
        if vit_dim != d_model:
            self.vis_proj = nn.Linear(vit_dim, d_model, bias=False)
        else:
            self.vis_proj = nn.Identity()

        assert vit_dim == d_model or self.vis_proj is not None, (
            f"DINOv2 output dim {vit_dim} != d_model {d_model}. "
            f"Add vis_proj linear layer or set d_model={vit_dim}."
        )

        # === State & action encoders ===
        self.state_encoder = StateEncoder(obs_dim, encoder_hidden_dim, d_model)
        self.action_encoder = ActionEncoder(action_dim, encoder_hidden_dim, d_model)

        # === Learnable [READ] token ===
        self.readout_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # === Causal transformer decoder ===
        self.transformer = CausalTransformerDecoder(
            d_model=d_model,
            n_layers=n_transformer_layers,
            n_heads=n_transformer_heads,
            d_ff=d_ff,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        # === Readout projection → conditioning vector ===
        self.cond_dim = d_dit
        self.readout_proj = nn.Sequential(
            nn.Linear(d_model, d_dit),
            nn.SiLU(),
            nn.LayerNorm(d_dit),
        )

        # === DiT action head ===
        self.dit_head = DiTActionHead(
            action_dim=action_dim,
            d_model=d_dit,
            n_layers=n_dit_layers,
            n_heads=n_dit_heads,
            d_ff=d_dit_ff,
            cond_dim=d_dit,
            dropout=dropout if dropout < 0.1 else 0.0,  # lighter dropout for DiT
        )

        # === Precompute sequence structure ===
        n_vision = n_cameras * self.n_patches
        tokens_per_group = n_vision + 2  # vision + state + action
        self.tokens_per_group = tokens_per_group
        self.total_tokens = tokens_per_group * history_length + 1  # +1 READ
        self.group_sizes = [tokens_per_group] * history_length + [1]

        # Cache attention mask (registered as buffer, moves with model)
        mask = build_block_causal_mask(self.group_sizes, device=torch.device("cpu"))
        self.register_buffer("attn_mask", mask, persistent=False)

        # Cache positional encoding
        pe = sinusoidal_positional_encoding(self.total_tokens, d_model, torch.device("cpu"))
        self.register_buffer("pos_enc", pe, persistent=False)

    def encode(self, batch: dict) -> torch.Tensor:
        """Encode observation history → conditioning vector [B, cond_dim].

        batch keys:
            images:      {cam_name: [B, H, C, h, w]}
            state:       [B, H, state_dim]
            prev_actions: [B, H, action_dim]
        """
        images_dict = batch["images"]
        state = batch["state"]           # [B, H, obs_dim]
        prev_actions = batch["prev_actions"]  # [B, H, action_dim]
        B = state.shape[0]
        H = self.history_length

        # --- Encode images through DINOv2 ---
        # Stack all cameras and history frames for a single batched ViT forward.
        # We batch all cameras × history frames together to maximise GPU utilisation.
        cam_keys = list(images_dict.keys())
        n_cams = len(cam_keys)

        # Check if batch contains precomputed features (shape [..., n_patches, vit_dim])
        # instead of raw images (shape [..., C, H, W]).
        first_val = images_dict[cam_keys[0]]
        using_precomputed = first_val.ndim == 4 and first_val.shape[-1] == self.vit.d_model

        if using_precomputed:
            # Features: [B, H, n_patches, vit_dim] — skip ViT entirely
            cam_tokens_list = []
            for cam_name in cam_keys:
                feats = images_dict[cam_name]  # [B, H, n_patches, vit_dim]
                if self.vis_pool is not None:
                    # Pool: [B*H, vit_dim, n_patches] → [B*H, vit_dim, vis_tokens]
                    BH = B * H
                    feats_flat = feats.reshape(BH, feats.shape[2], feats.shape[3])  # [B*H, np, d]
                    feats_flat = self.vis_pool(feats_flat.transpose(1, 2)).transpose(1, 2)  # [B*H, vt, d]
                    feats = feats_flat.reshape(B, H, self.n_patches, -1)
                feats = self.vis_proj(feats)  # [B, H, n_patches, d_model]
                cam_tokens_list.append(feats)
        else:
            # Raw images: run all cameras × history through ViT in one fused batch
            all_imgs = []
            for cam_name in cam_keys:
                cam_imgs = images_dict[cam_name]  # [B, H, C, h, w]
                all_imgs.append(cam_imgs.reshape(B * H, *cam_imgs.shape[2:]))
            # Batch: [B*H*n_cams, C, h, w]
            all_flat = torch.cat(all_imgs, dim=0)
            all_patches = self.vit(all_flat)  # [B*H*n_cams, raw_patches, vit_dim]

            if self.vis_pool is not None:
                # Pool patches: [B*H*n_cams, vit_dim, raw_patches] → vis_tokens
                all_patches = self.vis_pool(
                    all_patches.transpose(1, 2)
                ).transpose(1, 2)  # [B*H*n_cams, vis_tokens, vit_dim]

            all_patches = self.vis_proj(all_patches)  # [B*H*n_cams, n_patches, d_model]

            # Split back by camera
            chunks = all_patches.split(B * H, dim=0)  # n_cams × [B*H, n_patches, d_model]
            cam_tokens_list = [
                c.reshape(B, H, self.n_patches, self.d_model) for c in chunks
            ]

        # Concatenate camera tokens: [B, H, n_cameras * n_patches, d_model]
        vision_tokens = torch.cat(cam_tokens_list, dim=2)

        # --- Encode state and actions ---
        state_tokens = self.state_encoder(state).unsqueeze(2)         # [B, H, 1, d_model]
        action_tokens = self.action_encoder(prev_actions).unsqueeze(2)  # [B, H, 1, d_model]

        # --- Assemble token sequence ---
        # Per timestep: [vision, state, action]
        timestep_tokens = torch.cat([vision_tokens, state_tokens, action_tokens], dim=2)
        # [B, H, tokens_per_group, d_model]

        # Flatten to sequence: [B, H * tokens_per_group, d_model]
        sequence = timestep_tokens.reshape(B, H * self.tokens_per_group, self.d_model)

        # Append [READ] token
        readout = self.readout_token.expand(B, -1, -1)
        sequence = torch.cat([sequence, readout], dim=1)  # [B, total_tokens, d_model]

        # --- Add positional encoding ---
        sequence = sequence + self.pos_enc.unsqueeze(0)

        # --- Transformer decoder with block-causal mask ---
        sequence = self.transformer(sequence, attn_mask=self.attn_mask)

        # --- Extract [READ] token → conditioning ---
        readout_out = sequence[:, -1]  # [B, d_model]
        conditioning = self.readout_proj(readout_out)  # [B, cond_dim]

        return conditioning

    def forward(self, batch: dict) -> dict:
        """Training forward pass with flow matching loss.

        batch must include "action": [B, action_dim] target action.
        """
        conditioning = self.encode(batch)
        target_action = batch["action"]  # [B, action_dim]
        B = target_action.shape[0]

        # Flow matching: sample random timestep and noise
        t = torch.rand(B, device=target_action.device)
        noise = torch.randn_like(target_action)

        # Interpolate: a_t = (1-t)*noise + t*target
        t_expand = t.unsqueeze(-1)
        noisy_action = (1.0 - t_expand) * noise + t_expand * target_action

        # Target velocity: v* = target - noise (straight-line flow)
        target_velocity = target_action - noise

        # Predict velocity
        pred_velocity = self.dit_head(noisy_action, t, conditioning)

        # MSE loss on velocity field
        loss = F.mse_loss(pred_velocity, target_velocity)

        return {"loss": loss}

    @torch.no_grad()
    def predict(self, batch: dict, num_steps: int = None) -> torch.Tensor:
        """Inference: sample action via Euler ODE integration.

        Args:
            batch: observation batch (no "action" key needed)
            num_steps: Euler integration steps (default: self.num_flow_steps)
        Returns:
            action: [B, action_dim] sampled clean action
        """
        if num_steps is None:
            num_steps = self.num_flow_steps

        conditioning = self.encode(batch)
        B = conditioning.shape[0]

        # Start from Gaussian noise
        action = torch.randn(B, self.action_dim, device=conditioning.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=conditioning.device)
            velocity = self.dit_head(action, t, conditioning)
            action = action + velocity * dt

        return action


@register("architecture", "vla_flow_matching")
def build_vla_policy(**kwargs):
    """Build VLA policy from config dict.

    All parameters are passed through from robotics.architecture config.
    """
    # Remove 'type' key if present (consumed by registry)
    kwargs.pop("type", None)
    return VLAPolicy(**kwargs)

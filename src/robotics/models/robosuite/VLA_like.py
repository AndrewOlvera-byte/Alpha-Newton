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
from src.robotics.models.AttentionPool import AttentionPool


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
        # Multi-task: number of distinct tasks. When > 1, a learned task token is
        # prepended to the causal transformer sequence (so every frame can attend
        # to it) and a separate task embedding biases the DiT conditioning vector.
        # Set n_tasks=1 (default) for single-task — adds zero overhead.
        n_tasks: int = 1,
        # ViT params
        vit_model: str = "facebook/dinov2-small",
        vit_img_size: int = 224,
        freeze_vit: bool = True,
        # Vision token pooling: pool ViT patch tokens down to this many per camera.
        # DINOv2 ViT-S/14 @ 224px → 256 patches. Pooling to 64 reduces sequence
        # length from 1543 to 391 tokens (4× speedup in attention).
        # Set to 0 to disable pooling and use all patches.
        vis_tokens_per_camera: int = 64,
        # Vision pooling method: "attention" (default, Perceiver-style) or "avg"
        vis_pool_type: str = "attention",
        # Flow matching
        num_flow_steps: int = 10,
        # Action chunking: predict K future actions per step.
        # K > 1 gives the DiT a sequence to attend over (Diffusion Policy style).
        # At inference, execute the first action and discard the rest.
        action_chunk_size: int = 1,
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
        self.action_chunk_size = action_chunk_size

        # === Vision encoder (frozen DINOv2) ===
        self.vit = DINOv2Encoder(model_name=vit_model, img_size=vit_img_size, freeze=freeze_vit)
        vit_dim = self.vit.d_model  # 384 for ViT-S

        # === Vision token pooling ===
        raw_patches = self.vit.n_patches  # 256 for 224×224 with patch 14
        if vis_tokens_per_camera > 0 and vis_tokens_per_camera < raw_patches:
            if vis_pool_type == "attention":
                self.vis_pool = AttentionPool(
                    d_model=vit_dim,
                    n_queries=vis_tokens_per_camera,
                    n_heads=min(4, vit_dim // 64),
                )
            else:
                self.vis_pool = nn.AdaptiveAvgPool1d(vis_tokens_per_camera)
            self.n_patches = vis_tokens_per_camera
            self._pool_type = vis_pool_type
        else:
            self.vis_pool = None
            self.n_patches = raw_patches
            self._pool_type = "none"

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
        )

        # === DiT action head ===
        self.dit_head = DiTActionHead(
            action_dim=action_dim,
            d_model=d_dit,
            n_layers=n_dit_layers,
            n_heads=n_dit_heads,
            d_ff=d_dit_ff,
            cond_dim=d_dit,
            dropout=dropout,
            action_chunk_size=action_chunk_size,
        )

        # === Precompute sequence structure ===
        n_vision = n_cameras * self.n_patches
        tokens_per_group = n_vision + 2  # vision + state + action
        self.tokens_per_group = tokens_per_group
        self.n_tasks = n_tasks

        # Task embeddings — only allocated when n_tasks > 1.
        # task_emb:     [n_tasks, d_model] — task token prepended to the causal
        #               transformer sequence as group 0. Every subsequent group
        #               can attend back to it via the block-causal mask, giving
        #               the full history a shared task context signal.
        # task_emb_dit: [n_tasks, d_dit]  — additive bias to the DiT conditioning
        #               vector, so the flow-matching action head is also task-aware.
        if n_tasks > 1:
            self.task_emb = nn.Embedding(n_tasks, d_model)
            self.task_emb_dit = nn.Embedding(n_tasks, d_dit)
            nn.init.normal_(self.task_emb.weight, std=0.02)
            nn.init.normal_(self.task_emb_dit.weight, std=0.02)
            # Task token is group 0; history groups 1..H+1; readout is last group.
            self.group_sizes = [1] + [tokens_per_group] * history_length + [1]
            self.total_tokens = 1 + tokens_per_group * history_length + 1
        else:
            self.group_sizes = [tokens_per_group] * history_length + [1]
            self.total_tokens = tokens_per_group * history_length + 1

        # Cache attention mask (registered as buffer, moves with model)
        mask = build_block_causal_mask(self.group_sizes, device=torch.device("cpu"))
        self.register_buffer("attn_mask", mask, persistent=False)

        # Cache positional encoding
        pe = sinusoidal_positional_encoding(self.total_tokens, d_model, torch.device("cpu"))
        self.register_buffer("pos_enc", pe, persistent=False)

    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply vision pooling. patches: [N, n_patches, vit_dim]."""
        if self.vis_pool is None:
            return patches
        if self._pool_type == "attention":
            return self.vis_pool(patches)  # [N, n_queries, vit_dim]
        else:
            # AdaptiveAvgPool1d: needs [N, C, L] layout
            return self.vis_pool(patches.transpose(1, 2)).transpose(1, 2)

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
        cam_keys = list(images_dict.keys())
        n_cams = len(cam_keys)

        # Check if batch contains precomputed features
        first_val = images_dict[cam_keys[0]]
        using_precomputed = first_val.ndim == 4 and first_val.shape[-1] == self.vit.d_model

        if using_precomputed:
            cam_tokens_list = []
            for cam_name in cam_keys:
                feats = images_dict[cam_name]  # [B, H, n_patches, vit_dim]
                BH = B * H
                feats_flat = feats.reshape(BH, feats.shape[2], feats.shape[3])
                feats_flat = self._pool_patches(feats_flat)
                feats = feats_flat.reshape(B, H, self.n_patches, -1)
                feats = self.vis_proj(feats)
                cam_tokens_list.append(feats)
        else:
            # Raw images: run all cameras × history through ViT in one fused batch
            all_imgs = []
            for cam_name in cam_keys:
                cam_imgs = images_dict[cam_name]  # [B, H, C, h, w]
                all_imgs.append(cam_imgs.reshape(B * H, *cam_imgs.shape[2:]))
            all_flat = torch.cat(all_imgs, dim=0)
            all_patches = self.vit(all_flat)  # [B*H*n_cams, raw_patches, vit_dim]

            all_patches = self._pool_patches(all_patches)
            all_patches = self.vis_proj(all_patches)

            chunks = all_patches.split(B * H, dim=0)
            cam_tokens_list = [
                c.reshape(B, H, self.n_patches, self.d_model) for c in chunks
            ]

        # Concatenate camera tokens: [B, H, n_cameras * n_patches, d_model]
        vision_tokens = torch.cat(cam_tokens_list, dim=2)

        # --- Encode state and actions ---
        state_tokens = self.state_encoder(state).unsqueeze(2)         # [B, H, 1, d_model]
        action_tokens = self.action_encoder(prev_actions).unsqueeze(2)  # [B, H, 1, d_model]

        # --- Assemble token sequence ---
        timestep_tokens = torch.cat([vision_tokens, state_tokens, action_tokens], dim=2)

        # Flatten to sequence: [B, H * tokens_per_group, d_model]
        sequence = timestep_tokens.reshape(B, H * self.tokens_per_group, self.d_model)

        # Prepend task token (multi-task mode only).
        # Placed at position 0 so every history frame can attend back to it.
        if self.n_tasks > 1 and "task_id" in batch:
            task_token = self.task_emb(batch["task_id"]).unsqueeze(1)  # [B, 1, d_model]
            sequence = torch.cat([task_token, sequence], dim=1)

        # Append [READ] token
        readout = self.readout_token.expand(B, -1, -1)
        sequence = torch.cat([sequence, readout], dim=1)

        # --- Add positional encoding ---
        sequence = sequence + self.pos_enc.unsqueeze(0)

        # --- Transformer decoder with block-causal mask ---
        sequence = self.transformer(sequence, attn_mask=self.attn_mask)

        # --- Extract [READ] token → conditioning ---
        readout_out = sequence[:, -1]  # [B, d_model]
        conditioning = self.readout_proj(readout_out)  # [B, cond_dim]

        # Task-conditioned bias to DiT action head.
        # Additive injection (not concatenation) keeps d_dit fixed regardless of n_tasks.
        if self.n_tasks > 1 and "task_id" in batch:
            conditioning = conditioning + self.task_emb_dit(batch["task_id"])

        return conditioning

    def forward(self, batch: dict) -> dict:
        """Training forward pass with flow matching loss.

        batch must include "action": [B, action_dim] or [B, K, action_dim].
        """
        conditioning = self.encode(batch)
        target_action = batch["action"]  # [B, action_dim] or [B, K, action_dim]
        B = target_action.shape[0]
        K = self.action_chunk_size

        # Reshape to chunk form if needed
        if target_action.ndim == 2 and K > 1:
            # Should not happen if data pipeline is correct, but handle gracefully
            target_action = target_action.unsqueeze(1).expand(-1, K, -1)
        elif target_action.ndim == 2 and K == 1:
            target_action = target_action.unsqueeze(1)  # [B, 1, action_dim]

        # Flow matching: sample random timestep and noise
        t = torch.rand(B, device=target_action.device)
        noise = torch.randn_like(target_action)  # [B, K, action_dim]

        # Interpolate: a_t = (1-t)*noise + t*target
        t_expand = t.view(B, 1, 1)
        noisy_action = (1.0 - t_expand) * noise + t_expand * target_action

        # Target velocity: v* = target - noise (straight-line flow)
        target_velocity = target_action - noise

        # Predict velocity: [B, K, action_dim]
        pred_velocity = self.dit_head(noisy_action, t, conditioning)

        # MSE loss on velocity field — keep per-sample for t-bucket diagnostics
        loss_per_sample = F.mse_loss(pred_velocity, target_velocity, reduction="none").mean(dim=(-1, -2))  # [B]
        loss = loss_per_sample.mean()

        return {"loss": loss, "_t": t.detach(), "_loss_per_sample": loss_per_sample.detach()}

    @torch.no_grad()
    def predict(self, batch: dict, num_steps: int = None) -> torch.Tensor:
        """Inference: sample action via Euler ODE integration.

        Args:
            batch: observation batch (no "action" key needed)
            num_steps: Euler integration steps (default: self.num_flow_steps)
        Returns:
            action: [B, action_dim] — first action from chunk (execute this one)
        """
        if num_steps is None:
            num_steps = self.num_flow_steps
        assert num_steps > 0, f"num_flow_steps must be > 0, got {num_steps}"

        conditioning = self.encode(batch)
        B = conditioning.shape[0]
        K = self.action_chunk_size

        # Start from Gaussian noise: [B, K, action_dim]
        action = torch.randn(B, K, self.action_dim, device=conditioning.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=conditioning.device)
            velocity = self.dit_head(action, t, conditioning)
            action = action + velocity * dt

        # Clamp normalized action to ±3σ before returning.
        # Prevents large DiT velocity accumulations from producing extreme denormalized
        # actions that get silently clipped by the environment — critical for clean RL handoff.
        action = action.clamp(-3.0, 3.0)

        # Return first action from chunk
        return action[:, 0]


@register("architecture", "vla_flow_matching")
def build_vla_policy(**kwargs):
    """Build VLA policy from config dict.

    All parameters are passed through from robotics.architecture config.
    """
    # Remove 'type' key if present (consumed by registry)
    kwargs.pop("type", None)
    return VLAPolicy(**kwargs)

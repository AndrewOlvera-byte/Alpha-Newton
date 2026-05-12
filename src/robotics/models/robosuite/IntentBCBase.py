import torch
import torch.nn as nn

from src.robotics.models.ActionEncoder import ActionEncoder
from src.robotics.models.AttentionPool import AttentionPool
from src.robotics.models.DecoderTransformer import CausalTransformerDecoder
from src.robotics.models.StateEncoder import StateEncoder
from src.robotics.models.ViT import DINOv2Encoder


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
    """Build a block-causal attention mask for history-group decoding."""
    total = sum(group_sizes)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)

    row_start = 0
    for i, size_i in enumerate(group_sizes):
        col_start = 0
        for j in range(i + 1):
            size_j = group_sizes[j]
            mask[row_start: row_start + size_i, col_start: col_start + size_j] = True
            col_start += size_j
        row_start += size_i

    return mask.unsqueeze(0).unsqueeze(0)


class VisionIntentBCBackbone(nn.Module):
    """Shared visuomotor backbone that exposes sequence tokens plus an intent token."""

    def __init__(
        self,
        d_model: int = 384,
        n_transformer_layers: int = 6,
        n_transformer_heads: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        action_dim: int = 7,
        obs_dim: int = 19,
        n_cameras: int = 2,
        history_length: int = 3,
        n_tasks: int = 1,
        vit_model: str = "facebook/dinov2-small",
        vit_img_size: int = 224,
        freeze_vit: bool = True,
        vis_tokens_per_camera: int = 64,
        vis_pool_type: str = "attention",
        gradient_checkpointing: bool = False,
        encoder_hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.history_length = history_length
        self.n_cameras = n_cameras
        self.n_tasks = n_tasks

        self.vit = DINOv2Encoder(model_name=vit_model, img_size=vit_img_size, freeze=freeze_vit)
        vit_dim = self.vit.d_model
        raw_patches = self.vit.n_patches

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

        self.vis_proj = nn.Linear(vit_dim, d_model, bias=False) if vit_dim != d_model else nn.Identity()
        self.state_encoder = StateEncoder(obs_dim, encoder_hidden_dim, d_model)
        self.action_encoder = ActionEncoder(action_dim, encoder_hidden_dim, d_model)
        self.readout_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.transformer = CausalTransformerDecoder(
            d_model=d_model,
            n_layers=n_transformer_layers,
            n_heads=n_transformer_heads,
            d_ff=d_ff,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        n_vision = n_cameras * self.n_patches
        self.tokens_per_group = n_vision + 2

        if n_tasks > 1:
            self.task_emb = nn.Embedding(n_tasks, d_model)
            nn.init.normal_(self.task_emb.weight, std=0.02)
            self.group_sizes = [1] + [self.tokens_per_group] * history_length + [1]
            self.total_tokens = 1 + self.tokens_per_group * history_length + 1
        else:
            self.group_sizes = [self.tokens_per_group] * history_length + [1]
            self.total_tokens = self.tokens_per_group * history_length + 1

        mask = build_block_causal_mask(self.group_sizes, device=torch.device("cpu"))
        self.register_buffer("attn_mask", mask, persistent=False)
        pe = sinusoidal_positional_encoding(self.total_tokens, d_model, torch.device("cpu"))
        self.register_buffer("pos_enc", pe, persistent=False)

    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if self.vis_pool is None:
            return patches
        if self._pool_type == "attention":
            return self.vis_pool(patches)
        return self.vis_pool(patches.transpose(1, 2)).transpose(1, 2)

    def forward(self, batch: dict) -> dict:
        images_dict = batch["images"]
        state = batch["state"]
        prev_actions = batch["prev_actions"]
        batch_size = state.shape[0]
        history = self.history_length

        cam_keys = list(images_dict.keys())
        first_val = images_dict[cam_keys[0]]
        using_precomputed = first_val.ndim == 4 and first_val.shape[-1] == self.vit.d_model

        if using_precomputed:
            cam_tokens_list = []
            for cam_name in cam_keys:
                feats = images_dict[cam_name]
                feats = feats.reshape(batch_size * history, feats.shape[2], feats.shape[3])
                feats = self._pool_patches(feats)
                feats = feats.reshape(batch_size, history, self.n_patches, -1)
                cam_tokens_list.append(self.vis_proj(feats))
        else:
            all_imgs = []
            for cam_name in cam_keys:
                cam_imgs = images_dict[cam_name]
                all_imgs.append(cam_imgs.reshape(batch_size * history, *cam_imgs.shape[2:]))
            all_flat = torch.cat(all_imgs, dim=0)
            all_patches = self.vit(all_flat)
            all_patches = self._pool_patches(all_patches)
            all_patches = self.vis_proj(all_patches)
            chunks = all_patches.split(batch_size * history, dim=0)
            cam_tokens_list = [chunk.reshape(batch_size, history, self.n_patches, self.d_model) for chunk in chunks]

        vision_tokens = torch.cat(cam_tokens_list, dim=2)
        state_tokens = self.state_encoder(state).unsqueeze(2)
        action_tokens = self.action_encoder(prev_actions).unsqueeze(2)
        timestep_tokens = torch.cat([vision_tokens, state_tokens, action_tokens], dim=2)
        sequence = timestep_tokens.reshape(batch_size, history * self.tokens_per_group, self.d_model)

        if self.n_tasks > 1 and "task_id" in batch:
            task_token = self.task_emb(batch["task_id"]).unsqueeze(1)
            sequence = torch.cat([task_token, sequence], dim=1)

        readout = self.readout_token.expand(batch_size, -1, -1)
        sequence = torch.cat([sequence, readout], dim=1)
        sequence = sequence + self.pos_enc.unsqueeze(0)
        decoded = self.transformer(sequence, attn_mask=self.attn_mask)

        return {
            "decoded_sequence": decoded[:, :-1],
            "intent_token": decoded[:, -1],
        }

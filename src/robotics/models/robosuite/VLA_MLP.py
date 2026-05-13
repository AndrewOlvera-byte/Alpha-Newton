"""
Visuomotor Policy with deterministic MLP control head.

Registered as architecture type "vla_mlp_bc".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import register
from src.robotics.loss import gaussian_nll_loss
from src.robotics.models.robosuite.IntentBCBase import VisionIntentBCBackbone
from src.robotics.models.robosuite.MLPController import MLPController


class VLAMLPPolicy(nn.Module):
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
        action_chunk_size: int = 1,
        gradient_checkpointing: bool = False,
        encoder_hidden_dim: int = 128,
        controller_hidden_dim: int = 512,
        controller_cross_attention_heads: int = 4,
        controller_ff_dim: int = 1024,
        bc_loss_type: str = "mse",
        **kwargs,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.bc_loss_type = bc_loss_type
        self.backbone = VisionIntentBCBackbone(
            d_model=d_model,
            n_transformer_layers=n_transformer_layers,
            n_transformer_heads=n_transformer_heads,
            d_ff=d_ff,
            dropout=dropout,
            action_dim=action_dim,
            obs_dim=obs_dim,
            n_cameras=n_cameras,
            history_length=history_length,
            n_tasks=n_tasks,
            vit_model=vit_model,
            vit_img_size=vit_img_size,
            freeze_vit=freeze_vit,
            vis_tokens_per_camera=vis_tokens_per_camera,
            vis_pool_type=vis_pool_type,
            gradient_checkpointing=gradient_checkpointing,
            encoder_hidden_dim=encoder_hidden_dim,
        )
        self.controller = MLPController(
            action_dim=action_dim,
            d_model=d_model,
            hidden_dim=controller_hidden_dim,
            n_heads=controller_cross_attention_heads,
            d_ff=controller_ff_dim,
            dropout=dropout,
            action_chunk_size=action_chunk_size,
            predict_log_var=(bc_loss_type == "gnll"),
        )

    def _target_to_chunk(self, target_action: torch.Tensor) -> torch.Tensor:
        if target_action.ndim == 2:
            target_action = target_action.unsqueeze(1)
        return target_action

    def forward(self, batch: dict) -> dict:
        encoded = self.backbone(batch)
        ctrl_out = self.controller(encoded["decoded_sequence"], encoded["intent_token"])
        target_action = self._target_to_chunk(batch["action"])

        if self.bc_loss_type == "gnll":
            pred_mu, pred_log_var = ctrl_out
            loss, loss_per_sample = gaussian_nll_loss(pred_mu, pred_log_var, target_action)
        else:
            loss_per_sample = F.mse_loss(ctrl_out, target_action, reduction="none").mean(dim=(-1, -2))
            loss = loss_per_sample.mean()
            loss_per_sample = loss_per_sample.detach()

        return {"loss": loss, "_loss_per_sample": loss_per_sample}

    @torch.no_grad()
    def predict(self, batch: dict, num_steps: int = None) -> torch.Tensor:
        del num_steps
        encoded = self.backbone(batch)
        ctrl_out = self.controller(encoded["decoded_sequence"], encoded["intent_token"])
        if self.bc_loss_type == "gnll":
            mu, _ = ctrl_out
            return mu[:, 0].clamp(-5.0, 5.0)
        return ctrl_out[:, 0].clamp(-5.0, 5.0)


@register("architecture", "vla_mlp_bc")
def build_vla_mlp_policy(**kwargs):
    kwargs.pop("type", None)
    return VLAMLPPolicy(**kwargs)

"""
Visuomotor Policy with diagonal-Gaussian control head.

Registered as architecture type "vla_gaussian_bc".
"""
import torch
import torch.nn as nn

from src.core.registry import register
from src.robotics.models.robosuite.GaussianController import GaussianController
from src.robotics.models.robosuite.IntentBCBase import VisionIntentBCBackbone


class VLAGaussianPolicy(nn.Module):
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
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
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
        self.controller = GaussianController(
            action_dim=action_dim,
            d_model=d_model,
            hidden_dim=controller_hidden_dim,
            n_heads=controller_cross_attention_heads,
            d_ff=controller_ff_dim,
            dropout=dropout,
            action_chunk_size=action_chunk_size,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def _target_to_chunk(self, target_action: torch.Tensor) -> torch.Tensor:
        if target_action.ndim == 2:
            target_action = target_action.unsqueeze(1)
        return target_action

    def forward(self, batch: dict) -> dict:
        encoded = self.backbone(batch)
        mean, log_std = self.controller(encoded["decoded_sequence"], encoded["intent_token"])
        target_action = self._target_to_chunk(batch["action"])
        nll = self.controller.nll(target_action, mean, log_std)
        loss_per_sample = nll.mean(dim=(-1, -2))
        return {"loss": loss_per_sample.mean(), "_loss_per_sample": loss_per_sample.detach()}

    @torch.no_grad()
    def predict(self, batch: dict, num_steps: int = None) -> torch.Tensor:
        del num_steps
        encoded = self.backbone(batch)
        mean, _ = self.controller(encoded["decoded_sequence"], encoded["intent_token"])
        return mean[:, 0].clamp(-5.0, 5.0)


@register("architecture", "vla_gaussian_bc")
def build_vla_gaussian_policy(**kwargs):
    kwargs.pop("type", None)
    return VLAGaussianPolicy(**kwargs)

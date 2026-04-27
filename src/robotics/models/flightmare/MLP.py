"""Low-latency vision + state fusion trunk for Flightmare drone BC / PPO.

Design rationale (informed by Loquercio et al. DroNet 2018, "High-Speed Flight
in the Wild" Sci. Robotics 2021, Kaufmann et al. Swift Nature 2023, and the
Stable-Baselines3 / OpenAI-Baselines PPO conventions):

* Frozen ImageNet-pretrained CNN as the visual encoder. ResNet-18 with global
  average pooling is the canonical drone-BC backbone (DroNet inherited it,
  Loquercio's wilderness flight system uses a similarly small frozen CNN).
  MobileNetV3-Small offers a smaller / faster alternative for tight edge
  budgets, and EfficientNet-V2-S is supported for higher capacity. A frozen
  pretrained ViT (DINOv2 small) is also available for ablations, but in
  latency-critical drone control a CNN with GAP consistently dominates.
* Single frame per camera in the hot path. Drone state at typical control
  rates (50-100 Hz) is approximately Markov, so frame stacking yields little
  benefit and roughly doubles vision-encoder cost. Short proprioceptive
  history can still be injected via a flat state vector if desired.
* Global average pooling rather than attention pooling. Cheaper and better
  matched to small frozen CNNs whose spatial maps are already low-resolution.
* channels_last memory format on the CNN, BatchNorm kept in eval(), and no
  Python control flow in the forward path - all torch.compile friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVisionEncoder(nn.Module):
    """Frozen pretrained vision backbone producing one feature vector per image.

    Backbones (selected via ``backbone`` arg):
      * ``resnet18`` (default)     - 512-d, ~11M params, drone-BC standard
      * ``resnet34``               - 512-d, ~21M params
      * ``mobilenet_v3_small``     - 576-d, ~2.5M params, lowest latency
      * ``mobilenet_v3_large``     - 960-d
      * ``efficientnet_v2_s``      - 1280-d, ~21M params
      * ``dinov2_small``           - 384-d, ViT-S/14 via transformers, for
                                     ablations (heavier than ResNet-18 in
                                     wall-clock on small batch sizes).

    All return ``[B, feature_dim]`` via global average pooling.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    _CNN_FEATURE_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "efficientnet_v2_s": 1280,
    }

    def __init__(
        self,
        backbone: str = "resnet18",
        img_size: int = 224,
        freeze: bool = True,
        pretrained: bool = True,
        channels_last: bool = True,
        dinov2_model_name: str = "facebook/dinov2-small",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.img_size = img_size
        self.frozen = freeze
        self._channels_last = channels_last and not backbone.startswith("dinov2")

        if backbone in self._CNN_FEATURE_DIMS:
            from torchvision import models as tvm

            weights = "DEFAULT" if pretrained else None
            net = getattr(tvm, backbone)(weights=weights)
            if backbone.startswith("resnet"):
                self.features = nn.Sequential(*list(net.children())[:-1])
            else:
                self.features = nn.Sequential(net.features, net.avgpool)
            self.feature_dim = self._CNN_FEATURE_DIMS[backbone]
            self._is_vit = False
            
        elif backbone == "dinov2_small":
            from transformers import Dinov2Model

            self.features = Dinov2Model.from_pretrained(dinov2_model_name)
            self.feature_dim = self.features.config.hidden_size
            self._is_vit = True
        else:
            raise ValueError(
                f"backbone '{backbone}' not supported. "
                f"Pick from {list(self._CNN_FEATURE_DIMS)} or 'dinov2_small'."
            )

        if freeze:
            self.features.requires_grad_(False)
            self.features.eval()
        if self._channels_last:
            self.features = self.features.to(memory_format=torch.channels_last)

        self.register_buffer("pixel_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.features.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``[B, 3, H, W]`` in ``[0, 1]`` -> ``[B, feature_dim]``."""
        if images.shape[-1] != self.img_size or images.shape[-2] != self.img_size:
            images = F.interpolate(
                images,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.pixel_mean) / self.pixel_std
        if self._channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(not self.frozen):
            if self._is_vit:
                out = self.features(pixel_values=images)
                feat = out.last_hidden_state[:, 0]  # CLS token
            else:
                feat = self.features(images).flatten(1)
        return feat


class MLPFusion(nn.Module):
    """Vision + proprio state (+ prev action) -> compact fused trunk feature.

    The fusion trunk is intentionally small (a few LayerNorm + Linear + SiLU
    blocks) to keep policy latency low. BatchNorm is avoided so single-sample
    inference at deployment does not require running-stat calibration.

    Image input formats accepted in ``forward``:
      * ``batch["images"]`` is a tensor ``[B, 3, H, W]`` (single camera), or
      * ``batch["images"]`` is a dict ``{cam_name: [B, 3, H, W]}`` with
        ``len == n_cameras``. Camera ordering follows iteration order of the
        dict (insertion-ordered in modern Python).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        backbone: str = "resnet18",
        img_size: int = 224,
        pretrained: bool = True,
        freeze_vision: bool = True,
        n_cameras: int = 1,
        include_prev_action: bool = True,
        state_hidden_dim: int = 128,
        state_embed_dim: int = 128,
        trunk_hidden_dim: int = 256,
        trunk_depth: int = 2,
        dropout: float = 0.0,
        channels_last: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_cameras = n_cameras
        self.include_prev_action = include_prev_action

        self.vision = TinyVisionEncoder(
            backbone=backbone,
            img_size=img_size,
            freeze=freeze_vision,
            pretrained=pretrained,
            channels_last=channels_last,
        )

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_hidden_dim, state_embed_dim),
        )

        if include_prev_action:
            self.action_encoder = nn.Sequential(
                nn.LayerNorm(action_dim),
                nn.Linear(action_dim, state_hidden_dim),
                nn.SiLU(),
                nn.Linear(state_hidden_dim, state_embed_dim),
            )
            extra = state_embed_dim
        else:
            self.action_encoder = None
            extra = 0

        fused_in = self.vision.feature_dim * n_cameras + state_embed_dim + extra
        layers: list[nn.Module] = [
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, trunk_hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(max(0, trunk_depth - 1)):
            layers.append(nn.Linear(trunk_hidden_dim, trunk_hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.feature_dim = trunk_hidden_dim

    def encode_images(self, images) -> torch.Tensor:
        if isinstance(images, dict):
            cam_tensors = list(images.values())
            B = cam_tensors[0].shape[0]
            stacked = torch.cat(cam_tensors, dim=0)  # [B*n_cam, 3, H, W]
            feats = self.vision(stacked)
            feats = feats.reshape(self.n_cameras, B, -1).permute(1, 0, 2).reshape(B, -1)
        else:
            feats = self.vision(images)
        return feats

    def forward(self, batch: dict) -> torch.Tensor:
        feats = self.encode_images(batch["images"])
        state_emb = self.state_encoder(batch["state"])
        parts = [feats, state_emb]
        if self.include_prev_action:
            parts.append(self.action_encoder(batch["prev_actions"]))
        fused = torch.cat(parts, dim=-1)
        return self.trunk(fused)

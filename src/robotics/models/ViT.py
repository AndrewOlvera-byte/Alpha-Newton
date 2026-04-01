import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 vision encoder.

    Input:  [B, C, H, W] float32 [0, 1]
    Output: [B, n_patches, 384] patch token embeddings
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name: str = "facebook/dinov2-small", img_size: int = 224, freeze: bool = True):
        super().__init__()
        from transformers import Dinov2Model

        self.vit = Dinov2Model.from_pretrained(model_name)
        self.img_size = img_size
        self.d_model = self.vit.config.hidden_size  # 384 for ViT-S
        self.patch_size = self.vit.config.patch_size  # 14
        self.n_patches = (img_size // self.patch_size) ** 2

        # ImageNet normalization as buffers (move with model to device)
        self.register_buffer("pixel_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        if freeze:
            self.vit.requires_grad_(False)
            self.vit.eval()

        self.frozen = freeze

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] float32 in [0, 1]
        Returns:
            [B, n_patches, d_model] patch token embeddings
        """
        # Resize if needed
        if images.shape[-1] != self.img_size or images.shape[-2] != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # ImageNet normalization
        images = (images - self.pixel_mean) / self.pixel_std

        with torch.set_grad_enabled(not self.frozen):
            outputs = self.vit(pixel_values=images)

        # Remove CLS token, keep only patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:]
        return patch_tokens

"""Image augmentation for Flightmare drone BC.

Choice of augmentations is conservative on purpose: drone visuomotor BC is
extremely sensitive to anything that breaks the geometric correspondence
between the image and the action label. We avoid rotations and large crops
(which would invalidate the body-frame waypoint and the CTBR/motor labels).

Photometric augmentations are kept moderate, in line with what worked for
"Learning High-Speed Flight in the Wild" (Loquercio et al. 2021): brightness,
contrast, saturation, hue jitter; small Gaussian noise; optional motion blur
to mimic high-speed flight.

All augs run on uint8 [B?, C, H, W] tensors via ``torchvision.transforms.v2``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_train_aug(image_size: int, motion_blur: bool = True) -> nn.Module:
    from torchvision.transforms import v2 as T  # type: ignore
    layers: list[nn.Module] = [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    ]
    if motion_blur:
        layers.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3))
    layers += [
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=image_size, antialias=True),
        T.CenterCrop(size=image_size),
    ]
    return nn.Sequential(*layers)


def build_eval_aug(image_size: int) -> nn.Module:
    from torchvision.transforms import v2 as T  # type: ignore
    return nn.Sequential(
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=image_size, antialias=True),
        T.CenterCrop(size=image_size),
    )

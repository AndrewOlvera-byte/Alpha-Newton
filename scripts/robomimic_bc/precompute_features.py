"""
Precompute DINOv2 ViT-S/14 features from rendered images in HDF5.

Stores features under obs/{cam_key}_features alongside the image data.
This eliminates the ViT from the training loop entirely, giving a
10-30× throughput improvement for large-scale BC training.

With --n-aug N > 1, each frame is passed through the ViT N times with
independent random augmentations (random crop + color jitter). Features
are stored as [T, N_aug, n_patches, d_model]. The dataset randomly picks
one aug view per training step, effectively multiplying dataset size by N.

Usage:
    # Single pass (no aug, backward-compatible):
    python scripts/precompute_features.py \\
        --hdf5 data/robomimic/lift/ph/image.hdf5 \\
        --cameras agentview_image robot0_eye_in_hand_image

    # 4× augmented (recommended for small datasets):
    python scripts/precompute_features.py \\
        --hdf5 data/robomimic/lift/ph/image.hdf5 \\
        --cameras agentview_image robot0_eye_in_hand_image \\
        --n-aug 4

The script is idempotent: skips cameras where features already exist
with the correct n-aug dimension.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_vit(model_name: str, img_size: int, device: torch.device):
    from transformers import Dinov2Model

    vit = Dinov2Model.from_pretrained(model_name).to(device).eval()
    vit.requires_grad_(False)
    n_patches = (img_size // vit.config.patch_size) ** 2
    d_model = vit.config.hidden_size

    # ImageNet normalization buffers
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"[ViT] {model_name}: patch_size={vit.config.patch_size}, "
          f"n_patches={n_patches}, d_model={d_model}")
    return vit, pixel_mean, pixel_std, n_patches, d_model


def build_aug_pipeline(img_size: int) -> T.Compose:
    """Random crop + color jitter applied before ViT encoding."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ])


def build_clean_pipeline(img_size: int) -> T.Compose:
    """Center crop only — used for the first (canonical) aug view."""
    return T.Compose([
        T.CenterCrop(img_size),
    ])


@torch.no_grad()
def encode_batch(vit, pixel_mean, pixel_std, imgs_chw: torch.Tensor) -> torch.Tensor:
    """imgs_chw: [B, C, H, W] float [0,1] on device → [B, n_patches, d] float16."""
    imgs_norm = (imgs_chw - pixel_mean) / pixel_std
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = vit(pixel_values=imgs_norm)
    patches = out.last_hidden_state[:, 1:]  # remove CLS token
    return patches.to(torch.float16)


@torch.no_grad()
def encode_demo_all_augs(
    vit,
    pixel_mean,
    pixel_std,
    imgs_uint8: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
    n_aug: int,
) -> np.ndarray:
    """Encode one demo with N aug views.

    Args:
        imgs_uint8: [T, H, W, C] uint8
    Returns:
        features: [T, N_aug, n_patches, d_model] float16
    """
    T_len = imgs_uint8.shape[0]
    aug_views = []

    for aug_i in range(n_aug):
        # First view is canonical (center crop only); rest are random
        pipeline = build_clean_pipeline(img_size) if aug_i == 0 else build_aug_pipeline(img_size)
        view_features = []

        for start in range(0, T_len, batch_size):
            batch_np = imgs_uint8[start: start + batch_size]  # [B, H, W, C]
            t = torch.from_numpy(batch_np).to(device, non_blocking=True)
            t = t.permute(0, 3, 1, 2).float() / 255.0  # [B, C, H, W]

            if t.shape[-1] != img_size or t.shape[-2] != img_size:
                t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)

            t = pipeline(t)
            feats = encode_batch(vit, pixel_mean, pixel_std, t)  # [B, n_patches, d]
            view_features.append(feats.cpu().numpy())

        aug_views.append(np.concatenate(view_features, axis=0))  # [T, n_patches, d]

    # Stack: [T, N_aug, n_patches, d_model]
    return np.stack(aug_views, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to image.hdf5")
    parser.add_argument(
        "--cameras", nargs="+",
        default=["agentview_image", "robot0_eye_in_hand_image"],
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--n-aug", type=int, default=4,
        help="Number of augmented views per frame (1 = canonical only, no aug). "
             "Features stored as [T, N_aug, n_patches, d_model]. "
             "Dataset randomly picks one view per training step."
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    vit, pixel_mean, pixel_std, n_patches, d_model = load_vit(args.model, args.img_size, device)

    print(f"[Features] n_aug={args.n_aug} — each frame encoded {args.n_aug}× "
          f"({'canonical + random aug' if args.n_aug > 1 else 'canonical only'})")

    with h5py.File(args.hdf5, "a") as f:
        demo_keys = sorted(f["data"].keys())
        print(f"[Features] {args.hdf5}: {len(demo_keys)} demos")

        for cam_key in args.cameras:
            feat_key = cam_key.replace("_image", "_features")
            print(f"\n[Features] Camera: {cam_key} → {feat_key}")

            # Check how many demos already have features with the right n_aug shape
            already_done = 0
            needs_redo = []
            for dk in demo_keys:
                path = f"data/{dk}/obs/{feat_key}"
                if path in f:
                    existing_shape = f[path].shape  # [T] or [T, N_aug, n_patches, d] or [T, n_patches, d]
                    existing_n_aug = existing_shape[1] if existing_shape.ndim == 4 else 1
                    if existing_n_aug == args.n_aug:
                        already_done += 1
                    else:
                        needs_redo.append(dk)
                else:
                    needs_redo.append(dk)

            if already_done == len(demo_keys):
                print(f"  All {len(demo_keys)} demos already have {feat_key} "
                      f"with n_aug={args.n_aug}, skipping.")
                continue

            if needs_redo:
                print(f"  {already_done}/{len(demo_keys)} already done with correct n_aug. "
                      f"Processing {len(needs_redo)} demos.")

            for dk in tqdm(needs_redo, desc=f"  demos [{feat_key}]"):
                feat_path = f"data/{dk}/obs/{feat_key}"

                # Delete existing if wrong n_aug
                if feat_path in f:
                    del f[feat_path]

                imgs = f[f"data/{dk}/obs/{cam_key}"][:]  # [T, H, W, C] uint8

                if args.n_aug == 1:
                    # Backward-compatible: store [T, n_patches, d_model]
                    features = encode_demo_all_augs(
                        vit, pixel_mean, pixel_std, imgs,
                        args.img_size, args.batch_size, device, n_aug=1,
                    )[:, 0]  # [T, n_patches, d_model]
                    chunks = (1, n_patches, d_model)
                else:
                    # Store [T, N_aug, n_patches, d_model]
                    features = encode_demo_all_augs(
                        vit, pixel_mean, pixel_std, imgs,
                        args.img_size, args.batch_size, device, n_aug=args.n_aug,
                    )
                    chunks = (1, args.n_aug, n_patches, d_model)

                f.create_dataset(
                    feat_path,
                    data=features,
                    compression="lzf",
                    chunks=chunks,
                )

    shape_desc = f"[T, {args.n_aug}, {n_patches}, {d_model}]" if args.n_aug > 1 else f"[T, {n_patches}, {d_model}]"
    print(f"\n[Features] Done. Features stored as float16 {shape_desc} in {args.hdf5}")
    print("Run training with the same image.hdf5 — dataset will auto-detect features and aug dimension.")


if __name__ == "__main__":
    main()

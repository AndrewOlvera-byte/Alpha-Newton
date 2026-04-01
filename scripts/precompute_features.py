"""
Precompute DINOv2 ViT-S/14 features from rendered images in HDF5.

Stores features under obs/{cam_key}_features alongside the image data.
This eliminates the ViT from the training loop entirely, giving a
10-30× throughput improvement for large-scale BC training.

Usage:
    python scripts/precompute_features.py \\
        --hdf5 data/robomimic/lift/ph/image.hdf5 \\
        --cameras agentview_image robot0_eye_in_hand_image \\
        --img-size 224 \\
        --batch-size 256

The script is idempotent: skips cameras where features already exist.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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


@torch.no_grad()
def encode_images(vit, pixel_mean, pixel_std, imgs_uint8: np.ndarray,
                  img_size: int, batch_size: int, device: torch.device) -> np.ndarray:
    """Encode [N, H, W, C] uint8 images → [N, n_patches, d_model] float16."""
    N = imgs_uint8.shape[0]
    all_features = []

    for start in tqdm(range(0, N, batch_size), desc="  encoding", leave=False):
        batch = imgs_uint8[start : start + batch_size]  # [B, H, W, C]
        # HWC uint8 → CHW float [0,1]
        t = torch.from_numpy(batch).to(device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float() / 255.0  # [B, C, H, W]

        if t.shape[-1] != img_size or t.shape[-2] != img_size:
            t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)

        t = (t - pixel_mean) / pixel_std

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = vit(pixel_values=t)
        patches = out.last_hidden_state[:, 1:]  # remove CLS, [B, n_patches, d]
        all_features.append(patches.to(torch.float16).cpu().numpy())

    return np.concatenate(all_features, axis=0)


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
    args = parser.parse_args()

    device = torch.device(args.device)
    vit, pixel_mean, pixel_std, n_patches, d_model = load_vit(args.model, args.img_size, device)

    with h5py.File(args.hdf5, "a") as f:
        demo_keys = sorted(f["data"].keys())
        print(f"[Features] {args.hdf5}: {len(demo_keys)} demos")

        for cam_key in args.cameras:
            feat_key = cam_key.replace("_image", "_features")
            print(f"\n[Features] Camera: {cam_key} → {feat_key}")

            # Check how many demos already have features
            already_done = sum(
                1 for dk in demo_keys
                if f"data/{dk}/obs/{feat_key}" in f
            )
            if already_done == len(demo_keys):
                print(f"  All {len(demo_keys)} demos already have {feat_key}, skipping.")
                continue
            if already_done > 0:
                print(f"  {already_done}/{len(demo_keys)} already done, continuing from where we left off.")

            for dk in tqdm(demo_keys, desc=f"  demos [{feat_key}]"):
                feat_path = f"data/{dk}/obs/{feat_key}"
                if feat_path in f:
                    continue  # already computed

                imgs = f[f"data/{dk}/obs/{cam_key}"][:]  # [T, H, W, C] uint8
                features = encode_images(
                    vit, pixel_mean, pixel_std, imgs,
                    args.img_size, args.batch_size, device,
                )  # [T, n_patches, d_model] float16

                f.create_dataset(
                    feat_path,
                    data=features,
                    compression="lzf",  # fast, lossless
                    chunks=(1, n_patches, d_model),
                )

    print(f"\n[Features] Done. Features stored as float16 in {args.hdf5}")
    print("Run training with the same image.hdf5 — dataset will auto-detect features.")


if __name__ == "__main__":
    main()

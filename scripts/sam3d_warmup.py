#!/usr/bin/env python3
"""
One-time warmup script for SAM-3D Objects.

This script:
- Forces TorchInductor compile + autotune
- Populates persistent kernel cache
- Avoids mesh / fp16 crashes
"""

import sys
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = "/home/ferdinand/sam_project"
SAM3D_NOTEBOOK = f"{PROJECT_ROOT}/sam-3d-objects/notebook"
CONFIG_PATH = f"{PROJECT_ROOT}/sam-3d-objects/checkpoints/hf/pipeline.yaml"
DUMMY_IMAGE_PATH = f"{PROJECT_ROOT}/dummy_input/dummy_input.png"

sys.path.insert(0, SAM3D_NOTEBOOK)

from inference import Inference, load_image

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

IMAGE_SIZE = (512, 512)
assert torch.cuda.is_available(), "CUDA required"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def resize_image_np(image_np: np.ndarray, size=(512, 512)) -> np.ndarray:
    pil = Image.fromarray(image_np)
    pil = pil.resize(size, resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def make_full_mask(size=(512, 512)) -> np.ndarray:
    return np.full(size, 255, dtype=np.uint8)


def make_synthetic_pointmap(h=512, w=512, z=1.0) -> torch.Tensor:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = (xs - w / 2) / (w / 2)
    ys = (ys - h / 2) / (h / 2)
    zs = np.full((h, w), z, dtype=np.float32)
    return torch.from_numpy(np.stack([xs, ys, zs], axis=-1))


# ---------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------

def main():
    print("\n=== SAM3D WARMUP START ===")

    print("Loading inference pipeline (compile=True)")
    inference = Inference(CONFIG_PATH, compile=True)

    print("Loading dummy image")
    image = load_image(DUMMY_IMAGE_PATH, convert_rgb=True)
    image = resize_image_np(image, IMAGE_SIZE)

    mask = make_full_mask(IMAGE_SIZE)
    pointmap = make_synthetic_pointmap(*IMAGE_SIZE)

    # ------------------------------------------------------------
    # Stage 1 warmup (cheap, avoids mesh decoder)
    # ------------------------------------------------------------
    print("Warmup pass 1: stage1_only=True")
    with torch.inference_mode():
        inference._pipeline.run(
            image=image,
            mask=mask,
            seed=0,
            stage1_only=True,
            pointmap=pointmap,
        )

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Full pipeline warmup (NO mesh / texture)
    # ------------------------------------------------------------
    print("Warmup pass 2: full pipeline (safe)")
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=False):
            inference._pipeline.run(
                image=image,
                mask=mask,
                seed=1,
                pointmap=pointmap,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=False,
            )

    torch.cuda.synchronize()

    print("\n=== SAM3D WARMUP COMPLETE ===")
    print("TorchInductor cache is now populated.")
    print("You can start the server.\n")


if __name__ == "__main__":
    main()

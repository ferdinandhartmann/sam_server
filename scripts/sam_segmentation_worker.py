"""Segmentation worker for SAM 3.

Loads the SAM3 segmentation model once, waits for pending jobs, and writes mask
artifacts that downstream workers can consume.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_bbox, plot_mask
from sklearn.cluster import KMeans
from skimage.color import lab2rgb, rgb2lab
from matplotlib.colors import to_rgb

from config import SEGMENTATION_STAGE, WORKER_LOG_PREFIX, WORKER_POLL_SECONDS
from job_state import claim_next_job, prerequisites_for, set_artifacts, set_stage_status


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def generate_colors(n_colors: int = 256, n_samples: int = 5000) -> np.ndarray:
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(lab)
    centers_lab = kmeans.cluster_centers_
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


def plot_mask_overlay(image_path: Path, inference_state, colors: np.ndarray) -> Path:
    plt.figure(figsize=(12, 8))
    img = Image.open(image_path)
    plt.imshow(img)
    nb_objects = len(inference_state["scores"])
    save_path = image_path.parent / "segmentation_visualization.png"
    print(f"{WORKER_LOG_PREFIX} [segmentation] found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = colors[i % len(colors)]
        plot_mask(inference_state["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = inference_state["scores"][i].item()
        plot_bbox(
            h,
            w,
            inference_state["boxes"][i].cpu(),
            text=f"id={i}, prob={prob:.2f}",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    plt.savefig(save_path)
    return save_path


def save_masks_as_pngs(image_array: np.ndarray, inference_state, output_dir: Path) -> Path:
    output_dir.mkdir(exist_ok=True)
    primary_mask_path = output_dir / "mask_0.png"
    for idx, mask in enumerate(inference_state["masks"]):
        mask_np = mask.squeeze(0).cpu().numpy().astype("uint8")
        mask_img = Image.fromarray(mask_np * 255)
        save_path = output_dir / f"mask_{idx}.png"
        mask_img.save(save_path)
        if idx == 0:
            primary_mask_path = save_path

        rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = image_array[..., :3]
        rgba[..., 3] = mask_np * 255
        obj_img = Image.fromarray(rgba)
        obj_img.save(output_dir / f"object_{idx}.png")
        print(f"{WORKER_LOG_PREFIX} [segmentation] saved object {idx} to {save_path}")
    return primary_mask_path


def run_segmentation(job_dir: Path, processor: Sam3Processor, colors: np.ndarray) -> None:
    image_path = job_dir / "input.png"
    seg_dir = job_dir / "segmentation"
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="object")

    mask_path = save_masks_as_pngs(image_np, inference_state, seg_dir)
    preview_path = plot_mask_overlay(image_path, inference_state, colors)

    set_artifacts(job_dir, mask=str(mask_path), segmentation_preview=str(preview_path))


def main() -> None:
    print(f"{WORKER_LOG_PREFIX} [segmentation] loading SAM3 model")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)
    colors = generate_colors(n_colors=128, n_samples=5000)
    print(f"{WORKER_LOG_PREFIX} [segmentation] model ready")

    while True:
        job_dir = claim_next_job(SEGMENTATION_STAGE, prerequisites_for(SEGMENTATION_STAGE))
        if not job_dir:
            time.sleep(WORKER_POLL_SECONDS)
            continue

        try:
            run_segmentation(job_dir, processor, colors)
        except Exception as exc:  # noqa: BLE001 - worker errors need visibility
            print(f"{WORKER_LOG_PREFIX} [segmentation] failed: {exc}")
            set_stage_status(job_dir, SEGMENTATION_STAGE, "failed", error=str(exc))
        else:
            set_stage_status(job_dir, SEGMENTATION_STAGE, "done")


if __name__ == "__main__":
    main()
    

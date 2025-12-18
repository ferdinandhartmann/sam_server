# scripts/sam_segmentation_worker.py

import os, sys

print("\033[92mLoading SAM3_segmentation libraries and model...\033[0m")

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
import time
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results, plot_bbox, plot_mask

# sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

######

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from matplotlib.colors import to_rgb


def save_masks_as_pngs(inference_state, output_dir, img_np):
    for idx, mask in enumerate(inference_state["masks"]):
        mask_np = mask.squeeze(0).cpu().numpy().astype("uint8")
        # Create an RGBA image: copy RGB, set alpha from mask
        rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img_np[..., :3]  # Only take RGB channels if img_np has alpha
        rgba[..., 3] = mask_np * 255
        obj_img = Image.fromarray(rgba)
        obj_save_path = os.path.join(output_dir, f"{idx}.png")
        obj_img.save(obj_save_path)
        print(f"Saved object {idx} to {obj_save_path}")

def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb

def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)
    
def visualize_segmentation_results(image_path, output_dir, inference_state, colors):
    plt.figure(figsize=(12, 8))
    img = Image.open(image_path)
    plt.imshow(img)
    results = inference_state
    nb_objects = len(inference_state["scores"])
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = colors[i % len(colors)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "segmentation_results.png")
    plt.savefig(save_path)
    print(f"Plotted results saved to {save_path}")


def run_sam(model, image_path, output_dir, done_dir, colors):

    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="mouse")

    # img0 = Image.open(image_path)
    # plot_results(img0, inference_state)

    img_np = np.array(image)
    save_masks_as_pngs(inference_state, done_dir, img_np)
    # Save the raw image in the done_dir
    raw_img_save_path = os.path.join(done_dir, "job.png")
    image.save(raw_img_save_path)

    visualize_segmentation_results(image_path, output_dir, inference_state, colors)

#######

COLORS = generate_colors(n_colors=128, n_samples=5000)

model = build_sam3_image_model()

PATH = "/home/ferdinand/sam_project/sam_server/worker_data/sam3_worker"
job_name = "job.png"
done_name = "job.png"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
DONE_DIR = os.path.join(os.path.dirname(PATH), "sam_3d_worker", "input")
for d in [INPUT_DIR, OUTPUT_DIR, DONE_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(INPUT_DIR, job_name)
# DONE_PATH = os.path.join(OUTPUT_DIR, done_name)

print("\033[92mSam_segmentation worker ready\033[0m")


while True:
    if not os.path.exists(IMAGE_PATH):
        time.sleep(0.1)
        continue

    start_time = time.time()
    print(f"\033[95m[SAM3_WORKER] Job started\033[0m")
    run_sam(model, IMAGE_PATH, OUTPUT_DIR, DONE_DIR, COLORS)
    elapsed_time = time.time() - start_time
    print(f"\033[95m[SAM3_WORKER] Time taken: {elapsed_time:.2f} seconds\033[0m")

    os.remove(IMAGE_PATH)
    
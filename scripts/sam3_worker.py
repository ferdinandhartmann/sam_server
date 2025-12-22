# scripts/sam_segmentation_worker.py

import os, sys

from utils import ColorPrint
print = ColorPrint(worker_name="SAM3", default_color="yellow")

print("Loading libraries and model...")

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
    return True

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
    
def visualize_segmentation_results(image_path, output_dir, inference_state, colors, safe_prompt):
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
    save_path = os.path.join(output_dir, f"{safe_prompt}_segmentation_results.png")
    plt.savefig(save_path)
    print(f"Plotted results saved to {save_path}")


def run_sam(model, image_path, prompt_path, output_dir, done_dir, colors, final_output_dir):
    
    print("Starting inference...")

    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    prompt = "object"
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip() or prompt
        print(f"Using prompt from file: \"{prompt}\".")
    else:
        print(f"! No prompt file found at {prompt_path}, using default prompt \"{prompt}\".")
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # check if there are masks detected
    if len(inference_state["masks"]) == 0:
        print("No masks detected!!!!, skipping saving masks and visualization.")
        open(os.path.join(os.path.dirname(done_dir), "sam3_nomaskdetected.flag"), "a").close()
        return
    
    print(f"Detected {len(inference_state['masks'])} masks, saving masks and visualization...")

    # Save the raw image in the done_dir
    # Sanitize prompt to create a valid filename
    safe_prompt = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in prompt.lower())
    safe_prompt = safe_prompt.replace(' ', '_').strip('_')
    raw_img_save_path = os.path.join(done_dir, f"{safe_prompt}.png")
    image.save(raw_img_save_path)
    print(f"Saved raw image to {raw_img_save_path}")
    
    img_np = np.array(image)
    if save_masks_as_pngs(inference_state, done_dir, img_np) == True:
            # Create a "sam3_worker.finished" file in the done_dir
        finished_flag_path = os.path.join(done_dir, "sam3_worker_finished.flag")
        open(finished_flag_path, "a").close()
        print(f"Created finished flag at {finished_flag_path}")
    else :
        print("Error saving masks as PNGs, skipping creating finished flag.")

    visualize_segmentation_results(image_path, final_output_dir, inference_state, colors, safe_prompt)
    print("Visualization complete.")

#######

COLORS = generate_colors(n_colors=128, n_samples=5000)

model = build_sam3_image_model()

def create_all_folders():
    for d in [INPUT_DIR, OUTPUT_DIR, DONE_DIR, READY_DIR]:
        os.makedirs(d, exist_ok=True)

PATH = "/home/ferdinand/sam_project/sam_server/worker_data/sam3_worker"
job_name = "job.jpg"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
DONE_DIR = os.path.join(os.path.dirname(PATH), "sam_3d_worker", "input")
READY_DIR = os.path.join(os.path.dirname(PATH), "workers_ready")
create_all_folders()
PROMPT_PATH = os.path.join(INPUT_DIR, "prompt.txt")
FINAL_OUTPUT_DIR = os.path.join(os.path.dirname(PATH), "final_output")
IMAGE_PATH = os.path.join(INPUT_DIR, job_name)


open(os.path.join(READY_DIR, "sam3_worker.ready"), "a").close()
print("Ready! Waiting for jobs...")


while True:
    if( not os.path.exists(INPUT_DIR) ):
        create_all_folders()
        time.sleep(0.1)
        continue
    
    if not os.path.exists(IMAGE_PATH):
        time.sleep(0.1)
        continue

    start_time = time.time()
    print(f"Job started")
    
    done_flag_path = os.path.join(OUTPUT_DIR, "done.flag")
    if os.path.exists(done_flag_path):
        os.remove(done_flag_path)
    
    run_sam(model, IMAGE_PATH, PROMPT_PATH, OUTPUT_DIR, DONE_DIR, COLORS, FINAL_OUTPUT_DIR)
    
    elapsed_time = time.time() - start_time
    print(f"Job finished! ({elapsed_time:.2f})s")
    
    # open(os.path.join(OUTPUT_DIR, "done.flag"), "a").close()

    os.remove(IMAGE_PATH)
    if os.path.exists(PROMPT_PATH):
        os.remove(PROMPT_PATH)
    
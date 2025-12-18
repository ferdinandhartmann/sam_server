# sam_3d_worker.py

import sys, os

print("Loading SAM3D libraries and model...")

import os
import imageio
import uuid
from IPython.display import Image as ImageDisplay
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

def save_gif(model_output, output_dir, image_name):
    # render gaussian splat
    scene_gs = make_scene(model_output)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    video = render_video(
        scene_gs,
        r=1,
        fov=60,
        pitch_deg=15,
        yaw_start_deg=-45,
        resolution=512,
    )["color"]

    # save video as gif
    imageio.mimsave(
        os.path.join(f"{output_dir}/{image_name}.gif"),
        video,
        format="GIF",
        duration=1000 / 20,  # default assuming 20fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )
    

PATH = "/home/ferdinand/sam_project/sam_server/sam_3d_worker"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = "/home/ferdinand/sam_project/sam-3d-objects/checkpoints/hf/pipeline.yaml"

IMAGE_PATH = f"{INPUT_DIR}/input_image.png"


print("SAM-3D worker ready")


inference = Inference(config_path, compile=False)

######

IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))
image = load_image(IMAGE_PATH, convert_rgb=True)
mask = load_single_mask(INPUT_DIR, index=0)
# display_image(image, masks=[mask])

######

# run model
model_output = inference(image, mask, seed=42)

# export gaussian splat (as point cloud)
model_output["gs"].save_ply(f"{OUTPUT_DIR}/{IMAGE_NAME}.ply")

# render and save gif
save_gif(model_output, OUTPUT_DIR, IMAGE_NAME)





while True:
    if not os.path.exists(os.path.join(INPUT_DIR, "job.txt")):
        time.sleep(0.1)
        continue

    with open(os.path.join(INPUT_DIR, "job.txt")) as f:
        image_path, out_path = f.read().strip().split(",")

    run_sam(image_path, out_path)

    os.remove(os.path.join(INPUT_DIR, "job.txt"))
    open(os.path.join(OUTPUT_DIR, "done.txt"), "w").close()
    
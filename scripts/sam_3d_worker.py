# scripts/sam_3d_worker.py

import sys, os

print("\033[92mLoading SAM3D libraries and model...\033[0m")

sys.path.insert(0, "/home/ferdinand/sam_project/sam-3d-objects/notebook")

import time
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
    

def run_sam3d(config_path, image_path, done_dir, output_dir):
        
    inference = Inference(config_path, compile=False)

    ######

    IMAGE_NAME = os.path.basename(os.path.dirname(image_path))
    image = load_image(image_path, convert_rgb=True)
    input_dir = os.path.dirname(image_path)
    mask = load_single_mask(input_dir, index=0)
    # display_image(image, masks=[mask])

    ######

    # run model
    model_output = inference(image, mask, seed=42)

    # export gaussian splat (as point cloud)
    model_output["gs"].save_ply(f"{done_dir}/job.ply")

    # render and save gif
    save_gif(model_output, output_dir, "splatting_visualization")



config_path = "/home/ferdinand/sam_project/sam-3d-objects/checkpoints/hf/pipeline.yaml"

PATH = "/home/ferdinand/sam_project/sam_server/worker_data/sam_3d_worker"
job_name = "job.png"
done_name = "job.png"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
DONE_DIR = os.path.join(os.path.dirname(PATH), "mesh_worker", "input")
for d in [INPUT_DIR, OUTPUT_DIR, DONE_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(INPUT_DIR, job_name)
# DONE_PATH = os.path.join(OUTPUT_DIR, done_name)

print("\033[92mSAM-3D worker ready\033[0m")


while True:
    if not os.path.exists(IMAGE_PATH):
        time.sleep(0.1)
        continue

    start_time = time.time()
    print(f"\033[95m[SAM_3D_WORKER] Job started\033[0m")
    run_sam3d(config_path, IMAGE_PATH, DONE_DIR, OUTPUT_DIR)
    elapsed_time = time.time() - start_time
    print(f"\033[95m[SAM_3D_WORKER] Time taken: {elapsed_time:.2f} seconds\033[0m")
    
    # Remove all files in the input folder
    for f in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
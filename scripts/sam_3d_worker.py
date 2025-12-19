# scripts/sam_3d_worker.py

import sys, os

from utils import ColorPrint
print = ColorPrint(worker_name="SAM_3D", default_color="orange")

print("Loading libraries and model...")

sys.path.insert(0, "/home/ferdinand/sam_project/sam-3d-objects/notebook")

import time
# import imageio
# import uuid
# from IPython.display import Image as ImageDisplay
# from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer
import trimesh

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
    
def clean_mesh(m):
    # remove degenerate faces via mask
    mask = m.nondegenerate_faces()
    if mask is not None:
        m = m.submesh([mask], append=True)

    # remove unused vertices
    m.remove_unreferenced_vertices()

    # final validation
    m.process(validate=True)
    return m

def rescale_to_match(source: trimesh.Trimesh, target: trimesh.Trimesh):
    # Compute bounding boxes
    src_extents = source.bounding_box.extents
    tgt_extents = target.bounding_box.extents

    # Uniform scale factor (preserve proportions)
    scale = (src_extents / tgt_extents).min()

    target.apply_scale(scale)

    # Align centers
    src_center = source.bounding_box.centroid
    tgt_center = target.bounding_box.centroid
    target.apply_translation(src_center - tgt_center)

    return target

def create_voxel_collision_mesh(mesh, voxel_scale=64.0):

    # Voxel resolution control (lower = coarser)
    voxel_pitch = mesh.scale / voxel_scale  # try 64, 128, 256

    vox = mesh.voxelized(pitch=voxel_pitch)
    vox = vox.fill()
    # 3. Reconstruct surface
    collision = vox.marching_cubes
    collision = clean_mesh(collision)
    collision = rescale_to_match(mesh, collision)

    print(f"Collision mesh: {len(collision.faces)} faces")
    print("Collision watertight:", collision.is_watertight)
    assert collision.is_watertight, "Collision mesh is NOT watertight!"
    # print("Collision Euler number:", collision.euler_number)
    # print("Collision volume:", collision.volume)
    
    return collision

def create_convex_hull_mesh(mesh, reduce_percent=0.9):
    # Simplify mesh to reduce number of faces
    simplified_mesh = mesh.simplify_quadric_decimation(reduce_percent)
    convexhull = simplified_mesh.convex_hull
    convexhull = clean_mesh(convexhull)
    convexhull = rescale_to_match(mesh, convexhull)
    print(f"Convex hull mesh has {len(convexhull.faces)} faces")
    return convexhull


def run_sam3d(config_path, image_path, done_dir, output_dir):
    
    print("Starting inference...")
        
    # inference = Inference(config_path, compile=False)

    # ######

    # IMAGE_NAME = os.path.basename(os.path.dirname(image_path))
    # image = load_image(image_path, convert_rgb=True)
    # input_dir = os.path.dirname(image_path)
    # mask = load_single_mask(input_dir, index=0)
    # # display_image(image, masks=[mask])

    # ######

    # # run model
    # model_output = inference(image, mask, seed=42)
    
    # WITH_MESH_POSTPROCESS = True
    # WITH_TEXTURE_BAKING = True
    # model_output = inference._pipeline.postprocess_slat_output(
    #     model_output,
    #     with_mesh_postprocess=WITH_MESH_POSTPROCESS,
    #     with_texture_baking=WITH_TEXTURE_BAKING,
    #     use_vertex_color=not WITH_TEXTURE_BAKING,
    # )
    
    # mesh = model_output["glb"]  # trimesh object
    mesh_path = os.path.join(output_dir, "object_mesh.glb")
    # mesh.export(mesh_path)
    # print(f"Exported .glb mesh")
    
    # Import and export to .obj with texture and material
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.export(os.path.join(done_dir, "object_visual.obj"))
    print(f"Exported visual mesh")    
    
    # Ensure mesh is a single unified mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    print(f"Mesh has {len(mesh.faces)} faces")
        
    # create convex hull
    create_convex_hull_mesh(mesh, reduce_percent=0.93).export(os.path.join(done_dir, "object_collision.obj"))    
    print(f"Exported convex hull collision mesh")
    
    # # create voxel-based watertight collision mesh
    # create_voxel_collision_mesh(mesh, voxel_scale=64.0).export(os.path.join(done_dir, "collision.obj"))


    # export gaussian splat (as point cloud) and gif visualization
    model_output["gs"].save_ply(f"{output_dir}/gsplat.ply")
    save_gif(model_output, output_dir, "splatting_visualization")
    print(f"Exported gaussian splat and gif visualization")



config_path = "/home/ferdinand/sam_project/sam-3d-objects/checkpoints/hf/pipeline.yaml"

PATH = "/home/ferdinand/sam_project/sam_server/worker_data/sam_3d_worker"
job_name = "job.png"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
DONE_DIR = os.path.join(os.path.dirname(PATH), "final_output")
for d in [INPUT_DIR, OUTPUT_DIR, DONE_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(INPUT_DIR, job_name)

print("Ready! Waiting for jobs...")


while True:
    if not os.path.exists(IMAGE_PATH):
        time.sleep(0.1)
        continue

    start_time = time.time()
    print(f"Job started")
    
    run_sam3d(config_path, IMAGE_PATH, DONE_DIR, OUTPUT_DIR)
    
    elapsed_time = time.time() - start_time
    print(f"Job finished! ({elapsed_time:.2f})s")
    
    # Remove all files in the input folder
    for f in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

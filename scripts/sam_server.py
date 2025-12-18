from fastapi import FastAPI, UploadFile
import subprocess
import uuid
import os
import shutil

# Absolute paths to Python executables
SAM3_PY   = "/home/ferdinand/miniforge3/envs/sam3/bin/python"
SAM3D_PY = "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python"
BASE_PY  = "/home/ferdinand/miniforge3/bin/python"

BASE_JOBS = "/home/ferdinand/sam_server/jobs"
os.makedirs(BASE_JOBS, exist_ok=True)

app = FastAPI()


@app.post("/reconstruct")
async def reconstruct(image: UploadFile):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_JOBS, job_id)
    os.makedirs(job_dir)

    # ---- save input ----
    img_path = os.path.join(job_dir, "input.png")
    with open(img_path, "wb") as f:
        f.write(await image.read())

    mask_path = os.path.join(job_dir, "mask.png")
    ply_path  = os.path.join(job_dir, "object.ply")
    obj_path  = os.path.join(job_dir, "object.obj")
    stl_path  = os.path.join(job_dir, "object.stl")
    png_path  = os.path.join(job_dir, "preview.png")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU

    # ---- SAM ----
    subprocess.run([
        SAM3_PY, "scripts/run_sam.py",
        "--image", img_path,
        "--out", mask_path
    ], check=True, env=env)

    # ---- SAM-3D ----
    subprocess.run([
        SAM3D_PY, "scripts/run_sam3d.py",
        "--image", img_path,
        "--mask", mask_path,
        "--out", ply_path
    ], check=True, env=env)

    # ---- Mesh conversion ----
    subprocess.run([
        BASE_PY, "scripts/convert_ply.py",
        "--in", ply_path,
        "--out-obj", obj_path,
        "--out-stl", stl_path
    ], check=True)

    # ---- Preview render ----
    subprocess.run([
        BASE_PY, "scripts/render_preview.py",
        "--mesh", obj_path,
        "--out", png_path
    ], check=True)

    return {
        "job_id": job_id,
        "obj": obj_path,
        "stl": stl_path,
        "png": png_path
    }

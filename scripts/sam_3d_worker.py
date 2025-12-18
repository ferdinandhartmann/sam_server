"""3D reconstruction worker.

Loads the SAM-3D model once, then processes jobs after segmentation masks are
available. It exports a PLY file for downstream conversion and a GIF preview to
help with debugging.
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio
from inference import (
    Inference,
    load_image,
    load_single_mask,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)

from config import DEFAULT_SAM3D_CONFIG, RECONSTRUCTION_STAGE, WORKER_LOG_PREFIX, WORKER_POLL_SECONDS
from job_state import claim_next_job, prerequisites_for, set_artifacts, set_stage_status


import sys
print("[sam3d] PYTHONPATH:", sys.path)


def save_gif(model_output, output_dir: Path, image_name: str) -> Path:
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

    gif_path = output_dir / f"{image_name}.gif"
    imageio.mimsave(
        gif_path,
        video,
        format="GIF",
        duration=1000 / 20,
        loop=0,
    )
    return gif_path


def reconstruct(job_dir: Path, inference: Inference) -> None:
    image_path = job_dir / "input.png"
    mask_dir = job_dir / "segmentation"
    output_dir = job_dir
    output_dir.mkdir(exist_ok=True)

    image = load_image(str(image_path), convert_rgb=True)
    mask = load_single_mask(str(mask_dir), index=0)

    model_output = inference(image, mask, seed=42)

    ply_path = output_dir / "object.ply"
    model_output["gs"].save_ply(str(ply_path))

    gif_path = save_gif(model_output, output_dir, "preview")

    set_artifacts(job_dir, ply=str(ply_path), reconstruction_preview=str(gif_path))


def main() -> None:
    print(f"{WORKER_LOG_PREFIX} [sam-3d] loading config from {DEFAULT_SAM3D_CONFIG}")
    inference = Inference(DEFAULT_SAM3D_CONFIG, compile=False)
    print(f"{WORKER_LOG_PREFIX} [sam-3d] model ready")

    while True:
        job_dir = claim_next_job(RECONSTRUCTION_STAGE, prerequisites_for(RECONSTRUCTION_STAGE))
        if not job_dir:
            time.sleep(WORKER_POLL_SECONDS)
            continue

        try:
            reconstruct(job_dir, inference)
        except Exception as exc:  # noqa: BLE001 - worker needs to surface failures
            print(f"{WORKER_LOG_PREFIX} [sam-3d] failed: {exc}")
            set_stage_status(job_dir, RECONSTRUCTION_STAGE, "failed", error=str(exc))
        else:
            set_stage_status(job_dir, RECONSTRUCTION_STAGE, "done")


if __name__ == "__main__":
    main()
    

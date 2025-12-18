from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, UploadFile

from config import (
    CONVERSION_STAGE,
    JOBS_ROOT,
    SERVER_HOST,
    SERVER_PORT,
    WORKER_LOG_PREFIX,
)
from job_state import (
    create_job_directory,
    current_artifacts,
    load_state,
    set_stage_status,
    wait_for_stage,
)
from worker_manager import WorkerManager

manager = WorkerManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"{WORKER_LOG_PREFIX} starting worker subprocesses")
    manager.start()
    try:
        yield
    finally:
        print(f"{WORKER_LOG_PREFIX} stopping worker subprocesses")
        manager.stop()


app = FastAPI(lifespan=lifespan)


def _job_dir(job_id: str) -> Path:
    candidate = JOBS_ROOT / job_id
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return candidate


@app.post("/reconstruct")
async def reconstruct(image: UploadFile) -> Dict[str, str]:
    job_dir = create_job_directory()
    input_path = job_dir / "input.png"

    with open(input_path, "wb") as f:
        f.write(await image.read())

    success = await asyncio.to_thread(
        wait_for_stage,
        job_dir,
        CONVERSION_STAGE,
        timeout_seconds=3600,
        poll_seconds=1.0,
    )

    if not success:
        state = load_state(job_dir)
        failed_stage = next(
            (
                (stage, details)
                for stage, details in state["stages"].items()
                if details.get("status") == "failed"
            ),
            None,
        )
        error_message = (
            failed_stage[1].get("error")
            or f"Stage {failed_stage[0]} failed"
            if failed_stage
            else "Job did not finish"
        )
        set_stage_status(job_dir, CONVERSION_STAGE, "failed", error=error_message)
        raise HTTPException(status_code=500, detail=error_message)

    artifacts = current_artifacts(job_dir)
    return {
        "job_id": job_dir.name,
        "obj": artifacts.get("visual_obj"),
        "stl": artifacts.get("visual_stl"),
        "preview": artifacts.get("reconstruction_preview"),
    }


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> Dict:
    job_dir = _job_dir(job_id)
    return load_state(job_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.sam_server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)

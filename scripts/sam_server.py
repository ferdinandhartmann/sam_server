from fastapi import FastAPI, UploadFile, Form, HTTPException
from pathlib import Path
import uuid
import shutil
import subprocess
import time
import threading
import os
from fastapi.responses import FileResponse
from collections import defaultdict

from scripts.utils import ColorPrint
print = ColorPrint(worker_name="SAM_SERVER", default_color="magenta")

READY_DIR = Path("worker_data/workers_ready")
JOBS = Path("worker_data")
JOBS.mkdir(exist_ok=True)
old_job_id = None

# Track downloaded files for each job
job_download_tracker = defaultdict(set)

def delete_worker_data():
    worker_data_dir = "worker_data"
    if os.path.exists(worker_data_dir):
        for folder in os.listdir(worker_data_dir):
            folder_path = os.path.join(worker_data_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)
    else:
        print(f"Folder not found: {worker_data_dir}")

delete_worker_data()

app = FastAPI()

def launch_workers():
    global workers_ready
    print("Launching worker processes...")
    subprocess.Popen(["python3", "scripts/start_workers.py"])
    while not all((READY_DIR / f).exists() for f in ["sam3_worker.ready", "sam_3d_worker.ready"]):
        print("Waiting for workers to be ready...")
        time.sleep(2.0)
    print("All workers are ready!")

@app.on_event("startup")
def startup():
    print("Starting up FastAPI server...")
    threading.Thread(target=launch_workers, daemon=True).start()

@app.get("/ready")
def ready():
    ready = all(
        (READY_DIR / f).exists()
        for f in ["sam3_worker.ready", "sam_3d_worker.ready"]
    )
    print(f"Ready check: {ready}")
    return {"ready": ready}


def archive_and_clear_worker_data(prompt_name):
    worker_data_dir = Path("worker_data")
    archive_dir = Path(f"worker_data_finished/worker_data_{prompt_name}")

    if worker_data_dir.exists():
        # Create the archive directory
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy worker_data to the archive directory
        shutil.copytree(worker_data_dir, archive_dir, dirs_exist_ok=True)
        print(f"Archived worker data to {archive_dir}")

        # Clear the worker_data directory
        for folder in worker_data_dir.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)
            else:
                folder.unlink()
        
        READY_DIR = Path("worker_data/workers_ready")
        READY_DIR.mkdir(exist_ok=True)
        
        print("Cleared worker_data directory.")
        
    else:
        print("worker_data directory does not exist.")

@app.post("/submit")
async def submit(image: UploadFile, prompt: str = Form(...)):
    if not all(
        (READY_DIR / f).exists()
        for f in ["sam3_worker.ready", "sam_3d_worker.ready"]
    ):
        print("Workers not ready, rejecting job submission.")
        raise HTTPException(503, "Workers not ready")
    
    if old_job_id is not None:
        archive_and_clear_worker_data(old_job_id)

    job_id = str(uuid.uuid4())
    job_dir = JOBS / "sam3_worker/input"
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"Received job {job_id}, saving image and prompt...")

    with open(job_dir / "job.jpg", "wb") as f:
        shutil.copyfileobj(image.file, f)

    (job_dir / "prompt.txt").write_text(prompt)
    print(f"Job {job_id} submitted successfully.")

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    out = JOBS / "sam_3d_worker/output"
    if (out / "done.flag").exists():
        print(f"Job {job_id} is done.")
        return {"status": "done"}
    elif (out / "sam3_nomaskdetected.flag").exists():
        print(f"Job {job_id} completed with no masks detected.")
        return {"status": "no_masks_detected"}
    else:
        print(f"Job {job_id} is still processing.")
        return {"status": "processing"}

@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    file_path = JOBS / "final_output" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")  
    print(f"Download requested for job {job_id}, file {filename}")

    # Track the downloaded file
    job_download_tracker[job_id].add(filename)

    # Check if all files for the job have been downloaded
    job_output = JOBS / "final_output"
    all_files = {f.name for f in job_output.iterdir() if f.is_file()}
    if job_download_tracker[job_id] == all_files:
        print(f"All files for job {job_id} have been downloaded.")
        old_job_id = job_id
        # remove the done flag so the job no longer reports as done
        done_flag = JOBS / "sam_3d_worker/output" / "done.flag"
        if done_flag.exists():
            done_flag.unlink()
            print(f"Removed done.flag for job {job_id}")

        # set module-level old_job_id for later archiving (avoid 'global' after assignment)
        globals()['old_job_id'] = job_id

        # cleanup download tracker for this job
        job_download_tracker.pop(job_id, None)

    return FileResponse(path=file_path, filename=filename)

@app.get("/list/{job_id}")
def list_files(job_id: str):
    job_output = JOBS / "final_output"
    if not job_output.exists():
        return {"files": []}
    files = [f.name for f in job_output.iterdir() if f.is_file()]
    print(f"Listing files for job {job_id}: {files}")
    return {"files": files}

@app.get("/health")
def health():
    """Simple health check for the server"""
    return {"status": "ok"}

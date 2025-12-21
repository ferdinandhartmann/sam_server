# scripts/start_all_workers.py

import subprocess
import signal
import sys
import shutil
import os

from utils import ColorPrint
print = ColorPrint(worker_name="All Worker Starter", default_color="cyan")


SAM3_PY   = "/home/ferdinand/miniforge3/envs/sam3/bin/python"
SAM3D_PY  = "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python"

workers = [
    [SAM3_PY, "scripts/sam3_worker.py"],
    [SAM3D_PY, "scripts/sam_3d_worker.py"],
]

# Delete all existing folders in worker_data directory
worker_data_dir = "worker_data"
if os.path.exists(worker_data_dir):
    for folder in os.listdir(worker_data_dir):
        folder_path = os.path.join(worker_data_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Deleting folder: {folder_path}")
            shutil.rmtree(folder_path)
else:
    print(f"Folder not found: {worker_data_dir}")

procs = []

def shutdown(sig, frame):
    print("Shutting down workers...")
    for p in procs:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

for cmd in workers:
    print("Starting:", cmd)
    procs.append(subprocess.Popen(cmd))

print("All workers running.")
signal.pause()

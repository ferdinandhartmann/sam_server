# scripts/start_all_workers.py

import subprocess
import signal
import sys
import shutil
import os

SAM3_PY   = "/home/ferdinand/miniforge3/envs/sam3/bin/python"
SAM3D_PY  = "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python"
BASE_PY   = "/home/ferdinand/miniforge3/bin/python"

workers = [
    [SAM3_PY, "scripts/sam3_worker.py"],
    [SAM3D_PY, "scripts/sam_3d_worker.py"],
    [BASE_PY, "scripts/mesh_worker.py"],
]

folders_to_delete = [
    "mesh_worker",
    "sam_3d_worker",
    "sam3_worker"
]

for folder in folders_to_delete:
    if os.path.exists(folder):
        print(f"Deleting folder: {folder}")
        shutil.rmtree(folder)
    else:
        print(f"Folder not found: {folder}")

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

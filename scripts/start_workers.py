import subprocess
import signal
import sys
from pathlib import Path
import time
import os
import shutil

from utils import ColorPrint
print = ColorPrint(worker_name="All Worker Starter", default_color="purple")

SAM3_PY   = "/home/ferdinand/miniforge3/envs/sam3/bin/python"
SAM3D_PY  = "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python"

WORKERS = {
    "sam3":   [SAM3_PY, "scripts/sam3_worker.py"],
    "sam3d":  [SAM3D_PY, "scripts/sam_3d_worker.py"],
}

READY_DIR = Path("worker_data/workers_ready")
READY_DIR.mkdir(exist_ok=True)

procs = {}

def start_all():
    for name, cmd in WORKERS.items():
        print(f"Starting {name}")
        procs[name] = subprocess.Popen(cmd)

def wait_until_ready():
    print("Waiting for workers...")
    while True:
        ready = all((READY_DIR / f"{name}.ready").exists() for name in WORKERS)
        if ready:
            print("All workers ready")
            return
        time.sleep(0.5)

def shutdown(sig, frame):
    print("Stopping workers")
    for p in procs.values():
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    start_all()
    wait_until_ready()

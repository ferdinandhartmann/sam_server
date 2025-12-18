import subprocess
import signal
import sys

SAM3_PY   = "/home/ferdinand/miniforge3/envs/sam3/bin/python"
SAM3D_PY  = "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python"
BASE_PY   = "/home/ferdinand/miniforge3/bin/python"

workers = [
    [SAM3_PY, "sam_worker.py"],
    [SAM3D_PY, "sam3d_worker.py"],
    [BASE_PY, "mesh_worker.py"],
]


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

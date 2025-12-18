"""Start and stop dedicated worker processes in their configured environments."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

from config import CONVERTER_PYTHON, RECONSTRUCTION_PYTHON, SEGMENTATION_PYTHON, WORKER_LOG_PREFIX, ROOT_DIR

WORKERS: List[List[str]] = [
    [SEGMENTATION_PYTHON, str(Path(__file__).resolve().parent / "sam_segmentation_worker.py")],
    [RECONSTRUCTION_PYTHON, str(Path(__file__).resolve().parent / "sam_3d_worker.py")],
    [CONVERTER_PYTHON, str(Path(__file__).resolve().parent / "converter_worker.py")],
]


class WorkerManager:
    def __init__(self) -> None:
        self._procs: List[subprocess.Popen] = []

    def start(self) -> None:
        for cmd in WORKERS:
            print(f"{WORKER_LOG_PREFIX} starting {' '.join(cmd)}")
            env = os.environ.copy()
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            
            env["PYTHONPATH"] = (
                "/home/ferdinand/sam_project/sam3:"
                "/home/ferdinand/sam_project/sam-3d-objects:"
                + env.get("PYTHONPATH", "")
            )
                        
            self._procs.append(
                subprocess.Popen(cmd, env=env, cwd=ROOT_DIR)
            )

    def stop(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                print(f"{WORKER_LOG_PREFIX} stopping pid {proc.pid}")
                proc.terminate()
        for proc in self._procs:
            if proc.poll() is None:
                proc.wait(timeout=10)
        self._procs.clear()

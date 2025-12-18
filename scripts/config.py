"""Shared configuration for the SAM worker pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# Base repo directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Job storage
JOBS_ROOT = Path(os.environ.get("SAM_SERVER_JOBS", ROOT_DIR / "jobs"))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# Python executables for each worker
SEGMENTATION_PYTHON = os.environ.get("SAM_SEGMENTATION_PYTHON", "/home/ferdinand/miniforge3/envs/sam3/bin/python")
RECONSTRUCTION_PYTHON = os.environ.get("SAM_3D_PYTHON", "/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python")
# CONVERTER_PYTHON = os.environ.get("SAM_CONVERTER_PYTHON", "/home/ferdinand/miniforge3/envs/sam_server/bin/python")
CONVERTER_PYTHON = os.environ.get("SAM_CONVERTER_PYTHON", "/home/ferdinand/miniforge3/envs/sam3/bin/python")

# Worker behavior
WORKER_POLL_SECONDS = float(os.environ.get("SAM_WORKER_POLL_SECONDS", "1.0"))
WORKER_LOG_PREFIX = os.environ.get("SAM_WORKER_LOG_PREFIX", "[worker]")

# Application settings
SERVER_HOST = os.environ.get("SAM_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SAM_SERVER_PORT", "8000"))

# Model configuration paths
DEFAULT_SAM3D_CONFIG = os.environ.get("/home/ferdinand/sam_project/sam-3d-objects/checkpoints/hf/pipeline.yaml"),


# Stage names used across workers
SEGMENTATION_STAGE = "segmentation"
RECONSTRUCTION_STAGE = "reconstruction"
CONVERSION_STAGE = "conversion"
ALL_STAGES = [SEGMENTATION_STAGE, RECONSTRUCTION_STAGE, CONVERSION_STAGE]

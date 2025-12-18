"""Helpers for reading and writing job state files shared by the workers."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, Optional

from config import ALL_STAGES, CONVERSION_STAGE, JOBS_ROOT, RECONSTRUCTION_STAGE, SEGMENTATION_STAGE

STATE_FILE = "state.json"


def _state_path(job_dir: Path) -> Path:
    return job_dir / STATE_FILE


def _atomic_write(path: Path, data: Dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def _blank_state(job_id: str) -> Dict:
    return {
        "job_id": job_id,
        "stages": {
            stage: {"status": "pending", "error": None} for stage in ALL_STAGES
        },
        "artifacts": {},
    }


def create_job_directory() -> Path:
    job_id = str(uuid.uuid4())
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    _atomic_write(_state_path(job_dir), _blank_state(job_id))
    return job_dir


def load_state(job_dir: Path) -> Dict:
    return json.loads(_state_path(job_dir).read_text())


def update_state(job_dir: Path, state: Dict) -> None:
    _atomic_write(_state_path(job_dir), state)


def set_stage_status(job_dir: Path, stage: str, status: str, *, error: Optional[str] = None) -> None:
    state = load_state(job_dir)
    state["stages"][stage]["status"] = status
    state["stages"][stage]["error"] = error
    update_state(job_dir, state)


def set_artifacts(job_dir: Path, **artifacts: str) -> None:
    state = load_state(job_dir)
    state["artifacts"].update(artifacts)
    update_state(job_dir, state)


def claim_next_job(stage: str, prerequisites: Iterable[str]) -> Optional[Path]:
    for job_dir in sorted(JOBS_ROOT.iterdir()):
        if not job_dir.is_dir():
            continue
        state = load_state(job_dir)
        stage_state = state["stages"].get(stage)
        if not stage_state or stage_state["status"] != "pending":
            continue
        if not all(state["stages"].get(req, {}).get("status") == "done" for req in prerequisites):
            continue
        stage_state["status"] = "in_progress"
        update_state(job_dir, state)
        return job_dir
    return None


def stage_complete(job_dir: Path, stage: str) -> bool:
    return load_state(job_dir)["stages"].get(stage, {}).get("status") == "done"


def wait_for_stage(job_dir: Path, stage: str, *, timeout_seconds: float, poll_seconds: float = 0.5) -> bool:
    import time

    end_time = time.time() + timeout_seconds
    while time.time() < end_time:
        state = load_state(job_dir)
        status = state["stages"].get(stage, {}).get("status")
        if status == "done":
            return True
        if status == "failed":
            return False
        if any(details.get("status") == "failed" for details in state["stages"].values()):
            return False
        time.sleep(poll_seconds)
    return False


def current_artifacts(job_dir: Path) -> Dict[str, str]:
    return load_state(job_dir).get("artifacts", {})


def prerequisites_for(stage: str) -> Iterable[str]:
    if stage == SEGMENTATION_STAGE:
        return []
    if stage == RECONSTRUCTION_STAGE:
        return [SEGMENTATION_STAGE]
    if stage == CONVERSION_STAGE:
        return [RECONSTRUCTION_STAGE]
    return []

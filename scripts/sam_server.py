# scripts/sam_server.py

import argparse
import atexit
import cgi
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config_utils import load_config


class WorkerCoordinator:
    """Manage worker processes and current job state."""

    def __init__(self, base_dir=None):
        base_path = Path(base_dir).expanduser() if base_dir else Path(
            "/home/ferdinand/sam_project/sam_server/worker_data"
        )
        self.base_dir = base_path
        self.sam3_input = self.base_dir / "sam3_worker" / "input" / "job.png"
        self.sam3_prompt = self.sam3_input.with_name("prompt.txt")
        self.sam3d_input = self.base_dir / "sam_3d_worker" / "input" / "job.png"
        self.mesh_input = self.base_dir / "mesh_worker" / "input" / "job.ply"
        self.mesh_results = self.base_dir / "mesh_worker" / "results"
        self.jobs_dir = Path.cwd() / "server_jobs"
        self.jobs_dir.mkdir(exist_ok=True)

        for folder in [self.sam3_input.parent, self.sam3d_input.parent, self.mesh_input.parent, self.mesh_results]:
            folder.mkdir(parents=True, exist_ok=True)

        self._active_job = None
        self._lock = threading.Lock()
        self.status = {}
        self.worker_process = self._start_workers()
        atexit.register(self.shutdown)

    def _start_workers(self):
        """Spawn the worker supervisor script."""
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "start_all_workers.py"),
        ]
        print(f"Starting worker supervisor: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def shutdown(self):
        if self.worker_process and self.worker_process.poll() is None:
            print("Terminating worker supervisor...")
            try:
                self.worker_process.send_signal(signal.SIGTERM)
                self.worker_process.wait(timeout=10)
            except Exception:
                self.worker_process.kill()

    def _copy_results(self, job_id):
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(exist_ok=True)
        artifacts = {
            "ply": self.mesh_results / "job.ply",
            "visual_obj": self.mesh_results / "object_visual.obj",
            "collision_obj": self.mesh_results / "object_collision.obj",
        }

        copied = []
        for name, path in artifacts.items():
            if path.exists():
                target = job_dir / f"{job_id}_{name}{path.suffix}"
                shutil.copy2(path, target)
                copied.append(target)
        return job_dir, copied

    def _zip_results(self, job_dir, artifacts):
        zip_path = job_dir / f"{job_dir.name}_results.zip"
        shutil.make_archive(zip_path.with_suffix(""), "zip", job_dir)
        return zip_path

    def _wait_for_file(self, path, stage_name, job_id, timeout=3600):
        start = time.time()
        while time.time() - start < timeout:
            if path.exists():
                return True
            time.sleep(0.5)
        self.status[job_id] = {"state": "failed", "message": f"Timeout waiting for {stage_name}"}
        return False

    def submit_job(self, image_bytes, prompt_text):
        with self._lock:
            if self._active_job:
                raise RuntimeError("Another job is still running. Try again later.")
            job_id = uuid.uuid4().hex
            self._active_job = job_id

        self.status[job_id] = {"state": "received", "message": "Job accepted"}

        def _runner():
            try:
                # Write inputs for SAM3 worker
                self.sam3_input.write_bytes(image_bytes)
                self.sam3_prompt.write_text(prompt_text or "", encoding="utf-8")
                self.status[job_id] = {"state": "sam3", "message": "Waiting for SAM3 segmentation"}

                if not self._wait_for_file(self.sam3d_input, "SAM3 masks", job_id):
                    return

                self.status[job_id] = {"state": "sam3d", "message": "Waiting for 3D reconstruction"}
                if not self._wait_for_file(self.mesh_input, "SAM-3D output", job_id):
                    return

                self.status[job_id] = {"state": "meshing", "message": "Waiting for mesh conversion"}
                visual_obj = self.mesh_results / "object_visual.obj"
                collision_obj = self.mesh_results / "object_collision.obj"
                if not self._wait_for_file(visual_obj, "mesh results", job_id):
                    return
                if not self._wait_for_file(collision_obj, "mesh results", job_id):
                    return

                job_dir, artifacts = self._copy_results(job_id)
                zip_path = self._zip_results(job_dir, artifacts)
                self.status[job_id] = {
                    "state": "finished",
                    "message": "Job completed",
                    "zip_path": str(zip_path),
                }
            finally:
                with self._lock:
                    self._active_job = None

        threading.Thread(target=_runner, daemon=True).start()
        return job_id

    def get_status(self, job_id):
        return self.status.get(job_id)


class SamRequestHandler(BaseHTTPRequestHandler):
    manager: WorkerCoordinator = None

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/job":
            self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)
            return

        ctype, _ = cgi.parse_header(self.headers.get("content-type"))
        if ctype != "multipart/form-data":
            self._send_json({"error": "Use multipart/form-data"}, status=HTTPStatus.BAD_REQUEST)
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        image_field = form["image"] if "image" in form else None
        prompt_field = form["prompt"] if "prompt" in form else None
        if not image_field or not image_field.file:
            self._send_json({"error": "Image file missing"}, status=HTTPStatus.BAD_REQUEST)
            return

        prompt_text = prompt_field.value if prompt_field is not None else ""
        image_bytes = image_field.file.read()

        try:
            job_id = self.manager.submit_job(image_bytes, prompt_text)
        except RuntimeError as exc:  # another job in progress
            self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            return

        self._send_json({"job_id": job_id, "status": "queued"}, status=HTTPStatus.ACCEPTED)

    def do_GET(self):
        if self.path.startswith("/status/"):
            job_id = self.path.split("/status/")[-1]
            status = self.manager.get_status(job_id)
            if not status:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(status)
            return

        if self.path.startswith("/result/"):
            job_id = self.path.split("/result/")[-1]
            status = self.manager.get_status(job_id)
            if not status or status.get("state") != "finished":
                self._send_json({"error": "Result not ready"}, status=HTTPStatus.NOT_FOUND)
                return

            zip_path = Path(status["zip_path"])
            if not zip_path.exists():
                self._send_json({"error": "Artifacts missing"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            data = zip_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f"attachment; filename={zip_path.name}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        return  # silence default logging


def run_server(host="0.0.0.0", port=8000, base_dir=None):
    SamRequestHandler.manager = WorkerCoordinator(base_dir=base_dir)
    with ThreadingHTTPServer((host, port), SamRequestHandler) as httpd:
        print(f"Serving SAM pipeline on http://{host}:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="SAM worker orchestrator")
    parser.add_argument("--config", help="Path to config.json", default=None)
    parser.add_argument("--host", help="Bind host (overrides config)")
    parser.add_argument("--port", type=int, help="Bind port (overrides config)")
    parser.add_argument("--base-dir", help="Worker data directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config).get("server", {})
    host = args.host or cfg.get("host", "0.0.0.0")
    port = args.port or int(cfg.get("port", 8000))
    base_dir = args.base_dir or cfg.get("base_dir")

    run_server(host=host, port=port, base_dir=base_dir)


if __name__ == "__main__":
    main()

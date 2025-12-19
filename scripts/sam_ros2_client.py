"""
ROS2 node that mirrors the behaviour of the standalone SAM HTTP client.

Parameters:
- server_url (string): URL of the SAM server (default: http://localhost:8000).
- watch_dir (string): Google Drive synced folder to watch for jobs.
- used_dir (string): Folder to archive processed inputs.
- output_dir (string): Folder to store downloaded artifacts.
- poll_interval (double): Seconds between directory scans (default: 5.0).
"""

import shutil
import threading
import time
from pathlib import Path

import argparse

import rclpy
from rclpy.node import Node

from sam_http_client import _find_jobs, download_results, poll_status, send_job
from config_utils import load_config


class SamRosClient(Node):
    def __init__(self, config=None):
        super().__init__("sam_drive_watcher")

        config = config or {}

        self.server_url = self.declare_parameter("server_url", config.get("server_url", "http://localhost:8000")).value

        watch_dir_value = self.declare_parameter("watch_dir", config.get("watch_dir", "")).value
        used_dir_value = self.declare_parameter("used_dir", config.get("used_dir", "")).value
        output_dir_value = self.declare_parameter("output_dir", config.get("output_dir", "")).value

        self.watch_dir = Path(watch_dir_value).expanduser() if watch_dir_value else None
        self.used_dir = Path(used_dir_value).expanduser() if used_dir_value else None
        self.output_dir = Path(output_dir_value).expanduser() if output_dir_value else None
        self.poll_interval = float(self.declare_parameter("poll_interval", config.get("poll_interval", 5.0)).value)

        if self.watch_dir:
            if not self.used_dir:
                self.used_dir = self.watch_dir / "used"
            if not self.output_dir:
                self.output_dir = self.watch_dir / "results"
        else:
            if not self.used_dir:
                self.used_dir = Path() / "used"
            if not self.output_dir:
                self.output_dir = Path() / "results"

        self.used_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._processing = False
        self.timer = self.create_timer(self.poll_interval, self._tick)
        self.get_logger().info(f"Watching {self.watch_dir} for SAM jobs")

    def _tick(self):
        if self._processing:
            return
        if not self.watch_dir:
            self.get_logger().warning("watch_dir parameter not set; idle")
            return
        if not self.watch_dir.exists():
            self.get_logger().warning(f"Watch dir {self.watch_dir} missing")
            return

        jobs = list(_find_jobs(self.watch_dir))
        if not jobs:
            return

        image_path, prompt_path = jobs[0]
        self._processing = True
        threading.Thread(target=self._process_job, args=(image_path, prompt_path), daemon=True).start()

    def _process_job(self, image_path, prompt_path):
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            self.get_logger().info(f"Submitting {image_path.name}")
            job_id = send_job(self.server_url, image_path, prompt_text)
            status = poll_status(self.server_url, job_id, interval=self.poll_interval)
            if status.get("state") == "finished":
                download_results(self.server_url, job_id, self.output_dir)
            timestamp = int(time.time())
            shutil.move(image_path, self.used_dir / f"{image_path.stem}_{timestamp}{image_path.suffix}")
            shutil.move(prompt_path, self.used_dir / f"{prompt_path.stem}_{timestamp}{prompt_path.suffix}")
        finally:
            self._processing = False


def main(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to config.json")
    parsed_args, remaining = parser.parse_known_args(args=args)

    cfg = load_config(parsed_args.config).get("ros2", {})

    rclpy.init(args=remaining)
    node = SamRosClient(config=cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

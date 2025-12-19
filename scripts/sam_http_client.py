"""
Simple HTTP client for the SAM server.

Features:
- Submit a single job from a provided image and prompt text.
- Poll server progress and download artifacts when finished.
- Optional watch mode for a Google Drive synced folder on a laptop.
"""

import argparse
import shutil
import time
from pathlib import Path

import requests

from config_utils import load_config


def send_job(server_url, image_path, prompt_text):
    image_path = Path(image_path).expanduser()
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {"prompt": prompt_text}
        response = requests.post(f"{server_url}/job", files=files, data=data, timeout=30)
    if response.status_code not in (200, 202):
        raise RuntimeError(f"Server error: {response.status_code} {response.text}")
    return response.json()["job_id"]


def poll_status(server_url, job_id, interval=2):
    while True:
        resp = requests.get(f"{server_url}/status/{job_id}", timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch status: {resp.text}")
        data = resp.json()
        state = data.get("state")
        message = data.get("message", "")
        print(f"[STATUS] {state}: {message}")
        if state in {"finished", "failed"}:
            return data
        time.sleep(interval)


def download_results(server_url, job_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(f"{server_url}/result/{job_id}", timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download results: {resp.text}")

    zip_path = output_dir / f"{job_id}.zip"
    with open(zip_path, "wb") as f:
        f.write(resp.content)

    shutil.unpack_archive(zip_path, output_dir / job_id)
    print(f"Artifacts saved to {output_dir / job_id}")
    return zip_path


def _find_jobs(folder):
    folder = Path(folder).expanduser()
    for image_path in folder.glob("*.png"):
        prompt_path = image_path.with_suffix(".txt")
        if prompt_path.exists():
            yield image_path, prompt_path


def watch_drive_folder(server_url, watch_dir, used_dir=None, poll_interval=5, output_dir=None):
    watch_dir = Path(watch_dir).expanduser()
    used_dir = Path(used_dir).expanduser() if used_dir else watch_dir / "used"
    output_dir = Path(output_dir).expanduser() if output_dir else watch_dir / "results"
    used_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Watching {watch_dir} for new images + prompt.txt ...")
    while True:
        jobs = list(_find_jobs(watch_dir))
        if not jobs:
            time.sleep(poll_interval)
            continue

        for image_path, prompt_path in jobs:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            print(f"Submitting {image_path.name} with prompt from {prompt_path.name}")
            job_id = send_job(server_url, image_path, prompt_text)
            status = poll_status(server_url, job_id, interval=poll_interval)
            if status.get("state") == "finished":
                download_results(server_url, job_id, output_dir)
            timestamp = int(time.time())
            shutil.move(image_path, used_dir / f"{image_path.stem}_{timestamp}{image_path.suffix}")
            shutil.move(prompt_path, used_dir / f"{prompt_path.stem}_{timestamp}{prompt_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="SAM server client")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--server", help="SAM server URL")
    parser.add_argument("--image", help="Image to send")
    parser.add_argument("--prompt", help="Prompt text or path to file")
    parser.add_argument("--watch", help="Google Drive folder to watch for jobs")
    parser.add_argument("--used", help="Directory to move processed inputs")
    parser.add_argument("--output", help="Directory to store results")
    parser.add_argument("--poll-interval", type=float, help="Seconds between polling")
    args = parser.parse_args()

    cfg = load_config(args.config).get("http_client", {})
    server_url = args.server or cfg.get("server_url", "http://localhost:8000")
    watch_dir = args.watch or cfg.get("watch_dir")
    used_dir = args.used or cfg.get("used_dir")
    output_dir = args.output or cfg.get("output_dir")
    poll_interval = args.poll_interval or float(cfg.get("poll_interval", 5.0))

    if watch_dir:
        watch_drive_folder(server_url, watch_dir, used_dir=used_dir, poll_interval=poll_interval, output_dir=output_dir)
        return

    if not args.image:
        raise SystemExit("Provide --image or --watch")

    prompt_text = ""
    if args.prompt:
        prompt_path = Path(args.prompt).expanduser()
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else args.prompt

    job_id = send_job(server_url, args.image, prompt_text)
    status = poll_status(server_url, job_id, interval=poll_interval)
    if status.get("state") == "finished":
        download_results(server_url, job_id, output_dir or "client_results")


if __name__ == "__main__":
    main()

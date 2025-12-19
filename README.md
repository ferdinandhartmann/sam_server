# SAM Server

This repository provides a lightweight HTTP server that orchestrates the existing SAM3 → SAM-3D → mesh workers, plus clients for sending jobs from a laptop (standalone Python and ROS2).

## Server

Start the orchestrator (it will launch the worker supervisor script automatically):

```bash
python scripts/sam_server.py --config config.json
```

Configuration lives in `config.json` (see `config.example.json` for defaults) and supports overrides for the server bind host/port and worker data directory. CLI flags take precedence over config values.

`config.json` sections:

- `server`: `host`, `port`, `base_dir` for the worker data hierarchy.
- `http_client`: defaults for `server_url`, optional `watch_dir`, `used_dir`, `output_dir`, and `poll_interval` when monitoring a Google Drive folder.
- `ros2`: the same client defaults applied to the ROS2 node (can still be overridden with `--ros-args`).

API endpoints:

- `POST /job` — multipart form with fields `image` (PNG) and `prompt` (text). Returns a `job_id`.
- `GET /status/<job_id>` — returns JSON progress (`received`, `sam3`, `sam3d`, `meshing`, `finished`).
- `GET /result/<job_id>` — downloads a ZIP containing the generated `job.ply`, `object_visual.obj`, and `object_collision.obj`.

Artifacts for each job are stored under `server_jobs/<job_id>`.

## Standalone Python client

Submit a single image + prompt:

```bash
python scripts/sam_http_client.py --config config.json --image /path/to/job.png --prompt "a coffee mug"
```

Watch a Google Drive-synced folder for new `*.png` + matching `*.txt` prompt files (processed inputs are moved to a `used/` subfolder):

```bash
python scripts/sam_http_client.py --config config.json --watch "~/Google Drive/SAMJobs"
```

Results are unpacked into `results/` inside the watched folder (or a custom `--output` directory).

## ROS2 drive-watcher node

The ROS2 node mirrors the standalone client behaviour using ROS parameters:

```bash
ros2 run <your_package> sam_ros2_client.py --config /path/to/config.json --ros-args \
  -p server_url:=http://<server>:8000 \
  -p watch_dir:=/path/to/GoogleDrive/SAMJobs \
  -p poll_interval:=5.0
```

Processed inputs are archived under `used/` and artifacts are stored under `results/` inside the watch directory by default.

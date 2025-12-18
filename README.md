# sam_server

A FastAPI service that orchestrates segmentation, 3D reconstruction, and mesh
conversion workers. Each worker loads its model once at startup, waits for jobs,
and processes them in order: segmentation → 3D → conversion.

## Requirements

- Python environments for each worker (configure paths via environment
  variables):
  - `SAM_SEGMENTATION_PYTHON` – Python executable for the SAM3 segmentation
    worker.
  - `SAM_3D_PYTHON` – Python executable for the SAM-3D reconstruction worker.
  - `SAM_CONVERTER_PYTHON` – Python executable for the Open3D conversion worker.
- A valid SAM-3D config file (defaults to `checkpoints/hf/pipeline.yaml`, override with `SAM3D_CONFIG_PATH`).
- CUDA-ready GPU if you want hardware acceleration.
- Python dependencies installed inside the respective environments (`fastapi`,
  `uvicorn`, `sam3`, `sam-3d-objects`, `open3d`, etc.).

## Directory layout

- `scripts/` – API server, workers, and utilities.
- `jobs/` – Created automatically; contains per-job folders with `state.json`
  and artifacts.

## Running the server and workers (single command)

From the repository root:

```bash
export SAM_SEGMENTATION_PYTHON=/home/ferdinand/miniforge3/envs/sam3/bin/python
export SAM_3D_PYTHON=/home/ferdinand/miniforge3/envs/sam3d-objects/bin/python
export SAM_CONVERTER_PYTHON=/home/ferdinand/miniforge3/envs/sam_server/bin/python
export SAM3D_CONFIG_PATH=/path/to/pipeline.yaml  # optional if using the default layout

python scripts/sam_server.py
```

This command:
1. Starts all three workers in their respective environments.
2. Starts the FastAPI server on `0.0.0.0:8000`.
3. Loads all models once; each worker then polls the shared `jobs/` directory
   for work.

### Running workers without the API (optional)

If you need the workers without the HTTP server (e.g., during debugging):

```bash
python scripts/start_all_workers.py
```

Press `CTRL+C` to stop them.

## Submitting a job from the same machine

```bash
python scripts/sam_client.py path/to/image.png --host 127.0.0.1 --port 8000
```

The response contains the job ID and paths to the OBJ/STL/GIF artifacts inside
`jobs/<job_id>/`.

## Submitting a job from another machine

Ensure the server machine is reachable and open the desired port. Then point the
client to the server IP:

```bash
python scripts/sam_client.py path/to/image.png --host <SERVER_IP> --port 8000
```

## Job lifecycle

1. `POST /reconstruct` saves the uploaded image to a new job directory and
   returns once conversion finishes (or fails).
2. The segmentation worker writes masks into `jobs/<job_id>/segmentation/` and
   updates `state.json`.
3. The 3D worker consumes the mask, produces `object.ply`, and saves a GIF
   preview.
4. The converter worker turns the PLY into OBJ/STL files (visual + collision
   variants).
5. You can poll `GET /jobs/{job_id}` for status or artifacts at any point.

## Configuration

Environment variables (optional):

- `SAM_SERVER_HOST` / `SAM_SERVER_PORT` – host/port for the FastAPI server.
- `SAM_SERVER_JOBS` – override the `jobs/` directory location.
- `SAM_WORKER_POLL_SECONDS` – polling interval for workers.
- `SAM_WORKER_LOG_PREFIX` – prefix for worker log lines.

## Notes

- Workers assume GPU resources are available. Set `CUDA_VISIBLE_DEVICES` in the
  environment if you need to pin devices.
- All artifacts are written inside the job directory. The API response returns
  the relevant paths.
- The pipeline is intentionally file-based to keep the environments isolated per
  worker.

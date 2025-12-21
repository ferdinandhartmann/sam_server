# SAM Server

The SAM Server is a FastAPI-based application designed to manage and process 3D object-related tasks. It supports worker processes for handling jobs and provides endpoints for job submission, status checking, and downloading results.

## Getting Started

### Manual Worker Start
1. Navigate to the project directory:
    ```bash
    cd ~/sam_project/sam_server
    ```
2. Activate the conda environment:
    ```bash
    conda activate sam_server
    ```
3. Start all workers manually:
    ```bash
    python3 scripts/start_all_workers.py
    ```
4. Place the input image and a prompt .txt file with the same base name as the image (containing the prompt) into:
    ```
    worker_data/sam_3d_worker/input/
    ```

### Running the Server
1. Navigate to the project directory:
    ```bash
    cd ~/sam_project/sam_server
    ```
2. Activate the conda environment:
    ```bash
    conda activate sam_server
    ```
3. Start the server using the provided script:
   ```bash
   ./scripts/start_sam_server.sh
   ```
4. Access the server at `http://0.0.0.0:8000`.

## API Endpoints

### `/ready`
- **Description**: Check if all workers are ready.
- **Method**: `GET`

### `/submit`
- **Description**: Submit a job with an image and prompt.
- **Method**: `POST`

### `/status/{job_id}`
- **Description**: Check the status of a submitted job.
- **Method**: `GET`

### `/download/{job_id}/{filename}`
- **Description**: Download the result of a completed job.
- **Method**: `GET`

### `/health`
- **Description**: Perform a health check on the server.
- **Method**: `GET`

## Directory Structure
- `scripts/`: Contains server and worker scripts.
- `worker_data/`: Stores input, output, and intermediate files for workers.
- `README.md`: Documentation for the SAM Server.


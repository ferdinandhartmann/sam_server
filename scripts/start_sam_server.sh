export TORCHINDUCTOR_CACHE_DIR=/home/ferdinand/.cache/torchinductor
# export TORCHINDUCTOR_CACHE_DIR=/home/ferdinand/.cache/torch/inductor
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export CUDA_VISIBLE_DEVICES=0 

uvicorn scripts.sam_server:app \
    --host 0.0.0.0 \
    --port 8000 \
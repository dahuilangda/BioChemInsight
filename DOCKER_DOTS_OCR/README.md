
# DOCKER_DOTS_OCR Service Setup Guide

Self-hosted OCR container based on DotsOCR. Follow the steps below to download weights, configure Docker, and run the service.

## 1. Prerequisites

1. **NVIDIA Docker** (GPU acceleration required)
2. **Python 3.8+** (used for downloading model weights)
3. **Python packages**
	```bash
	pip install modelscope huggingface_hub
	```

## 2. Download Model Weights

Run the helper script in this directory:
```bash
python download_model.py
```
Weights will be stored under `weights/DotsOCR/`.

## 3. Configure Docker

### Key Files
- `docker-compose.yml`: Orchestrates the service with GPU access, port mapping, and weight mounts.
- `Dockerfile`: Reference if you need to customize the base image.

### docker-compose.yml Settings to Review
- `shm_size: 2g`: Required for stable NCCL communication; keep it at 2 GB or higher.
- `NVIDIA_VISIBLE_DEVICES=0,1`: Lists the GPU indices the container may use.
- `volumes`: Ensure the mounted weight directory matches your local path.

## 4. Run and Manage the Service

### Pull the Image
```bash
docker compose pull
```

### Start the Service
```bash
docker compose up -d
```

### Tail Logs
```bash
docker compose logs -f
```

### Stop the Service
```bash
docker compose down
```
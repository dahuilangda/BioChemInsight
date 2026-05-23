# DOCKER_PADDLE_OCR

Standalone PaddleOCR queue stack with Redis, FastAPI API, and Celery worker.

## Start

```bash
cd DOCKER_PADDLE_OCR
docker compose up -d --build
```

Default network:
- `172.201.0.0/24`

Services:
- `paddle-ocr-redis`
- `paddle-ocr-api`
- `paddle-ocr-worker`

Health:

```bash
curl http://localhost:8010/healthz
```

## API

Submit OCR:

```bash
curl -X POST http://localhost:8010/v1/pdf-to-markdown \
  -F "file=@data/demo.pdf;type=application/pdf" \
  -F "page_start=1" \
  -F "page_end=3"
```

The submit response contains `job_id`. Poll a job:

```bash
curl http://localhost:8010/v1/jobs/<job_id>
```

Fetch a finished markdown result:

```bash
curl http://localhost:8010/v1/jobs/<job_id>/result
```

Download a finished Word job:

```bash
curl -L http://localhost:8010/v1/jobs/<job_id>/download -o result.docx
```

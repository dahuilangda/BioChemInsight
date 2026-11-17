# DOCKER_PADDLE_OCR

Standalone PaddleOCR-VL-1.6 queue stack with Redis, FastAPI API, and Celery worker.

The recognition pipeline is `PaddleOCRVL(pipeline_version="v1.6")`. The
VL decoder's `generation_config.json` in the local model cache is patched
at pipeline init with `no_repeat_ngram_size=20` (env
`PADDLEOCR_NO_REPEAT_NGRAM_SIZE`): dense chemical tables drive the
decoder into SMILES repetition loops (arginine/aspartate motifs), and
the model's `generate()` forwards only `max_new_tokens`/`use_cache`, so
per-call values never reach it. Note: on dense sheets the page-level
pipeline reads the first table region completely — full-sheet coverage
comes from the harness's block-crop reader (`utils/vl_block_reader.py`).

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

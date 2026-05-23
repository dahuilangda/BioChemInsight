from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import redis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .ocr_core import SUPPORTED_IMAGE_SUFFIXES

LOGGER = logging.getLogger("paddle_ocr_server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

PADDLEOCR_REDIS_URL = (os.getenv("PADDLEOCR_REDIS_URL", "redis://redis:6379/0") or "redis://redis:6379/0").strip()
PADDLEOCR_TASK_ROOT = Path(os.getenv("PADDLEOCR_TASK_ROOT", "/workspace/output") or "/workspace/output")
PADDLEOCR_JOB_TTL_SECONDS = int(os.getenv("PADDLEOCR_JOB_TTL_SECONDS", "86400") or "86400")

app = FastAPI(
    title="PaddleOCR Queue Service",
    description="Queue-backed PaddleOCR service exposed through FastAPI and Celery.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.Redis.from_url(PADDLEOCR_REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"paddleocr:job:{job_id}"


def _result_path(job_id: str) -> Path:
    return PADDLEOCR_TASK_ROOT / job_id / "result.json"


def _normalize_lang(lang: str | None) -> str:
    normalized = (lang or "auto").strip().lower()
    return normalized or "auto"


def _hash_contents(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _build_job_id(
    document_key: str,
    page_start: int,
    page_end: int,
    lang: str,
    return_raw: bool,
    endpoint: str,
    page_number_offset: int = 0,
) -> str:
    payload = {
        "document_key": document_key,
        "page_start": int(page_start),
        "page_end": int(page_end),
        "lang": lang,
        "return_raw": bool(return_raw),
        "endpoint": endpoint,
        "page_number_offset": int(page_number_offset),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _store_job(payload: Dict[str, Any]) -> None:
    redis_client.setex(_job_key(payload["job_id"]), PADDLEOCR_JOB_TTL_SECONDS, json.dumps(payload, ensure_ascii=False))


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    raw = redis_client.get(_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw)


def _load_result(job_id: str) -> Optional[Dict[str, Any]]:
    path = _result_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _enqueue_job(
    endpoint: str,
    file: UploadFile,
    contents: bytes,
    page_start: int,
    page_end: int,
    lang: str,
    return_raw: bool,
    document_key: str = "",
    page_number_offset: int = 0,
    source_suffix: str = ".pdf",
) -> Dict[str, Any]:
    document_key = document_key.strip() or _hash_contents(contents)
    job_id = _build_job_id(document_key, page_start, page_end, lang, return_raw, endpoint, page_number_offset=page_number_offset)
    existing = _load_job(job_id)
    if existing and existing.get("status") in {"pending", "running", "completed"}:
        return existing

    task_dir = PADDLEOCR_TASK_ROOT / job_id
    task_dir.mkdir(parents=True, exist_ok=True)
    source_suffix = source_suffix if source_suffix.startswith(".") else f".{source_suffix}"
    source_path = task_dir / f"input{source_suffix.lower()}"
    source_path.write_bytes(contents)

    payload = {
        "job_id": job_id,
        "status": "pending",
        "endpoint": endpoint,
        "source_path": str(source_path),
        "source_suffix": source_suffix.lower(),
        "document_key": document_key,
        "page_start": page_start,
        "page_end": page_end,
        "lang": lang,
        "return_raw": return_raw,
        "page_number_offset": page_number_offset,
        "filename": file.filename or "document.pdf",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _store_job(payload)

    from .celery_app import celery_app

    celery_app.send_task("paddleocr.run_job", args=[job_id], queue="paddleocr")
    return payload


@app.get("/healthz")
async def healthz() -> dict:
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception as exc:
        redis_status = f"error: {exc}"
    return {
        "status": "ok",
        "redis": redis_status,
        "task_root": str(PADDLEOCR_TASK_ROOT),
    }


@app.post("/v1/pdf-to-markdown")
async def pdf_to_markdown_endpoint(
    file: UploadFile = File(...),
    page_start: int = Form(1),
    page_end: int = Form(-1),
    lang: str = Form("auto"),
    return_raw: bool = Form(False),
    document_key: str = Form(""),
    page_number_offset: int = Form(0),
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job = _enqueue_job(
        endpoint="pdf-to-markdown",
        file=file,
        contents=contents,
        page_start=page_start,
        page_end=page_end,
        lang=_normalize_lang(lang),
        return_raw=return_raw,
        document_key=document_key,
        page_number_offset=page_number_offset,
        source_suffix=".pdf",
    )
    return job


@app.post("/v1/image-to-markdown")
async def image_to_markdown_endpoint(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
    return_raw: bool = Form(False),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    job = _enqueue_job(
        endpoint="image-to-markdown",
        file=file,
        contents=contents,
        page_start=1,
        page_end=1,
        lang=_normalize_lang(lang),
        return_raw=return_raw,
        source_suffix=suffix,
    )
    return job


@app.post("/v1/pdf-to-word")
async def pdf_to_word_endpoint(
    file: UploadFile = File(...),
    page_start: int = Form(1),
    page_end: int = Form(-1),
    lang: str = Form("auto"),
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    job = _enqueue_job(
        endpoint="pdf-to-word",
        file=file,
        contents=contents,
        page_start=page_start,
        page_end=page_end,
        lang=_normalize_lang(lang),
        return_raw=False,
        source_suffix=".pdf",
    )
    return job


@app.post("/v1/image-to-word")
async def image_to_word_endpoint(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    job = _enqueue_job(
        endpoint="image-to-word",
        file=file,
        contents=contents,
        page_start=1,
        page_end=1,
        lang=_normalize_lang(lang),
        return_raw=False,
        source_suffix=suffix,
    )
    return job


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    result = _load_result(job_id)
    return result if result is not None else job


@app.get("/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    result = _load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return result


@app.get("/v1/jobs/{job_id}/download")
async def download_job_result(job_id: str) -> FileResponse:
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    result = _load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    download_path = result.get("download_path")
    if not download_path:
        raise HTTPException(status_code=400, detail="Job has no downloadable artifact.")
    path = Path(download_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Download artifact missing.")
    media_type = result.get("content_type") or ("application/zip" if path.suffix.lower() == ".zip" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=result.get("download_name") or path.name,
    )

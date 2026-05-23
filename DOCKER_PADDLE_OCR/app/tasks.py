from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import redis

from .ocr_core import PaddleOCRCore

LOGGER = logging.getLogger("paddle_ocr_tasks")

PADDLEOCR_REDIS_URL = (os.getenv("PADDLEOCR_REDIS_URL", "redis://redis:6379/0") or "redis://redis:6379/0").strip()
PADDLEOCR_TASK_ROOT = Path(os.getenv("PADDLEOCR_TASK_ROOT", "/workspace/output") or "/workspace/output")
PADDLEOCR_TASK_RESULT_TTL_SECONDS = int(os.getenv("PADDLEOCR_TASK_RESULT_TTL_SECONDS", "86400") or "86400")
PADDLEOCR_TASK_LOCK_TTL_SECONDS = int(os.getenv("PADDLEOCR_TASK_LOCK_TTL_SECONDS", "86400") or "86400")

redis_client = redis.Redis.from_url(PADDLEOCR_REDIS_URL, decode_responses=True)
ocr_core = PaddleOCRCore()


def _job_key(job_id: str) -> str:
    return f"paddleocr:job:{job_id}"


def _lock_key(job_id: str) -> str:
    return f"paddleocr:lock:{job_id}"


def _result_path(job_id: str) -> Path:
    return PADDLEOCR_TASK_ROOT / job_id / "result.json"


def _load_job(job_id: str) -> Dict[str, Any]:
    raw = redis_client.get(_job_key(job_id))
    if not raw:
        raise KeyError(job_id)
    return json.loads(raw)


def _store_job(job_id: str, payload: Dict[str, Any]) -> None:
    key = _job_key(job_id)
    ttl = redis_client.ttl(key)
    redis_client.set(key, json.dumps(payload, ensure_ascii=False))
    if ttl and ttl > 0:
        redis_client.expire(key, ttl)


def _update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    job = _load_job(job_id)
    job.update(fields)
    job["updated_at"] = time.time()
    _store_job(job_id, job)
    return job


def _release_lock(job_id: str) -> None:
    redis_client.delete(_lock_key(job_id))


def _write_result(job_id: str, payload: Dict[str, Any]) -> Path:
    job_dir = PADDLEOCR_TASK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(job_id)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    redis_client.expire(_job_key(job_id), PADDLEOCR_TASK_RESULT_TTL_SECONDS)
    return result_path


def _build_markdown_response(job: Dict[str, Any], page_markdowns, raw_predictions, processed_pages) -> Dict[str, Any]:
    offset = int(job.get("page_number_offset") or 0)
    source_pages = [offset + int(page_number) for page_number in processed_pages]
    response: Dict[str, Any] = {
        "job_id": job["job_id"],
        "status": "completed",
        "endpoint": job["endpoint"],
        "page_markdowns": page_markdowns,
        "pages": [
            {"page_number": page_number, "markdown": page_markdown}
            for page_number, page_markdown in zip(source_pages, page_markdowns)
        ],
        "page_numbers": source_pages,
        "page_count": len(processed_pages),
    }
    if job.get("return_raw"):
        response["raw_predictions"] = raw_predictions
    return response


def _build_file_response(job: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    artifact_dir = PADDLEOCR_TASK_ROOT / job["job_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"artifact{output_path.suffix.lower()}"
    if output_path.resolve() != artifact_path.resolve():
        shutil.copy2(output_path, artifact_path)
    return {
        "job_id": job["job_id"],
        "status": "completed",
        "endpoint": job["endpoint"],
        "download_name": f"{Path(job.get('filename') or 'document').stem}_paddleocr{artifact_path.suffix}",
        "download_path": str(artifact_path),
        "content_type": "application/zip" if artifact_path.suffix.lower() == ".zip" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }


def run_job(job_id: str) -> Dict[str, Any]:
    job = _load_job(job_id)
    if job.get("status") in {"completed", "failed", "canceled"}:
        return job
    if not redis_client.set(_lock_key(job_id), "1", nx=True, ex=PADDLEOCR_TASK_LOCK_TTL_SECONDS):
        return job

    _update_job(job_id, status="running", started_at=time.time())
    source_path = Path(job["source_path"])
    try:
        endpoint = job["endpoint"]
        if endpoint == "pdf-to-markdown":
            page_markdowns, raw_predictions, processed_pages = ocr_core.process_pdf(
                pdf_path=source_path,
                page_start=int(job["page_start"]),
                page_end=int(job["page_end"]),
                lang=job.get("lang"),
                return_raw=bool(job.get("return_raw")),
            )
            result = _build_markdown_response(job, page_markdowns, raw_predictions, processed_pages)
        elif endpoint == "image-to-markdown":
            markdown, raw_predictions = ocr_core.process_image(
                contents=source_path.read_bytes(),
                suffix=job.get("source_suffix") or source_path.suffix or ".png",
                lang=job.get("lang"),
                return_raw=bool(job.get("return_raw")),
            )
            result = {
                "job_id": job_id,
                "status": "completed",
                "endpoint": endpoint,
                "page_markdowns": [markdown],
                "pages": [{"page_number": 1, "markdown": markdown}],
                "page_numbers": [1],
                "page_count": 1,
            }
            if job.get("return_raw"):
                result["raw_predictions"] = raw_predictions
        elif endpoint == "pdf-to-word":
            output_path = ocr_core.process_pdf_to_word(
                contents=source_path.read_bytes(),
                page_start=int(job["page_start"]),
                page_end=int(job["page_end"]),
                lang=job.get("lang"),
            )
            result = _build_file_response(job, output_path)
        elif endpoint == "image-to-word":
            output_path = ocr_core.process_image_to_word(
                contents=source_path.read_bytes(),
                suffix=job.get("source_suffix") or source_path.suffix or ".png",
                lang=job.get("lang"),
            )
            result = _build_file_response(job, output_path)
        else:
            raise RuntimeError(f"Unknown endpoint: {endpoint}")

        result_path = _write_result(job_id, result)
        _update_job(job_id, status="completed", result_path=str(result_path), finished_at=time.time())
        return result
    except Exception as exc:
        LOGGER.exception("PaddleOCR job %s failed", job_id)
        _update_job(job_id, status="failed", error=str(exc), finished_at=time.time())
        raise
    finally:
        _release_lock(job_id)
        try:
            if source_path.exists():
                source_path.unlink(missing_ok=True)
        except Exception:
            pass

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import requests


def _get_job_status(server_url: str, job_id: str) -> Dict[str, Any]:
    response = requests.get(f"{server_url.rstrip('/')}/v1/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"PaddleOCR job {job_id} returned an invalid status payload.")
    return payload


def _get_job_result(server_url: str, job_id: str) -> Dict[str, Any]:
    response = requests.get(f"{server_url.rstrip('/')}/v1/jobs/{job_id}/result", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"PaddleOCR job {job_id} returned an invalid result payload.")
    return payload


def _wait_for_job_result(server_url: str, job_id: str, timeout_seconds: int, poll_interval_seconds: float = 2.0) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    while True:
        status_payload = _get_job_status(server_url, job_id)
        status = str(status_payload.get("status") or "").lower()
        if status == "completed":
            return _get_job_result(server_url, job_id)
        if status == "failed":
            raise RuntimeError(str(status_payload.get("error") or f"PaddleOCR job {job_id} failed."))
        if time.monotonic() >= deadline:
            raise requests.Timeout(f"PaddleOCR job {job_id} did not complete within {int(timeout_seconds)}s.")
        time.sleep(poll_interval_seconds)


def request_pdf_to_markdown(
    pdf_file: str | os.PathLike[str],
    page_start: int,
    page_end: int,
    lang: str,
    return_raw: bool,
    server_url: str,
    *,
    document_key: str,
    page_number_offset: int = 0,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    endpoint = f"{server_url.rstrip('/')}/v1/pdf-to-markdown"
    with open(pdf_file, "rb") as pdf_stream:
        response = requests.post(
            endpoint,
            files={"file": (Path(pdf_file).name or "document.pdf", pdf_stream, "application/pdf")},
            data={
                "page_start": str(page_start),
                "page_end": str(page_end),
                "lang": lang,
                "return_raw": "true" if return_raw else "false",
                "document_key": document_key,
                "page_number_offset": str(page_number_offset),
            },
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR queue service returned an invalid submit payload.")
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError("PaddleOCR queue service did not return a job_id.")
    return _wait_for_job_result(server_url, job_id, timeout_seconds)

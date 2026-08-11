from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import requests


def check_server_health(server_url: str, timeout: int = 10) -> bool:
    """Return *True* if the OCR service is reachable and Redis is connected."""
    try:
        response = requests.get(
            f"{server_url.rstrip('/')}/healthz",
            timeout=timeout,
        )
        if response.status_code != 200:
            return False
        payload = response.json()
        return str(payload.get("redis", "")).startswith("ok")
    except requests.RequestException:
        return False


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


# Transient network errors that should NOT abort polling — the next poll
# cycle may succeed once the network recovers.
_TRANSIENT_NETWORK_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
)


def _wait_for_job_result(
    server_url: str,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Poll job status until completion, failure, or deadline.

    Transient network errors (``ConnectionError``, ``Timeout``) during
    individual poll requests are tolerated — they do not abort the wait.
    The overall *timeout_seconds* deadline is still enforced, so a sustained
    outage eventually surfaces as ``requests.Timeout``.
    """
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    while True:
        try:
            status_payload = _get_job_status(server_url, job_id)
        except _TRANSIENT_NETWORK_ERRORS:
            # Network hiccup during status poll — check deadline, then retry.
            if time.monotonic() >= deadline:
                raise requests.Timeout(
                    f"PaddleOCR job {job_id} did not complete within {int(timeout_seconds)}s."
                )
            time.sleep(poll_interval_seconds)
            continue

        status = str(status_payload.get("status") or "").lower()
        if status == "completed":
            # Fetching the result may also hit a transient error; allow one
            # fall-through to the deadline check below.
            try:
                return _get_job_result(server_url, job_id)
            except _TRANSIENT_NETWORK_ERRORS:
                if time.monotonic() >= deadline:
                    raise requests.Timeout(
                        f"PaddleOCR job {job_id} did not complete within {int(timeout_seconds)}s."
                    )
                time.sleep(poll_interval_seconds)
                continue
        if status == "failed":
            raise RuntimeError(
                str(status_payload.get("error") or f"PaddleOCR job {job_id} failed.")
            )
        if time.monotonic() >= deadline:
            raise requests.Timeout(
                f"PaddleOCR job {job_id} did not complete within {int(timeout_seconds)}s."
            )
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
    submit_timeout: int = 120,
    max_submit_retries: int = 2,
    retry_backoff_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Submit a PDF OCR job and block until the result is ready.

    Parameters
    ----------
    timeout_seconds
        Overall deadline for the job to reach *completed* status.
    submit_timeout
        Per-request HTTP timeout for the initial ``POST`` that uploads the
        PDF and enqueues the job.
    max_submit_retries
        Number of times to retry the **submit** step on transient network
        errors (``ConnectionError`` / ``Timeout``).  The job-wait phase has
        its own internal resilience (see :func:`_wait_for_job_result`).
    retry_backoff_seconds
        Base delay between submit retries; actual delay is
        ``retry_backoff_seconds * 2 ** attempt`` (with default ``max_submit_retries=2``:
        5 s at attempt 0, 10 s at attempt 1; the final attempt raises on failure).
    """
    endpoint = f"{server_url.rstrip('/')}/v1/pdf-to-markdown"
    last_exc: Exception | None = None

    # --- Phase 1: submit the job (retry on transient network errors) --- #
    # The try is scoped to ONLY the POST + job_id extraction.  The wait phase
    # is deliberately outside this loop so that a wait-deadline Timeout
    # propagates directly to the caller instead of being retried as a submit
    # failure (which would blow the timeout budget ~3× and delay split-retry).
    job_id: str | None = None
    for attempt in range(max(max_submit_retries, 0) + 1):
        try:
            with open(pdf_file, "rb") as pdf_stream:
                response = requests.post(
                    endpoint,
                    files={
                        "file": (
                            Path(pdf_file).name or "document.pdf",
                            pdf_stream,
                            "application/pdf",
                        )
                    },
                    data={
                        "page_start": str(page_start),
                        "page_end": str(page_end),
                        "lang": lang,
                        "return_raw": "true" if return_raw else "false",
                        "document_key": document_key,
                        "page_number_offset": str(page_number_offset),
                    },
                    timeout=submit_timeout,
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "PaddleOCR queue service returned an invalid submit payload."
                )
            job_id = str(payload.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("PaddleOCR queue service did not return a job_id.")
            break  # submit succeeded — exit retry loop
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < max(max_submit_retries, 0):
                wait = retry_backoff_seconds * (2 ** attempt)
                print(
                    f"PaddleOCR submit attempt {attempt + 1}/{max_submit_retries + 1} "
                    f"failed ({type(exc).__name__}: {exc}); retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
                continue
            raise

    if job_id is None:
        raise last_exc if last_exc is not None else RuntimeError(
            "PaddleOCR submit failed without recording an exception."
        )

    # --- Phase 2: wait for the job to complete (no submit-retry here) --- #
    return _wait_for_job_result(server_url, job_id, timeout_seconds)

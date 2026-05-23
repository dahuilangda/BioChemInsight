from __future__ import annotations

import os

from celery import Celery

PADDLEOCR_REDIS_URL = (os.getenv("PADDLEOCR_REDIS_URL", "redis://redis:6379/0") or "redis://redis:6379/0").strip()

celery_app = Celery("paddle_ocr_queue", broker=PADDLEOCR_REDIS_URL, backend=PADDLEOCR_REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@celery_app.task(name="paddleocr.run_job")
def run_job(job_id: str) -> None:
    from .tasks import run_job as run_ocr_job

    run_ocr_job(job_id)

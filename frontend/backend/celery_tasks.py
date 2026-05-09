from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .celery_app import celery_app
from .work_queue import clear_inflight, get_job


@celery_app.task(name="frontend.backend.celery_tasks.run_queued_task")
def run_queued_task(task_id: str) -> None:
    job = get_job(task_id)
    if not job:
        clear_inflight(task_id)
        return

    try:
        from . import main

        task_record = main.task_manager.get(task_id)
        if task_record is not None and task_record.status == "canceled":
            return

        task_name = job.get("task_name")
        args = job.get("args") or []
        kwargs = job.get("kwargs") or {}

        if task_name == "auto_detect_plan":
            asyncio.run(main.launch_auto_detect_task(task_id, *args, **kwargs))
        elif task_name == "structure_extraction":
            asyncio.run(main.launch_structure_task(task_id, *args, **kwargs))
        elif task_name == "bioactivity_extraction":
            asyncio.run(main.launch_assay_task(task_id, *args, **kwargs))
        elif task_name == "data_merge":
            asyncio.run(main.launch_merge_task(task_id, *args, **kwargs))
        elif task_name == "full_pipeline":
            asyncio.run(main.launch_full_pipeline_task(task_id, *args, **kwargs))
        else:
            main.task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Unknown queued task type",
                error=f"Unsupported task_name: {task_name}",
            )
    finally:
        clear_inflight(task_id)

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_job(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be a JSON object")
    return payload


async def _run_job(job: dict) -> None:
    from . import main

    task_id = str(job.get("task_id") or "")
    task_name = str(job.get("task_name") or "")
    args = job.get("args") or []
    kwargs = job.get("kwargs") or {}
    if not task_id:
        raise ValueError("Missing task_id")
    if not isinstance(args, list):
        raise ValueError("Job args must be a list")
    if not isinstance(kwargs, dict):
        raise ValueError("Job kwargs must be an object")

    task_record = main.task_manager.get(task_id)
    if task_record is not None and task_record.status in {"completed", "failed", "canceled"}:
        return

    if task_name == "auto_detect_plan":
        await main.launch_auto_detect_task(task_id, *args, **kwargs)
    elif task_name == "structure_extraction":
        await main.launch_structure_task(task_id, *args, **kwargs)
    elif task_name == "bioactivity_extraction":
        await main.launch_assay_task(task_id, *args, **kwargs)
    elif task_name == "data_merge":
        await main.launch_merge_task(task_id, *args, **kwargs)
    elif task_name == "full_pipeline":
        await main.launch_full_pipeline_task(task_id, *args, **kwargs)
    else:
        main.task_manager.update(
            task_id,
            status="failed",
            progress=1.0,
            message="Unknown queued task type",
            error=f"Unsupported task_name: {task_name}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one BioChemInsight queue job in an isolated child process.")
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()
    job = _load_job(args.job_file)
    asyncio.run(_run_job(job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

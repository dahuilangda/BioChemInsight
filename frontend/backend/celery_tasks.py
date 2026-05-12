from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .celery_app import celery_app
from .work_queue import acquire_execution_lock, clear_inflight, get_job, release_execution_lock


def _float_setting(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw not in (None, ""):
        return float(raw)
    try:
        import constants

        return float(getattr(constants, name, default))
    except Exception:
        return float(default)


TASK_CHILD_POLL_SECONDS = max(0.5, _float_setting("TASK_CHILD_POLL_SECONDS", 2.0))
TASK_CHILD_TERMINATE_GRACE_SECONDS = max(1.0, _float_setting("TASK_CHILD_TERMINATE_GRACE_SECONDS", 20.0))
TASK_CHILD_TIMEOUT_SECONDS = max(0.0, _float_setting("TASK_CHILD_TIMEOUT_SECONDS", 0.0))


class ChildProcessStopped(Exception):
    pass


def _terminate_child(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_job_in_child(task_id: str, job: dict) -> None:
    from . import main

    payload = dict(job)
    payload["task_id"] = task_id
    job_file = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", prefix=f"biocheminsight_job_{task_id}_", delete=False) as fh:
            json.dump(payload, fh, ensure_ascii=False)
            job_file = fh.name

        command = [
            sys.executable,
            "-m",
            "frontend.backend.task_child_runner",
            "--job-file",
            job_file,
        ]
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), start_new_session=True)
        task = main.task_manager.get(task_id)
        if task is not None:
            metadata = dict(task.metadata or {})
            metadata.update(
                {
                    "execution_mode": "isolated_child_process",
                    "child_pid": process.pid,
                    "child_started_at": time.time(),
                }
            )
            main.task_manager.update(task_id, metadata=metadata)
        started = time.monotonic()
        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code == 0:
                    return
                task = main.task_manager.get(task_id)
                if task is not None and task.status in {"completed", "failed", "canceled"}:
                    return
                raise ChildProcessStopped(f"Task child process exited with code {return_code}")

            task = main.task_manager.get(task_id)
            if task is None:
                _terminate_child(process, TASK_CHILD_TERMINATE_GRACE_SECONDS)
                raise ChildProcessStopped("Task disappeared while child process was running")
            if task.status == "canceled":
                _terminate_child(process, TASK_CHILD_TERMINATE_GRACE_SECONDS)
                return

            elapsed = time.monotonic() - started
            if TASK_CHILD_TIMEOUT_SECONDS > 0 and elapsed > TASK_CHILD_TIMEOUT_SECONDS:
                _terminate_child(process, TASK_CHILD_TERMINATE_GRACE_SECONDS)
                main.task_manager.update(
                    task_id,
                    status="failed",
                    progress=1.0,
                    message="Task timed out",
                    error=f"Child process exceeded {int(TASK_CHILD_TIMEOUT_SECONDS)}s",
                )
                return
            time.sleep(TASK_CHILD_POLL_SECONDS)
    finally:
        if job_file:
            try:
                os.unlink(job_file)
            except OSError:
                pass


@celery_app.task(name="frontend.backend.celery_tasks.run_queued_task")
def run_queued_task(task_id: str) -> None:
    job = get_job(task_id)
    if not job:
        clear_inflight(task_id)
        return

    lock_token = acquire_execution_lock(task_id)
    if lock_token is None:
        # A duplicate Celery delivery can happen after Docker/Redis recovery.
        # Another worker thread already owns this BioChemInsight task id, so do
        # not clear the inflight/job records here.
        return

    try:
        from . import main

        task_record = main.task_manager.get(task_id)
        if task_record is not None and task_record.status in {"completed", "failed", "canceled"}:
            return

        _run_job_in_child(task_id, job)
    except ChildProcessStopped as exc:
        from . import main

        task = main.task_manager.get(task_id)
        if task is not None and task.status not in {"completed", "failed", "canceled"}:
            main.task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Task child process stopped",
                error=str(exc),
            )
    finally:
        clear_inflight(task_id)
        release_execution_lock(task_id, lock_token)

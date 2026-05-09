from __future__ import annotations

import os
import time

from .celery_app import celery_app
from .work_queue import inflight_count, mark_inflight, pop_next_job, requeue_front

try:
    import constants as project_constants
except ImportError:  # pragma: no cover
    project_constants = None


def _int_setting(name: str, default: int) -> int:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return int(env_value)
    if project_constants is not None and hasattr(project_constants, name):
        return int(getattr(project_constants, name))
    return default


def _float_setting(name: str, default: float) -> float:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return float(env_value)
    if project_constants is not None and hasattr(project_constants, name):
        return float(getattr(project_constants, name))
    return default


MAX_RUNNING = _int_setting("DISPATCHER_MAX_CONCURRENT_TASKS", _int_setting("MAX_CONCURRENT_TASKS", 2))
POLL_SECONDS = _float_setting("QUEUE_DISPATCHER_POLL_SECONDS", 1.0)


def dispatch_once() -> int:
    dispatched = 0
    while inflight_count() < MAX_RUNNING:
        job = pop_next_job()
        if not job:
            break
        task_id = job["task_id"]
        mark_inflight(task_id)
        try:
            celery_app.send_task(
                "frontend.backend.celery_tasks.run_queued_task",
                args=[task_id],
                queue="compute",
            )
        except Exception:
            requeue_front(task_id)
            raise
        dispatched += 1
    return dispatched


def main() -> None:
    print(f"Queue dispatcher started: max_running={MAX_RUNNING}, poll={POLL_SECONDS}s", flush=True)
    while True:
        try:
            dispatch_once()
        except Exception as exc:
            print(f"Queue dispatcher error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

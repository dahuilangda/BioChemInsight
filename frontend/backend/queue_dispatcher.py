from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Iterable

from .celery_app import celery_app
from .task_manager import create_task_manager
from .work_queue import (
    clear_inflight,
    get_job,
    inflight_count,
    inflight_task_ids,
    mark_inflight,
    pop_next_job,
    release_execution_lock,
    requeue_front,
)

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
STRUCTURE_MAX_RUNNING = max(1, _int_setting("STRUCTURE_TASK_CONCURRENCY", MAX_RUNNING))
POLL_SECONDS = _float_setting("QUEUE_DISPATCHER_POLL_SECONDS", 1.0)
# If Docker is force-recreated while jobs are running, Redis can preserve
# inflight ids even when Celery no longer has the corresponding task active.
# After this grace period, dispatcher verifies Celery active/reserved state and
# requeues orphaned running tasks. Set 0 to disable.
STALE_RUNNING_SECONDS = _float_setting("QUEUE_DISPATCHER_STALE_RUNNING_SECONDS", 300.0)
TERMINAL_STATUSES = {"completed", "failed", "canceled"}
STRUCTURE_HEAVY_TASKS = {"structure_extraction", "full_pipeline"}
task_manager = create_task_manager()


def _iter_celery_requests(payload: Any) -> Iterable[dict]:
    if not isinstance(payload, dict):
        return []
    requests = []
    for items in payload.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                request = item.get("request") if isinstance(item.get("request"), dict) else item
                requests.append(request)
    return requests


def _request_mentions_task(request: dict, task_id: str) -> bool:
    args = request.get("args")
    if isinstance(args, (list, tuple)):
        return bool(args and args[0] == task_id)
    if isinstance(args, str):
        return task_id in args
    return False


def _active_celery_task_ids() -> set[str] | None:
    try:
        inspector = celery_app.control.inspect(timeout=1.0)
        payloads = [inspector.active(), inspector.reserved(), inspector.scheduled()]
    except Exception as exc:
        print(f"Queue dispatcher could not inspect Celery workers: {exc}", flush=True)
        return None
    if all(payload is None for payload in payloads):
        return None
    active_ids: set[str] = set()
    for payload in payloads:
        for request in _iter_celery_requests(payload):
            args = request.get("args")
            if isinstance(args, (list, tuple)) and args:
                active_ids.add(str(args[0]))
            elif isinstance(args, str):
                # Celery may render args as a string in some transports.
                for token in args.replace("'", " ").replace('"', " ").replace("[", " ").replace("]", " ").split():
                    if len(token) == 32:
                        active_ids.add(token)
    return active_ids


def prune_stale_inflight() -> int:
    """Release inflight slots for terminal/missing tasks.

    Inflight lives in Redis so dispatcher restarts preserve concurrency state.
    If a running job is canceled, completes, or fails while the worker/container
    is interrupted before ``clear_inflight`` runs, the task id can remain in the
    inflight set forever. That blocks new jobs even though Celery has no active
    task. Before each dispatch loop, free those terminal or orphaned slots.
    """
    pruned = 0
    active_celery_ids: set[str] | None = None
    for task_id in inflight_task_ids():
        task = task_manager.get(task_id)
        job = get_job(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            if task is not None and task.status == "canceled":
                if active_celery_ids is None:
                    active_celery_ids = _active_celery_task_ids()
                if active_celery_ids is not None and task_id in active_celery_ids:
                    continue
            clear_inflight(task_id)
            release_execution_lock(task_id)
            pruned += 1
            continue
        if job is None:
            task_manager.update(
                task_id,
                status="failed",
                progress=1.0,
                message="Task stopped after restart",
                error=(
                    "The dispatcher lost the queued job payload during a restart, "
                    "so this task cannot be resumed automatically. Please start it again."
                ),
            )
            clear_inflight(task_id)
            release_execution_lock(task_id)
            pruned += 1
            continue
        if STALE_RUNNING_SECONDS > 0 and task.status == "running":
            age_seconds = (datetime.utcnow() - task.updated_at).total_seconds()
            if age_seconds < STALE_RUNNING_SECONDS:
                continue
            if active_celery_ids is None:
                active_celery_ids = _active_celery_task_ids()
            if active_celery_ids is None or task_id in active_celery_ids:
                continue
            task_manager.update(
                task_id,
                status="pending",
                message="Requeued after stale dispatcher inflight recovery",
            )
            release_execution_lock(task_id)
            requeue_front(task_id)
            pruned += 1
    return pruned


def dispatch_once() -> int:
    pruned = prune_stale_inflight()
    if pruned:
        print(f"Queue dispatcher pruned {pruned} stale inflight task(s)", flush=True)
    dispatched = 0
    structure_running = 0
    for task_id in inflight_task_ids():
        job = get_job(task_id)
        if job and job.get("task_name") in STRUCTURE_HEAVY_TASKS:
            structure_running += 1
    while inflight_count() < MAX_RUNNING:
        job = pop_next_job()
        if not job:
            break
        task_id = job["task_id"]
        if job.get("task_name") in STRUCTURE_HEAVY_TASKS and structure_running >= STRUCTURE_MAX_RUNNING:
            requeue_front(task_id)
            break
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
        if job.get("task_name") in STRUCTURE_HEAVY_TASKS:
            structure_running += 1
        dispatched += 1
    return dispatched


def main() -> None:
    print(
        f"Queue dispatcher started: max_running={MAX_RUNNING}, "
        f"structure_max_running={STRUCTURE_MAX_RUNNING}, poll={POLL_SECONDS}s",
        flush=True,
    )
    while True:
        try:
            dispatch_once()
        except Exception as exc:
            print(f"Queue dispatcher error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

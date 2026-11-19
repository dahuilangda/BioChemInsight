from __future__ import annotations

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .celery_app import celery_app
from .task_manager import create_task_manager
from .work_queue import (
    claim_next_job,
    clear_inflight,
    get_job,
    has_execution_lock,
    inflight_task_ids,
    is_task_queued,
    iter_job_task_ids,
    queued_task_ids,
    record_queue_event,
    release_execution_lock,
    requeue_front,
    reserve_orphan,
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


MAX_RUNNING = _int_setting("DISPATCHER_MAX_CONCURRENT_TASKS", _int_setting("MAX_CONCURRENT_TASKS", 3))
STRUCTURE_MAX_RUNNING = max(1, _int_setting("STRUCTURE_TASK_CONCURRENCY", MAX_RUNNING))
POLL_SECONDS = _float_setting("QUEUE_DISPATCHER_POLL_SECONDS", 1.0)
RECOVERY_INTERVAL_SECONDS = max(POLL_SECONDS, _float_setting("QUEUE_DISPATCHER_RECOVERY_INTERVAL_SECONDS", 15.0))
# If Docker is force-recreated while jobs are running, Redis can preserve
# inflight ids even when Celery no longer has the corresponding task active.
# After this grace period, dispatcher verifies Celery active/reserved state and
# requeues orphaned running tasks. Set 0 to disable.
STALE_RUNNING_SECONDS = _float_setting("QUEUE_DISPATCHER_STALE_RUNNING_SECONDS", 300.0)
PENDING_STALE_SECONDS = _float_setting("QUEUE_DISPATCHER_PENDING_STALE_SECONDS", 600.0)
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


def _active_celery_task_ids() -> set[str] | None:
    def inspect_state(state):
        inspector = celery_app.control.inspect(timeout=1.0)
        return getattr(inspector, state)()

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            payloads = list(executor.map(inspect_state, ('active', 'reserved', 'scheduled')))
    except Exception as exc:
        print(f"Queue dispatcher could not inspect Celery workers: {exc}", flush=True)
        return None
    if any(not isinstance(payload, dict) or not payload for payload in payloads):
        return None
    if any(payload.keys() != payloads[0].keys() for payload in payloads[1:]):
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
    inspected_workers = False
    for task_id in inflight_task_ids():
        task = task_manager.get(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            if has_execution_lock(task_id) or (task is not None and task.status == "canceled"):
                if task is not None and (datetime.utcnow() - task.updated_at).total_seconds() < POLL_SECONDS:
                    continue
                if not inspected_workers:
                    active_celery_ids = _active_celery_task_ids()
                    inspected_workers = True
                if active_celery_ids is None or task_id in active_celery_ids:
                    continue
            clear_inflight(task_id)
            release_execution_lock(task_id)
            pruned += 1
            continue
        if get_job(task_id) is None:
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
        if STALE_RUNNING_SECONDS > 0 and task.status in {"running", "pending"}:
            # Pending tasks normally leave this state within seconds (child
            # boot); a pending task still inflight after the stale window is
            # stranded (worker died before writing "running").
            stale_window = STALE_RUNNING_SECONDS if task.status == "running" else max(
                STALE_RUNNING_SECONDS, PENDING_STALE_SECONDS
            )
            age_seconds = (datetime.utcnow() - task.updated_at).total_seconds()
            if age_seconds < stale_window:
                continue
            if not inspected_workers:
                active_celery_ids = _active_celery_task_ids()
                inspected_workers = True
            if active_celery_ids is None or task_id in active_celery_ids:
                continue
            task_manager.update(
                task_id,
                status="pending",
                message="Requeued after stale dispatcher inflight recovery",
            )
            if requeue_front(task_id, recover=True):
                record_queue_event(task_id, "requeued", reason="stale_inflight")
            pruned += 1
    return pruned


def recover_orphaned_jobs() -> int:
    """Requeue persisted jobs that are no longer inflight or queued.

    A worker/container restart can leave this state:
    - task registry says ``running``;
    - job payload still exists;
    - execution lock still exists;
    - task id is absent from inflight and every partition queue.

    The normal stale-inflight path cannot see those jobs, so the UI would show
    a permanent running task. Requeue only after the same stale grace period and
    after Celery confirms the task is not active/reserved/scheduled.
    """
    recovered = 0
    scheduled_ids = queued_task_ids()
    scheduled_ids.update(inflight_task_ids())
    active_celery_ids: set[str] | None = None
    for task_id in iter_job_task_ids():
        if task_id in scheduled_ids:
            continue
        task = task_manager.get(task_id)
        job = get_job(task_id)
        if task is None or job is None:
            clear_inflight(task_id)
            release_execution_lock(task_id)
            recovered += 1
            continue
        partition_id = str(job.get("partition_id") or "unknown")
        if is_task_queued(task_id, partition_id):
            continue
        if task.status in TERMINAL_STATUSES:
            clear_inflight(task_id)
            release_execution_lock(task_id)
            recovered += 1
            continue
        if task.status not in {"running", "pending"}:
            continue
        if task.status == "running" and STALE_RUNNING_SECONDS > 0:
            age_seconds = (datetime.utcnow() - task.updated_at).total_seconds()
            if age_seconds < STALE_RUNNING_SECONDS:
                continue
        if active_celery_ids is None:
            active_celery_ids = _active_celery_task_ids()
        if active_celery_ids is None or task_id in active_celery_ids:
            continue
        if not reserve_orphan(task_id, partition_id):
            continue
        task = task_manager.get(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            clear_inflight(task_id)
            release_execution_lock(task_id)
            recovered += 1
            continue
        _kill_orphan_child(task_id, task)
        task_manager.update(
            task_id,
            status="pending",
            message="Requeued after stale orphaned job recovery",
        )
        if requeue_front(task_id, recover=True):
            record_queue_event(task_id, "requeued", reason="orphaned")
            recovered += 1
    return recovered


def _kill_orphan_child(task_id: str, task) -> None:
    try:
        child_pid = int((task.metadata or {}).get("child_pid") or 0)
    except (TypeError, ValueError):
        return
    if child_pid <= 1:
        return
    try:
        with open(f"/proc/{child_pid}/cmdline", encoding="utf-8", errors="replace") as handle:
            args = handle.read().rstrip("\0").split("\0")
    except OSError:
        return
    if len(args) < 5 or args[1:4] != ["-m", "frontend.backend.task_child_runner", "--job-file"]:
        return
    if not Path(args[4]).name.startswith(f"biocheminsight_job_{task_id}_"):
        return
    try:
        os.killpg(child_pid, signal.SIGKILL)
        print(
            f"Queue dispatcher killed orphan child {child_pid} for task {task_id}",
            flush=True,
        )
    except OSError:
        pass


def dispatch_once(*, recover_orphans: bool = True) -> int:
    pruned = prune_stale_inflight()
    if pruned:
        print(f"Queue dispatcher pruned {pruned} stale inflight task(s)", flush=True)
    if recover_orphans:
        recovered = recover_orphaned_jobs()
        if recovered:
            print(f"Queue dispatcher recovered {recovered} orphaned job(s)", flush=True)
    dispatched = 0
    while True:
        job = claim_next_job(MAX_RUNNING, STRUCTURE_MAX_RUNNING, STRUCTURE_HEAVY_TASKS)
        if not job:
            break
        task_id = job["task_id"]
        record_queue_event(task_id, "claimed", enqueued_at=job.get("created_at"))
        try:
            celery_app.send_task(
                "frontend.backend.celery_tasks.run_queued_task",
                args=[task_id],
                headers={"claim_id": job["claim_id"]},
                queue="compute",
            )
        except Exception as exc:
            requeued = requeue_front(task_id)
            record_queue_event(task_id, "publish_failed", error_type=type(exc).__name__, requeued=requeued)
            raise
        dispatched += 1
    return dispatched


def main() -> None:
    print(
        f"Queue dispatcher started: max_running={MAX_RUNNING}, "
        f"structure_max_running={STRUCTURE_MAX_RUNNING}, poll={POLL_SECONDS}s",
        flush=True,
    )
    next_recovery = 0.0
    while True:
        now = time.monotonic()
        recover_orphans = now >= next_recovery
        if recover_orphans:
            next_recovery = now + RECOVERY_INTERVAL_SECONDS
        try:
            dispatch_once(recover_orphans=recover_orphans)
        except Exception as exc:
            print(f"Queue dispatcher error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

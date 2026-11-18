from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import redis

from .task_manager import TASK_KEY_PREFIX

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
WORK_QUEUE_KEY_PREFIX = os.getenv("WORK_QUEUE_KEY_PREFIX", f"{TASK_KEY_PREFIX}:work_queue")
TASK_EXECUTION_LOCK_SECONDS = int(os.getenv("QUEUE_TASK_EXECUTION_LOCK_SECONDS", "86400") or "86400")

_ENQUEUE_SCRIPT = """
if not redis.call('SET', KEYS[4], ARGV[2], 'NX') then
  return 0
end
redis.call('RPUSH', KEYS[3], ARGV[1])
if redis.call('SADD', KEYS[2], ARGV[3]) == 1 then
  redis.call('RPUSH', KEYS[1], ARGV[3])
end
return 1
"""

_CLAIM_NEXT_SCRIPT = """
if redis.call('SCARD', KEYS[3]) >= tonumber(ARGV[3]) then
  return nil
end

local restricted = {}
for i = 5, #ARGV do
  restricted[ARGV[i]] = true
end
local restricted_running = 0
for _, task_id in ipairs(redis.call('SMEMBERS', KEYS[3])) do
  local raw = redis.call('GET', ARGV[2] .. task_id)
  if raw and restricted[cjson.decode(raw).task_name] then
    restricted_running = restricted_running + 1
  end
end
local allow_restricted = restricted_running < tonumber(ARGV[4])

local function take_job(queue_key)
  if allow_restricted then
    while true do
      local task_id = redis.call('LPOP', queue_key)
      if not task_id then return nil end
      local raw = redis.call('GET', ARGV[2] .. task_id)
      if raw and redis.call('SISMEMBER', KEYS[3], task_id) == 0 then
        return task_id, raw
      end
    end
  end
  for _, task_id in ipairs(redis.call('LRANGE', queue_key, 0, -1)) do
    local raw = redis.call('GET', ARGV[2] .. task_id)
    if not raw or redis.call('SISMEMBER', KEYS[3], task_id) == 1 then
      redis.call('LREM', queue_key, 0, task_id)
    elseif not restricted[cjson.decode(raw).task_name] then
      redis.call('LREM', queue_key, 0, task_id)
      return task_id, raw
    end
  end
end

for _, partition_id in ipairs(redis.call('LRANGE', KEYS[1], 0, -1)) do
  local queue_key = ARGV[1] .. partition_id .. ':queue'
  local task_id, raw = take_job(queue_key)
  if redis.call('LLEN', queue_key) == 0 then
    redis.call('LREM', KEYS[1], 0, partition_id)
    redis.call('SREM', KEYS[2], partition_id)
  elseif task_id then
    redis.call('LREM', KEYS[1], 0, partition_id)
    redis.call('RPUSH', KEYS[1], partition_id)
  end
  if task_id then
    redis.call('SADD', KEYS[3], task_id)
    return raw
  end
end

return nil
"""

_CANCEL_QUEUED_SCRIPT = """
if redis.call('SISMEMBER', KEYS[5], ARGV[1]) == 1 then
  return 0
end
local removed = redis.call('LREM', KEYS[3], 0, ARGV[1])
redis.call('DEL', KEYS[4])
if redis.call('LLEN', KEYS[3]) == 0 then
  redis.call('SREM', KEYS[2], ARGV[2])
  redis.call('LREM', KEYS[1], 0, ARGV[2])
end
return removed
"""

_REQUEUE_FRONT_SCRIPT = """
local raw = redis.call('GET', KEYS[4])
if not raw or redis.call('LPOS', KEYS[3], ARGV[1]) then
  return 0
end
redis.call('LPUSH', KEYS[3], ARGV[1])
redis.call('SREM', KEYS[5], ARGV[1])
if redis.call('SADD', KEYS[2], ARGV[2]) == 1 then
  redis.call('LPUSH', KEYS[1], ARGV[2])
end
return 1
"""


_REDIS_CLIENT: redis.Redis | None = None
_REDIS_CLIENT_LOCK = threading.Lock()


def get_redis() -> redis.Redis:
    """Shared client: redis-py connections are pooled and thread-safe, and a
    single client avoids building a new pool per call."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        with _REDIS_CLIENT_LOCK:
            if _REDIS_CLIENT is None:
                _REDIS_CLIENT = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _REDIS_CLIENT


def partitions_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partitions"


def active_partitions_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:active_partitions"


def inflight_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:inflight"


def execution_lock_key(task_id: str) -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:execution_lock:{task_id}"


def partition_queue_key(partition_id: str) -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partition:{partition_id}:queue"


def partition_queue_prefix() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partition:"


def job_key(task_id: str) -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:job:{task_id}"


def job_prefix() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:job:"


def iter_job_task_ids() -> Iterable[str]:
    """Yield task ids that still have a persisted queue job payload."""
    prefix = job_prefix()
    for key in get_redis().scan_iter(f"{prefix}*"):
        if key.startswith(prefix):
            yield key[len(prefix):]


def enqueue_task(
    task_id: str,
    task_name: str,
    partition_id: str,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    r = get_redis()
    partition_id = partition_id or "unknown"
    payload = {
        "task_id": task_id,
        "task_name": task_name,
        "partition_id": partition_id,
        "args": args or [],
        "kwargs": kwargs or {},
        "created_at": datetime.utcnow().isoformat(),
    }
    queue_key = partition_queue_key(partition_id)
    r.eval(
        _ENQUEUE_SCRIPT,
        4,
        partitions_key(),
        active_partitions_key(),
        queue_key,
        job_key(task_id),
        task_id,
        json.dumps(payload, ensure_ascii=False),
        partition_id,
    )


def get_job(task_id: str) -> Optional[Dict[str, Any]]:
    raw = get_redis().get(job_key(task_id))
    if not raw:
        return None
    return json.loads(raw)


def clear_inflight(task_id: str) -> None:
    r = get_redis()
    pipe = r.pipeline()
    pipe.srem(inflight_key(), task_id)
    pipe.delete(job_key(task_id))
    pipe.execute()


def acquire_execution_lock(task_id: str, ttl_seconds: int = TASK_EXECUTION_LOCK_SECONDS) -> Optional[str]:
    """Return a lock token when this worker may execute task_id.

    Redis/Celery can redeliver an old message after a Docker restart while the
    dispatcher also requeues the same BioChemInsight task id. The per-task lock
    prevents those duplicate Celery deliveries from running the expensive
    pipeline twice.
    """
    token = uuid.uuid4().hex
    acquired = get_redis().set(execution_lock_key(task_id), token, nx=True, ex=max(60, int(ttl_seconds)))
    return token if acquired else None


def release_execution_lock(task_id: str, token: Optional[str] = None) -> None:
    r = get_redis()
    key = execution_lock_key(task_id)
    if token is None:
        r.delete(key)
        return
    pipe = r.pipeline()
    while True:
        try:
            pipe.watch(key)
            current = pipe.get(key)
            if current != token:
                pipe.unwatch()
                return
            pipe.multi()
            pipe.delete(key)
            pipe.execute()
            return
        except redis.WatchError:
            continue


def inflight_task_ids() -> List[str]:
    return list(get_redis().smembers(inflight_key()) or [])


def inflight_count() -> int:
    return int(get_redis().scard(inflight_key()) or 0)


def is_task_queued(task_id: str, partition_id: Optional[str] = None) -> bool:
    r = get_redis()
    if partition_id and r.lpos(partition_queue_key(partition_id), task_id) is not None:
        return True
    partitions = [item for item in r.lrange(partitions_key(), 0, -1) if item != partition_id]
    if not partitions:
        return False
    pipe = r.pipeline(transaction=False)
    for candidate in partitions:
        pipe.lpos(partition_queue_key(candidate), task_id)
    return any(position is not None for position in pipe.execute())


def cancel_queued_task(task_id: str) -> bool:
    """Remove a task from the Redis dispatch queue if it has not started."""
    r = get_redis()
    job = get_job(task_id)
    if not job:
        return False
    partition_id = job.get("partition_id") or "unknown"
    return bool(r.eval(
        _CANCEL_QUEUED_SCRIPT,
        5,
        partitions_key(),
        active_partitions_key(),
        partition_queue_key(partition_id),
        job_key(task_id),
        inflight_key(),
        task_id,
        partition_id,
    ))


_QUEUE_POSITIONS_CACHE: tuple[float, Dict[str, int]] | None = None
_QUEUE_POSITIONS_TTL_SECONDS = float(os.getenv("QUEUE_POSITIONS_TTL_SECONDS", "2") or "2")


def get_queue_positions() -> Dict[str, int]:
    global _QUEUE_POSITIONS_CACHE
    cached = _QUEUE_POSITIONS_CACHE
    if cached is not None and time.monotonic() - cached[0] < _QUEUE_POSITIONS_TTL_SECONDS:
        return cached[1]
    r = get_redis()
    partitions = r.lrange(partitions_key(), 0, -1)
    pipe = r.pipeline(transaction=False)
    for partition in partitions:
        pipe.lrange(partition_queue_key(partition), 0, -1)
    queues = deque(map(deque, pipe.execute()))
    positions: Dict[str, int] = {}
    pos = 1
    while queues:
        queue = queues.popleft()
        if queue:
            positions[queue.popleft()] = pos
            pos += 1
            if queue:
                queues.append(queue)
    _QUEUE_POSITIONS_CACHE = (time.monotonic(), positions)
    return positions


def claim_next_job(
    max_running: int,
    restricted_limit: int,
    restricted_tasks: Iterable[str],
) -> Optional[Dict[str, Any]]:
    r = get_redis()
    raw = r.eval(
        _CLAIM_NEXT_SCRIPT,
        3,
        partitions_key(),
        active_partitions_key(),
        inflight_key(),
        partition_queue_prefix(),
        job_prefix(),
        max_running,
        restricted_limit,
        *sorted(restricted_tasks),
    )
    if raw:
        return json.loads(raw)
    return None


def requeue_front(task_id: str) -> bool:
    job = get_job(task_id)
    if not job:
        clear_inflight(task_id)
        return False
    partition_id = job.get("partition_id") or "unknown"
    r = get_redis()
    return bool(
        r.eval(
            _REQUEUE_FRONT_SCRIPT,
            5,
            partitions_key(),
            active_partitions_key(),
            partition_queue_key(partition_id),
            job_key(task_id),
            inflight_key(),
            task_id,
            partition_id,
        )
    )

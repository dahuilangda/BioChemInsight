from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

from .task_manager import TASK_KEY_PREFIX

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
WORK_QUEUE_KEY_PREFIX = os.getenv("WORK_QUEUE_KEY_PREFIX", f"{TASK_KEY_PREFIX}:work_queue")

_ENQUEUE_SCRIPT = """
redis.call('SET', KEYS[4], ARGV[2])
redis.call('RPUSH', KEYS[3], ARGV[1])
if redis.call('SADD', KEYS[2], ARGV[3]) == 1 then
  redis.call('RPUSH', KEYS[1], ARGV[3])
end
return 1
"""

_POP_NEXT_SCRIPT = """
local partition_count = redis.call('LLEN', KEYS[1])
if partition_count < 1 then
  return nil
end

for i = 1, partition_count do
  local partition_id = redis.call('LPOP', KEYS[1])
  if not partition_id then
    return nil
  end

  local queue_key = ARGV[1] .. partition_id .. ':queue'
  local task_id = redis.call('LPOP', queue_key)
  local remaining = redis.call('LLEN', queue_key)

  if remaining > 0 then
    redis.call('RPUSH', KEYS[1], partition_id)
  else
    redis.call('SREM', KEYS[2], partition_id)
  end

  if task_id then
    local raw = redis.call('GET', ARGV[2] .. task_id)
    if raw then
      return raw
    end
  end
end

return nil
"""

_REQUEUE_FRONT_SCRIPT = """
local raw = redis.call('GET', KEYS[4])
if not raw then
  return 0
end
redis.call('LPUSH', KEYS[3], ARGV[1])
redis.call('SREM', KEYS[5], ARGV[1])
if redis.call('SADD', KEYS[2], ARGV[2]) == 1 then
  redis.call('LPUSH', KEYS[1], ARGV[2])
end
return 1
"""


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def partitions_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partitions"


def active_partitions_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:active_partitions"


def inflight_key() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:inflight"


def partition_queue_key(partition_id: str) -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partition:{partition_id}:queue"


def partition_queue_prefix() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:partition:"


def job_key(task_id: str) -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:job:{task_id}"


def job_prefix() -> str:
    return f"{WORK_QUEUE_KEY_PREFIX}:job:"


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


def mark_inflight(task_id: str) -> None:
    get_redis().sadd(inflight_key(), task_id)


def clear_inflight(task_id: str) -> None:
    r = get_redis()
    pipe = r.pipeline()
    pipe.srem(inflight_key(), task_id)
    pipe.delete(job_key(task_id))
    pipe.execute()


def inflight_task_ids() -> List[str]:
    return list(get_redis().smembers(inflight_key()) or [])


def inflight_count() -> int:
    return int(get_redis().scard(inflight_key()) or 0)


def cancel_queued_task(task_id: str) -> bool:
    """Remove a task from the Redis dispatch queue if it has not started."""
    r = get_redis()
    job = get_job(task_id)
    if not job:
        return False
    if r.sismember(inflight_key(), task_id):
        return False

    partition_id = job.get("partition_id") or "unknown"
    queue_key = partition_queue_key(partition_id)
    removed = int(r.lrem(queue_key, 0, task_id) or 0)
    pipe = r.pipeline()
    pipe.delete(job_key(task_id))
    if r.llen(queue_key) == 0:
        pipe.srem(active_partitions_key(), partition_id)
        pipe.lrem(partitions_key(), 0, partition_id)
    pipe.execute()
    return removed > 0


def get_queue_positions() -> Dict[str, int]:
    """Return positions for queued task ids."""
    r = get_redis()
    partitions = r.lrange(partitions_key(), 0, -1)
    queues = {partition: r.lrange(partition_queue_key(partition), 0, -1) for partition in partitions}
    positions: Dict[str, int] = {}
    pos = 1
    while True:
        progressed = False
        for partition in partitions:
            queue = queues.get(partition) or []
            if queue:
                task_id = queue.pop(0)
                positions[task_id] = pos
                pos += 1
                progressed = True
        if not progressed:
            break
    return positions


def pop_next_job() -> Optional[Dict[str, Any]]:
    """Pop one queued job."""
    r = get_redis()
    raw = r.eval(
        _POP_NEXT_SCRIPT,
        2,
        partitions_key(),
        active_partitions_key(),
        partition_queue_prefix(),
        job_prefix(),
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

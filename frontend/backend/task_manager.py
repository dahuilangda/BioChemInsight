from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis


TASK_KEY_PREFIX = os.getenv("TASK_KEY_PREFIX", "biocheminsight")


@dataclass
class Task:
    """Represents a long-running backend job."""

    id: str
    type: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    pdf_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result_path: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        payload = asdict(self)
        if not include_data:
            payload.pop("data", None)
        payload["task_id"] = payload.pop("id")
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _task_to_storage(task: Task) -> str:
    payload = asdict(task)
    payload["created_at"] = task.created_at.isoformat()
    payload["updated_at"] = task.updated_at.isoformat()
    return json.dumps(_json_safe(payload), ensure_ascii=False)


def _task_from_storage(raw: str | bytes | Dict[str, Any]) -> Task:
    payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw) if not isinstance(raw, dict) else raw
    payload["created_at"] = _parse_datetime(payload.get("created_at"))
    payload["updated_at"] = _parse_datetime(payload.get("updated_at"))
    return Task(**payload)


class RedisTaskManager:
    """Redis-backed task registry so task cards survive web/worker restarts."""

    def __init__(self, redis_url: str, key_prefix: str = TASK_KEY_PREFIX) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis = redis.Redis.from_url(redis_url, decode_responses=False)

    @property
    def index_key(self) -> str:
        return f"{self.key_prefix}:tasks:index"

    def task_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:tasks:{task_id}"

    def create(self, task_type: str, pdf_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(id=task_id, type=task_type, pdf_id=pdf_id, params=params or {}, metadata=metadata or {})
        pipe = self.redis.pipeline()
        pipe.set(self.task_key(task_id), _task_to_storage(task))
        pipe.zadd(self.index_key, {task_id: task.created_at.timestamp()})
        pipe.execute()
        return task

    def get(self, task_id: str) -> Optional[Task]:
        raw = self.redis.get(self.task_key(task_id))
        if raw is None:
            return None
        return _task_from_storage(raw)

    def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        task = self.get(task_id)
        if not task:
            return None
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, _json_safe(value))
        task.updated_at = datetime.utcnow()
        self.redis.set(self.task_key(task_id), _task_to_storage(task))
        return task

    def list(self) -> List[Task]:
        task_ids = self.redis.zrevrange(self.index_key, 0, -1)
        tasks: List[Task] = []
        for raw_task_id in task_ids:
            task_id = raw_task_id.decode("utf-8") if isinstance(raw_task_id, bytes) else str(raw_task_id)
            task = self.get(task_id)
            if task is not None:
                tasks.append(task)
        return tasks


def create_task_manager() -> RedisTaskManager:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return RedisTaskManager(redis_url)


def ensure_task(task: Optional[Task], task_id: str) -> Task:
    if task is None:
        raise KeyError(f"Task '{task_id}' not found")
    return task

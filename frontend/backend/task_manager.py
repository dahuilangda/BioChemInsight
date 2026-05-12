from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis


TASK_KEY_PREFIX = os.getenv("TASK_KEY_PREFIX", "biocheminsight")


_NON_TERMINAL_MESSAGE_PREFIXES = (
    "Auto-detecting ",
    "Compiling ",
    "Detecting ",
    "Extracting ",
    "Loading ",
    "Merging ",
    "Post-processing ",
    "Preparing ",
    "Processing ",
)


def _looks_like_non_terminal_message(message: str) -> bool:
    return message.startswith(_NON_TERMINAL_MESSAGE_PREFIXES)


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
        status = str(payload.get("status") or "")
        if status in {"completed", "canceled"}:
            # Older persisted task records may have a terminal status with the
            # last in-flight progress/message (for example "Processing page …"
            # at 50%). Normalize the API representation so Jobs is consistent
            # without mutating historical Redis records.
            payload["progress"] = 1.0
        if status == "completed":
            message = str(payload.get("message") or "").strip()
            if not message or _looks_like_non_terminal_message(message):
                payload["message"] = "Completed"
        elif status == "canceled" and not str(payload.get("message") or "").strip():
            payload["message"] = "Canceled"
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


def _task_summary_payload(task: Task) -> Dict[str, Any]:
    status = str(task.status or "")
    progress = 1.0 if status in {"completed", "canceled"} else float(task.progress or 0.0)
    message = str(task.message or "").strip()
    if status == "completed" and (not message or _looks_like_non_terminal_message(message)):
        message = "Completed"
    elif status == "canceled" and not message:
        message = "Canceled"
    return {
        "task_id": task.id,
        "type": task.type,
        "status": status,
        "progress": progress,
        "message": message,
        "pdf_id": task.pdf_id,
        "result_path": task.result_path,
        "error": task.error,
        "params": {},
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


_TASK_SUMMARY_FIELDS = {
    "id",
    "type",
    "status",
    "progress",
    "message",
    "pdf_id",
    "result_path",
    "error",
    "created_at",
    "updated_at",
}


def _skip_json_string(raw: str, index: int) -> int:
    index += 1
    while index < len(raw):
        char = raw[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return index


def _skip_json_container(raw: str, index: int) -> int:
    stack = [raw[index]]
    index += 1
    in_string = False
    while index < len(raw) and stack:
        char = raw[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack[-1] == "[":
            stack.pop()
        index += 1
    return index


def _skip_json_value(raw: str, index: int) -> int:
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw):
        return index
    char = raw[index]
    if char == '"':
        return _skip_json_string(raw, index)
    if char in "{[":
        return _skip_json_container(raw, index)
    while index < len(raw) and raw[index] not in ",}":
        index += 1
    return index


def _task_summary_from_storage(raw: str | bytes | Dict[str, Any]) -> Dict[str, Any]:
    """Read list-view fields without decoding large params/data payloads."""
    if isinstance(raw, dict):
        payload = raw
    else:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        decoder = json.JSONDecoder()
        payload: Dict[str, Any] = {}
        index = text.find("{") + 1
        if index <= 0:
            raise ValueError("Invalid task JSON")
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text) or text[index] == "}":
                break
            key, index = decoder.raw_decode(text, index)
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text) or text[index] != ":":
                raise ValueError("Invalid task JSON")
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            if key in _TASK_SUMMARY_FIELDS:
                payload[key], index = decoder.raw_decode(text, index)
            else:
                index = _skip_json_value(text, index)
            while index < len(text) and text[index].isspace():
                index += 1
            if index < len(text) and text[index] == ",":
                index += 1

    status = str(payload.get("status") or "")
    message = str(payload.get("message") or "").strip()
    if status in {"completed", "canceled"}:
        payload["progress"] = 1.0
    if status == "completed" and (not message or _looks_like_non_terminal_message(message)):
        payload["message"] = "Completed"
    elif status == "canceled" and not message:
        payload["message"] = "Canceled"
    return {
        "task_id": str(payload.get("id") or ""),
        "type": str(payload.get("type") or ""),
        "status": status,
        "progress": float(payload.get("progress") or 0.0),
        "message": str(payload.get("message") or ""),
        "pdf_id": payload.get("pdf_id"),
        "result_path": payload.get("result_path"),
        "error": payload.get("error"),
        "params": {},
        "created_at": str(payload.get("created_at") or ""),
        "updated_at": str(payload.get("updated_at") or ""),
    }


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

    def task_summary_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:tasks:{task_id}:summary"

    def create(self, task_type: str, pdf_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(id=task_id, type=task_type, pdf_id=pdf_id, params=params or {}, metadata=metadata or {})
        pipe = self.redis.pipeline()
        pipe.set(self.task_key(task_id), _task_to_storage(task))
        pipe.set(self.task_summary_key(task_id), json.dumps(_task_summary_payload(task), ensure_ascii=False))
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
        next_status = str(fields.get("status", task.status) or "")
        if next_status in {"completed", "canceled"}:
            fields["progress"] = 1.0
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, _json_safe(value))
        task.updated_at = datetime.utcnow()
        pipe = self.redis.pipeline()
        pipe.set(self.task_key(task_id), _task_to_storage(task))
        pipe.set(self.task_summary_key(task_id), json.dumps(_task_summary_payload(task), ensure_ascii=False))
        pipe.execute()
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

    def list_summaries(self) -> List[Dict[str, Any]]:
        task_ids = self.redis.zrevrange(self.index_key, 0, -1)
        if not task_ids:
            return []
        normalized_ids = [raw_task_id.decode("utf-8") if isinstance(raw_task_id, bytes) else str(raw_task_id) for raw_task_id in task_ids]
        keys = [self.task_summary_key(task_id) for task_id in normalized_ids]
        summaries: List[Dict[str, Any]] = []
        missing_ids: List[str] = []
        for task_id, raw in zip(normalized_ids, self.redis.mget(keys)):
            if raw is None:
                missing_ids.append(task_id)
                continue
            try:
                summaries.append(json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw))
            except Exception:
                missing_ids.append(task_id)
        if missing_ids:
            pipe = self.redis.pipeline()
            for task_id in missing_ids:
                pipe.get(self.task_key(task_id))
            fallback_raw_items = pipe.execute()
            pipe = self.redis.pipeline()
            for task_id, raw in zip(missing_ids, fallback_raw_items):
                if raw is None:
                    continue
                try:
                    summary = _task_summary_from_storage(raw)
                except Exception:
                    summary = _task_from_storage(raw).to_dict()
                    summary["params"] = {}
                summaries.append(summary)
                pipe.set(self.task_summary_key(task_id), json.dumps(summary, ensure_ascii=False))
            pipe.execute()
        return summaries


def create_task_manager() -> RedisTaskManager:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return RedisTaskManager(redis_url)


def ensure_task(task: Optional[Task], task_id: str) -> Task:
    if task is None:
        raise KeyError(f"Task '{task_id}' not found")
    return task

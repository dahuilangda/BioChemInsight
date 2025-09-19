from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


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


class TaskManager:
    """In-memory task registry with thread-safe updates."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def create(self, task_type: str, pdf_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(id=task_id, type=task_type, pdf_id=pdf_id, params=params or {})
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.utcnow()
            return task

    def list(self) -> List[Task]:
        with self._lock:
            return list(self._tasks.values())


def ensure_task(task: Optional[Task], task_id: str) -> Task:
    if task is None:
        raise KeyError(f"Task '{task_id}' not found")
    return task

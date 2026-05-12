from __future__ import annotations

import subprocess
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from frontend.backend import celery_tasks


@dataclass
class DummyTask:
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)


class DummyTaskManager:
    def __init__(self, task: DummyTask) -> None:
        self.task = task
        self.updates: list[dict[str, Any]] = []
        self.get_calls = 0

    def get(self, task_id: str) -> DummyTask:
        self.get_calls += 1
        if self.get_calls >= 2:
            self.task.status = "canceled"
        return self.task

    def update(self, task_id: str, **fields: Any) -> DummyTask:
        self.updates.append(fields)
        for key, value in fields.items():
            setattr(self.task, key, value)
        return self.task


class ChildRunnerTests(unittest.TestCase):
    def test_run_job_in_child_terminates_process_on_cancel(self) -> None:
        task = DummyTask()
        manager = DummyTaskManager(task)
        fake_main = types.ModuleType("frontend.backend.main")
        fake_main.task_manager = manager

        process = subprocess.Popen(["sleep", "30"], start_new_session=True)

        class FakePopen:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._process = process
                self.pid = process.pid

            def poll(self) -> int | None:
                return self._process.poll()

        with (
            mock.patch.dict(sys.modules, {"frontend.backend.main": fake_main}),
            mock.patch.object(celery_tasks, "TASK_CHILD_POLL_SECONDS", 0.01),
            mock.patch.object(celery_tasks, "TASK_CHILD_TERMINATE_GRACE_SECONDS", 0.5),
            mock.patch.object(celery_tasks, "TASK_CHILD_TIMEOUT_SECONDS", 0),
            mock.patch.object(celery_tasks.subprocess, "Popen", FakePopen),
        ):
            celery_tasks._run_job_in_child("abc123", {"task_name": "full_pipeline", "args": [], "kwargs": {}})

        self.assertIsNotNone(process.poll())
        self.assertTrue(
            any(update.get("metadata", {}).get("execution_mode") == "isolated_child_process" for update in manager.updates)
        )


if __name__ == "__main__":
    unittest.main()

"""Persistent client for the MolSight inference process."""
from __future__ import annotations

from contextlib import suppress
import json
import os
import queue
import subprocess
import threading

_WORKER_SCRIPT = os.environ.get(
    "MOLSIGHT_WORKER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "molsight_worker.py"))
_VENV_PYTHON = os.environ.get(
    "MOLSIGHT_VENV_PYTHON", "/data/molsight_venv/bin/python")


class MolsightService:
    """Lazy persistent MolSight worker client (fork-safe via lock)."""

    def __init__(self):
        self._process = None
        self._replies: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._startup_timeout = float(os.environ.get("MOLSIGHT_STARTUP_TIMEOUT_S", "120"))
        self._prediction_timeout = float(os.environ.get("MOLSIGHT_PREDICTION_TIMEOUT_S", "90"))
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return (os.environ.get("MOLSIGHT_ENABLED", "1") != "0"
                and os.path.exists(_WORKER_SCRIPT)
                and os.path.exists(_VENV_PYTHON))

    def _start(self):
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("MOLSIGHT_GPU", "0"))
        with open(os.environ.get(
            "MOLSIGHT_WORKER_LOG", "/tmp/molsight_worker_stderr.log"
        ), "ab") as log:
            self._process = subprocess.Popen(
                [_VENV_PYTHON, _WORKER_SCRIPT,
                 "--device", os.environ.get("MOLSIGHT_DEVICE", "cuda:0")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True, bufsize=1, env=env)
        self._replies = queue.SimpleQueue()
        threading.Thread(
            target=self._read_replies, args=(self._process.stdout, self._replies),
            daemon=True,
        ).start()

    @staticmethod
    def _read_replies(stream, replies):
        try:
            with stream:
                for line in stream:
                    replies.put(line)
        finally:
            replies.put("")

    def _read_reply(self, timeout: float) -> str:
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"MolSight worker did not respond in {timeout:g}s") from exc
        if not reply:
            raise RuntimeError("MolSight worker closed its response stream")
        return reply

    def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            process.kill()
            process.wait()
            with suppress(OSError):
                process.stdin.close()

    def _ensure_started(self):
        if self._process is None or self._process.poll() is not None:
            self._stop()
            self._start()
            line = self._read_reply(self._startup_timeout)
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("event") != "ready":
                raise RuntimeError(f"MolSight worker failed to start: {line[:200]}")

    def predict_many(self, image_paths: list[str], max_retries: int = 1) -> list[dict]:
        """Return ordered predictions and retry only unfinished inputs."""
        if not image_paths:
            return []
        with self._lock:
            results: list[dict | None] = [None] * len(image_paths)
            last_error = "worker_unavailable"
            for _ in range(max_retries + 1):
                try:
                    self._ensure_started()
                    outstanding = [i for i, r in enumerate(results) if r is None]
                    # Interleaved write/read — write-all-then-read-all deadlocks
                    # once the OS pipe buffer fills.
                    for i in outstanding:
                        self._process.stdin.write(
                            json.dumps({"image_path": image_paths[i]}) + "\n")
                        self._process.stdin.flush()
                        line = self._read_reply(self._prediction_timeout)
                        result = json.loads(line)
                        if not isinstance(result, dict):
                            raise RuntimeError("MolSight worker returned a non-object response")
                        results[i] = result
                    break
                except (RuntimeError, json.JSONDecodeError, OSError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"[:200]
                    self._stop()
            return [r if r is not None else {"ok": False, "error": last_error}
                    for r in results]

    def predict(self, image_path: str) -> dict:
        return self.predict_many([image_path])[0]

    def shutdown(self):
        with self._lock:
            self._stop()


_SERVICE: MolsightService | None = None
_SERVICE_LOCK = threading.Lock()


def molsight_service() -> MolsightService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MolsightService()
        return _SERVICE

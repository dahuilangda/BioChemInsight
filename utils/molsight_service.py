"""MolSight integration service (§44.70 routing harness).

Production client for the MolSight OCSR model — the only backend in the stack
that decodes large macrocycles (≥20-membered rings, cyclic peptides) exactly.
Routing role: complete structures AND large-macrocycle markush images where
keeper breaks the ring and MarkushGlyph gets the composition wrong.

Same worker protocol as markush_glyph_service.py:
  - persistent worker subprocess inside the isolated molsight_venv
  - lazy start, batch predict_many with interleaved write/read, death-detection
  - fail-open: every failure degrades to {"ok": False} — the caller falls back

Environment:
  MOLSIGHT_VENV_PYTHON   molsight venv python (default /data/molsight_venv/bin/python)
  MOLSIGHT_WORKER        worker script path (default repo utils/molsight_worker.py)
  MOLSIGHT_ENABLED       set "0" to disable (default on; worker missing ⇒ off)
  MOLSIGHT_DEVICE        torch device passed to the worker (default cuda:0)
  CUDA_VISIBLE_DEVICES   pin the GPU for the worker subprocess
"""
from __future__ import annotations

import json
import os
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
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return (os.environ.get("MOLSIGHT_ENABLED", "1") != "0"
                and os.path.exists(_WORKER_SCRIPT)
                and os.path.exists(_VENV_PYTHON))

    def _start(self):
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("MOLSIGHT_GPU", "0"))
        self._process = subprocess.Popen(
            [_VENV_PYTHON, _WORKER_SCRIPT,
             "--device", os.environ.get("MOLSIGHT_DEVICE", "cuda:0")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)

    def _ensure_started(self):
        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass
            self._start()
            line = self._process.stdout.readline()
            ready = json.loads(line) if line else {}
            if ready.get("event") != "ready":
                raise RuntimeError(f"molsight worker failed to start: {line[:200]}")

    def predict_many(self, image_paths: list[str], max_retries: int = 1) -> list[dict]:
        """Batch prediction; returns one result dict per input, order-stable."""
        with self._lock:
            results: list[dict | None] = [None] * len(image_paths)
            for attempt in range(max_retries + 1):
                try:
                    self._ensure_started()
                    outstanding = [i for i, r in enumerate(results) if r is None]
                    # Interleaved write/read — write-all-then-read-all deadlocks
                    # once the OS pipe buffer fills.
                    for i in outstanding:
                        self._process.stdin.write(
                            json.dumps({"image_path": image_paths[i]}) + "\n")
                        self._process.stdin.flush()
                        line = self._process.stdout.readline()
                        if not line:
                            raise RuntimeError("worker died (empty response)")
                        results[i] = json.loads(line)
                    break
                except (RuntimeError, json.JSONDecodeError, OSError):
                    if attempt >= max_retries:
                        break
                    try:
                        if self._process is not None:
                            self._process.kill()
                    except OSError:
                        pass
                    self._process = None
            return [r if r is not None else {"ok": False, "error": "worker_unavailable"}
                    for r in results]

    def predict(self, image_path: str) -> dict:
        return self.predict_many([image_path])[0]

    def shutdown(self):
        with self._lock:
            if self._process is not None:
                try:
                    self._process.stdin.close()
                    self._process.terminate()
                except OSError:
                    pass
                self._process = None


_SERVICE: MolsightService | None = None
_SERVICE_LOCK = threading.Lock()


def molsight_service() -> MolsightService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MolsightService()
        return _SERVICE

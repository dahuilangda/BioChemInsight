#!/bin/bash
set -e

cd /app
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000 &
backend_pid=$!

cd /app/frontend/ui
python /app/static-server.py dist --host 0.0.0.0 --port 3000 &
frontend_pid=$!

wait -n "$backend_pid" "$frontend_pid"
exit_code=$?
kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
exit "$exit_code"

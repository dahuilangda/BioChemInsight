"""Frontend review workspace URL construction."""
from __future__ import annotations

import os


def review_workspace_url(task_id: str) -> str:
    base = os.environ.get("BIOCHEMINSIGHT_FRONTEND_URL",
                          os.environ.get("BIOCHEMINSIGHT_URL",
                                         "http://127.0.0.1:8000"))
    return f"{base.rstrip('/')}/review/{task_id}"

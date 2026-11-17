"""Extract and review source-linked compound measurements."""

import asyncio

from skills.registry import register
from utils.measurement_review import review_measurements


@register(
    "activity_tables",
    "model",
    "Extract and review compound measurements with source evidence.",
    input_schema={
        "type": "object",
        "properties": {
            "markdown": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "audit_path": {"type": "string", "minLength": 1},
        },
        "anyOf": [{"required": ["markdown"]}, {"required": ["text"]}],
    },
)
async def parse(payload: dict) -> dict:
    source = payload.get("markdown", payload.get("text"))
    if not isinstance(source, str) or not source.strip():
        return {"ok": False, "error": "Non-empty measurement source required"}
    audit_path = payload.get("audit_path")
    if audit_path is not None and (
        not isinstance(audit_path, str) or not audit_path.strip()
    ):
        return {"ok": False, "error": "audit_path must be a non-empty path"}
    return await asyncio.to_thread(review_measurements, source, audit_path=audit_path)

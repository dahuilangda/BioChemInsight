"""Read sequence identifiers from structure figures."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from skills.registry import register
from utils.model_harness import (
    ModelContractError,
    ModelEvidenceError,
    require_json_object,
)

_PROMPT = """Read the sequence identifier printed on this structure figure.
The images show the bottom label strip, top label strip, and full crop of the same figure.
Copy the identifier and its printed label. A FIG number is not a sequence identifier.
Use status="found" only when the annotation unambiguously labels the whole figure.
Use status="absent" when no sequence annotation is visible.
Use status="uncertain" for illegible annotations or competing labels whose ownership
cannot be determined. Do not choose a label by position or infer missing digits.
For absent or uncertain results, seq_id, label_text, and confidence must be null.
Return only JSON with status, seq_id, label_text, and confidence.
"""


def _band_geometry(height: int, width: int) -> list[tuple[str, int, int]] | None:
    """Return label strips for crops that are not already band-shaped."""
    if width <= 0 or height * 3 <= width:
        return None
    band_px = int(min(max(0.35 * height, 60), 320))
    return [
        ("band_bottom", max(0, height - band_px), height),
        ("band_top", 0, min(height, band_px)),
    ]


def _prepare_views(
    image: np.ndarray, debug_dir: str | None = None, stem: str = "crop"
) -> list[tuple[str, bytes]]:
    """Encode the views sent to the model: bottom band, top band, then the
    full crop. White framing keeps glyphs at crop edges legible."""
    height, width = image.shape[:2]
    views: list[tuple[str, np.ndarray]] = []
    for name, y0, y1 in _band_geometry(height, width) or []:
        strip = image[y0:y1]
        if strip.shape[1] < 1000:
            scale = 2.0
            strip = cv2.resize(
                strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        views.append((name, strip))
    framed = cv2.copyMakeBorder(
        image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    views.append(("full", framed))

    encoded: list[tuple[str, bytes]] = []
    for name, view in views:
        ok, buf = cv2.imencode(".png", view)
        if not ok:
            raise RuntimeError(f"could not encode {name} view of the crop")
        encoded.append((name, buf.tobytes()))
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, f"{stem}_view_{name}.png")
            if not cv2.imwrite(path, view):
                raise OSError(f"Could not write debug view: {path}")
    return encoded


_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "seq_id", "label_text", "confidence"],
    "properties": {
        "status": {"type": "string", "enum": ["found", "absent", "uncertain"]},
        "seq_id": {"type": ["integer", "null"], "minimum": 1},
        "label_text": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    },
}


def parse_annotation(text: str) -> dict:
    result = require_json_object(text, "seq_id_vision", allow_wrapped=False)
    if set(result) != set(_RESPONSE_SCHEMA["required"]):
        raise ModelContractError("Sequence annotation fields do not match the contract")
    status = result["status"]
    if not isinstance(status, str) or status not in {"found", "absent", "uncertain"}:
        raise ModelContractError("Unknown sequence annotation status")
    if status != "found":
        if any(
            result[key] is not None for key in ("seq_id", "label_text", "confidence")
        ):
            raise ModelContractError(
                "Absent annotations require null annotation fields"
            )
        if status == "uncertain":
            raise ModelEvidenceError("Sequence annotation is ambiguous or illegible")
        return result
    seq_id, label, confidence = (
        result["seq_id"],
        result["label_text"],
        result["confidence"],
    )
    if type(seq_id) is not int or seq_id < 1:
        raise ModelContractError("seq_id must be a positive integer")
    if not isinstance(label, str) or not label.strip():
        raise ModelContractError("A detected annotation requires its printed label")
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ModelContractError(
            "confidence must be a finite number between zero and one"
        )
    return result


def _read_annotation(image_path: str, debug_dir: str | None, audit_path) -> dict:
    from utils.llm_utils import run_vision_json_task

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    views = _prepare_views(image, debug_dir, stem=Path(image_path).stem)
    with TemporaryDirectory(prefix="sequence-annotation-") as directory:
        paths = []
        for name, content in views:
            path = Path(directory) / f"{name}.png"
            path.write_bytes(content)
            paths.append(path)
        result = run_vision_json_task(
            task_name="seq_id_vision",
            image_files=paths,
            prompt=_PROMPT,
            parser=parse_annotation,
            preserve_response=True,
            response_schema=_RESPONSE_SCHEMA,
            audit_path=audit_path,
            metadata={"source_image": image_path, "views": [name for name, _ in views]},
        )
    return {
        "ok": True,
        **result,
        "found": result["status"] == "found",
        "views_sent": [name for name, _ in views],
        **(
            {"reason": "no_seq_id_label_visible"}
            if result["status"] == "absent"
            else {}
        ),
    }


@register(
    "seq_id_vision",
    "model",
    "Read the printed sequence identifier from a structure crop.",
    input_schema={
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "minLength": 1},
            "crop_path": {"type": "string", "minLength": 1},
            "debug_dir": {"type": "string", "minLength": 1},
            "audit_path": {"type": "string", "minLength": 1},
        },
        "anyOf": [{"required": ["image_path"]}, {"required": ["crop_path"]}],
    },
)
async def read_seq_id_label(payload: dict) -> dict:
    image_path = payload.get("image_path", payload.get("crop_path"))
    if not isinstance(image_path, str) or not image_path.strip():
        return {"ok": False, "error": "image_path or crop_path is required"}
    if not Path(image_path).is_file():
        return {"ok": False, "error": f"Image not found: {image_path}"}
    for field in ("debug_dir", "audit_path"):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return {"ok": False, "error": f"{field} must be a non-empty path"}
    return await asyncio.to_thread(
        _read_annotation,
        image_path,
        payload.get("debug_dir"),
        payload.get("audit_path"),
    )

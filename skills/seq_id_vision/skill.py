"""Atomic skill: read the SEQ ID annotation off a structure crop, visually.

Given a cropped structure figure (e.g. a boxed patent FIG structure or an
already-cut label band), a vision model looks for the printed "SEQ ID NO: N"
annotation (or an equivalent printed variant such as "SEQ ID No.17" /
"(SEQ ID NO: 17)") and copies the number out verbatim. The result feeds the
figure FSM's LABEL stage, which stores it as FigureGoal.seq_id so the
activity-join can link the figure to the sequence table on the same page.

Design notes:
- Visual reading only, deliberately: patent figure labels use dozens of
  micro-variants ("SEQ ID No:", "SEQ.ID.NO", italic "No", smudged digits)
  that a regex on OCR text quietly misses or mis-copies; the model reads
  the glyphs directly instead.
- A missing annotation is a named result (found=False), never an exception
  and never a guessed number: callers branch on `found`, they do not catch.
- Up to three views are sent per call (bottom label band, top label
  band, full crop) because the patent convention usually prints the label
  under the box — but above-the-figure annotations with a pointer squiggle
  (e.g. "(SEQ ID NO:1)" over a peptide row) are equally common; the two
  band strips match services/structure_api._label_bands, and the full crop
  covers side-of-box and inside-box annotations.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import base64

import cv2
import numpy as np

from skills.registry import register

_PROMPT = """The images show views of the same cropped structure figure \
from a patent page: zoomed label strips first (bottom strip, then top \
strip), then the full crop.

TASK: read the printed sequence identifier annotation on this figure. The \
first images are zoomed label strips (bottom strip, then top strip); the \
last image is the full crop. Look \
for text of the form "SEQ ID NO: 17" — patents also print it as "SEQ ID \
NO. 17", "SEQ ID No:17", "SEQ ID NO 17", "Sequence ID NO: 17", often in \
parentheses "(SEQ ID NO: 17)". A figure tag such as "FIG. 8B" may sit \
nearby; the figure number is NOT the sequence id.

RULES:
- Copy the number EXACTLY as printed. Never infer or guess a number from \
context, sequence length, or position in the patent.
- If several SEQ ID annotations are visible, report the one that labels \
the figure block as a whole: the annotation attached to the block heading \
or pointer (e.g. written right after a block name like "B2. (SEQ ID \
No:22)"), otherwise the topmost annotation. Do not report a label that \
marks only one strand or chain of a multi-strand drawing.
- If no SEQ ID annotation is visible in either view, set "found" to false \
and all other fields to null.

Respond with ONLY this JSON object — the output must start with '{':
{"found": true|false, "seq_id": <integer or null>, "label_text": "<exact \
printed text or null>", "confidence": <0.0-1.0 or null>}
"""


def _band_geometry(height: int, width: int) -> list[tuple[str, int, int]] | None:
    """Label strips as (name, y0, y1), or None if the crop is already
    band-shaped.

    Patent convention prints the label under the figure box, with the
    above-the-figure pointer annotation ("(SEQ ID NO:1)" over a peptide
    row) as the second convention — the same two strips
    services/structure_api._label_bands reads for OCR. A thin horizontal
    input is itself a label band, so re-cutting it would discard glyph
    rows."""
    if width <= 0 or height * 3 <= width:   # aspect >= 3:1 is band-shaped
        return None
    band_px = int(min(max(0.35 * height, 60), 320))
    return [("band_bottom", max(0, height - band_px), height),
            ("band_top", 0, min(height, band_px))]


def _prepare_views(image: np.ndarray, debug_dir: str | None = None,
                   stem: str = "crop") -> list[tuple[str, bytes]]:
    """Encode the views sent to the model: bottom band, top band, then the
    full crop. White framing keeps glyphs at crop edges legible."""
    height, width = image.shape[:2]
    views: list[tuple[str, np.ndarray]] = []
    for name, y0, y1 in (_band_geometry(height, width) or []):
        strip = image[y0:y1]
        # Small bands upscale ~2x so digit glyphs give the model more
        # pixels; patent label type is ~10 pt and reads poorly at 1x.
        if strip.shape[1] < 1000:
            scale = 2.0
            strip = cv2.resize(strip, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        views.append((name, strip))
    framed = cv2.copyMakeBorder(image, 16, 16, 16, 16,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    views.append(("full", framed))

    encoded: list[tuple[str, bytes]] = []
    for name, view in views:
        ok, buf = cv2.imencode(".png", view)
        if not ok:
            # Corrupt pixels are an infrastructure fault, not a missing
            # annotation: fail loudly instead of sending a broken view.
            raise RuntimeError(f"could not encode {name} view of the crop")
        encoded.append((name, buf.tobytes()))
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_view_{name}.png"),
                        view)
    return encoded


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _parse_model_json(text: str) -> dict | None:
    """Brace-matched JSON extraction; tolerant of markdown fences and
    preambles around the object."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                import json
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _call_vision_model(views: list[tuple[str, bytes]], timeout: float) -> str:
    """One OpenAI-compatible vision call with all views attached."""
    import constants
    from openai import OpenAI

    url = getattr(constants, "VISUAL_MODEL_URL", None)
    key = getattr(constants, "VISUAL_MODEL_KEY", None)
    name = getattr(constants, "VISUAL_MODEL_NAME", None)
    if not (url and key and name):
        raise ValueError(
            "seq_id_vision requires VISUAL_MODEL_NAME, VISUAL_MODEL_URL and "
            "VISUAL_MODEL_KEY in constants.py")
    # Hard ceiling shared with the other vision consumers so one hung crop
    # cannot pin a harness worker beyond the pipeline's per-call budget.
    timeout = min(timeout, getattr(
        constants, "VISION_MODEL_HARD_TIMEOUT_CEILING_SECONDS", 120))
    content: list[dict] = [{"type": "text", "text": _PROMPT}]
    for _, png_bytes in views:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(png_bytes)}})
    client = OpenAI(base_url=url, api_key=key, timeout=timeout)
    completion = client.chat.completions.create(
        model=name,
        messages=[
            {"role": "system", "content": "You are a precise information "
             "extraction assistant. Return JSON only."},
            {"role": "user", "content": content},
        ],
        temperature=0, max_tokens=200, timeout=timeout)
    return completion.choices[0].message.content or ""


def _coerce_seq_id(value) -> int | None:
    """Accept only an exact integer (int or all-digit string); anything
    else would be a silently altered reading of the printed text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


@register("seq_id_vision", "model",
          "structure crop -> SEQ ID NO annotation read visually "
          "(found, seq_id, label_text, confidence)")
async def read_seq_id_label(payload: dict) -> dict:
    """One structure crop -> the printed SEQ ID annotation, read visually.

    Payload: {"image_path": "/path/crop.png", "debug_dir": optional dir for
    the sent view images} ("crop_path" accepted as an alias, the figure FSM
    field name). Missing/unreadable input is an {"ok": false, "error"} named
    result; a crop without a visible annotation is {"ok": true, "found":
    false, "reason": "no_seq_id_label_visible"}."""
    image_path = payload.get("image_path") or payload.get("crop_path")
    if not image_path:
        return {"ok": False,
                "error": "image_path (or crop_path) is required"}
    if not os.path.isfile(image_path):
        return {"ok": False, "error": f"image not found: {image_path}"}
    image = cv2.imread(str(image_path))
    if image is None:
        return {"ok": False, "error": f"unreadable image: {image_path}"}

    views = _prepare_views(image, payload.get("debug_dir"),
                           stem=os.path.splitext(os.path.basename(str(image_path)))[0])

    import constants
    timeout = float(getattr(constants, "VISION_MODEL_TIMEOUT_SECONDS", 90))
    raw = _call_vision_model(views, timeout)
    parsed = _parse_model_json(raw)
    if parsed is None:
        return {"ok": True, "found": False, "seq_id": None,
                "label_text": None, "confidence": None,
                "reason": "unparseable_model_response",
                "raw_response": raw[:300]}
    if not parsed.get("found"):
        return {"ok": True, "found": False, "seq_id": None,
                "label_text": None,
                "confidence": parsed.get("confidence"),
                "reason": "no_seq_id_label_visible"}
    seq_id = _coerce_seq_id(parsed.get("seq_id"))
    if seq_id is None:
        # The model claims an annotation but could not produce an integer:
        # reporting a half-read would corrupt the ledger join, so this is a
        # named not-found with the actual text preserved for audit.
        return {"ok": True, "found": False, "seq_id": None,
                "label_text": parsed.get("label_text"),
                "confidence": parsed.get("confidence"),
                "reason": "model_seq_id_not_integer",
                "raw_response": raw[:300]}
    confidence = parsed.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None
    return {"ok": True, "found": True, "seq_id": seq_id,
            "label_text": parsed.get("label_text"),
            "confidence": confidence,
            "views_sent": [name for name, _ in views]}

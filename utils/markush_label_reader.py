"""Offloaded R-group label reader (§44.19, 20.75-pt lever).

The graph decoder reads superscript R-labels poorly (MarkushGrapher-2's
no-OCR ablation collapses label accuracy 56->4); the labels live in small
image regions next to the dummy atoms whose coordinates the decode already
provides. This module crops those regions, composites them into one indexed
grid, and reads every label in a single VLM call — offloading label identity
from the decoder to a dedicated reading pass.

Prototype scope: functions are pure (crop/grid/read/parse) and unit-tested
on synthetic renders; production wiring and bulk validation against the 248
numbering-only failures are follow-ups.
"""
from __future__ import annotations

from typing import Any

_LABEL_CROP_FRACTION = 0.12   # window half-width as a fraction of image min side
_LABEL_GRID_CELL = 160        # px per grid cell after upscale
_LABEL_GRID_COLUMNS = 6


def crop_label_windows(image, points, frac: float = _LABEL_CROP_FRACTION):
    """Crop a square window around each (x, y) point.

    *image* is a BGR ndarray; *points* are normalized [0..1] coordinates as
    produced by the decode's coords head. Returns list of crops (same order).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    if image is None or getattr(image, "size", 0) == 0:
        return []
    height, width = image.shape[:2]
    half = int(max(height, width) * frac) or 8
    crops = []
    for point in points or []:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            crops.append(None)
            continue
        cx, cy = int(x * width), int(y * height)
        x1, x2 = max(0, cx - half), min(width, cx + half)
        y1, y2 = max(0, cy - half), min(height, cy + half)
        crop = image[y1:y2, x1:x2]
        crops.append(crop if crop.size else None)
    return crops


def build_label_grid(crops, cell: int = _LABEL_GRID_CELL, columns: int = _LABEL_GRID_COLUMNS):
    """Composite crops into one white grid with index labels (1-based)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    valid = [(index + 1, crop) for index, crop in enumerate(crops or []) if crop is not None]
    if not valid:
        return None
    rows = (len(valid) + columns - 1) // columns
    grid = np.full((rows * cell, columns * cell, 3), 255, dtype=np.uint8)
    for slot, (number, crop) in enumerate(valid):
        row, col = divmod(slot, columns)
        resized = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_CUBIC)
        grid[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = resized
        cv2.putText(grid, str(number), (col * cell + 4, row * cell + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return grid


def build_label_read_prompt(count: int, known_labels=None) -> str:
    known = ', '.join(known_labels) if known_labels else '(unrestricted)'
    return (
        f"The image is a grid of {count} numbered crops. Each crop shows one "
        "attachment point of a chemical structure with a superscript R-group "
        "label next to it (for example R1, R2, R31, R52). Read the label in "
        "EACH numbered crop. Transcribe exactly the number shown — do not "
        "guess from position or frequency. "
        f"Plausible labels in this document: {known}. "
        'Return JSON {"labels": [{"cell": 1, "label": "R1"}]} for every cell; '
        'use "unknown" when no label is legible.'
    )


def parse_label_read_payload(response_text: str) -> dict:
    import json
    import re
    text = str(response_text or '').strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: first balanced JSON object in the response.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("label read payload has invalid labels array")
    out = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        try:
            cell = int(item.get("cell"))
        except (TypeError, ValueError):
            continue
        label = str(item.get("label") or "").strip()
        out[cell] = label
    return {"labels": out}


def read_labels_for_points(image, points, known_labels=None,
                           vision_read_fn=None, audit_path=None):
    """End-to-end: crops -> grid -> one VLM call -> {dummy_index: label}.

    *vision_read_fn* defaults to the production vision task runner; tests
    inject a stub. Returns {'labels': {index: label}, 'cells': total}.
    """
    crops = crop_label_windows(image, points)
    grid = build_label_grid(crops)
    if grid is None:
        return {"labels": {}, "cells": 0}
    if vision_read_fn is None:
        from utils.llm_utils import run_vision_json_task
        def vision_read_fn(image_file, prompt, audit_path=None, metadata=None):
            return run_vision_json_task(
                task_name="read_markush_labels_grid",
                image_file=image_file,
                prompt=prompt,
                parser=parse_label_read_payload,
                audit_path=audit_path,
                metadata=metadata,
            )
    import os
    import tempfile
    path = tempfile.mktemp(suffix="_label_grid.png")
    try:
        import cv2
        cv2.imwrite(path, grid)
        payload = vision_read_fn(
            path,
            build_label_read_prompt(len(crops), known_labels),
            audit_path=audit_path,
            metadata={"source": "label_offload_reader", "cells": len(crops)},
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    labels = {}
    for cell, label in (payload.get("labels") or {}).items():
        index = int(cell) - 1
        if 0 <= index < len(crops) and label and label.lower() != "unknown":
            labels[index] = label
    return {"labels": labels, "cells": len(crops)}

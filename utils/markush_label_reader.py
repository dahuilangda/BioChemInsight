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

import torch
import torch.nn as nn
import torch.nn.functional as F

_LABEL_CROP_FRACTION = 0.12   # window half-width as a fraction of image max side
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


def dummy_points_from_raw(raw: Any):
    """Dummy-only attachment points in decoded atom order.

    atom_sets[i] mirrors the i-th decoded atom (model.py builds it straight
    from the chartok symbol stream), so filtering on '*' here both removes
    non-dummy proposals and fixes the slot order for SMILES rewriting.
    Returns (points, dummy_atom_indices).
    """
    points, indices = [], []
    for index, atom in enumerate(raw.get("atom_sets") or []):
        if not isinstance(atom, dict):
            continue
        if "*" not in str(atom.get("atom_symbol") or ""):
            continue
        coord = atom.get("coords")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                points.append([float(coord[0]), float(coord[1])])
                indices.append(index)
            except (TypeError, ValueError):
                continue
    return points, indices


def normalize_label_number(label) -> int | None:
    """'R1a' -> 1, 'R12' -> 12, 'unknown' -> None (first digit group)."""
    import re
    digits = re.findall(r"\d+", str(label or ""))
    return int(digits[0]) if digits else None


def rewrite_dummy_isotopes(smiles: str, numbers) -> str:
    """Rewrite the k-th [n*] isotope in *smiles* with numbers[k] in order.

    None entries (unreadable labels) leave that dummy unchanged; fewer
    numbers than dummies leave the tail unchanged.
    """
    import re
    if not smiles:
        return smiles
    ordered = list(numbers or [])

    def _sub(match):
        try:
            number = next(iterator)
        except StopIteration:
            return match.group(0)
        if number is None:
            return match.group(0)
        return f"[{int(number)}*]"

    # Mixed-notation guard: if the SMILES contains any bare [*] (no isotope)
    # alongside [n*] dummies, regex-match order may desync from the dummy
    # order the numbers were built for — leave such smiles untouched rather
    # than silently corrupt labels.
    if re.search(r"(?<!\d)\*", str(smiles)):
        return str(smiles)
    iterator = iter(ordered)
    return re.sub(r"\[\d+\*\]", _sub, str(smiles))


def labels_to_numbers(labels: dict, count: int) -> list:
    """Reader slot map {cell: label} -> per-slot normalized numbers."""
    return [normalize_label_number(labels.get(slot)) for slot in range(count)]


class LabelReaderCNN(nn.Module):
    """Mirror of the training-time architecture (train_label_reader_head)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def read_labels_with_classifier(image, points, checkpoint_path,
                                window: int = 96):
    """Classifier-backed label read: crop windows around each dummy point
    (offset grid, since labels sit up-right of the atom), classify each with
    the cells-bbox-trained CNN, and take the majority confident class.

    Returns {'labels': {slot: number}, 'cells': len(points)}.
    """
    import torch

    import cv2
    import numpy as np

    if image is None or getattr(image, "size", 0) == 0 or not points:
        return {"labels": {}, "cells": len(points or [])}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    class_to_index = {number: index for index, number in enumerate(classes)}
    from utils.markush_label_reader import LabelReaderCNN
    model = LabelReaderCNN(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    height, width = image.shape[:2]
    half = window
    offsets = [(0, 0), (0.30, -0.25), (0.35, 0.05), (0.0, -0.30), (-0.15, -0.15)]
    slot_numbers = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for slot, point in enumerate(points):
        try:
            cx = int(float(point[0]) * width)
            cy = int(float(point[1]) * height)
        except (TypeError, ValueError, IndexError):
            continue
        votes: dict[int, int] = {}
        for ox_frac, oy_frac in offsets:
            ox = int(ox_frac * window)
            oy = int(oy_frac * window)
            x1 = max(0, min(width - 2, cx + ox - half))
            y1 = max(0, min(height - 2, cy + oy - half))
            x2 = max(x1 + 2, min(width, x1 + 2 * half))
            y2 = max(y1 + 2, min(height, y1 + 2 * half))
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                probabilities = F.softmax(model(tensor), dim=1)[0]
            confidence_value, class_index = probabilities.max(0)
            if float(confidence_value) >= 0.6:
                votes[class_index.item()] = votes.get(class_index.item(), 0) + 1
        if not votes:
            continue
        best_class, count = max(votes.items(), key=lambda kv: kv[1])
        if count >= 2:
            slot_numbers[slot] = classes[best_class]
    return {"labels": slot_numbers, "cells": len(points)}


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
    handle, path = tempfile.mkstemp(suffix="_label_grid.png")
    os.close(handle)
    try:
        import cv2
        if not cv2.imwrite(path, grid):
            return {"labels": {}, "cells": len(crops)}
        try:
            payload = vision_read_fn(
                path,
                build_label_read_prompt(sum(1 for c in crops if c is not None),
                                        known_labels),
                audit_path=audit_path,
                metadata={"source": "label_offload_reader", "cells": len(crops)},
            )
        except Exception:
            # One malformed response must not kill the caller: degrade to no
            # labels read, identical to every cell coming back 'unknown'.
            return {"labels": {}, "cells": len(crops)}
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

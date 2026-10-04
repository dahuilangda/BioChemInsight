"""Grid-merged compound-ID extraction (harness roadmap: VLM batch merge).

Composites N structure crops into one indexed grid and reads every compound
ID in a single VLM call, replacing N per-structure calls. Falls back to the
caller's per-structure path on any failure. Accuracy contract: the grid
prompt carries the same schema rules as structure_to_id; validation compares
both paths on real pages.
"""
from __future__ import annotations

import cv2
import numpy as np

GRID_CELL = 200
GRID_COLUMNS = 4


def build_id_grid(images):
    """Composite BGR or grayscale crops into one indexed grid.

    Aspect-preserving (white-padded, no squash) with a red border strip as
    the cell marker (MolGrapher-style; overlay text can collide with the
    molecule or the identifier being read). Caps at 16 tiles per grid.
    """
    tiles = []
    for index, image in enumerate(images[:16]):
        if image is None or getattr(image, "size", 0) == 0:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cell = GRID_CELL
        border = 6
        inner = cell - 2 * border
        h, w = image.shape[:2]
        scale = min(inner / max(1, h), inner / max(1, w))
        resized = cv2.resize(image, (max(1, int(w * scale)),
                                     max(1, int(h * scale))),
                             interpolation=cv2.INTER_CUBIC)
        tile = np.full((cell, cell, 3), 255, dtype=np.uint8)
        oh, ow = resized.shape[:2]
        y0, x0 = border + (inner - oh) // 2, border + (inner - ow) // 2
        tile[y0:y0 + oh, x0:x0 + ow] = resized
        tile[:border, :] = (0, 0, 255)
        cv2.putText(tile, str(index + 1), (4, cell - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    rows = (len(tiles) + GRID_COLUMNS - 1) // GRID_COLUMNS
    grid = np.full((rows * GRID_CELL, GRID_COLUMNS * GRID_CELL, 3), 255,
                   dtype=np.uint8)
    for slot, tile in enumerate(tiles):
        row, col = divmod(slot, GRID_COLUMNS)
        grid[row * GRID_CELL:(row + 1) * GRID_CELL,
             col * GRID_CELL:(col + 1) * GRID_CELL] = tile
    return grid


def build_id_grid_prompt(count: int):
    return (
        f"The image is a grid of {count} numbered crops of chemical structures "
        "from a patent. For EACH numbered crop, read the compound identifier "
        "attached to the red-boxed molecule (for example 1, 2, 15, Ia, Ex-3). "
        "Reading order: table row labels and headings near the structure win "
        "over unrelated page text. "
        'Return JSON {"ids": [{"cell": 1, "compound_id": "12", "confidence": '
        '"high"}]} for EVERY cell; use compound_id "None" when the crop shows '
        "no identifier."
    )


def parse_id_grid_payload(response_text: str) -> dict:
    import json
    import re
    text = str(response_text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    ids = payload.get("ids")
    if not isinstance(ids, list):
        raise ValueError("id grid payload has invalid ids array")
    out = {}
    confidences = {}
    for item in ids:
        if not isinstance(item, dict):
            continue
        try:
            cell = int(item.get("cell"))
        except (TypeError, ValueError):
            continue
        out[cell] = str(item.get("compound_id") or "").strip()
        try:
            confidences[cell] = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidences[cell] = None
    return {"ids": out, "confidences": confidences}

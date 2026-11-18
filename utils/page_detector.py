"""Whole-page chemical structure detection."""
from __future__ import annotations

import os
from pathlib import Path

_MODEL = None
_WEIGHTS = Path('models/moldetv2/moldet_v2_yolo11n_960_doc.pt')
_CONF = 0.25
_IMGSZ = 960


def _weights_path() -> Path:
    configured = os.getenv('MOLDET_WEIGHTS')
    if configured:
        return Path(configured).expanduser()
    local = Path(__file__).resolve().parents[1] / _WEIGHTS
    if local.is_file():
        return local
    mounted = Path('/data/BioChemInsight') / _WEIGHTS
    return mounted if mounted.is_file() else local


def detect_structures_on_page(page_bgr, conf: float = _CONF) -> list:
    """Detect chemical structures on a page. Returns [(x0, y0, x1, y1, score)]."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(_weights_path())
    results = _MODEL.predict(page_bgr, imgsz=_IMGSZ, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            score = box.conf[0].item()
            h, w = page_bgr.shape[:2]
            x0, y0 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x0 > 10 and y2 - y0 > 10:
                boxes.append((x0, y0, x2, y2, score))
    return boxes

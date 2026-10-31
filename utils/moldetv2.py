"""MolDetv2-Doc-n (YOLO11n @ 960) whole-page chemical structure detector.

UniParser MolDetv2: mAP50 0.9936 / mAP50:95 0.9544 on MolDet-Bench-Doc
(447 patent/document PDF pages), 3.9ms whole-page single pass on a 4070
Ti SUPER. Trained on 60,000 human-annotated patent/paper/book PDF pages.
Replaces the Mask R-CNN + tile + mask-morphology chain that fragmented
large macrocycles into overlapping partial boxes.
"""
from __future__ import annotations

_MODEL = None
_WEIGHTS = '/data/BioChemInsight/models/moldetv2/moldet_v2_yolo11n_960_doc.pt'
_CONF = 0.25
_IMGSZ = 960


def moldetv2_detect(page_bgr, conf: float = _CONF) -> list:
    """Detect chemical structures on a page. Returns [(x0, y0, x1, y1, score)]."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(_WEIGHTS)
    results = _MODEL.predict(page_bgr, imgsz=_IMGSZ, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            score = box.conf[0].item()
            # Clip to image bounds
            h, w = page_bgr.shape[:2]
            x0, y0 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x0 > 10 and y2 - y0 > 10:
                boxes.append((x0, y0, x2, y2, score))
    return boxes

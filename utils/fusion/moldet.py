"""MolDetv2 (YOLO11) detector replacing the fused detection head in cascade.

Uni-Parser's
official combo IS "YOLO11 detection + independent recognizer", Apache-2.0,
mAP50 0.9872 @1.5ms on real documents. Boxes + adaptive expansion feed the
existing cascade recognition (production crop transform w/ 512 pre-upscale).
"""
from __future__ import annotations

import numpy as np
import torch


class MolDetDetector:
    def __init__(self, weights_path: str, device: str = "cuda",
                 conf: float = 0.25, iou: float = 0.45):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def detect(self, page_bgr: np.ndarray, with_scores: bool = False):
        """Return normalized xyxy boxes (N, 4), best-score first.

        with_scores=True also returns the YOLO confidences (N,) — the
        detection->gate channel that the short-circuit and risk scoring
        consume.
        """
        result = self.model.predict(page_bgr, conf=self.conf, iou=self.iou,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return (empty, np.zeros(0, dtype=np.float32)) if with_scores else empty
        h, w = page_bgr.shape[:2]
        xyxy = result.boxes.xyxy.cpu().numpy()
        boxes = np.stack([
            xyxy[:, 0] / w, xyxy[:, 1] / h,
            xyxy[:, 2] / w, xyxy[:, 3] / h,
        ], axis=1).astype(np.float32)
        if with_scores:
            return boxes, result.boxes.conf.cpu().numpy().astype(np.float32)
        return boxes


def expand_boxes(boxes: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    """Adaptive box expansion (MolDet-Bench practice) to avoid truncation."""
    if boxes.shape[0] == 0:
        return boxes
    w = (boxes[:, 2] - boxes[:, 0]) * ratio
    h = (boxes[:, 3] - boxes[:, 1]) * ratio
    out = boxes.copy()
    out[:, 0] -= w; out[:, 1] -= h
    out[:, 2] += w; out[:, 3] += h
    return np.clip(out, 0.0, 1.0)

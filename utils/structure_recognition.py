"""Unified structure detection and recognition entry points.

DECIMER-style Mask R-CNN segmentation and MolNexTR graph decoding solve
different stages of the structure extraction pipeline. This module keeps them
behind one small API so callers do not each reimplement model lookup, loading,
locking, and result normalization.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:  # optional runtime configuration
    import constants as project_constants
except ImportError:  # pragma: no cover
    project_constants = None

from utils.molecule_segmentation import apply_masks, get_expanded_masks


MOLNEXTR_MODEL_FILE = "molnextr_best.pth"
MOLNEXTR_POSTPROCESS_WORKERS = max(
    1, int(getattr(project_constants, "MOLNEXTR_POSTPROCESS_WORKERS", 1) or 1)
)
MOLNEXTR_PREPROCESS_LONG_EDGE = max(
    0, int(getattr(project_constants, "MOLNEXTR_PREPROCESS_LONG_EDGE", 512) or 0)
)
_molnextr_lock = threading.Lock()
_segmentation_lock = threading.Lock()
_molnextr_model = None
_molnextr_model_path: str | None = None
_molnextr_device: torch.device | None = None


@dataclass(frozen=True)
class DetectedStructure:
    image: np.ndarray
    bbox: list[int]
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class StructureDetectionResult:
    structures: list[DetectedStructure]
    masks: np.ndarray


@dataclass(frozen=True)
class StructurePrediction:
    smiles: str
    molblock: str = ""
    raw: Any = None
    elapsed_seconds: float = 0.0


def normalize_segment_array(segment: np.ndarray | None) -> np.ndarray | None:
    if not isinstance(segment, np.ndarray) or len(segment.shape) != 3:
        return None
    if segment.shape[2] == 4:
        segment = segment[:, :, :3]
    elif segment.shape[2] != 3:
        return None
    if segment.dtype != np.uint8:
        if segment.max() <= 1.0:
            segment = (segment * 255).astype(np.uint8)
        else:
            segment = segment.astype(np.uint8)
    return segment


def extract_molblock(prediction: Any) -> str:
    if not isinstance(prediction, dict):
        return ""
    for key in ("predicted_molfile", "molfile", "molblock", "molfile_v3", "molfileV3"):
        value = prediction.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def sort_segments_bboxes(segments, bboxes, masks, same_row_pixel_threshold=50):
    if len(bboxes) == 0:
        return segments, bboxes, masks

    bbox_with_indices = [(bbox, idx) for idx, bbox in enumerate(bboxes)]
    sorted_bbox_with_indices = sorted(bbox_with_indices, key=lambda item: item[0][0])

    rows = []
    current_row = [sorted_bbox_with_indices[0]]
    for bbox_with_idx in sorted_bbox_with_indices[1:]:
        if abs(bbox_with_idx[0][0] - current_row[-1][0][0]) < same_row_pixel_threshold:
            current_row.append(bbox_with_idx)
        else:
            rows.append(sorted(current_row, key=lambda item: item[0][1]))
            current_row = [bbox_with_idx]
    rows.append(sorted(current_row, key=lambda item: item[0][1]))

    sorted_indices = [bbox_with_idx[1] for row in rows for bbox_with_idx in row]
    sorted_segments = [segments[idx] for idx in sorted_indices]
    sorted_bboxes = [bboxes[idx] for idx in sorted_indices]
    sorted_masks = np.stack([masks[:, :, idx] for idx in sorted_indices], axis=-1)
    return sorted_segments, sorted_bboxes, sorted_masks


def resolve_molnextr_model_path() -> str:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path("/app/models") / MOLNEXTR_MODEL_FILE,
        root / "models" / MOLNEXTR_MODEL_FILE,
        Path.cwd() / "models" / MOLNEXTR_MODEL_FILE,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"{MOLNEXTR_MODEL_FILE} not found. Build/download the model into /app/models or ./models."
    )


def _load_molnextr(model_path: str | None = None, device: torch.device | None = None):
    global _molnextr_model, _molnextr_model_path, _molnextr_device

    resolved_path = model_path or resolve_molnextr_model_path()
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _molnextr_lock:
        if (
            _molnextr_model is not None
            and _molnextr_model_path == resolved_path
            and _molnextr_device == resolved_device
        ):
            return _molnextr_model

        from utils.MolNexTR import molnextr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading MolNexTR model from: {resolved_path}")
        _molnextr_model = molnextr(
            resolved_path,
            resolved_device,
            postprocess_workers=MOLNEXTR_POSTPROCESS_WORKERS,
            preprocess_long_edge=MOLNEXTR_PREPROCESS_LONG_EDGE,
        )
        _molnextr_model_path = resolved_path
        _molnextr_device = resolved_device
        return _molnextr_model


class StructureRecognizer:
    def __init__(
        self,
        model_path: str | None = None,
        device: torch.device | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_molnextr(self.model_path, self.device)
        return self._model

    def detect_segments(self, page_bgr: np.ndarray) -> StructureDetectionResult:
        with _segmentation_lock:
            masks = get_expanded_masks(page_bgr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        segments, bboxes = apply_masks(page_bgr, masks)
        if len(segments) > 0:
            segments, bboxes, masks = sort_segments_bboxes(segments, bboxes, masks)

        detected = []
        for idx, segment in enumerate(segments):
            normalized = normalize_segment_array(segment)
            if normalized is None:
                continue
            mask = masks[:, :, idx] if masks is not None and masks.shape[2] > idx else None
            detected.append(
                DetectedStructure(
                    image=normalized,
                    bbox=np.asarray(bboxes[idx]).astype(int).tolist(),
                    mask=mask,
                )
            )
        return StructureDetectionResult(structures=detected, masks=masks)

    def predict_segment_file(self, segment_file: str) -> StructurePrediction:
        return self._predict_molnextr(segment_file)

    def _predict_molnextr(self, segment_file: str) -> StructurePrediction:
        start = time.monotonic()
        result = self.model.predict_final_results(
            segment_file,
            return_atoms_bonds=True,
            return_confidence=True,
        ) or {}
        elapsed = time.monotonic() - start
        if isinstance(result, dict):
            smiles = result.get("predicted_smiles") or ""
            molblock = extract_molblock(result)
        else:
            smiles = result or ""
            molblock = ""
        return StructurePrediction(
            smiles=smiles,
            molblock=molblock,
            raw=result,
            elapsed_seconds=elapsed,
        )

    def health_check(self) -> dict[str, Any]:
        path = resolve_molnextr_model_path()
        payload: dict[str, Any] = {"model_path": path}
        model = _load_molnextr(path, self.device)
        payload["device"] = str(model.device)
        payload["loaded"] = True
        return payload

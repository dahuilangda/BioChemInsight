"""Page-level dataset adapter for fusion training (MG2 page_image + refs).

Stage-P1 data source: ``data/raw/markushgrapher2/uspto-mol-m-54k/*.parquet``
rows carry ``page_image`` (1024×1024), ``cells`` (OCR boxes) and the
markush ``annotation``/``cxsmiles``. Each page holds exactly one structure,
which bootstraps the detection head (box ≈ full page) and lets the existing
recognition refs be reused verbatim — the fusion model sees pages, the
labels stay those of ``build_real_markushgrapher_ocsr.py``.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_GLOB = "uspto-mol-m-54k/*.parquet"


def _decode_image(payload: Any) -> np.ndarray:
    import cv2

    if isinstance(payload, bytes):
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    else:
        image = np.asarray(payload)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
    if image is None or image.size == 0:
        raise ValueError("unreadable page_image payload")
    return image


@dataclass
class PageSample:
    image: np.ndarray          # (H, W, 3) uint8
    boxes: np.ndarray          # (K, 4) normalized xyxy
    page_size: tuple[int, int]

    def tensors(self, size: int = 384) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2

        resized = cv2.resize(self.image, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.from_numpy(self.boxes.astype(np.float32))


class MG2PageDataset(Dataset):
    """Iterate MG2 parquet rows as page-level (image, full-page-box) samples.

    The recognition targets are NOT rebuilt here — Stage-B plugs the existing
    refs builder (labels from the v3 markush shards via ``source_row``).
    This adapter only handles the page/box side that the two-stage pipeline
    never had.
    """

    def __init__(self, raw_root: str | Path, subset_glob: str = DEFAULT_GLOB,
                 limit: int = 0, full_page_box: bool = True) -> None:
        import pyarrow.parquet as pq

        self.samples: list[dict[str, Any]] = []
        paths = sorted(Path(raw_root).glob(subset_glob))
        if not paths:
            raise FileNotFoundError(f"no parquet under {raw_root}/{subset_glob}")
        taken = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            wanted = [n for n in ("page_image", "cxsmiles", "page_image_path", "cells")
                      if n in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(columns=wanted, batch_size=64):
                for row in batch.to_pylist():
                    try:
                        image = _decode_image(row.get("page_image"))
                    except Exception:
                        continue
                    height, width = image.shape[:2]
                    boxes = np.array([[0.02, 0.02, 0.98, 0.98]], dtype=np.float32) \
                        if full_page_box else self._cells_to_boxes(row.get("cells"))
                    self.samples.append({
                        "image": image,
                        "boxes": boxes,
                        "page_size": (height, width),
                        "cxsmiles": str(row.get("cxsmiles") or ""),
                        "source_parquet": str(path),
                    })
                    taken += 1
                    if limit and taken >= limit:
                        return

    @staticmethod
    def _cells_to_boxes(cells: Any) -> np.ndarray:
        try:
            parsed = json.loads(cells) if isinstance(cells, str) else (cells or [])
        except json.JSONDecodeError:
            parsed = []
        boxes = [[0.02, 0.02, 0.98, 0.98]]
        for cell in parsed[:0]:  # placeholder: OCR cells are labels, not structures
            break
        return np.array(boxes, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "image": sample["image"],
            "boxes": sample["boxes"],
            "page_size": sample["page_size"],
            "cxsmiles": sample["cxsmiles"],
        }

    def iterate_tensors(self, size: int = 384) -> Iterator[PageSample]:
        for sample in self.samples:
            yield PageSample(sample["image"], sample["boxes"], tuple(sample["page_size"]))

"""Page-level dataset adapter for fusion training (MG2 page_image + refs).

v2 (post-audit): MG2 ``page_image`` is a pyarrow struct ``{'bytes','path'}``
and must be decoded through ``payload['bytes']`` — the naive ``np.asarray``
branch silently dropped **every** row (audit finding). Decoding is now
lazy (row-pointer based): 54k pages × ~3MB never all live in memory.
``test-*.parquet`` shards are excluded to avoid leaking the official eval
split into training.
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

    raw: bytes | None = None
    if isinstance(payload, dict):
        raw = payload.get("bytes")
    elif isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    elif isinstance(payload, np.ndarray):
        raw = payload.tobytes() if payload.dtype == np.uint8 else None
    if not raw:
        raise ValueError("page_image payload carries no bytes")
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        # payloads may be stored via PIL (e.g. png-encoded differently)
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as handle:
            image = np.asarray(handle.convert("RGB"))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


@dataclass
class PageSample:
    image: np.ndarray
    boxes: np.ndarray
    page_size: tuple[int, int]

    def tensors(self, size: int = 384) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2

        resized = cv2.resize(self.image, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.from_numpy(self.boxes.astype(np.float32))


class MG2PageDataset(Dataset):
    """Lazy page-level view over MG2 parquets.

    ``__getitem__`` decodes one page on demand; the index only stores
    (parquet_path, row_offset, cxsmiles) tuples plus a full-page box.
    """

    def __init__(self, raw_root: str | Path, subset_glob: str = DEFAULT_GLOB,
                 limit: int = 0) -> None:
        import pyarrow.parquet as pq

        self.index: list[dict[str, Any]] = []
        self.decode_failures = 0
        self._row_cache: dict[tuple[str, int], bytes | None] = {}
        paths = sorted(Path(raw_root).glob(subset_glob))
        paths = [p for p in paths if "test-" not in p.name]  # keep eval split out
        if not paths:
            raise FileNotFoundError(f"no train parquet under {raw_root}/{subset_glob}")
        taken = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            wanted = [n for n in ("page_image", "cxsmiles", "page_image_path")
                      if n in parquet.schema_arrow.names]
            offset = 0
            for batch in parquet.iter_batches(columns=wanted, batch_size=64):
                for row in batch.to_pylist():
                    self.index.append({
                        "parquet": str(path),
                        "row": offset,
                        "cxsmiles": str(row.get("cxsmiles") or ""),
                    })
                    offset += 1
                    taken += 1
                    if limit and taken >= limit:
                        return
        self._row_cache: dict[tuple[str, int], bytes | None] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _payload(self, parquet: str, row: int) -> bytes | None:
        key = (parquet, row)
        if key in self._row_cache:
            return self._row_cache[key]
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["page_image"])
        cell = table.column("page_image")[row].as_py()
        payload = cell.get("bytes") if isinstance(cell, dict) else cell
        payload = bytes(payload) if payload is not None else None
        # cache only a bounded window to keep memory flat
        if len(self._row_cache) > 64:
            self._row_cache.clear()
        self._row_cache[key] = payload
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.index[index]
        payload = self._payload(meta["parquet"], meta["row"])
        try:
            image = _decode_image(payload)
        except Exception:
            self.decode_failures += 1
            raise
        height, width = image.shape[:2]
        return {
            "image": image,
            "boxes": np.array([[0.02, 0.02, 0.98, 0.98]], dtype=np.float32),
            "page_size": (height, width),
            "cxsmiles": meta["cxsmiles"],
        }

    def iterate_tensors(self, size: int = 384) -> Iterator[PageSample]:
        for index in range(len(self)):
            item = self[index]
            yield PageSample(item["image"], item["boxes"], item["page_size"])


def cells_placeholder(cells: Any) -> np.ndarray:
    """OCR cells are labels, not structure boxes — raising beats pretending."""
    raise NotImplementedError(
        "deriving structure boxes from OCR cells is not implemented; use "
        "full-page boxes for Stage P1 or provide annotated boxes"
    )

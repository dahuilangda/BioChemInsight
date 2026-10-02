"""MolMole Patent300: real patent pages with structure detection annotations.

300 pages, 2,482 structure bboxes (category_id 1 = molecule; category 2 =
labels/text we exclude). This is the real-page detection supervision the
fusion head needs to close the synthetic-to-real domain gap (design §28).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORY_STRUCTURE = 1


class MolMolePageDataset(Dataset):
    """Pages + normalized xyxy GT boxes for the structure category.

    Images download lazily from the HF mirror on first access (row-pointer
    style, same pattern as MG2PageDataset); annotations are local.
    """

    def __init__(self, annotation_path: str | Path, image_root: str | Path,
                 limit: int = 0, train_fraction: float = 0.85,
                 split: str = "train", seed: int = 13) -> None:
        payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        images = payload["images"]
        # deterministic split by hashed filename (stable across runs)
        import random

        rng = random.Random(seed)
        names = sorted(im["file_name"] for im in images)
        shuffled = names[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_fraction)
        keep = set(shuffled[:cut] if split == "train" else shuffled[cut:])
        self.image_root = Path(image_root)
        self.entries: list[dict[str, Any]] = []
        for im in images:
            if im["file_name"] not in keep:
                continue
            boxes = [[b["bbox"][0] / im["width"], b["bbox"][1] / im["height"],
                      (b["bbox"][0] + b["bbox"][2]) / im["width"],
                      (b["bbox"][1] + b["bbox"][3]) / im["height"]]
                     for b in im.get("dla_bboxes", [])
                     if b.get("category_id") == CATEGORY_STRUCTURE]
            if not boxes:
                continue
            self.entries.append({
                "file": im["file_name"],
                "width": im["width"], "height": im["height"],
                "boxes": np.array(boxes, dtype=np.float32).clip(0.0, 1.0),
            })
            if limit and len(self.entries) >= limit:
                break
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _image(self, name: str) -> np.ndarray:
        import cv2

        if name in self._cache:
            return self._cache[name]
        local = self.image_root / name
        if not local.exists():
            from huggingface_hub import hf_hub_download

            import os as _os
            _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            remote = hf_hub_download(
                "doxa-friend/MolMole_Patent300",
                f"images_300/{name}",
                repo_type="dataset",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(Path(remote).read_bytes())
        img = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[name] = img
        return img

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return {
            "image": self._image(entry["file"]),
            "boxes": entry["boxes"],
            "file": entry["file"],
        }

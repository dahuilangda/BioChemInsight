"""Synthetic multi-structure pages for fusion Stage P2.

Stage A's honest boundary (design §13) was single-structure pages with
box ~= full page. This generator pastes K real structure crops (from the
MolScribe uspto_mol TIF pool — real patent ink) onto a white canvas in a
reading-order grid, with margins, optional ruling lines and text-noise
strips, and records per-structure GT boxes + SMILES. That gives the
multi-instance detection/regression signal P2 needs without any new
annotation budget.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_crop(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable crop {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image[..., ::-1]  # BGR -> RGB


@dataclass
class SynthPageConfig:
    canvas: int = 1024
    min_structures: int = 2
    max_structures: int = 6
    margin: int = 24
    gap: int = 40
    ruling_line_prob: float = 0.3
    text_noise_prob: float = 0.4
    seed: int = 0
    scale_jitter: tuple[float, float] = (0.7, 1.0)


class SyntheticMultiStructureDataset(Dataset):
    """Pages with K pasted structures; GT boxes + per-structure SMILES.

    ``index_path``: the MolScribe uspto_mol CSV (file_path, SMILES).
    ``image_root``: directory that CSV's file_path values are relative to.
    """

    def __init__(self, index_path: str | Path, image_root: str | Path,
                 pages: int = 20000, config: SynthPageConfig | None = None) -> None:
        import csv as csv_mod
        import os

        self.cfg = config or SynthPageConfig()
        self.image_root = Path(image_root)
        self.pages = pages
        self.rng = random.Random(self.cfg.seed)

        csv_mod.field_size_limit(10 ** 9)
        pool: list[tuple[str, str]] = []
        with open(index_path, newline="", encoding="utf-8") as handle:
            for row in csv_mod.DictReader(handle):
                smiles = (row.get("SMILES") or "").strip()
                rel = (row.get("file_path") or "").strip()
                if smiles and rel and "*" not in smiles and "[R" not in smiles:
                    if os.path.exists(self.image_root / rel):
                        pool.append((rel, smiles))
        self.pool = pool
        if not self.pool:
            raise ValueError("empty crop pool")
        # deterministic per-page RNG streams (worker-safe, replayable)
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return self.pages

    def _render(self, index: int) -> dict[str, Any]:
        import cv2

        rng = random.Random(self.cfg.seed + index)
        cfg = self.cfg
        canvas = np.full((cfg.canvas, cfg.canvas, 3), 255, dtype=np.uint8)

        k = rng.randint(cfg.min_structures, cfg.max_structures)
        cols = 2 if k <= 4 else 3
        rows = (k + cols - 1) // cols
        cell_w = (cfg.canvas - 2 * cfg.margin - (cols - 1) * cfg.gap) // cols
        cell_h = (cfg.canvas - 2 * cfg.margin - (rows - 1) * cfg.gap) // rows

        picks = rng.sample(range(len(self.pool)), k)
        boxes: list[list[float]] = []
        smiles_list: list[str] = []
        placed = 0
        for row in range(rows):
            for col in range(cols):
                if placed >= k:
                    break
                rel, smiles = self.pool[picks[placed]]
                try:
                    crop = _load_crop(str(self.image_root / rel))
                except Exception:
                    try:
                        rel2 = rel.replace("uspto_mol/", "", 1)
                        crop = _load_crop(str(self.image_root / rel2))
                    except Exception:
                        k -= 1
                        continue
                scale = rng.uniform(*cfg.scale_jitter)
                h, w = crop.shape[:2]
                target_w = min(int(cell_w * scale), cell_w)
                target_h = int(h * target_w / max(1, w))
                if target_h > cell_h:
                    target_h = cell_h
                    target_w = int(w * target_h / max(1, h))
                if target_w < 8 or target_h < 8:
                    placed += 1
                    continue
                crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
                x0 = cfg.margin + col * (cell_w + cfg.gap) + rng.randint(0, max(0, cell_w - target_w))
                y0 = cfg.margin + row * (cell_h + cfg.gap) + rng.randint(0, max(0, cell_h - target_h))
                canvas[y0: y0 + target_h, x0: x0 + target_w] = crop
                boxes.append([
                    x0 / cfg.canvas, y0 / cfg.canvas,
                    (x0 + target_w) / cfg.canvas, (y0 + target_h) / cfg.canvas,
                ])
                smiles_list.append(smiles)
                placed += 1

        if rng.random() < cfg.ruling_line_prob:  # horizontal separator
            y = rng.randint(cfg.margin, cfg.canvas - cfg.margin)
            cv2.line(canvas, (cfg.margin, y), (cfg.canvas - cfg.margin, y),
                     (120, 120, 120), thickness=2)
        if rng.random() < cfg.text_noise_prob:  # fake caption strip
            y = rng.randint(cfg.margin, cfg.canvas - 60)
            n_words = rng.randint(3, 9)
            x = cfg.margin
            for _ in range(n_words):
                wl = rng.randint(40, 120)
                cv2.putText(canvas, "Example " + str(rng.randint(1, 99)), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
                x += wl + rng.randint(6, 20)

        return {
            "image": canvas,
            "boxes": np.array(boxes or [[0.02, 0.02, 0.98, 0.98]], dtype=np.float32),
            "smiles": smiles_list,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[index] = self._render(index)
        return self._cache[index]

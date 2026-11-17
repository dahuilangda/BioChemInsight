"""VL block reader: PaddleOCR-VL-1.6 per-compound-block table reading.

Why this shape: full-page VL reads hit decoder repetition loops in
SMILES cells (arginine/aspartate motifs), and the PP-OCR-only chain is
brittle on ultra-tight sheet layouts (cross-block attr association).
Reading one compound block at a time fixes both: the block's attr,
notation and SMILES travel together (no cross-block association), the
output is short (loops never ignite), and `no_repeat_ngram_size` stops
residual loops at the cost of a truncated SMILES tail — which the mass
contract rejects anyway.

The block geometry comes from the PP-OCR line boxes' printed-id rhythm
(cod runs in the leftmost column), not from any layout template.
"""
from __future__ import annotations

import json
import re
import statistics
import threading

_SINGLE_FLIGHT = threading.BoundedSemaphore(1)

_FCEL = re.compile(r"<fcel>|<ecel>|<lcel>|<ucel>|<cel>")
_NL = re.compile(r"<nl>")


def block_bands(lines: list[dict], col0_max_x: float = 560,
                gap: float = 120, page_height: float | None = None
                ) -> list[tuple[float, float, str]]:
    """Compound-block bands from the printed-id rhythm: pure-digit cells
    in the leftmost column, runs of the same id grouped by vertical
    proximity. Each band extends to the next band's top (a block's rows
    can outrun its own id cells); the last band runs to the page
    bottom. Returns (y_top, y_bottom, cid) in reading order."""
    cids = []
    for line in lines:
        text = line["text"].strip()
        if line["box"][0] < col0_max_x and text.isdigit() and len(text) <= 4:
            cids.append((line["box"][1], line["box"][3], text))
    cids.sort()
    runs: list[list] = []
    for y1, y2, cid in cids:
        if runs and runs[-1][2] == cid and y1 - runs[-1][1] < gap:
            runs[-1][1] = y2
        else:
            runs.append([y1, y2, cid])
    bands = []
    for i, (y1, _y2, cid) in enumerate(runs):
        bottom = (runs[i + 1][0] - 10 if i + 1 < len(runs)
                  else (page_height or _y2 + 400))
        bands.append((y1, bottom, cid))
    return bands


def parse_vl_table(text: str) -> list[list[str]]:
    """VL cell stream (<fcel>/<ecel>/<lcel>...<nl>) -> rows of cells.
    <lcel>/<ucel> marks are span continuations: they contribute no
    text."""
    rows: list[list[str]] = []
    for raw in _NL.split(text):
        cells = _FCEL.split(raw)
        row = [c.strip() for c in cells if c and c.strip()]
        if row:
            rows.append(row)
    return rows


class VLBlockReader:
    """Lazy-loading single-flight reader over one GPU."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 no_repeat_ngram_size: int = 20):
        self.model_path = model_path
        self.device = device
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForImageTextToText,
                                  AutoProcessor)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype=torch.bfloat16).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def read_block(self, sheet: "PIL.Image.Image",
                   band: tuple[float, float, str],
                   pad_top: float = 30, pad_bottom: float = 40) -> list[list[str]]:
        """One compound band -> grid rows (cells, no tags)."""
        import torch
        self._load()
        y1, y2, _cid = band
        w, h = sheet.size
        crop = sheet.crop((0, max(0, int(y1) - pad_top), w,
                           min(h, int(y2) + pad_bottom)))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": "Table Recognition:"},
        ]}]
        with _SINGLE_FLIGHT:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                images_kwargs={"size": {"shortest_edge": 112896,
                                        "longest_edge": 1280 * 28 * 28}},
            ).to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False,
                    no_repeat_ngram_size=self.no_repeat_ngram_size)
            text = self._processor.decode(
                out[0][inputs["input_ids"].shape[-1]:-1])
        return parse_vl_table(text)

    def read_sheet(self, sheet: "PIL.Image.Image",
                   lines: list[dict]) -> dict[str, list[list[str]]]:
        """cid-rhythm bands -> {cid: grid rows}."""
        bands = block_bands(lines, page_height=sheet.size[1])
        out: dict[str, list[list[str]]] = {}
        for band in bands:
            try:
                out[band[2]] = self.read_block(sheet, band)
            except Exception as exc:
                out[band[2]] = [[f"__error__ {type(exc).__name__}: "
                                 f"{str(exc)[:80]}"]]
        return out

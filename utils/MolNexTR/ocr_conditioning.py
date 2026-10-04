"""OCR conditioning encoder (§44.49): minimal viable MG2-style joint stream.

Projects ChemicalOCR cells (text+bbox) into the MoE decoder's d_model space,
producing a variable-length memory sequence that is prepended to the Swin
encoder features before autoregressive decoding. The decoder's existing
cross-attention handles the concatenated memory without architecture change.

Design (per MG2 engineering dossier §44.36):
  - word-level embedding (simple whitespace tokenization, vocab 512)
  - learned 2D bbox embedding (64 bins per coordinate, x/y summed)
  - linear projection to d_model
  - zero-initialised output gate (starts as no-op; training learns to open)

Usage:
    encoder = OcrConditioningEncoder(d_model=256)
    encoder.train()  # only this module's parameters are optimised
    memory = encoder(cells, d_model)  # (1, seq_len, d_model)
    # prepend to encoder features before decode:
    features = torch.cat([memory, features], dim=1)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OcrConditioningEncoder(nn.Module):
    def __init__(self, d_model: int = 256, vocab_size: int = 512,
                 bbox_bins: int = 64, max_tokens: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.bbox_x0 = nn.Embedding(bbox_bins, d_model)
        self.bbox_y0 = nn.Embedding(bbox_bins, d_model)
        self.bbox_x1 = nn.Embedding(bbox_bins, d_model)
        self.bbox_y1 = nn.Embedding(bbox_bins, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Zero-initialised gate: training starts from no-op and learns to open.
        self.gate = nn.Parameter(torch.zeros(1))
        self._bbox_bins = bbox_bins

    def _tokenize_word(self, word: str, vocab_size: int) -> list[int]:
        """Deterministic word→token list (char-hash based, no external tokenizer)."""
        tokens = []
        for char in word.lower():
            tokens.append(hash(char) % vocab_size)
        if not tokens:
            tokens = [0]
        return tokens

    def _bbox_to_bins(self, bbox: list[float]) -> list[int]:
        return [min(self._bbox_bins - 1, max(0, int(v * self._bbox_bins)))
                for v in bbox[:4]]

    def forward(self, cells: list[dict]) -> torch.Tensor:
        """cells: [{bbox: [x0,y0,x1,y1] normalised, text: str}] → (1, seq, d_model)."""
        embeddings = []
        for cell in cells or []:
            text = str(cell.get("text") or "").strip()
            bbox = cell.get("bbox")
            if not text or not bbox or len(bbox) < 4:
                continue
            bins = self._bbox_to_bins(bbox)
            bbox_emb = (self.bbox_x0(torch.tensor(bins[0])) + self.bbox_y0(torch.tensor(bins[1]))
                        + self.bbox_x1(torch.tensor(bins[2])) + self.bbox_y1(torch.tensor(bins[3])))
            for char_token in self._tokenize_word(text, 512):
                emb = self.word_embedding(torch.tensor(char_token)) + bbox_emb
                embeddings.append(self.proj(emb.unsqueeze(0)).squeeze(0))
        if not embeddings:
            return torch.zeros(1, 1, self.d_model,
                               device=next(self.parameters()).device)
        seq = torch.stack(embeddings).unsqueeze(0)  # (1, seq, d_model)
        gate = torch.sigmoid(self.gate)
        return seq * gate


def prepend_ocr_memory(features: torch.Tensor,
                       ocr_memory: torch.Tensor | None) -> torch.Tensor:
    """Prepend OCR memory to encoder features (per-sample batch of 1)."""
    if ocr_memory is None:
        return features
    return torch.cat([ocr_memory, features], dim=1)

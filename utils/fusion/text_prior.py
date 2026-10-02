"""VTL text-prior injection for the frozen Swin+MoE cascade decoder.

MarkushGrapher-2 protocol (design §37 literature): freeze the visual
encoder and the MoE decoder entirely; train ONLY a lightweight text
projection that encodes OCR cells (character tokens + normalized bboxes)
into the decoder's memory space, concatenated to the visual tokens.
Literature delta on markush tasks: 3 -> 55 CXSMILES accuracy (M2S), with
a small gain on complete structures (89.1 -> 89.8).

This module implements the minimal BioChemInsight-compatible version:
- inputs: page crop (visual tokens from the frozen Swin) + MG2-style OCR
  cells [{bbox: [x1,y1,x2,y2], text: str}]
- a 2-layer transformer text encoder over char+box embeddings (d=256 to
  match dec_hidden), projected to 1024 and ADDED to the visual tokens
  (residual, zero-initialized output so training starts at the baseline).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def build_char_vocab():
    import string

    chars = string.ascii_letters + string.digits + "'-()[]{}+=,/·\u2032 "
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad
    return stoi, len(chars) + 1


class TextPriorProjector(nn.Module):
    """OCR char+bbox tokens -> decoder memory-space residual."""

    def __init__(self, out_dim: int = 1024, d_model: int = 256, layers: int = 2,
                 max_text_tokens: int = 96):
        super().__init__()
        self.stoi, vocab = build_char_vocab()
        self.char_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.box_mlp = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_text_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        # zero-init output so step-0 == the frozen baseline (Bridging lesson)
        self.out_proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.max_text_tokens = max_text_tokens

    def encode_cells(self, cells: list[dict]) -> torch.Tensor:
        """cells -> (T, d_model) token tensor (device: module's)."""
        device = self.char_emb.weight.device
        ids, boxes = [], []
        for cell in cells or []:
            bbox = cell.get("bbox") or [0, 0, 0, 0]
            boxes.append([float(v) for v in bbox[:4]])
            for ch in str(cell.get("text") or "")[:12]:
                ids.append(self.stoi.get(ch, 0))
                # duplicate the box per char (char-level VTL)
                if len(ids) >= self.max_text_tokens:
                    break
            if len(ids) >= self.max_text_tokens:
                break
        if not ids:
            return torch.zeros(1, self.box_mlp[0].out_features, device=device)
        while len(ids) < 2:  # transformer needs >=2 tokens
            ids.append(0)
            boxes.append(boxes[0])
        boxes = boxes + boxes[:1] * (len(ids) - len(boxes))
        t_ids = torch.tensor(ids[: self.max_text_tokens], device=device).unsqueeze(0)
        t_box = torch.tensor(boxes[: len(ids[: self.max_text_tokens])],
                              device=device, dtype=torch.float).unsqueeze(0)
        x = self.char_emb(t_ids) + self.box_mlp(t_box)
        x = x + self.pos_emb[:, : x.shape[1]]
        return self.encoder(x).squeeze(0)  # (T, d_model)

    def forward(self, visual_tokens: torch.Tensor, cells: list[dict]) -> torch.Tensor:
        """Add projected text tokens (padded to T) onto the visual memory.

        visual_tokens: (B, L, C). Text residual is zero-padded to L — the
        decoder sees baseline visual tokens plus text evidence.
        """
        text = self.encode_cells(cells)  # (T, d)
        proj = self.out_proj(text)       # (T, C)
        B, L, C = visual_tokens.shape
        residual = torch.zeros(B, L, C, device=visual_tokens.device,
                               dtype=visual_tokens.dtype)
        T = min(proj.shape[0], L)
        residual[:, :T, :] = proj[:T, :].to(visual_tokens.dtype)
        return visual_tokens + residual

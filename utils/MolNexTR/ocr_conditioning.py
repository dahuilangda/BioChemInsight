"""OCR conditioning encoder (): MG2-style joint OCR token stream.

Projects ChemicalOCR cells (text + normalized bbox) into the MoE decoder's
encoder-feature space (``args.encoder_dim``), producing a fixed-length
key-padded memory that is concatenated to the Swin encoder features BEFORE the
decoder's ``enc_transform`` — the exact point where the inference path
(``MoEDecoder.decode``) already prepends ``_current_ocr_memory``.

Tensor contract (collate-friendly, batched, no Python loops over cells):
    cells: float32 (B, S, 4 + T) with S = OCR_CELL_SLOTS, T = OCR_TEXT_CHARS
        cells[..., 0:4]  normalized bbox x0, y0, x1, y1 in [0, 1] (augmented
                         frame); a cell is valid iff cells[..., 0] >= 0
        cells[..., 4:]   deterministic char ids in [0, vocab_size) stored as
                         float, -1 = absent; text left-packed per cell
    returns (memory, pad_mask):
        memory:   (B, S, out_dim) zero-gated token memory
        pad_mask: (B, 1, S) bool, True = masked out (invalid cell)

Samples with no valid cells get exactly one unmasked learned null token, so
cross-attention never attends over an all-masked key set (softmax NaN guard).

Char ids are crc32-based: Python's ``hash()`` is seed-randomised per process
and must NEVER be used for model inputs.
"""
from __future__ import annotations

import zlib

import torch
import torch.nn as nn

OCR_CELL_SLOTS = 16
OCR_TEXT_CHARS = 16
OCR_BBOX_COLS = 4
OCR_CELL_FEATURES = OCR_BBOX_COLS + OCR_TEXT_CHARS


def ocr_char_id(char: str, vocab_size: int) -> int:
    """Deterministic, process-independent char → id (crc32, not Python hash)."""
    return zlib.crc32(char.encode("utf-8")) % max(1, int(vocab_size))


def encode_ocr_cells(cells: list[dict] | None,
                     slots: int = OCR_CELL_SLOTS,
                     text_chars: int = OCR_TEXT_CHARS,
                     vocab_size: int = 512) -> torch.Tensor:
    """ChemicalOCR cells → float32 (slots, 4 + text_chars) tensor.

    Rows beyond ``slots`` are dropped; text beyond ``text_chars`` is truncated
    (R-labels are 1-4 glyphs; long cells carry no identity signal).
    """
    out = torch.full((slots, OCR_CELL_FEATURES), -1.0)
    for slot, cell in enumerate((cells or [])[:slots]):
        text = str(cell.get("text") or "").strip()
        bbox = cell.get("bbox")
        if not text or not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
            continue
        out[slot, 0:4] = torch.tensor([x0, y0, x1, y1])
        ids = [ocr_char_id(ch, vocab_size) for ch in text[:text_chars]]
        if ids:
            out[slot, OCR_BBOX_COLS:OCR_BBOX_COLS + len(ids)] = torch.tensor(
                ids, dtype=torch.float32)
    return out


class OcrConditioningEncoder(nn.Module):
    """ChemicalOCR cells → gated memory tokens in the encoder-feature space.

    Word-level (per-cell) tokens: MG2 fuses OCR at word granularity; R-labels
    are 1-4 glyphs so a cell IS the word. The output gate is zero-initialised
    (Flamingo-style): training starts from a no-op and learns to open, so a
    freshly-enabled run cannot perturb the frozen decoder's behaviour.
    """

    def __init__(self, out_dim: int, d_model: int = 256, vocab_size: int = 512,
                 bbox_bins: int = 64,
                 cell_slots: int = OCR_CELL_SLOTS,
                 text_chars: int = OCR_TEXT_CHARS):
        super().__init__()
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.bbox_bins = int(bbox_bins)
        self.cell_slots = int(cell_slots)
        self.text_chars = int(text_chars)
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.bbox_x0 = nn.Embedding(bbox_bins, d_model)
        self.bbox_y0 = nn.Embedding(bbox_bins, d_model)
        self.bbox_x1 = nn.Embedding(bbox_bins, d_model)
        self.bbox_y1 = nn.Embedding(bbox_bins, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.out_dim)
        self.null_token = nn.Parameter(torch.zeros(self.out_dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S, F) cells → (memory (B, S, out_dim), pad_mask (B, 1, S))."""
        if cells.dim() != 3 or cells.size(-1) < OCR_CELL_FEATURES:
            raise ValueError(
                f"ocr cells tensor must be (B, S, {OCR_CELL_FEATURES}); got "
                f"{tuple(cells.shape)}")
        cells = cells.float()
        batch, slots = cells.shape[0], cells.shape[1]
        bbox = cells[:, :, :OCR_BBOX_COLS].clamp(0.0, 1.0)
        valid = cells[:, :, 0] >= 0.0                                   # (B, S)
        bins = (bbox * self.bbox_bins).long().clamp(0, self.bbox_bins - 1)
        bbox_emb = (
            self.bbox_x0(bins[:, :, 0]) + self.bbox_y0(bins[:, :, 1])
            + self.bbox_x1(bins[:, :, 2]) + self.bbox_y1(bins[:, :, 3])
        )                                                                # (B, S, D)
        char_raw = cells[:, :, OCR_BBOX_COLS:]                           # (B, S, T)
        char_mask = char_raw >= 0.0
        char_ids = char_raw.long().clamp(0, self.vocab_size - 1)
        word = self.word_embedding(char_ids) * char_mask.unsqueeze(-1)   # (B, S, T, D)
        text_emb = word.sum(dim=2) / char_mask.sum(
            dim=2, keepdim=True).clamp(min=1).to(word.dtype)             # (B, S, D)
        token = self.out_proj(self.norm(self.proj(text_emb + bbox_emb)))
        # tanh gate (Flamingo-style): tanh(0) = 0 → a freshly enabled module
        # emits EXACTLY zero memory, so enabling cannot perturb behaviour.
        token = token * torch.tanh(self.gate)

        # Null token for samples with zero valid cells: keep exactly one
        # unmasked key so softmax never sees an all -inf row.
        has_any = valid.any(dim=1)                                       # (B,)
        pad_mask = ~valid
        if not bool(has_any.all()):
            empty = ~has_any
            token = token.clone()
            token[empty, 0] = self.null_token.to(token.dtype)
            pad_mask = pad_mask.clone()
            pad_mask[empty, 0] = False
        return token, pad_mask.unsqueeze(1)


def build_ocr_memory(ocr_encoder: nn.Module | None,
                     cells: torch.Tensor | None,
                     device: torch.device,
                     dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Shared MoE-side helper: cells batch → (memory, pad_mask) or None."""
    if ocr_encoder is None or cells is None:
        return None
    memory, pad_mask = ocr_encoder(cells.to(device))
    return memory.to(dtype), pad_mask.to(device)

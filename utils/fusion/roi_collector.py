"""Differentiable RoI collection: page features -> per-box token grids.

Contract (see research/fusion_design.md §3): the MolNexTR/MoE decoder
consumes ``(features (N, 144, 1024), hiddens=[multiscale])``. This module
samples per-box token grids from the Swin multi-scale hiddens so the
existing decoder can run **unmodified** on page-level detections.

Shape conventions
-----------------
- ``fine``   : (B, C_f, H, W)  e.g. 24×24×512   (hiddens[2], main signal)
- ``coarse`` : (B, C_c, H/2, W/2) e.g. 12×12×1024 (hiddens[3], context)
- ``boxes``  : list[Tensor(K, 4)] in normalized xyxy page coordinates
- output per page: token grid (K, 144, out_dim) with out_dim == encoder_dim
  (1024 by default, matching ``enc_pos_emb = Embedding(144, 1024)``).

Everything is differentiable (torchvision ``roi_align``), so detection and
recognition can be trained jointly end-to-end (FOTS/ABCNet lesson).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

TOKENS_PER_SIDE = 12  # MolNexTR: 12*12 = 144 encoder tokens
CONTEXT_SIDE = 6      # coarse context grid sampled at 6*6 then tiled to 12*12


class RoICollector(nn.Module):
    """Sample per-box (12, 12) token grids from two feature scales.

    Fine features are roi-aligned at 12×12 (the recognition signal); coarse
    features at 6×6 (page context), broadcast onto the same grid; both are
    projected and summed into the decoder's expected channel width.
    """

    def __init__(self, fine_channels: int, coarse_channels: int, out_dim: int = 1024) -> None:
        super().__init__()
        self.fine_proj = nn.Conv2d(fine_channels, out_dim, kernel_size=1)
        self.coarse_proj = nn.Conv2d(coarse_channels, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        fine: torch.Tensor,      # (B, C_f, Hf, Wf)
        coarse: torch.Tensor,    # (B, C_c, Hc, Wc)
        boxes: list[torch.Tensor],  # per image (K_i, 4) normalized xyxy
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return token grids shaped for the existing decoder.

        Returns
        -------
        tokens : (sum(K), 144, out_dim) — per-box encoder "features"
        hiddens : list with one entry per box: [(144, C_f), (144, C_c)]
            (per-token channel features, kept for multiscale consumers such
            as the attachment head; spatial layout flattened row-major).
        """
        if fine.device != coarse.device:
            raise ValueError("fine/coarse on different devices")
        device = fine.device
        height, width = fine.shape[-2:]
        total_boxes = sum(box.shape[0] for box in boxes)
        if total_boxes == 0:
            empty_tokens = fine.new_zeros((0, TOKENS_PER_SIDE ** 2, self.out_dim))
            return empty_tokens, []

        # roi_align wants absolute pixel coords: scale normalized boxes once.
        scale = torch.tensor([width, height, width, height], device=device)
        abs_boxes = [box * scale for box in boxes]

        fine_rois = roi_align(fine, abs_boxes, output_size=(TOKENS_PER_SIDE, TOKENS_PER_SIDE),
                              spatial_scale=1.0, aligned=True)
        coarse_rois = roi_align(coarse, abs_boxes, output_size=(CONTEXT_SIDE, CONTEXT_SIDE),
                                spatial_scale=1.0, aligned=True)

        fine_tokens = self.fine_proj(fine_rois)                     # (K, out, 12, 12)
        coarse_tokens = self.coarse_proj(coarse_rois)               # (K, out, 6, 6)
        coarse_up = F.interpolate(
            coarse_tokens, size=(TOKENS_PER_SIDE, TOKENS_PER_SIDE), mode="nearest-exact"
        )
        merged = self.norm((fine_tokens + coarse_up).permute(0, 2, 3, 1))  # (K,12,12,out)
        tokens = merged.flatten(1, 2)                                # (K, 144, out)

        fine_flat = fine_rois.flatten(2).transpose(1, 2)             # (K, 144, C_f)
        coarse_flat = coarse_up.flatten(2).transpose(1, 2)           # (K, 144, C_c)
        hiddens = [
            [f_box, c_box] for f_box, c_box in zip(fine_flat, coarse_flat)
        ]
        return tokens, hiddens


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou: float = 0.5) -> torch.Tensor:
    """Thin wrapper keeping the detection postprocess testable."""
    from torchvision.ops import nms

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return nms(boxes, scores, iou)

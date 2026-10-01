"""Differentiable RoI collection: page features -> per-box token grids.

Contract (v2, corrected by independent audit 2026-10-01):

* Production ``swin_base`` hiddens are **3D ``(B, L, C)`` flattened grids**
  (e.g. (1, 9216, 128) / (1, 2304, 256) / (1, 576, 512) / (1, 144, 1024));
  the 4D branch exists only for encoder names containing ``'patch'``
  (components.py ``forwards()``). :func:`hidden_to_bchw` accepts both.
* The production MoE multiscale attachment head consumes **3 levels with
  channels (256, 512, 1024)** laid out ``(B, L, C)`` with L a perfect
  square (attachment_set.py ``_project_multiscale_levels``). We therefore
  sample three scales per box — mid (48×48/256), fine (24×24/512),
  coarse (12×12/1024) — and hand the decoder ``hiddens`` in the same
  coarse-to-fine order it sees from the encoder.

Token grids stay 12×12 = 144 to match ``enc_pos_emb = Embedding(144, 1024)``
and the ``features`` input of ``MoEDecoder.decode``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align

TOKENS_PER_SIDE = 12
CONTEXT_SIDE = 6


def hidden_to_bchw(hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Normalize an encoder hidden to (B, C, H, W); returns (tensor, side).

    3D input (B, L, C) is unfolded via its perfect-square L; 4D input
    (B, H, W, C) is permuted. Non-square 4D grids are rejected loudly.
    """
    if hidden.dim() == 3:
        batch, length, channels = hidden.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise ValueError(f"hidden length {length} is not a perfect square")
        return hidden.transpose(1, 2).reshape(batch, channels, side, side), side
    if hidden.dim() == 4:
        batch, height, width, channels = hidden.shape
        if height != width:
            raise ValueError(f"non-square hidden grid {height}x{width} unsupported")
        return hidden.permute(0, 3, 1, 2), height
    raise ValueError(f"unexpected hidden rank {hidden.dim()}")


class RoICollector(nn.Module):
    """Sample per-box token grids + 3-level per-RoI hiddens.

    Recognition signal comes from roi-aligning the fine level at 12×12;
    context from the coarse level at 6×6 (broadcast onto the grid). The
    three hiddens scales are re-sampled per box at 12×12 and returned in
    the decoder's expected coarse→fine order (256, 512, 1024).
    """

    LEVEL_SPECS = (("mid", 256), ("fine", 512), ("coarse", 1024))

    def __init__(self, mid_channels: int, fine_channels: int, coarse_channels: int,
                 out_dim: int = 1024) -> None:
        super().__init__()
        self.fine_proj = nn.Conv2d(fine_channels, out_dim, kernel_size=1)
        self.coarse_proj = nn.Conv2d(coarse_channels, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim
        self.level_channels = (mid_channels, fine_channels, coarse_channels)

    def forward(
        self,
        levels_bchw: dict[str, torch.Tensor],        # {'mid','fine','coarse'} (B,C,H,W)
        boxes: list[torch.Tensor],                    # per image (K,4) normalized xyxy
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        device = levels_bchw["fine"].device
        total = sum(box.shape[0] for box in boxes)
        if total == 0:
            empty = levels_bchw["fine"].new_zeros((0, TOKENS_PER_SIDE ** 2, self.out_dim))
            return empty, []

        fine = levels_bchw["fine"]
        # Each level has its own grid side: normalize -> level pixels per
        # level. Using one level's scale for another silently samples the
        # wrong image region (round-2 audit's decisive finding).
        abs_boxes_by_level = {
            name: [box * torch.tensor(
                [t.shape[-1], t.shape[-2], t.shape[-1], t.shape[-2]], device=device)
                for box in boxes]
            for name, t in levels_bchw.items()
        }

        fine_rois = roi_align(fine, abs_boxes_by_level["fine"],
                              (TOKENS_PER_SIDE, TOKENS_PER_SIDE),
                              spatial_scale=1.0, aligned=True)
        coarse_rois = roi_align(levels_bchw["coarse"], abs_boxes_by_level["coarse"],
                                (CONTEXT_SIDE, CONTEXT_SIDE),
                                spatial_scale=1.0, aligned=True)
        fine_tokens = self.fine_proj(fine_rois)
        coarse_up = F.interpolate(self.coarse_proj(coarse_rois),
                                  size=(TOKENS_PER_SIDE, TOKENS_PER_SIDE),
                                  mode="nearest-exact")
        merged = self.norm((fine_tokens + coarse_up).permute(0, 2, 3, 1))
        tokens = merged.flatten(1, 2)                                  # (K, 144, out)

        hidden_levels: list[torch.Tensor] = []
        for name in ("mid", "fine", "coarse"):  # decoder expects (256, 512, 1024)
            rois = roi_align(levels_bchw[name], abs_boxes_by_level[name],
                             (TOKENS_PER_SIDE, TOKENS_PER_SIDE),
                             spatial_scale=1.0, aligned=True)
            hidden_levels.append(rois.flatten(2).transpose(1, 2))      # (K, 144, C)
        per_box_hiddens = [
            [hidden_levels[0][k], hidden_levels[1][k], hidden_levels[2][k]]
            for k in range(total)
        ]
        return tokens, per_box_hiddens


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou: float = 0.5) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return nms(boxes, scores, iou)

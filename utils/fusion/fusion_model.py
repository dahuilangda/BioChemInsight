"""Fused page-level recognizer: encoder + light detection head + RoI + MoE.

v2 (post-audit, 2026-10-01). Corrections vs the first prototype:

* production ``swin_base`` hiddens are 3D (B, L, C) — normalized via
  :func:`utils.fusion.roi_collector.hidden_to_bchw`, never assumed 4D;
* heads build **eagerly** in ``__init__`` (lazy creation broke the
  optimizer/DDP contract and created inference tensors);
* per-RoI hiddens carry **3 levels (256, 512, 1024)** matching the
  production multiscale attachment head;
* AET now actually mixes predicted boxes into recognition training
  (the previous version's predicted branch was unreachable dead code);
* detection losses are computed inside ``forward`` with NaN guards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fusion.per_box_refs import expand_refs_for_boxes
from utils.fusion.roi_collector import RoICollector, hidden_to_bchw, nms_boxes


@dataclass
class FusionConfig:
    mid_level: int = 1        # 48×48/256
    fine_level: int = 2       # 24×24/512
    coarse_level: int = 3     # 12×12/1024
    mid_channels: int = 256
    fine_channels: int = 512
    coarse_channels: int = 1024
    tokens: int = 144
    encoder_dim: int = 1024
    num_convs: int = 2
    hid: int = 256
    aet_gt_ratio: float = 0.5
    nms_iou: float = 0.5
    score_threshold: float = 0.5
    max_boxes: int = 32
    stage: str = "a"          # 'a' = detection only, 'ab' = joint


class LightDetectionHead(nn.Module):
    """Anchor-free single-class head (DINO-lite): ltrb + centerness."""

    def __init__(self, in_channels: int, cfg: FusionConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(cfg.num_convs):
            layers += [nn.Conv2d(current, cfg.hid, 3, padding=1),
                       nn.GroupNorm(32, cfg.hid), nn.ReLU(inplace=True)]
            current = cfg.hid
        self.tower = nn.Sequential(*layers)
        self.box_reg = nn.Conv2d(cfg.hid, 4, 3, padding=1)
        self.centerness = nn.Conv2d(cfg.hid, 1, 3, padding=1)
        nn.init.normal_(self.box_reg.weight, std=0.01)
        nn.init.normal_(self.centerness.weight, std=0.01)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tower = self.tower(feats)
        return self.box_reg(tower).relu(), self.centerness(tower).sigmoid().squeeze(1)


def _decode_ltrb(ltrb_flat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    x, y = points[..., 0], points[..., 1]
    return torch.stack([x - ltrb_flat[..., 0], y - ltrb_flat[..., 1],
                        x + ltrb_flat[..., 2], y + ltrb_flat[..., 3]], dim=-1)


def _feature_points(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).view(-1, 2)


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def detection_losses(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                     gt_boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    """Assignment-free detection losses with NaN guards.

    ``pred_boxes`` are decoded xyxy boxes (not ltrb). Each prediction
    targets its best-IoU ground truth when overlap is real (>0.3);
    otherwise it becomes a background target for the score head.
    """
    zero = pred_boxes.sum() * 0.0
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return {"loss_det": zero}
    ious = _iou_matrix(gt_boxes, pred_boxes)          # (G, P)
    matched = ious.max(dim=0).values > 0.3
    if matched.any():
        best_gt = ious.argmax(dim=0)
        box_loss = F.l1_loss(pred_boxes[matched], gt_boxes[best_gt][matched])
    else:
        box_loss = zero
    score_loss = F.binary_cross_entropy(
        pred_scores.clamp(1e-4, 1 - 1e-4), matched.float().clamp(0.0, 1.0)
    )
    return {"loss_det": box_loss + score_loss}


class FusedStructureRecognizer(nn.Module):
    """encoder(page) -> detection head -> RoI collector -> MoE decoder."""

    def __init__(self, encoder: nn.Module, decoder: Any, cfg: FusionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.encoder = encoder
        self.decoder = decoder
        # eager build: optimizer/DDP must see these at construction time
        self._head = LightDetectionHead(self.cfg.fine_channels, self.cfg)
        self._collector = RoICollector(self.cfg.mid_channels, self.cfg.fine_channels,
                                       self.cfg.coarse_channels, self.cfg.encoder_dim)

    # -- helpers --------------------------------------------------------------
    def _levels(self, hiddens: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "mid": hidden_to_bchw(hiddens[self.cfg.mid_level])[0],
            "fine": hidden_to_bchw(hiddens[self.cfg.fine_level])[0],
            "coarse": hidden_to_bchw(hiddens[self.cfg.coarse_level])[0],
        }

    def detect(self, hiddens: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Return (per-image xyxy boxes, per-image scores, per-image flat boxes/scores for loss)."""
        fine = self._levels(hiddens)["fine"]
        ltrb, centerness = self._head(fine)
        batch, _, height, width = ltrb.shape
        points = _feature_points(height, width, ltrb.device)
        boxes = _decode_ltrb(ltrb.view(batch, 4, -1).permute(0, 2, 1), points)  # (B, HW, 4)
        scores = centerness.view(batch, -1)
        out_boxes, out_scores = [], []
        for i in range(batch):
            keep = scores[i] > self.cfg.score_threshold
            page_boxes = boxes[i][keep].clamp(0.0, 1.0)
            page_scores = scores[i][keep]
            if page_boxes.numel():
                idx = nms_boxes(page_boxes, page_scores, self.cfg.nms_iou)
                page_boxes = page_boxes[idx][: self.cfg.max_boxes]
                page_scores = page_scores[idx][: self.cfg.max_boxes]
            out_boxes.append(page_boxes)
            out_scores.append(page_scores)
        return out_boxes, out_scores, [boxes[i] for i in range(batch)], [scores[i] for i in range(batch)]

    # -- training ---------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        refs: dict[str, torch.Tensor],
        structure_labels: torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
        box_structure_ids: list[list[int]] | None = None,
        structure_counts: list[int] | None = None,
    ) -> dict[str, Any]:
        features, hiddens = self.encoder(images)
        levels = self._levels(hiddens)
        pred_boxes, _, flat_boxes, flat_scores = self.detect(hiddens)

        losses = {}
        if gt_boxes is not None:
            page_losses = [
                detection_losses(flat_boxes[i], flat_scores[i], gt_boxes[i])
                for i in range(len(gt_boxes))
            ]
            losses["loss_det"] = torch.stack([p["loss_det"] for p in page_losses]).mean()

        if self.cfg.stage == "a":
            # detection-only stage: skip the (expensive) recognition decoder
            return {"detection": {"boxes": pred_boxes}, "losses": losses,
                    "features": features, "hiddens": hiddens}

        # AET (ABCNet): mix predicted boxes into recognition training.
        if gt_boxes is None:
            chosen = pred_boxes
        elif self.training:
            chosen = [
                gt if torch.rand((), device=images.device) < self.cfg.aet_gt_ratio else pred
                for gt, pred in zip(gt_boxes, pred_boxes)
            ]
        else:
            chosen = gt_boxes

        tokens, roi_hiddens = self._collector(levels, chosen)
        if tokens.shape[0] == 0:
            losses["loss_recognition_absent"] = features.sum() * 0.0
            return {"detection": {"boxes": pred_boxes}, "losses": losses,
                    "features": features, "hiddens": hiddens}

        assert tokens.shape[1] == features.shape[1] == self.cfg.tokens, (
            f"token contract violated: {tokens.shape[1]} vs {features.shape[1]}"
        )
        flat = [
            torch.stack([h[level] for h in roi_hiddens], dim=0)
            for level in range(len(roi_hiddens[0]))
        ]
        per_page_counts = [int(b.shape[0]) for b in chosen]
        expanded = expand_refs_for_boxes(
            refs, per_page_counts,
            structure_labels=structure_labels,
            box_structure_ids=box_structure_ids,
            structure_counts=structure_counts,
        )
        recognition = self.decoder(tokens, expanded.refs,
                                   structure_labels=expanded.structure_labels,
                                   encoder_hiddens=flat)
        return {"detection": {"boxes": pred_boxes}, "recognition": recognition,
                "losses": losses, "features": tokens, "hiddens": flat,
                "per_box_refs": expanded}

    # -- inference -----------------------------------------------------------------
    @torch.inference_mode()
    def recognize_page(self, images: torch.Tensor, expected_structure_types=None) -> list[dict[str, Any]]:
        features, hiddens = self.encoder(images)
        levels = self._levels(hiddens)
        boxes, scores, _, _ = self.detect(hiddens)
        tokens, roi_hiddens = self._collector(levels, boxes)
        if tokens.shape[0] == 0:
            return [{"boxes": b, "predictions": []} for b in boxes]
        flat = [
            torch.stack([h[level] for h in roi_hiddens], dim=0)
            for level in range(len(roi_hiddens[0]))
        ]
        per_box_types = expected_structure_types
        if expected_structure_types is not None:
            flat_types = []
            for page_index, page_boxes in enumerate(boxes):
                page_type = expected_structure_types[page_index]
                flat_types.extend([page_type] * int(page_boxes.shape[0]))
            per_box_types = flat_types
        decoded = self.decoder.decode(tokens, hiddens=flat,
                                      expected_structure_types=per_box_types)
        results, index = [], 0
        for page_boxes in boxes:
            count = page_boxes.shape[0]
            results.append({"boxes": page_boxes, "predictions": decoded[index: index + count]})
            index += count
        return results

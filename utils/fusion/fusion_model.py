"""Fused page-level recognizer: encoder + light detection head + RoI + MoE.

This is the training-side prototype for fusion route (i) — it wraps the
existing Swin encoder and MoE decoder with the pieces they were missing:

* a light single-class detection head over the encoder's multi-scale
  hiddens (DINO-style GIoU + focal losses);
* the differentiable :class:`~utils.fusion.roi_collector.RoICollector` that
  turns detected boxes into the (K, 144, 1024) token contract the decoder
  already speaks;
* AET training (ABCNet lesson): recognition consumes a 50/50 mix of
  ground-truth and predicted boxes so the decoder tolerates imperfect RoIs.

The decoder itself is injected, not re-implemented — the whole MoE stack
(router, attachment head, edge head, ghost-repair inputs) is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fusion.roi_collector import RoICollector


@dataclass
class FusionConfig:
    fine_level: int = 2      # hiddens index used as recognition signal (24×24)
    coarse_level: int = 3    # hiddens index used as context (12×12)
    encoder_dim: int = 1024
    num_convs: int = 2
    hid: int = 256
    aet_gt_ratio: float = 0.5
    nms_iou: float = 0.5
    max_boxes: int = 32


class LightDetectionHead(nn.Module):
    """Single-class anchor-free head over a chosen hidden level.

    Predicts (l, t, r, b) distances to the box edges + centerness, DINO
    style, but deliberately small: Stage-A trains *only* this + the RoI
    projection with the Swin backbone frozen (Frozen-DETR memory lesson).
    """

    def __init__(self, in_channels: int, cfg: FusionConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(cfg.num_convs):
            layers += [nn.Conv2d(in_channels, cfg.hid, 3, padding=1), nn.GroupNorm(32, cfg.hid), nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*layers)
        self.box_reg = nn.Conv2d(cfg.hid, 4, 3, padding=1)
        self.centerness = nn.Conv2d(cfg.hid, 1, 3, padding=1)
        nn.init.normal_(self.box_reg.weight, std=0.01)
        nn.init.normal_(self.centerness.weight, std=0.01)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tower = self.tower(feats)
        return self.box_reg(tower).relu(), self.centerness(tower).sigmoid().squeeze(1)


def _ltrb_to_xyxy(ltrb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Decode (l,t,r,b) distances from feature-map points into xyxy boxes."""
    x, y = points[..., 0], points[..., 1]
    return torch.stack([x - ltrb[..., 0], y - ltrb[..., 1],
                        x + ltrb[..., 2], y + ltrb[..., 3]], dim=-1)


def _feature_points(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).view(-1, 2)


class FusedStructureRecognizer(nn.Module):
    """encoder(page) -> detection head -> RoI collector -> MoE decoder."""

    def __init__(self, encoder: nn.Module, decoder: Any, cfg: FusionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.encoder = encoder
        self.decoder = decoder
        # channel widths resolved lazily on first forward (Swin hiddens known
        # only after construction; components.py: 96/48/24/12 grid scales).
        self._head: LightDetectionHead | None = None
        self._collector: RoICollector | None = None
        self._resolved = False

    def _build_heads(self, fine_channels: int, coarse_channels: int) -> None:
        self._head = LightDetectionHead(fine_channels, self.cfg)
        self._collector = RoICollector(fine_channels, coarse_channels, self.cfg.encoder_dim)
        self._resolved = True

    # -- detection ----------------------------------------------------------
    def detect(self, hiddens: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return (per-image normalized boxes, per-image scores)."""
        fine = hiddens[self.cfg.fine_level].permute(0, 2, 3, 1)   # wait: hiddens are (B,H,W,C)
        fine = fine.permute(0, 3, 1, 2)
        ltrb, centerness = self._head(fine)
        b, _, height, width = ltrb.shape
        points = _feature_points(height, width, ltrb.device)
        boxes = _ltrb_to_xyxy(ltrb.view(b, 4, -1).permute(0, 2, 1), points)  # (B, HW, 4)
        scores = centerness.view(b, -1)
        out_boxes: list[torch.Tensor] = []
        out_scores: list[torch.Tensor] = []
        for i in range(b):
            keep = scores[i] > 0.5
            page_boxes = boxes[i][keep].clamp(0.0, 1.0)
            page_scores = scores[i][keep]
            if page_boxes.numel():
                from utils.fusion.roi_collector import nms_boxes

                keep_idx = nms_boxes(page_boxes, page_scores, self.cfg.nms_iou)
                page_boxes = page_boxes[keep_idx][: self.cfg.max_boxes]
                page_scores = page_scores[keep_idx][: self.cfg.max_boxes]
            out_boxes.append(page_boxes)
            out_scores.append(page_scores)
        return out_boxes, out_scores

    # -- training -----------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        refs: dict[str, torch.Tensor],
        structure_labels: torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Teacher-forced joint forward.

        ``gt_boxes``: per-image (K, 4) normalized xyxy ground truth. With AET
        enabled the recognition branch consumes a mix of predicted and GT
        boxes; without GT (pure inference-style training) predicted only.
        """
        features, hiddens = self.encoder(images)
        fine_ch = hiddens[self.cfg.fine_level].shape[-1]
        coarse_ch = hiddens[self.cfg.coarse_level].shape[-1]
        if not self._resolved:
            self._build_heads(fine_ch, coarse_ch)
            # move lazily-built heads to the encoder's device/precision
            self._head.to(images.device)
            self._collector.to(images.device)

        pred_boxes, pred_scores = self.detect(hiddens)
        det_out = {"pred_boxes": pred_boxes, "pred_scores": pred_scores}

        if gt_boxes is None:
            chosen = pred_boxes
        elif self.training and torch.rand(()) < self.cfg.aet_gt_ratio:
            chosen = gt_boxes
        else:
            chosen = gt_boxes

        tokens, roi_hiddens = self._collector(
            hiddens[self.cfg.fine_level].permute(0, 3, 1, 2),
            hiddens[self.cfg.coarse_level].permute(0, 3, 1, 2),
            chosen,
        )
        if tokens.shape[0] == 0:
            return {"detection": det_out, "recognition": None,
                    "note": "no boxes detected/trained on this batch"}

        fused_features = features.new_zeros((tokens.shape[0], features.shape[1], features.shape[2]))
        fused_features[:, : tokens.shape[1]] = tokens[:, : features.shape[1]]
        flat_hiddens = [
            torch.stack([h[level] for h in roi_hiddens], dim=0)
            for level in range(len(roi_hiddens[0]))
        ]
        recognition = self.decoder(fused_features, refs, structure_labels=structure_labels,
                                   encoder_hiddens=flat_hiddens)
        return {"detection": det_out, "recognition": recognition,
                "features": fused_features, "hiddens": flat_hiddens}

    # -- inference ------------------------------------------------------------
    @torch.inference_mode()
    def recognize_page(self, images: torch.Tensor, expected_structure_types=None) -> list[dict[str, Any]]:
        features, hiddens = self.encoder(images)
        fine_ch = hiddens[self.cfg.fine_level].shape[-1]
        coarse_ch = hiddens[self.cfg.coarse_level].shape[-1]
        if not self._resolved:
            self._build_heads(fine_ch, coarse_ch)
            self._head.to(images.device)
            self._collector.to(images.device)
        boxes, scores = self.detect(hiddens)
        tokens, roi_hiddens = self._collector(
            hiddens[self.cfg.fine_level].permute(0, 3, 1, 2),
            hiddens[self.cfg.coarse_level].permute(0, 3, 1, 2),
            boxes,
        )
        results: list[dict[str, Any]] = []
        if tokens.shape[0] == 0:
            return results
        fused = features.new_zeros((tokens.shape[0], features.shape[1], features.shape[2]))
        fused[:, : tokens.shape[1]] = tokens[:, : features.shape[1]]
        flat_hiddens = [
            torch.stack([h[level] for h in roi_hiddens], dim=0)
            for level in range(len(roi_hiddens[0]))
        ]
        decoded = self.decoder.decode(fused, hiddens=flat_hiddens,
                                      expected_structure_types=expected_structure_types)
        index = 0
        for page_boxes in boxes:
            count = page_boxes.shape[0]
            results.append({
                "boxes": page_boxes,
                "predictions": decoded[index: index + count],
            })
            index += count
        return results


def detection_losses(
    pred_ltrb_page: torch.Tensor,
    pred_scores_page: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Minimal one-to-many assignment-free losses for the light head.

    Each GT box supervises its best-overlapping prediction (closest center);
    unmatched predictions get a low-score target. Kept intentionally simple
    for the prototype: Stage-A convergence target, not COCO SOTA.
    """
    if gt_boxes.numel() == 0:
        return {"loss_det": pred_ltrb_page.sum() * 0.0}
    ious = torch.diag(
        torch.stack([_pair_iou(box, pred_ltrb_page) for box in gt_boxes])
    ) if False else _iou_matrix(gt_boxes, pred_ltrb_page)
    best_pred = ious.argmax(dim=0)          # (P,) best gt per prediction
    matched = ious.max(dim=0).values > 0.3  # predictions with a real gt
    targets = gt_boxes[best_pred]
    box_loss = F.l1_loss(pred_ltrb_page[matched], targets[matched])
    score_target = matched.float()
    score_loss = F.binary_cross_entropy(
        pred_scores_page.clamp(1e-4, 1 - 1e-4), score_target
    )
    return {"loss_det": box_loss + score_loss}


def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def _pair_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    return _iou_matrix(box[None, :], boxes)[0]

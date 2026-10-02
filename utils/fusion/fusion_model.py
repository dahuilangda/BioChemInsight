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

from utils.fusion.anchor_frame import reanchor_predictions, reanchor_refs
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
    # tuned on P2 multi-structure holdout: F1 0.296 -> 0.557 over the
    # default (0.5, NMS 0.5); see design doc §14 sweep
    score_threshold: float = 0.90
    nms_iou: float = 0.30
    max_boxes: int = 32
    stage: str = "a"          # 'a' = detection only, 'ab' = joint
    residual_collector: bool = False  # B2: zero-init residual token path
    # detection feature level: 2 = fine 24x24/512ch (synthetic-domain default),
    # 1 = mid 48x48/256ch (real-page small structures need the denser grid;
    # 24x24 gives Patent300 structures 1.5x0.6 anchor points — design §30)
    detection_level: int = 2


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
                     gt_boxes: torch.Tensor,
                     points: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """FCOS center-assignment detection losses.

    A grid cell is POSITIVE when its center falls inside a GT box (the
    standard FCOS/DINO rule); its regression target is the ltrb distance
    to that box's edges. Background cells carry only the score loss. This
    replaces the earlier IoU-threshold matching, which deadlocked on real
    pages: at init only 3/2304 cells reached IoU>0.3, so BCE's negative
    dominance drove centerness to all-zero and positives never reappeared
    (design §31).

    ``points``: (P, 2) normalized cell centers matching ``pred_boxes``'s
    grid order; callers that lack it fall back to IoU matching.
    """
    zero = pred_boxes.sum() * 0.0
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return {"loss_det": zero}
    if points is None or points.shape[0] != pred_boxes.shape[0]:
        # legacy fallback: best-IoU matching
        ious = _iou_matrix(gt_boxes, pred_boxes)
        matched = ious.max(dim=0).values > 0.3
        best_gt = ious.argmax(dim=0)
    else:
        px = points[:, 0:1]   # (P, 1)
        py = points[:, 1:2]   # (P, 1)
        inside = ((px > gt_boxes[None, :, 0]) &
                  (px < gt_boxes[None, :, 2]) &
                  (py > gt_boxes[None, :, 1]) &
                  (py < gt_boxes[None, :, 3]))                    # (P, G)
        # each positive cell targets the SMALLEST containing box (handles
        # nested boxes the way FPN level assignment would)
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                 (gt_boxes[:, 3] - gt_boxes[:, 1]))               # (G,)
        big = torch.where(inside, areas[None, :], torch.full_like(areas[None, :], float("inf")))
        best_gt = big.argmin(dim=1)                               # (P,)
        matched = inside.any(dim=1)
    if matched.any():
        box_loss = F.l1_loss(pred_boxes[matched], gt_boxes[best_gt][matched])
    else:
        box_loss = zero
    probs = pred_scores.clamp(1e-4, 1 - 1e-4)
    logits = torch.log(probs) - torch.log1p(-probs)
    score_loss = F.binary_cross_entropy_with_logits(
        logits, matched.float().clamp(0.0, 1.0)
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
        head_channels = {1: self.cfg.mid_channels, 2: self.cfg.fine_channels}[
            self.cfg.detection_level]
        self._head = LightDetectionHead(head_channels, self.cfg)
        self._collector = RoICollector(self.cfg.mid_channels, self.cfg.fine_channels,
                                       self.cfg.coarse_channels, self.cfg.encoder_dim,
                                       residual=self.cfg.residual_collector)

    # -- helpers --------------------------------------------------------------
    def _levels(self, hiddens: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "mid": hidden_to_bchw(hiddens[self.cfg.mid_level])[0],
            "fine": hidden_to_bchw(hiddens[self.cfg.fine_level])[0],
            "coarse": hidden_to_bchw(hiddens[self.cfg.coarse_level])[0],
        }

    def detect(self, hiddens: list[torch.Tensor]):
        """Return (per-image xyxy boxes, per-image scores, per-image flat boxes/scores for loss)."""
        level_name = {1: "mid", 2: "fine"}[self.cfg.detection_level]
        feats = self._levels(hiddens)[level_name]
        ltrb, centerness = self._head(feats)
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
        return (out_boxes, out_scores, [boxes[i] for i in range(batch)],
                [scores[i] for i in range(batch)], points)

    @staticmethod
    def _align_ids(chosen, gt_boxes, box_structure_ids):
        """Map structure ids onto the *chosen* boxes (AET-safe).

        When the chosen boxes ARE the ground truth, caller ids pass through.
        When AET picked predicted boxes, each predicted box inherits the
        structure of its best-IoU ground-truth box on that page (matching
        the recognition target to whatever region the RoI actually shows).
        """
        if gt_boxes is None or box_structure_ids is None:
            return box_structure_ids
        aligned: list[list[int]] = []
        for page, chosen_page in enumerate(chosen):
            gt_page = gt_boxes[page]
            ids_page = box_structure_ids[page] if page < len(box_structure_ids) else None
            if ids_page is None or len(chosen_page) == len(gt_page) and torch.equal(
                    chosen_page, gt_page):
                aligned.append(list(ids_page) if ids_page is not None
                               else [0] * len(chosen_page))
                continue
            if gt_page.numel() == 0:
                aligned.append([0] * len(chosen_page))
                continue
            ious = _iou_matrix(chosen_page, gt_page)  # (K_pred, K_gt)
            best = ious.argmax(dim=1)
            page_ids = []
            for k in range(chosen_page.shape[0]):
                gt_index = int(best[k])
                page_ids.append(ids_page[gt_index] if gt_index < len(ids_page) else 0)
            aligned.append(page_ids)
        return aligned

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
        pred_boxes, _, flat_boxes, flat_scores, grid_points = self.detect(hiddens)

        losses = {}
        if gt_boxes is not None:
            page_losses = [
                detection_losses(flat_boxes[i], flat_scores[i], gt_boxes[i],
                                 points=grid_points)
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
        aligned_ids = self._align_ids(chosen, gt_boxes, box_structure_ids)
        device = images.device
        refs_on_device = refs
        if refs and next(iter(refs.values())) is not None:
            first = refs["chartok_coords"][0] if "chartok_coords" in refs else next(
                (v for v in refs.values() if torch.is_tensor(v) and v.dim() > 0), None)
            if first is not None and first.device != device:
                def _move(value):
                    if torch.is_tensor(value):
                        return value.to(device, non_blocking=True)
                    if isinstance(value, list):
                        return [_move(v) for v in value]
                    return value
                refs_on_device = {k: _move(v) for k, v in refs.items()}
                structure_labels = (
                    structure_labels.to(device) if structure_labels is not None else None
                )
        expanded = expand_refs_for_boxes(
            refs, per_page_counts,
            structure_labels=structure_labels,
            box_structure_ids=aligned_ids,
            structure_counts=structure_counts,
        )
        reanchored_refs = reanchor_refs(expanded.refs, chosen)
        recognition = self.decoder(tokens, reanchored_refs,
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
        boxes, scores, _, _, _ = self.detect(hiddens)
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
        for page_index, page_boxes in enumerate(boxes):
            count = page_boxes.shape[0]
            page_predictions = decoded[index: index + count]
            if isinstance(page_predictions, list) and page_predictions:
                page_predictions = reanchor_predictions(page_predictions, page_boxes)
            results.append({"boxes": page_boxes, "predictions": page_predictions})
            index += count
        return results


class CascadeStructureRecognizer(nn.Module):
    """M1 cascade path: detect once, then re-crop at native resolution.

    The RoI route was falsified three ways (design §16-18): the information
    bottleneck sits BEFORE roi_align — a page squeezed to 384² gives each
    structure only a sliver of the 12×12 coarse grid, while the decoder was
    trained on crops that own the full grid. Cascade closes that gap by
    construction: the detected box is cropped from the FULL-RESOLUTION page
    image, resized to 384² exactly like the two-stage pipeline, and passed
    through the stock encoder + MoE decoder (weights untouched). Only the
    detection head is new — which is exactly the Phase-1 efficiency
    architecture (one detection forward replaces tiles/mask-completion, the
    crop forward replaces nothing).
    """

    def __init__(self, encoder: nn.Module, decoder: Any,
                 head: LightDetectionHead, cfg: FusionConfig,
                 fine_level: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.detection_head = head
        self.cfg = cfg
        self.fine_level = fine_level

    def detect(self, page_tensor_384: torch.Tensor) -> list[torch.Tensor]:
        with torch.inference_mode():
            _, hiddens = self.encoder(page_tensor_384)
        fine = hidden_to_bchw(hiddens[self.fine_level])[0]
        ltrb, centerness = self.detection_head(fine)
        batch, _, height, width = ltrb.shape
        points = _feature_points(height, width, ltrb.device)
        boxes = _decode_ltrb(ltrb.view(batch, 4, -1).permute(0, 2, 1), points)
        scores = centerness.view(batch, -1)
        out: list[torch.Tensor] = []
        for i in range(batch):
            keep = scores[i] > self.cfg.score_threshold
            page_boxes = boxes[i][keep].clamp(0.0, 1.0)
            page_scores = scores[i][keep]
            if page_boxes.numel():
                idx = nms_boxes(page_boxes, page_scores, self.cfg.nms_iou)
                page_boxes = page_boxes[idx][: self.cfg.max_boxes]
            out.append(page_boxes)
        return out

    @staticmethod
    def tighten_box_with_ink(box: torch.Tensor, page_gray: "torch.Tensor",
                             margin: int = 4, shrink_steps: int = 6) -> torch.Tensor:
        """Shrink a detected box to the ink it contains.

        The Stage-A head emits loose boxes on real layouts (trained on pasted
        synthetics with full-cell margins). The two-stage pipeline achieves
        tight crops via mask expansion; here we take the cheap deterministic
        proxy: inside the detected box, find the ink (dark pixel) bounding
        region and re-emit that (with a small margin), iteratively shrinking
        when the first pass still catches stray ink rows/columns.

        ``page_gray``: (H, W) float tensor of the full-res page in [0, 1].
        """
        height, width = page_gray.shape
        x1 = int(max(0.0, float(box[0])) * width)
        y1 = int(max(0.0, float(box[1])) * height)
        x2 = int(min(1.0, float(box[2])) * width)
        y2 = int(min(1.0, float(box[3])) * height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return box
        window = page_gray[y1:y2, x1:x2]
        for _ in range(shrink_steps):
            ink = window < 0.75
            if not ink.any():
                return box
            ys, xs = torch.where(ink)
            ny1, ny2 = int(ys.min()) , int(ys.max()) + 1
            nx1, nx2 = int(xs.min()), int(xs.max()) + 1
            if (ny2 - ny1) < 8 or (nx2 - nx1) < 8:
                break
            new_window = window[ny1:ny2, nx1:nx2]
            if new_window.numel() == window.numel():
                break
            window = new_window
            y1, x1 = y1 + ny1, x1 + nx1
            y2, x2 = y1 + window.shape[0], x1 + window.shape[1]
        # re-add margin, clamp
        y1 = max(0, y1 - margin); x1 = max(0, x1 - margin)
        y2 = min(height, y2 + margin); x2 = min(width, x2 + margin)
        return torch.tensor([x1 / width, y1 / height,
                             x2 / width, y2 / height], device=box.device)

    _CROP_TRANSFORM = None

    @classmethod
    def production_crop_transform(cls, crop_bgr: "np.ndarray") -> "torch.Tensor":
        """The exact inference preprocessing the decoder was trained under:
        long-edge-512 pre-upscale -> CropWhite(pad=50) -> Resize(384) ->
        ToGray -> Normalize. Skipping CropWhite collapses exact to 0
        (design §21); skipping the pre-upscale changes CropWhite's RELATIVE
        pad (50px absolute) — a 130px-tall crop gets 2x50px margins and the
        ink fills only ~57% of the canvas instead of ~84%, shrinking line
        sizes to ~0.68x of the training distribution (operator audit §36)."""
        import cv2 as _cv2
        import numpy as _np
        from utils.MolNexTR.dataset import get_transforms
        from utils.MolNexTR.model import resize_small_image_to_long_edge

        if cls._CROP_TRANSFORM is None:
            cls._CROP_TRANSFORM = get_transforms(384, 384, augment=False)
        crop_bgr = resize_small_image_to_long_edge(crop_bgr)
        out = cls._CROP_TRANSFORM(image=crop_bgr, keypoints=[])
        return out["image"]

    @torch.inference_mode()
    def recognize_page(
        self,
        page_image_full: torch.Tensor,      # (1, 3, H, W) normalized like training
        page_tensor_384: torch.Tensor,      # (1, 3, 384, 384) same normalization
        expected_structure_types=None,
        tighten_with_ink: bool = False,
        markush_repair_recognizer=None,
    ) -> list[dict[str, Any]]:
        """Detect on the 384 view; re-crop each box from the full-res page."""
        boxes = self.detect(page_tensor_384)
        if tighten_with_ink:
            gray = page_image_full[0].mean(dim=0)  # (H, W), already normalized
            gray = (gray * 0.226) + 0.447          # back to [0,1] luminance-ish
            boxes = [torch.stack([self.tighten_box_with_ink(b, gray) for b in page_boxes])
                     if page_boxes.numel() else page_boxes
                     for page_boxes in boxes]
        results: list[dict[str, Any]] = []
        for page_index, page_boxes in enumerate(boxes):
            count = page_boxes.shape[0]
            if count == 0:
                results.append({"boxes": page_boxes, "predictions": []})
                continue
            height = page_image_full.shape[2]
            width = page_image_full.shape[3]
            crops_uint8 = []
            # Route every crop through the PRODUCTION preprocessing (CropWhite
            # -> Resize 384 -> ToGray -> Normalize): margin-bearing crops
            # without CropWhite collapse exact to 0 (design §21). Bilinear
            # re-scaling of the normalized full-page tensor is replaced by
            # cutting from the raw-resolution page through this transform.
            import numpy as _np

            page_np = page_image_full  # keep tensor API for box math above
            tensors = []
            for box in page_boxes:
                h, w = page_image_full.shape[2], page_image_full.shape[3]
                y1b = int(max(0.0, float(box[1])) * h)
                x1b = int(max(0.0, float(box[0])) * w)
                y2b = int(min(1.0, float(box[3])) * h)
                x2b = int(min(1.0, float(box[2])) * w)
                crop = page_image_full[0, :, y1b:y2b, x1b:x2b].cpu().numpy()
                # (3, h, w) -> denorm -> uint8 gray -> (h, w, 3) for albumentations
                crop = ((crop * 0.226) + 0.447)
                crop = _np.clip(crop * 255.0, 0, 255).astype(_np.uint8)
                if crop.ndim == 3:            # (C, h, w)
                    crop = crop[0]            # all 3 channels identical (gray)
                crop = _np.repeat(crop[:, :, None], 3, axis=2)   # (h, w, 3)
                crops_uint8.append(crop)
                tensors.append(self.production_crop_transform(crop))
            crop_batch = torch.stack(tensors).to(page_image_full.device)
            features, hiddens = self.encoder(crop_batch)
            # The multiscale attachment head hard-requires 3 square-grid levels
            # with (256, 512, 1024) channels — arbitrary-size crops violate
            # that. Pad each level's L up to the next perfect square so the
            # square-map contract holds; padded tail tokens are zeros and the
            # head's pointer/heatmap layers tolerate them (confidence-weighted).
            import math as _math

            flat_hiddens = []
            for h in hiddens:
                if h.dim() != 3:
                    h = h.flatten(1, 2)
                k, length, channels = h.shape
                side = _math.isqrt(length)
                if side * side != length:
                    side = side + 1
                    pad = side * side - length
                    h = torch.nn.functional.pad(h, (0, 0, 0, pad))
                flat_hiddens.append(h)
            types = None
            if expected_structure_types is not None:
                t = expected_structure_types[page_index]
                types = [t] * count
            decoded = self.decoder.decode(features, hiddens=flat_hiddens,
                                          expected_structure_types=types)
            if isinstance(decoded, list):
                decoded = reanchor_predictions(decoded, page_boxes)
                if markush_repair_recognizer is not None:
                    from utils.fusion.markush_repair import apply_markush_attachment_repair

                    decoded = [
                        apply_markush_attachment_repair(d, markush_repair_recognizer, c)
                        for d, c in zip(decoded, crops_uint8)
                    ]
            results.append({"boxes": page_boxes, "predictions": decoded})
        return results


class DomainRoutedDetector(nn.Module):
    """Per-domain detection: a tiny layout classifier picks head+threshold.

    The round-a14c7b0 finding: one head cannot serve both domains at once —
    the mg2 head (full-page single structure) achieves top box quality on
    crop-like layouts, the synth/mix heads cover multi-structure layouts,
    and the score threshold that maximizes detection F1 (0.90) starves the
    cascade downstream. Routing keeps each head's specialist quality while
    the classifier (a 2-conv net over the 384 view) picks which to apply,
    and each domain carries its own cascade threshold.

    Layout proxy for training the router: connected-component count of the
    ink mask after morphological cleanup — cheap, no labels needed beyond
    the pages we already have.
    """

    def __init__(self, encoder: nn.Module, heads: dict[str, LightDetectionHead],
                 fine_level: int = 2, thresholds: dict[str, float] | None = None,
                 nms_ious: dict[str, float] | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.fine_level = fine_level
        self.thresholds = thresholds or {"single": 0.90, "multi": 0.50}
        self.nms_ious = nms_ious or {"single": 0.30, "multi": 0.50}
        self.domain = None  # set by route() or manually

    @staticmethod
    def ink_component_count(bgr: "np.ndarray") -> int:
        import cv2 as _cv2

        gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
        binary = (gray < 200).astype("uint8") * 255
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (9, 9))
        binary = _cv2.morphologyEx(binary, _cv2.MORPH_CLOSE, kernel)
        binary = _cv2.morphologyEx(binary, _cv2.MORPH_OPEN, kernel)
        n, _ = _cv2.connectedComponents(binary)
        return max(0, n - 1)  # minus background

    def route(self, bgr: "np.ndarray") -> str:
        components = self.ink_component_count(bgr)
        return "multi" if components >= 3 else "single"

    @torch.inference_mode()
    def detect(self, page_tensor_384: torch.Tensor, domain: str | None = None):
        _, hiddens = self.encoder(page_tensor_384)
        fine = hidden_to_bchw(hiddens[self.fine_level])[0]
        domain = domain or self.domain or "single"
        head = self.heads[domain]
        ltrb, centerness = head(fine)
        batch, _, height, width = ltrb.shape
        points = _feature_points(height, width, ltrb.device)
        boxes = _decode_ltrb(ltrb.view(batch, 4, -1).permute(0, 2, 1), points)
        scores = centerness.view(batch, -1)
        threshold = self.thresholds.get(domain, 0.9)
        nms_iou = self.nms_ious.get(domain, 0.5)
        out: list[torch.Tensor] = []
        for i in range(batch):
            keep = scores[i] > threshold
            page_boxes = boxes[i][keep].clamp(0.0, 1.0)
            page_scores = scores[i][keep]
            if page_boxes.numel():
                idx = nms_boxes(page_boxes, page_scores, nms_iou)
                page_boxes = page_boxes[idx][:32]
            out.append(page_boxes)
        return out, domain


class RoutedCascadeRecognizer(nn.Module):
    """Cascade recognizer driven by DomainRoutedDetector."""

    def __init__(self, encoder: nn.Module, decoder: Any,
                 detector: DomainRoutedDetector) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.detector = detector

    @torch.inference_mode()
    def recognize_page(self, page_bgr: "np.ndarray", expected_types=None):
        import cv2 as _cv2
        import numpy as _np

        device = next(self.encoder.parameters()).device
        domain = self.detector.route(page_bgr)
        r = _cv2.resize(page_bgr, (384, 384), interpolation=_cv2.INTER_AREA).astype(_np.float32) / 255.0
        gray = (r.mean(axis=2, keepdims=True) - 0.447) / 0.226
        t384 = torch.from_numpy(_np.repeat(gray, 3, axis=2).transpose(2, 0, 1)).unsqueeze(0).to(device)
        boxes, domain = self.detector.detect(t384, domain=domain)
        page_boxes = boxes[0]
        count = int(page_boxes.shape[0])
        if count == 0:
            return {"domain": domain, "boxes": page_boxes.cpu(), "predictions": []}
        full = (page_bgr.astype(_np.float32) / 255.0).mean(axis=2, keepdims=True)
        full = (full - 0.447) / 0.226
        page_image_full = torch.from_numpy(
            _np.repeat(full, 3, axis=2).transpose(2, 0, 1)).unsqueeze(0).to(device)
        h, w = page_bgr.shape[:2]
        tensors = []
        for box in page_boxes:
            y1 = int(max(0.0, float(box[1])) * h); x1 = int(max(0.0, float(box[0])) * w)
            y2 = int(min(1.0, float(box[3])) * h); x2 = int(min(1.0, float(box[2])) * w)
            if y2 - y1 < 8 or x2 - x1 < 8:
                continue
            crop = page_image_full[0, :, y1:y2, x1:x2].cpu().numpy()
            crop = ((crop * 0.226) + 0.447)
            crop = _np.clip(crop * 255.0, 0, 255).astype(_np.uint8)
            if crop.ndim == 3:
                crop = crop[0]
            crop = _np.repeat(crop[:, :, None], 3, axis=2)
            from utils.fusion.fusion_model import CascadeStructureRecognizer
            tensors.append(CascadeStructureRecognizer.production_crop_transform(crop))
        if not tensors:
            return {"domain": domain, "boxes": page_boxes.cpu(), "predictions": []}
        crop_batch = torch.stack(tensors).to(device)
        features, hiddens = self.encoder(crop_batch)
        import math as _math

        flat_hiddens = []
        for hlevel in hiddens:
            if hlevel.dim() != 3:
                hlevel = hlevel.flatten(1, 2)
            k, length, channels = hlevel.shape
            side = _math.isqrt(length)
            if side * side != length:
                side += 1
                hlevel = torch.nn.functional.pad(hlevel, (0, 0, 0, side * side - length))
            flat_hiddens.append(hlevel)
        types = None
        if expected_types is not None:
            types = [expected_types] * count if isinstance(expected_types, str) else list(expected_types) * 1
        decoded = self.decoder.decode(features, hiddens=flat_hiddens,
                                      expected_structure_types=types)
        if isinstance(decoded, list):
            decoded = reanchor_predictions(decoded, page_boxes)
        return {"domain": domain, "boxes": page_boxes.cpu(), "predictions": decoded}

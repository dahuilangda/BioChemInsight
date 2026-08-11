"""Runtime inference for the wavy-bond Mask R-CNN detector (Phase 1/2/3).

A lazy-loaded detector that, given a fragment crop image, returns the detected
wavy attachment regions (mask + bbox + confidence + edge side). Used by:
- Phase 2: image inpaint of the wavy edge band before MolNexTR re-decode (kills
  the ghost carbon). We use the detected BBOX to localize the edge, then inpaint
  the whole foreground-ink band on that edge (the wavy zigzag + its connector).
  This is more robust than the raw predicted MASK, whose synthetic->real pixel
  precision is poor (domain gap).
- Phase 3: authoritative attachment evidence to synthesize a dummy atom and
  unblock the assembly pipeline.

Complete rows are never passed through (zero overhead); a single config flag
disables the detector.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

_DEFAULT_CHECKPOINT = "experiments/wavy_seg/wavy_maskrcnn_best.pth"


def build_wavy_maskrcnn(num_classes: int = 2, pretrained_backbone: bool = True) -> MaskRCNN:
    """Mask R-CNN for attachment-point detection. resnet50+FPN backbone.

    num_classes=2: legacy wavy-only detector (bg + wavy).
    num_classes=5: multi-class detector (bg + wavy + rgroup + asterisk + dashed)
    — the DECIMER+MolNexTR fusion detector. Kept self-contained (no training/
    import) so it works in the runtime image where training/ is absent.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    return MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=384,
        max_size=384,
    )


@dataclass(frozen=True)
class AttachmentDetection:
    mask: np.ndarray          # (H,W) bool, attachment pixels in image coords
    bbox: tuple[int, int, int, int]  # (x1,y1,x2,y2) in image coords
    confidence: float
    side: str                 # "left"|"right"|"top"|"bottom"
    cx: float                 # normalized center x
    cy: float                 # normalized center y
    class_name: str = "wavy"  # "wavy"|"rgroup"|"asterisk"|"dashed" (multi-class)


def _foreground_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fixed = (gray < 235).astype(np.uint8) * 255
    fg = cv2.bitwise_or(otsu, fixed)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _edge_band_slice(image_shape, bbox, side, band_fraction=0.30):
    """Row/col slices of an edge band on the image, sized to the structure."""
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    bw = max(6, int(round(width * band_fraction)))
    bh = max(6, int(round(height * band_fraction)))
    h, w = image_shape[:2]
    if side == "left":
        return slice(y1, min(y2 + bh // 2, h)), slice(max(0, x1 - 2), min(x2, x1 + bw))
    if side == "right":
        return slice(y1, min(y2 + bh // 2, h)), slice(max(x1, x2 - bw), min(w, x2 + 2))
    if side == "top":
        return slice(max(0, y1 - 2), min(y2, y1 + bh)), slice(x1, min(x2 + bw // 2, w))
    return slice(max(y1, y2 - bh), min(h, y2 + 2)), slice(x1, min(x2 + bw // 2, w))


class AttachmentDetector:
    """Lazy Mask R-CNN wavy-bond detector. Thread-safe singleton per checkpoint."""

    _lock = threading.Lock()

    def __init__(self, checkpoint_path: str | Path = _DEFAULT_CHECKPOINT, device: torch.device | None = None, num_classes: int = 2):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = int(num_classes)
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"attachment detector checkpoint not found: {self.checkpoint_path}")
            from torchvision.transforms import functional as TF
            self._TF = TF
            model = build_wavy_maskrcnn(num_classes=self.num_classes, pretrained_backbone=False)
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.to(self.device).eval()
            self._model = model

    def detect(self, image_rgb: np.ndarray, *, score_threshold: float = 0.05) -> list[AttachmentDetection]:
        """Detect wavy attachment bonds in an RGB image. Returns detections
        sorted by confidence (highest first). Empty list if none."""
        if image_rgb is None or image_rgb.size == 0:
            return []
        self._ensure_loaded()
        from PIL import Image as PILImage
        h, w = image_rgb.shape[:2]
        pil = PILImage.fromarray(image_rgb.astype(np.uint8)).resize((384, 384), PILImage.LANCZOS)
        img_t = self._TF.to_tensor(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(img_t)[0]
        scores = out["scores"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        labels = out["labels"].cpu().numpy()
        keep = scores >= score_threshold
        dets: list[AttachmentDetection] = []
        for i in np.where(keep)[0]:
            m = masks[i] > 0.5
            if m.sum() < 3:
                continue
            m_full = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            x1, y1, x2, y2 = boxes[i]
            sx, sy = w / 384.0, h / 384.0
            bx_full = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            cx_n = float((x1 + x2) / 2.0 / 384.0)
            cy_n = float((y1 + y2) / 2.0 / 384.0)
            side = self._edge_side(cx_n, cy_n)
            class_name = self._label_to_class(int(labels[i]))
            dets.append(AttachmentDetection(
                mask=m_full, bbox=bx_full, confidence=float(scores[i]),
                side=side, cx=cx_n, cy=cy_n, class_name=class_name,
            ))
        dets.sort(key=lambda d: -d.confidence)
        return dets

    @staticmethod
    def _label_to_class(label: int) -> str:
        # COCO category ids: 1=wavy, 2=rgroup, 3=asterisk, 4=dashed.
        # Legacy 2-class model only emits label 1 (=wavy).
        return {1: "wavy", 2: "rgroup", 3: "asterisk", 4: "dashed"}.get(label, "wavy")

    def inpaint_wavy_edge_band(
        self, image_rgb: np.ndarray, detection: AttachmentDetection, *, band_fraction: float = 0.30
    ) -> np.ndarray:
        """Inpaint the foreground ink in the detected wavy edge band.

        Uses the detection's edge SIDE (robust from the bbox center) rather than
        the raw predicted MASK (whose pixel precision is unreliable across the
        synthetic->real gap). Inpaints all foreground ink in that edge band so
        the wavy zigzag + its connector vanish before re-decode. Returns a new
        image; the input is untouched.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        bbox = _foreground_bbox(gray)
        if bbox is None:
            return image_rgb
        rs, cs = _edge_band_slice(image_rgb.shape, bbox, detection.side, band_fraction)
        out = image_rgb.copy()
        band = out[rs, cs]
        if band.size == 0:
            return image_rgb
        band_gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY) if band.ndim == 3 else band
        ink = ((band_gray < 160).astype(np.uint8)) * 255
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
        if int(ink.sum()) > 0:
            out[rs, cs] = cv2.inpaint(band, ink, 3, cv2.INPAINT_TELEA)
        return out

    @staticmethod
    def _edge_side(cx: float, cy: float) -> str:
        candidates = (("left", cx), ("right", 1 - cx), ("top", cy), ("bottom", 1 - cy))
        return min(candidates, key=lambda item: item[1])[0]

    def inpaint_wavy_mask_pixels(
        self,
        image_rgb: np.ndarray,
        detection: AttachmentDetection,
        *,
        dilate: int = 2,
    ) -> tuple[np.ndarray, bool]:
        """Erase only the detector's predicted wavy pixels, keeping the connector.

        Returns (out_image, changed); the input is untouched.
        """
        mask = getattr(detection, "mask", None)
        if not isinstance(mask, np.ndarray) or not mask.any():
            return image_rgb, False
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        h, w = image_rgb.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        ink = mask.astype(np.uint8) * 255
        if int(dilate) > 0:
            ink = cv2.dilate(
                ink,
                np.ones((2 * int(dilate) + 1, 2 * int(dilate) + 1), np.uint8),
                iterations=1,
            )
        if not ink.any():
            return image_rgb, False
        out = cv2.inpaint(image_rgb, ink, 3, cv2.INPAINT_TELEA)
        return out, True


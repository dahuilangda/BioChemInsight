"""
PyTorch-based molecule structure segmentation, replacing DECIMER's TensorFlow Mask R-CNN.

Provides the same interface as decimer_segmentation:
    - get_expanded_masks(image) -> np.array
    - apply_masks(image, masks) -> (segments, bboxes)

Uses torchvision Mask R-CNN with ResNet-101 + FPN backbone,
loaded from converted PyTorch weights (mask_rcnn_molecule.pth).
"""

import os
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import Conv2dNormActivation
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.morphology import dilation, erosion

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_device = None

MEAN_PIXEL = np.array([123.7, 116.8, 103.9], dtype=np.float32)  # RGB
MIN_DIM = 800
MAX_DIM = 1024
DET_CONFIDENCE = 0.3
DET_MASK_IOU_THRESHOLD = 0.85
DET_MASK_COVERAGE_THRESHOLD = 0.9
DET_BBOX_CONTAINMENT_TOLERANCE = 0.02
DET_AGGREGATE_CHILD_CONTAINMENT_TOLERANCE = 0.04
DET_AGGREGATE_MIN_AREA_RATIO = 1.75
DET_AGGREGATE_MIN_CHILD_AREA = 400
DET_COMPONENT_MIN_AREA = 50
DET_COMPONENT_MIN_AREA_RATIO = 0.08
DET_COMPONENT_MIN_BBOX_AREA = 2000
DET_COMPONENT_MIN_BBOX_SIDE = 30
DET_AGGREGATE_MIN_CHILDREN = 2
DET_AGGREGATE_MIN_CHILD_UNION_RATIO = 0.45
DET_AGGREGATE_MIN_CHILD_SPAN_RATIO = 0.70
DET_FOREGROUND_SPLIT_MIN_AREA_RATIO = 0.45
DET_FOREGROUND_SPLIT_MIN_COMPONENTS = 2


class MatterportRPNHead(nn.Module):
    """RPN head matching the Matterport/DECIMER Mask R-CNN architecture."""

    def __init__(self, in_channels=256, hidden_channels=512, num_anchors=3):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                hidden_channels,
                kernel_size=3,
                norm_layer=None,
            )
        )
        self.cls_logits = nn.Conv2d(hidden_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            hidden = self.conv(feature)
            logits.append(self.cls_logits(hidden))
            bbox_reg.append(self.bbox_pred(hidden))
        return logits, bbox_reg


def _get_weights_path():
    """Find mask_rcnn_molecule.pth."""
    candidates = [
        # Same directory as this file
        os.path.join(os.path.dirname(__file__), "mask_rcnn_molecule.pth"),
        # models/ directory
        os.path.join(os.path.dirname(__file__), "..", "models", "mask_rcnn_molecule.pth"),
        "/app/models/mask_rcnn_molecule.pth",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "mask_rcnn_molecule.pth not found. Run scripts/convert_decimer_weights.py first."
    )


def _load_model():
    """Load the Mask R-CNN model (lazy, once)."""
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with same architecture as convert_decimer_weights.py
    backbone = resnet_fpn_backbone("resnet101", pretrained=False, trainable_layers=5)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )

    rpn_head = MatterportRPNHead()

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
    )

    weights_path = _get_weights_path()
    state_dict = torch.load(weights_path, map_location=_device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(_device)
    model.eval()
    _model = model
    return _model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _resize_image(image, min_dim=MIN_DIM, max_dim=MAX_DIM):
    """Resize image maintaining aspect ratio, pad to square.

    Args:
        image: np.array (H, W, 3), BGR from cv2.imread

    Returns:
        resized: np.array (max_dim, max_dim, 3) padded with zeros
        scale: float
        padding: (top, left)
    """
    h, w = image.shape[:2]
    scale = max(min_dim / min(h, w), max_dim / max(h, w))
    scale = min(scale, max_dim / max(h, w))  # don't exceed max_dim

    if scale != 1.0:
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    else:
        new_h, new_w = h, w

    # Ensure within bounds
    new_h = min(new_h, max_dim)
    new_w = min(new_w, max_dim)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    top = (max_dim - new_h) // 2
    left = (max_dim - new_w) // 2
    padded = np.zeros((max_dim, max_dim, 3), dtype=image.dtype)
    padded[top:top + new_h, left:left + new_w] = resized

    return padded, scale, (top, left)


def _mold_image(image_bgr):
    """Preprocess image for model input.

    Args:
        image_bgr: np.array (H, W, 3) BGR

    Returns:
        tensor: torch.Tensor (1, 3, H, W) normalized
        meta: dict with original shape, scale, padding for unmolding
    """
    original_shape = image_bgr.shape[:2]

    # Convert BGR -> RGB
    image_rgb = image_bgr[:, :, ::-1].copy()

    # Resize and pad
    padded, scale, (top, left) = _resize_image(image_rgb)

    # Subtract mean pixel
    molded = padded.astype(np.float32) - MEAN_PIXEL

    # HWC -> CHW, numpy -> tensor
    tensor = torch.from_numpy(molded.transpose(2, 0, 1)).float()

    meta = {
        "original_shape": original_shape,
        "scale": scale,
        "padding": (top, left),
        "padded_shape": padded.shape[:2],
    }

    return tensor.unsqueeze(0).to(_device), meta


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def _unmold_detections(boxes, scores, masks, meta):
    """Convert model output back to original image coordinates.

    Args:
        boxes: torch.Tensor (N, 4) in [x1, y1, x2, y2] format (padded image coords)
        scores: torch.Tensor (N,)
        masks: torch.Tensor (N, 1, H, W) or (N, H, W)
        meta: dict from _mold_image

    Returns:
        final_masks: np.array (H_orig, W_orig, N) bool
        final_bboxes: np.array (N, 4) [y1, x1, y2, x2]
        final_scores: np.array (N,)
    """
    if boxes.numel() == 0:
        return np.zeros((*meta["original_shape"], 0), dtype=bool), \
               np.zeros((0, 4), dtype=np.int32), \
               np.zeros((0,), dtype=np.float32)

    orig_h, orig_w = meta["original_shape"]
    top, left = meta["padding"]
    scale = meta["scale"]

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    if masks.dim() == 4:
        masks_np = masks[:, 0].cpu().numpy()  # (N, H, W)
    else:
        masks_np = masks.cpu().numpy()

    n = boxes_np.shape[0]
    final_masks = np.zeros((orig_h, orig_w, n), dtype=bool)
    final_bboxes = np.zeros((n, 4), dtype=np.int32)

    for i in range(n):
        # Unpad bbox
        x1, y1, x2, y2 = boxes_np[i]
        x1 = max(0, (x1 - left) / scale)
        y1 = max(0, (y1 - top) / scale)
        x2 = min(orig_w, (x2 - left) / scale)
        y2 = min(orig_h, (y2 - top) / scale)

        final_bboxes[i] = [int(y1), int(x1), int(y2), int(x2)]

        # Resize mask to original image coordinates
        mask = masks_np[i]
        # Remove padding
        pad_h, pad_w = meta["padded_shape"]
        mask = mask[top:top + pad_h, left:left + pad_w]
        if mask.size > 0:
            mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
            final_masks[:, :, i] = mask > 0.5

    return final_masks, final_bboxes, scores_np


# ---------------------------------------------------------------------------
# Mask expansion (ported from DECIMER segmentation)
# ---------------------------------------------------------------------------

def _binarize_image(image_array, threshold=0.72):
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    grayscale = rgb2gray(image_rgb)
    return grayscale > threshold


def _determine_depiction_size_with_buffer(bboxes):
    bboxes_array = np.asarray(bboxes)
    heights = bboxes_array[:, 2] - bboxes_array[:, 0]
    widths = bboxes_array[:, 3] - bboxes_array[:, 1]
    return int(1.1 * np.max(heights)), int(1.1 * np.max(widths))


def _detect_horizontal_and_vertical_lines(image, max_depiction_size):
    binarized = (~image).astype(np.uint8) * 255
    structure_height, structure_width = max_depiction_size
    structure_height = max(1, int(structure_height))
    structure_width = max(1, int(structure_width))

    horizontal_kernel = np.ones((1, structure_width), dtype=np.uint8)
    vertical_kernel = np.ones((structure_height, 1), dtype=np.uint8)
    horizontal_mask = cv2.morphologyEx(
        binarized, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    ) == 255
    vertical_mask = cv2.morphologyEx(
        binarized, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    ) == 255
    return horizontal_mask | vertical_mask


def _find_equidistant_points(x1, y1, x2, y2, num_points=5):
    points = []
    for idx in range(num_points + 1):
        t = idx / num_points
        points.append((x1 * (1 - t) + x2 * t, y1 * (1 - t) + y2 * t))
    return points


def _detect_lines(image, max_depiction_size, segmentation_mask):
    image_uint8 = (~image).astype(np.uint8) * 255
    structure_context = dilation(segmentation_mask, footprint=np.ones((9, 9), dtype=bool))
    lines = cv2.HoughLinesP(
        image_uint8,
        1,
        np.pi / 180,
        threshold=5,
        minLineLength=max(1, int(max(max_depiction_size) / 4)),
        maxLineGap=10,
    )
    exclusion_mask = np.zeros_like(image_uint8, dtype=np.uint8)
    if lines is None:
        return exclusion_mask.astype(bool)

    height, width = segmentation_mask.shape
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points_in_structure = False
        for x, y in _find_equidistant_points(x1, y1, x2, y2, num_points=7):
            px = min(width - 1, max(0, int(x)))
            py = min(height - 1, max(0, int(y)))
            if structure_context[py, px]:
                points_in_structure = True
                break
        if not points_in_structure:
            cv2.line(exclusion_mask, (x1, y1), (x2, y2), 255, 2)
    return exclusion_mask.astype(bool)


def _get_seeds(image_array, mask_array, exclusion_mask):
    mask_y_values, mask_x_values = np.where(mask_array)
    if len(mask_y_values) == 0:
        return []

    mask_y_diff = mask_y_values.max() - mask_y_values.min()
    mask_x_diff = mask_x_values.max() - mask_x_values.min()
    x_min_limit = mask_x_values.min() + mask_x_diff / 10
    x_max_limit = mask_x_values.max() - mask_x_diff / 10
    y_min_limit = mask_y_values.min() + mask_y_diff / 10
    y_max_limit = mask_y_values.max() - mask_y_diff / 10

    mask_coordinates = set(zip(mask_y_values, mask_x_values))
    image_y_values, image_x_values = np.where(~image_array)
    image_coordinates = set(zip(image_y_values, image_x_values))

    seed_pixels = []
    for y_coord, x_coord in mask_coordinates & image_coordinates:
        if x_coord < x_min_limit or x_coord > x_max_limit:
            continue
        if y_coord < y_min_limit or y_coord > y_max_limit:
            continue
        if exclusion_mask[y_coord, x_coord]:
            continue
        seed_pixels.append((x_coord, y_coord))
    return seed_pixels


def _expand_mask(image_array, seed_pixels):
    if not seed_pixels:
        return np.zeros_like(image_array, dtype=bool)

    labeled_array, _ = ndimage.label(~image_array)
    expanded = np.zeros_like(image_array, dtype=bool)
    processed_labels = set()
    for x, y in seed_pixels:
        if expanded[y, x]:
            continue
        label_value = labeled_array[y, x]
        if label_value > 0 and label_value not in processed_labels:
            expanded[labeled_array == label_value] = True
            processed_labels.add(label_value)
    return expanded


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1], dtype=np.int32)


def _mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def _mask_coverage(mask_a, mask_b):
    """How much of mask_a is covered by mask_b."""
    area_a = mask_a.sum()
    if area_a == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(area_a)


def _bbox_contains(container, inner, tolerance=DET_BBOX_CONTAINMENT_TOLERANCE):
    """Return True if container almost fully contains inner."""
    container = np.asarray(container, dtype=np.float32)
    inner = np.asarray(inner, dtype=np.float32)
    inner_h = max(1.0, inner[2] - inner[0])
    inner_w = max(1.0, inner[3] - inner[1])
    return (
        container[0] <= inner[0] + tolerance * inner_h
        and container[1] <= inner[1] + tolerance * inner_w
        and container[2] >= inner[2] - tolerance * inner_h
        and container[3] >= inner[3] - tolerance * inner_w
    )


def _bbox_area(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _split_connected_masks(mask):
    labeled, num_labels = ndimage.label(mask)
    if num_labels <= 1:
        return [mask]

    raw_components = []
    areas = []
    for label_idx in range(1, num_labels + 1):
        component = labeled == label_idx
        if component.any():
            raw_components.append(component)
            areas.append(int(component.sum()))
    if not raw_components:
        return []

    largest_area = max(areas)
    min_area = max(DET_COMPONENT_MIN_AREA, int(largest_area * DET_COMPONENT_MIN_AREA_RATIO))
    components = [
        component
        for component, area in zip(raw_components, areas)
        if area >= min_area
    ]
    if components:
        return components
    return [raw_components[int(np.argmax(areas))]]


def _bbox_union(bboxes):
    if not bboxes:
        return None
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return np.array(
        [
            bboxes[:, 0].min(),
            bboxes[:, 1].min(),
            bboxes[:, 2].max(),
            bboxes[:, 3].max(),
        ],
        dtype=np.float32,
    )


def _bbox_span_ratio(inner_union, outer):
    if inner_union is None:
        return 0.0
    inner_area = _bbox_area(inner_union)
    outer_area = _bbox_area(outer)
    if outer_area <= 0:
        return 0.0
    return float(inner_area) / float(outer_area)


def _bbox_is_substantial(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    height = float(bbox[2] - bbox[0])
    width = float(bbox[3] - bbox[1])
    return (
        height >= DET_COMPONENT_MIN_BBOX_SIDE
        and width >= DET_COMPONENT_MIN_BBOX_SIDE
        and height * width >= DET_COMPONENT_MIN_BBOX_AREA
    )


def _mask_union_area(masks):
    if not masks:
        return 0
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    return int(union.sum())


def _remove_long_ruling_lines(foreground):
    height, width = foreground.shape
    horizontal_kernel = np.ones((1, max(20, int(width * 0.55))), dtype=np.uint8)
    vertical_kernel = np.ones((max(20, int(height * 0.18)), 1), dtype=np.uint8)
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    ruling_lines = cv2.bitwise_or(horizontal, vertical)
    return cv2.bitwise_and(foreground, cv2.bitwise_not(ruling_lines))


def _split_component_by_foreground_regions(image_bgr, component_mask):
    bbox = _mask_bbox(component_mask)
    if bbox is None:
        return [component_mask]
    y1, x1, y2, x2 = [int(v) for v in bbox]
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return [component_mask]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = (gray < 200).astype(np.uint8) * 255
    foreground = _remove_long_ruling_lines(foreground)
    if not np.any(foreground):
        return [component_mask]

    height, width = foreground.shape
    kernel_size = max(9, min(31, int(round(min(height, width) * 0.05))))
    dilated = cv2.dilate(foreground, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
    if num_labels <= 2:
        return [component_mask]

    component_mask_crop = component_mask[y1:y2, x1:x2]
    candidates = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label_idx]]
        if area < DET_COMPONENT_MIN_AREA:
            continue
        region = labels == label_idx
        if not np.any(region & component_mask_crop):
            continue
        candidates.append((area, x, y, w, h, region))
    if len(candidates) < DET_FOREGROUND_SPLIT_MIN_COMPONENTS:
        return [component_mask]

    largest_area = max(item[0] for item in candidates)
    min_area = max(DET_COMPONENT_MIN_AREA, int(largest_area * DET_FOREGROUND_SPLIT_MIN_AREA_RATIO))
    selected = [item for item in candidates if item[0] >= min_area]
    if len(selected) < DET_FOREGROUND_SPLIT_MIN_COMPONENTS:
        return [component_mask]

    split_masks = []
    for _, x, y, w, h, region in selected:
        split = np.zeros_like(component_mask, dtype=bool)
        region_crop = region & component_mask_crop
        if not np.any(region_crop):
            region_crop = np.zeros_like(component_mask_crop, dtype=bool)
            region_crop[y:y + h, x:x + w] = True
            region_crop &= component_mask_crop
        split[y1:y2, x1:x2] = region_crop
        split_bbox = _mask_bbox(split)
        if split_bbox is not None and _bbox_is_substantial(split_bbox):
            split_masks.append(split)
    return split_masks or [component_mask]


def _suppress_aggregate_mask_candidates(mask_candidates):
    """Drop broad model candidates that only aggregate more specific candidates."""
    if len(mask_candidates) <= 1:
        return mask_candidates

    suppress_indices = set()
    for idx, candidate in enumerate(mask_candidates):
        candidate_area = _bbox_area(candidate["bbox"])
        if candidate_area <= 0:
            continue
        contained_children = []
        for other_idx, other in enumerate(mask_candidates):
            if other_idx == idx:
                continue
            other_area = _bbox_area(other["bbox"])
            if other_area < DET_AGGREGATE_MIN_CHILD_AREA:
                continue
            if candidate_area < other_area * DET_AGGREGATE_MIN_AREA_RATIO:
                continue
            if not _bbox_contains(
                candidate["bbox"],
                other["bbox"],
                tolerance=DET_AGGREGATE_CHILD_CONTAINMENT_TOLERANCE,
            ):
                continue
            if _mask_iou(candidate["mask"], other["mask"]) >= DET_MASK_IOU_THRESHOLD:
                continue
            contained_children.append(other)

        if len(contained_children) >= DET_AGGREGATE_MIN_CHILDREN:
            child_union_bbox = _bbox_union([child["bbox"] for child in contained_children])
            child_span_ratio = _bbox_span_ratio(child_union_bbox, candidate["bbox"])
            child_union_area = _mask_union_area([child["mask"] for child in contained_children])
            candidate_mask_area = max(1, int(candidate["mask"].sum()))
            child_union_ratio = float(child_union_area) / float(candidate_mask_area)
            if (
                child_span_ratio < DET_AGGREGATE_MIN_CHILD_SPAN_RATIO
                and child_union_ratio < DET_AGGREGATE_MIN_CHILD_UNION_RATIO
            ):
                continue
            suppress_indices.add(idx)
            continue
        if contained_children:
            child = contained_children[0]
            child_score = float(child.get("score", 0.0))
            candidate_score = float(candidate.get("score", 0.0))
            if child_score >= candidate_score * 0.95:
                suppress_indices.add(idx)

    if not suppress_indices:
        return mask_candidates
    return [candidate for idx, candidate in enumerate(mask_candidates) if idx not in suppress_indices]


def _dedupe_mask_candidates(mask_candidates):
    """Remove duplicate or nested masks while keeping the strongest candidates."""
    if not mask_candidates:
        return []

    mask_candidates = _suppress_aggregate_mask_candidates(mask_candidates)
    if not mask_candidates:
        return []

    ordered = sorted(
        mask_candidates,
        key=lambda item: (-float(item["score"]), int(item["mask"].sum())),
    )
    accepted = []
    seen_hashes = set()
    for candidate in ordered:
        mask_hash = hash(candidate["mask"].tobytes())
        if mask_hash in seen_hashes:
            continue
        if any(
            _mask_coverage(candidate["mask"], kept["mask"]) >= DET_MASK_COVERAGE_THRESHOLD
            or _mask_iou(candidate["mask"], kept["mask"]) >= DET_MASK_IOU_THRESHOLD
            or _bbox_contains(kept["bbox"], candidate["bbox"])
            for kept in accepted
        ):
            continue
        accepted.append(candidate)
        seen_hashes.add(mask_hash)

    accepted.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    return accepted


def _complete_structure_mask(image_bgr, masks, bboxes, scores=None):
    if masks.size == 0 or masks.shape[2] == 0:
        return masks, bboxes

    binarized = _binarize_image(image_bgr, threshold=0.72)
    blur_factor = max(2, int(image_bgr.shape[1] / 185))
    kernel = np.ones((blur_factor, blur_factor), dtype=np.uint8)
    eroded_image = erosion(binarized, footprint=kernel)
    max_depiction_size = _determine_depiction_size_with_buffer(bboxes)

    horizontal_vertical_lines = _detect_horizontal_and_vertical_lines(
        eroded_image, max_depiction_size
    )
    segmentation_mask = masks.any(axis=2)
    hough_lines = _detect_lines(
        binarized,
        max_depiction_size,
        segmentation_mask=segmentation_mask,
    )
    exclusion_mask = horizontal_vertical_lines | dilation(hough_lines, footprint=kernel)
    image_with_exclusion = ~((~eroded_image) & (~exclusion_mask))

    expanded_candidates = []
    for idx in range(masks.shape[2]):
        mask = masks[:, :, idx]
        score = 0.0 if scores is None else float(scores[idx])
        seeds = _get_seeds(image_with_exclusion, mask, exclusion_mask)
        expanded = _expand_mask(image_with_exclusion, seeds)
        candidate_mask = expanded if expanded.any() else mask
        for component in _split_connected_masks(candidate_mask):
            for split_component in _split_component_by_foreground_regions(image_bgr, component):
                bbox = _mask_bbox(split_component)
                if bbox is None:
                    continue
                expanded_candidates.append(
                    {
                        "mask": split_component,
                        "bbox": bbox,
                        "score": score,
                    }
                )

    unique_candidates = _dedupe_mask_candidates(expanded_candidates)
    if not unique_candidates:
        return np.empty((image_bgr.shape[0], image_bgr.shape[1], 0), dtype=bool), np.empty((0, 4), dtype=np.int32)

    expanded_stack = np.stack([item["mask"] for item in unique_candidates], axis=-1)
    expanded_bboxes = np.stack([item["bbox"] for item in unique_candidates], axis=0).astype(np.int32)
    return expanded_stack, expanded_bboxes


# ---------------------------------------------------------------------------
# Public API (same interface as decimer_segmentation)
# ---------------------------------------------------------------------------

def get_expanded_masks(image: np.array) -> np.array:
    """Detect molecular structures in image and return expanded masks.

    Args:
        image: np.array (H, W, 3), BGR format (from cv2.imread)

    Returns:
        np.array of shape (H, W, N) where N is number of detected structures.
        Each slice [:, :, i] is a boolean mask for one structure.
    """
    model = _load_model()

    with torch.no_grad():
        input_tensor, meta = _mold_image(image)
        predictions = model(input_tensor)[0]

    boxes = predictions["boxes"]
    scores = predictions["scores"]
    raw_masks = predictions["masks"]

    # Filter by confidence
    keep = scores > DET_CONFIDENCE
    boxes = boxes[keep]
    scores = scores[keep]
    raw_masks = raw_masks[keep]

    # Unmold to original coordinates
    masks, bboxes, scores = _unmold_detections(boxes, scores, raw_masks, meta)

    if masks.shape[2] == 0:
        return masks

    # Expand masks to capture complete structures
    masks, bboxes = _complete_structure_mask(image, masks, bboxes, scores=scores)

    return masks


def apply_masks(image: np.array, masks: np.array):
    """Apply masks to image, returning cropped segments and bounding boxes.

    Args:
        image: np.array (H, W, 3), BGR format
        masks: np.array (H, W, N), boolean masks from get_expanded_masks

    Returns:
        segments: list of np.array, each RGBA cropped image (H_seg, W_seg, 4)
        bboxes: list of tuples (y0, x0, y1, x1) in pixel coordinates
    """
    segments = []
    bboxes_out = []

    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        if not mask.any():
            continue

        # Find bbox of mask
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # Add small margin
        margin = 2
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(image.shape[0], y1 + margin)
        x1 = min(image.shape[1], x1 + margin)

        # Crop the original page by bbox.  The previous alpha-mask path used a
        # normal binary threshold as alpha, which made black molecule strokes
        # transparent and saved nearly blank dotted crops.
        crop = image[y0:y1, x0:x1].copy()

        segments.append(crop)
        bboxes_out.append((y0, x0, y1, x1))

    return segments, bboxes_out

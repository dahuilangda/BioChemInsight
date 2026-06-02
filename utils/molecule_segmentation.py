import os
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import Conv2dNormActivation
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.morphology import erosion

try:
    import constants as project_constants
except ImportError:  # pragma: no cover - optional local configuration
    project_constants = None

_model = None
_device = None

MEAN_PIXEL = np.array([123.7, 116.8, 103.9], dtype=np.float32)
MIN_DIM = 800
MAX_DIM = 1024
DET_CONFIDENCE = float(getattr(project_constants, "STRUCTURE_DETECTION_CONFIDENCE", 0.05))
DET_MODEL_SCORE_THRESH = float(getattr(project_constants, "STRUCTURE_DET_MODEL_SCORE_THRESH", DET_CONFIDENCE) or DET_CONFIDENCE)
DET_MODEL_NMS_THRESH = float(getattr(project_constants, "STRUCTURE_DET_MODEL_NMS_THRESH", 0.50) or 0.50)
MATTERPORT_BOX_STD = (0.1, 0.1, 0.2, 0.2)
TORCHVISION_BOX_CODER_WEIGHTS = tuple(1.0 / value for value in MATTERPORT_BOX_STD)
DET_MASK_IOU_THRESHOLD = 0.85
DET_MASK_COVERAGE_THRESHOLD = 0.9
DET_BBOX_OVERLAP_COVERAGE_THRESHOLD = float(
    getattr(project_constants, "STRUCTURE_DET_BBOX_OVERLAP_COVERAGE_THRESHOLD", 0.78) or 0.78
)
DET_BBOX_CONTAINMENT_TOLERANCE = 0.02
DET_AGGREGATE_CHILD_CONTAINMENT_TOLERANCE = 0.04
DET_AGGREGATE_MIN_AREA_RATIO = 1.75
DET_AGGREGATE_MIN_CHILD_AREA = 400
DET_COMPONENT_MIN_AREA = 50
DET_COMPONENT_MIN_AREA_RATIO = 0.08
DET_COMPONENT_MIN_BBOX_AREA = 2000
DET_COMPONENT_MIN_BBOX_SIDE = 30
DET_RAW_MIN_BBOX_SIDE = int(getattr(project_constants, "STRUCTURE_DET_RAW_MIN_BBOX_SIDE", 18) or 18)
DET_RAW_MIN_BBOX_AREA = int(getattr(project_constants, "STRUCTURE_DET_RAW_MIN_BBOX_AREA", 600) or 600)
DET_RAW_MAX_BBOX_PAGE_FRACTION = float(
    getattr(project_constants, "STRUCTURE_DET_RAW_MAX_BBOX_PAGE_FRACTION", 0.92) or 0.92
)
DET_RAW_NMS_IOU_THRESHOLD = float(getattr(project_constants, "STRUCTURE_DET_RAW_NMS_IOU_THRESHOLD", 0.55) or 0.55)
DET_AGGREGATE_MIN_CHILDREN = 2
DET_AGGREGATE_MIN_CHILD_UNION_RATIO = 0.45
DET_AGGREGATE_MIN_CHILD_SPAN_RATIO = 0.70
DET_FOREGROUND_SPLIT_MIN_AREA_RATIO = 0.45
DET_FOREGROUND_SPLIT_MIN_COMPONENTS = 2
DET_MAX_LINE_CONTEXT_KERNEL = int(getattr(project_constants, "STRUCTURE_DET_MAX_LINE_CONTEXT_KERNEL", 101) or 101)
DET_MAX_RULING_KERNEL = int(getattr(project_constants, "STRUCTURE_DET_MAX_RULING_KERNEL", 401) or 401)
DET_MAX_EXPANDED_CANDIDATES = int(getattr(project_constants, "STRUCTURE_DET_MAX_EXPANDED_CANDIDATES", 80) or 80)
DET_MAX_RAW_DETECTIONS = int(getattr(project_constants, "STRUCTURE_DET_MAX_RAW_DETECTIONS", 60) or 60)
DET_MODEL_DETECTIONS_PER_IMG = int(
    getattr(project_constants, "STRUCTURE_DET_MODEL_DETECTIONS_PER_IMG", max(100, DET_MAX_RAW_DETECTIONS * 2))
    or max(100, DET_MAX_RAW_DETECTIONS * 2)
)
DET_MAX_SEED_PIXELS = int(getattr(project_constants, "STRUCTURE_DET_MAX_SEED_PIXELS", 12000) or 12000)
DET_MAX_MASK_PAGE_FRACTION = float(getattr(project_constants, "STRUCTURE_DET_MAX_MASK_PAGE_FRACTION", 0.72) or 0.72)
DET_MAX_LOCAL_EXPANSION_PIXELS = int(
    getattr(project_constants, "STRUCTURE_DET_MAX_LOCAL_EXPANSION_PIXELS", 900_000) or 900_000
)
DET_LOCAL_EXPANSION_PAD_RATIO = float(getattr(project_constants, "STRUCTURE_DET_LOCAL_EXPANSION_PAD_RATIO", 0.12) or 0.12)
DET_LOCAL_EXPANSION_MAX_PAD = int(getattr(project_constants, "STRUCTURE_DET_LOCAL_EXPANSION_MAX_PAD", 96) or 96)
DET_LOCAL_EXPANSION_MIN_PAD = int(getattr(project_constants, "STRUCTURE_DET_LOCAL_EXPANSION_MIN_PAD", 36) or 36)
DET_TILE_PASS_ENABLED = bool(getattr(project_constants, "STRUCTURE_DET_TILE_PASS_ENABLED", True))
DET_TILE_SIZE = int(getattr(project_constants, "STRUCTURE_DET_TILE_SIZE", 760) or 760)
DET_TILE_STRIDE_RATIO = float(getattr(project_constants, "STRUCTURE_DET_TILE_STRIDE_RATIO", 0.62) or 0.62)
DET_MAX_TILE_PASSES = int(getattr(project_constants, "STRUCTURE_DET_MAX_TILE_PASSES", 8) or 8)
DET_FINAL_MIN_BBOX_SIDE = int(getattr(project_constants, "STRUCTURE_DET_FINAL_MIN_BBOX_SIDE", 18) or 18)
DET_FINAL_MIN_BBOX_AREA = int(getattr(project_constants, "STRUCTURE_DET_FINAL_MIN_BBOX_AREA", 450) or 450)
DET_FINAL_MIN_INK_PIXELS = int(getattr(project_constants, "STRUCTURE_DET_FINAL_MIN_INK_PIXELS", 35) or 35)
DET_FINAL_MIN_INK_DENSITY = float(getattr(project_constants, "STRUCTURE_DET_FINAL_MIN_INK_DENSITY", 0.018) or 0.018)
DET_FINAL_MAX_RULING_LINE_RATIO = float(
    getattr(project_constants, "STRUCTURE_DET_FINAL_MAX_RULING_LINE_RATIO", 0.38) or 0.38
)
DET_LOCAL_FOREGROUND_LINK_PAD = int(getattr(project_constants, "STRUCTURE_DET_LOCAL_FOREGROUND_LINK_PAD", 76) or 76)


class MatterportRPNHead(nn.Module):
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


class MatterportBoxHead(nn.Module):
    def __init__(self, in_channels=256, resolution=7, representation_size=1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * resolution * resolution, representation_size)
        self.bn6 = nn.BatchNorm1d(representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.bn7 = nn.BatchNorm1d(representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.relu(self.bn7(self.fc7(x)))
        return x


class MatterportMaskHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x


def _get_weights_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "mask_rcnn_molecule.pth"),
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
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    box_head = MatterportBoxHead()
    mask_head = MatterportMaskHead()

    model = MaskRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        box_roi_pool=roi_pooler,
        mask_head=mask_head,
        mask_roi_pool=mask_roi_pooler,
        box_score_thresh=DET_MODEL_SCORE_THRESH,
        box_nms_thresh=DET_MODEL_NMS_THRESH,
        box_detections_per_img=DET_MODEL_DETECTIONS_PER_IMG,
    )
    model.transform = GeneralizedRCNNTransform(
        min_size=MAX_DIM,
        max_size=MAX_DIM,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size_divisible=1,
    )
    model.rpn.box_coder.weights = TORCHVISION_BOX_CODER_WEIGHTS
    model.roi_heads.box_coder.weights = TORCHVISION_BOX_CODER_WEIGHTS

    weights_path = _get_weights_path()
    state_dict = torch.load(weights_path, map_location=_device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(_device)
    model.eval()
    _model = model
    return _model


def _resize_image(image, min_dim=MIN_DIM, max_dim=MAX_DIM):
    h, w = image.shape[:2]
    scale = max(min_dim / min(h, w), max_dim / max(h, w))
    scale = min(scale, max_dim / max(h, w))

    if scale != 1.0:
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    else:
        new_h, new_w = h, w

    new_h = min(new_h, max_dim)
    new_w = min(new_w, max_dim)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (max_dim - new_h) // 2
    left = (max_dim - new_w) // 2
    padded = np.zeros((max_dim, max_dim, 3), dtype=image.dtype)
    padded[top:top + new_h, left:left + new_w] = resized

    return padded, scale, (top, left)


def _mold_image(image_bgr):
    original_shape = image_bgr.shape[:2]

    image_rgb = image_bgr[:, :, ::-1].copy()

    padded, scale, (top, left) = _resize_image(image_rgb)
    resized_h = min(int(round(original_shape[0] * scale)), MAX_DIM)
    resized_w = min(int(round(original_shape[1] * scale)), MAX_DIM)

    molded = padded.astype(np.float32) - MEAN_PIXEL

    tensor = torch.from_numpy(molded.transpose(2, 0, 1)).float()

    meta = {
        "original_shape": original_shape,
        "scale": scale,
        "padding": (top, left),
        "padded_shape": padded.shape[:2],
        "resized_shape": (resized_h, resized_w),
    }

    return tensor.unsqueeze(0).to(_device), meta


def _unmold_detections(boxes, scores, masks, meta):
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
        masks_np = masks[:, 0].cpu().numpy()
    else:
        masks_np = masks.cpu().numpy()

    n = boxes_np.shape[0]
    final_masks = np.zeros((orig_h, orig_w, n), dtype=bool)
    final_bboxes = np.zeros((n, 4), dtype=np.int32)

    for i in range(n):
        x1, y1, x2, y2 = boxes_np[i]
        x1 = max(0, (x1 - left) / scale)
        y1 = max(0, (y1 - top) / scale)
        x2 = min(orig_w, (x2 - left) / scale)
        y2 = min(orig_h, (y2 - top) / scale)

        final_bboxes[i] = [int(y1), int(x1), int(y2), int(x2)]

        mask = masks_np[i]
        resized_h, resized_w = meta["resized_shape"]
        mask = mask[top:top + resized_h, left:left + resized_w]
        if mask.size > 0:
            mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
            final_masks[:, :, i] = mask > 0.5

    return final_masks, final_bboxes, scores_np


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
    structure_height = min(structure_height, max(1, DET_MAX_RULING_KERNEL))
    structure_width = min(structure_width, max(1, DET_MAX_RULING_KERNEL))

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
    context_size = max(9, int(max(max_depiction_size) * 1.1))
    context_size = min(context_size, max(9, DET_MAX_LINE_CONTEXT_KERNEL))
    if context_size % 2 == 0:
        context_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (context_size, context_size))
    structure_context = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
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

    candidate_mask = mask_array & (~image_array) & (~exclusion_mask)
    if mask_x_diff > 0:
        candidate_mask[:, :max(0, int(x_min_limit))] = False
        candidate_mask[:, min(candidate_mask.shape[1], int(x_max_limit) + 1):] = False
    if mask_y_diff > 0:
        candidate_mask[:max(0, int(y_min_limit)), :] = False
        candidate_mask[min(candidate_mask.shape[0], int(y_max_limit) + 1):, :] = False

    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        return []
    if len(ys) > DET_MAX_SEED_PIXELS:
        step = max(1, len(ys) // DET_MAX_SEED_PIXELS)
        ys = ys[::step]
        xs = xs[::step]
    return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))


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


def _clip_bbox(bbox, height, width, pad=0):
    bbox = np.asarray(bbox, dtype=np.int32)
    y1, x1, y2, x2 = [int(v) for v in bbox]
    pad = max(0, int(pad or 0))
    return (
        max(0, y1 - pad),
        max(0, x1 - pad),
        min(int(height), y2 + pad),
        min(int(width), x2 + pad),
    )


def _bbox_within_crop(bbox, crop_y1, crop_x1, crop_y2, crop_x2):
    y1, x1, y2, x2 = [int(v) for v in bbox]
    return np.array(
        [
            max(0, y1 - crop_y1),
            max(0, x1 - crop_x1),
            min(crop_y2 - crop_y1, y2 - crop_y1),
            min(crop_x2 - crop_x1, x2 - crop_x1),
        ],
        dtype=np.int32,
    )


def _place_local_mask(local_mask, page_shape, crop_y1, crop_x1):
    page_mask = np.zeros(page_shape, dtype=bool)
    y2 = min(page_shape[0], crop_y1 + local_mask.shape[0])
    x2 = min(page_shape[1], crop_x1 + local_mask.shape[1])
    if y2 > crop_y1 and x2 > crop_x1:
        page_mask[crop_y1:y2, crop_x1:x2] = local_mask[:y2 - crop_y1, :x2 - crop_x1]
    return page_mask


def _mask_candidate_hash(mask, bbox):
    bbox = np.asarray(bbox, dtype=np.int32)
    y1, x1, y2, x2 = [int(v) for v in bbox]
    crop = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return hash((tuple(int(v) for v in bbox), crop.tobytes()))


def _mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def _mask_coverage(mask_a, mask_b):
    area_a = mask_a.sum()
    if area_a == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(area_a)


def _bbox_contains(container, inner, tolerance=DET_BBOX_CONTAINMENT_TOLERANCE):
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


def _bbox_iou(box_a, box_b):
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])
    intersection = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _bbox_overlap_coverage(box_a, box_b):
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])
    intersection = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    if intersection <= 0:
        return 0.0
    smaller = min(_bbox_area(a), _bbox_area(b))
    return float(intersection / smaller) if smaller > 0 else 0.0


def _raw_detection_quality_score(mask, bbox, score, page_area):
    bbox_area = _bbox_area(bbox)
    if bbox_area <= 0:
        return 0.0
    mask_area = max(0, int(mask.sum()))
    fill_ratio = min(1.0, float(mask_area) / float(bbox_area))
    size_penalty = 0.5 if bbox_area / max(1, page_area) > DET_MAX_MASK_PAGE_FRACTION else 1.0
    return float(score) * (0.35 + 0.65 * fill_ratio) * size_penalty


def _filter_raw_detections(masks, bboxes, scores, page_shape):
    if masks.size == 0 or masks.shape[2] == 0:
        return masks, bboxes, scores

    height, width = page_shape[:2]
    page_area = max(1, int(height) * int(width))
    candidates = []
    for idx in range(masks.shape[2]):
        bbox = np.asarray(bboxes[idx], dtype=np.int32)
        y1, x1, y2, x2 = [int(v) for v in bbox]
        bbox_h = max(0, y2 - y1)
        bbox_w = max(0, x2 - x1)
        bbox_area = bbox_h * bbox_w
        if bbox_h < DET_RAW_MIN_BBOX_SIDE or bbox_w < DET_RAW_MIN_BBOX_SIDE:
            continue
        if bbox_area < DET_RAW_MIN_BBOX_AREA:
            continue
        if bbox_area / page_area > DET_RAW_MAX_BBOX_PAGE_FRACTION:
            continue
        mask = masks[:, :, idx]
        if not mask.any():
            continue
        score = float(scores[idx]) if scores is not None and len(scores) > idx else 0.0
        quality = _raw_detection_quality_score(mask, bbox, score, page_area)
        candidates.append({
            "idx": idx,
            "bbox": bbox,
            "mask": mask,
            "score": score,
            "quality": quality,
        })
    if not candidates:
        return (
            np.empty((height, width, 0), dtype=bool),
            np.empty((0, 4), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    candidates.sort(key=lambda item: item["quality"], reverse=True)
    kept = []
    for candidate in candidates:
        if any(_bbox_iou(candidate["bbox"], kept_item["bbox"]) >= DET_RAW_NMS_IOU_THRESHOLD for kept_item in kept):
            continue
        if any(
            _bbox_contains(kept_item["bbox"], candidate["bbox"])
            and _mask_coverage(candidate["mask"], kept_item["mask"]) >= DET_MASK_COVERAGE_THRESHOLD
            for kept_item in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= DET_MAX_RAW_DETECTIONS:
            break

    kept.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    filtered_masks = np.stack([item["mask"] for item in kept], axis=-1).astype(bool)
    filtered_bboxes = np.stack([item["bbox"] for item in kept], axis=0).astype(np.int32)
    filtered_scores = np.asarray([item["score"] for item in kept], dtype=np.float32)
    return filtered_masks, filtered_bboxes, filtered_scores


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


def _large_connected_component_count(mask, min_area=DET_COMPONENT_MIN_AREA):
    labeled, num_labels = ndimage.label(mask)
    if num_labels <= 0:
        return 0
    count = 0
    for label_idx in range(1, num_labels + 1):
        if int((labeled == label_idx).sum()) >= min_area:
            count += 1
    return count


def _mask_bbox_fill_ratio(mask, bbox):
    mask_bbox = _mask_bbox(mask)
    if mask_bbox is None:
        return 0.0
    bbox_area = _bbox_area(bbox)
    if bbox_area <= 0:
        return 0.0
    return float(_bbox_area(mask_bbox)) / float(bbox_area)


def _remove_long_ruling_lines(foreground):
    height, width = foreground.shape
    horizontal_kernel = np.ones((1, max(20, int(width * 0.55))), dtype=np.uint8)
    vertical_kernel = np.ones((max(20, int(height * 0.18)), 1), dtype=np.uint8)
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    ruling_lines = cv2.bitwise_or(horizontal, vertical)
    return cv2.bitwise_and(foreground, cv2.bitwise_not(ruling_lines))


def _ruling_line_ratio(foreground):
    height, width = foreground.shape
    if height <= 0 or width <= 0 or not np.any(foreground):
        return 0.0
    horizontal_kernel = np.ones((1, max(8, int(width * 0.45))), dtype=np.uint8)
    vertical_kernel = np.ones((max(8, int(height * 0.45)), 1), dtype=np.uint8)
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    line_pixels = int(cv2.bitwise_or(horizontal, vertical).sum() // 255)
    ink_pixels = int((foreground > 0).sum())
    return float(line_pixels) / float(max(1, ink_pixels))


def _bond_like_line_counts(foreground):
    lines = cv2.HoughLinesP(
        foreground,
        1,
        np.pi / 180,
        threshold=12,
        minLineLength=12,
        maxLineGap=4,
    )
    if lines is None:
        return 0, 0
    diagonal = 0
    axis_aligned = 0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 8:
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        if 12.0 <= angle <= 78.0:
            diagonal += 1
        elif angle < 8.0 or angle > 82.0:
            axis_aligned += 1
    return diagonal, axis_aligned


def _trim_line_polluted_candidate(candidate, foreground):
    cleaned = _remove_long_ruling_lines(foreground)
    if not np.any(cleaned):
        return candidate, False

    bbox = np.asarray(candidate["bbox"], dtype=np.int32)
    y1, x1, y2, x2 = [int(value) for value in bbox]
    bbox_area = max(1, (y2 - y1) * (x2 - x1))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    components = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = [int(value) for value in stats[label_idx]]
        if area < max(80, int(bbox_area * 0.01)):
            continue
        if _component_is_line_like(x, y, w, h, area):
            continue
        component_mask = (labels == label_idx).astype(np.uint8) * 255
        diagonal_lines, axis_aligned_lines = _bond_like_line_counts(component_mask)
        aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
        if diagonal_lines < 2 and (area < 250 or aspect > 4.5):
            continue
        components.append({
            "label": label_idx,
            "area": area,
            "bbox": np.array([y, x, y + h, x + w], dtype=np.int32),
            "score": area + diagonal_lines * 60 - axis_aligned_lines * 10,
        })
    if not components:
        return candidate, False

    components.sort(key=lambda item: item["score"], reverse=True)
    selected = [components[0]]
    selected_bbox = components[0]["bbox"].copy()
    changed = True
    while changed:
        changed = False
        expanded = np.array([
            selected_bbox[0] - 8,
            selected_bbox[1] - 8,
            selected_bbox[2] + 8,
            selected_bbox[3] + 8,
        ], dtype=np.int32)
        for component in components:
            if any(component["label"] == item["label"] for item in selected):
                continue
            box = component["bbox"]
            overlap = max(0, min(expanded[2], box[2]) - max(expanded[0], box[0])) * max(
                0, min(expanded[3], box[3]) - max(expanded[1], box[1])
            )
            if overlap <= 0:
                continue
            selected.append(component)
            selected_bbox = _bbox_union([item["bbox"] for item in selected]).astype(np.int32)
            changed = True

    local_mask = np.zeros_like(cleaned, dtype=bool)
    for component in selected:
        local_mask |= labels == component["label"]
    local_bbox = _mask_bbox(local_mask)
    if local_bbox is None:
        return candidate, False

    new_bbox = np.array([
        y1 + int(local_bbox[0]),
        x1 + int(local_bbox[1]),
        y1 + int(local_bbox[2]),
        x1 + int(local_bbox[3]),
    ], dtype=np.int32)
    new_area = _bbox_area(new_bbox)
    old_area = _bbox_area(bbox)
    if new_area <= 0 or new_area >= old_area * 0.90:
        return candidate, False
    if new_area < DET_FINAL_MIN_BBOX_AREA:
        return candidate, False

    updated_mask = np.zeros_like(candidate["mask"], dtype=bool)
    updated_mask[y1:y2, x1:x2] = local_mask
    updated = dict(candidate)
    updated["mask"] = updated_mask
    updated["bbox"] = new_bbox
    updated["score"] = float(candidate.get("score", 1.0)) * 0.97
    return updated, True


def _component_is_line_like(x, y, w, h, area):
    aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
    fill = float(area) / max(1.0, float(w * h))
    return aspect > 7.5 and fill > 0.35


def _expand_candidate_to_neighboring_foreground(candidate, image_bgr):
    bbox = np.asarray(candidate["bbox"], dtype=np.int32)
    height, width = image_bgr.shape[:2]
    y1, x1, y2, x2 = [int(v) for v in bbox]
    bbox_h = max(1, y2 - y1)
    bbox_w = max(1, x2 - x1)
    pad = min(
        DET_LOCAL_EXPANSION_MAX_PAD,
        max(DET_LOCAL_FOREGROUND_LINK_PAD, int(round(max(bbox_h, bbox_w) * 0.65))),
    )
    crop_y1, crop_x1, crop_y2, crop_x2 = _clip_bbox(bbox, height, width, pad=pad)
    crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return candidate

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = (gray < 205).astype(np.uint8) * 255
    foreground = _remove_long_ruling_lines(foreground)
    if not np.any(foreground):
        return candidate

    kernel = np.ones((5, 5), dtype=np.uint8)
    linked = cv2.dilate(foreground, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
    if num_labels <= 1:
        return candidate

    local_bbox = np.array([y1 - crop_y1, x1 - crop_x1, y2 - crop_y1, x2 - crop_x1], dtype=np.int32)
    local_mask = candidate["mask"][crop_y1:crop_y2, crop_x1:crop_x2]
    selected_boxes = [local_bbox]
    for label_idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label_idx]]
        if area < 12 or _component_is_line_like(x, y, w, h, area):
            continue
        component_region = labels == label_idx
        if np.any(component_region & local_mask):
            selected_boxes.append(np.array([y, x, y + h, x + w], dtype=np.int32))
            continue

        comp_bbox = np.array([y, x, y + h, x + w], dtype=np.int32)
        comp_aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
        comp_fill = float(area) / max(1.0, float(w * h))
        if comp_aspect > 4.5 and comp_fill < 0.25:
            continue
        if w > max(80, bbox_w * 1.15) or h > max(80, bbox_h * 1.25):
            continue
        vertical_overlap = max(0, min(local_bbox[2], comp_bbox[2]) - max(local_bbox[0], comp_bbox[0]))
        vertical_coverage = vertical_overlap / max(1, min(local_bbox[2] - local_bbox[0], comp_bbox[2] - comp_bbox[0]))
        horizontal_gap = max(0, max(local_bbox[1], comp_bbox[1]) - min(local_bbox[3], comp_bbox[3]))
        center_gap_y = abs(((local_bbox[0] + local_bbox[2]) / 2.0) - ((comp_bbox[0] + comp_bbox[2]) / 2.0))
        if (
            horizontal_gap <= max(12, int(max(bbox_h, bbox_w) * 0.32))
            and (vertical_coverage >= 0.20 or center_gap_y <= max(bbox_h, h) * 0.65)
        ):
            selected_boxes.append(comp_bbox)

    if len(selected_boxes) <= 1:
        return candidate

    union = _bbox_union(selected_boxes)
    if union is None:
        return candidate
    union = np.asarray([
        union[0] + crop_y1,
        union[1] + crop_x1,
        union[2] + crop_y1,
        union[3] + crop_x1,
    ], dtype=np.int32)
    union = np.asarray(_clip_bbox(union, height, width, pad=0), dtype=np.int32)
    if _bbox_area(union) <= _bbox_area(bbox):
        return candidate
    if _bbox_area(union) > _bbox_area(bbox) * 4.5:
        return candidate
    union_h = max(1, int(union[2] - union[0]))
    union_w = max(1, int(union[3] - union[1]))
    if union_w > max(180, int(bbox_w * 2.8)) or union_h > max(120, int(bbox_h * 2.4)):
        return candidate

    expanded_mask = candidate["mask"].copy()
    uy1, ux1, uy2, ux2 = [int(v) for v in union]
    expanded_mask[uy1:uy2, ux1:ux2] |= (cv2.cvtColor(image_bgr[uy1:uy2, ux1:ux2], cv2.COLOR_BGR2GRAY) < 235)
    updated = dict(candidate)
    updated["bbox"] = union
    updated["mask"] = expanded_mask
    updated["score"] = float(candidate.get("score", 1.0)) * 0.98
    return updated


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

    foreground_labels, foreground_components = ndimage.label(foreground > 0)
    if foreground_components <= 1:
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
    if split_masks and len(split_masks) < DET_FOREGROUND_SPLIT_MIN_COMPONENTS:
        split_union_bbox = _bbox_union([_mask_bbox(split) for split in split_masks])
        original_bbox = _mask_bbox(component_mask)
        if _bbox_span_ratio(split_union_bbox, original_bbox) < DET_AGGREGATE_MIN_CHILD_SPAN_RATIO:
            return [component_mask]
    return split_masks or [component_mask]


def _suppress_aggregate_mask_candidates(mask_candidates):
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
            if (
                _large_connected_component_count(candidate["mask"], min_area=DET_COMPONENT_MIN_AREA) <= 1
                and _mask_bbox_fill_ratio(candidate["mask"], candidate["bbox"]) >= 0.82
            ):
                continue
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

    if not suppress_indices:
        return mask_candidates
    return [candidate for idx, candidate in enumerate(mask_candidates) if idx not in suppress_indices]


def _dedupe_mask_candidates(mask_candidates):
    if not mask_candidates:
        return []

    mask_candidates = _suppress_aggregate_mask_candidates(mask_candidates)
    if not mask_candidates:
        return []

    ordered = sorted(
        mask_candidates,
        key=lambda item: (-_bbox_area(item["bbox"]), -int(item["mask"].sum()), -float(item["score"])),
    )
    accepted = []
    seen_hashes = set()
    for candidate in ordered:
        mask_hash = _mask_candidate_hash(candidate["mask"], candidate["bbox"])
        if mask_hash in seen_hashes:
            continue
        if any(
            _mask_coverage(candidate["mask"], kept["mask"]) >= DET_MASK_COVERAGE_THRESHOLD
            or _mask_iou(candidate["mask"], kept["mask"]) >= DET_MASK_IOU_THRESHOLD
            or _bbox_contains(kept["bbox"], candidate["bbox"])
            or _bbox_overlap_coverage(candidate["bbox"], kept["bbox"]) >= DET_BBOX_OVERLAP_COVERAGE_THRESHOLD
            for kept in accepted
        ):
            continue
        accepted.append(candidate)
        seen_hashes.add(mask_hash)

    accepted.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    return accepted


def _complete_single_structure_mask(image_bgr, binarized, eroded_image, mask, bbox, score):
    height, width = image_bgr.shape[:2]
    bbox = np.asarray(bbox, dtype=np.int32)
    bbox_area = _bbox_area(bbox)
    page_area = max(1, height * width)
    mask_area = int(mask.sum())
    if bbox_area <= 0 or mask_area <= 0:
        return [{"mask": mask, "bbox": bbox, "score": score}]
    if bbox_area / page_area > DET_MAX_MASK_PAGE_FRACTION or mask_area / page_area > DET_MAX_MASK_PAGE_FRACTION:
        return [{"mask": mask, "bbox": bbox, "score": score}]

    y1, x1, y2, x2 = [int(v) for v in bbox]
    bbox_h = max(1, y2 - y1)
    bbox_w = max(1, x2 - x1)
    pad_ratio = DET_LOCAL_EXPANSION_PAD_RATIO
    if max(bbox_h, bbox_w) < 90:
        pad_ratio = max(pad_ratio, 0.85)
    elif max(bbox_h, bbox_w) < 140:
        pad_ratio = max(pad_ratio, 0.55)
    pad = min(
        DET_LOCAL_EXPANSION_MAX_PAD,
        max(DET_LOCAL_EXPANSION_MIN_PAD, int(round(max(bbox_h, bbox_w) * pad_ratio))),
    )
    crop_y1, crop_x1, crop_y2, crop_x2 = _clip_bbox(bbox, height, width, pad=pad)
    crop_h = max(0, crop_y2 - crop_y1)
    crop_w = max(0, crop_x2 - crop_x1)
    if crop_h <= 0 or crop_w <= 0 or crop_h * crop_w > DET_MAX_LOCAL_EXPANSION_PIXELS:
        return [{"mask": mask, "bbox": bbox, "score": score}]

    local_binarized = binarized[crop_y1:crop_y2, crop_x1:crop_x2]
    local_eroded = eroded_image[crop_y1:crop_y2, crop_x1:crop_x2]
    local_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    local_bbox = _bbox_within_crop(bbox, crop_y1, crop_x1, crop_y2, crop_x2)
    max_depiction_size = _determine_depiction_size_with_buffer(np.asarray([local_bbox], dtype=np.int32))

    horizontal_vertical_lines = _detect_horizontal_and_vertical_lines(
        local_eroded,
        max_depiction_size,
    )
    hough_lines = _detect_lines(
        local_binarized,
        max_depiction_size,
        segmentation_mask=local_mask,
    )
    blur_factor = max(2, int(crop_w / 185))
    kernel = np.ones((blur_factor, blur_factor), dtype=np.uint8)
    exclusion_mask = horizontal_vertical_lines | cv2.dilate(hough_lines.astype(np.uint8), kernel, iterations=1).astype(bool)
    image_with_exclusion = ~((~local_eroded) & (~exclusion_mask))

    seeds = _get_seeds(image_with_exclusion, local_mask, exclusion_mask)
    expanded = _expand_mask(image_with_exclusion, seeds)
    candidate_mask = expanded if expanded.any() else local_mask

    candidates = []
    for component in _split_connected_masks(candidate_mask):
        page_component = _place_local_mask(component, (height, width), crop_y1, crop_x1)
        split_components = _split_component_by_foreground_regions(image_bgr, page_component)
        if len(split_components) > 1:
            split_components = [page_component] + split_components
        for split_component in split_components:
            split_bbox = _mask_bbox(split_component)
            if split_bbox is None:
                continue
            candidates.append({"mask": split_component, "bbox": split_bbox, "score": score})
            if len(candidates) >= DET_MAX_EXPANDED_CANDIDATES:
                break
        if len(candidates) >= DET_MAX_EXPANDED_CANDIDATES:
            break
    return candidates or [{"mask": mask, "bbox": bbox, "score": score}]


def _complete_structure_mask(image_bgr, masks, bboxes, scores=None):
    if masks.size == 0 or masks.shape[2] == 0:
        return masks, bboxes

    binarized = _binarize_image(image_bgr, threshold=0.72)
    blur_factor = max(2, int(image_bgr.shape[1] / 185))
    kernel = np.ones((blur_factor, blur_factor), dtype=np.uint8)
    eroded_image = erosion(binarized, footprint=kernel)

    expanded_candidates = []
    for idx in range(masks.shape[2]):
        mask = masks[:, :, idx]
        score = 0.0 if scores is None else float(scores[idx])
        bbox = np.asarray(bboxes[idx], dtype=np.int32)
        expanded_candidates.extend(
            _complete_single_structure_mask(
                image_bgr,
                binarized,
                eroded_image,
                mask,
                bbox,
                score,
            )
        )
        if len(expanded_candidates) >= DET_MAX_EXPANDED_CANDIDATES:
            break

    unique_candidates = _dedupe_mask_candidates(expanded_candidates)
    if not unique_candidates:
        return np.empty((image_bgr.shape[0], image_bgr.shape[1], 0), dtype=bool), np.empty((0, 4), dtype=np.int32)

    expanded_stack = np.stack([item["mask"] for item in unique_candidates], axis=-1)
    expanded_bboxes = np.stack([item["bbox"] for item in unique_candidates], axis=0).astype(np.int32)
    return expanded_stack, expanded_bboxes


def _tile_origins(length, tile_size, stride, max_tiles):
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    final_origin = max(0, length - tile_size)
    if not origins or origins[-1] != final_origin:
        origins.append(final_origin)
    if len(origins) <= max_tiles:
        return origins
    if max_tiles <= 1:
        return [0]
    sampled = []
    for idx in range(max_tiles):
        source_idx = round(idx * (len(origins) - 1) / (max_tiles - 1))
        sampled.append(origins[source_idx])
    return list(dict.fromkeys(sampled))


def _detect_masks_single_pass(image):
    model = _load_model()

    with torch.no_grad():
        input_tensor, meta = _mold_image(image)
        predictions = model(input_tensor)[0]

    boxes = predictions["boxes"]
    scores = predictions["scores"]
    raw_masks = predictions["masks"]

    keep = scores > DET_CONFIDENCE
    boxes = boxes[keep]
    scores = scores[keep]
    raw_masks = raw_masks[keep]
    if scores.numel() > DET_MODEL_DETECTIONS_PER_IMG:
        top_indices = torch.argsort(scores, descending=True)[:DET_MODEL_DETECTIONS_PER_IMG]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        raw_masks = raw_masks[top_indices]

    masks, bboxes, scores = _unmold_detections(boxes, scores, raw_masks, meta)
    masks, bboxes, scores = _filter_raw_detections(masks, bboxes, scores, image.shape)
    if masks.shape[2] == 0:
        return masks, bboxes, scores
    masks, bboxes = _complete_structure_mask(image, masks, bboxes, scores=scores)
    scores = np.ones((masks.shape[2],), dtype=np.float32)
    return masks, bboxes, scores


def _translate_tile_candidates(tile_masks, tile_bboxes, offset_y, offset_x, page_shape):
    page_h, page_w = page_shape[:2]
    candidates = []
    for idx in range(tile_masks.shape[2]):
        tile_mask = tile_masks[:, :, idx]
        if not tile_mask.any():
            continue
        mask = np.zeros((page_h, page_w), dtype=bool)
        y2 = min(page_h, offset_y + tile_mask.shape[0])
        x2 = min(page_w, offset_x + tile_mask.shape[1])
        if y2 <= offset_y or x2 <= offset_x:
            continue
        mask[offset_y:y2, offset_x:x2] = tile_mask[:y2 - offset_y, :x2 - offset_x]
        bbox = np.asarray(tile_bboxes[idx], dtype=np.int32).copy()
        bbox[[0, 2]] += int(offset_y)
        bbox[[1, 3]] += int(offset_x)
        bbox = np.asarray(_clip_bbox(bbox, page_h, page_w, pad=0), dtype=np.int32)
        candidates.append({"mask": mask, "bbox": bbox, "score": 1.0})
    return candidates


def _detect_multiscale_mask_candidates(image):
    masks, bboxes, scores = _detect_masks_single_pass(image)
    height, width = image.shape[:2]
    candidates = [
        {"mask": masks[:, :, idx], "bbox": np.asarray(bboxes[idx], dtype=np.int32), "score": float(scores[idx])}
        for idx in range(masks.shape[2])
    ]

    if not DET_TILE_PASS_ENABLED:
        return candidates
    tile_size = max(256, min(DET_TILE_SIZE, max(height, width)))
    stride = max(128, int(round(tile_size * DET_TILE_STRIDE_RATIO)))
    y_origins = _tile_origins(height, tile_size, stride, DET_MAX_TILE_PASSES)
    x_origins = _tile_origins(width, tile_size, stride, DET_MAX_TILE_PASSES)
    for y0 in y_origins:
        for x0 in x_origins:
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            if y1 - y0 < 220 or x1 - x0 < 220:
                continue
            if y0 == 0 and x0 == 0 and y1 == height and x1 == width:
                continue
            tile = image[y0:y1, x0:x1]
            tile_masks, tile_bboxes, tile_scores = _detect_masks_single_pass(tile)
            candidates.extend(_translate_tile_candidates(tile_masks, tile_bboxes, y0, x0, image.shape))
            if len(candidates) >= DET_MAX_EXPANDED_CANDIDATES:
                break
        if len(candidates) >= DET_MAX_EXPANDED_CANDIDATES:
            break
    return candidates


def _filter_final_mask_candidates(candidates, image_bgr):
    filtered = []
    page_shape = image_bgr.shape
    page_area = max(1, int(page_shape[0]) * int(page_shape[1]))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    for candidate in candidates or []:
        bbox = np.asarray(candidate["bbox"], dtype=np.int32)
        bbox_h = max(0, int(bbox[2] - bbox[0]))
        bbox_w = max(0, int(bbox[3] - bbox[1]))
        bbox_area = bbox_h * bbox_w
        if bbox_h < DET_FINAL_MIN_BBOX_SIDE or bbox_w < DET_FINAL_MIN_BBOX_SIDE:
            continue
        if bbox_area < DET_FINAL_MIN_BBOX_AREA:
            continue
        if min(bbox_h, bbox_w) < 32 and bbox_area < 1600:
            continue
        if bbox_area / page_area > DET_RAW_MAX_BBOX_PAGE_FRACTION:
            continue
        mask_area = int(candidate["mask"].sum())
        if mask_area < max(18, int(bbox_area * 0.006)):
            continue
        y1, x1, y2, x2 = [int(v) for v in bbox]
        crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        foreground = (crop < 205).astype(np.uint8) * 255
        ruling_line_ratio = _ruling_line_ratio(foreground)
        if ruling_line_ratio > DET_FINAL_MAX_RULING_LINE_RATIO:
            continue
        diagonal_lines, axis_aligned_lines = _bond_like_line_counts(foreground)
        if ruling_line_ratio > 0.14 and diagonal_lines < 18 and axis_aligned_lines > diagonal_lines * 2:
            continue
        if ruling_line_ratio > 0.14 and axis_aligned_lines > diagonal_lines * 1.3:
            trimmed_candidate, was_trimmed = _trim_line_polluted_candidate(candidate, foreground)
            if was_trimmed:
                candidate = trimmed_candidate
                bbox = np.asarray(candidate["bbox"], dtype=np.int32)
                bbox_h = max(0, int(bbox[2] - bbox[0]))
                bbox_w = max(0, int(bbox[3] - bbox[1]))
                bbox_area = bbox_h * bbox_w
                if bbox_h < DET_FINAL_MIN_BBOX_SIDE or bbox_w < DET_FINAL_MIN_BBOX_SIDE:
                    continue
                if bbox_area < DET_FINAL_MIN_BBOX_AREA:
                    continue
                y1, x1, y2, x2 = [int(v) for v in bbox]
                crop = gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                foreground = (crop < 205).astype(np.uint8) * 255
                diagonal_lines, axis_aligned_lines = _bond_like_line_counts(foreground)
                if diagonal_lines < 6 and axis_aligned_lines > diagonal_lines:
                    continue
        ink_pixels = int((crop < 205).sum())
        ink_density = ink_pixels / max(1, bbox_area)
        if ink_pixels < DET_FINAL_MIN_INK_PIXELS:
            continue
        if ink_density < DET_FINAL_MIN_INK_DENSITY:
            continue
        aspect_ratio = max(bbox_w / max(1, bbox_h), bbox_h / max(1, bbox_w))
        if aspect_ratio > 6.0 and ink_density < 0.08:
            continue
        filtered.append(candidate)
    return filtered


def get_expanded_masks(image: np.array) -> np.array:
    candidates = _detect_multiscale_mask_candidates(image)
    candidates = _dedupe_mask_candidates(candidates)
    candidates = _filter_final_mask_candidates(candidates, image)
    candidates = _dedupe_mask_candidates(candidates)
    if not candidates:
        return np.empty((image.shape[0], image.shape[1], 0), dtype=bool)
    return np.stack([item["mask"] for item in candidates], axis=-1).astype(bool)


def apply_masks(image: np.array, masks: np.array):
    segments = []
    bboxes_out = []

    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        margin = 2
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(image.shape[0], y1 + margin)
        x1 = min(image.shape[1], x1 + margin)

        crop = image[y0:y1, x0:x1].copy()

        segments.append(crop)
        bboxes_out.append((y0, x0, y1, x1))

    return segments, bboxes_out

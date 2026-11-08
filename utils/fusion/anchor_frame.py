"""Anchor frame conversion between page coordinates and RoI frames.

Why this exists (fusion design / round-2 P1): the MoE decoder and its
attachment head live in the **RoI frame** — the detected box resampled onto
a 12×12 grid, anisotropically stretched. Supervision and thresholds,
however, were defined on full images:

* training refs carry ``attachment_points`` / ``attachment_anchor_points`` /
  ``attachment_atom_coords`` normalized to the *input image* frame
  (dataset.py ``attachment_targets``, bms_collate);
* production gating uses ``attachment_set_max_anchor_distance = 0.25``
  in that same frame (experiments/moe/production/moe_config.json).

With fusion the "input image" of the decoder is the RoI, so:

* **supervision** must be re-anchored: page-normalized (x, y) -> RoI frame
  ``(x - x1) / (x2 - x1)``, with points falling outside the box masked out
  (they are not visible in that RoI);
* **thresholds** are anisotropic: a page-frame distance ``d`` becomes
  ``d_x / (x2 - x1)`` along x and ``d_y / (y2 - y1)`` along y in the RoI
  frame, so the per-axis acceptance thresholds are ``base / box_width`` and
  ``base / box_height``;
* **predictions** (anchor points emitted by the attachment head) must be
  mapped back RoI -> page for downstream markush assembly, which lives in
  page coordinates.
"""
from __future__ import annotations

import torch

EPS = 1e-6

ANCHOR_COORD_KEYS = (
    "attachment_points",
    "attachment_anchor_points",
    "attachment_atom_coords",
)


def _box_tensors(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    width = (x2 - x1).clamp(min=EPS)
    height = (y2 - y1).clamp(min=EPS)
    return x1, y1, width, height


def page_to_roi(points: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map page-normalized points into the RoI frame of ``box``.

    Parameters
    ----------
    points : (..., 2) page-normalized xy
    box : (4,) normalized xyxy

    Returns
    -------
    roi_points : (..., 2) same shape, coordinates in [0, 1] of the box
    valid : (...,) bool — False where the source point lay outside the box
    """
    x1, y1, width, height = _box_tensors(box)
    rx = (points[..., 0] - x1) / width
    ry = (points[..., 1] - y1) / height
    roi = torch.stack([rx, ry], dim=-1)
    valid = (rx >= -EPS) & (rx <= 1 + EPS) & (ry >= -EPS) & (ry <= 1 + EPS)
    return roi, valid


def roi_to_page(points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Inverse map: RoI-frame points back to page-normalized coordinates."""
    x1, y1, width, height = _box_tensors(box)
    return torch.stack([x1 + points[..., 0] * width,
                        y1 + points[..., 1] * height], dim=-1)


def anchor_distance_thresholds(box: torch.Tensor, base: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis RoI-frame thresholds equivalent to ``base`` in page frame.

    A page-frame tolerance of ``base`` spans ``base / box_width`` of the box
    along x (same for y), which is the distance the RoI-frame consumer must
    accept. This keeps the *physical* tolerance identical to production
    instead of silently shrinking it on small boxes.
    """
    _, _, width, height = _box_tensors(box)
    return base / width, base / height


def within_page_threshold(
    delta: torch.Tensor, box: torch.Tensor, base: float = 0.25
) -> torch.Tensor:
    """True where RoI-frame offsets are within the page-frame tolerance.

    ``delta``: (..., 2) RoI-frame offsets (predicted minus target). The check
    converts them back to page units per axis before comparing, i.e. it is
    the exact production ``max_anchor_distance`` semantics preserved under
    the anisotropic RoI resampling.
    """
    _, _, width, height = _box_tensors(box)
    page_dx = delta[..., 0] * width
    page_dy = delta[..., 1] * height
    return (page_dx.abs() <= base) & (page_dy.abs() <= base)


def reanchor_refs(
    refs: dict[str, torch.Tensor], boxes: list[torch.Tensor],
    mask_keys: tuple[str, ...] = ANCHOR_COORD_KEYS,
) -> dict[str, torch.Tensor]:
    """Rewrite anchor-coordinate supervision from page frame to RoI frames.

    ``boxes`` is the per-page box list in RoICollector order (page-major);
    refs are already per-box expanded (``expand_refs_for_boxes`` output).
    Rows carry one of the two padded layouts:

    * ``(N, K, 2)`` per-attachment points — masked by writing ``-1`` (the
      collate padding value) where the point lies outside its box;
    * ``(N, K)`` index/count tensors are left untouched (indices refer to
      atoms whose *coordinates* were re-anchored, not to pixels).
    """
    flat_boxes = torch.cat([b for b in boxes if b.numel()], dim=0) if sum(
        b.numel() for b in boxes) else torch.zeros(0, 4)
    if flat_boxes.shape[0] != _rows(refs):
        raise ValueError(
            f"reanchor_refs: {flat_boxes.shape[0]} boxes but refs have "
            f"{_rows(refs)} rows (expand_refs_for_boxes must run first)"
        )
    out = dict(refs)
    for key in mask_keys:
        value = refs.get(key)
        if not torch.is_tensor(value) or value.dim() < 2 or value.shape[-1] != 2:
            continue
        rows, k = value.shape[0], value.shape[1]
        reanchored = torch.empty_like(value)
        for row in range(rows):
            roi, valid = page_to_roi(value[row], flat_boxes[row])
            roi = torch.where(valid.unsqueeze(-1), roi, torch.full_like(roi, -1.0))
            reanchored[row] = roi
        out[key] = reanchored
    return out


def reanchor_predictions(
    predictions: list[dict], boxes: torch.Tensor
) -> list[dict]:
    """Map predicted anchor/attachment points from RoI frame back to page.

    ``boxes``: (K, 4) the boxes that produced ``predictions`` (same order).
    Handles both dict payloads with tensor ``anchor_points`` and plain
    tensors; unknown payloads pass through unchanged.
    """
    out: list[dict] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            out.append(prediction)
            continue
        converted = dict(prediction)
        points = converted.get("anchor_points")
        if torch.is_tensor(points) and points.dim() == 2 and points.shape[-1] == 2:
            converted["anchor_points"] = roi_to_page(points, boxes[index])
            converted["anchor_points_frame"] = "page"
        attachment = converted.get("attachment_set_predictions")
        if torch.is_tensor(attachment) and attachment.dim() == 2 and attachment.shape[-1] == 2:
            converted["attachment_set_predictions"] = roi_to_page(attachment, boxes[index])
        out.append(converted)
    return out


def _rows(refs: dict[str, torch.Tensor]) -> int:
    for key in ANCHOR_COORD_KEYS:
        value = refs.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.shape[0])
    for value in refs.values():
        if torch.is_tensor(value) and value.dim() >= 1:
            return int(value.shape[0])
    raise ValueError("refs carry no row axis")

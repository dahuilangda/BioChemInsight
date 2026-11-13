"""Shape-checked objectives and supervision for atom/chemical-edge learning.

Indices and coordinates are image-local. Coordinates use normalized (x, y)
positions in [0, 1]; missing supervision is represented by explicit masks or
-100 labels, never by invented negative edges or coordinate targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

IGNORE_INDEX = -100


def _floating(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating-point tensor')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _mask(value: Tensor | None, shape: torch.Size | tuple, device: torch.device, name: str) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if value.dtype != torch.bool or value.shape != shape or value.device != device:
        raise ValueError(f'{name} must be a boolean tensor with the same shape and device')
    return value


def _indices(value: Tensor, name: str, upper: int, device: torch.device | None = None) -> Tensor:
    if value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional int64 tensor')
    if device is not None and value.device != device:
        raise ValueError(f'{name} has a different device')
    if ((value < 0) | (value >= upper)).any():
        raise ValueError(f'{name} is outside the valid index range')
    return value


def _coordinates(value: Tensor, name: str) -> Tensor:
    result = _floating(value, name)
    if result.ndim != 2 or result.shape[-1] != 2 or ((result < 0) | (result > 1)).any():
        raise ValueError(f'{name} must be N x 2 normalized (x, y) coordinates in [0, 1]')
    return result


def _focal_parameters(alpha: float, gamma: float) -> None:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be finite and strictly between zero and one')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')


def focal_objectness_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> Tensor:
    """Stable binary focal loss; broadcasting targets or masks is forbidden.

    Targets must be binary, or -100 for missing labels. An explicit false mask
    also removes a query from both the loss and its mean denominator. Empty or
    wholly missing supervision returns a differentiable zero. Half/BF16 inputs
    reduce in FP32; float64 stays float64 for gradient checks. For distributed
    accumulation, request ``sum`` and divide by the global valid-query count.
    """
    value = _floating(logits, 'logits')
    if not isinstance(targets, Tensor) or targets.shape != logits.shape or targets.device != logits.device:
        raise ValueError('targets must have exactly the logits shape and device; broadcasting is forbidden')
    _focal_parameters(alpha, gamma)
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError('reduction must be mean, sum, or none')
    valid = _mask(valid_mask, logits.shape, logits.device, 'valid_mask') & targets.ne(IGNORE_INDEX)
    selected = targets[valid]
    if not torch.isfinite(selected).all() or ((selected != 0) & (selected != 1)).any():
        raise ValueError('valid objectness targets must be zero or one')
    # Avoid NaNs from intentionally unannotated/masked positions.
    truth = torch.where(valid, targets, torch.zeros_like(targets)).to(value.dtype)
    probability = value.sigmoid()
    p_t = torch.where(truth.bool(), probability, 1 - probability)
    alpha_t = torch.where(truth.bool(), alpha, 1 - alpha)
    loss = F.binary_cross_entropy_with_logits(value, truth, reduction='none')
    loss = (loss * alpha_t * (1 - p_t).pow(gamma)).masked_fill(~valid, 0)
    if reduction == 'none':
        return loss
    total = loss.sum()
    if reduction == 'sum':
        return total
    return total / valid.sum().clamp_min(1)


def masked_coordinate_l1(
    coordinates: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    reduction: str = 'mean',
) -> Tensor:
    """Sum x/y L1 per annotated atom, then mean over annotated atoms.

    The mask has exactly the coordinate tensor's leading shape. Missing target
    pairs may contain NaNs but are never consumed; prediction tensors must be
    finite everywhere. A wholly missing task returns a differentiable zero.
    """
    value = _floating(coordinates, 'coordinates')
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError('coordinates must have two components on the last axis')
    if (not isinstance(targets, Tensor) or not targets.is_floating_point()
            or targets.shape != coordinates.shape or targets.device != coordinates.device):
        raise ValueError('coordinate targets must have exactly the prediction shape and device')
    mask = _mask(valid_mask, coordinates.shape[:-1], coordinates.device, 'valid_mask')
    if not torch.isfinite(targets[mask]).all():
        raise ValueError('valid coordinate targets contain nonfinite values')
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError('reduction must be mean, sum, or none')
    truth = torch.where(mask.unsqueeze(-1), targets, 0).to(value.dtype)
    loss = (value - truth).abs().sum(-1).masked_fill(~mask, 0)
    if reduction == 'none':
        return loss
    total = loss.sum()
    return total if reduction == 'sum' else total / mask.sum().clamp_min(1)


@torch.no_grad()
def hungarian_atom_cost(
    objectness_logits: Tensor,
    coordinates: Tensor,
    target_coordinates: Tensor,
    *,
    class_weight: float = 1.0,
    coordinate_weight: float = 5.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Return Q x N assignment costs: focal positive-minus-negative plus L1.

    This computes no assignment and transfers nothing to the CPU. Callers can
    pass the result to a linear-sum solver. Ground-truth counts/coordinates
    belong only to supervised matching, never to image-only inference.
    """
    logits = _floating(objectness_logits, 'objectness_logits')
    points = _coordinates(coordinates, 'coordinates')
    gold = _coordinates(target_coordinates, 'target_coordinates')
    if logits.ndim != 1 or logits.shape[0] != points.shape[0]:
        raise ValueError('objectness_logits must have one entry per predicted coordinate')
    if points.device != logits.device or gold.device != logits.device:
        raise ValueError('all matching tensors must share a device')
    _focal_parameters(alpha, gamma)
    if (not math.isfinite(class_weight) or not math.isfinite(coordinate_weight)
            or class_weight < 0 or coordinate_weight < 0 or class_weight + coordinate_weight <= 0):
        raise ValueError('matching weights must be finite, nonnegative, and not both zero')
    dtype = torch.promote_types(logits.dtype, torch.promote_types(points.dtype, gold.dtype))
    logits, points, gold = logits.to(dtype), points.to(dtype), gold.to(dtype)
    probability = logits.sigmoid()
    positive = alpha * (1 - probability).pow(gamma) * F.softplus(-logits)
    negative = (1 - alpha) * probability.pow(gamma) * F.softplus(logits)
    return class_weight * (positive - negative).unsqueeze(1) + coordinate_weight * torch.cdist(points, gold, p=1)


def merge_chemical_probabilities(probabilities: Tensor) -> Tensor:
    """Merge normalized seven-class probabilities to [none,single,double,triple,aromatic].

    The input must already be probabilities. Classes 1, 5 and 6 are ordinary
    or depicted-stereo single bonds. This operation marginalizes their mass;
    it does not infer wedge direction. Passing raw logits is an error.
    """
    value = _floating(probabilities, 'probabilities')
    if value.ndim < 1 or value.shape[-1] != 7:
        raise ValueError('probabilities must have seven classes on the last axis')
    tolerance = 5e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if (value < 0).any() or (value > 1).any() or not torch.allclose(
        value.sum(-1), torch.ones_like(value[..., 0]), atol=tolerance, rtol=0,
    ):
        raise ValueError('expected normalized probabilities, not logits or arbitrary scores')
    return torch.stack((value[..., 0], value[..., [1, 5, 6]].sum(-1),
                        value[..., 2], value[..., 3], value[..., 4]), dim=-1)


def logits_to_chemical_probabilities(logits: Tensor) -> Tensor:
    """Softmax seven-class logits, then marginalize single-bond subclasses."""
    value = _floating(logits, 'logits')
    if value.ndim < 1 or value.shape[-1] != 7:
        raise ValueError('logits must have seven classes on the last axis')
    return merge_chemical_probabilities(value.softmax(-1))


@dataclass(frozen=True)
class DenoisingTargets:
    points: Tensor
    objectness_targets: Tensor
    objectness_valid_mask: Tensor
    coordinate_targets: Tensor
    coordinate_valid_mask: Tensor


def make_denoising_targets(
    gold_points: Tensor,
    positive_indices: Tensor,
    negative_points: Tensor,
    *,
    negative_exclusion_radius: float,
    noise_std: float = 0.015,
    generator: torch.Generator | None = None,
) -> DenoisingTargets:
    """Build unbatched DN anchors with explicit negative-near-atom masking.

    Positive anchors are noisy gold positions; their targets remain the exact
    gold positions. Random negative anchors within the exclusion radius of any
    gold atom have unknown objectness, not a fabricated negative label. The
    radius uses normalized image coordinates and must be chosen for the actual
    pixel/bond scale. Empty gold or negative sets are supported.
    """
    gold = _coordinates(gold_points, 'gold_points')
    negative = _coordinates(negative_points, 'negative_points')
    indices = _indices(positive_indices, 'positive_indices', len(gold), gold.device)
    if negative.device != gold.device:
        raise ValueError('gold_points and negative_points must share a device')
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')
    if not math.isfinite(negative_exclusion_radius) or negative_exclusion_radius < 0:
        raise ValueError('negative_exclusion_radius must be finite and nonnegative')
    negative = negative.to(gold.dtype)
    positives = gold[indices]
    noise = torch.randn(positives.shape, dtype=gold.dtype, device=gold.device, generator=generator) * noise_std
    anchors = torch.cat(((positives + noise).clamp(0, 1), negative))
    positive_count = len(indices)
    targets = torch.zeros(len(anchors), dtype=gold.dtype, device=gold.device)
    targets[:positive_count] = 1
    valid = torch.ones(len(anchors), dtype=torch.bool, device=gold.device)
    if len(gold) and len(negative):
        valid[positive_count:] = torch.cdist(negative, gold).amin(-1) > negative_exclusion_radius
    coordinate_targets = torch.zeros_like(anchors)
    coordinate_targets[:positive_count] = positives
    coordinate_valid = torch.zeros_like(valid)
    coordinate_valid[:positive_count] = True
    return DenoisingTargets(anchors, targets, valid, coordinate_targets, coordinate_valid)


@dataclass(frozen=True)
class MatchedTargets:
    objectness_targets: Tensor
    objectness_valid_mask: Tensor
    coordinate_targets: Tensor
    coordinate_valid_mask: Tensor
    element_targets: Tensor
    edge_targets: Tensor


def matched_supervision(
    num_queries: int,
    num_gold: int,
    query_indices: Tensor,
    gold_indices: Tensor,
    *,
    gold_coordinates: Tensor | None = None,
    gold_coordinate_mask: Tensor | None = None,
    gold_elements: Tensor | None = None,
    gold_edges: Tensor | None = None,
    complete_annotations: bool = True,
) -> MatchedTargets:
    """Vectorize a one-to-one matching into query-indexed supervision.

    Unknown element/edge labels use -100. Unmatched query pairs are ignored by
    the edge loss, rather than asserted to be no-bond. With complete annotations
    every gold atom must be matched, and unmatched queries are known no-object.
    For partial annotations set ``complete_annotations=False``: unmatched
    objectness is then unknown too. Coordinate masks concern entire (x,y) pairs.
    """
    if not isinstance(num_queries, int) or not isinstance(num_gold, int) or num_queries < 0 or num_gold < 0:
        raise ValueError('num_queries and num_gold must be nonnegative integers')
    q = _indices(query_indices, 'query_indices', num_queries)
    g = _indices(gold_indices, 'gold_indices', num_gold, q.device)
    if q.shape != g.shape or q.unique().numel() != q.numel() or g.unique().numel() != g.numel():
        raise ValueError('matching must contain equally sized, unique query/gold index lists')
    if complete_annotations and len(g) != num_gold:
        raise ValueError('complete supervision requires every gold atom to be matched; check query capacity')
    device = q.device
    objectness = torch.zeros(num_queries, device=device)
    objectness[q] = 1
    objectness_valid = torch.full((num_queries,), complete_annotations, dtype=torch.bool, device=device)
    objectness_valid[q] = True
    coordinates = torch.zeros(num_queries, 2, device=device)
    coordinate_valid = torch.zeros(num_queries, dtype=torch.bool, device=device)
    if gold_coordinates is None:
        if gold_coordinate_mask is not None:
            raise ValueError('gold_coordinate_mask requires gold_coordinates')
    else:
        if (not gold_coordinates.is_floating_point() or gold_coordinates.shape != (num_gold, 2)
                or gold_coordinates.device != device):
            raise ValueError('gold_coordinates must be num_gold x 2 floating coordinates on the matching device')
        coord_mask = _mask(gold_coordinate_mask, (num_gold,), device, 'gold_coordinate_mask')
        _coordinates(gold_coordinates[coord_mask], 'valid gold_coordinates')
        coordinates[q] = torch.where(coord_mask[g, None], gold_coordinates[g], 0).to(coordinates.dtype)
        coordinate_valid[q] = coord_mask[g]
    elements = torch.full((num_queries,), IGNORE_INDEX, dtype=torch.long, device=device)
    if gold_elements is not None:
        if gold_elements.dtype != torch.long or gold_elements.shape != (num_gold,) or gold_elements.device != device:
            raise ValueError('gold_elements must be a num_gold int64 vector on the matching device')
        if ((gold_elements != IGNORE_INDEX) & ((gold_elements < 1) | (gold_elements > 118))).any():
            raise ValueError('gold_elements must contain atomic numbers 1..118 or -100')
        elements[q] = gold_elements[g]
    edges = torch.full((num_queries, num_queries), IGNORE_INDEX, dtype=torch.long, device=device)
    if gold_edges is not None:
        if gold_edges.dtype != torch.long or gold_edges.shape != (num_gold, num_gold) or gold_edges.device != device:
            raise ValueError('gold_edges must be a num_gold square int64 tensor on the matching device')
        if ((gold_edges != IGNORE_INDEX) & ((gold_edges < 0) | (gold_edges > 4))).any():
            raise ValueError('chemical edge targets must be 0..4 or -100')
        if not torch.equal(gold_edges, gold_edges.T):
            raise ValueError('chemical edge targets must be symmetric; directional stereo needs separate supervision')
        edges[q[:, None], q[None, :]] = gold_edges[g[:, None], g[None, :]]
        edges.fill_diagonal_(IGNORE_INDEX)
    return MatchedTargets(objectness, objectness_valid, coordinates, coordinate_valid, elements, edges)

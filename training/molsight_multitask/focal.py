"""Versioned chemical hard-negative weighting; not enabled in a trainer yet."""
import math
import torch
import torch.nn.functional as F


def chemical_focal_sums(logits, targets, *, positive_weight=0.5, gamma=2.0, detach_weights=True):
    """Balance per-image positive CE and focal-weighted negative CE.

    Input is B x N x N x 5 (native chemical) or x 7 (MolSight subclasses), with chemical labels 0..4 or -100. Single-bond
    subclasses 1/5/6 are marginalized. Crucially, p_none is a probability over
    chemical CLASSES, never over candidate atoms. Negative means divide by the
    fixed number of labeled negatives, not their weight sum. With detached
    weights this is the explicitly chosen focal-style surrogate; set False for
    differentiating through the focal factor. Callers must record that choice.
    Return a sum plus supervised-image count for global/DDP normalization.
    """
    if (not logits.is_floating_point() or logits.ndim != 4 or logits.shape[-1] not in (5, 7)
            or logits.shape[1] != logits.shape[2] or targets.shape != logits.shape[:-1]
            or targets.device != logits.device or targets.dtype != torch.long):
        raise ValueError('Require floating BxNxNx(5 or 7) logits and same-device aligned int64 labels')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite edge logits')
    if ((targets != -100) & ((targets < 0) | (targets > 4))).any():
        raise ValueError('Chemical labels must be 0..4 or -100')
    if targets.diagonal(dim1=1, dim2=2).ne(-100).any():
        raise ValueError('Self-pairs must be explicitly masked')
    if not math.isfinite(positive_weight) or not 0 < positive_weight < 1:
        raise ValueError('Positive weight must lie strictly between zero and one')
    if not math.isfinite(gamma) or gamma < 0 or type(detach_weights) is not bool:
        raise ValueError('Invalid focal configuration')
    value = logits.float()
    grouped = value if value.shape[-1] == 5 else torch.stack((
        value[..., 0], torch.logsumexp(value[..., [1, 5, 6]], -1),
        value[..., 2], value[..., 3], value[..., 4]), -1)
    loss = F.cross_entropy(grouped.reshape(-1, 5), targets.reshape(-1),
                           ignore_index=-100, reduction='none').reshape(targets.shape)
    p_none = grouped.softmax(dim=-1)[..., 0]
    weights = (1 - p_none).clamp_min(0).pow(gamma)
    if detach_weights:
        weights = weights.detach()
    positive, negative = targets.gt(0), targets.eq(0)
    positives, negatives = positive.sum((1, 2)), negative.sum((1, 2))
    positive_mean = (loss * positive).sum((1, 2)) / positives.clamp_min(1)
    negative_mean = (loss * negative * weights).sum((1, 2)) / negatives.clamp_min(1)
    positive_scale = positives.gt(0).float() * positive_weight
    negative_scale = negatives.gt(0).float() * (1 - positive_weight)
    scale = positive_scale + negative_scale
    per_image = (positive_scale * positive_mean + negative_scale * negative_mean) / scale.clamp_min(1e-12)
    return {'loss_sum': per_image.sum(), 'images': scale.gt(0).sum(),
            'positive_pairs': positives.sum(), 'negative_pairs': negatives.sum()}

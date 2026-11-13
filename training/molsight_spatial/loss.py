"""One matched detection objective, reusing reviewed matching/DN mathematics."""
import math
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from training.ocsr_reboot.objectives import (
    hungarian_atom_cost, matched_supervision, focal_objectness_loss,
    masked_coordinate_l1, make_denoising_targets,
)
from training.molsight_multitask.focal import chemical_focal_sums


@dataclass(frozen=True)
class LossConfig:
    objectness_weight: float = 1.
    balance_objectness: bool = False
    count_weight: float = 0.
    edge_negative_ratio: float = 0.
    stereo_negative_ratio: float = 0.
    coordinate_weight: float = 5.
    auxiliary_weight: float = .5
    state_weight: float = .2
    stereo_weight: float = .2
    edge_weight: float = 1.
    denoising_weight: float = .5
    gamma: float = 2.
    detach_focal_weights: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if name not in ['detach_focal_weights','balance_objectness'] and (not math.isfinite(value) or value < 0):
                raise ValueError('Invalid loss weight '+name)
        if type(self.detach_focal_weights) is not bool:
            raise ValueError('Focal gradient policy must be explicit')
        if type(self.balance_objectness) is not bool:raise ValueError('Objectness balance policy must be explicit')


def balanced_objectness_loss(logits,targets,valid_mask):
    """Equal total positive/negative weight across different molecule sizes."""
    pointwise=focal_objectness_loss(logits,targets,valid_mask=valid_mask,alpha=.5,reduction='none')
    means=[pointwise[mask].mean() for mask in [valid_mask & targets.eq(1),valid_mask & targets.eq(0)] if mask.any()]
    return sum(means)/len(means) if means else logits.sum()*0


def mine_pair_targets(logits,targets,ratio):
    """Keep every positive and a fixed-size set of hardest known negatives.

    K scales with labelled positive pairs, rather than all N² easy non-bonds.
    Unknown/diagonal pairs remain unknown; only training receives these labels.
    """
    if not math.isfinite(ratio) or ratio<=0:raise ValueError('Invalid negative mining ratio')
    if logits.ndim!=4 or logits.shape[1]!=logits.shape[2] or logits.shape[:-1]!=targets.shape or logits.shape[-1]<2 or targets.dtype!=torch.long:
        raise ValueError('Invalid pair targets')
    if targets.device!=logits.device or not torch.isfinite(logits).all():raise ValueError('Invalid pair logits')
    if ((targets!=-100)&((targets<0)|(targets>=logits.shape[-1]))).any():raise ValueError('Invalid pair label')
    result=targets.contiguous().clone()
    with torch.no_grad():
        hardness=-logits.detach().float().log_softmax(-1)[...,0]
        for b in range(len(targets)):
            negative=targets[b].eq(0);available=int(negative.sum())
            k=min(available,max(32,math.ceil(int(targets[b].gt(0).sum())*ratio)))
            result[b][negative]=-100
            if k:
                indices=hardness[b].masked_fill(~negative,float('-inf')).flatten().topk(k).indices
                result[b].view(-1)[indices]=0
    return result


def dn_targets(sample, *, positive_count=32, negative_count=32):
    points = sample['coords']
    if len(points) < 1:
        raise ValueError('Complete molecules must contain atoms')
    selected = torch.randperm(len(points), device=points.device)[:positive_count]
    negative = torch.rand(negative_count, 2, device=points.device)
    if len(points) > 1:
        distances = torch.cdist(points.float(), points.float())
        distances.fill_diagonal_(float('inf'))
        radius = float(distances.amin(-1).median()) * .3
    else:
        radius = .01
    return make_denoising_targets(points, positive_indices=selected,
        negative_points=negative, negative_exclusion_radius=radius, noise_std=.01)


def masked_ce(logits, labels):
    if logits.shape[:-1] != labels.shape or labels.dtype != torch.long:
        raise ValueError('State logits and labels differ')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite state logits')
    valid = labels.ne(-100)
    if not valid.any():
        return logits.sum() * 0
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any():
        raise ValueError('State label outside declared vocabulary')
    return F.cross_entropy(logits[valid].float(), labels[valid])


def detection_loss(output, sample, *, config=LossConfig(), denoising=None):
    """Batch-one effective objective. Gold is consumed only in training here."""
    if output['objectness'].ndim != 2 or output['objectness'].shape[0] != 1:
        raise ValueError('Use a single unpadded image for spatial supervision')
    count = output['objectness'].shape[1]
    n = len(sample['coords'])
    if not 0 < n <= count:
        raise ValueError('Query capacity does not cover the complete molecule')
    cost = hungarian_atom_cost(output['objectness'][0], output['coords'][0], sample['coords'])
    query, gold = linear_sum_assignment(cost.detach().double().cpu().numpy())
    q = torch.as_tensor(query, dtype=torch.long, device=cost.device)
    g = torch.as_tensor(gold, dtype=torch.long, device=cost.device)
    matched = matched_supervision(count, n, q, g, gold_coordinates=sample['coords'],
        gold_elements=sample['elements'], gold_edges=sample['edges'])
    values = {
        'objectness': config.objectness_weight * (balanced_objectness_loss(output['objectness'][0],matched.objectness_targets,matched.objectness_valid_mask)
            if config.balance_objectness else focal_objectness_loss(output['objectness'][0], matched.objectness_targets,
            valid_mask=matched.objectness_valid_mask)),
        'coordinates': config.coordinate_weight * masked_coordinate_l1(output['coords'][0],
            matched.coordinate_targets, matched.coordinate_valid_mask),
    }
    if config.count_weight:
        # Supervised training regularizer only; inference still uses the fixed threshold.
        values['expected_cardinality']=config.count_weight * F.smooth_l1_loss(
            output['objectness'][0].float().sigmoid().sum()/n,output['objectness'].new_tensor(1.).float())
    layers = output.get('aux_coords', [])
    values['auxiliary_coordinates'] = sum((masked_coordinate_l1(layer[0], matched.coordinate_targets,
        matched.coordinate_valid_mask) for layer in layers), output['coords'].sum()*0) * config.auxiliary_weight / max(1, len(layers))
    for name in ['elements', 'charge', 'isotope', 'explicit_h', 'no_implicit', 'radicals', 'aromatic']:
        values['state:'+name] = config.state_weight * masked_ce(output[name][0, q], sample[name][g])
    edge_targets=matched.edge_targets.unsqueeze(0)
    if config.edge_negative_ratio:
        edge_targets=mine_pair_targets(output['edges'],edge_targets,config.edge_negative_ratio)
    edges = chemical_focal_sums(output['edges'], edge_targets,
        gamma=config.gamma, detach_weights=config.detach_focal_weights)
    values['chemical_edges'] = config.edge_weight * edges['loss_sum'] / edges['images'].clamp_min(1)
    for name in ['wedges', 'double_stereo']:
        predicted = output[name][0][q[:, None], q[None, :]]
        target = sample[name][g[:, None], g[None, :]]
        if config.stereo_negative_ratio:
            target=mine_pair_targets(predicted.unsqueeze(0),target.unsqueeze(0),config.stereo_negative_ratio)[0]
        # Balance positive/none classes so sparse visible stereo is not swamped.
        valid = target.ne(-100)
        positive, negative = valid & target.gt(0), valid & target.eq(0)
        terms = [masked_ce(predicted, target.masked_fill(~mask, -100)) for mask in [positive, negative] if mask.any()]
        values['stereo:'+name] = config.stereo_weight * (sum(terms) / len(terms) if terms else predicted.sum()*0)
    if denoising is not None:
        dn = output['dn']
        values['dn_objectness'] = config.denoising_weight * focal_objectness_loss(
            dn['objectness'][0], denoising.objectness_targets, valid_mask=denoising.objectness_valid_mask)
        values['dn_coordinates'] = config.denoising_weight * config.coordinate_weight * masked_coordinate_l1(
            dn['coords'][0], denoising.coordinate_targets, denoising.coordinate_valid_mask)
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite matched objective')
    return {'loss': sum(values.values()), 'terms': values,
            'matched_atoms': n, 'query_indices': q, 'gold_indices': g}

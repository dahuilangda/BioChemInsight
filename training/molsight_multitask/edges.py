"""Chemical edge supervision without invented drawing wedge labels."""
import torch
import torch.nn.functional as F


CHEMICAL_TYPES = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}


def chemical_targets(annotation):
    count = len(annotation['atoms'])
    if [atom['atom_id'] for atom in annotation['atoms']] != list(range(count)):
        raise ValueError('Chemical atom identifiers are not contiguous')
    targets = torch.zeros(count, count, dtype=torch.long)
    targets.fill_diagonal_(-100)
    seen = set()
    for bond in annotation['bonds']:
        first, second = bond['begin'], bond['end']
        if not 0 <= first < count or not 0 <= second < count or first == second:
            raise ValueError('Invalid chemical bond endpoints')
        identity = tuple(sorted((first, second)))
        if identity in seen:
            raise ValueError('Repeated chemical bond')
        seen.add(identity)
        if bond['chemical_type'] not in CHEMICAL_TYPES:
            raise ValueError(f'Unsupported chemical bond type: {bond["chemical_type"]}')
        targets[first, second] = targets[second, first] = CHEMICAL_TYPES[bond['chemical_type']]
    return targets


def chemical_edge_sums(logits, targets, *, positive_weight=0.5):
    """Return a differentiable sum and valid-image denominator for DDP.

    MolSight's seven classes use 1,5,6 for ordinary/depicted stereo single
    bonds. Chemical-only labels cannot identify the depicted wedge direction,
    so marginalize those classes rather than supervising an invented wedge.
    Balance positive and negative pair means within each supervised image;
    the caller must globally reduce `images` before scaling backward.
    """
    if logits.ndim != 4 or logits.shape[-1] != 7 or logits.shape[:-1] != targets.shape:
        raise ValueError('Expected B x N x N x 7 logits and aligned targets')
    if logits.shape[1] != logits.shape[2]:
        raise ValueError('Edge matrices must be square')
    if not 0 < positive_weight < 1:
        raise ValueError('Positive weight must be between zero and one')
    if targets.dtype != torch.long or ((targets != -100) & ((targets < 0) | (targets > 4))).any():
        raise ValueError('Chemical labels must be 0..4 or -100')
    value = logits.float()
    grouped = torch.stack((value[..., 0], torch.logsumexp(value[..., [1, 5, 6]], dim=-1),
                           value[..., 2], value[..., 3], value[..., 4]), dim=-1)
    losses = F.cross_entropy(grouped.reshape(-1, 5), targets.reshape(-1),
                             ignore_index=-100, reduction='none').reshape(targets.shape)
    positive, negative = targets.gt(0), targets.eq(0)
    positive_counts = positive.sum((1, 2))
    negative_counts = negative.sum((1, 2))
    positive_mean = (losses * positive).sum((1, 2)) / positive_counts.clamp_min(1)
    negative_mean = (losses * negative).sum((1, 2)) / negative_counts.clamp_min(1)
    positive_scale = positive_counts.gt(0).float() * positive_weight
    negative_scale = negative_counts.gt(0).float() * (1 - positive_weight)
    scale = positive_scale + negative_scale
    per_image = (positive_scale * positive_mean + negative_scale * negative_mean) / scale.clamp_min(1e-12)
    return {'loss_sum': per_image.sum(), 'images': scale.gt(0).sum(),
            'positive_pairs': positive_counts.sum(), 'negative_pairs': negative_counts.sum()}

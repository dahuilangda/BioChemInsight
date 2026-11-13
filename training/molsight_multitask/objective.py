"""Character-token and chemical-edge objectives with separate denominators."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from molsight.loss import Criterion
from molsight_multitask.edges import chemical_edge_sums


class GraphCriterion(nn.Module):
    def __init__(self, args, tokenizer, *, edge_weight=0.1):
        super().__init__()
        if not math.isfinite(edge_weight) or edge_weight < 0:
            raise ValueError('Chemical edge weight must be finite and nonnegative')
        configuration = dict(vars(args))
        configuration['formats'] = ['char']
        self.character = Criterion(SimpleNamespace(**configuration), tokenizer)
        self.edge_weight = edge_weight

    def normalizers(self, batch):
        result = self.character.normalizers(batch)
        targets = batch.get('chemical_edges')
        result['chemical_edges'] = (targets.ge(0).any(dim=(1, 2)).sum().float()
                                    if targets is not None else result['sequence'].new_zeros(()))
        return result

    def forward(self, logits, edge_pred, loc_pred, label, chemical_edges=None,
                normalizers=None, world_size=1, **kwargs):
        if normalizers is None:
            normalizers = self.normalizers({'label': label, 'chemical_edges': chemical_edges})
        losses = self.character(logits=logits, edge_pred=None, loc_pred=None, label=label,
                                 normalizers=normalizers, world_size=world_size)
        if chemical_edges is None or not chemical_edges.ge(0).any():
            # Do not require/use a graph head for character-only replay.
            losses['chemical_edges'] = logits.float().sum() * 0
        else:
            if edge_pred is None:
                raise ValueError('Annotated chemical edges require an active edge head')
            total = chemical_edge_sums(edge_pred, chemical_edges)['loss_sum']
            losses['chemical_edges'] = (total * self.edge_weight * world_size
                                        / normalizers['chemical_edges'].clamp_min(1))
        return losses

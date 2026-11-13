"""Validate annotated visible units through fixed expansion and sparse assembly.

No image recognizer calls this supervised admission entry point.
"""
import torch
from training.molsight_dense.model import STATE_CLASSES,CHEMICAL,REVERSE
from training.molsight_dense.annotation import annotation_roundtrip
from .registry import TEMPLATES
from .model import expand_visible


def visible_annotation_roundtrip(target,coords):
    def logits(labels,n):return torch.nn.functional.one_hot(labels.clamp_min(0),n).float().unsqueeze(0)*40-20
    states={name:logits(target[name],n) for name,n in STATE_CLASSES.items()}
    states.update(aliases=logits(target['aliases'],len(TEMPLATES)+1),joint_bonds=logits(target['joint_bonds'],9),double_defined=logits(target['double_defined'],2))
    points,expanded,_=expand_visible(torch.as_tensor(coords).float().unsqueeze(0),states)
    joint=expanded['joint_bonds'][0].argmax(-1)
    if not torch.equal(joint,torch.tensor(REVERSE,device=joint.device)[joint.T]):
        raise ValueError('Annotated directed bonds are not reversal-consistent')
    chemical=torch.tensor(CHEMICAL,device=joint.device)[joint]
    atomic={name:expanded[name][0].argmax(-1) for name in ['elements','charge','isotope','radicals','aromatic']}
    atomic.update(explicit_h=expanded['hydrogens'][0].argmax(-1),no_implicit=torch.ones(len(joint),dtype=torch.long,device=joint.device),
        edges=chemical,wedges=torch.where((joint==5)|(joint==6),joint-4,0),
        double_stereo=expanded['double_defined'][0].argmax(-1)*chemical.eq(2))
    return annotation_roundtrip(atomic,points[0])

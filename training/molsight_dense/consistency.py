"""Training-only graph constraints; no test labels or inference graph repair.

Valence targets are measured on each annotated graph, rather than a hand-made
table which would reject valid charged atoms, aromatic heterocycles or sulfur.
Aliases receive degree supervision, but no invented hidden hydrogen targets.
"""
import math
import torch
from torch.nn import functional as F


def hardest_mean(values, fraction=.1):
    values=values.reshape(-1)
    return values.topk(max(1,math.ceil(len(values)*fraction))).values.mean() if len(values) else values.sum()


def hard_atom_loss(read,sample):
    terms=[]
    for name in ['elements','charge','isotope','hydrogens','radicals','aromatic']:
        labels=sample[name];valid=labels.ne(-100)
        if valid.any():
            ce=F.cross_entropy(read[name][0].float(),labels,reduction='none',ignore_index=-100)
            terms.append(hardest_mean(ce[valid]))
    return torch.stack(terms).mean() if terms else read['elements'].sum()*0


def chemical_consistency(read,sample):
    logits=read['joint_bonds'][0].float();n=logits.shape[0]
    valid=~torch.eye(n,dtype=torch.bool,device=logits.device)
    probabilities=logits.softmax(-1)*valid[...,None]
    orders=logits.new_tensor([0,1,2,3,1.5,1,1,1,1])
    labels=sample['joint_bonds'];known=labels.ne(-100)&valid
    if not torch.equal(known,valid):raise ValueError('Consistency needs fully labelled visible pairs')
    true_order=orders[labels.clamp_min(0)]*valid
    expected_order=(probabilities*orders).sum(-1).sum(-1)
    expected_degree=probabilities[...,1:].sum(-1).sum(-1)
    true_degree=(labels.gt(0)&valid).sum(-1).float()
    degree=F.smooth_l1_loss(expected_degree,true_degree)
    ordinary=sample['hydrogens'].ne(-100)
    if ordinary.any():
        expected_h=(read['hydrogens'][0].float().softmax(-1)*torch.arange(9,device=logits.device)).sum(-1)
        total=expected_order+expected_h
        target=true_order.sum(-1)+sample['hydrogens'].clamp_min(0)
        valence=F.smooth_l1_loss(total[ordinary],target[ordinary])
    else:valence=logits.sum()*0
    # Only ordinary pairs have annotated aromatic atom identities.
    atom_aromatic=read['aromatic'][0].float().softmax(-1)[:,1]
    eligible=ordinary[:,None]&ordinary[None,:]&valid
    incompatible=probabilities[...,4]*(1-atom_aromatic[:,None]*atom_aromatic[None,:])
    # Normalize by atoms, not N² easy non-bonds, to retain long-graph gradients.
    aromatic=incompatible[eligible].sum()/max(1,int(ordinary.sum()))
    return {'degree':degree,'valence':valence,'aromatic':aromatic}


def center_recall(logits,indices):
    selected=logits[indices[:,1],indices[:,0]].float()
    return hardest_mean(-F.logsigmoid(selected))

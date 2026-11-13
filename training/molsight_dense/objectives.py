"""Pixel-grounded labels, total chemical hydrogen and joint bond states."""
import torch
from torch.nn import functional as F
from rdkit import Chem
from training.molsight_spatial.data import GraphDataset
from training.molsight_spatial.loss import mine_pair_targets,masked_ce


def chemical_targets(molecule,sample):
        if [a.GetAtomicNum() for a in molecule.GetAtoms()]!=sample['elements'].tolist():
            raise ValueError('Canonical hydrogen labels lost atom correspondence')
        sample=dict(sample)
        sample['hydrogens']=torch.tensor([a.GetTotalNumHs() for a in molecule.GetAtoms()],dtype=torch.long)
        if sample['hydrogens'].max()>8:raise ValueError('Unsupported hydrogen count')
        joint=sample['edges'].clone()
        wedge=sample['wedges']
        for i,j in zip(*torch.where(wedge>0)):
            if joint[i,j]!=1 or wedge[j,i]!=0:raise ValueError('Invalid visible wedge annotation')
            joint[i,j]=4+wedge[i,j];joint[j,i]=6+wedge[i,j]
        sample['joint_bonds']=joint
        sample['double_defined']=sample['double_stereo'].masked_fill(sample['edges']!=2,-100)
        return sample


class DenseDataset(GraphDataset):
    def __init__(self,*args,preprocessing='letterbox',**kwargs):
        super().__init__(*args,**kwargs);self.preprocessing=preprocessing
    def get(self,index,*,epoch):
        sample=super().get(index,epoch=epoch)
        if self.preprocessing=='molsight_native':
            from PIL import Image
            from .inputs import native_input
            record=self.records[index]
            with Image.open(record['file_path']) as image:
                sample['image'],sample['coords'],sample['augmentation']=native_input(image,record['coords_pixels'],self.size,
                    seed=sample['augmentation']['seed'],augment=self.augment)
        return chemical_targets(Chem.MolFromSmiles(sample['smiles']),sample)


def center_targets(points,height,width):
    if points.ndim!=2 or points.shape[-1]!=2 or not len(points):raise ValueError('Invalid atom points')
    scaled=points*points.new_tensor([width,height])
    indices=scaled.floor().long()
    if (indices<0).any() or (indices[:,0]>=width).any() or (indices[:,1]>=height).any():
        raise ValueError('Center outside feature map')
    flat=indices[:,1]*width+indices[:,0]
    if flat.unique().numel()!=len(points):raise ValueError('Two atoms collide in a heatmap cell; increase input resolution')
    target=points.new_zeros((height,width))
    # Finite local Gaussian support avoids O(H*W*N) target construction.
    shifts=torch.cartesian_prod(torch.arange(-3,4,device=points.device),torch.arange(-3,4,device=points.device))
    locations=indices[:,None,:]+shifts.flip(-1)[None,:,:]
    x,y=locations[...,0],locations[...,1]
    valid=(x>=0)&(x<width)&(y>=0)&(y<height)
    values=torch.exp(-shifts.float().square().sum(-1)/2).expand(len(points),-1)
    target.view(-1).scatter_reduce_(0,(y[valid]*width+x[valid]),values[valid],reduce='amax',include_self=True)
    return target,indices,scaled-indices


def center_loss(logits,target):
    value=logits.float();positive=target.eq(1);negative=target.lt(1)
    probability=value.sigmoid()
    positive_loss=-F.logsigmoid(value)*(1-probability).square()*positive
    negative_loss=-F.logsigmoid(-value)*probability.square()*(1-target).pow(4)*negative
    return (positive_loss.sum()+negative_loss.sum())/positive.sum().clamp_min(1)


def balanced_pair_loss(logits,targets,ratio=4.):
    selected=mine_pair_targets(logits,targets,ratio)
    masks=[selected.gt(0),selected.eq(0)]
    terms=[masked_ce(logits,selected.masked_fill(~mask,-100)) for mask in masks if mask.any()]
    return sum(terms)/len(terms) if terms else logits.sum()*0


def objective(model,encoded,sample,*,reader_enabled=True,noise_pixels=.75,chemistry_weight=0.,hard_weight=0.,center_recall_weight=0.,noise_seed=None):
    h,w=encoded['heatmap'].shape[-2:]
    target,indices,offsets=center_targets(sample['coords'],h,w)
    predicted=encoded['offset'][0,:,indices[:,1],indices[:,0]].transpose(0,1)
    terms={'centers':center_loss(encoded['heatmap'][0,0],target),
        'offset':F.l1_loss(predicted.float(),offsets.float())}
    if center_recall_weight:
        from .consistency import center_recall
        terms['center_recall']=center_recall_weight*center_recall(encoded['heatmap'][0,0],indices)
    # Reader labels are supervised only; ordinary inference always uses peaks.
    generator=None
    if noise_seed is not None:
        generator=torch.Generator(device=sample['coords'].device);generator.manual_seed(noise_seed)
    noise=torch.randn(sample['coords'].shape,device=sample['coords'].device,dtype=sample['coords'].dtype,generator=generator)
    points=(sample['coords']+noise*noise_pixels/sample['image'].shape[-1]).clamp(0,1)
    read=model.head.read(encoded,points.unsqueeze(0),sample['image'].shape[-1],with_bonds=reader_enabled)
    for name in ['elements','charge','isotope','hydrogens','radicals','aromatic']:
        terms['state:'+name]=.2*masked_ce(read[name][0],sample[name])
    if 'aliases' in read:
        labels=sample['aliases'];masks=[labels.gt(0),labels.eq(0)]
        alias_terms=[masked_ce(read['aliases'][0],labels.masked_fill(~mask,-100)) for mask in masks if mask.any()]
        terms['state:aliases']=.5*sum(alias_terms)/len(alias_terms)
    if reader_enabled:
        terms['bonds']=balanced_pair_loss(read['joint_bonds'],sample['joint_bonds'].unsqueeze(0))
        terms['double_defined']=.2*masked_ce(read['double_defined'][0],sample['double_defined'])
        if chemistry_weight:
            from .consistency import chemical_consistency
            terms.update({'chemistry:'+k:chemistry_weight*v for k,v in chemical_consistency(read,sample).items()})
    if hard_weight:
        from .consistency import hard_atom_loss
        terms['hard_atoms']=hard_weight*hard_atom_loss(read,sample)
    if hasattr(model.head,'validity'):
        from training.molsight_proposals.objectives import validity_loss
        terms['proposal_validity']=.5*validity_loss(model.head,encoded,sample,points,0 if noise_seed is None else noise_seed+1777)
    if not all(torch.isfinite(v).all() for v in terms.values()):raise FloatingPointError('Nonfinite dense objective')
    return sum(terms.values()),terms

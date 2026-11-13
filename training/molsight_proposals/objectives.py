import torch
from torch.nn import functional as F


def negative_points(encoded,sample,seed,max_count=128):
    """Training-only negatives: real wrong peaks and ink without a true center."""
    probability=encoded['heatmap'][0,0].detach().float().sigmoid();h,w=probability.shape
    local=F.max_pool2d(probability[None,None],3,1,1)[0,0]
    y,x=torch.where((probability>=.1)&probability.eq(local))
    offset=encoded['offset'][0,:,y,x].detach().float().T
    peaks=(torch.stack((x,y),-1)+offset)/offset.new_tensor([w,h])
    gt=sample['coords'];edges=sample['edges'];i,j=torch.where(torch.triu(edges>0,diagonal=1))
    radius=(torch.linalg.vector_norm(gt[i]-gt[j],dim=-1).median()*.3 if len(i) else gt.new_tensor(.01))
    if len(peaks):
        valid=torch.cdist(peaks,gt).amin(-1)>radius
        ids=torch.where(valid)[0];ids=ids[probability[y[ids],x[ids]].argsort(descending=True)[:max_count//2]]
        peaks=peaks[ids]
    # This is foreground mining, not a uniformly white-background classifier.
    iy,ix=torch.where(sample['image'][0]<.7)
    generator=torch.Generator(device=gt.device);generator.manual_seed(seed)
    if len(ix):
        ids=torch.randperm(len(ix),device=gt.device,generator=generator)[:max_count*8]
        ink=torch.stack((ix[ids],iy[ids]),-1).float()/sample['image'].shape[-1]
        ink=ink[torch.cdist(ink,gt).amin(-1)>radius][:max_count-len(peaks)]
        peaks=torch.cat((peaks,ink))
    return peaks.detach()


def validity_loss(head,encoded,sample,points,seed):
    positive=head.proposal_logits(encoded,points.unsqueeze(0))[0].float()
    terms=[F.cross_entropy(positive,torch.ones(len(points),dtype=torch.long,device=points.device))]
    negatives=negative_points(encoded,sample,seed)
    if len(negatives):
        logits=head.proposal_logits(encoded,negatives.unsqueeze(0))[0].float()
        terms.append(F.cross_entropy(logits,torch.zeros(len(negatives),dtype=torch.long,device=points.device)))
    return torch.stack(terms).mean()

"""Sample chemical identity before drawing view to avoid duplicate-view bias."""
from training.ocsr_reboot.protocol import canonical_smiles
import numpy as np


def identity_pools(records,view_pools):
    result={}
    for domain,buckets in view_pools.items():
        result[domain]={}
        for size,indices in buckets.items():
            pool={}
            for index in indices:
                identity=canonical_smiles(records[index]['SMILES'])
                pool.setdefault(identity,[]).append(index)
            result[domain][size]=pool
    return result


def choose_view(pool,rng):
    identity=sorted(pool)[int(rng.integers(len(pool)))]
    views=pool[identity]
    return views[int(rng.integers(len(views)))]


def choose_size(buckets,rng,power=0.):
    sizes=sorted(buckets)
    if not 0<=power<=1:raise ValueError('Size prior power must be in [0,1]')
    if power==0:return int(rng.choice(sizes))
    counts=np.asarray([len(buckets[size]) for size in sizes],float)
    weights=counts**power;weights/=weights.sum()
    return int(rng.choice(sizes,p=weights))

"""Use MolSight's established white-crop geometry with synchronized point labels."""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from PIL import Image
from molsight.dataset import get_transforms
from training.molsight_spatial.augment import transform


@lru_cache(maxsize=8)
def geometry(size):
    # Debug mode omits normalization; photometric transforms and normalization
    # are applied exactly once afterwards by the shared coordinate transform.
    return get_transforms(SimpleNamespace(input_size=size,encoder='efficientvit'),augment=False,rotate=False,debug=True)


def native_input(image,points,size,*,seed=0,augment=False):
    prepared=geometry(size)(image=np.asarray(image.convert('RGB')),keypoints=np.asarray(points,dtype=np.float32).reshape(-1,2))
    pixels=Image.fromarray(prepared['image'])
    tensor,coords,trace=transform(pixels,np.asarray(prepared['keypoints']).reshape(-1,2),size,seed=seed,augment=augment)
    trace['preprocessing']='molsight_native_white_crop'
    return tensor,coords,trace

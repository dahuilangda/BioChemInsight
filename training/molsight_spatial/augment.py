"""Deterministic, orientation-preserving image/coordinate transforms."""
import io
import cv2
import numpy as np
from PIL import Image
import torch


def transform(image, points, size, *, seed=0, augment=False):
    image = np.asarray(image.convert('RGB'))
    points = np.asarray(points, dtype=np.float64)
    h,w = image.shape[:2]
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Invalid pixel points')
    if (points<0).any() or (points[:,0]>=w).any() or (points[:,1]>=h).any():raise ValueError('Source coordinates out of bounds')
    factor = min(size/w,size/h)
    width,height = max(1,round(w*factor)),max(1,round(h*factor))
    resized = cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC)
    left,top = (size-width)//2,(size-height)//2
    canvas=np.full((size,size,3),255,np.uint8);canvas[top:top+height,left:left+width]=resized
    xy=points*np.array([width/w,height/h])+[left,top]
    trace={'seed':int(seed),'augment':augment,'resize':[width,height],'padding':[left,top]}
    if augment:
        rng=np.random.default_rng(seed)
        ink=np.any(canvas<245,axis=-1); ys,xs=np.where(ink)
        corners=np.array([[xs.min(),ys.min()],[xs.max(),ys.min()],[xs.max(),ys.max()],[xs.min(),ys.max()]]) if len(xs) else xy
        accepted=False
        for _ in range(8):
            angle=float(rng.uniform(-4,4));scale=float(rng.uniform(.96,1.02))
            matrix=cv2.getRotationMatrix2D((size/2,size/2),angle,scale)
            matrix[:,2]+=rng.uniform(-.015,.015,2)*size
            transformed=xy@matrix[:,:2].T+matrix[:,2]
            bounds=corners@matrix[:,:2].T+matrix[:,2]
            if (bounds>=2).all() and (bounds<size-2).all() and (transformed>=0).all() and (transformed<size).all():
                canvas=cv2.warpAffine(canvas,matrix,(size,size),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
                xy=transformed;trace['affine']=matrix.tolist();accepted=True;break
        trace['affine_accepted']=accepted
        if rng.random()<.4:
            quality=int(rng.integers(65,96));encoded=io.BytesIO();Image.fromarray(canvas).save(encoded,format='JPEG',quality=quality)
            canvas=np.array(Image.open(io.BytesIO(encoded.getvalue())).convert('RGB'));trace['jpeg_quality']=quality
        if rng.random()<.3:
            sigma=float(rng.uniform(.2,.7));canvas=cv2.GaussianBlur(canvas,(3,3),sigma);trace['blur_sigma']=sigma
        if rng.random()<.3:
            shrink=float(rng.uniform(.7,.95));side=max(1,round(size*shrink))
            canvas=cv2.resize(cv2.resize(canvas,(side,side),interpolation=cv2.INTER_AREA),(size,size),interpolation=cv2.INTER_LINEAR)
            trace['resolution_factor']=shrink
    gray=cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)
    value=np.repeat(gray[...,None],3,axis=-1).astype(np.float32)/255
    value=(value-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(value.transpose(2,0,1).copy()),torch.tensor(xy/size,dtype=torch.float32),trace

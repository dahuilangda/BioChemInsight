"""Data admission for the actual image crop; no coordinate correction/clipping."""
import numpy as np


def assert_native_crop_contains_points(image,points,pad=5):
    pixels=np.asarray(image.convert('RGB'));height,width=pixels.shape[:2]
    xy=np.asarray(points,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(xy).all() or (xy<0).any() or (xy[:,0]>=width).any() or (xy[:,1]>=height).any():raise ValueError('Original point outside image')
    ink=np.any(pixels!=255,axis=-1);y,x=np.where(ink)
    if not len(x):raise ValueError('Blank drawing')
    left,top,right,bottom=int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1
    shifted=xy-[left,top]+pad;size=np.array([right-left+2*pad,bottom-top+2*pad])
    if (shifted<0).any() or (shifted>=size).any():raise ValueError('Observed unit outside native white crop')
    return {'ink_bbox':[left,top,right,bottom],'pad':pad,'points_survive_crop':True}

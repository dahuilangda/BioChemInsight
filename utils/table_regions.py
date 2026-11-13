"""Host-owned table windows bound dense visual transcription without guessing cells."""
from pathlib import Path
import re
import unicodedata


def _centers(mask,axis,minimum):
    import numpy as np
    hits=np.flatnonzero((mask>0).sum(axis=axis)>=minimum)
    groups=[]
    for index in hits:
        if not groups or index-groups[-1][-1]>3:groups.append([int(index)])
        else:groups[-1].append(int(index))
    return [round(sum(group)/len(group)) for group in groups]


def table_windows(page,image_path,output_dir,max_rows=24,max_cells=64):
    import cv2
    import numpy as np
    from PIL import Image
    image=Image.open(image_path).convert('RGB');gray=np.asarray(image.convert('L'))
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,12)
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(30,image.width//30),1)))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(30,image.height//35))))
    contours,_=cv2.findContours(cv2.bitwise_or(horizontal,vertical),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    grids=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if w<image.width*.25 or w*h<image.width*image.height*.035:continue
        rows=[y+v for v in _centers(horizontal[y:y+h,x:x+w],1,w*.65)]
        columns=[x+v for v in _centers(vertical[y:y+h,x:x+w],0,h*.65)]
        if len(rows)>=4 and len(columns)>=3:grids.append((x,y,w,h,rows,columns))
    grids.sort(key=lambda grid:(grid[1],grid[0]))
    if not grids:return []
    words=page.get_text('words');sx=image.width/page.rect.width;sy=image.height/page.rect.height
    context=' '.join(str(word[4]) for word in words if not any(x<=((word[0]+word[2])/2*sx)<=x+w and y<=((word[1]+word[3])/2*sy)<=y+h for x,y,w,h,_,_ in grids))[:5000]
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    windows=[]
    for table_index,(x,y,w,h,rows,columns) in enumerate(grids):
        headers=[]
        for left,right in zip(columns,columns[1:]):
            text=' '.join(str(word[4]) for word in words if left<((word[0]+word[2])/2*sx)<right and rows[0]<((word[1]+word[3])/2*sy)<rows[1])
            headers.append(re.sub(r'\W+','',unicodedata.normalize('NFKC',text).casefold()))
        repeat=[i for i,text in enumerate(headers) if headers and headers[0] and text==headers[0]]
        stride=repeat[1] if len(repeat)>1 else len(headers)
        groups=[(i,min(i+stride,len(headers))) for i in range(0,len(headers),stride)] if stride>0 and repeat==list(range(0,len(headers),stride)) else [(0,len(headers))]
        for group_index,(first,last) in enumerate(groups):
            limit=max(1,min(max_rows,max_cells//max(1,last-first)))
            for start in range(1,len(rows)-1,limit):
                end=min(len(rows)-1,start+limit)
                left,right=columns[first],columns[last]
                header=image.crop((left,rows[0],right+2,rows[1]+2))
                body=image.crop((left,rows[start],right+2,rows[end]+2))
                combined=Image.new('RGB',(max(header.width,body.width),header.height+body.height+8),'white')
                combined.paste(header,(0,0));combined.paste(body,(0,header.height+8))
                path=output/f'table_{table_index}_group_{group_index}_rows_{start}_{end}.png';combined.save(path)
                windows.append({'image':str(path),'table_index':table_index,'group_index':group_index,'row_start':start,'row_end':end,
                    'source_region':f'grid:{table_index}:columns:{first}-{last}:rows:{start}-{end}',
                    'bbox':[left/image.width,rows[start]/image.height,right/image.width,rows[end]/image.height],
                    'context':context})
    # Small pages need no extra requests. A grid of a few cells can also be a
    # figure frame; keep the full-page path unless tiling actually reduces it.
    return windows if len(windows)>1 else []

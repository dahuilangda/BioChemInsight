"""Image-only experimental recognizer CLI; no production-model replacement."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .artifact import load_artifact,export_artifact
from .augment import transform
from .data import digest


def main(args):
    torch.set_num_threads(2)
    model,meta=load_artifact(args.artifact,args.device)
    with Image.open(args.image) as image:
        tensor,_,trace=transform(image,np.empty((0,2)),meta['input_size'],augment=False)
    with torch.no_grad():result=model.predict_image(tensor.unsqueeze(0).to(args.device),objectness_threshold=meta['objectness_threshold'])
    result.update(input_sha256=digest(args.image),artifact_sha256=digest(args.artifact/'artifact.json'),
        optimizer_updates=0,scope='Image-only spatial graph; no oracle inputs or AR-as-spatial fallback.')
    text=json.dumps(result,indent=2)
    if args.out:args.out.write_text(text+'\n')
    else:print(text)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--out',type=Path);main(parser.parse_args())

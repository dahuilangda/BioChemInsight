"""Evaluate a dense artifact on an admitted development manifest or frozen real pairs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from training.molsight_spatial.augment import transform
from training.molsight_spatial.data import digest
from training.ocsr_reboot.checkpoint import tree_sha256
from training.ocsr_reboot.protocol import canonical_smiles
from training.molsight_spatial.train import write
from .artifact import load_artifact
from .objectives import DenseDataset
from .train import evaluate


def main(args):
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=False)
    model,meta=load_artifact(args.artifact,args.device)
    before=tree_sha256(model.state_dict())
    if args.manifest:
        if 'aliases' in model.head.states:
            from training.molsight_visible.data import VisibleDataset
            dataset_type=VisibleDataset
        else:dataset_type=DenseDataset
        dataset=dataset_type(args.manifest,'validation',meta['input_size'],augment=False,preprocessing=meta.get('preprocessing','letterbox'))
        result=evaluate(model,dataset,args.device);write(args.out/'report.json',result)
    else:
        rows=json.loads(args.pairs.read_text());keys=[r['record_key'] for r in rows]
        if len(set(keys))!=len(keys) or not keys:raise ValueError('Invalid frozen cohort')
        for row in rows:
            canonical_smiles(row['SMILES'])
            if digest(row['file_path'])!=row['image_sha256']:raise ValueError('Cohort image changed')
        write(args.out/'plan.json',{'cohort_sha256':digest(args.pairs),'artifact_sha256':digest(args.artifact/'artifact.json'),
            'evaluator_sha256':digest(__file__),'records':len(rows),'optimizer_updates':0,'image_only':True})
        count=exact=parsed=0;groups={}
        with (args.out/'predictions.jsonl').open('x') as stream,torch.inference_mode():
            for row in rows:
                with Image.open(row['file_path']) as original:
                    if meta.get('preprocessing')=='molsight_native':
                        from .inputs import native_input
                        image,_,_=native_input(original,np.empty((0,2)),meta['input_size'],augment=False)
                    else:image,_,_=transform(original,np.empty((0,2)),meta['input_size'],augment=False)
                prediction=model.predict_image(image.unsqueeze(0).to(args.device))
                success=prediction['parse_success'] and canonical_smiles(prediction['smiles'])==canonical_smiles(row['SMILES'])
                result={'record_key':row['record_key'],'component':row['component'],'exact':bool(success),'prediction':prediction}
                stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
                count+=1;exact+=int(success);parsed+=int(prediction['parse_success'])
                for name in ['peptide_like_candidate','cyclic_peptide_like_candidate']:
                    if row.get(name):
                        group=groups.setdefault(name,{'count':0,'exact':0});group['count']+=1;group['exact']+=int(success)
                if count%25==0:write(args.out/'status.json',{'state':'running','completed':count,'exact':exact})
        if any(digest(r['file_path'])!=r['image_sha256'] for r in rows):raise ValueError('Image changed during evaluation')
        write(args.out/'report.json',{'records':count,'exact':exact,'parsed':parsed,'groups':groups,'optimizer_updates':0})
    if before!=tree_sha256(model.state_dict()):raise RuntimeError('Inference changed model state')
    write(args.out/'status.json',{'state':'complete','model_byte_unchanged':True,'optimizer_updates':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','out']:parser.add_argument('--'+name,type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True);source.add_argument('--manifest',type=Path);source.add_argument('--pairs',type=Path)
    parser.add_argument('--device',default='cuda:1');main(parser.parse_args())

"""Calibrate one global peak threshold on development data, never using test gold."""
import argparse,json
from dataclasses import replace
from collections import defaultdict
from pathlib import Path
import torch
from training.molsight_dense.artifact import load_artifact
from training.molsight_dense.objectives import DenseDataset
from training.molsight_visible.data import VisibleDataset
from training.molsight_visible.model import decode_visible
from training.molsight_dense.model import decode_graph
from training.molsight_spatial.data import digest
from training.ocsr_reboot.protocol import canonical_smiles
from training.ocsr_reboot.checkpoint import tree_sha256


def main(args):
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=False)
    model,meta=load_artifact(args.artifact,args.device);before=tree_sha256(model.state_dict())
    visible='aliases' in model.head.states
    dataset_type=VisibleDataset if visible else DenseDataset
    dataset=dataset_type(args.manifest,'validation',meta['input_size'],augment=False,preprocessing=meta.get('preprocessing','letterbox'))
    thresholds=sorted(set(args.thresholds));original=model.head.config;details=[];counts={str(t):defaultdict(lambda:{'records':0,'exact':0}) for t in thresholds}
    with torch.inference_mode():
        for i,record in enumerate(dataset.records):
            sample=dataset.get(i,epoch=0);encoded=model.encode(sample['image'].unsqueeze(0).to(args.device))
            model.head.config=replace(original,peak_threshold=thresholds[0])
            try:
                points,scores=model.head.peaks(encoded)
                states=model.head.read(encoded,points,meta['input_size']) if points.shape[1] else None
            except ValueError:points=scores=states=None
            predicted={}
            for threshold in thresholds:
                try:
                    if states is None:
                        model.head.config=replace(original,peak_threshold=threshold)
                        selected,_=model.head.peaks(encoded)
                        if not selected.shape[1]:raise ValueError('No peaks')
                        read=model.head.read(encoded,selected,meta['input_size'])
                    else:
                        keep=torch.where(scores>=threshold)[0];selected=points[:,keep]
                        if not len(keep):raise ValueError('No peaks')
                        read={name:(value[:,keep[:,None],keep[None,:]] if value.ndim==4 else value[:,keep]) for name,value in states.items()}
                    prediction=(decode_visible if visible else decode_graph)(selected,read)
                except ValueError:prediction={'parse_success':False,'smiles':None}
                predicted[str(threshold)]=prediction
            # All candidates above are predictions; gold is consulted only for scoring.
            gold=canonical_smiles(sample['smiles']);row={'record_key':record['record_key'],'domain':record['domain'],'exact':{}}
            for threshold,prediction in predicted.items():
                exact=prediction['parse_success'] and canonical_smiles(prediction['smiles'])==gold
                row['exact'][threshold]=bool(exact);counts[threshold][record['domain']]['records']+=1;counts[threshold][record['domain']]['exact']+=int(exact)
            details.append(row)
    model.head.config=original
    assert tree_sha256(model.state_dict())==before
    base=counts[str(thresholds[0])]
    eligible=[t for t in thresholds if all(counts[str(t)][domain]['exact']>=base[domain]['exact'] for domain in base)]
    best=max(eligible,key=lambda t:(sum(v['exact']/v['records'] for v in counts[str(t)].values()),-t))
    report={'artifact_sha256':digest(args.artifact/'artifact.json'),'development_manifest_sha256':digest(args.manifest),
        'calibrator_sha256':digest(__file__),'scope':'Development-only global calibration; no per-domain or per-image thresholds, no test labels.',
        'threshold':best,'baseline_threshold':thresholds[0],'counts':counts,'selection':'Highest macro exact without reducing any development domain relative to the lowest threshold.','rows':details}
    (args.out/'policy.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'threshold':best,'counts':counts}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','manifest','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--thresholds',type=float,nargs='+',default=[.3,.4,.5,.6,.7,.8]);parser.add_argument('--device',default='cuda:1');main(parser.parse_args())

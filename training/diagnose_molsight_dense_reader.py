"""Oracle-location reader ablation. NEVER an image-only molecular accuracy score."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from training.molsight_dense.artifact import load_artifact
from training.molsight_dense.objectives import DenseDataset
from training.molsight_dense.model import decode_graph
from training.molsight_spatial.data import digest
from training.ocsr_reboot.protocol import canonical_smiles
from training.ocsr_reboot.checkpoint import tree_sha256


def main(args):
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=False)
    model,meta=load_artifact(args.artifact,args.device);before=tree_sha256(model.state_dict())
    dataset=DenseDataset(args.manifest,'validation',meta['input_size'],augment=False)
    rows=[]
    for i,record in enumerate(dataset.records):
        sample=dataset.get(i,epoch=0);points=sample['coords'].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            encoded=model.encode(sample['image'].unsqueeze(0).to(args.device))
            full=model.head.read(encoded,points,meta['input_size'])
            original=model.head.bonds.tube_features
            def zero_line(*inputs):return torch.zeros_like(original(*inputs))
            with patch.object(model.head.bonds,'tube_features',side_effect=zero_line):
                removed=model.head.read(encoded,points,meta['input_size'])
            metrics={}
            for name,read in [('visual_line',full),('line_removed',removed)]:
                predicted=read['joint_bonds'][0].argmax(-1).cpu();gold=sample['joint_bonds']
                upper=torch.ones_like(gold,dtype=torch.bool).triu(diagonal=1)
                truth=upper&gold.gt(0);positive=upper&predicted.gt(0)
                true=int((truth&predicted.eq(gold)).sum());p=int(positive.sum());g=int(truth.sum())
                graph=decode_graph(points,read)
                exact=graph['parse_success'] and canonical_smiles(graph['smiles'])==canonical_smiles(sample['smiles'])
                metrics[name]={'tp':true,'pred':p,'gold':g,'oracle_location_exact':bool(exact)}
        rows.append({'record_key':record['record_key'],'domain':record['domain'],**metrics})
    summary={}
    for domain in sorted({r['domain'] for r in rows}):
        selected=[r for r in rows if r['domain']==domain];summary[domain]={}
        for name in ['visual_line','line_removed']:
            counts={k:sum(r[name][k] for r in selected) for k in ['tp','pred','gold','oracle_location_exact']}
            summary[domain][name]={**counts,'bond_f1':2*counts['tp']/max(1,counts['pred']+counts['gold'])}
    assert tree_sha256(model.state_dict())==before
    report={'scope':'ORACLE ATOM LOCATION DIAGNOSTIC; not image-only recognition accuracy. Only the bond-tube branch is zeroed; endpoint features and weights are unchanged.',
        'manifest_sha256':digest(args.manifest),'artifact_sha256':digest(args.artifact/'artifact.json'),'diagnostic_sha256':digest(__file__),'summary':summary,'rows':rows}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','manifest','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');main(parser.parse_args())

"""Post-inference atom/bond diagnostics on a labelled development manifest.

Gold participates only in scoring an already-computed image-only prediction.
These are diagnostics, never substitutes for molecular exact or real-patent tests.
"""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from training.molsight_spatial.artifact import load_artifact
from training.molsight_spatial.data import GraphDataset,digest
from training.ocsr_reboot.checkpoint import tree_sha256


def ratios(row,prefix):
    true,predicted,gold=(row[prefix+'_'+k] for k in ['true','predicted','gold'])
    return {'precision':true/predicted if predicted else 0.,'recall':true/gold if gold else 0.,
        'f1':2*true/(predicted+gold) if predicted+gold else 0.}


def main(args):
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    model,meta=load_artifact(args.artifact,args.device);before=tree_sha256(model.state_dict())
    groups=defaultdict(Counter);rows=[]
    for split in args.splits:
        dataset=GraphDataset(args.manifest,split,meta['input_size'],augment=False)
        for i,record in enumerate(dataset.records):
            sample=dataset.get(i,epoch=0)
            with torch.inference_mode():outputs=model(sample['image'].unsqueeze(0).to(args.device))
            # All selection and chemical decisions precede access to reference targets.
            chosen=torch.where(outputs['objectness'][0].sigmoid()>meta['objectness_threshold'])[0]
            coordinates=outputs['coords'][0,chosen].float().cpu()
            elements=outputs['elements'][0,chosen].argmax(-1).cpu()
            probability=outputs['edges'][0][chosen[:,None],chosen[None,:]].float().softmax(-1)
            predicted_bonds=((probability+probability.transpose(0,1))*.5).argmax(-1).cpu()
            predicted_states={name:outputs[name][0,chosen].argmax(-1).cpu() for name in ['charge','isotope','explicit_h','no_implicit','radicals','aromatic']}
            gold_edges=sample['edges'];gold_bond_indices=torch.where(torch.triu(gold_edges>0,diagonal=1))
            lengths=torch.linalg.vector_norm(sample['coords'][gold_bond_indices[0]]-sample['coords'][gold_bond_indices[1]],dim=-1)
            radius=float(lengths.median())*.3 if len(lengths) else .01
            mapping={}
            geometry_matches=0
            if len(chosen):
                distances=torch.cdist(coordinates,sample['coords'])
                geometry_valid=distances<=radius
                left,right=linear_sum_assignment((distances+~geometry_valid*1e6).numpy())
                geometry_matches=sum(int(geometry_valid[a,b]) for a,b in zip(left,right))
                valid=(distances<=radius)&elements[:,None].eq(sample['elements'][None,:])
                cost=distances+~valid*1e6
                left,right=linear_sum_assignment(cost.numpy())
                mapping={int(a):int(b) for a,b in zip(left,right) if valid[a,b]}
            true_bonds=0
            for a,b in zip(*torch.where(torch.triu(predicted_bonds>0,diagonal=1))):
                a,b=int(a),int(b)
                if a in mapping and b in mapping and predicted_bonds[a,b]==gold_edges[mapping[a],mapping[b]]:true_bonds+=1
            row={'record_key':record['record_key'],'split':split,'domain':record['domain'],
                'atom_true':len(mapping),'atom_predicted':len(chosen),'atom_gold':len(sample['coords']),
                'geometry_true':geometry_matches,'geometry_predicted':len(chosen),'geometry_gold':len(sample['coords']),
                'count_absolute_error':abs(len(chosen)-len(sample['coords'])),'count_exact':int(len(chosen)==len(sample['coords'])),
                'bond_true':true_bonds,'bond_predicted':int(torch.triu(predicted_bonds>0,diagonal=1).sum()),
                'bond_gold':len(gold_bond_indices[0]),'matching_radius_normalized':radius}
            for name,predicted in predicted_states.items():
                row['state_correct:'+name]=sum(int(predicted[a]==sample[name][b]) for a,b in mapping.items())
            row['state_all_correct']=sum(int(all(predicted[a]==sample[name][b] for name,predicted in predicted_states.items())) for a,b in mapping.items())
            rows.append(row)
            for group in [split+'/all',split+'/'+record['domain']]:
                groups[group]['records']+=1
                for key in ['atom_true','atom_predicted','atom_gold','geometry_true','geometry_predicted','geometry_gold',
                    'bond_true','bond_predicted','bond_gold','count_absolute_error','count_exact','state_all_correct',
                    *['state_correct:'+name for name in predicted_states]]:groups[group][key]+=row[key]
    assert tree_sha256(model.state_dict())==before
    report={'manifest_sha256':digest(args.manifest),'artifact_sha256':digest(args.artifact/'artifact.json'),
        'evaluator_sha256':digest(__file__),'optimizer_updates':0,
        'scope':'Image-only predictions; typed-atom matching within 0.3 median gold bond length is scoring only. Not molecular exact.',
        'groups':{key:{**value,'atom':ratios(value,'atom'),'geometry':ratios(value,'geometry'),'bond':ratios(value,'bond'),
            'count_mae':value['count_absolute_error']/value['records'],
            'conditional_state_accuracy':{name:value['state_correct:'+name]/value['atom_true'] if value['atom_true'] else 0. for name in predicted_states}}
            for key,value in groups.items()},'rows':rows}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','manifest','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--splits',nargs='+',default=['validation'])
    main(parser.parse_args())

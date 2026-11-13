"""Score cached real branch predictions; selection never receives the reference."""
import argparse,json,re
from pathlib import Path
from collections import defaultdict,Counter
import torch
from training.molsight_hybrid import select_prediction,complete_ar
from training.molsight_spatial.data import digest
from training.ocsr_reboot.protocol import canonical_smiles


def main(args):
    cohort=json.loads(args.pairs.read_text());keys={r['record_key'] for r in cohort}
    if len(keys)!=len(cohort):raise ValueError('Duplicate evaluation keys')
    ar_rows=[json.loads(x) for x in args.ar.read_text().splitlines()];ar={r['record_key']:r for r in ar_rows}
    ar_report=json.loads((args.ar.parent/'report.json').read_text())
    ar_pairs=json.loads((args.ar_pairs or args.pairs).read_text())
    if digest(args.ar_pairs or args.pairs)!=ar_report['pairs_sha256']:raise ValueError('AR cohort provenance mismatch')
    ar_inputs={r['record_key']:(r['file_path'],r['image_sha256']) for r in ar_pairs}
    meta=json.loads((args.artifact/'artifact.json').read_text())
    payload=torch.load(args.artifact/'model.pth',map_location='cpu',mmap=True,weights_only=False)
    if ar_report['checkpoint_sha256']!=payload['provenance']['base_sha256']:raise ValueError('Different MolSight base branches')
    if digest(args.artifact/'model.pth')!=meta['model_sha256']:raise ValueError('Artifact changed')
    if args.visual.suffix=='.json':
        content=json.loads(args.visual.read_text());visual_rows=content['rows']
        match=re.fullmatch(r'evaluation_(\d+)\.json',args.visual.name)
        if match and digest(args.visual.parent/f'update_{int(match[1]):06d}.pth')!=meta['model_sha256']:raise ValueError('Wrong checkpoint for cached development predictions')
    else:
        visual_rows=[json.loads(x) for x in args.visual.read_text().splitlines()]
        plan=json.loads((args.visual.parent/'plan.json').read_text())
        if plan['artifact_sha256']!=digest(args.artifact/'artifact.json'):raise ValueError('Visual cache artifact mismatch')
    visual={r['record_key']:r.get('prediction',r.get('spatial')) for r in visual_rows}
    if not keys<=set(ar) or not keys<=set(visual):raise ValueError('Incomplete cached branches')
    summary=defaultdict(lambda:defaultdict(Counter));details=[];regressions=defaultdict(list);contract_differences=[]
    for row in cohort:
        key=row['record_key']
        if ar_inputs[key]!=(row['file_path'],row['image_sha256']):raise ValueError('Branches evaluated different image bytes')
        gold=canonical_smiles(row['SMILES'])
        if canonical_smiles(ar[key]['gold'])!=gold:raise ValueError('Different chemical reference in cached branches')
        ar_text=complete_ar(ar[key]['prediction'],ar[key]['terminated'])
        ar_exact=ar_text==gold
        # Reference access happens only here, after each gold-blind selection.
        choices={name:select_prediction(ar[key]['prediction'],ar[key]['terminated'],visual[key],policy=name)
            for name in ['preserve_ar','cyclic_visible','peptide_visible']}
        outputs={'ar':{'parse_success':ar_text is not None,'smiles':ar_text},'visual':visual[key],**choices}
        scores={name:bool(value.get('parse_success')) and canonical_smiles(value['smiles'])==gold for name,value in outputs.items()}
        if scores['ar']!=bool(ar[key]['resolved_exact']):
            contract_differences.append({'record_key':key,'cached_exact':bool(ar[key]['resolved_exact']),
                'resolved_canonical_exact':scores['ar'],'raw_prediction':ar[key]['prediction'],'resolved_prediction':ar_text,'gold':gold})
        outputs['ar_cached']={'parse_success':bool(ar[key].get('parse_success',False))}
        scores['ar_cached']=bool(ar[key]['resolved_exact'])
        for name in choices:
            if ar_exact and not scores[name]:regressions[name].append(key)
        if ar_exact and not scores['preserve_ar']:raise AssertionError('Conservative policy lost a correct AR result')
        groups=['all']+([row['domain']] if row.get('domain') else [])
        for flag in ['peptide_like_candidate','cyclic_peptide_like_candidate']:
            if row.get(flag):groups.append(flag)
        for group in groups:
            for name,value in outputs.items():
                summary[group][name]['records']+=1;summary[group][name]['exact']+=int(scores[name]);summary[group][name]['parse']+=int(value.get('parse_success',False))
        details.append({'record_key':key,'component':row['component'],'scores':scores,'selected_branches':{name:value['branch'] for name,value in choices.items()}})
    args.out.mkdir(parents=True,exist_ok=False)
    report={'scope':'Explicit cached branch composition, never spatial-only accuracy. No reference fields are passed to the selector.',
        'cohort_sha256':digest(args.pairs),'ar_predictions_sha256':digest(args.ar),'visual_predictions_sha256':digest(args.visual),
        'selector_sha256':digest('training/molsight_hybrid.py'),'artifact_sha256':digest(args.artifact/'artifact.json'),
        'summary':{k:{name:dict(value) for name,value in group.items()} for k,group in summary.items()},
        'AR_scoring_contract_differences':contract_differences,
        'AR_comparison_contract':'Compare hybrid against ar using identical resolution/canonicalization; ar_cached retains the historic metric separately.',
        'AR_correct_overridden':dict(regressions),'rows':details}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['ar','visual','pairs','artifact','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ar-pairs',type=Path);main(parser.parse_args())

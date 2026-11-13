"""Image-only resolution cascade, preserving every valid primary prediction.

Read the original pixels again at each declared resolution. Never combine
atoms, edit bonds or use reference fields to choose a prediction.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from training.molsight_dense.artifact import load_artifact
from training.molsight_dense.inputs import native_input
from training.molsight_spatial.augment import transform
from training.molsight_spatial.data import digest
from training.ocsr_reboot.checkpoint import tree_sha256
from training.ocsr_reboot.protocol import canonical_smiles


def first_valid(primary,alternatives):
    if primary.get('parse_success'):return primary
    for predict in alternatives:
        try:
            prediction=predict()
        except Exception as error:
            # One failing resolution (e.g. OOM at 1024) must not kill the
            # cascade; later scales still get their chance.
            prediction={'parse_success':False,'error':f'{type(error).__name__}: {error}'}
        if prediction.get('parse_success'):return prediction
    return primary


@torch.inference_mode()
def predict_file(model,metadata,path,sizes=(768,1024,512),primary=None):
    if sizes[0]!=metadata['input_size'] or len(set(sizes))!=len(sizes) or any(n<128 or n%32 for n in sizes):
        raise ValueError('Primary resolution must match artifact; alternatives must be distinct multiples of 32')
    model.eval();device=next(model.parameters()).device;attempts=[]
    with Image.open(path) as original:
        image=original.convert('RGB')
        def predict(size):
            prep=native_input if metadata.get('preprocessing')=='molsight_native' else transform
            pixels,_,_=prep(image,np.empty((0,2)),size,augment=False)
            result=model.predict_image(pixels.unsqueeze(0).to(device));result={**result,'input_resolution':size}
            attempts.append({'size':size,'parse_success':bool(result.get('parse_success')),'error':result.get('error')})
            return result
        if primary is None:
            try:
                base=predict(sizes[0])
            except Exception as error:
                base={'parse_success':False,'error':f'{type(error).__name__}: {error}','input_resolution':sizes[0]}
        else:
            base={**primary,'input_resolution':sizes[0]}
        if primary is not None:attempts.append({'size':sizes[0],'cached':True,'parse_success':bool(base.get('parse_success')),'error':base.get('error')})
        result=first_valid(base,(lambda size=size:predict(size) for size in sizes[1:]))
    return {**result,'resolution_attempts':attempts,'selection':'first_strictly_assembled_graph_in_fixed_resolution_order'}


def main(a):
    torch.set_num_threads(2);rows=json.loads(a.pairs.read_text());a.out.mkdir(parents=True,exist_ok=False)
    if not rows or len({r['record_key'] for r in rows})!=len(rows):raise ValueError('Invalid cohort')
    for r in rows:
        canonical_smiles(r['SMILES'])
        if digest(r['file_path'])!=r['image_sha256']:raise ValueError('Image changed')
    model,metadata=load_artifact(a.artifact,a.device);before=tree_sha256(model.state_dict());cached={}
    if a.primary_cache:
        plan=json.loads((a.primary_cache.parent/'plan.json').read_text())
        if plan['artifact_sha256']!=digest(a.artifact/'artifact.json') or plan['cohort_sha256']!=digest(a.pairs):raise ValueError('Cache provenance mismatch')
        if plan.get('inference_policy_sha256'):raise ValueError('Calibrated cache needs an explicitly matching runtime policy')
        cached={r['record_key']:r['spatial'] for r in map(json.loads,a.primary_cache.read_text().splitlines())}
        if set(cached)!={r['record_key'] for r in rows}:raise ValueError('Incomplete cache')
    plan={'artifact_sha256':digest(a.artifact/'artifact.json'),'cohort_sha256':digest(a.pairs),'sizes':a.sizes,
        'evaluator_sha256':digest(__file__),'primary_cache_sha256':digest(a.primary_cache) if a.primary_cache else None,
        'image_only':True,'optimizer_updates':0,'selection':'fixed resolution order, first strict assembly; no graph repair'}
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');exact=parsed=0;began=time.monotonic();scales={}
    with (a.out/'predictions.jsonl').open('x') as stream:
        for i,r in enumerate(rows):
            result=predict_file(model,metadata,r['file_path'],a.sizes,cached.get(r['record_key']))
            ok=bool(result.get('parse_success')) and canonical_smiles(result['smiles'])==canonical_smiles(r['SMILES'])
            exact+=int(ok);parsed+=int(bool(result.get('parse_success')));size=result['input_resolution'];scales[str(size)]=scales.get(str(size),0)+1
            stream.write(json.dumps({'record_key':r['record_key'],'component':r['component'],'spatial':result,'spatial_exact':ok})+'\n');stream.flush()
            if (i+1)%25==0:(a.out/'status.json').write_text(json.dumps({'completed':i+1,'exact':exact,'seconds':time.monotonic()-began})+'\n')
    if tree_sha256(model.state_dict())!=before:raise ValueError('Inference changed model weights')
    if any(digest(r['file_path'])!=r['image_sha256'] for r in rows):raise ValueError('Image changed during evaluation')
    (a.out/'report.json').write_text(json.dumps({'records':len(rows),'exact':exact,'parsed':parsed,'selected_resolution':scales,
        'seconds':time.monotonic()-began,'optimizer_updates':0,'model_state_unchanged':True},indent=2)+'\n')
    (a.out/'status.json').write_text(json.dumps({'state':'complete','completed':len(rows)})+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact','pairs','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--sizes',nargs='+',type=int,default=[768,1024,512])
    p.add_argument('--primary-cache',type=Path);main(p.parse_args())

"""Controlled full recognizer adaptation with explicit diagnostics and provenance.

Native graph-head supervision is chemical-only in this first gate. Image-only
raw sequence and predicted graph are both saved; complete graph/stereo serving
acceptance is a separate, still-required gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .data import digest, prepare_sample
from .model import WarmStartRecognizer, tensor_digest
from .checkpoint import atomic_save_checkpoint, load_checkpoint_strict, audit_parameter_update
from .protocol import assert_disjoint_splits, canonical_smiles


def state_snapshot(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def gradient_summary(model):
    groups={name:{'tensors':0,'missing':0,'nonzero':0,'squared_norm':0.} for name in ['encoder','decoder','graph_head']}
    for name,parameter in model.named_parameters():
        group='encoder' if name.startswith('encoder.') else 'graph_head' if name.startswith('decoder.decoder.edges.') else 'decoder'
        value=groups[group];value['tensors']+=1
        if parameter.grad is None:value['missing']+=1;continue
        grad=parameter.grad.detach().float()
        if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite gradient '+name)
        square=float(grad.square().sum());value['squared_norm']+=square
        value['nonzero']+=int(square>0)
    for name,value in groups.items():
        value['norm']=math.sqrt(value.pop('squared_norm'))
        if value['norm']==0:raise RuntimeError('No effective gradient in '+name)
    return groups


def evaluate(model,records,samples,device,precision,max_tokens):
    from rdkit import Chem
    results=[]
    for record in records:
        started=time.monotonic()
        # Gold stays outside predict_image; no target counts, coords or refs.
        image=samples[record['record_key']]['image'].unsqueeze(0).to(device)
        error=None;prediction={};text='';terminated=False
        try:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
                prediction=model.predict_image(image,max_tokens=max_tokens,max_atoms=512)
            decoded=prediction['chartok_coords']
            tokens=decoded.get('_token_sequence',[])
            terminated=2 in tokens and (0 not in tokens[:tokens.index(2)])
            text=decoded.get('smiles','')
            exact=terminated and canonical_smiles(text)==canonical_smiles(record['SMILES'])
            connectivity=False
            mol=Chem.MolFromSmiles(text) if terminated else None
            gold=Chem.MolFromSmiles(record['SMILES'])
            if mol:
                Chem.RemoveStereochemistry(mol);Chem.RemoveStereochemistry(gold)
                connectivity=Chem.MolToSmiles(mol)==Chem.MolToSmiles(gold)
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)[:500];exact=False;connectivity=False
        results.append({'record_key':record['record_key'],'split':record['split'],
            'category':record['category'],'peptide_stratum':record.get('peptide_stratum'),
            'gold':record['SMILES'],'raw_sequence_smiles':text,'terminated':terminated,
            'graph_decode_issue':prediction.get('decode_quality_issue'),'raw_sequence_exact':bool(exact),'raw_sequence_connectivity_exact':bool(connectivity),
            'raw_prediction':prediction,'error':error,'seconds':time.monotonic()-started})
    summary={}
    for split in ['train','validation']:
        selected=[r for r in results if r['split']==split]
        summary[split]={'n':len(selected),'raw_sequence_exact':sum(r['raw_sequence_exact'] for r in selected),
            'terminated':sum(r['terminated'] for r in selected)}
    return {'scope':'Image-only raw sequence diagnostic; not complete raw-graph assembly or service acceptance.',
            'summary':summary,'rows':results}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(args):
    if args.updates < 1 or args.eval_every < 1 or args.warmup < 1 or args.cpu_threads < 1:
        raise ValueError('Training budgets must be positive')
    if any(not math.isfinite(rate) or rate <= 0 for rate in [args.encoder_lr,args.decoder_lr,args.head_lr]):
        raise ValueError('Every trained module needs a positive finite learning rate')
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in selected environment')
        torch.cuda.set_device(device)
    records=json.loads(args.manifest.read_text())['records']
    split_report=assert_disjoint_splits(records)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config={key:value for key,value in vars(args).items() if key not in ['out','resume']}
    config={key:str(value) if isinstance(value,Path) else value for key,value in config.items()}
    root=Path(__file__).resolve().parents[2]
    code={str(p.resolve().relative_to(root)):digest(p) for p in Path(__file__).parent.glob('*.py')}
    code.update({str(p):digest(p) for p in Path('utils/MolNexTR').rglob('*.py')})
    shared=Path('training/molsight_multitask/edges.py');code[str(shared)]=digest(shared)
    provenance={'manifest_sha256':digest(args.manifest),'base_checkpoint_sha256':digest(args.checkpoint),
        'source_sha256':code,'torch':str(torch.__version__),'cuda':torch.version.cuda}
    write_json(out/'plan.json',{'state':'preparing','configuration':config,'provenance':provenance,
        'splits':split_report,'required_gates':['actual-gradient','strict-reload','small-set-free-decode',
        'independent-three-domain-validation','complete-raw-graph-stereo','actual-service-image-only'],
        'interpretation':'Passing preflight or token learning alone does not achieve product recognition.'})
    model=WarmStartRecognizer(args.checkpoint,input_size=args.input_size).to(device)
    import rdkit, albumentations
    vocabulary={'stoi':model.tokenizer.stoi,'offset':model.tokenizer.offset,
        'maxx':model.tokenizer.maxx,'maxy':model.tokenizer.maxy,'sep_id':model.tokenizer.sep_id}
    provenance.update(vocabulary=vocabulary,vocabulary_sha256=digest(model.args.vocab_file),
        rdkit=str(rdkit.__version__),albumentations=str(albumentations.__version__))
    source_directory=out/'source_snapshot'
    source_directory.mkdir()
    for path,expected in code.items():
        if digest(path)!=expected: raise RuntimeError('Source changed during preparation: '+path)
        destination=source_directory/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if digest(destination)!=expected: raise RuntimeError('Source snapshot differs')
    shutil.copy2(model.args.vocab_file,source_directory/'vocabulary.json')
    write_json(out/'plan.json',{'state':'prepared','configuration':config,'provenance':provenance,
        'splits':split_report,'required_gates':['actual-gradient','strict-reload','small-set-free-decode',
        'independent-three-domain-validation','complete-raw-graph-stereo','actual-service-image-only'],
        'interpretation':'Passing preflight or token learning alone does not achieve product recognition.'})
    samples={row['record_key']:prepare_sample(row,model.tokenizer,args.input_size,args.max_tokens) for row in records}
    write_json(out/'prepared.json',{'vocabulary_size':len(model.tokenizer),'migrations':model.migrations,
        'samples':[{'record_key':row['record_key'],'split':row['split'],
            'tokens':int(samples[row['record_key']]['refs']['chartok_coords'][0].shape[-1]),
            'atoms':int(samples[row['record_key']]['refs']['edges'].shape[-1])} for row in records]})
    groups=[]
    for name,lr in [('encoder',args.encoder_lr),('decoder',args.decoder_lr),('graph_head',args.head_lr)]:
        parameters=[p for key,p in model.named_parameters() if (
            ('encoder' if key.startswith('encoder.') else 'graph_head' if key.startswith('decoder.decoder.edges.') else 'decoder')==name)]
        groups.append({'params':parameters,'lr':lr,'name':name})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/args.warmup))
    start=0
    if args.resume:
        restored=load_checkpoint_strict(args.resume,model,expected_config=config,expected_provenance=provenance,
            optimizer=optimizer,scheduler=scheduler,restore_rng=True)
        start=restored['step']
    train=[row for row in records if row['split']=='train']
    if not train:raise ValueError('No training samples')
    initial=state_snapshot(model)
    trained_names={name for name,p in model.named_parameters() if p.requires_grad}
    if not args.preflight:
        baseline=evaluate(model,records,samples,device,args.precision,args.max_tokens)
        write_json(out/'baseline.json',baseline)
        print(json.dumps({'baseline':baseline['summary']}),flush=True)
    started=time.monotonic()
    with (out/'updates.jsonl').open('x') as stream:
        for update in range(start+1,args.updates+1):
            # Full deterministic cycling makes every gate example receive supervision.
            row=train[args.preflight_record_index if args.preflight else (update-1)%len(train)];sample=samples[row['record_key']]
            model.train();optimizer.zero_grad(set_to_none=True)
            image=sample['image'].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                losses=model.losses(image,sample['refs'])
            if not all(torch.isfinite(value).all() for value in losses.values()):raise RuntimeError('Nonfinite objective')
            losses['loss'].backward()
            gradients=gradient_summary(model)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            snapshot={'update':update,'record_key':row['record_key'],
                **{key:float(value.detach()) for key,value in losses.items()},
                'gradients':gradients,'unclipped_gradient_norm':float(norm),
                'learning_rates':{group['name']:group['lr'] for group in optimizer.param_groups},
                'seconds':time.monotonic()-started,
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
            stream.write(json.dumps(snapshot,allow_nan=False)+'\n');stream.flush()
            if update==start+1 or update%10==0:print(json.dumps(snapshot),flush=True)
            if args.preflight or update%args.eval_every==0 or update==args.updates:
                checkpoint=out/f'update_{update:06d}.pth'
                saved=atomic_save_checkpoint(checkpoint,model,optimizer,step=update,config=config,
                    provenance=provenance,scheduler=scheduler,sampler_state={'next_index':update%len(train)})
                current=state_snapshot(model)
                audit=audit_parameter_update(initial,current,trainable_names=trained_names)
                changed_by_group={name:[] for name in ['encoder','decoder','graph_head']}
                for name in audit['changed']:
                    group='encoder' if name.startswith('encoder.') else 'graph_head' if name.startswith('decoder.decoder.edges.') else 'decoder'
                    if name in trained_names: changed_by_group[group].append(name)
                if any(not names for names in changed_by_group.values()):
                    raise RuntimeError('A trained module did not update parameters')
                audit['changed_by_group']={group:len(names) for group,names in changed_by_group.items()}
                # Verify full parameter bytes after strict reload of the saved file.
                load_checkpoint_strict(checkpoint,model,expected_config=config,expected_provenance=provenance,
                    optimizer=optimizer,scheduler=scheduler,restore_rng=False,expected_step=update)
                for key,value in model.state_dict().items():
                    if not torch.equal(value.detach().cpu(),current[key]):raise RuntimeError('Reload changed tensor '+key)
                write_json(out/f'parameter_audit_{update:06d}.json',{'checkpoint':saved,'audit':audit,
                    'strict_reload_equal':True,'gradients':gradients})
                del current
                if args.preflight:
                    write_json(out/'state.json',{'state':'preflight_passed','actual_updates':update,
                        'not_completed':['small-set-memorization','raw-graph-stereo','independent-generalization','service']})
                    return
                evaluation=evaluate(model,records,samples,device,args.precision,args.max_tokens)
                write_json(out/f'evaluation_{update:06d}.json',evaluation)
                print(json.dumps({'update':update,'evaluation':evaluation['summary']}),flush=True)
    for path,expected in code.items():
        if digest(path)!=expected:raise RuntimeError('Training source changed during run: '+path)
    write_json(out/'state.json',{'state':'training_finished_requires_acceptance','actual_updates':args.updates,
        'not_completed':['raw-graph-stereo','independent-generalization','actual-service']})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path('models/molnextr_best.pth'))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--input-size',type=int,default=384)
    parser.add_argument('--updates',type=int,default=1000);parser.add_argument('--eval-every',type=int,default=250)
    parser.add_argument('--warmup',type=int,default=50);parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--encoder-lr',type=float,default=1e-5);parser.add_argument('--decoder-lr',type=float,default=5e-5)
    parser.add_argument('--head-lr',type=float,default=2e-4);parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--max-tokens',type=int,default=4096);parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--preflight-record-index',type=int,default=0)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',type=Path)
    main(parser.parse_args())

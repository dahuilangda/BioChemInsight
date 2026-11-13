"""Single-GPU MolSight spatial adaptation with broad two-domain AR replay.

Only the supervised loss receives gold. Evaluation invokes predict_image(image)
with fixed thresholds. No oracle atom count or coordinates enter decoding.
"""
import argparse,copy,hashlib,json,math,os,random,shutil,time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from training.ocsr_reboot.checkpoint import atomic_save_checkpoint,load_checkpoint_strict,tree_sha256,audit_parameter_update
from training.ocsr_reboot.protocol import canonical_smiles
from .backbone import MolsightSpatial
from .data import GraphDataset,digest
from .loss import LossConfig,dn_targets,detection_loss
from .model import SpatialDetectorConfig


def write(path,data):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def move(sample,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in sample.items()}


def parameter_group(name):
    if name.startswith('detector.'):return 'detector'
    if name.startswith('backbone.image_encoder.') or name=='backbone.image_pos_embed':return 'encoder'
    if name.startswith('backbone.decoder.'):return 'decoder'
    raise ValueError('Unclassified trainable parameter: '+name)


def gradient_audit(model):
    groups={k:{'squared_norm':0.,'tensors':0,'missing':0,'nonzero':0} for k in ['encoder','decoder','detector']}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        group=parameter_group(name)
        row=groups[group];row['tensors']+=1
        if p.grad is None:row['missing']+=1;continue
        if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient '+name)
        square=float(p.grad.float().square().sum());row['squared_norm']+=square;row['nonzero']+=int(square>0)
    for key,row in groups.items():
        row['norm']=math.sqrt(row.pop('squared_norm'))
        if row['tensors'] and not row['norm']>0:raise RuntimeError('No effective gradient in '+key)
    return groups


@torch.no_grad()
def evaluate(model,dataset,device,threshold,include_ar=True):
    from molsight.inference import predict_images
    rows=[];model.eval()
    for i,record in enumerate(dataset.records):
        sample=move(dataset.get(i,epoch=0),device);image=sample['image'].unsqueeze(0)
        graph=model.predict_image(image,objectness_threshold=threshold)
        graph_exact=False
        if graph['parse_success']:
            graph_exact=canonical_smiles(graph['smiles'])==canonical_smiles(sample['smiles'])
        row={'record_key':record['record_key'],'domain':record['domain'],'gold':sample['smiles'],
            'spatial_parse_success':bool(graph['parse_success']),'spatial_exact':graph_exact,'spatial':graph}
        if include_ar:
            ar=predict_images(model.backbone,image)
            try:ar_exact=bool(ar['terminated'][0]) and canonical_smiles(ar['smiles'][0])==canonical_smiles(sample['smiles'])
            except ValueError:ar_exact=False
            row.update(ar_smiles=ar['smiles'][0],ar_terminated=bool(ar['terminated'][0]),ar_exact=ar_exact)
        rows.append(row)
    summary={}
    for domain in sorted({r['domain'] for r in rows}):
        values=[r for r in rows if r['domain']==domain]
        fields=['spatial_parse_success','spatial_exact']+(['ar_exact'] if include_ar else [])
        summary[domain]={'count':len(values),**{key:sum(bool(r[key]) for r in values) for key in fields}}
    return {'summary':summary,'rows':rows,'ar_evaluated':include_ar,
        'scope':'Unaugmented image-only development validation. Frozen backbone preservation is verified by byte comparison, not repeated AR inference.'}


def replay_loss(student,reference,batch):
    from molsight.policy import policy_mode
    from molsight.retention import distillation_sum
    image,label=batch['image'].unsqueeze(0),batch['label'].unsqueeze(0)
    with policy_mode(student,adapters=None):
        actual=student(image,label)['logits'][:,:-1].float()
        with torch.no_grad():expected=reference(image,label)['logits'][:,:-1].float()
    target=label[:,1:]
    ce=F.cross_entropy(actual.reshape(-1,actual.shape[-1]),target.reshape(-1),ignore_index=0)
    kl=distillation_sum(actual,expected,target)/target.ne(0).sum().clamp_min(1)
    return .25*ce+kl,ce,kl


def main(args):
    from molsight.prepared import PreparedDataset
    from molsight_multitask.replay import DomainReplaySampler
    if min(args.updates,args.eval_every,args.input_size,args.save_every,args.warmup,args.queries)<1:raise ValueError('Invalid training budget')
    if not 0<args.objectness_threshold<1:raise ValueError('Objectness threshold must be strictly between zero and one')
    if any(not math.isfinite(rate) or rate<=0 for rate in [args.encoder_lr,args.decoder_lr,args.detector_lr]):
        raise ValueError('Training learning rates must be finite and positive; use --preflight for zero-LR checks')
    if not torch.cuda.is_available():raise RuntimeError('Real spatial adaptation requires CUDA')
    torch.set_num_threads(2);device=torch.device(args.device);torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['out','resume','stop_after']}
    root=Path(__file__).resolve().parents[2]
    sources=[*Path(__file__).parent.glob('*.py'),*Path('training/ocsr_reboot').glob('*.py'),Path('training/molsight_multitask/focal.py'),Path('training/molsight_multitask/replay.py'),*Path('training/molsight_runtime/molsight').glob('*.py')]
    provenance={'manifest_sha256':digest(args.manifest),'checkpoint_sha256':digest(args.checkpoint),'vocabulary_sha256':digest(args.vocabulary),
        'admitted_manifest_sha256':digest(args.admitted_manifest),'source_sha256':{str(p.resolve()):digest(p) for p in sources},'torch':str(torch.__version__)}
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=args.resume is not None)
    detector_config=SpatialDetectorConfig(queries=args.queries)
    model=MolsightSpatial(args.checkpoint,args.vocabulary,detector_config).to(device)
    if args.spatial_warmstart:
        original_backbone=tree_sha256(model.backbone.state_dict())
        previous=torch.load(args.spatial_warmstart,map_location='cpu',mmap=True,weights_only=False)
        from .artifact import runtime_sources
        for relative,expected in runtime_sources(previous['provenance']).items():
            if digest(root/relative)!=expected:raise ValueError('Warmstart inference implementation changed: '+relative)
        load_checkpoint_strict(args.spatial_warmstart,model,expected_config=previous['config'],expected_provenance=previous['provenance'],restore_rng=False)
        if tree_sha256(model.backbone.state_dict())!=original_backbone:
            raise ValueError('Spatial warmstart must preserve the declared MolSight base checkpoint')
        prior_updates=previous['step']+previous['provenance'].get('spatial_warmstart',{}).get('total_prior_updates',0)
        provenance['spatial_warmstart']={'sha256':digest(args.spatial_warmstart),'step':previous['step'],
            'total_prior_updates':prior_updates,'optimizer_reset':True}
        del previous
    if args.backbone_mode=='frozen':model.backbone.requires_grad_(False)
    reference=copy.deepcopy(model.backbone).requires_grad_(False).eval() if args.backbone_mode=='joint' else None
    provenance['backbone_configuration']=model.configuration
    provenance['detector_configuration']=asdict(detector_config)
    provenance['relative_source_sha256']={str(p.resolve().relative_to(root)):digest(p) for p in sources}
    groups=defaultdict(list)
    for name,p in model.named_parameters():
        if p.requires_grad:groups[parameter_group(name)].append(p)
    rates={'encoder':args.encoder_lr,'decoder':args.decoder_lr,'detector':args.detector_lr}
    optimizer=torch.optim.AdamW([{'params':p,'lr':0. if args.preflight else rates[name],'name':name} for name,p in sorted(groups.items())],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/args.warmup))
    train=GraphDataset(args.manifest,'train',args.input_size,augment=True,seed=args.seed)
    validation=GraphDataset(args.manifest,'validation',args.input_size,augment=False,seed=args.seed)
    loss_configuration=LossConfig(objectness_weight=args.objectness_weight,balance_objectness=args.balance_objectness,count_weight=args.count_weight,
        edge_negative_ratio=args.edge_negative_ratio,stereo_negative_ratio=args.stereo_negative_ratio)
    prepared=PreparedDataset(args.admitted_manifest,'train',max_len=2048) if reference is not None else None
    pools={domain:[i for i,r in enumerate(train.records) if r['domain']==domain] for domain in ['small_molecule','linear_peptide','cyclic_peptide']}
    if any(not p for p in pools.values()):raise ValueError('Every training domain must be represented')
    start=0
    if args.resume:
        resume_metadata=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        if 'source_migration' in resume_metadata['provenance']:
            provenance['source_migration']=resume_metadata['provenance']['source_migration']
        del resume_metadata
        restored=load_checkpoint_strict(args.resume,model,expected_config=configuration,expected_provenance=provenance,optimizer=optimizer,scheduler=scheduler,restore_rng=True)
        start=restored['step']
        if restored['sampler_state']!={'next_update':start+1}:raise ValueError('Resume sampling cursor differs from checkpoint step')
        existing_log=out/'updates.jsonl'
        if existing_log.exists():
            logged=[json.loads(line)['update'] for line in existing_log.read_text().splitlines() if line.strip()]
            if not logged or logged[-1]!=start or any(b!=a+1 for a,b in zip(logged,logged[1:])):
                raise ValueError('Resume log does not end at the restored checkpoint; use a new output directory')
    anchors={domain:iter(DomainReplaySampler(prepared,domain,vocabulary_size=len(model.tokenizer),updates=args.updates,
        batch_size=1,seed=args.seed,start_update=start)) for domain in ['complete','peptide']} if reference is not None else {}
    write(out/'plan.json',{'configuration':configuration,'provenance':provenance,'detector':asdict(detector_config),
        'loss':asdict(loss_configuration),'sampling_schedule':['small_molecule','cyclic_peptide','small_molecule','linear_peptide','cyclic_peptide'],
        'cuda_training_determinism':'grid_sample backward may be nondeterministic; no bit-identical trajectory claim',
        'image_only_threshold':args.objectness_threshold,'replay':list(anchors),'preflight_zero_lr':args.preflight,
        'limitations':['Pilot is synthetic development data; successful updates or memorization are not product acceptance.']})
    snapshot=out/'source_snapshot'
    if not snapshot.exists():
        for path in sources:
            dest=snapshot/path.resolve().relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    baseline=None
    if not args.preflight:
        if args.resume:baseline=json.loads((args.resume.parent/'baseline.json').read_text())
        else:baseline=evaluate(model,validation,device,args.objectness_threshold,include_ar=reference is not None)
        write(out/'baseline.json',baseline)
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_digest=tree_sha256(initial_state) if args.preflight else None
    exposure=Counter();began=time.monotonic();stop=args.stop_after or (start+3 if args.preflight else args.updates)
    prior_updates=provenance.get('spatial_warmstart',{}).get('total_prior_updates',0)
    if not start<stop<=args.updates:raise ValueError('Invalid stop boundary')
    schedule=['small_molecule','cyclic_peptide','small_molecule','linear_peptide','cyclic_peptide']
    log_mode='a' if args.resume else 'x'
    with (out/'updates.jsonl').open(log_mode) as stream:
        for update in range(start+1,stop+1):
            data_update=prior_updates+update
            model.train();model.backbone.image_encoder.eval();optimizer.zero_grad(set_to_none=True)
            domain=(['small_molecule','linear_peptide','cyclic_peptide'][(update-1)%3] if args.preflight else schedule[(update-1)%len(schedule)])
            index=(max(pools[domain],key=lambda i:len(train.records[i]['coords_pixels'])) if args.preflight and domain=='cyclic_peptide' else int(np.random.default_rng(np.random.SeedSequence([args.seed,data_update])).choice(pools[domain])))
            sample=move(train.get(index,epoch=data_update),device);denoising=dn_targets(sample)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(sample['image'].unsqueeze(0),dn_points=denoising.points.unsqueeze(0))
                losses=detection_loss(prediction,sample,denoising=denoising,config=loss_configuration)
            losses['loss'].backward()
            replay={}
            for category,iterator in anchors.items():
                batch=move(prepared[next(iterator)[0]],device)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,ce,kl=replay_loss(model.backbone,reference,batch)
                (.5*loss).backward();replay[category]={'ce':float(ce.detach()),'kl':float(kl.detach())}
            gradients=gradient_audit(model)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            applied_rates={g['name']:g['lr'] for g in optimizer.param_groups}
            optimizer.step();scheduler.step();exposure[domain]+=1
            record={'update':update,'data_update':data_update,'domain':domain,'record_key':sample['record_key'],'atoms':len(sample['coords']),
                'loss':float(losses['loss'].detach()),'terms':{k:float(v.detach()) for k,v in losses['terms'].items()},
                'replay':replay,'gradients':gradients,'unclipped_gradient_norm':float(norm),'seconds':time.monotonic()-began,
                'learning_rates_used':applied_rates,'next_learning_rates':{g['name']:g['lr'] for g in optimizer.param_groups},'exposure_since_start':dict(exposure),
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({k:record[k] for k in ['update','domain','atoms','loss','unclipped_gradient_norm','seconds']}),flush=True)
            if args.preflight and tree_sha256(model.state_dict())!=initial_digest:raise RuntimeError('Zero-LR preflight changed parameters/buffers')
            if update%args.save_every==0 or update==stop:
                path=out/f'update_{update:06d}.pth'
                saved=atomic_save_checkpoint(path,model,optimizer,step=update,config=configuration,provenance=provenance,scheduler=scheduler,
                    sampler_state={'next_update':update+1})
                load_checkpoint_strict(path,model,expected_config=configuration,expected_provenance=provenance,optimizer=optimizer,scheduler=scheduler,restore_rng=False,expected_step=update)
                after={k:v.detach().cpu() for k,v in model.state_dict().items()}
                if args.preflight:
                    if tree_sha256(after)!=initial_digest:
                        raise RuntimeError('Strict reload changed zero-LR model state')
                    delta={'changed':[], 'all_finite':True, 'frozen_unchanged':True,
                           'before_sha256':initial_digest, 'after_sha256':tree_sha256(after)}
                else:
                    delta=audit_parameter_update(initial_state,after,trainable_names={name for name,p in model.named_parameters() if p.requires_grad})
                if not args.preflight:
                    prefixes=['detector.'] if args.backbone_mode=='frozen' else ['backbone.image_encoder.','backbone.decoder.','detector.']
                    for prefix in prefixes:
                        if not any(name.startswith(prefix) for name in delta['changed']):raise RuntimeError('No parameter changes in '+prefix)
                write(out/f'parameter_audit_{update:06d}.json',{'saved':saved,'strict_reload_equal':True,'parameter_changes':delta,'gradients':gradients,'zero_lr_unchanged':args.preflight})
            if not args.preflight and (update%args.eval_every==0 or update==stop):
                evaluation=evaluate(model,validation,device,args.objectness_threshold,include_ar=reference is not None);write(out/f'evaluation_{update:06d}.json',evaluation)
                if reference is not None and any(evaluation['summary'][d]['ar_exact']<baseline['summary'][d]['ar_exact'] for d in baseline['summary']):
                    write(out/'state.json',{'state':'stopped_for_development_AR_regression','update':update});return
    state='preflight_passed' if args.preflight else 'training_finished_requires_capability_acceptance' if stop==args.updates else 'checkpoint_boundary_requires_review'
    write(out/'state.json',{'state':state,'actual_updates':stop,'planned_updates':args.updates,'prior_updates':prior_updates})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','checkpoint','vocabulary','admitted-manifest','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--updates',type=int,default=500);parser.add_argument('--stop-after',type=int)
    parser.add_argument('--spatial-warmstart',type=Path)
    parser.add_argument('--eval-every',type=int,default=250);parser.add_argument('--save-every',type=int,default=250)
    parser.add_argument('--warmup',type=int,default=50);parser.add_argument('--queries',type=int,default=512);parser.add_argument('--input-size',type=int,default=768)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',default='cuda:0');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--backbone-mode',choices=['frozen','joint'],default='frozen')
    parser.add_argument('--encoder-lr',type=float,default=1e-7);parser.add_argument('--decoder-lr',type=float,default=1e-6);parser.add_argument('--detector-lr',type=float,default=1e-4)
    parser.add_argument('--objectness-threshold',type=float,default=.5)
    parser.add_argument('--balance-objectness',action='store_true');parser.add_argument('--objectness-weight',type=float,default=1.)
    parser.add_argument('--count-weight',type=float,default=0.)
    parser.add_argument('--edge-negative-ratio',type=float,default=0.)
    parser.add_argument('--stereo-negative-ratio',type=float,default=0.)
    main(parser.parse_args())

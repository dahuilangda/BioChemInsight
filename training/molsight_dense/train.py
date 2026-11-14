"""Train only dense visual heads on the actual pretrained MolSight backbone."""
import argparse,json,random,shutil,time
import math
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from training.ocsr_reboot.checkpoint import atomic_save_checkpoint,load_checkpoint_strict,tree_sha256,audit_parameter_update
from training.ocsr_reboot.protocol import canonical_smiles
from training.molsight_spatial.data import digest
from training.molsight_spatial.train import write,move
from .model import DenseMolsight,DenseConfig
from .objectives import DenseDataset,objective


@torch.no_grad()
def evaluate(model,dataset,device):
    rows=[];model.eval()
    for i,record in enumerate(dataset.records):
        sample=dataset.get(i,epoch=0)
        result=model.predict_image(sample['image'].unsqueeze(0).to(device))
        exact=result['parse_success'] and canonical_smiles(result['smiles'])==canonical_smiles(sample['smiles'])
        matches=0;count=result.get('predicted_count') or 0
        if count and len(result.get('points',[]))==count:
            points=torch.tensor(result['points']);elements=torch.tensor(result['elements'])
            distances=torch.cdist(points,sample['coords'])
            a,b=torch.where(torch.triu(sample['edges']>0,diagonal=1))
            radius=float(torch.linalg.vector_norm(sample['coords'][a]-sample['coords'][b],dim=-1).median())*.3 if len(a) else .01
            valid=(distances<=radius)&elements[:,None].eq(sample['elements'][None,:])
            a,b=linear_sum_assignment((distances+~valid*1e6).numpy())
            matches=sum(int(valid[x,y]) for x,y in zip(a,b))
        rows.append({'record_key':record['record_key'],'domain':record['domain'],'exact':bool(exact),
            'parse_success':bool(result['parse_success']),'atom_true':matches,'atom_predicted':count,'atom_gold':len(sample['coords']),
            'count_exact':count==len(sample['coords']),'prediction':result})
    summary={}
    for domain in sorted({r['domain'] for r in rows}):
        values=[r for r in rows if r['domain']==domain]
        tp=sum(r['atom_true'] for r in values);pred=sum(r['atom_predicted'] for r in values);gold=sum(r['atom_gold'] for r in values)
        summary[domain]={'records':len(values),'exact':sum(r['exact'] for r in values),'parse':sum(r['parse_success'] for r in values),
            'count_exact':sum(r['count_exact'] for r in values),'atom_precision':tp/max(1,pred),'atom_recall':tp/gold,'atom_f1':2*tp/max(1,pred+gold)}
    return {'summary':summary,'rows':rows,'scope':'All model predictions are image-only. Coordinates and counts enter only the post-inference scorer.'}


def main(args):
    if min(args.updates,args.save_every,args.eval_every,args.input_size)<1 or args.input_size%32 or args.localization_warmup<0:raise ValueError('Invalid training budget')
    if not math.isfinite(args.lr) or args.lr<=0:raise ValueError('Learning rate must be finite and positive')
    if any(not math.isfinite(v) or v<0 for v in [args.chemistry_weight,args.hard_weight,args.center_recall_weight]):raise ValueError('Invalid structural loss weight')
    if args.verify_proposals and not args.visible:raise ValueError('Proposal verifier requires the complete visible graph architecture')
    if not math.isfinite(args.size_prior_power) or not 0<=args.size_prior_power<=1:raise ValueError('Invalid size prior')
    train_sizes=sorted(set(args.train_sizes or [args.input_size]))
    if any(size<128 or size%32 for size in train_sizes):raise ValueError('Training sizes must be positive multiples of 32')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the full pretrained model')
    torch.set_num_threads(2);device=torch.device(args.device);torch.cuda.set_device(device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['out','resume','stop_after']}
    config=DenseConfig(peak_threshold=args.peak_threshold)
    if args.visible:
        from training.molsight_visible.model import VisibleMolsight
        from training.molsight_visible.data import VisibleDataset
        model_type,dataset_type=VisibleMolsight,VisibleDataset
    else:model_type,dataset_type=DenseMolsight,DenseDataset
    if args.verify_proposals:
        from training.molsight_proposals.model import ProposalMolsight
        model_type=ProposalMolsight
    model=model_type(args.checkpoint,args.vocabulary,config).to(device)
    train=dataset_type(args.manifest,'train',args.input_size,augment=True,seed=args.seed,preprocessing=args.preprocessing)
    validation=dataset_type(args.manifest,'validation',args.input_size,augment=False,seed=args.seed,preprocessing=args.preprocessing)
    root=Path(__file__).resolve().parents[2]
    sources=[*Path(__file__).parent.glob('*.py'),*Path('training/molsight_spatial').glob('*.py'),
        *Path('training/ocsr_reboot').glob('*.py'),Path('training/molsight_multitask/focal.py'),Path('training/molsight_multitask/replay.py'),
        *Path('training/molsight_runtime/molsight').glob('*.py'),Path('training/prepare_molsight_dense_styles.py')]
    if args.visible:sources += [*Path('training/molsight_visible').glob('*.py'),Path('training/prepare_molsight_visible.py')]
    sources += [Path('training/prepare_visible_hydrogen_views.py'),Path('training/prepare_molgrapher_complete.py')]
    if args.verify_proposals:sources += list(Path('training/molsight_proposals').glob('*.py'))
    provenance={'manifest_sha256':digest(args.manifest),'base_sha256':digest(args.checkpoint),'vocabulary_sha256':digest(args.vocabulary),
        'backbone_configuration':model.configuration,'dense_configuration':asdict(config),
        'relative_source_sha256':{str(p.resolve().relative_to(root)):digest(p) for p in sources}}
    if args.visible:
        from training.molsight_visible.registry import REGISTRY_SHA256
        provenance['visible_registry_sha256']=REGISTRY_SHA256
    if args.verify_proposals:provenance['proposal_validity']={'threshold':.5,'initial_policy':'All original proposals retained','negative_training':'wrong image peaks and non-center ink','new_head_lr_multiplier':10}
    if args.warmstart:
        previous=torch.load(args.warmstart,map_location='cpu',mmap=True,weights_only=False)
        from .artifact import inference_sources
        for relative,expected in inference_sources(previous['provenance']).items():
            if digest(root/relative)!=expected:raise ValueError('Warmstart inference source changed: '+relative)
        base_before=tree_sha256(model.backbone.state_dict())
        if (args.visible and previous['model_class']=='training.molsight_dense.model.DenseMolsight') or (args.verify_proposals and previous['model_class']=='training.molsight_visible.model.VisibleMolsight'):
            actual=tree_sha256({k:v for k,v in previous.items() if k!='payload_sha256'})
            if actual!=previous['payload_sha256']:raise ValueError('Corrupt migration source')
            missing,unexpected=model.load_state_dict(previous['model'],strict=False)
            expected=set()
            if previous['model_class']=='training.molsight_dense.model.DenseMolsight':
                expected|={'head.states.aliases.weight','head.states.aliases.bias'}|{k for k in model.state_dict() if k.startswith('head.context.')}
            if args.verify_proposals:expected|={'head.validity.weight','head.validity.bias'}
            if set(missing)!=expected or unexpected:raise ValueError('Unexpected visible-head migration keys')
            restored={k:model.state_dict()[k] for k in previous['model']}
            if tree_sha256(restored)!=tree_sha256(previous['model']):raise ValueError('Migration changed existing weights')
            provenance['head_migration']={'initialized_only':sorted(expected),'all_existing_state_byte_equal':True,'source_class':previous['model_class']}
        else:
            load_checkpoint_strict(args.warmstart,model,expected_config=previous['config'],expected_provenance=previous['provenance'],restore_rng=False)
        if base_before!=tree_sha256(model.backbone.state_dict()):raise ValueError('Warmstart changed declared MolSight base')
        provenance['warmstart']={'sha256':digest(args.warmstart),'prior_updates':previous['step']+previous['provenance'].get('warmstart',{}).get('prior_updates',0),
            'optimizer_reset':True,'loaded_model_sha256':tree_sha256(model.state_dict())}
        del previous
    rate=0. if args.preflight else args.lr
    parameters=list(model.head.parameters())
    if args.verify_proposals:
        new=list(model.head.validity.parameters());new_ids={id(p) for p in new}
        parameters=[{'params':[p for p in parameters if id(p) not in new_ids],'lr':rate},
            {'params':new,'lr':rate*10}]
    optimizer=torch.optim.AdamW(parameters,lr=rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/100))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=args.resume is not None)
    start=0
    if args.resume:
        restored=load_checkpoint_strict(args.resume,model,expected_config=configuration,expected_provenance=provenance,
            optimizer=optimizer,scheduler=scheduler,restore_rng=True)
        start=restored['step']
        if restored['sampler_state']!={'next_update':start+1}:raise ValueError('Resume cursor mismatch')
        if (out/'updates.jsonl').exists():
            logged=[json.loads(x)['update'] for x in (out/'updates.jsonl').read_text().splitlines()]
            if not logged or logged[-1]!=start:raise ValueError('Resume log has advanced past checkpoint')
    snapshot=out/'source_snapshot'
    if not snapshot.exists():
        for source in sources:
            target=snapshot/source.resolve().relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(out/'plan.json',{'configuration':configuration,'provenance':provenance,'new_architecture_heads_initialized':args.warmstart is None,
        'backbone':'Original pretrained MolSight, completely frozen including buffers.',
        'supervision':'Independent pixel centers, per-atom total H, joint chemical+apex categorical bond.',
        'reader_training':'GT points with pixel noise only during training; all evaluations use detected peaks.',
        'sampling':'domain schedule ' + args.domain_schedule + '; uniform available atom-size bins within domain.',
        'sampling_unit':args.sampling_unit,
        'gpu_determinism':'grid_sample backward may be nondeterministic; strict restored states do not imply identical future GPU trajectories.'})
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    frozen_hash=tree_sha256(model.backbone.state_dict())
    pools=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(train.records):
        n=r.get('chemical_atom_count',len(r['coords_pixels']));size=sum(n>limit for limit in [40,80,160,256])
        pools[r['domain']][size].append(i)
    if args.sampling_unit=='identity':
        from .sampling import identity_pools,choose_view
        identities=identity_pools(train.records,pools)
    from .sampling import choose_size
    domain_weights = {}
    for part in args.domain_schedule.split(','):
        name, _, count = part.strip().partition(':')
        name = name.strip()
        if not name:
            raise ValueError('Empty domain in --domain-schedule')
        domain_weights[name] = int(count or 1)
    unknown = set(domain_weights) - set(pools)
    if unknown:
        raise ValueError(f'--domain-schedule names domains absent from the manifest: {sorted(unknown)}')
    schedule = [name for name, count in domain_weights.items() for _ in range(max(1, count))]
    stop=args.stop_after or (start+3 if args.preflight else args.updates)
    if not start<stop<=args.updates:raise ValueError('Invalid training boundary')
    began=time.monotonic()
    with (out/'updates.jsonl').open('a' if args.resume else 'x') as stream:
        for update in range(start+1,stop+1):
            data_update=update+provenance.get('warmstart',{}).get('prior_updates',0)
            domain=(['small_molecule','linear_peptide','cyclic_peptide'][(update-1)%3] if args.preflight else schedule[(update-1)%5])
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,data_update]))
            size=choose_size(identities[domain] if args.sampling_unit=='identity' else pools[domain],rng,args.size_prior_power)
            index=(max((j for values in pools[domain].values() for j in values),key=lambda j:len(train.records[j]['coords_pixels']))
                if args.preflight and domain=='cyclic_peptide' else
                choose_view(identities[domain][size],rng) if args.sampling_unit=='identity' else int(rng.choice(pools[domain][size])))
            if args.preflight and args.visible and domain=='small_molecule':
                candidates=[j for values in pools[domain].values() for j in values if train.records[j].get('visible_graph')]
                if not candidates:raise ValueError('Visible preflight needs an actual abbreviation example')
                index=max(candidates,key=lambda j:train.records[j]['chemical_atom_count']-train.records[j]['visible_atom_count'])
            size_rng=np.random.default_rng(np.random.SeedSequence([args.seed,data_update,8531]))
            selected_size=(train_sizes[(update-1)%len(train_sizes)] if args.preflight else int(size_rng.choice(train_sizes)))
            retry_sizes=[s for s in train_sizes if s>=selected_size]
            retries=[]
            for resolution in retry_sizes:
                train.size=resolution;sample=train.get(index,epoch=data_update)
                cells=(sample['coords']*(resolution//2)).floor().long()
                if cells.unique(dim=0).shape[0]==len(cells):break
                retries.append(resolution)
            else:raise ValueError('Observed units collide at every admitted training resolution')
            sample=move(sample,device)
            model.train();model.backbone.eval();optimizer.zero_grad(set_to_none=True)
            reader=update>args.localization_warmup or args.preflight
            with torch.autocast('cuda',dtype=torch.bfloat16):
                encoded=model.encode(sample['image'].unsqueeze(0))
                loss,terms=objective(model,encoded,sample,reader_enabled=reader,
                    chemistry_weight=args.chemistry_weight,hard_weight=args.hard_weight,center_recall_weight=args.center_recall_weight,
                    noise_seed=args.seed*1000000+data_update)
            loss.backward()
            groups=defaultdict(float);missing=[]
            for name,p in model.head.named_parameters():
                if p.grad is None:
                    if reader or not name.startswith('bonds.'):missing.append(name)
                    continue
                if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient '+name)
                groups[name.split('.')[0]]+=float(p.grad.float().square().sum())
            if missing:raise RuntimeError('Missing active gradients '+str(missing))
            bond_supervised=reader and bool(sample['joint_bonds'].ne(-100).any())
            for name in ['pixel','semantic','fuse','center','offset','node','states']+(['context'] if hasattr(model.head,'context') else [])+(['validity'] if args.verify_proposals else [])+(['bonds'] if bond_supervised else []):
                if not groups[name]>0:raise RuntimeError('No gradient in '+name)
            norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),5.,error_if_nonfinite=True)
            rate=optimizer.param_groups[0]['lr'];used_rates=[g['lr'] for g in optimizer.param_groups];optimizer.step();scheduler.step()
            row={'update':update,'data_update':data_update,'domain':domain,'record_key':sample['record_key'],'atoms':len(sample['coords']),
                'reader_enabled':reader,'chemical_size_bucket':size,'training_input_size':resolution,'resolution_retries':retries,
                'loss':float(loss.detach()),'terms':{k:float(v.detach()) for k,v in terms.items()},
                'gradient_norms':{k:v**.5 for k,v in groups.items()},'unclipped_norm':float(norm),'lr_used':rate,
                'lr_by_group':used_rates,'seconds':time.monotonic()-began}
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if update%50==0 or args.preflight:print(json.dumps(row),flush=True)
            if update%args.save_every==0 or update==stop:
                if tree_sha256(model.backbone.state_dict())!=frozen_hash:raise RuntimeError('Frozen backbone changed')
                path=out/f'update_{update:06d}.pth'
                atomic_save_checkpoint(path,model,optimizer,step=update,config=configuration,provenance=provenance,
                    scheduler=scheduler,sampler_state={'next_update':update+1})
                load_checkpoint_strict(path,model,optimizer=optimizer,scheduler=scheduler,expected_config=configuration,
                    expected_provenance=provenance,restore_rng=False,expected_step=update)
                after={k:v.detach().cpu() for k,v in model.state_dict().items()}
                if args.preflight:
                    if tree_sha256(initial)!=tree_sha256(after):raise RuntimeError('Zero-LR model changed')
                    audit={'zero_lr_unchanged':True}
                else:audit=audit_parameter_update(initial,after,trainable_names={n for n,p in model.named_parameters() if p.requires_grad})
                write(out/f'audit_{update:06d}.json',{'strict_reload':True,'backbone_byte_unchanged':True,'parameters':audit})
            if not args.preflight and (update%args.eval_every==0 or update==stop):
                write(out/f'evaluation_{update:06d}.json',evaluate(model,validation,device))
    write(out/'state.json',{'state':'preflight_passed' if args.preflight else 'finished_requires_exact_acceptance' if stop==args.updates else 'checkpoint_boundary',
        'actual_updates':stop,'planned_updates':args.updates})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','checkpoint','vocabulary','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--stop-after',type=int);parser.add_argument('--warmstart',type=Path)
    parser.add_argument('--updates',type=int,default=5000);parser.add_argument('--eval-every',type=int,default=1000);parser.add_argument('--save-every',type=int,default=1000)
    parser.add_argument('--input-size',type=int,default=768);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--lr',type=float,default=3e-4);parser.add_argument('--peak-threshold',type=float,default=.3)
    parser.add_argument('--localization-warmup',type=int,default=300);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--visible',action='store_true')
    parser.add_argument('--verify-proposals',action='store_true')
    parser.add_argument('--sampling-unit',choices=['view','identity'],default='view',help='Identity then view prevents repeated renderings from biasing chemical exposure')
    parser.add_argument('--domain-schedule',default='small_molecule:2,linear_peptide:1,cyclic_peptide:2',help='Domain sampling weights as name:count pairs, comma separated')
    parser.add_argument('--size-prior-power',type=float,default=0.,help='0: uniform size buckets; 1: natural training frequency; .5: square-root balance')
    parser.add_argument('--train-sizes',type=int,nargs='+',help='Coordinate-synchronized training resolutions, independent of the fixed validation size')
    parser.add_argument('--chemistry-weight',type=float,default=0.)
    parser.add_argument('--hard-weight',type=float,default=0.)
    parser.add_argument('--center-recall-weight',type=float,default=0.)
    parser.add_argument('--preprocessing',choices=['letterbox','molsight_native'],default='letterbox')
    main(parser.parse_args())

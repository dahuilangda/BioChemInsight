"""Explicit MolSight character/chemical-edge adaptation with immutable plans.

This is a separate experiment driver; it never changes the running character
control. --preflight executes two zero-LR updates and checks unchanged weights.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from fit import evaluate, runtime_signature, save_checkpoint, validation_indices
from molsight.checkpoint import load_parameters, model_state, restore_rng
from molsight.cpu import loader_options
from molsight.evaluation import aggregate
from molsight.metrics import TokenMetrics
from molsight.model import MolsightModel
from molsight.prepared import PreparedDataset, collate_characters, file_hash
from molsight.retention import RetentionObjective
from molsight_multitask.replay import DomainReplaySampler
from molsight.sampling import DomainBatchSampler
from molsight.tokenizer import CharTokenizer
from molsight_multitask.data import GraphDataset, GraphStore, collate_graphs
from molsight_multitask.objective import GraphCriterion
from molsight_multitask.update import multitask_update


def validate_configuration(config):
    required = {'manifest', 'checkpoint', 'train_sidecar', 'validation_sidecar', 'out',
                'updates', 'warmup', 'batch_size', 'accumulation', 'lr', 'encoder_lr',
                'head_lr', 'edge_weight', 'edge_warmup', 'weights', 'peptide_group_weights',
                'stratum_weights', 'seed', 'workers', 'eval_every', 'save_every',
                'eval_batch_size', 'max_len', 'max_grad_norm', 'weight_decay',
                'retention_weight', 'retention_supervised_weight', 'retention_temperature',
                'retention_batch_size', 'retention_tolerance', 'retention_domains', 'excluded_domains'}
    if set(config) != required:
        raise ValueError(f'Configuration fields differ: missing={required-set(config)}, extra={set(config)-required}')
    for name in ('updates', 'batch_size', 'accumulation', 'eval_every', 'save_every',
                 'eval_batch_size', 'max_len', 'retention_batch_size'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'Invalid positive integer: {name}')
    for name in ('warmup', 'edge_warmup', 'workers', 'seed'):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f'Invalid nonnegative integer: {name}')
    if config['warmup'] > config['updates'] or config['edge_warmup'] > config['updates']:
        raise ValueError('Warmup exceeds training length')
    for name in ('lr', 'head_lr', 'max_grad_norm', 'retention_temperature'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'Invalid positive scalar: {name}')
    for name in ('encoder_lr', 'edge_weight', 'weight_decay', 'retention_weight',
                 'retention_supervised_weight', 'retention_tolerance'):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f'Invalid nonnegative scalar: {name}')
    excluded = config['excluded_domains']
    if (not isinstance(excluded, list) or len(excluded) != len(set(excluded))
            or set(excluded) & set(config['weights'])):
        raise ValueError('Excluded domains must be explicit, unique and disjoint from weights')
    domains = config['retention_domains']
    if (not domains or set(domains) - {'complete', 'peptide', 'attachment', 'markush'}
            or any(not math.isfinite(value) or value <= 0 for value in domains.values())
            or not math.isclose(sum(domains.values()), 1., abs_tol=1e-9)):
        raise ValueError('Replay domain weights must be positive and sum to one')
    if config['edge_weight'] == 0:
        raise ValueError('Graph experiment requires nonzero chemical edge weight')


def create_student(initial, tokenizer):
    configuration = dict(initial['plan']['configuration'])
    configuration.update(formats=['char', 'edges'], use_checkpoint=True, compile_layers=False)
    model = MolsightModel(SimpleNamespace(**configuration), tokenizer)
    state = model_state(initial)
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Initial checkpoint has nonfinite model tensors')
    destinations = model.state_dict()
    new = sorted(set(destinations) - set(state))
    if not new or any(not name.startswith('decoder.edge_predictor.') for name in new):
        raise ValueError('Initial checkpoint must extend by only the chemical edge head')
    migrated = {**state, **{name: destinations[name] for name in new}}
    load_parameters(model, migrated, source_vocabulary=initial['vocabulary'], target_vocabulary=tokenizer.stoi)
    return model, configuration, new


def parameter_groups(model, config, preflight):
    groups = {'decoder': [], 'encoder': [], 'chemical_edge_head': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = 'encoder' if name.startswith('image_encoder.') else (
            'chemical_edge_head' if name.startswith('decoder.edge_predictor.') else 'decoder')
        groups[kind].append(parameter)
    rates = {'decoder': config['lr'], 'encoder': config['encoder_lr'], 'chemical_edge_head': config['head_lr']}
    return [{'params': values, 'lr': 0. if preflight else rates[name], 'name': name}
            for name, values in groups.items() if values]


def run(args):
    config = json.loads(args.experiment.read_text())
    validate_configuration(config)
    if not torch.cuda.is_available():
        raise RuntimeError('This trainer requires CUDA; CPU objective tests use a separate harness')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rank, local, world = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'] + rank)
    torch.cuda.manual_seed_all(config['seed'] + rank)
    output = Path(config['out']) / ('preflight' if args.preflight else 'run')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=args.resume is not None)
    if world > 1:
        dist.barrier()
    specification = json.loads(Path(config['manifest']).read_text())
    tokenizer = CharTokenizer(specification['vocabulary'])
    initial = torch.load(config['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if initial['vocabulary'] != tokenizer.stoi or file_hash(specification['vocabulary']) != specification['vocabulary_sha256']:
        raise ValueError('Initial checkpoint/vocabulary mismatch')
    student, configuration, initialized = create_student(initial, tokenizer)
    if configuration['max_len'] != config['max_len']:
        raise ValueError('Experiment must retain the initial model context length')
    reference = MolsightModel(SimpleNamespace(**initial['plan']['configuration']), tokenizer)
    load_parameters(reference, model_state(initial), source_vocabulary=tokenizer.stoi, target_vocabulary=tokenizer.stoi)
    reference.requires_grad_(False).eval().to(device)
    student.image_encoder.requires_grad_(config['encoder_lr'] > 0)
    restart = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False) if args.resume else None
    if restart is not None:
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in model_state(restart).values()):
            raise ValueError('Resume checkpoint has nonfinite model tensors')
        load_parameters(student, model_state(restart), source_vocabulary=restart['vocabulary'], target_vocabulary=tokenizer.stoi)
    student.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(student, config, args.preflight), weight_decay=config['weight_decay'])
    def multiplier(step):
        if step < config['warmup']:
            return (step + 1) / max(1, config['warmup'])
        fraction = min(1., (step - config['warmup']) / max(1, config['updates'] - config['warmup']))
        return 0.5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    start = restart['update'] if restart else 0
    stop = args.stop_after or (2 if args.preflight else config['updates'])
    if not start < stop <= config['updates']:
        raise ValueError('Stop boundary must follow the resume update and lie within the plan')
    training = GraphDataset(config['manifest'], 'train', sidecar=config['train_sidecar'], max_len=config['max_len'])
    validation = PreparedDataset(config['manifest'], 'validation', max_len=config['max_len'])
    validation_graphs = GraphStore(config['validation_sidecar'], manifest_sha256=validation.manifest_sha256, split='validation')
    anchor_data = PreparedDataset(config['manifest'], 'train', max_len=config['max_len'])
    # The release-bound sampling tags are supplied with the experiment driver
    # independently below; preserve their hash and identity in the resume plan.
    tags_path = args.peptide_tags
    metadata = training.metadata(tags_path)
    sampler = DomainBatchSampler(metadata, updates=config['updates'], accumulation=config['accumulation'],
        batch_size=config['batch_size'], weights=config['weights'], rank=rank, world_size=world,
        seed=config['seed'], start_update=start, peptide_group_weights=config['peptide_group_weights'],
        stratum_weights=config['stratum_weights'], excluded_domains=config['excluded_domains'])
    anchors = {domain: DomainReplaySampler(anchor_data, domain, vocabulary_size=len(tokenizer), updates=config['updates'],
        batch_size=config['retention_batch_size'], rank=rank, world_size=world, seed=config['seed'], start_update=start)
        for domain in sorted(config['retention_domains'])}
    retention = RetentionObjective(reference, weight=config['retention_weight'],
        supervised_weight=config['retention_supervised_weight'], temperature=config['retention_temperature'])
    selected, evaluation_metadata = validation_indices(validation,
        {domain: 128 if domain == 'complete' else 192 if domain == 'peptide' else 64 for domain in config['weights']}, tags_path)
    sources = {str(path.resolve()): file_hash(path) for path in Path(__file__).parent.glob('*.py')}
    plan = {'schema_version': 'molsight_chemical_multitask_v1', 'configuration': configuration,
        'experiment': config, 'experiment_sha256': file_hash(args.experiment), 'world_size': world,
        'preflight': args.preflight, 'new_parameter_keys': initialized,
        'initial_checkpoint_sha256': file_hash(config['checkpoint']),
        'vocabulary_sha256': specification['vocabulary_sha256'], 'manifest_sha256': training.manifest_sha256,
        'peptide_tags_sha256': file_hash(tags_path) if tags_path else None,
        'sampler_sha256': sampler.definition_sha256, 'anchor_sampler_sha256': {domain: sampler.definition_sha256 for domain, sampler in anchors.items()},
        'graph_admission_sha256': {'train': training.graphs.admission_sha256, 'validation': validation_graphs.admission_sha256},
        'runtime': runtime_signature(), 'multitask_sources': sources, 'training_precision': 'bf16',
        'evaluation_precision': 'fp32', 'chemical_head_inference': 'auxiliary_only_no_drawing_stereo_claim'}
    baseline = None
    if restart:
        if restart['plan'] != plan or restart['validation_indices'] != selected:
            raise ValueError('Resume changes the training contract or validation cohort')
        optimizer.load_state_dict(restart['optimizer'])
        scheduler.load_state_dict(restart['scheduler'])
        scaler.load_state_dict(restart['scaler'])
        baseline = restart['baseline']
    distributed = DistributedDataParallel(student, device_ids=[local], broadcast_buffers=False) if world > 1 else student
    snapshot = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()} if args.preflight else None
    criterion = GraphCriterion(SimpleNamespace(**configuration), tokenizer, edge_weight=config['edge_weight']).to(device)
    loader = DataLoader(training, batch_sampler=sampler, collate_fn=collate_graphs,
        generator=torch.Generator().manual_seed(config['seed'] + rank), **loader_options(config['workers']))
    anchor_loaders = {domain: DataLoader(anchor_data, batch_sampler=sampler, collate_fn=collate_characters,
        num_workers=0, generator=torch.Generator().manual_seed(config['seed'] + rank + 0xA11CE + index))
        for index, (domain, sampler) in enumerate(anchors.items())}
    if baseline is None and not args.preflight:
        baseline, rows = evaluate(student, validation, selected, evaluation_metadata, device=device,
            rank=rank, world_size=world, batch_size=config['eval_batch_size'], precision='fp32')
        if rank == 0:
            (output / 'baseline.json').write_text(json.dumps({'metrics': baseline, 'records': rows,
                'peptide_strata': aggregate(rows, group_by='peptide_stratum')}, indent=2) + '\n')
    baseline = baseline or {}
    iterator = iter(loader)
    anchor_iterators = {domain: iter(loader) for domain, loader in anchor_loaders.items()}
    if restart:
        restore_rng(restart['rank_rng'][rank], device)
    if rank == 0:
        (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(json.dumps({'start_update': start, 'preflight': args.preflight, 'new_parameters': initialized}), flush=True)
    began = time.monotonic()
    for update in range(start, config['updates']):
        distributed.train()
        student.image_encoder.eval()
        criterion.edge_weight = config['edge_weight'] * min(1., (update + 1) / max(1, config['edge_warmup']))
        batches = [next(iterator) for _ in range(config['accumulation'])]
        observer = TokenMetrics(criterion.character.criterion['sequence'].criterion.ignore_indices, device)
        torch.cuda.reset_peak_memory_stats(device)
        losses, norm, stepped = multitask_update(distributed, criterion, batches, optimizer, scheduler,
            scaler, device, config['max_grad_norm'], amp=True, retention=retention,
            anchor_batch=[{'name': domain, 'weight': config['retention_domains'][domain], 'batch': next(iterator)}
                          for domain, iterator in anchor_iterators.items()], observer=observer)
        if not stepped:
            raise RuntimeError('BF16 optimizer update did not commit')
        graph_images = torch.tensor(sum(int(batch['chemical_edges'].ge(0).any(dim=(1, 2)).sum()) for batch in batches), device=device)
        if world > 1:
            dist.all_reduce(graph_images)
        record = {'update': update + 1, 'losses': losses, 'gradient_norm': float(norm),
            'metrics': observer.compute(), 'edge_weight': criterion.edge_weight,
            'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
            'domain': sampler.domain_for_update(update), 'graph_supervised_images': int(graph_images),
            'global_images': sum(len(batch['idx']) for batch in batches) * world,
            'replay_images': config['retention_batch_size'] * world * len(anchors),
            'input_resolutions': sorted({batch['image'].shape[-1] for batch in batches}),
            'peak_allocated_bytes_rank': torch.cuda.max_memory_allocated(device),
            'seconds': time.monotonic() - began}
        if rank == 0:
            with (output / 'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        stopping = update + 1 >= stop
        regression = False
        evaluation = not args.preflight and ((update + 1) % config['eval_every'] == 0 or update + 1 == config['updates'])
        if evaluation:
            scores, rows = evaluate(student, validation, selected, evaluation_metadata, device=device,
                rank=rank, world_size=world, batch_size=config['eval_batch_size'], precision='fp32')
            regression = any(scores[domain]['resolved_exact'] < baseline[domain]['resolved_exact'] - config['retention_tolerance']
                             for domain in config['retention_domains'])
            if rank == 0:
                (output / f'validation_{update+1:06d}.json').write_text(json.dumps({'metrics': scores,
                    'records': rows, 'peptide_strata': aggregate(rows, group_by='peptide_stratum'),
                    'input_resolutions': aggregate(rows, group_by='input_size'), 'retention_regression': regression}, indent=2) + '\n')
        if args.preflight:
            if any(not torch.equal(value, student.state_dict()[key].detach().cpu()) for key, value in snapshot.items()):
                raise RuntimeError('Zero-LR preflight changed model state')
        if stopping or evaluation or regression or (update + 1) % config['save_every'] == 0:
            suffix = '.regressed.pth' if regression else '.pth'
            if any(value.is_floating_point() and not torch.isfinite(value).all() for value in student.state_dict().values()):
                raise FloatingPointError('Nonfinite model state before checkpoint save')
            if any(torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all()
                   for state in optimizer.state.values() for value in state.values()):
                raise FloatingPointError('Nonfinite optimizer state before checkpoint save')
            save_checkpoint(output / f'update_{update+1:06d}{suffix}', distributed, optimizer, scheduler,
                scaler, update + 1, plan, tokenizer.stoi, baseline, selected, rank=rank, world_size=world, device=device)
        if regression:
            raise RuntimeError('Retained recognition domain regressed')
        if stopping:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--peptide-tags', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--stop-after', type=int)
    arguments = parser.parse_args()
    if arguments.stop_after is not None and arguments.stop_after < 1:
        parser.error('--stop-after must be positive')
    run(arguments)

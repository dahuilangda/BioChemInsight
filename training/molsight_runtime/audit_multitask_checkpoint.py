"""Audit persisted joint-training state without claiming recognition quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from molsight.checkpoint import model_state


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def audit(path):
    candidate = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    plan = candidate['plan']
    initial_path = plan['experiment']['checkpoint']
    initial = torch.load(initial_path, map_location='cpu', mmap=True, weights_only=False)
    old, new = model_state(initial), model_state(candidate)
    issues, groups = [], {}
    expected_new = set(plan['new_parameter_keys'])
    if set(new) - set(old) != expected_new or set(old) - set(new):
        issues.append('Unexpected model key changes')
    if candidate['vocabulary'] != initial['vocabulary']:
        issues.append('Vocabulary changed')
    if digest(initial_path) != plan['initial_checkpoint_sha256']:
        issues.append('Initial checkpoint hash changed')
    for name, value in new.items():
        if not value.is_floating_point():
            if name in old and not torch.equal(value, old[name]):
                issues.append(f'Nonfloating buffer changed: {name}')
            continue
        if not torch.isfinite(value).all():
            issues.append(f'Nonfinite model tensor: {name}')
        group = 'chemical_edge_head' if name in expected_new else 'encoder' if name.startswith('image_encoder.') else 'decoder'
        row = groups.setdefault(group, dict(tensors=0, compared_tensors=0, changed_tensors=0, elements=0,
            squared_delta=0., squared_initial=0., max_absolute_delta=0.))
        row['tensors'] += 1
        row['elements'] += value.numel()
        if name in old:
            row['compared_tensors'] += 1
            reference = old[name]
            if reference.shape != value.shape or reference.dtype != value.dtype:
                issues.append(f'Model tensor shape/dtype changed: {name}')
                continue
            delta = value.double() - reference.double()
            row['changed_tensors'] += int(not torch.equal(value, reference))
            row['squared_delta'] += float(delta.square().sum())
            row['squared_initial'] += float(reference.double().square().sum())
            row['max_absolute_delta'] = max(row['max_absolute_delta'], float(delta.abs().max()))
    for row in groups.values():
        row['relative_l2_change'] = math.sqrt(row['squared_delta'] / row['squared_initial']) if row['squared_initial'] else None
    optimizer = candidate['optimizer']
    optimization = []
    seen = set()
    for group in optimizer['param_groups']:
        steps = []
        if not math.isfinite(group['lr']):
            issues.append(f'Nonfinite learning rate: {group["name"]}')
        for parameter in group['params']:
            if parameter in seen:
                issues.append('Duplicate optimizer parameter')
            seen.add(parameter)
            state = optimizer['state'].get(parameter, {})
            for key, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    issues.append(f'Nonfinite optimizer tensor: {parameter}/{key}')
            if 'step' in state:
                step = float(state['step'])
                steps.append(step)
                if not 0 < step <= candidate['update']:
                    issues.append(f'Invalid optimizer step: {parameter}')
        optimization.append(dict(name=group['name'], parameters=len(group['params']),
            initialized_states=len(steps), step_range=[min(steps), max(steps)] if steps else None,
            next_learning_rate=group['lr']))
        if not steps:
            issues.append(f'Optimizer group never initialized: {group["name"]}')
    if not plan['preflight']:
        for group in ('encoder', 'decoder'):
            configured_lr = plan['experiment']['encoder_lr' if group == 'encoder' else 'lr']
            if configured_lr > 0 and groups[group]['changed_tensors'] == 0:
                issues.append(f'Trainable model group did not change: {group}')
    if len(candidate['rank_rng']) != plan['world_size']:
        issues.append('Rank RNG coverage mismatch')
    if candidate['scheduler']['last_epoch'] != candidate['update']:
        issues.append('Scheduler/update mismatch')
    for source, expected in plan['multitask_sources'].items():
        if digest(source) != expected:
            issues.append(f'Training source changed: {source}')
    return dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=digest(path),
        update=candidate['update'], passed=not issues, issues=issues, model_groups=groups,
        optimizer_groups=optimization, rank_rng_count=len(candidate['rank_rng']),
        expected_new_parameters=sorted(expected_new), scheduler=candidate['scheduler'],
        limitations=['State audit does not establish recognition improvement, inference reload, or exact restart.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(args.checkpoint)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)

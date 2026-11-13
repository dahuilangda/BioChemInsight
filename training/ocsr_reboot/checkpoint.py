"""Atomic, integrity-bound training checkpoints for trusted local runs.

Loading uses Python serialization for optimizer/RNG state. Only load files from
this trusted training workflow; integrity hashes are not signatures.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .protocol import ProtocolError


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def tree_sha256(value: Any) -> str:
    """Stable tensor/config/state digest independent of torch.save zip metadata."""
    digest = hashlib.sha256()

    def emit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b'tensor')
            digest.update(str(item.dtype).encode())
            digest.update(repr(tuple(item.shape)).encode())
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(data.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b'ndarray' + item.dtype.str.encode() + repr(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, np.generic):
            digest.update(b'numpy_scalar' + item.dtype.str.encode() + item.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b'mapping')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                emit(key); emit(item[key])
            digest.update(b'end_mapping')
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                emit(child)
            digest.update(b'end_sequence')
        elif item is None or type(item) in (str, int, float, bool, bytes):
            digest.update(type(item).__name__.encode() + b':' + repr(item).encode() + b'\0')
        else:
            raise ProtocolError(f'Unsupported checkpoint value: {type(item).__name__}')

    emit(value)
    return digest.hexdigest()


def _assert_finite(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ProtocolError('Nonfinite checkpoint tensor')
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(key); _assert_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: _assert_finite(item)
    elif isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            raise ProtocolError('Object-dtype checkpoint arrays are unsupported')
        if value.dtype.kind in 'biufc' and not np.isfinite(value).all():
            raise ProtocolError('Nonfinite checkpoint NumPy value')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError('Nonfinite checkpoint scalar')


def _class_name(instance: Any) -> str | None:
    if instance is None:
        return None
    cls = type(instance)
    return cls.__module__ + '.' + cls.__qualname__


def capture_rng_state() -> dict[str, Any]:
    # Avoid initializing CUDA merely to save a CPU unit-test checkpoint.
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _validate_rng_state(state)
    if state['torch_cuda'] is not None:
        if not torch.cuda.is_initialized() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ProtocolError('CUDA RNG device topology differs from the saved run')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """Preflight CPU state with private generators, never changing global RNG."""
    if not isinstance(state, Mapping) or set(state) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ProtocolError('Incomplete RNG state')
    try:
        random.Random().setstate(state['python'])
        np.random.RandomState().set_state(state['numpy'])
        torch.Generator(device='cpu').set_state(state['torch_cpu'])
    except (ValueError, TypeError, RuntimeError, KeyError) as exc:
        raise ProtocolError('Invalid CPU RNG state') from exc
    cuda_states = state['torch_cuda']
    if cuda_states is not None:
        if (not isinstance(cuda_states, (tuple, list)) or not torch.cuda.is_initialized()
                or len(cuda_states) != torch.cuda.device_count()):
            raise ProtocolError('CUDA RNG device topology differs from the saved run')
        for tensor in cuda_states:
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8
                    or tensor.ndim != 1 or tensor.device.type != 'cpu' or not tensor.numel()):
                raise ProtocolError('Invalid CUDA RNG byte tensor')


def _optimizer_names(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        selected = []
        for parameter in group['params']:
            if id(parameter) not in names or id(parameter) in seen:
                raise ProtocolError('Optimizer parameters are foreign or repeated')
            seen.add(id(parameter)); selected.append(names[id(parameter)])
        groups.append(selected)
    return groups


def atomic_save_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *,
    step: int, config: Mapping[str, Any], provenance: Mapping[str, Any],
    scheduler: Any = None, sampler_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if type(step) is not int or step < 0:
        raise ProtocolError('Checkpoint step must be an actual nonnegative optimizer-update count')
    if not config or not provenance:
        raise ProtocolError('Configuration and source/data/split provenance are required')
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ProtocolError('Scheduler is bound to a different optimizer')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 2, 'step': step,
        'model_class': _class_name(model),
        'model': _snapshot(model.state_dict()),
        'optimizer_class': _class_name(optimizer),
        'optimizer': _snapshot(optimizer.state_dict()),
        'optimizer_parameter_names': _optimizer_names(model, optimizer),
        'scheduler_class': _class_name(scheduler),
        'scheduler': _snapshot(scheduler.state_dict()) if scheduler is not None else None,
        'rng': capture_rng_state(), 'sampler_state': _snapshot(sampler_state),
        'config': _snapshot(dict(config)), 'provenance': _snapshot(dict(provenance)),
    }
    _assert_finite(payload)
    payload['payload_sha256'] = tree_sha256(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Link creates an immutable update checkpoint atomically and fails
            # on an existing name, including a concurrent writer's checkpoint.
            os.link(temporary, path)
            temporary.unlink()
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()
    return {'path': str(path.resolve()), 'step': step, 'payload_sha256': payload['payload_sha256']}


def load_checkpoint_strict(
    path: str | Path, model: torch.nn.Module, *,
    expected_config: Mapping[str, Any], expected_provenance: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None,
    restore_rng: bool = True, expected_step: int | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected_keys = {'schema_version', 'step', 'model_class', 'model', 'optimizer_class',
                     'optimizer', 'optimizer_parameter_names', 'scheduler_class',
                     'scheduler', 'rng', 'sampler_state', 'config', 'provenance', 'payload_sha256'}
    if set(payload) != expected_keys or payload['schema_version'] != 2:
        raise ProtocolError('Unsupported or incomplete checkpoint schema')
    # Validate everything before mutating a live model, including floating
    # metadata, scheduler state, sampler state and NumPy RNG arrays/scalars.
    _assert_finite(payload)
    _assert_finite(expected_config); _assert_finite(expected_provenance)
    actual_digest = tree_sha256({key: value for key, value in payload.items() if key != 'payload_sha256'})
    if actual_digest != payload['payload_sha256']:
        raise ProtocolError('Checkpoint integrity hash differs')
    if (tree_sha256(payload['config']) != tree_sha256(dict(expected_config))
            or tree_sha256(payload['provenance']) != tree_sha256(dict(expected_provenance))):
        raise ProtocolError('Configuration or source/data/split provenance differs')
    if payload['model_class'] != _class_name(model):
        raise ProtocolError('Model class differs')
    if optimizer is not None and payload['optimizer_class'] != _class_name(optimizer):
        raise ProtocolError('Optimizer class differs')
    if scheduler is not None and payload['scheduler_class'] != _class_name(scheduler):
        raise ProtocolError('Scheduler class differs')
    if scheduler is not None and (optimizer is None or scheduler.optimizer is not optimizer):
        raise ProtocolError('Scheduler restoration requires its matching optimizer')
    if type(payload['step']) is not int or payload['step'] < 0 or (expected_step is not None and payload['step'] != expected_step):
        raise ProtocolError('Actual optimizer-update count differs')
    current = model.state_dict()
    if set(current) != set(payload['model']):
        raise ProtocolError('Model parameter/buffer keys differ')
    for name, tensor in current.items():
        saved = payload['model'][name]
        if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
            raise ProtocolError(f'Model parameter/buffer shape or dtype differs: {name}')
    if optimizer is not None and _optimizer_names(model, optimizer) != payload['optimizer_parameter_names']:
        raise ProtocolError('Optimizer parameter group/name ordering differs')
    if scheduler is not None and payload['scheduler'] is None:
        raise ProtocolError('Missing scheduler state')
    if restore_rng:
        _validate_rng_state(payload['rng'])
    model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if tree_sha256(model.state_dict()) != tree_sha256(payload['model']):
        raise ProtocolError('Strict reload changed parameter bytes')
    if optimizer is not None and tree_sha256(optimizer.state_dict()) != tree_sha256(payload['optimizer']):
        raise ProtocolError('Strict reload changed optimizer state')
    if restore_rng:
        restore_rng_state(payload['rng'])
    return {'step': payload['step'], 'sampler_state': payload['sampler_state'],
            'payload_sha256': payload['payload_sha256'],
            'model_sha256': tree_sha256(model.state_dict())}


def audit_parameter_update(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], *,
    trainable_names: set[str], required_changed: set[str] | None = None,
) -> dict[str, Any]:
    if set(before) != set(after) or not trainable_names <= set(before):
        raise ProtocolError('Parameter audit keys differ')
    _assert_finite(after)
    changed, frozen_changed, unchanged_trainable = [], [], []
    for name in before:
        if before[name].shape != after[name].shape or before[name].dtype != after[name].dtype:
            raise ProtocolError(f'Parameter audit shape/dtype differs: {name}')
        if not torch.equal(before[name].cpu(), after[name].cpu()):
            changed.append(name)
            if name not in trainable_names: frozen_changed.append(name)
        elif name in trainable_names:
            unchanged_trainable.append(name)
    if frozen_changed:
        raise ProtocolError(f'Frozen parameters/buffers changed: {frozen_changed}')
    if not set(changed) & trainable_names:
        raise ProtocolError('No trainable parameter changed')
    if required_changed and not required_changed <= set(changed):
        raise ProtocolError(f'Required parameter did not change: {sorted(required_changed - set(changed))}')
    return {'changed': changed, 'unchanged_trainable': unchanged_trainable,
            'frozen_unchanged': True, 'all_finite': True,
            'before_sha256': tree_sha256(before), 'after_sha256': tree_sha256(after)}

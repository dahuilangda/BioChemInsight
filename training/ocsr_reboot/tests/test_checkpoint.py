import random

import numpy as np
import pytest
import torch

from training.ocsr_reboot import checkpoint as checkpoint_module

from training.ocsr_reboot.checkpoint import (
    atomic_save_checkpoint, audit_parameter_update, load_checkpoint_strict, tree_sha256,
)
from training.ocsr_reboot.protocol import ProtocolError

CONFIG = {'architecture': 'linear-test', 'lr': .01}
PROVENANCE = {'source_sha256': 'source', 'split_sha256': 'split', 'data_sha256': 'data'}


def make_training():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    return model, optimizer, scheduler


def step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(3, 2)
    loss = (model(inputs) - torch.randn(3, 1)).square().mean()
    loss.backward(); optimizer.step(); scheduler.step()


def test_exact_optimizer_rng_and_sampler_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11); np.random.seed(12); random.seed(13)
    model, optimizer, scheduler = make_training()
    step(model, optimizer, scheduler)
    path = tmp_path / 'update_000001.pth'
    saved = atomic_save_checkpoint(path, model, optimizer, scheduler=scheduler, step=1,
                                  config=CONFIG, provenance=PROVENANCE, sampler_state={'cursor': 16})
    expected_random = (random.random(), float(np.random.random()))
    step(model, optimizer, scheduler)
    expected_model = tree_sha256(model.state_dict())
    expected_optimizer = tree_sha256(optimizer.state_dict())
    restored, restored_optimizer, restored_scheduler = make_training()
    result = load_checkpoint_strict(path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler,
                                    expected_config=CONFIG, expected_provenance=PROVENANCE, expected_step=1)
    assert result['payload_sha256'] == saved['payload_sha256']
    assert result['sampler_state'] == {'cursor': 16}
    assert (random.random(), float(np.random.random())) == expected_random
    step(restored, restored_optimizer, restored_scheduler)
    assert tree_sha256(restored.state_dict()) == expected_model
    assert tree_sha256(restored_optimizer.state_dict()) == expected_optimizer
    assert list(tmp_path.glob('*.tmp')) == []


def test_metadata_mismatch_and_corrupt_payload_are_rejected(tmp_path):
    model, optimizer, scheduler = make_training()
    path = tmp_path / 'checkpoint.pth'
    atomic_save_checkpoint(path, model, optimizer, scheduler=scheduler, step=0, config=CONFIG, provenance=PROVENANCE)
    for wrong in ({'expected_config': {'different': 1}, 'expected_provenance': PROVENANCE},
                  {'expected_config': CONFIG, 'expected_provenance': {'different': 1}},
                  {'expected_config': CONFIG, 'expected_provenance': PROVENANCE, 'expected_step': 10}):
        with pytest.raises(ProtocolError): load_checkpoint_strict(path, model, **wrong)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['step'] = 1000
    torch.save(payload, path)
    with pytest.raises(ProtocolError, match='integrity'):
        load_checkpoint_strict(path, model, expected_config=CONFIG, expected_provenance=PROVENANCE)


def test_optimizer_parameter_order_must_match(tmp_path):
    model, optimizer, _ = make_training()
    path = tmp_path / 'checkpoint.pth'
    atomic_save_checkpoint(path, model, optimizer, step=0, config=CONFIG, provenance=PROVENANCE)
    reversed_optimizer = torch.optim.AdamW(list(model.parameters())[::-1], lr=.01)
    with pytest.raises(ProtocolError, match='ordering'):
        load_checkpoint_strict(path, model, optimizer=reversed_optimizer, expected_config=CONFIG, expected_provenance=PROVENANCE)


def test_update_snapshots_are_immutable_unless_replacement_is_explicit(tmp_path):
    model, optimizer, _ = make_training()
    path = tmp_path / 'checkpoint.pth'
    atomic_save_checkpoint(path, model, optimizer, step=1, config=CONFIG, provenance=PROVENANCE)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_save_checkpoint(path, model, optimizer, step=2, config=CONFIG, provenance=PROVENANCE)
    assert path.read_bytes() == original
    assert list(tmp_path.glob('*.tmp')) == []
    atomic_save_checkpoint(path, model, optimizer, step=2, config=CONFIG, provenance=PROVENANCE, overwrite=True)
    assert load_checkpoint_strict(path, model, expected_config=CONFIG, expected_provenance=PROVENANCE, expected_step=2)['step'] == 2


def test_nonfinite_model_and_dead_or_frozen_updates_are_rejected(tmp_path):
    model, optimizer, scheduler = make_training()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ProtocolError, match='No trainable'):
        audit_parameter_update(before, model.state_dict(), trainable_names=set(before))
    step(model, optimizer, scheduler)
    report = audit_parameter_update(before, model.state_dict(), trainable_names=set(before), required_changed=set(before))
    assert report['all_finite'] and report['frozen_unchanged']
    with pytest.raises(ProtocolError, match='Frozen'):
        audit_parameter_update(before, model.state_dict(), trainable_names={'weight'})
    with torch.no_grad(): model.weight.fill_(float('nan'))
    with pytest.raises(ProtocolError, match='Nonfinite'):
        atomic_save_checkpoint(tmp_path / 'bad.pth', model, optimizer, step=1, config=CONFIG, provenance=PROVENANCE)


def test_different_optimizer_algorithm_is_rejected_before_model_mutation(tmp_path):
    model, optimizer, _ = make_training()
    path = tmp_path / 'adamw.pth'
    atomic_save_checkpoint(path, model, optimizer, step=0, config=CONFIG, provenance=PROVENANCE)
    other_model = torch.nn.Linear(2, 1)
    before = tree_sha256(other_model.state_dict())
    # Adam and AdamW share state/group shapes, so parameter-name checks alone
    # cannot distinguish their mathematically different update algorithms.
    other_optimizer = torch.optim.Adam(other_model.parameters(), lr=.01)
    with pytest.raises(ProtocolError, match='Optimizer class'):
        load_checkpoint_strict(path, other_model, optimizer=other_optimizer,
                               expected_config=CONFIG, expected_provenance=PROVENANCE)
    assert tree_sha256(other_model.state_dict()) == before


def test_different_scheduler_algorithm_is_rejected_before_model_mutation(tmp_path):
    model, optimizer, scheduler = make_training()
    path = tmp_path / 'step_lr.pth'
    atomic_save_checkpoint(path, model, optimizer, scheduler=scheduler, step=0, config=CONFIG, provenance=PROVENANCE)
    other_model, other_optimizer, _ = make_training()
    before = tree_sha256(other_model.state_dict())
    other_scheduler = torch.optim.lr_scheduler.ExponentialLR(other_optimizer, gamma=.8)
    with pytest.raises(ProtocolError, match='Scheduler class'):
        load_checkpoint_strict(path, other_model, optimizer=other_optimizer, scheduler=other_scheduler,
                               expected_config=CONFIG, expected_provenance=PROVENANCE)
    assert tree_sha256(other_model.state_dict()) == before


@pytest.mark.parametrize('field', ['config', 'provenance', 'scheduler', 'rng', 'sampler_state'])
def test_nonfinite_entire_payload_is_rejected_on_save(tmp_path, monkeypatch, field):
    model, optimizer, scheduler = make_training()
    config, provenance, sampler_state = dict(CONFIG), dict(PROVENANCE), {'cursor': 0}
    bad = np.array([1., float('nan')]) if field in ('provenance', 'rng') else float('inf')
    if field == 'config': config['bad'] = bad
    elif field == 'provenance': provenance['bad'] = bad
    elif field == 'scheduler': monkeypatch.setattr(scheduler, 'state_dict', lambda: {'bad': bad})
    elif field == 'rng': monkeypatch.setattr(checkpoint_module, 'capture_rng_state', lambda: {'bad': bad})
    else: sampler_state['bad'] = np.float32(float('nan'))
    path = tmp_path / 'bad.pth'
    with pytest.raises(ProtocolError, match='Nonfinite'):
        atomic_save_checkpoint(path, model, optimizer, scheduler=scheduler, step=0,
                               config=config, provenance=provenance, sampler_state=sampler_state)
    assert not path.exists()
    assert list(tmp_path.glob('*.tmp')) == []


@pytest.mark.parametrize('field', ['config', 'provenance', 'scheduler', 'rng', 'sampler_state'])
def test_nonfinite_entire_payload_is_rejected_on_load_even_with_matching_digest(tmp_path, field):
    model, optimizer, scheduler = make_training()
    path = tmp_path / 'bad.pth'
    atomic_save_checkpoint(path, model, optimizer, scheduler=scheduler, step=0,
                           config=CONFIG, provenance=PROVENANCE, sampler_state={'cursor': 0})
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload[field]['bad'] = np.array([float('inf')])
    payload['payload_sha256'] = tree_sha256({key: value for key, value in payload.items() if key != 'payload_sha256'})
    torch.save(payload, path)
    before = tree_sha256(model.state_dict())
    with pytest.raises(ProtocolError, match='Nonfinite'):
        load_checkpoint_strict(path, model, expected_config=CONFIG, expected_provenance=PROVENANCE)
    assert tree_sha256(model.state_dict()) == before


def test_numeric_numpy_metadata_has_a_reproducible_digest(tmp_path):
    model, optimizer, _ = make_training()
    config = {**CONFIG, 'scale': np.array([1., 2.], dtype=np.float32), 'count': np.int64(3)}
    path = tmp_path / 'arrays.pth'
    atomic_save_checkpoint(path, model, optimizer, step=0, config=config, provenance=PROVENANCE)
    loaded = load_checkpoint_strict(path, model, expected_config=config, expected_provenance=PROVENANCE)
    assert loaded['step'] == 0


@pytest.mark.parametrize('rng_key', ['python', 'numpy', 'torch_cpu', 'torch_cuda'])
def test_invalid_rng_is_rejected_before_mutating_model_or_optimizer(tmp_path, rng_key):
    model, optimizer, _ = make_training()
    path = tmp_path / 'rng.pth'
    atomic_save_checkpoint(path, model, optimizer, step=0, config=CONFIG, provenance=PROVENANCE)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['rng'][rng_key] = [torch.zeros(1, dtype=torch.uint8)] if rng_key == 'torch_cuda' else None
    payload['payload_sha256'] = tree_sha256({key: value for key, value in payload.items() if key != 'payload_sha256'})
    torch.save(payload, path)
    restored, restored_optimizer, _ = make_training()
    before_model = tree_sha256(restored.state_dict())
    before_optimizer = tree_sha256(restored_optimizer.state_dict())
    with pytest.raises(ProtocolError, match='RNG'):
        load_checkpoint_strict(path, restored, optimizer=restored_optimizer,
                               expected_config=CONFIG, expected_provenance=PROVENANCE)
    assert tree_sha256(restored.state_dict()) == before_model
    assert tree_sha256(restored_optimizer.state_dict()) == before_optimizer

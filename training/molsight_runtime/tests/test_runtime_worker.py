import copy
import importlib.util
from pathlib import Path

import pytest
import torch

worker_path = Path(__file__).resolve().parents[3] / 'utils' / 'molsight_runtime_worker.py'
spec = importlib.util.spec_from_file_location('runtime_worker', worker_path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture():
    state = {'decoder.token_embedding.weight': torch.zeros(4, 8),
             'decoder.blocks.0.mlp.w12.weight': torch.zeros(8, 8),
             'image_pos_embed': torch.zeros(1, 8, 64, 64)}
    vocabulary = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'C': 3}
    checkpoint = {'vocabulary': vocabulary, 'plan': {'configuration': {
        'embed_dim': 8, 'dec_n_layer': 1, 'use_qknorm': True, 'use_swiglu': True,
        'use_rmsnorm': True, 'image_pos_grid_size': [64, 64], 'dec_n_head': 2,
        'lora': False, 'max_len': 2048}}}
    return checkpoint, state, vocabulary


def test_configuration_uses_recorded_grid_heads_and_decode_limit():
    checkpoint, state, vocabulary = fixture()
    configuration = worker.configuration_for(checkpoint, state, vocabulary, input_size=768, max_len=2048)
    assert configuration.image_pos_grid_size == [64, 64]
    assert configuration.input_size == 768
    assert configuration.dec_n_head == 2
    assert configuration.use_checkpoint is False


@pytest.mark.parametrize('mutation', ['vocabulary', 'depth', 'grid', 'adapter', 'length', 'resolution'])
def test_refuses_incompatible_model_contract(mutation):
    checkpoint, state, vocabulary = fixture()
    vocabulary = copy.deepcopy(vocabulary)
    size, length = 512, 2048
    if mutation == 'vocabulary':
        vocabulary['C'] = 4
    elif mutation == 'depth':
        checkpoint['plan']['configuration']['dec_n_layer'] = 2
    elif mutation == 'grid':
        checkpoint['plan']['configuration']['image_pos_grid_size'] = [32, 32]
    elif mutation == 'adapter':
        checkpoint['plan']['configuration']['lora'] = True
    elif mutation == 'length':
        length = 4096
    else:
        size = 640
    with pytest.raises(ValueError):
        worker.configuration_for(checkpoint, state, vocabulary, input_size=size, max_len=length)

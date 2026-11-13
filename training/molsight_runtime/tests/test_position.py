import pytest
import torch
from molsight.model import MolsightModel, get_2d_rff_pos_embed
from molsight.position import fourier_grid, recover_frequencies, resize_fourier_grid
from torch import nn


def test_original_position_grid_is_returned_unchanged():
    original = get_2d_rff_pos_embed((32, 48), 16)
    resized = resize_fourier_grid(original, (32, 48))
    assert resized is original
    assert torch.equal(resized, original)


def test_recovered_frequencies_reproduce_the_continuous_position_function():
    torch.manual_seed(941)
    frequencies = torch.randn(2, 32)
    source = fourier_grid((32, 48), frequencies)
    recovered = recover_frequencies(source)
    torch.testing.assert_close(recovered.float(), frequencies, atol=1e-6, rtol=1e-6)
    actual = resize_fourier_grid(source, (64, 96))
    expected = fourier_grid((64, 96), frequencies)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-5)


def test_non_fourier_position_buffers_are_rejected():
    original = get_2d_rff_pos_embed((32, 48), 16)
    original[:, :, 4, 8] += .2
    with pytest.raises(ValueError, match='consistent unaliased Fourier'):
        resize_fourier_grid(original, (64, 96))


def test_position_cache_is_invalidated_when_checkpoint_buffers_change():
    model = MolsightModel.__new__(MolsightModel)
    nn.Module.__init__(model)
    model.register_buffer('image_pos_embed', get_2d_rff_pos_embed((32, 48), 16))
    features = torch.zeros(1, 32, 64, 96)
    first = model.image_positions(features)
    assert model.image_positions(features) is first
    model.load_state_dict({'image_pos_embed': get_2d_rff_pos_embed((32, 48), 16)})
    second = model.image_positions(features)
    assert not torch.equal(first, second)
    assert model.image_positions(torch.zeros(1, 32, 32, 48)) is model.image_pos_embed


def test_inference_tensors_support_larger_position_grids():
    with torch.inference_mode():
        model = MolsightModel.__new__(MolsightModel)
        nn.Module.__init__(model)
        model.register_buffer('image_pos_embed', get_2d_rff_pos_embed((32, 48), 16))
        positions = model.image_positions(torch.zeros(1, 32, 64, 96))
    assert positions.shape == (1, 32, 64, 96)

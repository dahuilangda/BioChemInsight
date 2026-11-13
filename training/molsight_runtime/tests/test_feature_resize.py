import pytest
import torch
from torch import nn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_unit_scale_bilinear_resize_has_identity_values_and_gradients(dtype):
    torch.manual_seed(19)
    original = torch.randn(2, 5, 7, 9, dtype=dtype, requires_grad=True)
    replacement = original.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(original)
    expected = nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False)(
        original
    )
    actual = nn.Identity()(replacement)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(replacement.grad, original.grad, rtol=0, atol=0)

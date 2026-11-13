import pytest
import torch
from molsight_multitask.edges import chemical_edge_sums
from molsight_multitask.focal import chemical_focal_sums


def labels(n, batch=1):
    result = torch.zeros(batch, n, n, dtype=torch.long)
    result.diagonal(dim1=1, dim2=2).fill_(-100)
    return result


def test_probability_uses_classes_not_number_of_candidate_atoms():
    losses = []
    for n in [2, 9, 200]:
        logits = torch.zeros(1, n, n, 7)
        logits[..., 0] = 10
        result = chemical_focal_sums(logits, labels(n))
        p_none = logits[0, 0, 1].softmax(-1)[0]
        reference = -p_none.log() * (1 - p_none).square()
        assert torch.allclose(result['loss_sum'], reference, rtol=.01, atol=1e-14)
        losses.append(result['loss_sum'])
    assert torch.allclose(losses[0], losses[-1], rtol=1e-5, atol=1e-14)


def test_gamma_zero_matches_balanced_objective_and_gradient():
    torch.manual_seed(17)
    logits = torch.randn(2, 4, 4, 7, requires_grad=True)
    target = labels(4, batch=2)
    target[:, 0, 1] = target[:, 1, 0] = 1
    target[1] = -100
    actual = chemical_focal_sums(logits, target, gamma=0)
    reference = chemical_edge_sums(logits, target)
    assert actual['images'] == reference['images'] == 1
    assert torch.allclose(actual['loss_sum'], reference['loss_sum'])
    a = torch.autograd.grad(actual['loss_sum'], logits, retain_graph=True)[0]
    b = torch.autograd.grad(reference['loss_sum'], logits)[0]
    assert torch.allclose(a, b)


@pytest.mark.parametrize('detach', [False, True])
def test_common_per_pair_class_shift_preserves_loss_and_gradient(detach):
    torch.manual_seed(33)
    logits = torch.randn(1, 4, 4, 7, requires_grad=True)
    target = labels(4)
    shift = torch.randn(1, 4, 4, 1) * 4
    first = chemical_focal_sums(logits, target, detach_weights=detach)['loss_sum']
    second = chemical_focal_sums(logits + shift, target, detach_weights=detach)['loss_sum']
    a = torch.autograd.grad(first, logits, retain_graph=True)[0]
    b = torch.autograd.grad(second, logits)[0]
    assert torch.allclose(first, second, atol=1e-6)
    assert torch.allclose(a, b, atol=1e-6)


def test_missing_labels_produce_zero_grad_and_no_fake_image():
    logits = torch.randn(2, 3, 3, 7, requires_grad=True)
    result = chemical_focal_sums(logits, torch.full((2, 3, 3), -100))
    assert result['images'] == 0 and result['loss_sum'] == 0
    result['loss_sum'].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_wrong_shapes_unmasked_self_pairs_and_nonfinite_are_rejected():
    logits = torch.zeros(1, 3, 3, 7)
    with pytest.raises(ValueError):
        chemical_focal_sums(logits, labels(1))
    with pytest.raises(ValueError):
        chemical_focal_sums(logits, torch.zeros(1, 3, 3, dtype=torch.long))
    logits[0, 0, 1, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        chemical_focal_sums(logits, labels(3))

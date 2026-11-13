import pytest
import torch
from molsight_multitask.edges import chemical_edge_sums, chemical_targets


def targets(count):
    value = torch.zeros(1, count, count, dtype=torch.long)
    value[:, torch.arange(count), torch.arange(count)] = -100
    value[0, 0, 1] = value[0, 1, 0] = 1
    return value


def test_missing_supervision_is_differentiable_zero():
    logits = torch.randn(2, 3, 3, 7, requires_grad=True)
    result = chemical_edge_sums(logits, torch.full((2, 3, 3), -100, dtype=torch.long))
    result['loss_sum'].backward()
    assert result['images'] == 0
    assert logits.grad.count_nonzero() == 0


def test_chemical_single_bond_does_not_invent_drawing_wedge_labels():
    values = []
    for category in (1, 5, 6):
        logits = torch.zeros(1, 2, 2, 7)
        logits[..., category] = 5
        values.append(chemical_edge_sums(logits, targets(2))['loss_sum'])
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(values[0], values[2])


def test_large_none_class_does_not_drown_positive_supervision():
    small = chemical_edge_sums(torch.zeros(1, 3, 3, 7), targets(3))
    large = chemical_edge_sums(torch.zeros(1, 30, 30, 7), targets(30))
    torch.testing.assert_close(small['loss_sum'], large['loss_sum'])
    assert large['negative_pairs'] > small['negative_pairs']


def test_node_permutation_preserves_loss_and_gradients_are_finite():
    logits = torch.randn(1, 4, 4, 7, requires_grad=True)
    target = targets(4)
    original = chemical_edge_sums(logits, target)['loss_sum']
    order = torch.tensor([2, 0, 3, 1])
    changed = chemical_edge_sums(logits[:, order][:, :, order], target[:, order][:, :, order])['loss_sum']
    torch.testing.assert_close(original, changed)
    original.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert logits.grad[0, torch.arange(4), torch.arange(4)].count_nonzero() == 0


def test_targets_preserve_aromatic_and_closure_bonds_and_mask_diagonal():
    graph = {'atoms': [{'atom_id': i} for i in range(3)],
             'bonds': [{'begin': 0, 'end': 1, 'chemical_type': 'AROMATIC'},
                       {'begin': 1, 'end': 2, 'chemical_type': 'DOUBLE'},
                       {'begin': 2, 'end': 0, 'chemical_type': 'SINGLE'}]}
    value = chemical_targets(graph)
    assert value.tolist() == [[-100, 4, 1], [4, -100, 2], [1, 2, -100]]


def test_unknown_bond_type_is_not_mislabeled_single():
    graph = {'atoms': [{'atom_id': 0}, {'atom_id': 1}],
             'bonds': [{'begin': 0, 'end': 1, 'chemical_type': 'DATIVE'}]}
    with pytest.raises(ValueError, match='Unsupported'):
        chemical_targets(graph)


def test_empty_atom_batch_has_zero_contribution():
    logits = torch.empty(2, 0, 0, 7, requires_grad=True)
    result = chemical_edge_sums(logits, torch.empty(2, 0, 0, dtype=torch.long))
    result['loss_sum'].backward()
    assert result['images'] == 0 and result['loss_sum'] == 0
    assert logits.grad.shape == logits.shape

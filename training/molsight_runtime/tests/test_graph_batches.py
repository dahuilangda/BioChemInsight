import pytest
import torch
from molsight_multitask.data import collate_graphs
from molsight_multitask.edges import chemical_edge_sums


def sample(index, atoms):
    return {'idx': index, 'image': torch.zeros(3, 8, 8), 'label': torch.tensor([1, 3, 3, 2]),
            'smiles': 'CC', 'atom_indices': torch.tensor([1, 2][:atoms], dtype=torch.long),
            'chemical_edges': torch.tensor([[-100, 1], [1, -100]], dtype=torch.long)
            if atoms else torch.empty(0, 0, dtype=torch.long)}


def test_missing_graph_keeps_character_example_and_masks_all_edge_pairs():
    batch = collate_graphs([sample(0, 2), sample(1, 0)])
    assert batch['label'].shape[0] == 2
    assert batch['chemical_edges'][1].eq(-100).all()
    logits = torch.randn(2, 2, 2, 7, requires_grad=True)
    result = chemical_edge_sums(logits, batch['chemical_edges'])
    result['loss_sum'].backward()
    assert result['images'] == 1
    assert logits.grad[1].count_nonzero() == 0


def test_all_character_only_batch_has_empty_graph_dimensions():
    batch = collate_graphs([sample(0, 0), sample(1, 0)])
    assert batch['atom_indices'].shape == (2, 0)
    assert batch['chemical_edges'].shape == (2, 0, 0)
    assert batch['smiles'] == ['CC', 'CC']


def test_invalid_query_cannot_silently_select_padding_or_eos():
    value = sample(0, 2)
    value['atom_indices'][1] = 3
    with pytest.raises(ValueError, match='Atom query'):
        collate_graphs([value])

"""Numerical and semantic contracts; no encoder, datasets or GPU required."""
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from training.ocsr_reboot.objectives import (
    IGNORE_INDEX,
    focal_objectness_loss,
    hungarian_atom_cost,
    logits_to_chemical_probabilities,
    make_denoising_targets,
    matched_supervision,
    masked_coordinate_l1,
    merge_chemical_probabilities,
)


def test_focal_rejects_broadcasting_and_wrong_mask():
    logits = torch.zeros(1, 64)
    with pytest.raises(ValueError, match='broadcasting'):
        focal_objectness_loss(logits, torch.ones(1))
    with pytest.raises(ValueError, match='valid_mask'):
        focal_objectness_loss(logits, torch.ones_like(logits), valid_mask=torch.ones(64, dtype=torch.bool))
    with pytest.raises(ValueError, match='zero or one'):
        focal_objectness_loss(torch.zeros(1), torch.tensor([0.5]))


def test_dn_positive_and_negative_gradients_have_opposite_signs():
    gold = torch.tensor([[0.1, 0.2], [0.7, 0.8]])
    dn = make_denoising_targets(gold, torch.tensor([1, 0]), torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                                negative_exclusion_radius=0.03, noise_std=0)
    assert dn.objectness_targets.tolist() == [1, 1, 0, 0]
    assert dn.coordinate_valid_mask.tolist() == [True, True, False, False]
    torch.testing.assert_close(dn.coordinate_targets[:2], gold[[1, 0]])
    logits = torch.zeros(4, requires_grad=True)
    focal_objectness_loss(logits, dn.objectness_targets, valid_mask=dn.objectness_valid_mask).backward()
    assert (logits.grad[:2] < 0).all()
    assert (logits.grad[2:] > 0).all()


def test_dn_near_atom_random_negatives_are_unknown_not_false_negatives():
    gold = torch.tensor([[0.1, 0.2]])
    dn = make_denoising_targets(gold, torch.tensor([0]),
                                torch.tensor([[0.101, 0.201], [0.9, 0.9]]),
                                negative_exclusion_radius=0.02, noise_std=0)
    assert dn.objectness_valid_mask.tolist() == [True, False, True]
    logits = torch.zeros(3, requires_grad=True)
    loss = focal_objectness_loss(logits, dn.objectness_targets, valid_mask=dn.objectness_valid_mask)
    reference = focal_objectness_loss(logits[[0, 2]], dn.objectness_targets[[0, 2]])
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert logits.grad[1] == 0


def test_focal_extreme_logits_still_have_finite_corrective_gradients():
    logits = torch.tensor([-1000., 1000.], requires_grad=True)
    loss = focal_objectness_loss(logits, torch.tensor([1., 0.]))
    assert torch.isfinite(loss)
    loss.backward()
    assert (logits.grad != 0).all()
    assert logits.grad[0] < 0 < logits.grad[1]


@pytest.mark.parametrize('empty', [False, True])
def test_empty_or_missing_focal_returns_differentiable_zero(empty):
    logits = torch.zeros(0 if empty else 3, requires_grad=True)
    targets = torch.full_like(logits, IGNORE_INDEX)
    loss = focal_objectness_loss(logits, targets)
    assert loss == 0 and loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_focal_double_gradient_and_bfloat_reduction():
    logits = torch.tensor([-.8, .3], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 0])
    assert torch.autograd.gradcheck(lambda value: focal_objectness_loss(value, targets), (logits,))
    value = focal_objectness_loss(torch.zeros(2, dtype=torch.bfloat16), targets)
    assert value.dtype == torch.float32


def test_hungarian_prefers_high_confidence_at_equal_coordinates():
    probabilities = torch.tensor([.05, .5, .95])
    cost = hungarian_atom_cost(torch.logit(probabilities), torch.zeros(3, 2), torch.zeros(1, 2))
    query, gold = linear_sum_assignment(cost.numpy())
    assert query.tolist() == [2] and gold.tolist() == [0]
    assert cost[2, 0] < cost[1, 0] < cost[0, 0]
    assert not cost.requires_grad


def test_hungarian_coordinate_term_and_empty_target():
    logits = torch.zeros(2)
    points = torch.tensor([[0., 0.], [1., 1.]])
    cost = hungarian_atom_cost(logits, points, torch.tensor([[1., 1.]]), class_weight=0)
    assert cost[:, 0].tolist() == [10., 0.]
    assert hungarian_atom_cost(logits, points, torch.empty(0, 2)).shape == (2, 0)
    with pytest.raises(ValueError, match='one entry'):
        hungarian_atom_cost(torch.zeros(1, 2), points, torch.empty(0, 2))
    with pytest.raises(ValueError, match='normalized'):
        hungarian_atom_cost(logits, points + 1, points)


def test_chemical_logits_and_probability_apis_agree_on_marginalization():
    logits = torch.tensor([0., 1., 1.5, -3., -3., 1., 1.], requires_grad=True)
    raw = logits.softmax(-1)
    assert raw.argmax() == 2
    probabilities = logits_to_chemical_probabilities(logits)
    assert probabilities.argmax() == 1
    torch.testing.assert_close(probabilities, merge_chemical_probabilities(raw))
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.))
    (-probabilities[1].log()).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[[1, 5, 6]] < 0).all()
    with pytest.raises(ValueError, match='probabilities'):
        merge_chemical_probabilities(logits.detach())
    with pytest.raises(ValueError, match='seven'):
        logits_to_chemical_probabilities(torch.zeros(5))


def test_permutation_preserves_chemical_edges_and_elements():
    gold_coordinates = torch.tensor([[.1, .2], [.3, .4], [.5, .6]])
    elements = torch.tensor([6, 7, 8])
    edges = torch.tensor([[-100, 1, 0], [1, -100, 2], [0, 2, -100]])
    result = matched_supervision(5, 3, torch.tensor([4, 0, 2]), torch.tensor([2, 0, 1]),
                                 gold_coordinates=gold_coordinates, gold_elements=elements, gold_edges=edges)
    assert result.objectness_targets.tolist() == [1, 0, 1, 0, 1]
    assert result.objectness_valid_mask.all()
    assert result.element_targets.tolist() == [6, -100, 7, -100, 8]
    torch.testing.assert_close(result.coordinate_targets[[0, 2, 4]], gold_coordinates)
    torch.testing.assert_close(result.edge_targets[[0, 2, 4]][:, [0, 2, 4]], edges)
    assert result.edge_targets[1].eq(-100).all() and result.edge_targets[:, 3].eq(-100).all()


def test_missing_annotations_are_ignored_and_partial_objectness_is_unknown():
    result = matched_supervision(3, 4, torch.tensor([1]), torch.tensor([2]), complete_annotations=False)
    assert result.objectness_valid_mask.tolist() == [False, True, False]
    assert result.coordinate_valid_mask.sum() == 0
    assert result.element_targets.eq(IGNORE_INDEX).all()
    assert result.edge_targets.eq(IGNORE_INDEX).all()
    with pytest.raises(ValueError, match='every gold'):
        matched_supervision(3, 4, torch.tensor([1]), torch.tensor([2]))


def test_coordinate_mask_does_not_invent_targets_from_missing_nan_positions():
    result = matched_supervision(2, 2, torch.tensor([1, 0]), torch.tensor([0, 1]),
                                 gold_coordinates=torch.tensor([[float('nan'), float('nan')], [.3, .4]]),
                                 gold_coordinate_mask=torch.tensor([False, True]))
    assert result.coordinate_valid_mask.tolist() == [True, False]
    assert torch.isfinite(result.coordinate_targets).all()
    torch.testing.assert_close(result.coordinate_targets[0], torch.tensor([.3, .4]))


def test_empty_gold_is_known_background_and_dn_can_have_no_positives():
    result = matched_supervision(3, 0, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    assert result.objectness_targets.eq(0).all() and result.objectness_valid_mask.all()
    dn = make_denoising_targets(torch.empty(0, 2), torch.empty(0, dtype=torch.long), torch.tensor([[.5, .5]]),
                                negative_exclusion_radius=.03)
    assert dn.objectness_targets.tolist() == [0]
    assert dn.objectness_valid_mask.tolist() == [True]
    assert dn.coordinate_valid_mask.tolist() == [False]


def test_matching_rejects_duplicate_indices_and_directional_chemical_edges():
    with pytest.raises(ValueError, match='unique'):
        matched_supervision(2, 2, torch.tensor([0, 0]), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='symmetric'):
        matched_supervision(2, 2, torch.tensor([0, 1]), torch.tensor([0, 1]),
                            gold_edges=torch.tensor([[-100, 1], [0, -100]]))


def test_coordinate_loss_masks_missing_pairs_and_normalizes_by_atoms():
    prediction = torch.tensor([[.4, .7], [.2, .3]], requires_grad=True)
    target = torch.tensor([[.2, .3], [float('nan'), float('nan')]])
    loss = masked_coordinate_l1(prediction, target, torch.tensor([True, False]))
    torch.testing.assert_close(loss, torch.tensor(.6))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[1., 1.], [0., 0.]]))
    missing = masked_coordinate_l1(prediction, target, torch.tensor([False, False]))
    assert missing == 0 and missing.requires_grad
    with pytest.raises(ValueError, match='valid_mask'):
        masked_coordinate_l1(prediction, target, torch.tensor([True]))


def test_vectorized_matching_agrees_with_direct_permutation_for_random_graphs():
    generator = torch.Generator().manual_seed(14)
    for count in [1, 2, 7, 19]:
        queries = count + 4
        query_order = torch.randperm(queries, generator=generator)[:count]
        gold_order = torch.randperm(count, generator=generator)
        matrix = torch.randint(0, 5, (count, count), generator=generator)
        matrix = torch.triu(matrix, diagonal=1)
        matrix = matrix + matrix.T
        matrix.fill_diagonal_(-100)
        result = matched_supervision(queries, count, query_order, gold_order, gold_edges=matrix)
        expected = torch.full((queries, queries), -100, dtype=torch.long)
        for q_i, g_i in zip(query_order, gold_order):
            for q_j, g_j in zip(query_order, gold_order):
                expected[q_i, q_j] = matrix[g_i, g_j]
        torch.testing.assert_close(result.edge_targets, expected)

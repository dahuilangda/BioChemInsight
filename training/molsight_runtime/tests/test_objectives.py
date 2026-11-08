import pytest
import torch
import torch.nn.functional as F
from molsight.loss import SequenceLoss, SmoothedKLDLoss
from molsight.reward import stereochem_reward_fn, tanimoto_reward_fn


@pytest.mark.parametrize("ignore", [None, -100, 0])
def test_smoothing_distribution(ignore):
    loss = SmoothedKLDLoss(0.1, 5, ignore)
    distribution = loss.uniform.clone().scatter_(
        1, torch.tensor([[2]]), loss.confidence
    )
    torch.testing.assert_close(distribution.sum(), torch.tensor(1.0))
    if ignore == 0:
        assert distribution[0, 0] == 0


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_ignored_tokens_have_no_gradient(smoothing):
    logits = torch.randn(1, 4, 5, requires_grad=True)
    target = torch.tensor([[1, 3, 2, 0]])
    original = target.clone()
    loss = SequenceLoss(smoothing, 5, ignore_indices=(0, 3))(logits, target)
    loss.backward()
    assert logits.grad[0, 0].count_nonzero() == 0
    assert logits.grad[0, 2:].count_nonzero() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert torch.equal(target, original)


def test_zero_smoothing_matches_cross_entropy():
    logits = torch.randn(4, 5, requires_grad=True)
    target = torch.tensor([2, 1, -100, 3])
    torch.testing.assert_close(
        SmoothedKLDLoss(0, 5, -100)(logits, target), F.cross_entropy(logits, target)
    )


def test_all_ignored_is_differentiable_zero():
    logits = torch.randn(2, 5, requires_grad=True)
    loss = SmoothedKLDLoss(0.1, 5, -100)(logits, torch.full((2,), -100))
    loss.backward()
    assert loss == 0
    assert logits.grad.count_nonzero() == 0


@pytest.mark.parametrize(
    "prediction,target",
    [
        ("C[*:1]", "N[*:2]"),
        ("C[*:2]", "C[*:1]"),
        ("[13CH4]", "C"),
        ("F/C=C/F", "F/C=C\\F"),
        ("N[C@@H](C)O", "N[C@H](C)O"),
        ("CC", "CC.Cl"),
        ("", "CC"),
        ("invalid", "CC"),
    ],
)
def test_exact_reward_rejects_different_structures(prediction, target):
    assert stereochem_reward_fn([prediction], [target]).item() == 0


def test_attachment_exact_match():
    torch.testing.assert_close(
        stereochem_reward_fn(["[*:1]CC"], ["CC[*:1]"]), torch.tensor([0.6])
    )


@pytest.mark.parametrize("reward", [stereochem_reward_fn, tanimoto_reward_fn])
def test_invalid_gold_fails(reward):
    with pytest.raises(ValueError):
        reward(["CC"], ["invalid"])
    with pytest.raises(ValueError):
        reward([], ["CC"])


def test_edge_loss_uses_weighted_annotation_mean():
    from molsight.loss import EdgeLoss

    logits = torch.randn(2, 2, 7, requires_grad=True)
    targets = torch.tensor([[0, 2], [-100, 3]])
    weight = torch.ones(7)
    weight[0] = 0.1
    expected = F.cross_entropy(
        logits.reshape(-1, 7), targets.reshape(-1), weight=weight
    )
    torch.testing.assert_close(EdgeLoss()(logits, targets), expected)


def test_edge_loss_missing_annotations():
    from molsight.loss import EdgeLoss

    logits = torch.randn(2, 7, requires_grad=True)
    loss = EdgeLoss()(logits, torch.full((2,), -100))
    loss.backward()
    assert loss == 0
    assert logits.grad.count_nonzero() == 0

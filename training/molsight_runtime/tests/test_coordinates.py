from types import SimpleNamespace

import pytest
import torch
from molsight.loss import CoordLoss, CoordMLELoss


def test_coordinate_density_integrates_to_one():
    objective = CoordMLELoss(SimpleNamespace(n_coord_bins=3))
    points = torch.linspace(0, 1, 10001)
    targets = torch.stack((points, torch.full_like(points, -100)), dim=-1)[None]
    probabilities = torch.tensor([0.2, 0.3, 0.5]).expand(1, points.numel(), 3)
    sigma = torch.full((1, points.numel(), 1), 0.07)
    densities = []
    for target, weights, scale in zip(
        targets.unbind(1), probabilities.unbind(1), sigma.unbind(1)
    ):
        densities.append(
            (
                -objective(
                    weights[:, None], weights[:, None], scale[:, None], target[:, None]
                )
            ).exp()
        )
    integral = torch.trapezoid(torch.stack(densities), points)
    torch.testing.assert_close(integral, torch.tensor(1.0), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("sigma", [0.001, 0.7, 100000.0])
def test_coordinate_likelihood_finite_for_extreme_scales(sigma):
    logits = torch.tensor([[[20.0, -20.0, -20.0]]], requires_grad=True)
    scale = torch.tensor([[[sigma]]], requires_grad=True)
    targets = torch.tensor([[[1.0, 0.5]]])
    loss = CoordMLELoss(SimpleNamespace(n_coord_bins=3))(
        logits.softmax(-1), logits.softmax(-1), scale, targets
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(scale.grad).all()


def test_missing_coordinate_components_have_no_gradient():
    x_logits = torch.randn(1, 2, 3, requires_grad=True)
    y_logits = torch.randn(1, 2, 3, requires_grad=True)
    scale = torch.ones(1, 2, 1, requires_grad=True)
    targets = torch.tensor([[[0.5, -100.0], [-100.0, -100.0]]])
    loss = CoordMLELoss(SimpleNamespace(n_coord_bins=3))(
        x_logits.softmax(-1), y_logits.softmax(-1), scale, targets
    )
    loss.backward()
    assert x_logits.grad[0, 0].abs().sum() > 0
    assert x_logits.grad[0, 1].count_nonzero() == 0
    assert y_logits.grad.count_nonzero() == 0
    assert scale.grad[0, 1].count_nonzero() == 0


@pytest.mark.parametrize("regression", [False, True])
def test_empty_coordinate_annotations_are_differentiable_zero(regression):
    coordinates = torch.rand(1, 2, 2, requires_grad=True)
    logits = torch.randn(1, 2, 3, requires_grad=True)
    sigma = torch.ones(1, 2, 1, requires_grad=True)
    objective = CoordLoss(
        SimpleNamespace(n_coord_bins=3, regression=regression, label_smoothing=0.1),
        aux_loss=True,
    )
    loss = objective(
        (coordinates, logits.softmax(-1), logits.softmax(-1), sigma),
        torch.full((1, 2, 2), -100.0),
    )
    loss.backward()
    assert loss == 0


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -1.0, 1.1])
def test_invalid_coordinates_fail(target):
    objective = CoordMLELoss(SimpleNamespace(n_coord_bins=3))
    with pytest.raises(ValueError, match="targets"):
        objective(
            torch.ones(1, 1, 3) / 3,
            torch.ones(1, 1, 3) / 3,
            torch.ones(1, 1, 1),
            torch.tensor([[[target, 0.5]]]),
        )

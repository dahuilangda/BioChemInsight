from types import SimpleNamespace

import pytest
import torch
from molsight.inference import predict_images
from molsight.model import EdgePredictor, LocationPredictorRegression, LoRALinear
from torch import nn


@pytest.fixture
def fixture(monkeypatch):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.edge_predictor = EdgePredictor(8)
    model.decoder.loc_predictor = LocationPredictorRegression(8)
    model.adapter = LoRALinear(nn.Linear(8, 8), r=2, alpha=2)
    model.adapter.enabled = False
    model.sample_begin = 1
    model.train()
    hidden = torch.randn(2, 5, 8, requires_grad=True)

    def sample(policy, image, features, strategy, count):
        assert not policy.training
        assert not policy.adapter.enabled
        assert features is None and count == 1
        return (
            {"smiles": ["CC", ""], "indices": [[0, 1], []], "atoms": [["C", "C"], []]},
            {
                "hidden_states": hidden,
                "image_features": torch.randn(2, 4, 8),
                "raw_tokens": torch.tensor([[1, 10, 10, 2], [1, 10, 10, 10]]),
            },
        )

    monkeypatch.setattr("molsight.inference.sample_completions", sample)
    return SimpleNamespace(model=model, images=torch.randn(2, 8, 2, 2), hidden=hidden)


def test_coordinate_only_inference_does_not_require_an_edge_head(fixture):
    fixture.model.decoder.edge_predictor = None
    result = predict_images(fixture.model, fixture.images, formats=("char", "coords"))
    expected = fixture.model.decoder.loc_predictor._forward(fixture.hidden[0, :2])[0]
    torch.testing.assert_close(torch.tensor(result["coords"][0]), expected.detach())
    assert result["coords"][1] == []
    assert result["edges"] is None
    assert fixture.model.training
    assert not fixture.model.adapter.enabled
    assert all(parameter.grad is None for parameter in fixture.model.parameters())


def test_edge_inference_preserves_empty_completion_alignment(fixture):
    result = predict_images(fixture.model, fixture.images, formats=("char", "edges"))
    assert len(result["edges"]) == 2
    assert len(result["edges"][0]) == 2
    assert len(result["edges"][1]) == 0
    assert result["coords"] is None


def test_missing_prediction_head_is_explicit_and_restores_model_state(fixture):
    fixture.model.decoder.loc_predictor = None
    with pytest.raises(ValueError, match="no coordinate predictor"):
        predict_images(fixture.model, fixture.images, formats=("coords",))
    assert fixture.model.training
    assert not fixture.model.adapter.enabled


def test_char_inference_does_not_invoke_auxiliary_heads(fixture):
    fixture.model.decoder.edge_predictor = None
    fixture.model.decoder.loc_predictor = None
    result = predict_images(fixture.model, fixture.images)
    assert result["smiles"] == ["CC", ""]
    assert result["edges"] is None and result["coords"] is None
    assert result["terminated"] == [True, False]

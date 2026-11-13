import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from molsight.loss import GRPOLoss
from molsight.model import LoRALinear, MolsightModel, SMILESDecoder
from molsight.tokenizer import CharTokenizer


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_full_policy_updates_encoder_and_decoder_without_adapters(
    tmp_path, checkpoint, temperature
):
    torch.set_num_threads(1)
    torch.manual_seed(25)
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps(
            {
                "<pad>": 0,
                "<sos>": 1,
                "<eos>": 2,
                "<mask>": 3,
                "C": 4,
                "N": 5,
                "O": 6,
                "*": 7,
            }
        )
    )
    args = SimpleNamespace(
        max_len=8,
        embed_dim=32,
        dec_n_head=4,
        dec_n_layer=2,
        use_checkpoint=checkpoint,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        lora=False,
        formats=["char"],
        n_samples=2,
        policy_temperature=temperature,
    )
    model = MolsightModel.__new__(MolsightModel)
    nn.Module.__init__(model)
    model.tokenizer = CharTokenizer(path)
    model.initial_tokens, model.sample_begin, model.logit_filters, model.max_len = (
        [1],
        1,
        [],
        8,
    )
    model.image_encoder = nn.Conv2d(32, 32, 1)
    model.get_image_features = model.image_encoder
    model.decoder = SMILESDecoder(args, 8)
    nn.init.normal_(model.decoder.token_embedding.weight, std=0.05)
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    saved = {key: value.clone() for key, value in reference.state_dict().items()}
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    criterion = GRPOLoss(args, reference_model=reference)
    image = torch.randn(2, 32, 2, 2)
    loss, metrics = criterion(
        model, image, lambda samples: torch.arange(len(samples)).float()
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.image_encoder.weight.grad.abs().sum() > 0
    assert model.decoder.token_embedding.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reference.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    assert model.training
    assert metrics["kl"] >= 0


def test_trainable_reference_is_rejected():
    with pytest.raises(ValueError, match="must be frozen"):
        GRPOLoss(SimpleNamespace(n_samples=2), reference_model=nn.Linear(2, 2))

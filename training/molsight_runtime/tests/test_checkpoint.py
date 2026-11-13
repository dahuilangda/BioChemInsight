from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from molsight.checkpoint import (
    capture_rng,
    decoder_configuration,
    load_parameters,
    merge_adapters,
    model_state,
    restore_rng,
)
from molsight.model import LoRALinear, SMILESDecoder
from torch import nn


def test_optimizer_restart_restores_stochastic_update_trajectory(tmp_path):
    def build():
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        return model, optimizer, scheduler

    def update(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(3, 4) * (random.random() + np.random.random())
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    model, optimizer, scheduler = build()
    for _ in range(3):
        update(model, optimizer, scheduler)
    path = tmp_path / "restart.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
        },
        path,
    )
    expected = [update(model, optimizer, scheduler) for _ in range(4)]
    restored, optimizer2, scheduler2 = build()
    state = torch.load(path, weights_only=False)
    restored.load_state_dict(state["model"])
    optimizer2.load_state_dict(state["optimizer"])
    scheduler2.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    actual = [update(restored, optimizer2, scheduler2) for _ in range(4)]
    assert expected == actual
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def decoder(vocabulary=8, *, adapters=False):
    args = SimpleNamespace(
        max_len=12,
        embed_dim=32,
        dec_n_head=4,
        dec_n_layer=2,
        use_checkpoint=False,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        lora=adapters,
        formats=["char"],
    )
    model = nn.Module()
    model.decoder = SMILESDecoder(args, vocabulary)
    return model


def test_configuration_comes_from_weights():
    model = decoder()
    configuration = decoder_configuration(model.state_dict())
    assert configuration == {
        "embed_dim": 32,
        "dec_n_layer": 2,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rmsnorm": True,
    }
    state = dict(model.state_dict())
    state["decoder.blocks.3.attn.query.weight"] = torch.zeros(32, 32)
    with pytest.raises(ValueError, match="contiguous"):
        decoder_configuration(state)


def test_adapter_initialization_preserves_base_outputs():
    torch.manual_seed(51)
    original, adapted = decoder(), decoder(adapters=True)
    load_parameters(adapted, original.state_dict(), initialize_adapters=True)
    tokens, features = torch.tensor([[1, 4, 5]]), torch.randn(1, 4, 32)
    torch.testing.assert_close(
        original.decoder(tokens, features)[1], adapted.decoder(tokens, features)[1]
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_parameters(decoder(adapters=True), original.state_dict())


def test_vocabulary_extension_preserves_old_rows_and_requires_id_agreement():
    original, expanded = decoder(), decoder(9)
    source = {str(index): index for index in range(8)}
    target = {**source, "new": 8}
    load_parameters(
        expanded,
        original.state_dict(),
        source_vocabulary=source,
        target_vocabulary=target,
    )
    torch.testing.assert_close(
        expanded.decoder.token_embedding.weight[:8],
        original.decoder.token_embedding.weight,
    )
    with pytest.raises(ValueError, match="IDs"):
        load_parameters(
            expanded,
            original.state_dict(),
            source_vocabulary=source,
            target_vocabulary={**target, "0": 1},
        )
    with pytest.raises(ValueError, match="both vocabulary"):
        load_parameters(decoder(9), original.state_dict())


def test_merged_adapter_is_the_fixed_supervised_reference():
    adapted, merged = decoder(adapters=True), decoder()
    for module in adapted.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B.weight, std=0.03)
    load_parameters(merged, merge_adapters(adapted.state_dict(), alpha=16))
    tokens, features = torch.tensor([[1, 4, 5]]), torch.randn(1, 4, 32)
    torch.testing.assert_close(
        adapted.decoder(tokens, features)[1], merged.decoder(tokens, features)[1]
    )


def test_ddp_key_collisions_are_rejected():
    with pytest.raises(ValueError, match="collide"):
        model_state({"model": {"module.a": torch.ones(1), "a": torch.zeros(1)}})

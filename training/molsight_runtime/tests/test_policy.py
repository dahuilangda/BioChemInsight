from itertools import product
from types import SimpleNamespace

import pytest
import torch
from molsight.decoding import GreedyDecoder
from molsight.loss import GRPOLoss
from molsight.model import LoRALinear, MolsightModel, MultiHeadAttention, SMILESDecoder
from molsight.policy import (
    completion_mask,
    grouped_advantages,
    grpo_objective,
    policy_mode,
    sample_completions,
    token_logps,
    validate_adapter_policy,
)
from molsight.tokenizer import CharTokenizer
from torch import nn


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=list(product((False, True), repeat=3)))
def model(request, tmp_path):
    import json

    torch.manual_seed(41)
    args = SimpleNamespace(
        max_len=8,
        embed_dim=32,
        dec_n_head=4,
        dec_n_layer=2,
        use_checkpoint=request.param[0],
        use_qknorm=request.param[1],
        use_swiglu=False,
        use_rmsnorm=False,
        lora=True,
        formats=["sequence"],
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
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
    instance = MolsightModel.__new__(MolsightModel)
    nn.Module.__init__(instance)
    instance.tokenizer = CharTokenizer(vocab)
    instance.initial_tokens = [1]
    instance.sample_begin = 1
    instance.logit_filters = []
    instance.max_len = args.max_len
    instance.image_encoder = nn.Identity()
    instance.get_image_features = instance.image_encoder
    instance.decoder = SMILESDecoder(args, 8)
    for name, parameter in instance.named_parameters():
        parameter.requires_grad = "lora_A" in name or "lora_B" in name
        if "lora_B" in name:
            nn.init.normal_(parameter, std=0.03)
    previous_sdpa = MultiHeadAttention.use_sdpa
    MultiHeadAttention.use_sdpa = request.param[2]
    yield instance
    MultiHeadAttention.use_sdpa = previous_sdpa


def test_completion_boundaries():
    tokens = torch.tensor([[4, 2, 2, 0], [4, 0, 5, 2], [4, 5, 6, 4], [0, 0, 0, 0]])
    assert completion_mask(tokens).tolist() == [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
        [False, False, False, False],
    ]


def test_group_statistics():
    advantages, std = grouped_advantages(torch.tensor([1.0, 1.0, 0.0, 2.0]), 2)
    torch.testing.assert_close(advantages[:2], torch.zeros(2))
    torch.testing.assert_close(
        advantages[2:], torch.tensor([-1.0, 1.0]), atol=2e-4, rtol=0
    )
    assert torch.isfinite(std)
    with pytest.raises(ValueError):
        grouped_advantages(torch.tensor([1.0]), 1)


def test_clipped_objective_uses_old_policy():
    logps = torch.tensor([[0.4], [-0.4]], requires_grad=True)
    old = torch.zeros_like(logps, requires_grad=True)
    reference = torch.zeros_like(logps, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, _, clipped = grpo_objective(
        logps,
        old,
        reference,
        advantages,
        torch.ones_like(logps, dtype=torch.bool),
        0,
        0.2,
    )
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    loss.backward()
    assert logps.grad.count_nonzero() == 0
    assert old.grad is None and reference.grad is None and advantages.grad is None
    assert clipped == 1


def test_masked_tokens_have_no_gradient():
    logps = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    mask = torch.tensor([[True, False, False]])
    loss, _, _ = grpo_objective(
        logps, logps.detach(), logps.detach(), torch.tensor([1.0]), mask, 0.04, 0.2
    )
    loss.backward()
    torch.testing.assert_close(logps.grad, torch.tensor([[-1.0, 0.0, 0.0]]))


def test_policy_state_restored_after_exception(model):
    model.train()
    model.decoder.blocks[0].eval()
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    adapters[0].enabled = False
    training = [module.training for module in model.modules()]
    states = [module.enabled for module in adapters]
    with pytest.raises(RuntimeError), policy_mode(model):
        assert not model.training
        assert all(module.enabled for module in adapters)
        with policy_mode(model, adapters=False):
            assert not any(module.enabled for module in adapters)
        assert all(module.enabled for module in adapters)
        raise RuntimeError("interrupted")
    assert [module.training for module in model.modules()] == training
    assert [module.enabled for module in adapters] == states


def test_cached_rollout_probabilities_match_teacher_forcing(model):
    image = torch.randn(2, 32, 1, 2)
    with policy_mode(model), torch.no_grad():
        _, intermediate = sample_completions(
            model, image, image, GreedyDecoder(1.0, 2), 2
        )
        ids = intermediate["raw_tokens"]
        output = model(
            image=None, label=ids, image_features=image.repeat_interleave(2, dim=0)
        )
        expected = token_logps(output["logits"][:, :-1], ids[:, 1:])
    torch.testing.assert_close(
        intermediate["token_logps"], expected, atol=5e-5, rtol=5e-5
    )
    assert all(not module._forward_hooks for module in model.modules())


def test_sampling_exception_removes_hooks(model, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model, "generate", fail)
    image = torch.randn(2, 32, 1, 2)
    with pytest.raises(RuntimeError):
        sample_completions(model, image, image, GreedyDecoder(1.0, 2), 2)
    assert all(not module._forward_hooks for module in model.modules())


def test_grpo_backward_preserves_reference(model):
    image = torch.randn(2, 32, 1, 2)
    ids = torch.tensor([[1, 4, 5, 2], [1, 6, 4, 2]])
    with policy_mode(model, adapters=False), torch.no_grad():
        reference_before = model(image=image, label=ids)["logits"].clone()
    criterion = GRPOLoss(SimpleNamespace(n_samples=2))
    loss, metrics = criterion(
        model, image, lambda _: torch.tensor([0.0, 1.0, 1.0, 0.0])
    )
    assert model.training
    assert all(
        module.enabled for module in model.modules() if isinstance(module, LoRALinear)
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    optimizer.step()
    with policy_mode(model, adapters=False), torch.no_grad():
        reference_after = model(image=image, label=ids)["logits"]
    torch.testing.assert_close(reference_before, reference_after, rtol=0, atol=0)
    assert 0 <= metrics["clip_fraction"] <= 1


def test_reference_rejects_trainable_base(model):
    model.decoder.token_embedding.weight.requires_grad = True
    with pytest.raises(ValueError, match="frozen base"):
        validate_adapter_policy(model)


def test_checkpoint_backward_uses_forward_adapter_state(model):
    import copy

    direct = copy.deepcopy(model)
    direct.decoder.use_checkpoint = False
    model.decoder.use_checkpoint = True
    for instance in (model, direct):
        for module in instance.modules():
            if isinstance(module, LoRALinear):
                module.lora_dropout.p = 0.5
    image = torch.randn(2, 32, 1, 2)
    ids = torch.tensor([[1, 4, 5, 2], [1, 6, 4, 2]])
    for instance in (model, direct):
        with policy_mode(instance):
            output = instance(image=image, label=ids)["logits"]
            loss = output.square().mean()
        for module in instance.modules():
            if isinstance(module, LoRALinear):
                module.enabled = False
        loss.backward()
    for actual, expected in zip(model.parameters(), direct.parameters()):
        if expected.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-5)
    assert all(
        not module.enabled
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )


def test_scst_supports_multiple_images_and_samples(model):
    from molsight.loss import SCSTLoss

    criterion = SCSTLoss(SimpleNamespace(n_samples=3))
    image = torch.randn(2, 32, 1, 2)
    counts = []

    def reward(predictions):
        counts.append(len(predictions))
        return torch.arange(len(predictions), dtype=torch.float32) / max(
            len(predictions), 1
        )

    loss, metrics = criterion(model, image, reward)
    loss.backward()
    assert counts == [2, 6]
    assert torch.isfinite(loss)
    assert metrics["avg_baseline"] == 0.25
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_grpo_forward_backward(model, amp):
    model = model.cuda()
    image = torch.randn(2, 32, 1, 2, device="cuda")
    criterion = GRPOLoss(SimpleNamespace(n_samples=2))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        loss, metrics = criterion(
            model, image, lambda _: torch.tensor([0.0, 1.0, 1.0, 0.0], device="cuda")
        )
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert 0 <= metrics["clip_fraction"] <= 1
    assert all(not m._forward_hooks for m in model.modules())


def test_cpu_autocast_keeps_policy_gradients(model):
    image = torch.randn(2, 32, 1, 2)
    criterion = GRPOLoss(SimpleNamespace(n_samples=2))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = criterion(model, image, lambda _: torch.tensor([0.0, 1.0, 1.0, 0.0]))
    loss.backward()
    assert loss.requires_grad
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .model import LoRALinear
from .tokenizer import EOS_ID, PAD_ID


@contextmanager
def frozen_forward(device):
    with (
        torch.no_grad(),
        torch.autocast(
            device.type,
            enabled=torch.is_autocast_enabled(device.type),
            dtype=torch.get_autocast_dtype(device.type),
            cache_enabled=False,
        ),
    ):
        yield


@contextmanager
def policy_mode(model, adapters=True):
    modules = list(model.modules())
    training = [module.training for module in modules]
    adapter_states = [
        (module, module.enabled) for module in modules if isinstance(module, LoRALinear)
    ]
    try:
        model.eval()
        for module, _ in adapter_states:
            module.enabled = adapters
        yield
    finally:
        for module, enabled in adapter_states:
            module.enabled = enabled
        for module, state in zip(modules, training):
            module.training = state


def validate_adapter_policy(model):
    adapter_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for branch in (module.lora_A, module.lora_B)
        for parameter in branch.parameters()
    }
    if not adapter_parameters:
        raise ValueError("Adapter reference requires LoRA modules")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and id(parameter) not in adapter_parameters:
            raise ValueError(
                f"Adapter reference requires frozen base parameters: {name}"
            )
    if any(parameter.requires_grad for parameter in model.image_encoder.parameters()):
        raise ValueError("Cached image features require a frozen image encoder")
    if model.logit_filters:
        raise ValueError(
            "Policy probability calculation does not support logit filters"
        )


def completion_mask(tokens):
    eos = tokens.eq(EOS_ID)
    before_eos = eos.cumsum(dim=1) - eos.long() == 0
    before_pad = tokens.eq(PAD_ID).cumsum(dim=1) == 0
    return before_eos & before_pad


def token_logps(logits, tokens):
    return (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, tokens.unsqueeze(-1))
        .squeeze(-1)
    )


def grouped_advantages(rewards, group_size):
    if group_size < 2 or rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("GRPO requires complete groups with at least two samples")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    groups = rewards.detach().float().reshape(-1, group_size)
    std = groups.std(dim=1, correction=0, keepdim=True)
    advantages = (groups - groups.mean(dim=1, keepdim=True)) / (std + 1e-4)
    return advantages.reshape(-1), std.mean()


def grpo_objective(
    logps, old_logps, reference_logps, advantages, mask, beta, clip_epsilon
):
    if (
        logps.shape != old_logps.shape
        or logps.shape != reference_logps.shape
        or logps.shape != mask.shape
    ):
        raise ValueError(
            "Policy probabilities and completion masks must have identical shapes"
        )
    if advantages.shape != (logps.shape[0],):
        raise ValueError("Each completion requires one advantage")
    if beta < 0 or not 0 < clip_epsilon < 1:
        raise ValueError("Invalid KL coefficient or clipping range")
    logps = logps.float()
    old_logps, reference_logps = (
        old_logps.detach().float(),
        reference_logps.detach().float(),
    )
    ratio = torch.exp(logps - old_logps)
    advantage = advantages.detach().float().unsqueeze(1)
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    )
    delta = reference_logps - logps
    kl = (torch.expm1(delta) - delta).clamp_min(0)
    values = -surrogate + beta * kl
    if not torch.isfinite(values[mask]).all():
        raise FloatingPointError("Non-finite policy objective")
    lengths = mask.sum(dim=1).clamp_min(1)
    loss = (values.masked_fill(~mask, 0).sum(dim=1) / lengths).mean()
    mean_kl = (kl.masked_fill(~mask, 0).sum(dim=1) / lengths).mean()
    clip_fraction = (
        ((ratio - 1).abs() > clip_epsilon) & mask
    ).sum() / mask.sum().clamp_min(1)
    return loss, mean_kl, clip_fraction


def sample_completions(model, image, image_features, strategy, n_samples):
    cache, hooks = model.install_kv_cache_hooks()
    try:
        return model.generate(
            image=image,
            image_features=image_features,
            kv_cache=cache,
            decode_strategy=strategy,
            n_samples=n_samples,
        )
    finally:
        for hook in hooks:
            hook.remove()

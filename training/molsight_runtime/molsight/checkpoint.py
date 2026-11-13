"""Strict checkpoint loading and explicit vocabulary and adapter migrations."""

import re


def capture_rng(device=None):
    import random
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device is not None and torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device=None):
    import random
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def model_state(checkpoint):
    state = checkpoint["model"]
    normalized = {key.removeprefix("module."): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("Checkpoint keys collide after removing the DDP prefix")
    return normalized


def decoder_configuration(state):
    embedding = state["decoder.token_embedding.weight"]
    layers = sorted(
        {
            int(match[1])
            for key in state
            if (match := re.match(r"decoder\.blocks\.(\d+)\.", key))
        }
    )
    if not layers or layers != list(range(len(layers))):
        raise ValueError("Decoder checkpoint layers are not contiguous")
    return {
        "embed_dim": embedding.shape[1],
        "dec_n_layer": len(layers),
        "use_qknorm": "decoder.positional_embedding.weight" not in state,
        "use_swiglu": "decoder.blocks.0.mlp.w12.weight" in state,
        "use_rmsnorm": "decoder.blocks.0.self_attn_norm.bias" not in state,
    }


def model_configuration(state):
    configuration = decoder_configuration(state)
    if "image_pos_embed" in state:
        embedding = state["image_pos_embed"]
        if embedding.ndim != 4 or embedding.shape[:2] != (
            1,
            configuration["embed_dim"],
        ):
            raise ValueError("Invalid checkpoint image position dimensions")
        configuration["image_pos_grid_size"] = tuple(embedding.shape[-2:])
    return configuration


def load_parameters(
    model,
    state,
    *,
    source_vocabulary=None,
    target_vocabulary=None,
    initialize_adapters=False,
):
    from .model import LoRALinear

    destinations = model.state_dict()
    if (
        source_vocabulary is not None
        and target_vocabulary is not None
        and any(
            target_vocabulary.get(token) != index
            for token, index in source_vocabulary.items()
        )
    ):
        raise ValueError("Vocabulary migration must preserve all existing token IDs")
    aliases = {}
    adapter_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for suffix in ("weight", "bias"):
                aliases[f"{name}.{suffix}"] = f"{name}.base.{suffix}"
            adapter_keys.update((f"{name}.lora_A.weight", f"{name}.lora_B.weight"))
    migrated = {aliases.get(key, key): value for key, value in state.items()}
    if len(migrated) != len(state):
        raise ValueError("Checkpoint keys collide during adapter migration")
    embedding_key = "decoder.token_embedding.weight"
    if embedding_key in migrated and embedding_key in destinations:
        old, new = migrated[embedding_key], destinations[embedding_key]
        if old.shape != new.shape:
            if source_vocabulary is None or target_vocabulary is None:
                raise ValueError(
                    "Vocabulary expansion requires both vocabulary mappings"
                )
            if (
                old.shape[1:] != new.shape[1:]
                or old.shape[0] != len(source_vocabulary)
                or new.shape[0] != len(target_vocabulary)
                or new.shape[0] < old.shape[0]
                or any(
                    target_vocabulary.get(token) != index
                    for token, index in source_vocabulary.items()
                )
            ):
                raise ValueError(
                    "Vocabulary expansion must preserve all existing token IDs"
                )
            expanded = new.clone()
            expanded[: old.shape[0]].copy_(old)
            expanded[old.shape[0] :].copy_(old.mean(dim=0))
            migrated[embedding_key] = expanded
    missing = set(destinations) - set(migrated)
    if initialize_adapters and not (set(state) & adapter_keys):
        for key in missing & adapter_keys:
            migrated[key] = destinations[key]
    if set(migrated) != set(destinations):
        raise ValueError(
            f"Checkpoint key mismatch: missing={sorted(set(destinations) - set(migrated))}, "
            f"unexpected={sorted(set(migrated) - set(destinations))}"
        )
    model.load_state_dict(migrated, strict=True)


def merge_adapters(state, *, alpha):
    merged = dict(state)
    for key in state:
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key.removesuffix(".lora_A.weight")
        first = state[key].float()
        second = state[prefix + ".lora_B.weight"].float()
        base = state[prefix + ".base.weight"]
        merged[prefix + ".weight"] = (
            base.float() + second @ first * (alpha / first.shape[0])
        ).to(base.dtype)
        for suffix in (".lora_A.weight", ".lora_B.weight", ".base.weight"):
            del merged[prefix + suffix]
        bias = prefix + ".base.bias"
        if bias in merged:
            merged[prefix + ".bias"] = merged.pop(bias)
    return merged

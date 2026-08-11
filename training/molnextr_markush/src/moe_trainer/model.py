"""Encoder/decoder model wrapping and fine-tuning configuration.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA when WORLD_SIZE > 1")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


class EncoderMoETrainingModel(torch.nn.Module):
    """Joint DDP boundary for the visual encoder and direct graph experts."""

    def __init__(self, encoder: torch.nn.Module, moe: MoEDecoder):
        super().__init__()
        self.encoder = encoder
        self.moe = moe

    def forward(
        self,
        images: torch.Tensor,
        refs: dict,
        structure_labels: torch.Tensor | None = None,
    ) -> dict:
        features, encoder_hiddens = self.encoder(images)
        outputs = self.moe(
            features,
            refs,
            structure_labels=structure_labels,
            encoder_hiddens=encoder_hiddens,
        )
        outputs["encoder_features"] = features
        outputs["encoder_hiddens"] = encoder_hiddens
        return outputs


def configure_encoder_finetuning(
    encoder: torch.nn.Module,
    stages: int,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Unfreeze only the final Swin stages and final normalization."""

    requested = max(0, int(stages))
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if requested == 0:
        encoder.eval()
        return [], []
    transformer = getattr(encoder, "transformer", None)
    layers = list(getattr(transformer, "layers", ()))
    if not layers:
        raise ValueError("encoder stage fine-tuning requires a Swin-style encoder")
    if requested > len(layers):
        raise ValueError(
            f"encoder-finetune-stages={requested} exceeds available stages={len(layers)}"
        )
    selected_modules = layers[-requested:]
    final_norm = getattr(transformer, "norm", None)
    if final_norm is not None:
        selected_modules.append(final_norm)
    for module in selected_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = [
        parameter for parameter in encoder.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("encoder fine-tuning selected zero parameters")
    return trainable_parameters, trainable_names


_FRAGMENT_OUTPUT_EDGE_PREFIXES = (
    "decoder.chartok_coords.output_layer.",
    "decoder.edges.",
)


def decoder_scope_parameter_prefixes(
    decoder: torch.nn.Module,
    scope: str,
) -> tuple[str, ...]:
    normalized = str(scope).strip().lower()
    if normalized == "output_edge":
        return _FRAGMENT_OUTPUT_EDGE_PREFIXES
    if normalized != "last_cross_output_edge":
        raise ValueError(f"decoder scope {scope!r} does not use prefix selection")

    layer_root = "decoder.chartok_coords.decoder.transformer_layers."
    layer_indices = set()
    for name, _parameter in decoder.named_parameters():
        if not name.startswith(layer_root):
            continue
        index_text = name[len(layer_root):].split(".", 1)[0]
        if index_text.isdigit():
            layer_indices.add(int(index_text))
    if not layer_indices:
        raise RuntimeError(
            "last_cross_output_edge requires indexed MolNexTR transformer layers"
        )
    last_layer = max(layer_indices)
    last_layer_root = f"{layer_root}{last_layer}."
    return _FRAGMENT_OUTPUT_EDGE_PREFIXES + (
        last_layer_root + "context_attn.",
        last_layer_root + "layer_norm_2.",
    )


def decoder_trainability_manifest(
    decoder: torch.nn.Module,
    scope: str,
) -> dict:
    named_parameters = list(decoder.named_parameters())
    trainable_names = [
        name for name, parameter in named_parameters if parameter.requires_grad
    ]
    frozen_names = [
        name for name, parameter in named_parameters if not parameter.requires_grad
    ]
    return {
        "scope": str(scope),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_tensors": len(trainable_names),
        "frozen_parameter_tensors": len(frozen_names),
        "trainable_parameters": int(sum(
            parameter.numel()
            for _name, parameter in named_parameters
            if parameter.requires_grad
        )),
        "frozen_parameters": int(sum(
            parameter.numel()
            for _name, parameter in named_parameters
            if not parameter.requires_grad
        )),
    }


def configure_decoder_finetuning(
    decoder: torch.nn.Module,
    scope: str,
) -> tuple[list[torch.nn.Parameter], dict]:
    """Apply an explicit, auditable trainable boundary to one decoder expert."""

    normalized = str(scope).strip().lower()
    if normalized not in {"full", "output_edge", "last_cross_output_edge"}:
        raise ValueError(f"unsupported decoder train scope: {scope!r}")
    named_parameters = list(decoder.named_parameters())
    if not named_parameters:
        raise RuntimeError("decoder train scope selected a module with zero parameters")

    for _name, parameter in named_parameters:
        parameter.requires_grad_(normalized == "full")
    if normalized != "full":
        selected_prefixes = decoder_scope_parameter_prefixes(decoder, normalized)
        matched_prefixes = {prefix: False for prefix in selected_prefixes}
        for name, parameter in named_parameters:
            for prefix in selected_prefixes:
                if name.startswith(prefix):
                    parameter.requires_grad_(True)
                    matched_prefixes[prefix] = True
                    break
        missing = [prefix for prefix, matched in matched_prefixes.items() if not matched]
        if missing:
            raise RuntimeError(
                f"{normalized} decoder scope does not match the MolNexTR decoder "
                "structure; missing parameter prefixes: " + ", ".join(missing)
            )

    manifest = decoder_trainability_manifest(decoder, normalized)
    trainable_parameters = [
        parameter for _name, parameter in decoder.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(f"decoder train scope {normalized!r} selected zero parameters")
    return trainable_parameters, manifest


def validate_optimizer_parameter_coverage(
    moe: torch.nn.Module,
    encoder: torch.nn.Module,
    parameter_groups: list[dict],
) -> dict:
    """Require the optimizer to contain every and only trainable parameter once."""

    named_parameters = [
        (f"moe.{name}", parameter) for name, parameter in moe.named_parameters()
    ] + [
        (f"encoder.{name}", parameter)
        for name, parameter in encoder.named_parameters()
    ]
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    expected_ids = {
        id(parameter) for _name, parameter in named_parameters
        if parameter.requires_grad
    }
    actual_parameters = [
        parameter
        for group in parameter_groups
        for parameter in group.get("params", [])
    ]
    actual_ids = [id(parameter) for parameter in actual_parameters]
    duplicated_ids = sorted({
        parameter_id for parameter_id in actual_ids
        if actual_ids.count(parameter_id) > 1
    })
    if duplicated_ids:
        raise RuntimeError(
            "optimizer contains duplicate parameters: "
            + ", ".join(names_by_id.get(parameter_id, "<unknown>") for parameter_id in duplicated_ids[:5])
        )
    actual_id_set = set(actual_ids)
    unknown = sorted(actual_id_set - set(names_by_id))
    missing = sorted(expected_ids - actual_id_set)
    frozen = sorted(actual_id_set - expected_ids)
    if unknown or missing or frozen:
        raise RuntimeError(
            "optimizer/trainability contract mismatch: "
            f"unknown={len(unknown)} "
            f"missing={[names_by_id[item] for item in missing[:5]]} "
            f"frozen={[names_by_id[item] for item in frozen[:5] if item in names_by_id]}"
        )

    group_manifests = []
    for group in parameter_groups:
        group_parameters = list(group.get("params", []))
        group_manifests.append({
            "name": str(group.get("name", "unnamed")),
            "lr": float(group["lr"]),
            "parameter_names": [names_by_id[id(parameter)] for parameter in group_parameters],
            "parameter_tensors": len(group_parameters),
            "parameters": int(sum(parameter.numel() for parameter in group_parameters)),
        })
    return {
        "schema_version": "molnextr_trainable_parameter_manifest_v1",
        "parameter_names": sorted(names_by_id[item] for item in expected_ids),
        "parameter_tensors": len(expected_ids),
        "parameters": int(sum(parameter.numel() for parameter in actual_parameters)),
        "groups": group_manifests,
    }



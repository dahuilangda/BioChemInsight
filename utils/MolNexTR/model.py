"""model for MolNexTR"""

import argparse
import gc
import ctypes
import os
from pathlib import Path
from typing import List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rdkit import Chem

from .dataset import get_transforms
from .components import Encoder, Decoder
from .chemical import convert_graph_to_smiles
from .tokenization import get_tokenizer


def _markush_config_enabled(config):
    if not isinstance(config, dict):
        return False
    return bool((config.get("structure_conditioning") or {}).get("enabled")) or bool(
        (config.get("attachment_count_head") or {}).get("enabled")
    )

def _pad_chartok_for_sep(module, states):
    """Pad the chartok_coords output_layer + embedding +1 row for the <sep> tail
    token (Phase 2) when loading a pre-<sep> checkpoint (vocab 229 -> 230).
    Zero-init the <sep> row; byte-identical-complete is then enforced by the
    allow_sep=False decode flag (base/expert0 never emit <sep>). Only pads on an
    exact 1-row dim-0 shortfall, so checkpoints already sized for <sep> are unchanged."""
    import torch
    out_layer_keys = (
        "chartok_coords.output_layer.weight",
        "chartok_coords.output_layer.bias",
    )
    for name, param in module.named_parameters():
        is_out = any(name == k or name.endswith("." + k) for k in out_layer_keys)
        is_emb = (
            "chartok_coords" in name
            and "emb_luts" in name
            and name.endswith(".weight")
        )
        if not (is_out or is_emb):
            continue
        st = states.get(name)
        if st is None or st.dim() < 1:
            continue
        if st.shape[0] == param.shape[0] - 1:
            # Init the <sep> row as a COPY of the EOS row (id 2) so <sep> starts
            # with a high logit at the structure terminator and can compete with the
            # warm-started EOS during phase2_sep training (zero-init learned far too
            # slowly, ~0.006/epoch). For complete rows <sep> is masked at decode
            # (allow_sep=False) so this init never affects complete decode.
            pad = (st[2:3]).clone()
            states[name] = torch.cat([st, pad], dim=0)
    return states


def loading(module, module_states):
    """
    Loads the model's state_dict into a module, handling potential prefix mismatches.

    Args:
        module (torch.nn.Module): The module (model) to load the state_dict into.
        module_states (dict): The state dictionary to load.
    """
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    states = remove_prefix(module_states)
    states = _pad_chartok_for_sep(module, states)
    missing_keys, unexpected_keys = module.load_state_dict(states, strict=False)
    return

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]
DEFAULT_PREPROCESS_LONG_EDGE = 512
DEFAULT_MAX_INFERENCE_BATCH_SIZE = 1
_LIBC = None


def _molblock_backbone_key(molblock):
    if not molblock:
        return ""
    try:
        mol = Chem.MolFromMolBlock(
            str(molblock), sanitize=False, removeHs=False
        )
    except Exception:
        mol = None
    if mol is None:
        return ""
    editable = Chem.RWMol(mol)
    for atom_index in sorted(
        [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0],
        reverse=True,
    ):
        editable.RemoveAtom(atom_index)
    backbone = editable.GetMol()
    if backbone.GetNumAtoms() == 0:
        return ""
    try:
        Chem.SanitizeMol(backbone)
        return Chem.MolToSmiles(backbone, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _molblock_has_only_bonded_dummies(molblock):
    try:
        mol = Chem.MolFromMolBlock(
            str(molblock or ""), sanitize=False, removeHs=False
        )
    except Exception:
        mol = None
    if mol is None:
        return False
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    return bool(dummies and all(atom.GetDegree() > 0 for atom in dummies))


def _molblock_dummy_count(molblock):
    try:
        mol = Chem.MolFromMolBlock(
            str(molblock or ""), sanitize=False, removeHs=False
        )
    except Exception:
        mol = None
    if mol is None:
        return 0
    return int(sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()))


def _rollback_attachment_transaction(prediction, reason):
    base = prediction.get("_attachment_set_base_graph")
    if not isinstance(base, dict):
        return
    atom_data = prediction.get("chartok_coords")
    if not isinstance(atom_data, dict):
        return
    atom_data["symbols"] = list(base.get("symbols") or [])
    atom_data["coords"] = [list(point) for point in (base.get("coords") or [])]
    if "atom_scores" in atom_data or base.get("atom_scores"):
        atom_data["atom_scores"] = list(base.get("atom_scores") or [])
    prediction["edges"] = [list(row) for row in (base.get("edges") or [])]
    prediction["edge_scores"] = dict(base.get("edge_scores") or {})
    prediction["decode_atom_count"] = len(atom_data["symbols"])
    base_issue = base.get("decode_quality_issue")
    if base_issue:
        prediction["decode_quality_issue"] = base_issue
    else:
        prediction.pop("decode_quality_issue", None)
    accepted = list(prediction.get("attachment_set_predictions") or [])
    preserved = [item for item in accepted if item.get("action") == "preserve_existing"]
    rolled_back = [
        {**item, "reason": reason, "rolled_back_action": item.get("action")}
        for item in accepted
        if item.get("action") != "preserve_existing"
    ]
    prediction["attachment_set_predictions"] = preserved
    prediction["attachment_set_rejections"] = [
        *(prediction.get("attachment_set_rejections") or []),
        *rolled_back,
    ]
    transaction = dict(prediction.get("attachment_set_transaction") or {})
    transaction.update(
        {
            "status": "rolled_back",
            "reason": str(reason),
            "result_atom_count": len(atom_data["symbols"]),
        }
    )
    prediction["attachment_set_transaction"] = transaction


def _trim_process_memory():
    global _LIBC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name != "posix":
        return
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def resize_small_image_to_long_edge(image, target_long_edge=DEFAULT_PREPROCESS_LONG_EDGE):
    """Upscale small crops before MolNexTR's fixed 384px transform, preserving aspect ratio."""
    if image is None:
        return image
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= 0 or long_edge >= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)


class molnextr:
    """
    Main Interface for MolNexTR to get predictions
    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to run the model on, defaults to CPU if None.
    """
    def __init__(
        self,
        model_path,
        device=None,
        postprocess_workers=1,
        preprocess_long_edge=DEFAULT_PREPROCESS_LONG_EDGE,
        max_inference_batch_size=DEFAULT_MAX_INFERENCE_BATCH_SIZE,
        markush_config_path=None,
        moe_config_path=None,
    ):
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        args = self._get_args(model_states['args'])
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.postprocess_workers = max(1, int(postprocess_workers or 1))
        self.preprocess_long_edge = max(0, int(preprocess_long_edge or 0))
        self.max_inference_batch_size = max(1, int(max_inference_batch_size or 1))
        self.markush_config_path = str(markush_config_path) if markush_config_path else None
        self.markush_config = None
        self.markush_train_model = None
        self.moe_config_path = str(moe_config_path) if moe_config_path else None
        self.moe_config = None
        self.moe_decoder = None
        self._args = args
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        self._maybe_load_moe_inference(model_states)
        self.transform = get_transforms(args.input_size, args.input_size, augment=False)

    def _maybe_load_moe_inference(self, model_states):
        """Build the LoRA-adapter MoE decoder if a ``moe_config_path`` is given.

        The config JSON (``expert_kind == "lora"``) may specify:
          * num_experts (int, default 3), expert_names (list[str])
          * adapter_path (str|None): LoRA ``{A,B}`` checkpoint
            (``{"decoder": lora_state_dict}``); the frozen base weights come
            from the already-loaded ``self.decoder``.
          * router_path (str|None): learned gate checkpoint.
          * lora_rank / lora_alpha / include_output_layer / include_edges.
          * sidecar_confidence_threshold(s): minimum gate prob before a
            non-complete expert is activated (else the gate is zeroed ⇒ output
            matches the frozen base).
          * routing_strategy ("soft_mixture"|"sparse_top1").
          * use_confidence_head, valence_repair_enabled.

        Loading order matters (see method body): base weights into the
        LoRAMoELinear buffers first, then the adapter A/B, then the router.

        With no config this is a no-op and inference stays on the plain
        ``self.decoder`` (byte-identical to the frozen single-expert model).
        """
        if not self.moe_config_path:
            return
        from .moe import MoEDecoder, load_moe_config

        config = load_moe_config(self.moe_config_path)
        expert_kind = str(config.get("expert_kind", "lora"))
        if expert_kind not in ("lora", "full_mixture"):
            raise NotImplementedError(
                f"MoE config expert_kind={expert_kind!r} is not supported. Use "
                "'lora' (frozen base + LoRA adapter experts) or 'full_mixture' "
                "(frozen complete decoder + full-decoder specialist, logit mixture)."
            )
        num_experts = int(config.get("num_experts", 3))
        sep_extension_enabled = bool(
            config.get("sep_extension_enabled", False)
        )
        if sep_extension_enabled:
            representation = config.get("attachment_representation") or {}
            contract_errors = []
            if config.get("schema_version") != "molnextr_moe_native_attachment_v5":
                contract_errors.append("schema_version")
            if representation.get("schema_version") != "molnextr_single_attachment_extension_v1":
                contract_errors.append("attachment_representation")
            if representation.get("strict_failure") is not True:
                contract_errors.append("strict_failure")
            if representation.get("fallback") is not False:
                contract_errors.append("fallback")
            if expert_kind != "full_mixture" or num_experts < 3:
                contract_errors.append("fragment_expert")
            if config.get("attachment_set_decode_mode") != "direct_sidecar":
                contract_errors.append("direct_sidecar")
            if bool(config.get("route_fragment_to_base", False)):
                contract_errors.append("route_fragment_to_base")
            if bool(config.get("sidecar_broken_decode_fallback", False)):
                contract_errors.append("sidecar_broken_decode_fallback")
            if bool(config.get("per_bucket_decode_dispatch", False)):
                contract_errors.append("per_bucket_decode_dispatch")
            if bool(config.get("valence_repair_enabled", True)):
                contract_errors.append("valence_repair_enabled")
            if contract_errors:
                raise ValueError(
                    "invalid native attachment v5 inference contract: "
                    + ", ".join(contract_errors)
                )

        # The direct graph model may jointly fine-tune the final visual Swin
        # stage. Load that sparse delta before routing or decoding so production
        # uses the same image representation that was optimized during training.
        encoder_path = config.get("encoder_path")
        if encoder_path:
            encoder_checkpoint = torch.load(encoder_path, map_location="cpu")
            encoder_states = encoder_checkpoint.get(
                "encoder",
                encoder_checkpoint,
            )
            expected_names = set(
                encoder_checkpoint.get("trainable_parameter_names") or []
            )
            if expected_names and expected_names != set(encoder_states):
                raise ValueError(
                    "MolNexTR encoder checkpoint parameter manifest mismatch"
                )
            incompatible = self.encoder.load_state_dict(
                encoder_states,
                strict=False,
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "MolNexTR encoder checkpoint has unexpected parameters: "
                    + ", ".join(incompatible.unexpected_keys[:5])
                )

        moe_decoder = MoEDecoder(
            self._args,
            self.tokenizer,
            num_experts=num_experts,
            expert_names=config.get("expert_names"),
            expert_kind=expert_kind,
            lora_rank=int(config.get("lora_rank", 8)),
            lora_alpha=float(config.get("lora_alpha", 16)),
            include_output_layer=bool(config.get("include_output_layer", True)),
            include_edges=bool(config.get("include_edges", False)),
            sidecar_confidence_threshold=float(config.get("sidecar_confidence_threshold", 1.01)),
            sidecar_confidence_thresholds=config.get("sidecar_confidence_thresholds"),
            routing_strategy=config.get("routing_strategy", "soft_mixture"),
            router_kind=config.get("router_kind", "mean_mlp"),
            use_confidence_head=bool(config.get("use_confidence_head")),
            valence_repair_enabled=bool(config.get("valence_repair_enabled", True)),
            expected_fragment_star_logit_bias=float(config.get("expected_fragment_star_logit_bias", 0.0) or 0.0),
            expected_fragment_star_budget=int(config.get("expected_fragment_star_budget", 1) or 1),
            expected_fragment_max_atoms=int(config.get("expected_fragment_max_atoms", 30) or 0),
            mixture_complete_floor=float(config.get("mixture_complete_floor", 0.65)),
            full_mixture_sidecar_mode=config.get("full_mixture_sidecar_mode", "collapsed"),
            token_fusion_mode=config.get("token_fusion_mode", "fixed"),
            token_fusion_initial_sidecar_weight=float(
                config.get("token_fusion_initial_sidecar_weight", 0.2) or 0.2),
            token_fusion_dispatch=config.get("token_fusion_dispatch", "soft"),
            token_fusion_hard_threshold=float(
                config.get("token_fusion_hard_threshold", 0.5)
            ),
            specialist_ownership_scope=config.get(
                "specialist_ownership_scope", "full"
            ),
            decouple_fusion_policy_optimization=bool(
                config.get("decouple_fusion_policy_optimization", True)
            ),
            one_sided_fusion_oracle=bool(
                config.get("one_sided_fusion_oracle", True)
            ),
            attachment_set_enabled=bool(
                config.get("attachment_set_enabled", False)
            ),
            attachment_set_hidden_dim=int(
                config.get("attachment_set_hidden_dim", 256)
            ),
            attachment_set_num_queries=int(
                config.get("attachment_set_num_queries", 32)
            ),
            attachment_set_num_layers=int(
                config.get("attachment_set_num_layers", 3)
            ),
            attachment_set_num_heads=int(
                config.get("attachment_set_num_heads", 8)
            ),
            attachment_set_max_count=int(
                config.get("attachment_set_max_count", 30)
            ),
            attachment_set_min_confidence=float(
                config.get("attachment_set_min_confidence", 0.5)
            ),
            attachment_set_max_anchor_distance=float(
                config.get("attachment_set_max_anchor_distance", 0.25)
            ),
            attachment_set_decode_mode=config.get(
                "attachment_set_decode_mode", "token_mixture"
            ),
            attachment_set_feature_mode=config.get(
                "attachment_set_feature_mode", "single_scale"
            ),
            attachment_set_feature_levels=int(
                config.get("attachment_set_feature_levels", 3)
            ),
            attachment_set_max_feature_size=int(
                config.get("attachment_set_max_feature_size", 32)
            ),
            attachment_set_min_pointer_confidence=float(
                config.get("attachment_set_min_pointer_confidence", 0.15)
            ),
            attachment_set_heatmap_logit_scale=float(
                config.get("attachment_set_heatmap_logit_scale", 4.0)
            ),
            attachment_set_heatmap_prior_precision=float(
                config.get("attachment_set_heatmap_prior_precision", 8.0)
            ),
            sidecar_coordinate_context=config.get(
                "sidecar_coordinate_context", "full"
            ),
            fragment_terminal_action_head_enabled=bool(
                config.get("fragment_terminal_action_head_enabled", False)
            ),
            fragment_terminal_action_head_hidden_dim=int(
                config.get("fragment_terminal_action_head_hidden_dim", 128)
            ),
            fragment_structured_terminal_edge_enabled=bool(
                config.get("fragment_structured_terminal_edge_enabled", False)
            ),
        )
        # (1) expert0 = frozen base decoder. For 'lora' this loads base weights
        #     into the LoRAMoELinear buffers (W0/b0); for 'full_mixture' it loads
        #     the plain Decoder (load_base_weights_into_lora is a no-op without
        #     LoRA modules).
        moe_decoder.load_expert0_from_base(model_states["decoder"])
        # (1b) If expert0 was UNFROZEN (trained on fragments with anchor), load
        # the TRAINED expert0 checkpoint (overwrites the base weights). This is
        # required because load_expert0_from_base above loaded the ORIGINAL base.
        if config.get("frozen_expert0") is False:
            expert_paths = config.get("expert_paths") or []
            expert0_path = config.get("expert0_path") or (
                expert_paths[0] if len(expert_paths) > 0 else None)
            if expert0_path:
                e0_states = torch.load(expert0_path, map_location="cpu")
                moe_decoder.load_expert(0, e0_states.get("decoder", e0_states))
        # (2) trained expert artifact.
        if expert_kind == "lora":
            adapter_path = config.get("adapter_path")
            if adapter_path:
                adapter_states = torch.load(adapter_path, map_location="cpu")
                moe_decoder.load_adapter(adapter_states["decoder"])
        else:  # full_mixture: load trained sidecar decoder(s).
            expert_paths = config.get("expert_paths") or []
            expert1_path = config.get("expert1_path") or (
                expert_paths[1] if len(expert_paths) > 1 else None)
            if expert1_path:
                e1_states = torch.load(expert1_path, map_location="cpu")
                moe_decoder.load_expert(1, e1_states.get("decoder", e1_states))
            expert2_path = config.get("expert2_path") or (
                expert_paths[2] if len(expert_paths) > 2 else None)
            if expert2_path and len(moe_decoder.experts) > 2:
                e2_states = torch.load(expert2_path, map_location="cpu")
                moe_decoder.load_expert(2, e2_states.get("decoder", e2_states))
        # (3) router.
        router_path = config.get("router_path")
        if router_path:
            router_states = torch.load(router_path, map_location="cpu")
            moe_decoder.load_router(
                router_states,
                require_fusion_state=(moe_decoder.token_fusion_mode == "adaptive"),
                require_attachment_set_state=bool(
                    moe_decoder.attachment_set_enabled
                ),
                require_terminal_action_state=bool(
                    moe_decoder.fragment_terminal_action_head_enabled
                ),
            )
        # (4) optional confidence head.
        confidence_path = config.get("confidence_path")
        if confidence_path and moe_decoder.confidence_head is not None:
            conf_states = torch.load(confidence_path, map_location="cpu")
            head_states = conf_states.get("confidence_head", conf_states)
            moe_decoder.confidence_head.load_state_dict(head_states, strict=False)

        moe_decoder.to(self.device).eval()
        self.moe_decoder = moe_decoder
        self.moe_config = config
        # Phase 2 interim deploy safety: route fragments to frozen base (expert2
        # regresses real fragments until the <sep> retrain). Toggle via config.
        self.moe_decoder.route_fragment_to_base = bool(
            config.get("route_fragment_to_base", False)
        )
        # Per-bucket decode dispatch: markush (routed_idx 1) → direct_sidecar
        # (trained [n*] sidecar decoder); fragment (routed_idx 2) → residual_base
        # (decoupled attachment head on the frozen base). Each bucket uses its
        # correct from-root decoder instead of one global decode_mode. Default off
        # ⇒ byte-identical to pre-flag behavior. Toggle via config.
        self.moe_decoder.per_bucket_decode_dispatch = bool(
            config.get("per_bucket_decode_dispatch", False)
        )
        # Native sidecar output is the production default. Broken sidecar
        # replacement is an explicit legacy diagnostic only.
        self.moe_decoder.sidecar_broken_decode_fallback = bool(
            config.get("sidecar_broken_decode_fallback", False)
        )
        # Phase 2: enable <sep> emission for the trainable fragment sidecar (expert2).
        self.moe_decoder.sep_extension_enabled = sep_extension_enabled
        # Cardinality dummy-prune graph edit. Default off.
        prune_env = os.environ.get("MOLNEXTR_DUMMY_PRUNE_ENABLED", "").strip().lower()
        if prune_env in ("1", "true", "yes", "on"):
            self.moe_decoder.attachment_set_dummy_prune_enabled = True
        elif prune_env in ("0", "false", "no", "off"):
            self.moe_decoder.attachment_set_dummy_prune_enabled = False
        else:
            self.moe_decoder.attachment_set_dummy_prune_enabled = bool(
                config.get("attachment_set_dummy_prune_enabled", False)
            )

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model: allow encoder architecture override via env var (e.g.
        # MOLNEXTR_ENCODER_VARIANT=swin_large) for stronger visual features.
        _default_encoder = os.environ.get("MOLNEXTR_ENCODER_VARIANT", "swin_base").strip() or "swin_base"
        parser.add_argument('--encoder', type=str, default=_default_encoder)
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
        group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
        group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
        group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
        parser.add_argument('--continuous_coords', action='store_true')
        parser.add_argument('--compute_confidence', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value
        if args.vocab_file and not Path(args.vocab_file).exists():
            vocab_name = Path(args.vocab_file).name
            local_vocab = Path(__file__).resolve().parent / "vocab" / vocab_name
            if local_vocab.exists():
                args.vocab_file = str(local_vocab)
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        # When using a non-default encoder variant (e.g. swin_large via
        # MOLNEXTR_ENCODER_VARIANT), the encoder weights in molnextr_best.pth
        # (swin_base, dim=1024) won't match (dim=1536). Skip encoder loading
        # and let ImageNet pretrain + from-scratch training handle it.
        _skip_encoder = os.environ.get("MOLNEXTR_ENCODER_VARIANT", "").strip()
        if _skip_encoder and _skip_encoder != "swin_base":
            import warnings
            warnings.warn(
                f"encoder={_skip_encoder}: skipping base checkpoint encoder "
                f"weights (dim mismatch), using ImageNet pretrained"
            )
        else:
            loading(encoder, states['encoder'])
        # When encoder variant differs, decoder's enc_transform (Linear(1024→256))
        # mismatches the new encoder_dim (1536). Filter out incompatible keys.
        if _skip_encoder and _skip_encoder != "swin_base":
            dec_states = {k: v for k, v in states['decoder'].items()
                          if 'enc_trans_layer' not in k}
            missing, unexpected = decoder.load_state_dict(dec_states, strict=False)
        else:
            loading(decoder, states['decoder'])

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(
        self,
        input_images: List,
        return_atoms_bonds=False,
        return_confidence=False,
        batch_size=16,
        beam_size: int = 1,
        n_best: int = 1,
        expected_structure_types=None,
        attachment_priors=None,
    ):
        device = self.device
        outputs = []
        self.decoder.compute_confidence = return_confidence
        if self.moe_decoder is not None:
            self.moe_decoder.compute_confidence = return_confidence
            for _expert in self.moe_decoder.experts:
                _expert.compute_confidence = return_confidence
        internal_batch_size = min(max(1, int(batch_size or 1)), self.max_inference_batch_size)
        input_images = [
            resize_small_image_to_long_edge(image, self.preprocess_long_edge)
            for image in input_images
        ]
        expected_structure_types = list(expected_structure_types or [None] * len(input_images))
        if len(expected_structure_types) != len(input_images):
            raise ValueError("expected_structure_types must match input_images length")
        # Detection→graph fusion: per-sample detector attachment points (list of
        # {cx, cy, confidence, class} dicts) consumed inside the attachment_set
        # residual edit. None/short list ⇒ inert (byte-identical to base decode).
        attachment_priors = list(attachment_priors or [None] * len(input_images))
        if len(attachment_priors) < len(input_images):
            attachment_priors = attachment_priors + [None] * (len(input_images) - len(attachment_priors))

        for idx in range(0, len(input_images), internal_batch_size):
            batch_images = input_images[idx:idx+internal_batch_size]
            batch_expected = expected_structure_types[idx:idx+internal_batch_size]
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            images = torch.stack(images, dim=0).to(device)
            with torch.inference_mode():
                # Expose this batch's detector priors to the MoE decoder's
                # attachment_set residual edit (read by sample index b). Cleared
                # after decode so complete-only batches pay no overhead and no
                # state leaks across calls.
                if self.moe_decoder is not None:
                    self.moe_decoder._attachment_priors_batch = [
                        attachment_priors[idx + j] for j in range(len(batch_images))
                    ]
                features, hiddens = self.encoder(images)
                if self.moe_decoder is not None:
                    batch_predictions = self.moe_decoder.decode(
                        features,
                        hiddens,
                        beam_size=beam_size,
                        n_best=n_best,
                        expected_structure_types=batch_expected,
                    )
                elif self.markush_train_model is not None:
                    batch_predictions = self.markush_train_model.decode(
                        features,
                        hiddens,
                        condition_on_structure=bool(
                            (self.markush_config or {}).get("structure_conditioning", {}).get("enabled")
                        ),
                    )
                else:
                    batch_predictions = self.decoder.decode(features, hiddens)
                # Always clear the per-batch detector priors so no state leaks
                # into subsequent calls (complete-only batches, non-MoE paths).
                if self.moe_decoder is not None:
                    self.moe_decoder._attachment_priors_batch = None
            valid_items = [
                (item_index, pred)
                for item_index, pred in enumerate(batch_predictions)
                if not pred.get("decode_quality_issue")
            ]
            smiles_by_index = {item_index: "" for item_index in range(len(batch_predictions))}
            molblock_by_index = {item_index: "" for item_index in range(len(batch_predictions))}
            quality_issues_by_index = {item_index: [] for item_index in range(len(batch_predictions))}
            transaction_base_results = {}
            transaction_items = [
                (item_index, pred)
                for item_index, pred in valid_items
                if (pred.get("attachment_set_transaction") or {}).get("status")
                == "committed"
                and isinstance(pred.get("_attachment_set_base_graph"), dict)
            ]
            if transaction_items:
                base_nodes = [
                    pred["_attachment_set_base_graph"]["coords"]
                    for _, pred in transaction_items
                ]
                base_symbols = [
                    pred["_attachment_set_base_graph"]["symbols"]
                    for _, pred in transaction_items
                ]
                base_edges = [
                    pred["_attachment_set_base_graph"]["edges"]
                    for _, pred in transaction_items
                ]
                base_images = [
                    batch_images[item_index] for item_index, _ in transaction_items
                ]
                base_smiles, base_molblocks, _base_success, base_issues = (
                    convert_graph_to_smiles(
                        base_nodes,
                        base_symbols,
                        base_edges,
                        images=base_images,
                        num_workers=self.postprocess_workers,
                    )
                )
                for (item_index, pred), smiles, molblock, issue in zip(
                    transaction_items,
                    base_smiles,
                    base_molblocks,
                    base_issues,
                ):
                    transaction_base_results[item_index] = {
                        "smiles": smiles,
                        "molblock": molblock,
                        "issue": (
                            pred["_attachment_set_base_graph"].get(
                                "decode_quality_issue"
                            )
                            or issue
                            or ""
                        ),
                    }
            if valid_items:
                node_coords = [pred['chartok_coords']['coords'] for _, pred in valid_items]
                node_symbols = [pred['chartok_coords']['symbols'] for _, pred in valid_items]
                edges = [pred['edges'] for _, pred in valid_items]
                valid_images = [batch_images[item_index] for item_index, _ in valid_items]
                smiles_list, molblock_list, r_success, graph_quality_issues = convert_graph_to_smiles(
                    node_coords,
                    node_symbols,
                    edges,
                    images=valid_images,
                    num_workers=self.postprocess_workers,
                )
                for (item_index, _), smiles, molfile, issue in zip(
                    valid_items, smiles_list, molblock_list, graph_quality_issues
                ):
                    pred = batch_predictions[item_index]
                    base_result = transaction_base_results.get(item_index)
                    if base_result is not None:
                        transaction_reason = ""
                        try:
                            parsed_smiles = Chem.MolFromSmiles(smiles) if smiles else None
                        except Exception:
                            parsed_smiles = None
                        modified_ok = bool(
                            parsed_smiles is not None and molfile and not issue
                        )
                        base_ok = bool(
                            base_result["smiles"]
                            and base_result["molblock"]
                            and not base_result["issue"]
                        )
                        edit_actions = {
                            str(item.get("action") or "")
                            for item in pred.get("attachment_set_predictions") or []
                        }
                        append_items = [
                            item
                            for item in pred.get("attachment_set_predictions") or []
                            if item.get("action") == "append"
                        ]
                        if not modified_ok:
                            transaction_reason = "residual_graph_conversion_failed"
                        else:
                            target_cardinality = int(
                                (pred.get("attachment_set_transaction") or {}).get(
                                    "target_cardinality", 0
                                )
                            )
                            base_dummy_count = _molblock_dummy_count(
                                base_result["molblock"]
                            )
                            modified_dummy_count = _molblock_dummy_count(molfile)
                            if modified_dummy_count > max(
                                target_cardinality, base_dummy_count
                            ):
                                transaction_reason = "residual_cardinality_exceeded"
                            elif (
                                "append" in edit_actions
                                and target_cardinality > 0
                                and base_dummy_count >= target_cardinality
                            ):
                                transaction_reason = (
                                    "residual_existing_attachment_satisfied_cardinality"
                                )
                            elif (
                                append_items
                                and base_dummy_count > 0
                                and any(
                                    item.get("anchor_resolution")
                                    != "decoder_atom_pointer"
                                    for item in append_items
                                )
                            ):
                                transaction_reason = (
                                    "residual_continuous_anchor_cannot_extend_existing_set"
                                )
                        if not transaction_reason and "append" in edit_actions and not _molblock_has_only_bonded_dummies(
                            molfile
                        ):
                            transaction_reason = "residual_unbonded_dummy"
                        elif not transaction_reason and base_ok:
                            base_key = _molblock_backbone_key(base_result["molblock"])
                            modified_key = _molblock_backbone_key(molfile)
                            if not base_key or modified_key != base_key:
                                transaction_reason = "residual_backbone_changed"
                        if transaction_reason:
                            _rollback_attachment_transaction(pred, transaction_reason)
                            smiles = base_result["smiles"]
                            molfile = base_result["molblock"]
                            issue = base_result["issue"]
                        else:
                            transaction = dict(
                                pred.get("attachment_set_transaction") or {}
                            )
                            transaction["status"] = "validated"
                            transaction["backbone_invariant"] = bool(base_ok)
                            pred["attachment_set_transaction"] = transaction
                    molblock_by_index[item_index] = molfile
                    if issue:
                        quality_issues_by_index[item_index].append(issue)
                    else:
                        smiles_by_index[item_index] = smiles
            for item_index, pred in enumerate(batch_predictions):
                smiles = smiles_by_index[item_index]
                molfile = molblock_by_index[item_index]
                pred_dict = {"predicted_smiles": smiles, "predicted_molfile": molfile}
                if quality_issues_by_index[item_index]:
                    pred_dict["quality_issues"] = quality_issues_by_index[item_index]
                for _meta_key in (
                    "expert_weights",
                    "routed_expert",
                    "routing_forced_complete",
                    "routing_forced_default",
                    "routing_strategy",
                    "routing_confidence",
                    "routing_required_threshold",
                    "confidence",
                    "attachment_repair",
                    "token_fusion_mode",
                    "token_fusion_dispatch",
                    "token_fusion_applied",
                    "token_fusion_sidecar_mean",
                    "attachment_set_predictions",
                    "attachment_set_rejections",
                    "attachment_set_selected_count",
                    "attachment_set_cardinality",
                    "attachment_set_decode_mode",
                    "attachment_set_transaction",
                    "attachment_graph_consistency",
                    "attachment_extension",
                ):
                    if _meta_key in pred:
                        pred_dict[_meta_key] = pred[_meta_key]
                if pred.get("decode_quality_issue"):
                    pred_dict["quality_issues"] = [pred["decode_quality_issue"]]
                    pred_dict["decode_atom_count"] = pred.get("decode_atom_count")
                    outputs.append(pred_dict)
                    continue
                if return_atoms_bonds:
                    coords = pred['chartok_coords']['coords']
                    symbols = pred['chartok_coords']['symbols']
                    # get atoms info
                    atom_list = []
                    for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                        atom_dict = {"atom_number":f"{i}", "atom_symbol": symbol, "coords":(round(coord[0],3),round(coord[1],3))}
                        if return_confidence:
                            atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                        atom_list.append(atom_dict)
                    pred_dict["atom_sets"] = atom_list
                    # get bonds info
                    bond_list = []
                    num_atoms = len(symbols)
                    for i in range(num_atoms-1):
                        for j in range(i+1, num_atoms):
                            bond_type_int = pred['edges'][i][j]
                            if bond_type_int != 0:
                                bond_type_str = BOND_TYPES[bond_type_int]
                                bond_dict = {"atom_number":f"{i}","bond_type": bond_type_str, "endpoints": (i, j)}
                                if return_confidence:
                                    bond_dict["confidence"] = pred["edge_scores"].get((i, j))
                                bond_list.append(bond_dict)
                    pred_dict["bond_sets"] = bond_list
                outputs.append(pred_dict)
            del batch_images, images, features, hiddens
            if self.moe_decoder is not None:
                self.moe_decoder._clear_caches()
            else:
                for decoder in self.decoder.decoder.values():
                    if hasattr(decoder, "decoder") and hasattr(decoder.decoder, "state"):
                        decoder.decoder.state["cache"] = None
                        decoder.decoder.state["src"] = None
            del batch_predictions, smiles_by_index, molblock_by_index, quality_issues_by_index
            _trim_process_memory()
        del input_images
        _trim_process_memory()
        return outputs

    def predict_image(self, image, return_atoms_bonds=False, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single input image.
        
        Args:
            image (ndarray): Input image.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image.
        """
        return self.predict_images([
            image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def predict_image_files(
        self,
        image_files: List,
        return_atoms_bonds=False,
        return_confidence=False,
        batch_size=16,
        beam_size: int = 1,
        n_best: int = 1,
        expected_structure_types=None,
        attachment_priors=None,
    ):
        """
        Predicts SMILES and molecular structure from a list of image file paths.

        Args:
            image_files (List): List of image file paths.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.
            attachment_priors (list[list[dict]] | None): Per-file detector
                attachment points ({cx, cy, confidence, class}). When provided,
                these are fused into the attachment_set residual edit as
                full-schema proposals. None ⇒ inert (no-op decode).

        Returns:
            List: List of prediction results for each image file.
        """
        outputs = []
        expected_structure_types = list(expected_structure_types or [None] * len(image_files))
        if len(expected_structure_types) != len(image_files):
            raise ValueError("expected_structure_types must match image_files length")
        attachment_priors = list(attachment_priors or [None] * len(image_files))
        requested_batch_size = max(1, int(batch_size or 1))
        file_batch_size = min(requested_batch_size, self.max_inference_batch_size)
        for idx in range(0, len(image_files), file_batch_size):
            input_images = []
            for path in image_files[idx:idx + file_batch_size]:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_images.append(image)
            outputs.extend(
                self.predict_images(
                    input_images,
                    return_atoms_bonds=return_atoms_bonds,
                    return_confidence=return_confidence,
                    batch_size=file_batch_size,
                    beam_size=max(1, int(beam_size or 1)),
                    n_best=max(1, int(n_best or 1)),
                    expected_structure_types=expected_structure_types[idx:idx + file_batch_size],
                    attachment_priors=attachment_priors[idx:idx + file_batch_size],
                )
            )
            del input_images
            _trim_process_memory()
        return outputs

    def predict_final_results(self, image_file: str, return_atoms_bonds=False, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single image file path.
        
        Args:
            image_file (str): Path to the input image file.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image file.
        """
        return self.predict_image_files(
            [image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

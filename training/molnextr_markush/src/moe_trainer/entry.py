"""MoE training entry point — the main() training loop.

Extracted from the original ``tools/train_moe.py`` monolith.  This module
contains only the top-level ``main()`` orchestration; all library code lives
in the sibling modules.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403
from .args import *  # noqa: F401,F403
from .model import *  # noqa: F401,F403
from .runtime import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403
from .calibration import *  # noqa: F401,F403

def main():
    args = parse_args()
    validate_direct_sidecar_training_contract(args)
    if float(args.fragment_grpo_weight) < 0.0:
        raise ValueError("--fragment-grpo-weight must be non-negative")
    if float(args.fragment_grpo_reference_kl_weight) < 0.0:
        raise ValueError("--fragment-grpo-reference-kl-weight must be non-negative")
    if float(args.fragment_counterfactual_dpo_weight) < 0.0:
        raise ValueError("--fragment-counterfactual-dpo-weight must be non-negative")
    if float(args.fragment_counterfactual_dpo_beta) <= 0.0:
        raise ValueError("--fragment-counterfactual-dpo-beta must be positive")
    if float(args.fragment_counterfactual_edge_weight) < 0.0:
        raise ValueError("--fragment-counterfactual-edge-weight must be non-negative")
    if float(args.fragment_counterfactual_reward_margin) < 0.0:
        raise ValueError("--fragment-counterfactual-reward-margin must be non-negative")
    if float(args.fragment_search_policy_weight) < 0.0:
        raise ValueError("--fragment-search-policy-weight must be non-negative")
    if float(args.fragment_search_edge_weight) < 0.0:
        raise ValueError("--fragment-search-edge-weight must be non-negative")
    if float(args.fragment_search_policy_margin) < 0.0:
        raise ValueError("--fragment-search-policy-margin must be non-negative")
    if int(args.fragment_terminal_action_head_hidden_dim) < 32:
        raise ValueError(
            "--fragment-terminal-action-head-hidden-dim must be at least 32"
        )
    if float(args.fragment_terminal_action_head_loss_weight) < 0.0:
        raise ValueError(
            "--fragment-terminal-action-head-loss-weight must be non-negative"
        )
    if float(args.fragment_terminal_action_head_lr) <= 0.0:
        raise ValueError("--fragment-terminal-action-head-lr must be positive")
    if float(args.fragment_terminal_action_head_grad_clip) <= 0.0:
        raise ValueError(
            "--fragment-terminal-action-head-grad-clip must be positive"
        )
    if (
        float(args.fragment_terminal_action_head_loss_weight) > 0.0
        and not args.fragment_terminal_action_head_enabled
    ):
        raise ValueError(
            "terminal action head loss requires "
            "--fragment-terminal-action-head-enabled"
        )
    if (
        args.fragment_terminal_action_head_enabled
        and args.attachment_set_decode_mode != "direct_sidecar"
    ):
        raise ValueError(
            "fragment terminal action head requires direct_sidecar deployment"
        )
    if (
        args.fragment_structured_terminal_edge_enabled
        and args.attachment_set_decode_mode != "direct_sidecar"
    ):
        raise ValueError(
            "fragment structured terminal edge requires direct_sidecar deployment"
        )
    if int(args.fragment_grpo_every) < 1:
        raise ValueError("--fragment-grpo-every must be positive")
    if int(args.fragment_grpo_max_rows) < 1:
        raise ValueError("--fragment-grpo-max-rows must be positive")
    if int(args.fragment_grpo_group_size) < 2:
        raise ValueError("--fragment-grpo-group-size must be at least two")
    if float(args.fragment_grpo_temperature) <= 0.0:
        raise ValueError("--fragment-grpo-temperature must be positive")
    if not 0.0 < float(args.fragment_grpo_top_p) <= 1.0:
        raise ValueError("--fragment-grpo-top-p must be in (0, 1]")
    if (
        (
            float(args.fragment_grpo_weight) > 0.0
            or float(args.fragment_counterfactual_dpo_weight) > 0.0
            or float(args.fragment_counterfactual_edge_weight) > 0.0
            or float(args.fragment_search_policy_weight) > 0.0
            or float(args.fragment_search_edge_weight) > 0.0
        )
        and args.attachment_set_decode_mode != "direct_sidecar"
    ):
        raise ValueError("fragment on-policy optimization requires direct_sidecar deployment")
    if args.allow_attachment_set_pointer_upgrade and not args.resume_router:
        raise ValueError(
            "--allow-attachment-set-pointer-upgrade requires --resume-router"
        )
    if (
        args.allow_attachment_set_pointer_upgrade
        and args.attachment_set_feature_mode != "multiscale_pointer"
    ):
        raise ValueError(
            "--allow-attachment-set-pointer-upgrade requires "
            "--attachment-set-feature-mode multiscale_pointer"
        )
    if args.allow_attachment_set_heatmap_upgrade and not args.resume_router:
        raise ValueError(
            "--allow-attachment-set-heatmap-upgrade requires --resume-router"
        )
    if (
        args.allow_attachment_set_heatmap_upgrade
        and args.attachment_set_feature_mode != "multiscale_pointer_heatmap"
    ):
        raise ValueError(
            "--allow-attachment-set-heatmap-upgrade requires "
            "--attachment-set-feature-mode multiscale_pointer_heatmap"
        )
    if (
        args.allow_attachment_set_pointer_upgrade
        and args.allow_attachment_set_heatmap_upgrade
    ):
        raise ValueError("attachment-set architecture upgrade flags are mutually exclusive")
    if not args.build_data_only and args.unfreeze_expert0 and args.expert_kind != "full_mixture":
        raise ValueError("--unfreeze-expert0 is supported only with --expert-kind full_mixture")
    if (
        not args.build_data_only
        and args.token_fusion_mode == "adaptive"
        and args.expert_kind != "full_mixture"
    ):
        raise ValueError("adaptive token fusion requires --expert-kind full_mixture")
    if not 0.0 < float(args.token_fusion_hard_threshold) < 1.0:
        raise ValueError("--token-fusion-hard-threshold must be strictly between 0 and 1")
    if not 0 <= int(args.encoder_finetune_stages) <= 4:
        raise ValueError("--encoder-finetune-stages must be between 0 and 4")
    if args.resume_encoder and int(args.encoder_finetune_stages) == 0:
        raise ValueError("--resume-encoder requires --encoder-finetune-stages > 0")
    if (
        not args.build_data_only
        and args.fragment_decoder_train_scope != "full"
        and (
            args.expert_kind != "full_mixture"
            or args.full_mixture_sidecar_mode != "per_sidecar"
        )
    ):
        raise ValueError(
            f"--fragment-decoder-train-scope {args.fragment_decoder_train_scope} requires "
            "--expert-kind full_mixture and --full-mixture-sidecar-mode per_sidecar"
        )
    distributed, rank, local_rank, world_size = init_distributed()
    warm_start_provenance = None
    if is_rank0(rank):
        warm_start_provenance = {
            "expert1": checkpoint_provenance(args.resume_expert1),
            "expert2": checkpoint_provenance(args.resume_expert2),
            "encoder": checkpoint_provenance(args.resume_encoder),
            "router": checkpoint_provenance(args.resume_router),
            "adapter": checkpoint_provenance(args.resume_adapter),
            "attachment_set_pointer_upgrade": bool(
                args.allow_attachment_set_pointer_upgrade
            ),
            "attachment_set_heatmap_upgrade": bool(
                args.allow_attachment_set_heatmap_upgrade
            ),
        }
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    # First CUDA init in a fresh process can transiently fail; retry before CPU fallback.
    if torch.cuda.is_available():
        for _ in range(5):
            try:
                torch.cuda.init(); break
            except Exception:
                torch.cuda.sleep(1) if hasattr(torch.cuda, "sleep") else None
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank0_print(rank, f"device={device} distributed={distributed} world_size={world_size}")
    os.makedirs(args.output_dir, exist_ok=True)
    writer = make_summary_writer(args, rank)

    rank0_print(rank, "loading encoder + tokenizer from", args.base_checkpoint)
    base = molnextr(args.base_checkpoint, device=device)
    encoder_params, encoder_trainable_names = configure_encoder_finetuning(
        base.encoder,
        args.encoder_finetune_stages,
    )
    if args.resume_encoder:
        encoder_checkpoint = torch.load(args.resume_encoder, map_location="cpu")
        encoder_states = encoder_checkpoint.get("encoder", encoder_checkpoint)
        expected_names = set(encoder_trainable_names)
        if set(encoder_states) != expected_names:
            missing = sorted(expected_names - set(encoder_states))
            unexpected = sorted(set(encoder_states) - expected_names)
            raise ValueError(
                "resume encoder does not match selected fine-tune stages: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        incompatible = base.encoder.load_state_dict(encoder_states, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "resume encoder has unexpected parameters: "
                + ", ".join(incompatible.unexpected_keys[:5])
            )
    rank0_print(
        rank,
        f"  encoder fine-tune stages={int(args.encoder_finetune_stages)} "
        f"params={sum(parameter.numel() for parameter in encoder_params):,} "
        f"lr={float(args.encoder_lr):.2e}",
    )
    tokenizer = base.tokenizer
    model_args = set_train_args(base._args, args.output_dir)
    # Phase 2: base._args (checkpoint namespace) lacks the CLI flag — copy it so the
    # dataset picks it up via getattr(args, 'phase2_sep', False).
    model_args.phase2_sep = bool(getattr(args, "phase2_sep", False))

    rank0_print(rank, "building training dataframe...")
    sources = None
    cal_sources = None
    source_report = None
    real_original_report = None
    source_report_path = os.path.join(args.output_dir, "moe_source_report.json")
    source_root = str(args.production_source_root)
    if is_rank0(rank):
        if args.source_mode != "available_production_pose_factory":
            raise ValueError(f"unsupported source mode: {args.source_mode}")
        if args.frozen_source_report:
            if not args.reuse_df_cache:
                raise ValueError(
                    "--frozen-source-report requires --reuse-df-cache"
                )
            sources, cal_sources, source_report = load_frozen_source_partition(
                args.frozen_source_report,
                production_source_root=source_root,
            )
            rank0_print(
                rank,
                "  source discovery -> revalidated frozen whole-shard partition "
                f"{args.frozen_source_report}",
            )
        else:
            sources, cal_sources, source_report = discover_available_production_pose_factory(
                source_root,
                min_age_seconds=args.source_min_age_seconds,
                calibration_fraction=args.source_calibration_fraction,
                seed=args.seed,
            )
        with open(source_report_path, "w", encoding="utf-8") as handle:
            json.dump(source_report, handle, indent=2)
        rank0_print(rank, f"  source discovery -> {source_report_path}")
        for name, info in source_report.get("labels", {}).items():
            rank0_print(
                rank,
                f"  {name}: stable_csvs={info.get('stable_csvs')} "
                f"train_csvs={info.get('train_csvs')} calibration_csvs={info.get('calibration_csvs')}",
            )
    if args.reuse_df_cache and args.df_cache and os.path.exists(args.df_cache):
        df = pd.read_parquet(args.df_cache)
        real_frame, real_original_report = load_real_original_training_frame(
            args, rank
        )
        if real_frame is not None and len(real_frame):
            if "image_domain" in df.columns:
                df = df[
                    df["image_domain"].fillna("").astype(str).ne("real_original")
                ].copy()
            real_paths = set(real_frame["file_path"].astype(str))
            if set(df["file_path"].astype(str)) & real_paths:
                raise ValueError("cached pose rows overlap refreshed real-original rows")
            df = pd.concat([df, real_frame], ignore_index=True, sort=False)
            if is_rank0(rank):
                df.to_parquet(args.df_cache, index=False)
                rank0_print(
                    rank,
                    f"  refreshed {len(real_frame)} real-original rows in {args.df_cache}",
                )
        validate_dataframe_contract(
            df,
            context="reused training dataframe",
            native_attachment_extension=bool(args.phase2_sep),
            skip_fragment_linearization=bool(os.environ.get("MOE_SKIP_FRAGMENT_LIN_CHECK", "")),
        )
        rank0_print(rank, f"  loaded {len(df)} rows <- {args.df_cache}")
    elif distributed and not is_rank0(rank):
        if not args.df_cache:
            raise ValueError("DDP dataframe handoff requires --df-cache")
        dist.barrier()
        df = pd.read_parquet(args.df_cache)
        validate_dataframe_contract(
            df,
            context="DDP training dataframe",
            native_attachment_extension=bool(args.phase2_sep),
        )
    else:
        if sources is None:
            raise RuntimeError("rank0 source discovery did not produce training sources")
        df = build_moe_df(
            sources,
            args.per_label,
            out_path=None,
            tokenizer=tokenizer["chartok_coords"],
        )
        real_frame, real_original_report = load_real_original_training_frame(args, rank)
        if real_frame is not None and len(real_frame):
            duplicate_paths = set(df["file_path"].astype(str)) & set(
                real_frame["file_path"].astype(str)
            )
            if duplicate_paths:
                raise ValueError(
                    f"pose and real-original datasets overlap by {len(duplicate_paths)} files"
                )
            df = pd.concat([df, real_frame], ignore_index=True, sort=False)
        validate_dataframe_contract(
            df,
            context="rebuilt training dataframe",
            native_attachment_extension=bool(args.phase2_sep),
        )
        if args.df_cache and is_rank0(rank):
            Path(args.df_cache).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.df_cache, index=False)
            rank0_print(rank, f"  wrote {len(df)} rows -> {args.df_cache}")
        if distributed:
            dist.barrier()
    rank0_print(rank, f"  total rows: {len(df)}")
    if len(df) == 0 or "structure_type_label" not in df.columns:
        raise SystemExit("ERROR: 0 training rows built — check that the aggregate CSV paths "
                         "resolve and the atom-count alignment filter isn't dropping everything.")
    rank0_print(rank, f"  label counts: {df['structure_type_label'].value_counts().to_dict()}")
    dataframe_provenance = (
        checkpoint_provenance(args.df_cache)
        if is_rank0(rank) and args.df_cache and Path(args.df_cache).is_file()
        else None
    )
    if dataframe_provenance is not None:
        dataframe_provenance["rows"] = int(len(df))
        dataframe_provenance["label_counts"] = {
            str(int(label)): int(count)
            for label, count in df["structure_type_label"].value_counts().items()
        }
        rank0_print(
            rank,
            "  dataframe sha256: " + dataframe_provenance["sha256"],
        )
    if bool(args.attachment_set_enabled):
        attachment_capacity = validate_attachment_set_capacity(
            df,
            num_queries=args.attachment_set_num_queries,
            max_count=args.attachment_set_max_count,
        )
        rank0_print(rank, f"  attachment-set capacity: {attachment_capacity}")
    label_by_idx = df["structure_type_label"].to_numpy()
    if "coordinate_targets_available" in df.columns:
        coordinate_targets_by_idx = (
            df["coordinate_targets_available"].fillna(False).astype(bool).to_numpy()
        )
    else:
        coordinate_targets_by_idx = np.ones(len(df), dtype=bool)
    cal_df = None
    if distributed and not is_rank0(rank):
        cal_df = None
    else:
        if cal_sources is None:
            _train_sources, cal_sources, _report = discover_available_production_pose_factory(
                source_root,
                min_age_seconds=args.source_min_age_seconds,
                calibration_fraction=args.source_calibration_fraction,
                seed=args.seed,
            )
        rank0_print(rank, "building calibration dataframe...")
        cal_df = build_moe_df(
            cal_sources,
            args.calibration_max_per_label,
            out_path=None,
            tokenizer=tokenizer["chartok_coords"],
        )
        rank0_print(rank, f"  calibration rows: {len(cal_df)}")
    if args.build_data_only:
        rank0_print(rank, "build-data-only complete; exiting before training")
        if writer is not None:
            writer.close()
        if distributed:
            dist.destroy_process_group()
        return

    dataset = TrainDataset(model_args, df, tokenizer, split="train")
    sampling_report = {"strategy": "natural", "samples_per_epoch": int(len(dataset))}
    if args.sampling_strategy == "label_balanced":
        frag_weights = None
        if (
            args.fragment_oversample_tiny
            or args.fragment_oversample_wavy
            or "image_domain" in df.columns
        ):
            # Domain/size-aware oversampling for fragments: wavy attachment
            # evidence and rare tiny fragments would otherwise be under-sampled.
            smi_len = df["SMILES"].astype(str).str.len().to_numpy()
            frag_mask = label_by_idx == 2
            frag_smi_len = smi_len[frag_mask]
            # SMILES length is a robust heavy-atom proxy for fragments (*CCO=4,
            # *c1ccccc1=9, medium~14). Bucket: tiny/small/rest.
            w = np.ones_like(smi_len, dtype=np.float64)
            fpos = np.flatnonzero(frag_mask)
            if args.fragment_oversample_tiny:
                for idx, L in zip(fpos, frag_smi_len):
                    if L <= 8:
                        w[idx] = 8.0      # tiny fragments (<=~5 heavy atoms)
                    elif L <= 12:
                        w[idx] = 3.0      # small fragments
            wavy_mask = np.zeros_like(frag_mask, dtype=bool)
            if bool(args.fragment_oversample_wavy) and "attachment_render_mode" in df.columns:
                mode = df["attachment_render_mode"].astype(str).to_numpy()
                geom = df.get("attachment_render_geometry", pd.Series([""] * len(df))).astype(str).to_numpy()
                tight = df.get("real_tight_crop_style", pd.Series([False] * len(df))).astype(bool).to_numpy()
                external = df.get("terminal_wavy_externality_passed", pd.Series([False] * len(df))).astype(bool).to_numpy()
                wavy_mask = (
                    frag_mask
                    & (mode == "wavy")
                    & (geom == "custom_markush_attachment_perpendicular_wavy")
                    & tight
                    & external
                )
                w[wavy_mask] = np.maximum(w[wavy_mask], float(args.fragment_wavy_weight))
            real_original_mask = np.zeros(len(df), dtype=bool)
            if "image_domain" in df.columns:
                real_original_mask = df["image_domain"].fillna("").astype(str).eq(
                    "real_original"
                ).to_numpy()
                w[real_original_mask] = np.maximum(
                    w[real_original_mask],
                    float(args.real_original_weight),
                )
            # *CN oversampling: synthetic data under-represents the dominant
            # real-patent amine pattern (~55% vs ~9%), biasing toward C-C.
            cn_mask = np.zeros(len(df), dtype=bool)
            if args.fragment_oversample_cn and "fragment_backbone_smiles" in df.columns:
                bb = df["fragment_backbone_smiles"].fillna("").astype(str).to_numpy()
                # attachment carbon whose next heavy atom is N: backbone starts CN
                cn_mask = frag_mask & np.char.startswith(bb.astype("U"), "CN")
                w[cn_mask] = np.maximum(w[cn_mask], float(args.fragment_cn_weight))
            # Amide/ester oversampling on *C(=O)... backbones; RDKit canonicalizes
            # *C(=O) to "O=C..." so match both prefixes.
            amide_mask = np.zeros(len(df), dtype=bool)
            if args.fragment_oversample_amide and "fragment_backbone_smiles" in df.columns:
                bb = df["fragment_backbone_smiles"].fillna("").astype(str).to_numpy()
                bb_u = bb.astype("U")
                amide_mask = frag_mask & (
                    np.char.startswith(bb_u, "C(=O)") | np.char.startswith(bb_u, "O=C")
                )
                w[amide_mask] = np.maximum(w[amide_mask], float(args.fragment_amide_weight))
            tiny = int(((frag_smi_len <= 8)).sum())
            small = int(((frag_smi_len > 8) & (frag_smi_len <= 12)).sum())
            wavy = int(wavy_mask.sum())
            real_original = int(real_original_mask.sum())
            cn_count = int(cn_mask.sum())
            amide_count = int(amide_mask.sum())
            rank0_print(
                rank,
                f"  sampling priority: real_original(*{float(args.real_original_weight):g})="
                f"{real_original} wavy(*{float(args.fragment_wavy_weight):g})={wavy} "
                f"cn(*{float(args.fragment_cn_weight):g})={cn_count} "
                f"amide(*{float(args.fragment_amide_weight):g})={amide_count} "
                f"tiny(*8)={tiny} small(*3)={small} of {int(frag_mask.sum())} fragments",
            )
            frag_weights = w
        rows_per_label = int(args.epoch_rows_per_label or 0)
        if rows_per_label <= 0:
            counts = {
                int(label): int((label_by_idx == int(label)).sum())
                for label in sorted(np.unique(label_by_idx).tolist())
            }
            smallest = min(counts.values())
            sidecar_counts = [count for label, count in counts.items() if label > 0]
            sidecar_scale = max(sidecar_counts) if sidecar_counts else smallest
            cap = max(1, int(args.auto_epoch_rows_cap))
            rows_per_label = auto_rows_per_label(counts, cap)
            rank0_print(
                rank,
                f"  auto epoch rows/label: {rows_per_label} "
                f"(smallest={smallest}, sidecar_scale={sidecar_scale}, cap={cap})",
            )
        sampler = LabelBalancedSampler(
            label_by_idx,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
            rows_per_label=rows_per_label,
            shuffle=True,
            weights=frag_weights,
            focus_fraction=float(args.sampling_focus_fraction),
        )
        sampling_report = sampler.report()
        rank0_print(rank, f"  sampling: {sampling_report}")
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        ) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=bms_collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    if args.expert_kind not in ("lora", "full_mixture"):
        raise NotImplementedError(
            "expert_kind must be 'lora' or 'full_mixture'."
        )
    moe = MoEDecoder(
        model_args, tokenizer, num_experts=3,
        expert_kind=args.expert_kind, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        include_output_layer=args.include_output_layer, include_edges=args.include_edges,
        sidecar_confidence_threshold=args.sidecar_confidence_threshold,
        routing_strategy=args.routing_strategy,
        router_kind=args.router_kind,
        use_confidence_head=args.confidence_head,
        expected_fragment_star_logit_bias=float(args.expected_fragment_star_logit_bias),
        expected_fragment_star_budget=int(args.expected_fragment_star_budget),
        expected_fragment_max_atoms=int(args.expected_fragment_max_atoms),
        full_mixture_sidecar_mode=args.full_mixture_sidecar_mode,
        mixture_complete_floor=float(args.mixture_complete_floor),
        token_fusion_mode=args.token_fusion_mode,
        token_fusion_initial_sidecar_weight=float(args.token_fusion_initial_sidecar_weight),
        token_fusion_dispatch=str(args.token_fusion_dispatch),
        token_fusion_hard_threshold=float(args.token_fusion_hard_threshold),
        specialist_ownership_scope=str(args.specialist_ownership_scope),
        attachment_set_enabled=bool(args.attachment_set_enabled),
        attachment_set_hidden_dim=int(args.attachment_set_hidden_dim),
        attachment_set_num_queries=int(args.attachment_set_num_queries),
        attachment_set_num_layers=int(args.attachment_set_num_layers),
        attachment_set_num_heads=int(args.attachment_set_num_heads),
        attachment_set_max_count=int(args.attachment_set_max_count),
        attachment_set_min_confidence=float(args.attachment_set_min_confidence),
        attachment_set_max_anchor_distance=float(
            args.attachment_set_max_anchor_distance
        ),
        attachment_set_decode_mode=str(args.attachment_set_decode_mode),
        attachment_set_feature_mode=str(args.attachment_set_feature_mode),
        attachment_set_feature_levels=int(args.attachment_set_feature_levels),
        attachment_set_max_feature_size=int(
            args.attachment_set_max_feature_size
        ),
        attachment_set_min_pointer_confidence=float(
            args.attachment_set_min_pointer_confidence
        ),
        attachment_set_heatmap_logit_scale=float(
            args.attachment_set_heatmap_logit_scale
        ),
        attachment_set_heatmap_prior_precision=float(
            args.attachment_set_heatmap_prior_precision
        ),
        sidecar_coordinate_context=str(args.sidecar_coordinate_context),
        fragment_terminal_action_head_enabled=bool(
            args.fragment_terminal_action_head_enabled
        ),
        fragment_terminal_action_head_hidden_dim=int(
            args.fragment_terminal_action_head_hidden_dim
        ),
        fragment_structured_terminal_edge_enabled=bool(
            args.fragment_structured_terminal_edge_enabled
        ),
        decouple_fusion_policy_optimization=bool(
            args.decouple_fusion_policy_optimization
        ),
        one_sided_fusion_oracle=bool(args.one_sided_fusion_oracle),
        frozen_expert0=not bool(args.unfreeze_expert0),
    )
    rank0_print(rank, f"  expert_kind={moe.expert_kind} num_experts={moe.num_experts} "
                f"num_replaced={getattr(moe, 'num_replaced', 0)} "
                f"(lora rank={args.lora_rank} alpha={args.lora_alpha} "
                f"output_layer={args.include_output_layer} edges={args.include_edges})")
    states = torch.load(args.base_checkpoint, map_location="cpu")
    moe.load_expert0_from_base(states["decoder"])
    if args.expert_kind == "full_mixture":
        # Warm-start the specialist from the frozen base; fine-tuning learns
        # only the markush/fragment delta.
        moe.warm_start_specialist_from_base(states["decoder"])
    if args.resume_router:
        router_states = torch.load(args.resume_router, map_location="cpu")
        moe.load_router(
            router_states,
            require_fusion_state=(moe.token_fusion_mode == "adaptive"),
            require_attachment_set_state=bool(moe.attachment_set_enabled),
            require_terminal_action_state=bool(
                moe.fragment_terminal_action_head_enabled
            ),
            allow_attachment_set_pointer_upgrade=bool(
                args.allow_attachment_set_pointer_upgrade
            ),
            allow_attachment_set_heatmap_upgrade=bool(
                args.allow_attachment_set_heatmap_upgrade
            ),
        )
    if args.expert_kind == "lora" and getattr(args, "resume_adapter", None):
        adapter_states = torch.load(args.resume_adapter, map_location="cpu")
        moe.load_adapter(adapter_states.get("decoder", adapter_states))
    if args.expert_kind == "full_mixture" and getattr(args, "resume_expert1", None):
        e1_states = torch.load(getattr(args, "resume_expert1"), map_location="cpu")
        moe.load_expert(1, e1_states.get("decoder", e1_states))
    if args.expert_kind == "full_mixture" and getattr(args, "resume_expert2", None):
        e2_states = torch.load(getattr(args, "resume_expert2"), map_location="cpu")
        moe.load_expert(2, e2_states.get("decoder", e2_states))
    decoder_trainability = {}
    if args.expert_kind == "full_mixture":
        decoder_trainability["expert0_complete"] = decoder_trainability_manifest(
            moe.experts[0],
            "full" if args.unfreeze_expert0 else "frozen",
        )
        _expert1_params, decoder_trainability["expert1_markush"] = (
            configure_decoder_finetuning(moe.experts[1], "full")
        )
        if args.full_mixture_sidecar_mode == "per_sidecar":
            _expert2_params, decoder_trainability["expert2_fragment"] = (
                configure_decoder_finetuning(
                    moe.experts[2],
                    args.fragment_decoder_train_scope,
                )
            )
            # MoEDecoder.train() uses this training-only attribute to keep the
            # frozen transformer/embedding path deterministic while its heads learn.
            moe.fragment_decoder_train_scope = str(
                args.fragment_decoder_train_scope
            )
    else:
        decoder_trainability["shared_lora_decoder"] = (
            decoder_trainability_manifest(moe.experts[0], "lora_adapters")
        )
    for expert_name, manifest in decoder_trainability.items():
        rank0_print(
            rank,
            f"  {expert_name} scope={manifest['scope']} "
            f"trainable={manifest['trainable_parameters']:,} "
            f"frozen={manifest['frozen_parameters']:,}",
        )
    moe.to(device)
    joint_encoder_training = bool(encoder_params)
    training_root = (
        EncoderMoETrainingModel(base.encoder, moe).to(device)
        if joint_encoder_training
        else moe
    )
    train_model = DistributedDataParallel(
        training_root,
        device_ids=[local_rank],
        output_device=local_rank,
        # Batch-dependent graphs (conditional experts/relations) rule out
        # static_graph; unused-parameter discovery is required for DDP.
        find_unused_parameters=True,
    ) if distributed else training_root
    unwrapped_training_root = train_model.module if distributed else train_model
    model_for_loss = (
        unwrapped_training_root.moe
        if joint_encoder_training
        else unwrapped_training_root
    )
    encoder_for_training = base.encoder

    # The DETR-style heads are randomly initialized; sharing the sidecar's low
    # LR under-trains cardinality/localization.
    router_params = [param for param in model_for_loss.router.parameters() if param.requires_grad]
    attachment_set_params = [
        param for name, param in model_for_loss.named_parameters()
        if param.requires_grad and name.startswith("attachment_set_heads.")
    ]
    terminal_action_params = [
        param for name, param in model_for_loss.named_parameters()
        if param.requires_grad
        and name.startswith("fragment_terminal_action_head.")
    ]
    if args.unfreeze_expert0:
        expert0_params = [
            param for name, param in model_for_loss.named_parameters()
            if param.requires_grad and name.startswith("experts.0.")
            and "expert0_reference" not in name
        ]
        sidecar_params = [
            param for name, param in model_for_loss.named_parameters()
            if param.requires_grad
            and not name.startswith("experts.0.")
            and not name.startswith("expert0_reference.")
            and not name.startswith("router.")
            and not name.startswith("attachment_set_heads.")
            and not name.startswith("fragment_terminal_action_head.")
        ]
        trainable = (
            expert0_params + sidecar_params + attachment_set_params
            + terminal_action_params
            + router_params + encoder_params
        )
        rank0_print(
            rank,
            f"  trainable params: {sum(p.numel() for p in trainable):,} "
            f"(expert0={sum(p.numel() for p in expert0_params):,} @ lr={args.expert0_lr}, "
            f"sidecars={sum(p.numel() for p in sidecar_params):,} @ lr={args.lr}, "
            f"attachment_set={sum(p.numel() for p in attachment_set_params):,} "
            f"@ lr={args.attachment_set_lr}, "
            f"terminal_action={sum(p.numel() for p in terminal_action_params):,} "
            f"@ lr={args.fragment_terminal_action_head_lr}, "
            f"router={sum(p.numel() for p in router_params):,}, "
            f"encoder={sum(p.numel() for p in encoder_params):,} "
            f"@ lr={args.encoder_lr})",
        )
        parameter_groups = [
            {"name": "expert0", "params": expert0_params, "lr": float(args.expert0_lr)},
            {"name": "sidecars", "params": sidecar_params, "lr": float(args.lr)},
        ]
    else:
        expert_params = [
            param for name, param in model_for_loss.named_parameters()
            if param.requires_grad
            and not name.startswith("router.")
            and not name.startswith("attachment_set_heads.")
            and not name.startswith("fragment_terminal_action_head.")
        ]
        # Separate expert1/expert2 groups so markush can be frozen (lr=0)
        # while fragment trains.
        expert1_params = [
            param for name, param in model_for_loss.named_parameters()
            if param.requires_grad
            and name.startswith("experts.1.")
        ]
        expert2_params = [
            param for name, param in model_for_loss.named_parameters()
            if param.requires_grad
            and name.startswith("experts.2.")
        ]
        trainable = (
            expert_params + attachment_set_params + terminal_action_params
            + router_params + encoder_params
        )
        rank0_print(
            rank,
            f"  trainable params: {sum(p.numel() for p in trainable):,} "
            f"(expert={sum(p.numel() for p in expert_params):,} @ lr={args.lr}, "
            f"attachment_set={sum(p.numel() for p in attachment_set_params):,} "
            f"@ lr={args.attachment_set_lr}, "
            f"terminal_action={sum(p.numel() for p in terminal_action_params):,} "
            f"@ lr={args.fragment_terminal_action_head_lr}, "
            f"router={sum(p.numel() for p in router_params):,}, "
            f"encoder={sum(p.numel() for p in encoder_params):,} "
            f"@ lr={args.encoder_lr})",
        )
        parameter_groups = [
            {"name": "expert1_markush", "params": expert1_params, "lr": float(getattr(args, 'expert1_lr', 0.0) or args.lr)},
            {"name": "expert2_fragment", "params": expert2_params, "lr": float(args.lr)},
        ]
    if attachment_set_params:
        parameter_groups.append(
            {
                "name": "attachment_set",
                "params": attachment_set_params,
                "lr": float(args.attachment_set_lr),
            }
        )
    if terminal_action_params:
        parameter_groups.append(
            {
                "name": "terminal_action",
                "params": terminal_action_params,
                "lr": float(args.fragment_terminal_action_head_lr),
            }
        )
    if encoder_params:
        parameter_groups.append(
            {
                "name": "encoder",
                "params": encoder_params,
                "lr": float(args.encoder_lr),
            }
        )
    parameter_groups.append(
        {"name": "router", "params": router_params, "lr": float(args.router_lr)}
    )
    terminal_action_param_ids = {id(param) for param in terminal_action_params}
    non_terminal_trainable = [
        param for param in trainable
        if id(param) not in terminal_action_param_ids
    ]
    optimizer_trainability = validate_optimizer_parameter_coverage(
        model_for_loss,
        encoder_for_training,
        parameter_groups,
    )
    opt = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    sched = linear_warmup_cosine(opt, warmup_steps, total_steps, min_lr_frac=args.min_lr_frac)
    rank0_print(rank, f"  batches/epoch={len(loader)} steps/epoch={steps_per_epoch} "
                      f"total_steps={total_steps} warmup={warmup_steps}")

    log = []
    train_model.train()
    if (
        not args.unfreeze_expert0
        and bool(model_for_loss.experts[0].training)
    ):
        raise RuntimeError("frozen expert0 must remain in eval mode during MoE training")
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        opt.zero_grad(set_to_none=True)
        for it, (ids, imgs, refs) in enumerate(loader):
            accumulation_start = (it // args.grad_accum) * args.grad_accum
            accumulation_size = min(
                args.grad_accum,
                len(loader) - accumulation_start,
            )
            should_update = (
                (it + 1) % args.grad_accum == 0
                or (it + 1) == len(loader)
            )
            if distributed:
                # Equivalent to DDP.no_sync() for non-update micro-batches; the
                # next synchronized backward reduces accumulated gradients.
                train_model.require_backward_grad_sync = bool(should_update)
            imgs = imgs.to(device)
            struct = torch.tensor([int(label_by_idx[i]) for i in ids], device=device)
            if args.coordinate_missing_task_policy == "router_only":
                refs["decoder_task_mask"] = torch.tensor(
                    [bool(coordinate_targets_by_idx[i]) for i in ids],
                    dtype=torch.bool,
                )
            else:
                refs["decoder_task_mask"] = torch.ones(
                    len(ids), dtype=torch.bool
                )
            if not joint_encoder_training:
                # Legacy frozen-encoder path; production trains the encoder
                # inside the DDP boundary below.
                features, encoder_hiddens = base.encoder(imgs)
                features = features.detach()
                encoder_hiddens = [hidden.detach() for hidden in encoder_hiddens]
            do_conf = args.confidence_head and (it % max(1, args.conf_every) == 0)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                if joint_encoder_training:
                    fwd = train_model(
                        imgs,
                        refs,
                        structure_labels=struct,
                    )
                    features = fwd["encoder_features"]
                else:
                    fwd = train_model(
                        features,
                        refs,
                        structure_labels=struct,
                        encoder_hiddens=encoder_hiddens,
                    )
                losses = model_for_loss.compute_moe_loss(
                    fwd, structure_labels=struct,
                    load_balance_weight=args.load_balance_weight,
                    z_loss_weight=args.z_loss_weight,
                    structure_weight=args.structure_weight,
                    sidecar_dense_weight=args.sidecar_dense_weight,
                    sidecar_specialist_weight=args.sidecar_specialist_weight,
                    distill_complete_weight=args.distill_complete_weight,
                    mixture_token_ce_weight=args.mixture_token_ce_weight,
                    mixture_edge_ce_weight=args.mixture_edge_ce_weight,
                    edge_distill_weight=args.edge_distill_weight,
                    attachment_set_loss_weight=args.attachment_set_loss_weight,
                    attachment_set_point_loss_weight=(
                        args.attachment_set_point_loss_weight
                    ),
                    attachment_set_cardinality_loss_weight=(
                        args.attachment_set_cardinality_loss_weight
                    ),
                    attachment_set_relation_loss_weight=(
                        args.attachment_set_relation_loss_weight
                    ),
                    attachment_set_anchor_loss_weight=(
                        args.attachment_set_anchor_loss_weight
                    ),
                    attachment_set_pointer_loss_weight=(
                        args.attachment_set_pointer_loss_weight
                    ),
                    attachment_set_dummy_pointer_loss_weight=(
                        args.attachment_set_dummy_pointer_loss_weight
                    ),
                    attachment_set_heatmap_loss_weight=(
                        args.attachment_set_heatmap_loss_weight
                    ),
                    attachment_set_heatmap_cost_weight=(
                        args.attachment_set_heatmap_cost_weight
                    ),
                    attachment_set_heatmap_diversity_loss_weight=(
                        args.attachment_set_heatmap_diversity_loss_weight
                    ),
                    loss_reduction=args.loss_reduction,
                    anchor_l2_weight=args.anchor_l2_weight,
                    expert0_preserve_weight=(args.expert0_preserve_weight if args.unfreeze_expert0 else 0.0),
                    distill_temperature=args.distill_temperature,
                    routed_mixture_weight=args.routed_mixture_weight,
                    router_margin_weight=args.router_margin_weight,
                    router_margin=args.router_margin,
                    markush_symbol_weight=args.markush_symbol_weight,
                    fragment_symbol_weight=args.fragment_symbol_weight,
                    aromatic_symbol_weight=args.aromatic_symbol_weight,
                    unsaturated_symbol_weight=args.unsaturated_symbol_weight,
                    attachment_symbol_loss_weight=args.attachment_symbol_loss_weight,
                    variable_identity_loss_weight=args.variable_identity_loss_weight,
                    attachment_cardinality_loss_weight=args.attachment_cardinality_loss_weight,
                    fragment_eos_weight=args.fragment_eos_weight,
                    fragment_terminal_dummy_margin_weight=(
                        args.fragment_terminal_dummy_margin_weight
                    ),
                    fragment_terminal_dummy_margin=(
                        args.fragment_terminal_dummy_margin
                    ),
                    fragment_premature_eos_unlikelihood_weight=(
                        args.fragment_premature_eos_unlikelihood_weight
                    ),
                    fragment_terminal_action_head_loss_weight=(
                        args.fragment_terminal_action_head_loss_weight
                    ),
                    token_fusion_supervision_weight=args.token_fusion_supervision_weight,
                    token_fusion_attachment_target=args.token_fusion_attachment_target,
                    token_fusion_backbone_target=args.token_fusion_backbone_target,
                    token_fusion_oracle_temperature=args.token_fusion_oracle_temperature,
                    sidecar_nonzero_edge_weight=args.sidecar_nonzero_edge_weight,
                    dummy_edge_weight=args.dummy_edge_weight,
                    multiple_edge_weight=args.multiple_edge_weight,
                    aromatic_edge_weight=args.aromatic_edge_weight,
                    edge_valence_loss_weight=args.edge_valence_loss_weight,
                    expert_diversity_weight=args.expert_diversity_weight)
                total_loss = losses["loss"]
            # Confidence head runs in fp32 OUTSIDE autocast: its small MLP under
            # bf16/autocast raises "inference tensors cannot be saved for backward".
            conf_val = None
            if do_conf:
                # Real Tanimoto target: free-decode the mixture (no grad) vs gold.
                train_model.eval()
                with torch.no_grad():
                    dec_preds = model_for_loss.decode(features)
                train_model.train()
                coords_l, symbols_l, edges_l = [], [], []
                for p in dec_preds:
                    cc = p.get("chartok_coords", {})
                    if p.get("decode_quality_issue"):
                        coords_l.append([]); symbols_l.append([]); edges_l.append([])
                    else:
                        coords_l.append(cc.get("coords") or [])
                        symbols_l.append(cc.get("symbols") or [])
                        edges_l.append(p.get("edges") or [])
                smi_list, _, _, graph_quality_issues = convert_graph_to_smiles(
                    coords_l, symbols_l, edges_l, num_workers=1)
                smi_list = [
                    "" if issue else smiles
                    for smiles, issue in zip(smi_list, graph_quality_issues)
                ]
                gold = [str(df["SMILES"].iloc[i]) for i in ids]
                targets = torch.tensor(
                    [smi_tanimoto(p, g) for p, g in zip(smi_list, gold)],
                    device=device, dtype=torch.float32)
                conf_logits = model_for_loss.confidence_head_forward(fwd)
                conf_loss = confidence_loss(conf_logits, targets)
                total_loss = total_loss + args.conf_weight * conf_loss
                conf_val = float(conf_loss.detach())
            # SCST/REINFORCE on the deployed assembled-SMILES Tanimoto with a
            # self-critical greedy baseline; complete rows are skipped.
            scst_val = None
            if args.scst_weight > 0 and (it % max(1, args.scst_every) == 0):
                _l2t = {0: "complete", 1: "markush", 2: "fragment"}
                _exp = [_l2t.get(int(s.item())) for s in struct]
                _gold = [str(df["SMILES"].iloc[i]) for i in ids]
                (_rl_weights, _, _, _fd, _, _) = model_for_loss._compute_expected_weights(
                    features,
                    _exp,
                )
                _per_sc = (model_for_loss.expert_kind == "full_mixture"
                           and getattr(model_for_loss, "full_mixture_sidecar_mode", "collapsed") == "per_sidecar")

                def _asm(pred):
                    if pred.get("decode_quality_issue"):
                        return ""
                    _cc = pred.get("chartok_coords", {})
                    _smi, _, _, _iss = convert_graph_to_smiles(
                        [_cc.get("coords") or []], [_cc.get("symbols") or []],
                        [pred.get("edges") or []], num_workers=1)
                    return "" if (_iss and _iss[0]) else (_smi[0] if _smi else "")

                _terms = []
                _G = max(2, int(args.scst_group))  # GRPO group size
                _nrows = 0
                for _b in range(features.size(0)):
                    if bool(_fd[_b].item()) or _nrows >= args.scst_max_rows:
                        continue
                    _enc = features[_b:_b + 1]
                    _sc = model_for_loss._decode_constraints_for_expected(_exp[_b], {}, device)
                    _sei = (1 if _exp[_b] == "markush" else 2) if _per_sc else 1
                    if _per_sc:
                        _w2 = model_for_loss._mixture_weights_for_sidecar(
                            _rl_weights[_b],
                            _sei,
                        )
                    else:
                        _w2 = model_for_loss._mixture_weights_2(_rl_weights[_b])
                    # GRPO: sample G trajectories per row; near-saturated groups
                    # give ~zero advantage (no drift), diverse groups give signal.
                    _logps, _rewards = [], []
                    for _g in range(_G):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                            _pred_g, _logp_g = model_for_loss.decode_mixture_rl(
                                _enc, _w2, _sc, sample=True, sidecar_expert_idx=_sei)
                        _logps.append(_logp_g)
                        _rewards.append(smi_tanimoto(_asm(_pred_g), _gold[_b]))
                    _rewards = torch.tensor(_rewards, device=device, dtype=torch.float32)
                    _std = float(_rewards.std().item())
                    if _std < 1e-4:
                        _nrows += 1
                        continue  # degenerate group (all-same reward): no signal, skip
                    # Raw advantage (r_i - mean), NOT std-normalized: std-norm would
                    # re-amplify near-saturated rows' tiny differences into drift.
                    _adv = _rewards - _rewards.mean()
                    for _g in range(_G):
                        _a = float(_adv[_g])
                        if abs(_a) > 1e-6:
                            _terms.append(-_a * _logps[_g])
                    _nrows += 1
                if _terms:
                    _scst_loss = torch.stack(_terms).mean()
                    total_loss = total_loss + args.scst_weight * _scst_loss
                    scst_val = float(_scst_loss.detach())
            fragment_grpo_val = None
            fragment_grpo_reference_kl_val = None
            fragment_grpo_reward_val = None
            fragment_grpo_backbone_val = None
            fragment_grpo_attachment_val = None
            fragment_counterfactual_dpo_val = None
            fragment_counterfactual_reward_delta_val = None
            fragment_counterfactual_attachment_gain_val = None
            fragment_counterfactual_preference_rate_val = None
            fragment_counterfactual_edge_val = None
            fragment_counterfactual_edge_reward_gain_val = None
            fragment_counterfactual_edge_expected_reward_val = None
            fragment_search_policy_val = None
            fragment_search_policy_reward_gain_val = None
            fragment_search_policy_apply_rate_val = None
            fragment_search_edge_val = None
            fragment_search_edge_target_probability_val = None
            terminal_search_enabled = bool(
                args.fragment_search_policy_weight > 0.0
                or args.fragment_search_edge_weight > 0.0
            )
            terminal_objective_enabled = bool(
                args.fragment_counterfactual_dpo_weight > 0.0
                or args.fragment_counterfactual_edge_weight > 0.0
                or terminal_search_enabled
            )
            if (
                (
                    args.fragment_grpo_weight > 0.0
                    or terminal_objective_enabled
                )
                and (it + 1) % args.fragment_grpo_every == 0
            ):
                # Detached deployment-mode features: the on-policy auxiliary
                # updates only Expert2, not the shared visual encoder.
                if joint_encoder_training:
                    encoder_was_training = bool(encoder_for_training.training)
                    encoder_for_training.eval()
                    with torch.no_grad(), torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        rollout_features, _ = encoder_for_training(imgs)
                    if encoder_was_training:
                        encoder_for_training.train()
                else:
                    rollout_features = features
                rollout_features = rollout_features.detach()

                fragment_expert = model_for_loss.experts[2]
                fragment_expert_was_training = bool(fragment_expert.training)
                fragment_expert.eval()
                policy_terms = []
                counterfactual_terms = []
                counterfactual_edge_terms = []
                search_policy_terms = []
                search_edge_terms = []
                reference_kl_terms = []
                reward_metrics = []
                counterfactual_metrics = []
                counterfactual_edge_audits = []
                search_policy_metrics = []
                search_edge_audits = []
                rollout_rows = 0
                try:
                    fragment_constraints = (
                        model_for_loss._decode_constraints_for_expected(
                            "fragment",
                            {},
                            device,
                        )
                    )
                    for batch_index in range(rollout_features.size(0)):
                        if int(struct[batch_index].item()) != 2:
                            continue
                        if rollout_rows >= int(args.fragment_grpo_max_rows):
                            break
                        gold_smiles = str(df["SMILES"].iloc[ids[batch_index]])
                        rollout_feature = rollout_features[
                            batch_index : batch_index + 1
                        ]

                        if terminal_objective_enabled:
                            with torch.autocast(
                                "cuda",
                                dtype=torch.bfloat16,
                                enabled=(device.type == "cuda"),
                            ):
                                (
                                    greedy_pred,
                                    greedy_log_prob,
                                    greedy_reference_kl,
                                    greedy_info,
                                ) = model_for_loss.decode_direct_sidecar_rl(
                                    rollout_feature,
                                    fragment_constraints,
                                    sidecar_expert_idx=2,
                                    sample=False,
                                    reference_kl=(
                                        args.fragment_grpo_reference_kl_weight
                                        > 0.0
                                    ),
                                    return_terminal_action_logits=(
                                        args.fragment_search_policy_weight > 0.0
                                    ),
                                )
                            if args.fragment_grpo_reference_kl_weight > 0.0:
                                reference_kl_terms.append(greedy_reference_kl)
                            greedy_metrics = fragment_graph_reward(
                                decoded_prediction_smiles(greedy_pred),
                                gold_smiles,
                            )
                            if (
                                args.fragment_terminal_action_head_enabled
                                and args.fragment_search_policy_weight > 0.0
                                and int(greedy_info["generated_stars"]) == 1
                            ):
                                greedy_post_attachment_eos_loss = (
                                    terminal_action_pair_margin_loss(
                                        (
                                            greedy_info.get(
                                                "_terminal_stop_logits"
                                            )
                                            if greedy_info.get(
                                                "_terminal_stop_logits"
                                            ) is not None
                                            else greedy_info.get(
                                                "_terminal_action_logits"
                                            )
                                        ),
                                        target_star=bool(
                                            greedy_info.get(
                                                "_terminal_stop_logits"
                                            ) is not None
                                        ),
                                        target_logit_margin=(
                                            args.fragment_search_policy_margin
                                        ),
                                    )
                                )
                                if greedy_post_attachment_eos_loss is not None:
                                    search_policy_terms.append(
                                        greedy_post_attachment_eos_loss
                                    )
                            greedy_sequence = list(greedy_info["sequence"])
                            if (
                                args.fragment_counterfactual_edge_weight > 0.0
                                and int(greedy_info["generated_stars"]) == 1
                            ):
                                (
                                    greedy_edge_loss,
                                    _greedy_best_metrics,
                                    greedy_edge_audit,
                                ) = terminal_dummy_edge_expected_reward(
                                    greedy_pred,
                                    gold_smiles,
                                )
                                if greedy_edge_loss is not None:
                                    counterfactual_edge_terms.append(
                                        greedy_edge_loss
                                    )
                                if greedy_edge_audit.get("candidate_count", 0):
                                    counterfactual_edge_audits.append(
                                        greedy_edge_audit
                                    )
                            if (
                                args.fragment_search_edge_weight > 0.0
                                and int(greedy_info["generated_stars"]) == 1
                            ):
                                (
                                    greedy_search_edge_loss,
                                    _greedy_search_best_metrics,
                                    greedy_search_edge_audit,
                                ) = terminal_dummy_edge_expected_reward(
                                    greedy_pred,
                                    gold_smiles,
                                    distill_best_bonded_action=True,
                                )
                                greedy_search_reward_gain = float(
                                    (
                                        _greedy_search_best_metrics or {}
                                    ).get("reward", greedy_metrics["reward"])
                                    - greedy_metrics["reward"]
                                )
                                greedy_search_edge_audit[
                                    "deployed_reward_gain"
                                ] = greedy_search_reward_gain
                                greedy_search_edge_audit[
                                    "search_policy_applied"
                                ] = bool(
                                    greedy_search_reward_gain >= float(
                                        args.fragment_counterfactual_reward_margin
                                    )
                                )
                                if (
                                    greedy_search_edge_loss is not None
                                    and greedy_search_reward_gain >= float(
                                        args.fragment_counterfactual_reward_margin
                                    )
                                ):
                                    search_edge_terms.append(
                                        greedy_search_edge_loss
                                    )
                                if greedy_search_edge_audit.get(
                                    "candidate_count", 0
                                ):
                                    search_edge_audits.append(
                                        greedy_search_edge_audit
                                    )
                            if (
                                bool(greedy_info["eos_found"])
                                and int(greedy_info["generated_stars"]) == 0
                                and greedy_sequence
                                and int(greedy_sequence[-1]) == EOS_ID
                            ):
                                star_token_id = int(
                                    fragment_constraints[ATOM_FORMAT][
                                        "star_token_id"
                                    ]
                                )
                                eos_step = len(greedy_sequence) - 1
                                with torch.autocast(
                                    "cuda",
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda"),
                                ):
                                    (
                                        alternative_pred,
                                        alternative_log_prob,
                                        _,
                                        alternative_info,
                                    ) = model_for_loss.decode_direct_sidecar_rl(
                                        rollout_feature,
                                        fragment_constraints,
                                        sidecar_expert_idx=2,
                                        sample=False,
                                        reference_kl=False,
                                        return_terminal_action_logits=bool(
                                            args.fragment_terminal_action_head_enabled
                                        ),
                                        forced_token_at_step=(
                                            eos_step,
                                            star_token_id,
                                        ),
                                    )
                                alternative_metrics = fragment_graph_reward(
                                    decoded_prediction_smiles(alternative_pred),
                                    gold_smiles,
                                )
                                searched_edge_loss = None
                                searched_edge_audit = None
                                if args.fragment_counterfactual_edge_weight > 0.0:
                                    (
                                        alternative_edge_loss,
                                        best_edge_metrics,
                                        alternative_edge_audit,
                                    ) = terminal_dummy_edge_expected_reward(
                                        alternative_pred,
                                        gold_smiles,
                                    )
                                    if alternative_edge_loss is not None:
                                        counterfactual_edge_terms.append(
                                            alternative_edge_loss
                                        )
                                    if best_edge_metrics is not None:
                                        alternative_metrics = best_edge_metrics
                                    if alternative_edge_audit.get(
                                        "candidate_count", 0
                                    ):
                                        counterfactual_edge_audits.append(
                                            alternative_edge_audit
                                        )
                                if terminal_search_enabled:
                                    (
                                        searched_edge_loss,
                                        searched_best_metrics,
                                        searched_edge_audit,
                                    ) = terminal_dummy_edge_expected_reward(
                                        alternative_pred,
                                        gold_smiles,
                                        distill_best_bonded_action=True,
                                    )
                                    if searched_best_metrics is not None:
                                        # Terminal action = searched bonded graph,
                                        # not the forced-star edge argmax.
                                        alternative_metrics = searched_best_metrics
                                    if searched_edge_audit.get(
                                        "candidate_count", 0
                                    ):
                                        search_edge_audits.append(
                                            searched_edge_audit
                                        )
                                reward_delta = float(
                                    alternative_metrics["reward"]
                                    - greedy_metrics["reward"]
                                )
                                attachment_gain = float(
                                    alternative_metrics["attachment_valid"]
                                ) - float(greedy_metrics["attachment_valid"])
                                preference_applied = bool(
                                    abs(reward_delta)
                                    >= float(
                                        args.fragment_counterfactual_reward_margin
                                    )
                                )
                                if args.fragment_counterfactual_dpo_weight > 0.0:
                                    preference_loss = (
                                        reward_weighted_counterfactual_dpo_loss(
                                            alternative_log_prob,
                                            greedy_log_prob,
                                            reward_delta,
                                            beta=args.fragment_counterfactual_dpo_beta,
                                            reward_margin=(
                                                args.fragment_counterfactual_reward_margin
                                            ),
                                        )
                                    )
                                    if preference_loss is not None:
                                        counterfactual_terms.append(
                                            preference_loss
                                        )
                                if terminal_search_enabled:
                                    if searched_edge_audit is not None:
                                        searched_edge_audit[
                                            "deployed_reward_gain"
                                        ] = reward_delta
                                        searched_edge_audit[
                                            "search_policy_applied"
                                        ] = bool(
                                            reward_delta >= float(
                                                args.fragment_counterfactual_reward_margin
                                            )
                                        )
                                    boundary_action_logits = greedy_info.get(
                                        "_boundary_action_logits"
                                    )
                                    if boundary_action_logits is None:
                                        boundary_action_logits = greedy_info.get(
                                            "_terminal_action_logits"
                                        )
                                    searched_policy_loss = (
                                        searched_terminal_action_loss(
                                            boundary_action_logits,
                                            reward_delta,
                                            reward_margin=(
                                                args.fragment_counterfactual_reward_margin
                                            ),
                                            target_logit_margin=(
                                                args.fragment_search_policy_margin
                                            ),
                                        )
                                    )
                                    if (
                                        args.fragment_search_policy_weight > 0.0
                                        and searched_policy_loss is not None
                                    ):
                                        search_policy_terms.append(
                                            searched_policy_loss
                                        )
                                        post_attachment_eos_loss = (
                                            terminal_action_pair_margin_loss(
                                                (
                                                    alternative_info.get(
                                                        "_terminal_stop_logits"
                                                    )
                                                    if alternative_info.get(
                                                        "_terminal_stop_logits"
                                                    ) is not None
                                                    else alternative_info.get(
                                                        "_terminal_action_logits"
                                                    )
                                                ),
                                                target_star=bool(
                                                    alternative_info.get(
                                                        "_terminal_stop_logits"
                                                    ) is not None
                                                ),
                                                target_logit_margin=(
                                                    args.fragment_search_policy_margin
                                                ),
                                            )
                                        )
                                        if post_attachment_eos_loss is not None:
                                            search_policy_terms.append(
                                                post_attachment_eos_loss
                                            )
                                    if (
                                        args.fragment_search_edge_weight > 0.0
                                        and searched_edge_loss is not None
                                        and reward_delta >= float(
                                            args.fragment_counterfactual_reward_margin
                                        )
                                    ):
                                        search_edge_terms.append(
                                            searched_edge_loss
                                        )
                                    search_policy_metrics.append(
                                        {
                                            "reward_gain": reward_delta,
                                            "applied": bool(
                                                searched_policy_loss is not None
                                            ),
                                        }
                                    )
                                counterfactual_metrics.append(
                                    {
                                        "reward_delta": reward_delta,
                                        "attachment_gain": attachment_gain,
                                        "preference_applied": preference_applied,
                                    }
                                )

                        if args.fragment_grpo_weight > 0.0:
                            group_log_probs = []
                            group_rewards = []
                            for _ in range(int(args.fragment_grpo_group_size)):
                                with torch.autocast(
                                    "cuda",
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda"),
                                ):
                                    (
                                        rollout_pred,
                                        sequence_log_prob,
                                        rollout_reference_kl,
                                        _rollout_info,
                                    ) = model_for_loss.decode_direct_sidecar_rl(
                                        rollout_feature,
                                        fragment_constraints,
                                        sidecar_expert_idx=2,
                                        sample=True,
                                        temperature=args.fragment_grpo_temperature,
                                        top_p=args.fragment_grpo_top_p,
                                        reference_kl=(
                                            args.fragment_grpo_reference_kl_weight
                                            > 0.0
                                            and not terminal_objective_enabled
                                        ),
                                    )
                                metrics = fragment_graph_reward(
                                    decoded_prediction_smiles(rollout_pred),
                                    gold_smiles,
                                )
                                group_log_probs.append(sequence_log_prob)
                                group_rewards.append(float(metrics["reward"]))
                                reward_metrics.append(metrics)
                                if (
                                    args.fragment_grpo_reference_kl_weight > 0.0
                                    and not terminal_objective_enabled
                                ):
                                    reference_kl_terms.append(
                                        rollout_reference_kl
                                    )
                            group_loss, _advantages = group_relative_rloo_loss(
                                group_log_probs,
                                group_rewards,
                            )
                            if group_loss is not None:
                                policy_terms.append(group_loss)
                        rollout_rows += 1
                finally:
                    if fragment_expert_was_training:
                        fragment_expert.train()
                    model_for_loss._clear_caches()

                if policy_terms:
                    fragment_grpo_loss = torch.stack(policy_terms).mean()
                    total_loss = total_loss + (
                        float(args.fragment_grpo_weight)
                        * fragment_grpo_loss
                    )
                    fragment_grpo_val = float(fragment_grpo_loss.detach())
                if counterfactual_terms:
                    fragment_counterfactual_dpo_loss = torch.stack(
                        counterfactual_terms
                    ).mean()
                    total_loss = total_loss + (
                        float(args.fragment_counterfactual_dpo_weight)
                        * fragment_counterfactual_dpo_loss
                    )
                    fragment_counterfactual_dpo_val = float(
                        fragment_counterfactual_dpo_loss.detach()
                    )
                if counterfactual_edge_terms:
                    fragment_counterfactual_edge_loss = torch.stack(
                        counterfactual_edge_terms
                    ).mean()
                    total_loss = total_loss + (
                        float(args.fragment_counterfactual_edge_weight)
                        * fragment_counterfactual_edge_loss
                    )
                    fragment_counterfactual_edge_val = float(
                        fragment_counterfactual_edge_loss.detach()
                    )
                if search_policy_terms:
                    fragment_search_policy_loss = torch.stack(
                        search_policy_terms
                    ).mean()
                    total_loss = total_loss + (
                        float(args.fragment_search_policy_weight)
                        * fragment_search_policy_loss
                    )
                    fragment_search_policy_val = float(
                        fragment_search_policy_loss.detach()
                    )
                if search_edge_terms:
                    fragment_search_edge_loss = torch.stack(
                        search_edge_terms
                    ).mean()
                    total_loss = total_loss + (
                        float(args.fragment_search_edge_weight)
                        * fragment_search_edge_loss
                    )
                    fragment_search_edge_val = float(
                        fragment_search_edge_loss.detach()
                    )
                if reference_kl_terms:
                    fragment_grpo_reference_kl_loss = torch.stack(
                        reference_kl_terms
                    ).mean()
                    total_loss = total_loss + (
                        float(args.fragment_grpo_reference_kl_weight)
                        * fragment_grpo_reference_kl_loss
                    )
                    fragment_grpo_reference_kl_val = float(
                        fragment_grpo_reference_kl_loss.detach()
                    )
                if reward_metrics:
                    fragment_grpo_reward_val = float(
                        np.mean([item["reward"] for item in reward_metrics])
                    )
                    fragment_grpo_backbone_val = float(
                        np.mean(
                            [
                                item["backbone_tanimoto"]
                                for item in reward_metrics
                            ]
                        )
                    )
                    fragment_grpo_attachment_val = float(
                        np.mean(
                            [
                                float(item["attachment_valid"])
                                for item in reward_metrics
                            ]
                        )
                    )
                if counterfactual_metrics:
                    fragment_counterfactual_reward_delta_val = float(
                        np.mean(
                            [
                                item["reward_delta"]
                                for item in counterfactual_metrics
                            ]
                        )
                    )
                    fragment_counterfactual_attachment_gain_val = float(
                        np.mean(
                            [
                                item["attachment_gain"]
                                for item in counterfactual_metrics
                            ]
                        )
                    )
                    fragment_counterfactual_preference_rate_val = float(
                        np.mean(
                            [
                                float(item["preference_applied"])
                                for item in counterfactual_metrics
                            ]
                        )
                    )
                if counterfactual_edge_audits:
                    fragment_counterfactual_edge_reward_gain_val = float(
                        np.mean(
                            [
                                item["best_reward"] - item["no_bond_reward"]
                                for item in counterfactual_edge_audits
                            ]
                        )
                    )
                    fragment_counterfactual_edge_expected_reward_val = float(
                        np.mean(
                            [
                                item["expected_reward"]
                                for item in counterfactual_edge_audits
                            ]
                        )
                    )
                if search_policy_metrics:
                    applied_search_metrics = [
                        item for item in search_policy_metrics
                        if item["applied"]
                    ]
                    fragment_search_policy_reward_gain_val = float(
                        np.mean(
                            [item["reward_gain"] for item in search_policy_metrics]
                        )
                    )
                    fragment_search_policy_apply_rate_val = float(
                        len(applied_search_metrics) / len(search_policy_metrics)
                    )
                else:
                    fragment_search_policy_apply_rate_val = None
                if search_edge_audits:
                    applied_search_edge_audits = [
                        item for item in search_edge_audits
                        if float(item.get("deployed_reward_gain", float("-inf")))
                        >= float(args.fragment_counterfactual_reward_margin)
                    ]
                    if applied_search_edge_audits:
                        fragment_search_edge_target_probability_val = float(
                            np.mean(
                                [
                                    item["search_policy_target_probability"]
                                    for item in applied_search_edge_audits
                                ]
                            )
                        )
            (total_loss / accumulation_size).backward()

            if should_update:
                if non_terminal_trainable:
                    torch.nn.utils.clip_grad_norm_(
                        non_terminal_trainable,
                        args.grad_clip,
                    )
                if terminal_action_params:
                    torch.nn.utils.clip_grad_norm_(
                        terminal_action_params,
                        args.fragment_terminal_action_head_grad_clip,
                    )
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                global_step += 1
                if writer is not None:
                    writer.add_scalar("train/loss", float(total_loss.detach()), global_step)
                    group_lrs = {
                        str(group.get("name") or ""): float(group["lr"])
                        for group in opt.param_groups
                    }
                    writer.add_scalar("train/lr_experts", group_lrs.get("sidecars", group_lrs.get("expert2_fragment", 0.0)), global_step)
                    writer.add_scalar("train/lr_router", group_lrs["router"], global_step)
                    if "attachment_set" in group_lrs:
                        writer.add_scalar(
                            "train/lr_attachment_set",
                            group_lrs["attachment_set"],
                            global_step,
                        )
                    if "terminal_action" in group_lrs:
                        writer.add_scalar(
                            "train/lr_terminal_action",
                            group_lrs["terminal_action"],
                            global_step,
                        )
                    if "encoder" in group_lrs:
                        writer.add_scalar(
                            "train/lr_encoder",
                            group_lrs["encoder"],
                            global_step,
                        )
                    if args.unfreeze_expert0:
                        writer.add_scalar("train/lr_expert0", group_lrs["expert0"], global_step)
                    for key, value in losses.items():
                        if key == "loss" or value is None:
                            continue
                        value_f = finite_metric(value)
                        if value_f is not None:
                            writer.add_scalar(f"train/{key}", value_f, global_step)
                    if conf_val is not None:
                        writer.add_scalar("train/confidence_loss", conf_val, global_step)
                    if scst_val is not None:
                        writer.add_scalar("train/scst_loss", scst_val, global_step)
                    if fragment_grpo_val is not None:
                        writer.add_scalar(
                            "train/fragment_grpo_loss",
                            fragment_grpo_val,
                            global_step,
                        )
                    if fragment_grpo_reference_kl_val is not None:
                        writer.add_scalar(
                            "train/fragment_grpo_reference_kl",
                            fragment_grpo_reference_kl_val,
                            global_step,
                        )
                    if fragment_grpo_reward_val is not None:
                        writer.add_scalar(
                            "train/fragment_grpo_reward",
                            fragment_grpo_reward_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_grpo_backbone_tanimoto",
                            fragment_grpo_backbone_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_grpo_attachment_valid_rate",
                            fragment_grpo_attachment_val,
                            global_step,
                        )
                    if fragment_counterfactual_dpo_val is not None:
                        writer.add_scalar(
                            "train/fragment_counterfactual_dpo_loss",
                            fragment_counterfactual_dpo_val,
                            global_step,
                        )
                    if fragment_counterfactual_reward_delta_val is not None:
                        writer.add_scalar(
                            "train/fragment_counterfactual_reward_delta",
                            fragment_counterfactual_reward_delta_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_counterfactual_attachment_gain",
                            fragment_counterfactual_attachment_gain_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_counterfactual_preference_rate",
                            fragment_counterfactual_preference_rate_val,
                            global_step,
                        )
                    if fragment_counterfactual_edge_val is not None:
                        writer.add_scalar(
                            "train/fragment_counterfactual_edge_loss",
                            fragment_counterfactual_edge_val,
                            global_step,
                        )
                    if fragment_counterfactual_edge_reward_gain_val is not None:
                        writer.add_scalar(
                            "train/fragment_counterfactual_edge_reward_gain",
                            fragment_counterfactual_edge_reward_gain_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_counterfactual_edge_expected_reward",
                            fragment_counterfactual_edge_expected_reward_val,
                            global_step,
                        )
                    if fragment_search_policy_val is not None:
                        writer.add_scalar(
                            "train/fragment_search_policy_loss",
                            fragment_search_policy_val,
                            global_step,
                        )
                    if fragment_search_policy_reward_gain_val is not None:
                        writer.add_scalar(
                            "train/fragment_search_policy_reward_gain",
                            fragment_search_policy_reward_gain_val,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/fragment_search_policy_apply_rate",
                            fragment_search_policy_apply_rate_val,
                            global_step,
                        )
                    if fragment_search_edge_val is not None:
                        writer.add_scalar(
                            "train/fragment_search_edge_loss",
                            fragment_search_edge_val,
                            global_step,
                        )
                    if fragment_search_edge_target_probability_val is not None:
                        writer.add_scalar(
                            "train/fragment_search_edge_target_probability",
                            fragment_search_edge_target_probability_val,
                            global_step,
                        )

            if is_rank0(rank) and (it + 1) % 20 == 0:
                group_lrs = {
                    str(group.get("name") or ""): float(group["lr"])
                    for group in opt.param_groups
                }
                lr = group_lrs.get("sidecars", group_lrs.get("expert2_fragment", 0.0))
                router_lr = group_lrs["router"]
                attachment_lr = group_lrs.get("attachment_set")
                terminal_action_lr = group_lrs.get("terminal_action")
                encoder_lr = group_lrs.get("encoder")
                conf_str = f" conf={conf_val:.3f}" if conf_val is not None else ""
                scst_str = f" scst={scst_val:.3f}" if scst_val is not None else ""
                grpo_str = (
                    f" grpo={fmt_metric(fragment_grpo_val)}"
                    f" rkl={fmt_metric(fragment_grpo_reference_kl_val)}"
                    f" rr={fmt_metric(fragment_grpo_reward_val)}"
                    f" rbb={fmt_metric(fragment_grpo_backbone_val)}"
                    f" ratt={fmt_metric(fragment_grpo_attachment_val)}"
                    if fragment_grpo_reward_val is not None
                    else ""
                )
                counterfactual_str = (
                    f" cdpo={fmt_metric(fragment_counterfactual_dpo_val)}"
                    f" cdr={fmt_metric(fragment_counterfactual_reward_delta_val)}"
                    f" cda={fmt_metric(fragment_counterfactual_attachment_gain_val)}"
                    f" cpr={fmt_metric(fragment_counterfactual_preference_rate_val)}"
                    if fragment_counterfactual_reward_delta_val is not None
                    else ""
                )
                counterfactual_edge_str = (
                    f" ced={fmt_metric(fragment_counterfactual_edge_val)}"
                    f" ceg={fmt_metric(fragment_counterfactual_edge_reward_gain_val)}"
                    f" cer={fmt_metric(fragment_counterfactual_edge_expected_reward_val)}"
                    if fragment_counterfactual_edge_reward_gain_val is not None
                    else ""
                )
                search_str = (
                    f" spl={fmt_metric(fragment_search_policy_val)}"
                    f" spg={fmt_metric(fragment_search_policy_reward_gain_val)}"
                    f" spa={fmt_metric(fragment_search_policy_apply_rate_val)}"
                    f" sel={fmt_metric(fragment_search_edge_val)}"
                    f" sep={fmt_metric(fragment_search_edge_target_probability_val)}"
                    if fragment_search_policy_reward_gain_val is not None
                    or fragment_search_edge_target_probability_val is not None
                    else ""
                )
                msg = (f"epoch {epoch} it {it+1}/{len(loader)} step {global_step} "
                       f"loss={float(total_loss.detach()):.3f} "
                              f"tok={fmt_metric(losses['token_loss'])} edge={fmt_metric(losses['edge_loss'])} "
                              f"vval={fmt_metric(losses.get('edge_valence_loss'))} "
                              f"attach={fmt_metric(losses['attachment_token_loss'])} "
                              f"term={fmt_metric(losses['terminal_dummy_margin_loss'])} "
                              f"thead={fmt_metric(losses.get('fragment_terminal_action_head_loss'))} "
                              f"peos={fmt_metric(losses['premature_eos_unlikelihood_loss'])} "
                              f"var={fmt_metric(losses.get('variable_identity_loss'))} "
                              f"card={fmt_metric(losses['attachment_cardinality_loss'])} "
                              f"ptr={fmt_metric(losses.get('attachment_set_pointer_loss'))} "
                              f"dptr={fmt_metric(losses.get('attachment_set_dummy_pointer_loss'))} "
                              f"hmap={fmt_metric(losses.get('attachment_set_heatmap_loss'))} "
                              f"hdiv={fmt_metric(losses.get('attachment_set_heatmap_diversity_loss'))} "
                              f"mix={fmt_metric(losses['mixture_token_ce_loss'])}/"
                              f"{fmt_metric(losses['mixture_edge_ce_loss'])} "
                              f"gate={fmt_metric(losses['token_fusion_supervision_loss'])} "
                              f"alpha={fmt_metric(losses['fusion_alpha_backbone'])}/"
                              f"{fmt_metric(losses['fusion_alpha_attachment'])} "
                              f"lb={fmt_metric(losses['load_balance_loss'])} z={float(losses['z_loss']):.4f} "
                       f"struct={fmt_metric(losses['structure_loss'])} "
                       f"dist={fmt_metric(losses['token_distill_loss'])} "
                       f"mgn={fmt_metric(losses['router_margin_loss'])} "
                       f"div={fmt_metric(losses['expert_diversity_loss'])}{conf_str}{scst_str}{grpo_str}{counterfactual_str}{counterfactual_edge_str}{search_str} "
                       f"lr={lr:.2e}/{router_lr:.2e}"
                       + (
                           f" set_lr={attachment_lr:.2e}"
                           if attachment_lr is not None
                           else ""
                       )
                       + (
                           f" term_lr={terminal_action_lr:.2e}"
                           if terminal_action_lr is not None
                           else ""
                       )
                       + (
                           f" enc_lr={encoder_lr:.2e}"
                           if encoder_lr is not None
                           else ""
                       ))
                print(msg); log.append(msg)
        elapsed = time.time() - t0
        samples_per_second = (
            len(loader) * args.batch_size * world_size / max(elapsed, 1e-6)
        )
        memory_text = ""
        if device.type == "cuda":
            memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device),
                ],
                device=device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            peak_alloc_gib = float(memory[0].item()) / (1024 ** 3)
            peak_reserved_gib = float(memory[1].item()) / (1024 ** 3)
            memory_text = (
                f" peak_cuda={peak_alloc_gib:.2f}/{peak_reserved_gib:.2f}GiB"
            )
        rank0_print(
            rank,
            f"  epoch {epoch} done in {elapsed:.1f}s "
            f"throughput={samples_per_second:.1f} images/s{memory_text}",
        )
        if is_rank0(rank) and args.save_epoch_checkpoints:
            paths = save_moe_state(
                model_for_loss,
                model_args,
                args.output_dir,
                tag=f"epoch_{epoch:03d}",
                confidence_head=args.confidence_head,
                save_expert0=args.unfreeze_expert0,
                expert_kind=args.expert_kind,
                encoder=encoder_for_training,
                encoder_trainable_names=encoder_trainable_names,
                encoder_finetune_stages=args.encoder_finetune_stages,
            )
            ckpt_parts = []
            for key in (
                "adapter_path",
                "expert0_path",
                "expert1_path",
                "expert2_path",
                "encoder_path",
            ):
                if paths.get(key):
                    ckpt_parts.append(paths[key])
            ckpt_parts.append(paths["router_path"])
            msg = f"saved epoch checkpoint epoch={epoch} -> " + ", ".join(ckpt_parts)
            print(msg, flush=True)
            log.append(msg)

    # ---- save sidecar adapter + router (matching model.py loader) ----
    # model.py._maybe_load_moe_inference expects each expert ckpt to have a
    # 'decoder' key and the router ckpt to have a 'router' key.
    if distributed:
        dist.barrier()
    if not is_rank0(rank):
        if distributed:
            dist.destroy_process_group()
        return

    saved = save_moe_state(
        model_for_loss,
        model_args,
        args.output_dir,
        tag=None,
        confidence_head=args.confidence_head,
        save_expert0=args.unfreeze_expert0,
        expert_kind=args.expert_kind,
        encoder=encoder_for_training,
        encoder_trainable_names=encoder_trainable_names,
        encoder_finetune_stages=args.encoder_finetune_stages,
    )
    adapter_path = saved.get("adapter_path")
    expert0_path = saved.get("expert0_path")
    expert1_path = saved.get("expert1_path")
    expert2_path = saved.get("expert2_path")
    router_path = saved["router_path"]
    confidence_path = saved["confidence_path"]
    regime_path = saved.get("regime_path")
    encoder_path = saved.get("encoder_path")
    if adapter_path:
        print(f"saved adapter -> {adapter_path}  (LoRA, rank={args.lora_rank})")
    if expert0_path:
        print(f"saved expert0 -> {expert0_path}  (UNFROZEN, joint-trained)")
    if expert1_path:
        print(f"saved experts -> {expert1_path}, {expert2_path}")
    print(f"saved router  -> {router_path}")
    if confidence_path:
        print(f"saved conf    -> {confidence_path}")
    if encoder_path:
        print(f"saved encoder -> {encoder_path}")

    # ---- calibration/audit ----
    calibration_frame = cal_df if cal_df is not None and len(cal_df) else df
    threshold, cal = calibrate_threshold(
        model_for_loss.router, base.encoder, calibration_frame, (model_args, base), device,
        args.complete_recall_target, progress_every=args.calibration_progress_every)
    initial_sidecar_audit = audit_sidecar_threshold(
        model_for_loss.router, base.encoder, calibration_frame, (model_args, base), device,
        args.sidecar_confidence_threshold, progress_every=args.calibration_progress_every)
    sidecar_thresholds, sidecar_calibration = calibrate_sidecar_thresholds(
        model_for_loss.router,
        base.encoder,
        calibration_frame,
        (model_args, base),
        device,
        requested_threshold=args.sidecar_confidence_threshold,
        margin=args.sidecar_threshold_margin,
        max_per_label=args.calibration_max_per_label,
        progress_every=args.calibration_progress_every,
    )
    if args.attachment_set_decode_mode == "direct_sidecar":
        sidecar_calibration["policy"] = {
            "routing_rule": "explicit expected type else router argmax",
            "confidence_role": "diagnostic_only",
            "sidecar_to_expert0_fallback": False,
            "thresholds_affect_inference": False,
        }
    calibrated_sidecar_threshold = float(max(sidecar_thresholds[1:])) if len(sidecar_thresholds) > 1 else 1.01
    sidecar_audit = audit_sidecar_threshold(
        model_for_loss.router, base.encoder, calibration_frame, (model_args, base), device,
        sidecar_thresholds, progress_every=args.calibration_progress_every)
    print(f"legacy complete-p0 calibration: {cal} -> complete_confidence_threshold={threshold:.4f}")
    print(f"initial sidecar threshold audit: {initial_sidecar_audit}")
    print(f"sidecar threshold audit: {sidecar_audit}")

    # ---- emit ready-to-use MoE config ----
    def config_artifact_path(path):
        if not path:
            return None
        output_root = os.path.abspath(args.output_dir)
        artifact = os.path.abspath(path)
        relative = os.path.relpath(artifact, output_root)
        if relative == ".." or relative.startswith(".." + os.sep):
            return artifact
        return relative

    cfg = {
        "schema_version": (
            "molnextr_moe_native_attachment_v5"
            if args.phase2_sep
            else "molnextr_moe_direct_graph_v4"
        ),
        "num_experts": 3, "expert_names": EXPERT_NAMES,
        "expert_kind": str(args.expert_kind),
        "adapter_expert_names": ([] if args.expert_kind == "full_mixture" else EXPERT_NAMES[1:]),
        "num_adapter_experts": (0 if args.expert_kind == "full_mixture" else 2),
        "full_mixture_sidecar_mode": (
            str(args.full_mixture_sidecar_mode)
            if args.expert_kind == "full_mixture"
            else ""
        ),
        "fragment_decoder_train_scope": str(
            args.fragment_decoder_train_scope
        ),
        "sep_extension_enabled": bool(args.phase2_sep),
        "attachment_representation": (
            {
                "schema_version": "molnextr_single_attachment_extension_v1",
                "syntax": "backbone<sep>[zero_based_anchor:*]",
                "strict_failure": True,
                "fallback": False,
                "post_decode_graph_rewrite": False,
            }
            if args.phase2_sep
            else None
        ),
        "adapter_path": config_artifact_path(adapter_path),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "include_output_layer": bool(args.include_output_layer),
        "include_edges": bool(args.include_edges),
        "complete_confidence_threshold": threshold,
        "sidecar_confidence_threshold": float(calibrated_sidecar_threshold),
        "sidecar_confidence_thresholds": [float(x) for x in sidecar_thresholds],
        "routing_strategy": args.routing_strategy,
        "router_kind": str(args.router_kind),
        "token_fusion_mode": str(args.token_fusion_mode),
        "token_fusion_initial_sidecar_weight": float(
            args.token_fusion_initial_sidecar_weight
        ),
        "token_fusion_dispatch": str(args.token_fusion_dispatch),
        "token_fusion_hard_threshold": float(args.token_fusion_hard_threshold),
        "specialist_ownership_scope": str(args.specialist_ownership_scope),
        "attachment_set_enabled": bool(args.attachment_set_enabled),
        "attachment_set_hidden_dim": int(args.attachment_set_hidden_dim),
        "attachment_set_num_queries": int(args.attachment_set_num_queries),
        "attachment_set_num_layers": int(args.attachment_set_num_layers),
        "attachment_set_num_heads": int(args.attachment_set_num_heads),
        "attachment_set_max_count": int(args.attachment_set_max_count),
        "attachment_set_min_confidence": float(
            args.attachment_set_min_confidence
        ),
        "attachment_set_max_anchor_distance": float(
            args.attachment_set_max_anchor_distance
        ),
        "attachment_set_decode_mode": str(args.attachment_set_decode_mode),
        "per_bucket_decode_dispatch": False,
        "route_fragment_to_base": False,
        "sidecar_broken_decode_fallback": False,
        "valence_repair_enabled": False if args.phase2_sep else True,
        "attachment_set_feature_mode": str(args.attachment_set_feature_mode),
        "attachment_set_feature_levels": int(args.attachment_set_feature_levels),
        "attachment_set_max_feature_size": int(
            args.attachment_set_max_feature_size
        ),
        "attachment_set_min_pointer_confidence": float(
            args.attachment_set_min_pointer_confidence
        ),
        "attachment_set_heatmap_logit_scale": float(
            args.attachment_set_heatmap_logit_scale
        ),
        "attachment_set_heatmap_prior_precision": float(
            args.attachment_set_heatmap_prior_precision
        ),
        "sidecar_coordinate_context": str(args.sidecar_coordinate_context),
        "fragment_terminal_action_head_enabled": bool(
            args.fragment_terminal_action_head_enabled
        ),
        "fragment_terminal_action_head_hidden_dim": int(
            args.fragment_terminal_action_head_hidden_dim
        ),
        "fragment_structured_terminal_edge_enabled": bool(
            args.fragment_structured_terminal_edge_enabled
        ),
        "decouple_fusion_policy_optimization": bool(
            args.decouple_fusion_policy_optimization
        ),
        "one_sided_fusion_oracle": bool(args.one_sided_fusion_oracle),
        "expected_fragment_star_logit_bias": float(args.expected_fragment_star_logit_bias),
        "expected_fragment_star_budget": int(args.expected_fragment_star_budget),
        "expected_fragment_max_atoms": int(args.expected_fragment_max_atoms),
        "shared_expert0_weight": float(args.shared_expert0_weight),
        "mixture_complete_floor": float(args.mixture_complete_floor),
        "frozen_expert0": not args.unfreeze_expert0,
        "preserve_expert0": bool(args.unfreeze_expert0),
        "regime_conditioning": bool(args.regime_conditioning),
        "expert_paths": [
            config_artifact_path(expert0_path),
            config_artifact_path(expert1_path),
            config_artifact_path(expert2_path),
        ],
        "encoder_path": config_artifact_path(encoder_path),
        "encoder_finetune_stages": int(args.encoder_finetune_stages),
        "router_path": config_artifact_path(router_path),
        "use_confidence_head": bool(args.confidence_head),
        "confidence_path": config_artifact_path(confidence_path),
        "regime_path": config_artifact_path(regime_path),
        "training_contract": {
            "complete_route": (
                "joint_visual_encoder_plus_frozen_expert0_decoder_direct"
                if encoder_path
                else "frozen_encoder_plus_frozen_expert0_decoder_direct"
            ),
            "sidecar_route": "routed_expert_direct_no_fallback",
            "sidecar_accept_rule": (
                "explicit expected type or router argmax; confidence is diagnostic only"
                if args.attachment_set_decode_mode == "direct_sidecar"
                else "argmax != expert0 and top_prob >= sidecar_confidence_thresholds[argmax]"
            ),
            "task_loss_gate": "full_final_graph_symbol_eos_edge_topology",
            "graph_postprocess": "forbidden_for_direct_sidecar",
            "attachment_representation": (
                "native_backbone_sep_zero_based_anchor_single_dummy_edge_v1"
                if args.phase2_sep
                else "inline_dummy_atom"
            ),
            "attachment_extension_strict_failure": bool(args.phase2_sep),
            "coordinate_missing_task_policy": str(
                args.coordinate_missing_task_policy
            ),
            "sidecar_coordinate_context": str(
                args.sidecar_coordinate_context
            ),
            "fragment_structured_terminal_edge_enabled": bool(
                args.fragment_structured_terminal_edge_enabled
            ),
            "expert0_frozen": not args.unfreeze_expert0,
            "training_only_backbone_teacher": {
                "enabled": bool(
                    float(args.distill_complete_weight) > 0.0
                    or float(args.edge_distill_weight) > 0.0
                ),
                "token_scope": "sidecar_non_attachment_tokens",
                "edge_scope": "sidecar_non_dummy_edges",
                "token_weight": float(args.distill_complete_weight),
                "edge_weight": float(args.edge_distill_weight),
                "participates_in_inference": False,
            },
            "visual_encoder_jointly_trained": bool(encoder_path),
            "visual_encoder_finetune_stages": int(args.encoder_finetune_stages),
            "trainable_parameter_manifest": {
                "decoders": decoder_trainability,
                "optimizer": optimizer_trainability,
            },
            "threshold_calibration": sidecar_calibration,
            "source_mode": args.source_mode,
            "source_report_path": source_report_path,
            "training_dataframe": dataframe_provenance,
            "requires_production_gate": True,
            "production_source_root": str(source_root),
            "real_original_train_df": str(args.real_original_train_df),
            "real_original_train_report": str(args.real_original_train_report),
            "image_domain_counts": (
                {
                    str(key): int(value)
                    for key, value in df.get(
                        "image_domain",
                        pd.Series(["unknown"] * len(df)),
                    ).fillna("unknown").astype(str).value_counts().items()
                }
            ),
            "data_contract_version": MOE_DATA_CONTRACT_VERSION,
            "warm_start": warm_start_provenance,
            "sampling": sampling_report,
            "loss_weighting": {
                "markush_symbol_weight": float(args.markush_symbol_weight),
                "fragment_symbol_weight": float(args.fragment_symbol_weight),
                "attachment_symbol_loss_weight": float(args.attachment_symbol_loss_weight),
                "variable_identity_loss_weight": float(
                    args.variable_identity_loss_weight
                ),
                "attachment_cardinality_loss_weight": float(
                    args.attachment_cardinality_loss_weight
                ),
                "fragment_eos_weight": float(args.fragment_eos_weight),
                "fragment_terminal_dummy_margin_weight": float(
                    args.fragment_terminal_dummy_margin_weight
                ),
                "fragment_terminal_dummy_margin": float(
                    args.fragment_terminal_dummy_margin
                ),
                "fragment_premature_eos_unlikelihood_weight": float(
                    args.fragment_premature_eos_unlikelihood_weight
                ),
                "fragment_grpo_weight": float(args.fragment_grpo_weight),
                "fragment_grpo_every": int(args.fragment_grpo_every),
                "fragment_grpo_max_rows": int(args.fragment_grpo_max_rows),
                "fragment_grpo_group_size": int(
                    args.fragment_grpo_group_size
                ),
                "fragment_grpo_temperature": float(
                    args.fragment_grpo_temperature
                ),
                "fragment_grpo_top_p": float(args.fragment_grpo_top_p),
                "fragment_grpo_reference_kl_weight": float(
                    args.fragment_grpo_reference_kl_weight
                ),
                "fragment_counterfactual_dpo_weight": float(
                    args.fragment_counterfactual_dpo_weight
                ),
                "fragment_counterfactual_dpo_beta": float(
                    args.fragment_counterfactual_dpo_beta
                ),
                "fragment_counterfactual_edge_weight": float(
                    args.fragment_counterfactual_edge_weight
                ),
                "fragment_counterfactual_reward_margin": float(
                    args.fragment_counterfactual_reward_margin
                ),
                "fragment_search_policy_weight": float(
                    args.fragment_search_policy_weight
                ),
                "fragment_search_edge_weight": float(
                    args.fragment_search_edge_weight
                ),
                "fragment_search_policy_margin": float(
                    args.fragment_search_policy_margin
                ),
                "fragment_terminal_action_head_loss_weight": float(
                    args.fragment_terminal_action_head_loss_weight
                ),
                "fragment_terminal_action_head_lr": float(
                    args.fragment_terminal_action_head_lr
                ),
                "fragment_terminal_action_head_grad_clip": float(
                    args.fragment_terminal_action_head_grad_clip
                ),
                "fragment_grpo_rollout_contract": (
                    "direct_expert2_deployment_eval_mode_topology_sampling_"
                    "greedy_coordinates_encoder_detached_v1"
                ),
                "fragment_grpo_reward_contract": (
                    "0.55_backbone_plus_0.15_bonded_dummy_plus_0.10_"
                    "interaction_plus_0.20_attachment_site_exact_v2"
                ),
                "fragment_counterfactual_contract": (
                    "greedy_generated_prefix_eos_vs_terminal_dummy_completion_"
                    "reward_weighted_online_dpo_v1"
                ),
                "fragment_counterfactual_edge_contract": (
                    "exact_no_bond_or_single_bond_anchor_action_marginalization_"
                    "on_generated_terminal_dummy_v1"
                ),
                "fragment_search_policy_contract": (
                    "exact_terminal_eos_vs_bonded_dummy_graph_search_then_"
                    "pairwise_large_margin_policy_distillation_v1"
                ),
                "fragment_search_edge_contract": (
                    "exact_single_bond_anchor_search_then_structured_action_"
                    "cross_entropy_distillation_v1"
                ),
                "fragment_terminal_action_head_contract": (
                    "candidate_gated_hierarchical_eos_vs_attachment_residual_"
                    "with_detached_decoder_features_and_independent_optimizer_v1"
                ),
                "fragment_terminal_attachment_margin_contract": (
                    "native_sep_else_inline_dummy_vs_eos_teacher_forced_margin_v1"
                ),
                "fragment_structured_terminal_edge_contract": (
                    "exact_valence_constrained_terminal_dummy_single_bond_"
                    "anchor_map_no_action_when_no_legal_anchor_v1"
                ),
                "aromatic_symbol_weight": float(args.aromatic_symbol_weight),
                "unsaturated_symbol_weight": float(args.unsaturated_symbol_weight),
                "multiple_edge_weight": float(args.multiple_edge_weight),
                "aromatic_edge_weight": float(args.aromatic_edge_weight),
                "edge_valence_loss_weight": float(args.edge_valence_loss_weight),
                "sidecar_nonzero_edge_weight": float(args.sidecar_nonzero_edge_weight),
                "dummy_edge_weight": float(args.dummy_edge_weight),
                "sidecar_specialist_weight": float(args.sidecar_specialist_weight),
                "sidecar_dense_weight": float(args.sidecar_dense_weight),
                "distill_complete_weight": float(args.distill_complete_weight),
                "mixture_token_ce_weight": float(args.mixture_token_ce_weight),
                "mixture_edge_ce_weight": float(args.mixture_edge_ce_weight),
                "loss_reduction": str(args.loss_reduction),
                "token_fusion_supervision_weight": float(
                    args.token_fusion_supervision_weight
                ),
                "token_fusion_attachment_target": float(
                    args.token_fusion_attachment_target
                ),
                "token_fusion_backbone_target": float(
                    args.token_fusion_backbone_target
                ),
                "token_fusion_oracle_temperature": float(
                    args.token_fusion_oracle_temperature
                ),
                "decouple_fusion_policy_optimization": bool(
                    args.decouple_fusion_policy_optimization
                ),
                "one_sided_fusion_oracle": bool(args.one_sided_fusion_oracle),
                "token_fusion_dispatch": str(args.token_fusion_dispatch),
                "token_fusion_hard_threshold": float(
                    args.token_fusion_hard_threshold
                ),
                "specialist_ownership_scope": str(
                    args.specialist_ownership_scope
                ),
                "coordinate_missing_task_policy": str(
                    args.coordinate_missing_task_policy
                ),
                "edge_distill_weight": float(args.edge_distill_weight),
                "attachment_set_loss_weight": float(
                    args.attachment_set_loss_weight
                ),
                "attachment_set_point_loss_weight": float(
                    args.attachment_set_point_loss_weight
                ),
                "attachment_set_cardinality_loss_weight": float(
                    args.attachment_set_cardinality_loss_weight
                ),
                "attachment_set_relation_loss_weight": float(
                    args.attachment_set_relation_loss_weight
                ),
                "attachment_set_anchor_loss_weight": float(
                    args.attachment_set_anchor_loss_weight
                ),
                "attachment_set_pointer_loss_weight": float(
                    args.attachment_set_pointer_loss_weight
                ),
                "attachment_set_dummy_pointer_loss_weight": float(
                    args.attachment_set_dummy_pointer_loss_weight
                ),
                "attachment_set_heatmap_loss_weight": float(
                    args.attachment_set_heatmap_loss_weight
                ),
                "attachment_set_heatmap_cost_weight": float(
                    args.attachment_set_heatmap_cost_weight
                ),
                "attachment_set_heatmap_diversity_loss_weight": float(
                    args.attachment_set_heatmap_diversity_loss_weight
                ),
                "expert0_preserve_weight": float(args.expert0_preserve_weight),
                "anchor_l2_weight": float(args.anchor_l2_weight),
                "routed_mixture_weight": float(args.routed_mixture_weight),
                "router_margin_weight": float(args.router_margin_weight),
                "router_margin": float(args.router_margin),
                "load_balance_weight": float(args.load_balance_weight),
                "structure_weight": float(args.structure_weight),
                "fragment_oversample_wavy": bool(args.fragment_oversample_wavy),
                "fragment_wavy_weight": float(args.fragment_wavy_weight),
                "real_original_weight": float(args.real_original_weight),
                "sampling_focus_fraction": float(args.sampling_focus_fraction),
                "epoch_rows_per_label": int(args.epoch_rows_per_label),
                "auto_epoch_rows_cap": int(args.auto_epoch_rows_cap),
                "full_mixture_sidecar_mode": str(args.full_mixture_sidecar_mode),
                "fragment_decoder_train_scope": str(
                    args.fragment_decoder_train_scope
                ),
                "mixture_complete_floor": float(args.mixture_complete_floor),
            },
            "optimization": {
                "lr": float(args.lr),
                "encoder_lr": float(args.encoder_lr),
                "encoder_finetune_stages": int(args.encoder_finetune_stages),
                "expert0_lr": float(args.expert0_lr),
                "router_lr": float(args.router_lr),
                "attachment_set_lr": float(args.attachment_set_lr),
                "warmup_frac": float(args.warmup_frac),
                "min_lr_frac": float(args.min_lr_frac),
                "weight_decay": float(args.weight_decay),
                "grad_clip": float(args.grad_clip),
            },
        },
    }
    cfg_path = os.path.join(args.output_dir, "moe_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.output_dir, "train_log.txt"), "w") as f:
        f.write(
            "\n".join(log)
            + f"\ncomplete_confidence_threshold={threshold}\n"
            + f"requested_sidecar_confidence_threshold={args.sidecar_confidence_threshold}\n"
            + f"sidecar_confidence_threshold={calibrated_sidecar_threshold}\n"
            + f"sidecar_confidence_thresholds={json.dumps([float(x) for x in sidecar_thresholds])}\n"
            + f"routing_strategy={args.routing_strategy}\n"
            + f"router_kind={args.router_kind}\n"
            + f"token_fusion_mode={args.token_fusion_mode}\n"
            + f"token_fusion_initial_sidecar_weight={float(args.token_fusion_initial_sidecar_weight)}\n"
            + f"token_fusion_dispatch={args.token_fusion_dispatch}\n"
            + f"token_fusion_hard_threshold={float(args.token_fusion_hard_threshold)}\n"
            + f"specialist_ownership_scope={args.specialist_ownership_scope}\n"
            + f"coordinate_missing_task_policy={args.coordinate_missing_task_policy}\n"
            + f"sidecar_coordinate_context={args.sidecar_coordinate_context}\n"
            + f"encoder_finetune_stages={int(args.encoder_finetune_stages)}\n"
            + f"encoder_lr={float(args.encoder_lr)}\n"
            + f"loss_reduction={args.loss_reduction}\n"
            + f"expected_fragment_star_logit_bias={float(args.expected_fragment_star_logit_bias)}\n"
            + f"expected_fragment_star_budget={int(args.expected_fragment_star_budget)}\n"
            + f"shared_expert0_weight={float(args.shared_expert0_weight)}\n"
            + f"full_mixture_sidecar_mode={str(args.full_mixture_sidecar_mode)}\n"
            + f"fragment_decoder_train_scope={str(args.fragment_decoder_train_scope)}\n"
            + f"mixture_complete_floor={float(args.mixture_complete_floor)}\n"
            + f"loss_weighting={json.dumps(cfg['training_contract']['loss_weighting'])}\n"
            + f"optimization={json.dumps(cfg['training_contract']['optimization'])}\n"
            + f"cal={json.dumps(cal)}\n"
            + f"initial_sidecar_audit={json.dumps(initial_sidecar_audit)}\n"
            + f"sidecar_audit={json.dumps(sidecar_audit)}\n"
            + f"sidecar_calibration={json.dumps(sidecar_calibration)}\n"
        )
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"saved config -> {cfg_path}")
    print("\nDONE.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

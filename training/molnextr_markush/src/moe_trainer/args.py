"""Argument parsing and training-contract validation.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403
from .runtime import rank0_print  # noqa: F401,E402

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", default=str(REPO / "models" / "molnextr_best.pth"))
    p.add_argument("--output-dir", default=str(REPO / "experiments" / "moe" / "molnextr_moe_production_v1"))
    p.add_argument("--df-cache", default=str(REPO / "experiments" / "moe" / "molnextr_moe_production_v1_train_df.parquet"))
    p.add_argument("--reuse-df-cache", action=argparse.BooleanOptionalAction, default=False,
                   help="reuse --df-cache instead of rebuilding the dataframe")
    p.add_argument("--build-data-only", action="store_true",
                   help="build train/calibration dataframes and source report, then exit before model training")
    p.add_argument("--per-label", type=int, default=0,
                   help="pose-factory rows per label while rebuilding the dataframe; "
                        "does not truncate a reused cache or appended real-original rows. "
                        "Use --epoch-rows-per-label to bound training work.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--sampling-strategy", choices=["label_balanced", "natural"], default="label_balanced",
                   help="label_balanced draws equal rows per structure type each epoch; natural uses dataframe order")
    p.add_argument("--fragment-oversample-tiny", action="store_true",
                   help="size-aware oversampling within the fragment class: upweight tiny (short-SMILES) "
                        "fragments which are ~2%% of data and the main fragment over-generation failure mode, "
                        "so the fragment expert actually learns standalone tiny-fragment decoding")
    p.add_argument("--fragment-oversample-wavy", action=argparse.BooleanOptionalAction, default=True,
                   help="domain-aware oversampling within the fragment class: upweight accepted real-tight "
                        "terminal-wavy attachment renders, the production hard case for BioChemInsight "
                        "assembly. Requires dataframe metadata emitted by moe_dataset.py.")
    p.add_argument("--fragment-wavy-weight", type=float, default=12.0,
                   help="sampling weight for real-tight terminal-wavy fragment rows")
    p.add_argument("--fragment-oversample-cn", action="store_true",
                   help="upweight fragments whose attachment carbon bonds to nitrogen next "
                        "(*CN..., the dominant real-patent amine substituent pattern, ~55%% of "
                        "real wavy fragments but only ~9%% of synthetic). Corrects the C-C vs "
                        "C-N prior bias that makes the model insert a spurious extra carbon.")
    p.add_argument("--fragment-cn-weight", type=float, default=8.0,
                   help="sampling weight for attachment-carbon-to-nitrogen (*CN) fragments")
    p.add_argument("--fragment-oversample-amide", action="store_true",
                   help="upweight fragments whose attachment is an amide/ester carbonyl "
                        "(*C(=O)... backbone). Real patents have ~20%% amide substituents but "
                        "the decoder rearranges *C(=O)N into *CC(=O)N (amide ghost carbon). "
                        "Corrects the carbonyl-attachment prior.")
    p.add_argument("--fragment-amide-weight", type=float, default=6.0,
                   help="sampling weight for amide/ester (*C(=O)) attachment fragments")
    p.add_argument("--real-original-weight", type=float, default=12.0,
                   help="sampling priority for coordinate-optional original patent images")
    p.add_argument("--epoch-rows-per-label", type=int, default=0,
                   help="rows sampled per label per epoch for label_balanced; this is the "
                        "authoritative bound for smoke/probe runs. <=0 uses the coverage-aware auto policy")
    p.add_argument("--auto-epoch-rows-cap", type=int, default=35000,
                   help="auto policy cap per label; prevents a small complete set from limiting all sidecars")
    p.add_argument("--sampling-focus-fraction", type=float, default=0.10,
                   help="per-label fraction reserved for weighted real/tiny/wavy hard-domain rows")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="sidecar expert LR; Expert0 is frozen, so this should be a real fine-tuning LR")
    p.add_argument(
        "--encoder-finetune-stages",
        type=int,
        default=1,
        help="number of final Swin stages jointly optimized on complete, Markush, "
             "and fragment graph losses; 0 preserves the legacy frozen encoder",
    )
    p.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
        help="low learning rate for the selected visual encoder stages",
    )
    p.add_argument("--router-lr", type=float, default=3e-4)
    p.add_argument(
        "--attachment-set-lr",
        type=float,
        default=2e-4,
        help="learning rate for randomly initialized attachment-set heads; kept "
             "separate from the lower sidecar-decoder fine-tuning LR",
    )
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--min-lr-frac", type=float, default=0.05,
                   help="cosine schedule floor as a fraction of each param group's base LR")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # --- LoRA adapter-expert architecture (frozen-backbone MoE-LoRA) ---
    p.add_argument("--expert-kind", choices=["lora", "full_mixture"], default="full_mixture",
                   help="lora: frozen base decoder + N LoRA adapter experts (byte-identical complete). "
                        "full_mixture: frozen complete decoder (expert0) plus full-decoder sidecar(s); "
                        "the production default uses expert1=markush and expert2=fragment, each mixed "
                        "with expert0 at inference.")
    p.add_argument("--full-mixture-sidecar-mode", choices=["collapsed", "per_sidecar"], default="per_sidecar",
                   help="full_mixture architecture: collapsed keeps the legacy single expert1 for "
                        "markush+fragment; per_sidecar uses expert1=markush and expert2=fragment, "
                        "so attachment-fragment learning is not diluted by Markush syntax.")
    p.add_argument(
        "--fragment-decoder-train-scope",
        choices=["full", "output_edge", "last_cross_output_edge"],
        default="full",
        help="parameters optimized in the per-sidecar fragment decoder. output_edge "
             "trains only the atom-output and edge heads; last_cross_output_edge also "
             "trains the final decoder layer's visual cross-attention and its pre-norm. "
             "Both scopes freeze embeddings, self-attention, FFNs, earlier decoder "
             "layers, and the encoder projection.",
    )
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r (per expert)")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/rank)")
    p.add_argument("--include-output-layer", action=argparse.BooleanOptionalAction, default=True,
                   help="also adapt the atom output_layer (high leverage on token logits)")
    p.add_argument("--include-edges", action=argparse.BooleanOptionalAction, default=False,
                   help="also adapt the GraphPredictor edge MLP")
    p.add_argument("--expert-diversity-weight", type=float, default=0.0,
                   help="OMoE-style cosine penalty on the N experts' B_iA_i deltas (anti-collapse)")
    p.add_argument("--load-balance-weight", type=float, default=0.05,
                   help="Switch/GShard load-balance on the routed experts (ON by default for LoRA-MoE)")
    p.add_argument("--z-loss-weight", type=float, default=1e-3)
    p.add_argument("--structure-weight", type=float, default=0.5)
    p.add_argument("--sidecar-dense-weight", type=float, default=0.35)
    p.add_argument("--sidecar-specialist-weight", type=float, default=0.65)
    p.add_argument("--distill-complete-weight", type=float, default=0.0)
    p.add_argument("--mixture-token-ce-weight", type=float, default=0.0,
                   help="legacy mixture objective; must be zero for direct_sidecar")
    p.add_argument("--mixture-edge-ce-weight", type=float, default=0.0,
                   help="legacy edge-mixture objective; must be zero for direct_sidecar")
    p.add_argument("--loss-reduction", choices=["per_sample", "global"], default="per_sample",
                   help="per_sample prevents long structures from dominating tiny fragment/Markush rows")
    p.add_argument("--router-kind", choices=["mean_mlp", "attention_pool"], default="attention_pool",
                   help="attention_pool preserves small spatial attachment/label evidence lost by mean pooling")
    p.add_argument("--token-fusion-mode", choices=["fixed", "adaptive"], default="fixed",
                   help="adaptive learns a complete-vs-sidecar weight at every autoregressive step")
    p.add_argument("--token-fusion-initial-sidecar-weight", type=float, default=0.2)
    p.add_argument(
        "--token-fusion-dispatch",
        choices=["soft", "hard"],
        default="hard",
        help="hard dispatch gives each autoregressive token to exactly one expert; "
             "soft is retained only for legacy checkpoint reproduction",
    )
    p.add_argument(
        "--token-fusion-hard-threshold",
        type=float,
        default=0.5,
        help="calibrated sidecar gate threshold used by hard token/edge dispatch",
    )
    p.add_argument(
        "--specialist-ownership-scope",
        choices=["full", "attachment"],
        default="full",
        help="full trains the routed expert on the complete final graph; attachment "
             "is retained only for legacy residual checkpoint reproduction",
    )
    p.add_argument(
        "--coordinate-missing-task-policy",
        choices=["router_only", "masked_sequence"],
        default="masked_sequence",
        help="masked_sequence excludes unavailable x/y tokens while retaining exact "
             "symbol, EOS, dummy identity, and edge-topology supervision",
    )
    p.add_argument(
        "--sidecar-coordinate-context",
        choices=["full", "mask_y"],
        default="full",
        help="full preserves MolNexTR's pretrained coordinate causal chain. "
             "mask_y is retained only for historical checkpoint reproduction; "
             "it is not a production training or inference policy.",
    )
    p.add_argument(
        "--attachment-set-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train DETR-style routed attachment set heads on real/synthetic locations",
    )
    p.add_argument("--attachment-set-hidden-dim", type=int, default=256)
    p.add_argument("--attachment-set-num-queries", type=int, default=40)
    p.add_argument("--attachment-set-num-layers", type=int, default=3)
    p.add_argument("--attachment-set-num-heads", type=int, default=8)
    p.add_argument("--attachment-set-max-count", type=int, default=40)
    p.add_argument(
        "--attachment-set-min-confidence",
        type=float,
        default=0.1,
        help="low safety floor after cardinality-ranked query selection; DETR focal "
             "objectness is a ranking score and is not calibrated at 0.5",
    )
    p.add_argument("--attachment-set-max-anchor-distance", type=float, default=0.25)
    p.add_argument(
        "--attachment-set-feature-mode",
        choices=[
            "single_scale",
            "multiscale",
            "multiscale_pointer",
            "multiscale_pointer_heatmap",
        ],
        default="multiscale_pointer_heatmap",
    )
    p.add_argument("--attachment-set-feature-levels", type=int, default=3)
    p.add_argument("--attachment-set-max-feature-size", type=int, default=48)
    p.add_argument(
        "--attachment-set-min-pointer-confidence",
        type=float,
        default=0.15,
        help="minimum categorical decoder-atom pointer probability before a new "
             "dummy can be grafted; existing attachment evidence is never removed",
    )
    p.add_argument("--attachment-set-heatmap-logit-scale", type=float, default=4.0)
    p.add_argument(
        "--attachment-set-heatmap-prior-precision", type=float, default=8.0
    )
    p.add_argument(
        "--attachment-set-decode-mode",
        choices=["direct_sidecar", "token_mixture", "residual_base"],
        default="direct_sidecar",
    )
    p.add_argument("--attachment-set-loss-weight", type=float, default=1.0)
    p.add_argument("--attachment-set-point-loss-weight", type=float, default=5.0)
    p.add_argument("--attachment-set-cardinality-loss-weight", type=float, default=1.0)
    p.add_argument("--attachment-set-relation-loss-weight", type=float, default=1.0)
    p.add_argument("--attachment-set-anchor-loss-weight", type=float, default=5.0)
    p.add_argument("--attachment-set-pointer-loss-weight", type=float, default=3.0)
    p.add_argument("--attachment-set-dummy-pointer-loss-weight", type=float, default=3.0)
    p.add_argument("--attachment-set-heatmap-loss-weight", type=float, default=2.0)
    p.add_argument("--attachment-set-heatmap-cost-weight", type=float, default=2.0)
    p.add_argument(
        "--attachment-set-heatmap-diversity-loss-weight", type=float, default=0.5
    )
    p.add_argument(
        "--decouple-fusion-policy-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train adaptive fusion gates only with their calibrated oracle objective; "
             "mixture CE still trains specialists under the detached policy",
    )
    p.add_argument(
        "--one-sided-fusion-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="likelihood ratios may only lower backbone alpha or raise attachment/EOS alpha",
    )
    p.add_argument("--token-fusion-supervision-weight", type=float, default=0.0)
    p.add_argument("--token-fusion-attachment-target", type=float, default=0.95)
    p.add_argument("--token-fusion-backbone-target", type=float, default=0.15)
    p.add_argument("--token-fusion-oracle-temperature", type=float, default=1.0,
                   help="temperature for gold-token expert likelihood ratios used to supervise fusion")
    p.add_argument("--scst-weight", type=float, default=0.0,
                   help="SCST/REINFORCE weight (Bottleneck #1: objective misalignment). Fine-tune the "
                        "free-running mixture to MAXIMIZE the assembled-SMILES Tanimoto (the deploy "
                        "metric), greedy-decode reward as the self-critical baseline. The decoder has "
                        "otherwise only ever been trained on teacher-forced token CE. 0=off.")
    p.add_argument("--scst-every", type=int, default=20,
                   help="run an SCST step every N training steps (2 free-decodes per sidecar row)")
    p.add_argument("--scst-max-rows", type=int, default=4,
                   help="max sidecar rows decoded per SCST step (memory bound; each is an AR decode with grad)")
    p.add_argument("--scst-group", type=int, default=4,
                   help="GRPO: number of samples per sidecar row for group-normalized advantage "
                        "(r_i-mean)/std. >=2 enables GRPO (lower variance than single-sample SCST); "
                        "on near-saturated data std~0 -> near-zero gradient (no drift).")
    p.add_argument(
        "--fragment-grpo-weight",
        type=float,
        default=0.0,
        help="group-relative on-policy loss on production-identical direct fragment rollouts",
    )
    p.add_argument("--fragment-grpo-every", type=int, default=20)
    p.add_argument("--fragment-grpo-max-rows", type=int, default=2)
    p.add_argument("--fragment-grpo-group-size", type=int, default=4)
    p.add_argument("--fragment-grpo-temperature", type=float, default=0.8)
    p.add_argument("--fragment-grpo-top-p", type=float, default=0.95)
    p.add_argument(
        "--fragment-grpo-reference-kl-weight",
        type=float,
        default=0.02,
        help="generated-prefix KL on non-attachment backbone choices against frozen Expert0",
    )
    p.add_argument(
        "--fragment-counterfactual-dpo-weight",
        type=float,
        default=0.0,
        help="online preference loss between EOS and terminal-dummy completions at the greedy boundary",
    )
    p.add_argument("--fragment-counterfactual-dpo-beta", type=float, default=0.5)
    p.add_argument(
        "--fragment-counterfactual-edge-weight",
        type=float,
        default=0.0,
        help="exact expected-reward weight over terminal dummy anchor/bond actions",
    )
    p.add_argument(
        "--fragment-counterfactual-reward-margin",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--fragment-search-policy-weight",
        type=float,
        default=0.0,
        help="exact terminal-action search distillation weight at the greedy EOS prefix; "
             "search must prove a bonded dummy improves final graph reward",
    )
    p.add_argument(
        "--fragment-search-edge-weight",
        type=float,
        default=0.0,
        help="cross-entropy weight that distills the best enumerated terminal dummy "
             "anchor/bond action into the direct Expert2 edge argmax",
    )
    p.add_argument(
        "--fragment-search-policy-margin",
        type=float,
        default=2.0,
        help="target terminal-dummy minus EOS logit margin after exact graph search",
    )
    p.add_argument(
        "--fragment-terminal-action-head-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="factor Expert2 EOS-vs-terminal-dummy into a learned hierarchical "
             "action head; the base decoder and encoder are detached from its losses",
    )
    p.add_argument(
        "--fragment-terminal-action-head-hidden-dim",
        type=int,
        default=128,
    )
    p.add_argument(
        "--fragment-terminal-action-head-loss-weight",
        type=float,
        default=0.0,
        help="canonical-prefix binary action loss for the hierarchical terminal head",
    )
    p.add_argument(
        "--fragment-terminal-action-head-lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--fragment-terminal-action-head-grad-clip",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--fragment-structured-terminal-edge-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="decode a generated fragment terminal dummy with the exact "
             "valence-constrained single-bond anchor MAP",
    )
    p.add_argument("--anchor-l2-weight", type=float, default=1e-6)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument(
        "--edge-distill-weight",
        type=float,
        default=0.0,
        help="legacy Expert0 edge anchor; must be zero for direct_sidecar",
    )
    p.add_argument("--unfreeze-expert0", action="store_true",
                   help="UNFREEZE Expert 0 and joint-train it on complete+markush+fragment (breaks the "
                        "frozen-Expert0 OOD ceiling). A frozen reference decoder + functional-KL anchor "
                        "(--expert0-preserve-weight) preserve complete-molecule ability (LwF). The trained "
                        "Expert 0 is saved to moe_expert0.pth and the config points inference at it.")
    p.add_argument("--expert0-preserve-weight", type=float, default=1.0,
                   help="functional-KL weight anchoring the trainable Expert 0 to the frozen original on "
                        "complete rows (knowledge preservation; only active with --unfreeze-expert0)")
    p.add_argument("--expert0-lr", type=float, default=5e-6,
                   help="Expert 0 LR when --unfreeze-expert0 (lower than sidecars to limit forgetting)")
    p.add_argument("--expert1-lr", type=float, default=0.0,
                   help="Expert 1 (markush) LR. Default 0 = inherits --lr. Set to 0 explicitly to freeze markush decoder during fragment-only training.")
    p.add_argument("--detach-expert0-on-fragment", action="store_true",
                   help="DeepSeek-MoE shared/routed division of labor: on fragment rows Expert 0 is a "
                        "FIXED structure prior (detached, no gradient) because its complete-molecule prior "
                        "is incompatible with partial-molecule decoding (training on fragments degrades it). "
                        "The fragment routed specialist (Expert 2) does all the fragment learning. Markush "
                        "rows keep gradient (Expert 0 learns that compatible regime).")
    p.add_argument("--regime-conditioning", action="store_true",
                   help="ROOT FIX for the complete-prior/fragment conflict: add a learned per-regime bias "
                        "(complete/markush/fragment) to the encoder features feeding the decoders, so Expert 0 "
                        "is TOLD the output regime and can learn to STOP at the attachment '*' for fragments "
                        "(instead of its complete-molecule prior filling in). Standard multi-task decoder "
                        "conditioning. At inference the regime is the router's argmax prediction. Use this "
                        "INSTEAD of --detach-expert0-on-fragment (it lets Expert 0 genuinely learn fragments).")
    p.add_argument("--routed-mixture-weight", type=float, default=0.0,
                   help="CE weight on the inference-consistent w0*Expert0 + (1-w0)*matched-sidecar "
                        "mixture for sidecar rows; closes the train/inference routing gap")
    p.add_argument("--router-margin-weight", type=float, default=0.0,
                   help="multi-class hinge weight on router logits to widen the true-vs-competitor "
                        "gap, raising sidecar confidence above the accept threshold")
    p.add_argument("--router-margin", type=float, default=2.0,
                   help="target logit margin between the true expert and its strongest competitor")
    p.add_argument("--markush-symbol-weight", type=float, default=2.0,
                   help="Markush-row CE weight for dummy/R-group characters and numeric R labels")
    p.add_argument("--fragment-symbol-weight", type=float, default=1.5,
                   help="fragment-row CE weight for attachment characters; keep moderate so backbone "
                        "tokens remain a first-class objective")
    p.add_argument("--phase2-sep", action=argparse.BooleanOptionalAction, default=False,
                   help="Phase 2 (MolParser <sep>): emit a <sep>[anchor:*] suffix on the dummy-free "
                        "backbone for fragment rows (attachment metadata outside the AR trajectory). "
                        "Off by default; the trainable fragment sidecar then learns <sep> emission.")
    p.add_argument("--attachment-symbol-loss-weight", type=float, default=2.0,
                   help="auxiliary CE weight on sidecar rows at target '*' positions; this makes "
                        "explicit attachment evidence a first-class objective while avoiding the "
                        "early-training failure mode where fragment experts learn '*' but forget "
                        "the backbone")
    p.add_argument("--variable-identity-loss-weight", type=float, default=2.0,
                   help="auxiliary Markush CE on variable identity characters (for example "
                        "the 1 in [1*] or R1), preserving deterministic assembly labels")
    p.add_argument("--attachment-cardinality-loss-weight", type=float, default=0.25,
                   help="match the expected number of dummy tokens under teacher forcing")
    p.add_argument("--fragment-eos-weight", type=float, default=3.0,
                   help="upweight fragment EOS so tiny substituents stop instead of completing a molecule")
    p.add_argument(
        "--fragment-terminal-dummy-margin-weight",
        type=float,
        default=0.0,
        help="training-only pairwise ranking at the fragment attachment action: "
             "logit(<sep>) for native extensions, otherwise logit(*), must exceed "
             "logit(EOS); never applied as an inference bias",
    )
    p.add_argument(
        "--fragment-terminal-dummy-margin",
        type=float,
        default=2.0,
        help="required attachment-action-vs-EOS margin at the supervised boundary",
    )
    p.add_argument(
        "--fragment-premature-eos-unlikelihood-weight",
        type=float,
        default=0.0,
        help="training-only unlikelihood penalty on EOS before the true fragment EOS",
    )
    p.add_argument("--aromatic-symbol-weight", type=float, default=1.8,
                   help="sidecar-row CE weight for aromatic atom characters")
    p.add_argument("--unsaturated-symbol-weight", type=float, default=1.5,
                   help="sidecar-row CE weight for =/# and stereo slash characters")
    p.add_argument("--sidecar-nonzero-edge-weight", type=float, default=4.0,
                   help="sidecar-row CE weight for all real bonds, matching image-to-graph OCSR practice")
    p.add_argument("--dummy-edge-weight", type=float, default=12.0,
                   help="extra sidecar-row CE weight for bonds incident to dummy/R-group attachment atoms")
    p.add_argument("--multiple-edge-weight", type=float, default=1.5,
                   help="sidecar-row CE weight for double/triple edge classes")
    p.add_argument("--aromatic-edge-weight", type=float, default=1.8,
                   help="sidecar-row CE weight for aromatic edge class")
    p.add_argument("--edge-valence-loss-weight", type=float, default=0.0,
                   help="root-cause training penalty: a differentiable expected-valence "
                        "hinge on the edge softmax. Teaches the decoder not to put mass on "
                        "bond configurations that make an atom hypervalent (the dominant "
                        "cause of v5 invalid-SMILES failures). 0 disables; try 1.0-3.0. "
                        "Only acts on sidecar rows so the complete path stays byte-identical.")
    p.add_argument("--sidecar-confidence-threshold", type=float, default=0.5,
                   help="FLOOR for the per-sidecar accept threshold; calibration picks the data-driven "
                        "value (max complete-molecule prob on this sidecar + margin) which guarantees "
                        "zero complete->sidecar leakage while maximizing recall. Lower this only if you "
                        "want even more recall and accept re-checking leakage; the old 0.95 default "
                        "artificially capped recall (fragment accept ~66%%->93%% when lowered to ~0.6).")
    p.add_argument("--expected-fragment-star-logit-bias", type=float, default=0.0,
                   help="optional decode-time bias for '*' only when upstream expected type is fragment; "
                        "paired with a one-star budget so it cannot create repeated '*' streams. "
                        "Default 0 keeps old checkpoint behavior.")
    p.add_argument("--expected-fragment-star-budget", type=int, default=1,
                   help="maximum '*' tokens allowed during expected fragment decoding")
    p.add_argument("--expected-fragment-max-atoms", type=int, default=30,
                   help="fragment-only autoregressive atom cap; 0 disables it. This is never "
                        "applied to Markush decoding.")
    p.add_argument("--sidecar-threshold-margin", type=float, default=0.03,
                   help="safety margin above the calibration split's max complete->sidecar prob, to "
                        "absorb population variance so the zero-leakage guarantee holds beyond the "
                        "calibration sample")
    p.add_argument("--routing-strategy", choices=["soft_mixture", "sparse_top1", "shared_routed_top1"], default="soft_mixture",
                   help="inference routing written to moe_config.json; LoRA-MoE uses soft_mixture (the frozen W0 is already the always-on shared expert). shared_routed_top1 is accepted and mapped to soft_mixture.")
    p.add_argument("--shared-expert0-weight", type=float, default=0.35,
                   help="legacy no-op for old shared_routed_top1 configs; LoRA sidecar adapters always add residuals to frozen W0")
    p.add_argument("--mixture-complete-floor", type=float, default=0.0,
                   help="legacy mixture floor; must be zero for direct_sidecar")
    p.add_argument("--source-mode", choices=["available_production_pose_factory"], default="available_production_pose_factory",
                   help="discover currently available stable production pose-factory shard CSVs")
    p.add_argument("--production-source-root", default=str(DATA_ROOT / PRODUCTION_RELATIVE_ROOT),
                   help="root containing production ordinary/markush/fragment shard CSVs")
    p.add_argument(
        "--frozen-source-report",
        default="",
        help="Previously gated source-discovery report whose whole-shard train/calibration "
             "partition must be reused with --reuse-df-cache. Every referenced CSV is "
             "revalidated under --production-source-root; this is not a gate bypass.",
    )
    p.add_argument(
        "--real-original-train-df",
        default=str(
            DATA_ROOT
            / "data/generated/real_markushgrapher_ocsr_v2/train/data.parquet"
        ),
        help="pose-verified OCSR rows built from original patent image pixels",
    )
    p.add_argument(
        "--real-original-train-report",
        default=str(
            DATA_ROOT
            / "data/generated/real_markushgrapher_ocsr_v2/train/report.json"
        ),
    )
    p.add_argument(
        "--use-real-original-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--require-real-original-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail instead of continuing when the validated real-original dataset is absent",
    )
    p.add_argument("--source-min-age-seconds", type=float, default=120.0,
                   help="skip shard CSVs modified more recently than this, to avoid active writes")
    p.add_argument("--source-calibration-fraction", type=float, default=0.2,
                   help="fraction of stable shard CSVs held out for router calibration/evaluation")
    p.add_argument("--calibration-max-per-label", type=int, default=2048,
                   help="max rows per structure label for router threshold calibration; <=0 uses all calibration rows")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--complete-recall-target", type=float, default=1.0,
                   help="legacy report target; inference uses sidecar threshold")
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                   help="save expert/router checkpoints after each epoch for resumable long runs")
    p.add_argument("--calibration-progress-every", type=int, default=512,
                   help="print calibration progress every N rows; <=0 disables progress prints")
    p.add_argument("--resume-expert1", default=None)
    p.add_argument("--resume-expert2", default=None)
    p.add_argument(
        "--resume-encoder",
        default=None,
        help="resume the sparse jointly trained visual encoder checkpoint",
    )
    p.add_argument("--resume-router", default=None)
    p.add_argument(
        "--allow-attachment-set-pointer-upgrade",
        action="store_true",
        help=(
            "explicitly allow a resume router checkpoint with a multiscale "
            "attachment head to initialize a multiscale_pointer head; only "
            "the newly introduced pointer parameters may be missing"
        ),
    )
    p.add_argument(
        "--allow-attachment-set-heatmap-upgrade",
        action="store_true",
        help=(
            "explicitly allow a multiscale_pointer resume checkpoint to "
            "initialize multiscale_pointer_heatmap; only the new spatial "
            "distribution and sub-cell refinement parameters may be missing"
        ),
    )
    p.add_argument("--resume-adapter", default=None,
                   help="LoRA: resume adapter A/B params from a moe_adapter.pth checkpoint")
    p.add_argument("--confidence-head", action="store_true",
                   help="train the Protenix-style confidence head (Tanimoto distribution)")
    p.add_argument("--conf-every", type=int, default=10,
                   help="compute real-Tanimoto confidence targets every N optimizer micro-steps")
    p.add_argument("--conf-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_train_args(model_args, output_dir):
    """Populate the TrainDataset-required attrs on the checkpoint args namespace."""
    model_args.data_path = ""
    model_args.test_file = ""
    model_args.coords_file = "aux_file"        # read node_coords from df itself
    model_args.pseudo_coords = False           # coords are real (normalized_image)
    model_args.augment = True
    model_args.real_match = True   # comprehensive domain randomization (sim->real)
    model_args.mask_ratio = 0
    model_args.predict_coords = False
    model_args.save_path = output_dir
    model_args.raw_image_cache_size = 0
    model_args.clean_transform_source_arrows = []
    model_args.train_preprocess_long_edge = 0
    # Allow encoder architecture override (e.g. swin_large for stronger visual
    # features). The default 'swin_base' comes from molnextr_best.pth args.
    swin_variant = os.environ.get("MOLNEXTR_ENCODER_VARIANT", "").strip()
    if swin_variant:
        model_args.encoder = swin_variant
        rank0_print(0, f"  encoder override: {swin_variant}")
    os.makedirs(output_dir, exist_ok=True)
    return model_args


def validate_direct_sidecar_training_contract(args):
    if args.attachment_set_decode_mode != "direct_sidecar":
        return
    required = {
        "--expert-kind": (args.expert_kind, "full_mixture"),
        "--full-mixture-sidecar-mode": (
            args.full_mixture_sidecar_mode,
            "per_sidecar",
        ),
        "--specialist-ownership-scope": (
            args.specialist_ownership_scope,
            "full",
        ),
        "--coordinate-missing-task-policy": (
            args.coordinate_missing_task_policy,
            "masked_sequence",
        ),
        "--sidecar-coordinate-context": (
            args.sidecar_coordinate_context,
            "full",
        ),
        "--token-fusion-mode": (args.token_fusion_mode, "fixed"),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    zero_objectives = {
        "--mixture-token-ce-weight": args.mixture_token_ce_weight,
        "--mixture-edge-ce-weight": args.mixture_edge_ce_weight,
        "--token-fusion-supervision-weight": (
            args.token_fusion_supervision_weight
        ),
        "--mixture-complete-floor": args.mixture_complete_floor,
        "--expected-fragment-star-logit-bias": (
            args.expected_fragment_star_logit_bias
        ),
        "--fragment-premature-eos-unlikelihood-weight": (
            args.fragment_premature_eos_unlikelihood_weight
        ),
        "--scst-weight": args.scst_weight,
    }
    mismatches.extend(
        f"{name}={float(value)!r} (required 0.0)"
        for name, value in zero_objectives.items()
        if abs(float(value)) > 1.0e-12
    )
    if float(args.distill_complete_weight) < 0.0:
        mismatches.append("--distill-complete-weight must be non-negative")
    if float(args.edge_distill_weight) < 0.0:
        mismatches.append("--edge-distill-weight must be non-negative")
    if args.unfreeze_expert0:
        mismatches.append("--unfreeze-expert0 is forbidden")
    if (
        float(args.fragment_terminal_dummy_margin_weight) > 0.0
        and not args.phase2_sep
    ):
        mismatches.append(
            "--fragment-terminal-dummy-margin-weight is supported only "
            "for the native --phase2-sep attachment representation"
        )
    if mismatches:
        raise ValueError(
            "direct_sidecar is a no-mixture, frozen-Expert0 final-graph "
            "contract; invalid settings: " + "; ".join(mismatches)
        )



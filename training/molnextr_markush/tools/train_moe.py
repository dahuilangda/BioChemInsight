"""Joint LoRA-MoE trainer for MolNexTR.

This is the production-style MoE training (not an isolated classifier gate):
  * task loss on the gate-weighted mixture (token + edge CE) — the actual OCSR
    objective, learned through the gate so routing & specialization co-adapt;
  * Switch/GShard load-balancing loss  (N·Σ fᵢ·Pᵢ) — keeps sidecar adapters in use;
  * ST-MoE router z-loss (mean z²)     — router logit stability;
  * structure-type supervision         — strong prior (the data is labeled).

The complete route is the frozen original decoder with zero LoRA gate. Only the
Markush/fragment sidecar adapters and the router are optimized, so the
complete-molecule path cannot drift.

Formal two-GPU run:
  /home/dahuilangda/miniconda3/envs/llm/bin/python -m torch.distributed.run \
    --nproc_per_node=2 training/molnextr_markush/tools/train_moe.py \
    --per-label 0 --epochs 12 --batch-size 8 --grad-accum 2 \
    --lr 5e-5 --router-lr 1e-4 \
    --output-dir experiments/moe/molnextr_moe_production_v1
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.MolNexTR.model import molnextr  # noqa: E402
from utils.MolNexTR.moe import ATOM_FORMAT, EXPERT_NAMES, MoEDecoder  # noqa: E402
from utils.MolNexTR.moe_confidence import (  # noqa: E402
    confidence_loss,
    fragment_graph_reward,
    smi_tanimoto,
)
from utils.MolNexTR.dataset import TrainDataset, bms_collate  # noqa: E402
from utils.MolNexTR.chemical import convert_graph_to_smiles  # noqa: E402
from training.molnextr_markush.src.moe_dataset import (  # noqa: E402
    FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
    FRAGMENT_DECODER_LINEARIZATION_METHOD,
    MOE_DATA_CONTRACT_VERSION,
    build_moe_df,
    fragment_attachment_contract,
)
from utils.MolNexTR.tokenization import EOS_ID, atomwise_tokenizer  # noqa: E402
from utils.MolNexTR.utils import FORMAT_INFO  # noqa: E402
from training.molnextr_markush.src.moe_sources import (  # noqa: E402
    PRODUCTION_RELATIVE_ROOT,
    discover_available_production_pose_factory,
)

DATA_ROOT = Path(__file__).resolve().parents[1]


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


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def rank0_print(rank: int, *parts) -> None:
    if is_rank0(rank):
        print(*parts, flush=True)


def checkpoint_provenance(path: str | None) -> dict | None:
    if not path:
        return None
    resolved = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": resolved,
        "size_bytes": int(os.path.getsize(resolved)),
        "sha256": digest.hexdigest(),
    }


class LabelBalancedSampler(Sampler[int]):
    """Deterministic label-balanced sampler with DDP rank slicing.

    Optional per-row ``weights`` enable size-aware oversampling within a label
    (e.g. upweight tiny fragments, which are ~2% of data and the main source of
    fragment over-generation, so the fragment expert actually learns them).
    """

    def __init__(
        self,
        labels,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        rows_per_label: int = 0,
        shuffle: bool = True,
        weights=None,
        focus_fraction: float = 0.0,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.label_to_indices = {
            int(label): np.flatnonzero(self.labels == int(label)).astype(np.int64)
            for label in sorted(np.unique(self.labels).tolist())
        }
        if not self.label_to_indices:
            raise ValueError("LabelBalancedSampler got no labels")
        if int(rows_per_label or 0) > 0:
            self.rows_per_label = int(rows_per_label)
        else:
            self.rows_per_label = min(len(indices) for indices in self.label_to_indices.values())
        self.global_size = len(self.label_to_indices) * self.rows_per_label
        self.total_size = (self.global_size // self.num_replicas) * self.num_replicas
        if self.total_size <= 0:
            raise ValueError("LabelBalancedSampler total size is zero")
        self.num_samples = self.total_size // self.num_replicas
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.focus_fraction = min(0.75, max(0.0, float(focus_fraction or 0.0)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        pieces = []
        for label, indices in sorted(self.label_to_indices.items()):
            weighted = False
            w = None
            if self.weights is not None:
                w = self.weights[indices].astype(np.float64)
                w = np.where(w > 0, w, 1e-6)
                weighted = not np.allclose(w, w[0])
            focus_rows = (
                min(self.rows_per_label, int(round(self.rows_per_label * self.focus_fraction)))
                if weighted
                else 0
            )
            coverage_rows = self.rows_per_label - focus_rows
            # Treat epochs as contiguous windows over independently shuffled
            # full-corpus passes. This guarantees every row is visited before a
            # new coverage pass starts. A separate weighted, no-replacement
            # focus slice keeps real/tiny/wavy hard domains present every epoch
            # without replacing the natural-coverage stream.
            cursor = self.epoch * coverage_rows
            cycle_id, offset = divmod(cursor, len(indices))
            remaining = coverage_rows
            cycles = []
            while remaining > 0:
                rng = np.random.default_rng(
                    self.seed + label * 9176 + cycle_id * 104729
                )
                permutation = rng.permutation(indices)
                take = min(remaining, len(indices) - offset)
                cycles.append(permutation[offset:offset + take])
                remaining -= take
                cycle_id += 1
                offset = 0
            coverage = np.concatenate(cycles) if cycles else np.empty(0, dtype=np.int64)
            if focus_rows > 0:
                focus_rng = np.random.default_rng(
                    self.seed + self.epoch * 1009 + label * 9176 + 48611
                )
                available_mask = ~np.isin(indices, coverage, assume_unique=False)
                focus_pool = indices[available_mask]
                focus_weights = w[available_mask]
                if len(focus_pool) < focus_rows:
                    focus_pool = indices
                    focus_weights = w
                focus = focus_rng.choice(
                    focus_pool,
                    size=focus_rows,
                    replace=focus_rows > len(focus_pool),
                    p=focus_weights / focus_weights.sum(),
                )
                picked = np.concatenate([coverage, focus])
            else:
                picked = coverage
            pieces.append(picked)
        merged = np.concatenate(pieces)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch * 3571 + 8191)
            merged = merged[rng.permutation(len(merged))]
        merged = merged[: self.total_size]
        rank_indices = merged[self.rank:self.total_size:self.num_replicas]
        return iter(int(x) for x in rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def report(self) -> dict:
        return {
            "strategy": "label_balanced",
            "rows_per_label_per_epoch": int(self.rows_per_label),
            "global_samples_per_epoch": int(self.total_size),
            "rank_samples_per_epoch": int(self.num_samples),
            "label_counts_available": {
                str(label): int(len(indices)) for label, indices in self.label_to_indices.items()
            },
            "num_replicas": int(self.num_replicas),
            "focus_fraction": float(self.focus_fraction),
            "coverage_policy": "natural_full_corpus_windows_plus_weighted_focus_without_replacement",
        }


def auto_rows_per_label(label_counts: dict[int, int], cap: int) -> int:
    """Coverage-aware epoch size with a hard upper bound.

    Small complete sets must not limit sidecar coverage, but a large complete
    set must not silently disable ``--auto-epoch-rows-cap`` either.  The
    sidecar scale is therefore capped directly; smaller labels are sampled in
    deterministic coverage cycles by :class:`LabelBalancedSampler`.
    """
    counts = {int(label): int(count) for label, count in label_counts.items()}
    if not counts or min(counts.values()) <= 0:
        raise ValueError(f"label counts must be positive, got {counts}")
    sidecar_counts = [count for label, count in counts.items() if label > 0]
    reference = max(sidecar_counts) if sidecar_counts else max(counts.values())
    return max(1, min(reference, max(1, int(cap))))


def finite_metric(value) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def fmt_metric(value) -> str:
    value_f = finite_metric(value)
    return "n/a" if value_f is None else f"{value_f:.3f}"


def group_relative_rloo_loss(
    sequence_log_probs: list[torch.Tensor],
    rewards: list[float],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Unbiased leave-one-out group baseline without response-length scaling."""
    if len(sequence_log_probs) != len(rewards) or len(rewards) < 2:
        raise ValueError("group-relative RLOO requires matching groups of at least two")
    reward_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=sequence_log_probs[0].device,
    )
    baseline = (reward_tensor.sum() - reward_tensor) / (len(rewards) - 1)
    advantages = reward_tensor - baseline
    if float(advantages.abs().max().item()) <= 1.0e-8:
        return None, advantages
    loss = torch.stack(
        [
            -advantage.detach() * log_prob
            for advantage, log_prob in zip(advantages, sequence_log_probs)
        ]
    ).mean()
    return loss, advantages


def decoded_prediction_smiles(prediction: dict) -> str:
    if prediction.get("decode_quality_issue"):
        return ""
    atom_data = prediction.get(ATOM_FORMAT, {})
    decoded_smiles, _, _, graph_issues = convert_graph_to_smiles(
        [atom_data.get("coords") or []],
        [atom_data.get("symbols") or []],
        [prediction.get("edges") or []],
        num_workers=1,
    )
    if graph_issues and graph_issues[0]:
        return ""
    return decoded_smiles[0] if decoded_smiles else ""


def reward_weighted_counterfactual_dpo_loss(
    alternative_log_prob: torch.Tensor,
    greedy_log_prob: torch.Tensor,
    reward_delta: float,
    *,
    beta: float,
    reward_margin: float,
) -> torch.Tensor | None:
    if abs(float(reward_delta)) < float(reward_margin):
        return None
    direction = 1.0 if float(reward_delta) > 0.0 else -1.0
    preference_logit = (
        direction
        * float(beta)
        * (alternative_log_prob - greedy_log_prob)
    )
    return abs(float(reward_delta)) * F.softplus(-preference_logit)


def searched_terminal_action_loss(
    terminal_action_logits: torch.Tensor | None,
    reward_gain: float,
    *,
    reward_margin: float,
    target_logit_margin: float,
) -> torch.Tensor | None:
    """Distill a searched bonded-dummy action at the deployed EOS prefix.

    The rollout exposes exactly two logits in ``[EOS, terminal dummy]`` order.
    Search, rather than the forced token identity, decides whether the update is
    admissible.  Once the complete graph reward clears the margin, a pairwise
    large-margin logistic objective gives the boundary action direct credit
    instead of diluting it across the full alternative completion.
    """
    if float(reward_gain) < float(reward_margin):
        return None
    if (
        not isinstance(terminal_action_logits, torch.Tensor)
        or terminal_action_logits.numel() != 2
    ):
        return None
    return terminal_action_pair_margin_loss(
        terminal_action_logits,
        target_star=True,
        target_logit_margin=target_logit_margin,
    )


def terminal_action_pair_margin_loss(
    terminal_action_logits: torch.Tensor | None,
    *,
    target_star: bool,
    target_logit_margin: float,
) -> torch.Tensor | None:
    if (
        not isinstance(terminal_action_logits, torch.Tensor)
        or terminal_action_logits.numel() != 2
    ):
        return None
    pair = terminal_action_logits.reshape(2).float()
    if not bool(torch.isfinite(pair).all().detach().item()):
        return None
    preferred_margin = pair[1] - pair[0]
    if not bool(target_star):
        preferred_margin = -preferred_margin
    return F.softplus(float(target_logit_margin) - preferred_margin)


def terminal_dummy_edge_expected_reward(
    prediction: dict,
    gold_smiles: str,
    *,
    distill_best_bonded_action: bool = False,
    min_reward_gain: float | None = None,
) -> tuple[torch.Tensor | None, dict | None, dict]:
    """Exactly marginalize the terminal dummy's no-bond/anchor/bond actions."""
    atom_data = prediction.get(ATOM_FORMAT, {})
    symbols = list(atom_data.get("symbols") or [])
    coords = list(atom_data.get("coords") or [])
    edge_logits = prediction.get("_rollout_edge_logits")
    dummy_indices = [
        index for index, symbol in enumerate(symbols) if "*" in str(symbol)
    ]
    if (
        len(dummy_indices) != 1
        or not isinstance(edge_logits, torch.Tensor)
        or edge_logits.ndim != 4
        or edge_logits.size(0) != 1
        or edge_logits.size(1) != 7
        or edge_logits.size(2) != len(symbols)
        or edge_logits.size(3) != len(symbols)
    ):
        return None, None, {"candidate_count": 0, "reason": "missing_terminal_dummy_logits"}
    dummy_index = dummy_indices[0]
    anchors = [index for index in range(len(symbols)) if index != dummy_index]
    if not anchors:
        return None, None, {"candidate_count": 0, "reason": "missing_anchor"}

    directed = F.softmax(edge_logits.float()[0], dim=0).permute(1, 2, 0)
    pair_probabilities = []
    for anchor_index in anchors:
        forward = directed[dummy_index, anchor_index]
        reverse = directed[anchor_index, dummy_index]
        pair_probabilities.append(
            torch.stack(
                [
                    *[
                        0.5 * (forward[bond_type] + reverse[bond_type])
                        for bond_type in range(5)
                    ],
                    0.5 * (forward[5] + reverse[6]),
                    0.5 * (forward[6] + reverse[5]),
                ]
            ).clamp_min(1.0e-8)
        )
    pair_probabilities = torch.stack(pair_probabilities)
    no_bond_score = pair_probabilities[:, 0].log().sum()
    actions: list[tuple[int, int]] = [(-1, 0)]
    action_scores = [no_bond_score]
    for local_anchor, anchor_index in enumerate(anchors):
        bond_type = 1
        actions.append((anchor_index, bond_type))
        action_scores.append(
            no_bond_score
            - pair_probabilities[local_anchor, 0].log()
            + pair_probabilities[local_anchor, bond_type].log()
        )
    action_scores_tensor = torch.stack(action_scores)
    action_probabilities = F.softmax(action_scores_tensor, dim=0)

    raw_edges = prediction.get("edges") or []
    if len(raw_edges) != len(symbols) or any(
        len(row) != len(symbols) for row in raw_edges
    ):
        raw_edges = [[0] * len(symbols) for _ in symbols]
    candidate_edges = []
    for anchor_index, bond_type in actions:
        edges = [list(row) for row in raw_edges]
        for index in range(len(symbols)):
            edges[dummy_index][index] = 0
            edges[index][dummy_index] = 0
        if anchor_index >= 0:
            edges[dummy_index][anchor_index] = bond_type
            edges[anchor_index][dummy_index] = (
                6 if bond_type == 5 else 5 if bond_type == 6 else bond_type
            )
        candidate_edges.append(edges)

    decoded_smiles, _, _, graph_issues = convert_graph_to_smiles(
        [coords for _ in actions],
        [symbols for _ in actions],
        candidate_edges,
        num_workers=1,
    )
    metrics = [
        fragment_graph_reward(
            "" if issue else smiles,
            gold_smiles,
        )
        for smiles, issue in zip(decoded_smiles, graph_issues)
    ]
    rewards = torch.tensor(
        [float(item["reward"]) for item in metrics],
        dtype=torch.float32,
        device=action_scores_tensor.device,
    )
    best_index = int(torch.argmax(rewards).item())
    best_bonded_index = int(torch.argmax(rewards[1:]).item()) + 1
    reward_range = float((rewards.max() - rewards.min()).item())
    best_bonded_gain = float(
        (rewards[best_bonded_index] - rewards[0]).item()
    )
    audit = {
        "candidate_count": len(actions),
        "reward_range": reward_range,
        "no_bond_reward": float(rewards[0].item()),
        "best_reward": float(rewards[best_index].item()),
        "best_anchor": int(actions[best_index][0]),
        "best_bond_type": int(actions[best_index][1]),
        "best_bonded_reward": float(rewards[best_bonded_index].item()),
        "best_bonded_reward_gain": best_bonded_gain,
        "best_bonded_anchor": int(actions[best_bonded_index][0]),
        "best_bonded_bond_type": int(actions[best_bonded_index][1]),
        "expected_reward": float(
            (action_probabilities.detach() * rewards).sum().item()
        ),
    }
    if distill_best_bonded_action:
        search_policy_applied = bool(
            min_reward_gain is None
            or best_bonded_gain >= float(min_reward_gain)
        )
        audit["search_policy_target_probability"] = float(
            action_probabilities[best_bonded_index].detach().item()
        )
        audit["search_policy_applied"] = search_policy_applied
        if not search_policy_applied:
            return None, metrics[best_bonded_index], audit
        search_policy_loss = -F.log_softmax(
            action_scores_tensor.float(), dim=0
        )[best_bonded_index]
        return search_policy_loss, metrics[best_bonded_index], audit
    if reward_range <= 1.0e-8:
        return None, metrics[best_index], audit
    expected_advantage = (
        action_probabilities * (rewards - rewards[0])
    ).sum()
    return -expected_advantage, metrics[best_index], audit


def validate_dataframe_contract(
    frame: pd.DataFrame,
    *,
    context: str,
    native_attachment_extension: bool = False,
    skip_fragment_linearization: bool = False,
) -> None:
    required = {
        "file_path",
        "SMILES",
        "edges",
        "structure_type_label",
        "data_contract_version",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
    versions = set(frame["data_contract_version"].dropna().astype(str).unique().tolist())
    if versions != {MOE_DATA_CONTRACT_VERSION}:
        raise ValueError(
            f"{context} data contract mismatch: {sorted(versions)}; "
            f"expected only {MOE_DATA_CONTRACT_VERSION!r}. Rebuild the dataframe."
        )
    fragment_mask = frame["structure_type_label"].astype(int).eq(2)
    if bool(fragment_mask.any()):
        alignment_columns = {
            "atom_index_alignment_verified",
            "atom_index_alignment_method",
            "smiles_to_source_atom_indices",
            "source_dummy_atom_index",
            "smiles_dummy_atom_index",
            "atom_index_reordered",
            "decoder_smiles",
            "fragment_linearization_verified",
            "fragment_linearization_method",
            "fragment_backbone_smiles",
            "fragment_decoder_backbone_prefix",
            "decoder_to_smiles_atom_indices",
            "decoder_to_source_atom_indices",
            "chemical_smiles_dummy_atom_index",
            "decoder_dummy_atom_index",
            "fragment_dummy_is_final_token",
            "fragment_backbone_prefix_exact",
            "decoder_atom_index_reordered",
        }
        if native_attachment_extension:
            alignment_columns.add("node_coords")
        missing_alignment = sorted(alignment_columns - set(frame.columns))
        if missing_alignment:
            raise ValueError(
                f"{context} fragment rows lack atom-index alignment proof: "
                f"{missing_alignment}. Rebuild the dataframe."
            )
        fragment_rows = frame.loc[fragment_mask]
        if not fragment_rows["atom_index_alignment_verified"].fillna(False).astype(bool).all():
            raise ValueError(f"{context} contains unverified fragment atom-index alignment")
        methods = set(
            fragment_rows["atom_index_alignment_method"]
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )
        if methods != {FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD}:
            raise ValueError(
                f"{context} fragment alignment methods are {sorted(methods)}; "
                f"expected {FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD!r}"
            )
        if fragment_rows["smiles_to_source_atom_indices"].fillna("").astype(str).eq("").any():
            raise ValueError(f"{context} contains an empty fragment atom-index mapping")
        if (
            fragment_rows["source_dummy_atom_index"].fillna(-1).astype(int).lt(0).any()
            or fragment_rows["smiles_dummy_atom_index"].fillna(-1).astype(int).lt(0).any()
        ):
            raise ValueError(f"{context} contains invalid fragment dummy-index alignment")
        linearization_methods = set(
            fragment_rows["fragment_linearization_method"]
            .fillna("")
            .astype(str)
            .unique()
            .tolist()
        )
        if linearization_methods != {FRAGMENT_DECODER_LINEARIZATION_METHOD}:
            raise ValueError(
                f"{context} fragment linearization methods are "
                f"{sorted(linearization_methods)}; expected "
                f"{FRAGMENT_DECODER_LINEARIZATION_METHOD!r}"
            )
        required_true = (
            "fragment_linearization_verified",
            "fragment_dummy_is_final_token",
            "fragment_backbone_prefix_exact",
        )
        for column in required_true:
            if not fragment_rows[column].fillna(False).astype(bool).all():
                raise ValueError(f"{context} contains false fragment contract flag {column}")

        row_issues = []
        for row_index, row in fragment_rows.iterrows():
            issues = []
            decoder_smiles = str(row["decoder_smiles"] or "")
            tokens = atomwise_tokenizer(decoder_smiles)
            atom_tokens = [
                token
                for token in tokens
                if token.isalpha() or token.startswith("[") or token == "*"
            ]
            atom_count = len(atom_tokens)
            try:
                chemical_to_source = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["smiles_to_source_atom_indices"])
                    )
                ]
                decoder_to_chemical = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["decoder_to_smiles_atom_indices"])
                    )
                ]
                decoder_to_source = [
                    int(value)
                    for value in ast.literal_eval(
                        str(row["decoder_to_source_atom_indices"])
                    )
                ]
                edges = ast.literal_eval(str(row["edges"]))
                node_coords = (
                    ast.literal_eval(str(row["node_coords"]))
                    if native_attachment_extension
                    else []
                )
            except (SyntaxError, TypeError, ValueError) as exc:
                issues.append(f"unparseable_payload:{exc}")
                chemical_to_source = []
                decoder_to_chemical = []
                decoder_to_source = []
                edges = []
                node_coords = []
            if not tokens or tokens[-1] != "*":
                issues.append("dummy_is_not_final_plain_star_token")
            if atom_tokens.count("*") != 1:
                issues.append(f"decoder_dummy_count:{atom_tokens.count('*')}")
            prefix = str(row["fragment_decoder_backbone_prefix"] or "")
            if decoder_smiles != f"{prefix}*":
                issues.append("decoder_target_is_not_backbone_prefix_plus_dummy")
            if not str(row["fragment_backbone_smiles"] or ""):
                issues.append("missing_canonical_backbone_smiles")
            expected_permutation = list(range(atom_count))
            for name, values in (
                ("chemical_to_source", chemical_to_source),
                ("decoder_to_chemical", decoder_to_chemical),
                ("decoder_to_source", decoder_to_source),
            ):
                if len(values) != atom_count or sorted(values) != expected_permutation:
                    issues.append(f"{name}_is_not_permutation")
            chemical_dummy = int(row["chemical_smiles_dummy_atom_index"])
            decoder_dummy = int(row["decoder_dummy_atom_index"])
            source_dummy = int(row["source_dummy_atom_index"])
            if chemical_dummy != int(row["smiles_dummy_atom_index"]):
                issues.append("chemical_dummy_index_disagrees_with_alignment")
            if decoder_dummy != atom_count - 1:
                issues.append("decoder_dummy_is_not_last_atom")
            if decoder_to_chemical and (
                not (0 <= decoder_dummy < len(decoder_to_chemical))
                or decoder_to_chemical[decoder_dummy] != chemical_dummy
            ):
                issues.append("decoder_to_chemical_dummy_mapping_mismatch")
            if decoder_to_source and (
                not (0 <= decoder_dummy < len(decoder_to_source))
                or decoder_to_source[decoder_dummy] != source_dummy
            ):
                issues.append("decoder_to_source_dummy_mapping_mismatch")
            if (
                decoder_to_chemical
                and chemical_to_source
                and decoder_to_source
                and decoder_to_source
                != [chemical_to_source[index] for index in decoder_to_chemical]
            ):
                issues.append("composed_source_mapping_mismatch")
            attachment_ok, attachment_reason = fragment_attachment_contract(
                decoder_smiles,
                edges,
            )
            if not attachment_ok:
                issues.append(f"attachment_contract:{attachment_reason}")
            dummy_bond_types = [
                int(bond_type)
                for begin, end, bond_type in edges
                if int(begin) == decoder_dummy or int(end) == decoder_dummy
            ]
            if dummy_bond_types != [1]:
                issues.append(
                    f"terminal_dummy_bond_types:{dummy_bond_types}"
                )
            attachment_neighbors = [
                int(end) if int(begin) == decoder_dummy else int(begin)
                for begin, end, bond_type in edges
                if (
                    int(bond_type) != 0
                    and (
                        int(begin) == decoder_dummy
                        or int(end) == decoder_dummy
                    )
                )
            ]
            if native_attachment_extension and len(attachment_neighbors) != 1:
                issues.append(
                    f"native_extension_anchor_count:{len(attachment_neighbors)}"
                )
            elif native_attachment_extension:
                native_anchor = int(attachment_neighbors[0])
                if not 0 <= native_anchor < atom_count - 1:
                    issues.append(
                        f"native_extension_anchor_out_of_range:{native_anchor}"
                    )
                extension_text = f"[{native_anchor}:*]"
                native_target_length = (
                    2
                    + len(prefix)
                    + 2 * max(0, atom_count - 1)
                    + 1
                    + len(extension_text)
                )
                if native_target_length > int(
                    FORMAT_INFO["chartok_coords"]["max_len"]
                ):
                    issues.append(
                        f"native_extension_target_too_long:{native_target_length}"
                    )
            if native_attachment_extension and len(node_coords) != atom_count:
                issues.append(
                    f"decoder_coordinate_count:{len(node_coords)}!={atom_count}"
                )
            if bool(row["decoder_atom_index_reordered"]) != (
                decoder_to_chemical != expected_permutation
            ):
                issues.append("decoder_reordered_flag_mismatch")
            if issues and len(row_issues) < 10:
                row_issues.append((int(row_index), issues))
        if row_issues and not skip_fragment_linearization:
            raise ValueError(
                f"{context} contains invalid fragment decoder linearizations: "
                f"{row_issues}"
            )
    if frame["file_path"].astype(str).duplicated().any():
        raise ValueError(f"{context} contains duplicate file_path rows")
    unique_paths = frame["file_path"].astype(str).unique().tolist()
    missing_paths = [path for path in unique_paths if not os.path.isfile(path)]
    if missing_paths:
        preview = "; ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{context} contains {len(missing_paths)} missing/unmounted image "
            f"paths; refusing blank-image fallback. First paths: {preview}"
        )


def load_frozen_source_partition(
    report_path: str | Path,
    *,
    production_source_root: str | Path,
) -> tuple[dict[int, list[str]], dict[int, list[str]], dict]:
    """Load and revalidate a previously gated whole-shard source partition."""

    path = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen source report is missing: {path}")
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema_version") != "molnextr_moe_source_discovery_v1":
        raise ValueError("frozen source report schema mismatch")
    if report.get("source_mode") != "available_production_pose_factory":
        raise ValueError("frozen source report has an unsupported source mode")
    root = Path(production_source_root).resolve()
    labels = report.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("frozen source report has no label partitions")

    train: dict[int, list[str]] = {}
    calibration: dict[int, list[str]] = {}
    all_train: set[str] = set()
    all_calibration: set[str] = set()
    expected = {"complete": 0, "markush": 1, "fragment": 2}
    for bucket, expected_label in expected.items():
        partition = labels.get(bucket)
        if not isinstance(partition, dict):
            raise ValueError(f"frozen source report lacks {bucket!r}")
        if int(partition.get("label", -1)) != expected_label:
            raise ValueError(f"frozen source report label mismatch for {bucket}")
        train_paths = [str(Path(value).resolve()) for value in partition.get("train_csv_paths") or []]
        calibration_paths = [
            str(Path(value).resolve())
            for value in partition.get("calibration_csv_paths") or []
        ]
        if not train_paths or not calibration_paths:
            raise ValueError(f"frozen source report has an empty {bucket} partition")
        for source_path in [*train_paths, *calibration_paths]:
            candidate = Path(source_path)
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"frozen source CSV escapes production root: {candidate}"
                ) from exc
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"frozen source CSV is missing: {candidate}"
                )
        train[expected_label] = train_paths
        calibration[expected_label] = calibration_paths
        all_train.update(train_paths)
        all_calibration.update(calibration_paths)
    overlap = all_train & all_calibration
    if overlap:
        raise ValueError(
            f"frozen source train/calibration partitions overlap by {len(overlap)} CSVs"
        )
    validated = dict(report)
    validated["frozen_partition_contract"] = {
        "schema_version": "molnextr_frozen_source_partition_v1",
        "source_report_path": str(path),
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "production_source_root": str(root),
        "all_referenced_csvs_exist": True,
        "all_referenced_csvs_under_production_root": True,
        "train_calibration_overlap_count": 0,
        "train_csv_count": len(all_train),
        "calibration_csv_count": len(all_calibration),
    }
    return train, calibration, validated


def validate_attachment_set_capacity(
    frame: pd.DataFrame,
    *,
    num_queries: int,
    max_count: int,
) -> dict[str, int]:
    """Reject datasets that cannot be represented by the set predictor."""

    num_queries = int(num_queries)
    max_count = int(max_count)
    if num_queries <= 0:
        raise ValueError("--attachment-set-num-queries must be positive")
    if max_count <= 0:
        raise ValueError("--attachment-set-max-count must be positive")
    if max_count > num_queries:
        raise ValueError(
            "attachment-set cardinality cannot exceed the number of set queries: "
            f"max_count={max_count} num_queries={num_queries}"
        )
    if "SMILES" not in frame.columns:
        raise ValueError("attachment-set capacity audit requires the SMILES column")

    counts = frame["SMILES"].fillna("").astype(str).str.count(r"\*").astype(int)
    max_required = int(counts.max()) if len(counts) else 0
    capacity = min(num_queries, max_count)
    if max_required > capacity:
        examples = frame.loc[counts > capacity, "SMILES"].astype(str).head(3).tolist()
        raise ValueError(
            "attachment-set capacity is smaller than the training targets; "
            "refusing silent cardinality truncation: "
            f"required={max_required} capacity={capacity} examples={examples}"
        )
    return {
        "max_required": max_required,
        "num_queries": num_queries,
        "max_count": max_count,
    }


def load_real_original_training_frame(args, rank: int) -> tuple[pd.DataFrame | None, dict | None]:
    if not bool(args.use_real_original_data):
        return None, None
    data_path = Path(args.real_original_train_df)
    report_path = Path(args.real_original_train_report)
    if not data_path.exists() or not report_path.exists():
        message = (
            "validated real-original training data is missing; run "
            "tools/build_real_markushgrapher_ocsr.py first: "
            f"data={data_path} report={report_path}"
        )
        if bool(args.require_real_original_data):
            raise FileNotFoundError(message)
        rank0_print(rank, f"  WARNING: {message}")
        return None, None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError(f"real-original report is not passing: {report_path}")
    if report.get("schema_version") != "real_markushgrapher_ocsr_v2":
        raise ValueError("real-original report is not the pose-verified v2 contract")
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    if policy.get("original_image_pixels_preserved") is not True:
        raise ValueError("real-original report does not preserve source image pixels")
    if policy.get("coordinate_targets_are_source_derived") is not True:
        raise ValueError("real-original coordinates are not source-derived")
    if policy.get("training_requires_verified_pose") is not True:
        raise ValueError("real-original training rows are not pose-gated")
    if policy.get("data_contract_version") != MOE_DATA_CONTRACT_VERSION:
        raise ValueError("real-original report data contract version mismatch")
    frame = pd.read_parquet(data_path)
    validate_dataframe_contract(frame, context="real-original dataframe")
    if not set(frame["image_domain"].astype(str).unique()) <= {"real_original"}:
        raise ValueError("real-original dataframe contains a non-real image domain")
    if not frame["coordinate_targets_available"].astype(bool).all():
        raise ValueError("real-original training dataframe contains an unverified pose")
    if not frame["coordinate_pose_verified"].astype(bool).all():
        raise ValueError("real-original training dataframe failed the pose verification flag")
    if set(frame["node_coords_space"].astype(str).unique()) != {
        "normalized_image_cxsmiles_ocr_similarity"
    }:
        raise ValueError("real-original training dataframe has an invalid pose space")
    if not set(frame["structure_type_label"].astype(int).unique()) <= {1, 2}:
        raise ValueError("real-original dataframe contains complete/noise labels")
    rank0_print(
        rank,
        f"  real-original rows: {len(frame)} "
        f"labels={frame['structure_type_label'].value_counts().to_dict()}",
    )
    return frame, report


def make_summary_writer(args, rank: int):
    if not args.tensorboard or not is_rank0(rank):
        return None
    log_dir = args.tensorboard_dir or os.path.join(args.output_dir, "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requested but neither tensorboard nor tensorboardX is installed."
            ) from exc
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def linear_warmup_cosine(opt, warmup_steps, total_steps, min_lr_frac: float = 0.0):
    from torch.optim.lr_scheduler import LambdaLR
    floor = min(max(float(min_lr_frac or 0.0), 0.0), 0.95)
    def fn(step):
        if step < warmup_steps:
            return max(floor, max(1e-3, step / max(1, warmup_steps)))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
        return floor + (1.0 - floor) * cosine
    return LambdaLR(opt, lr_lambda=fn)


def save_moe_state(model_for_loss, model_args, output_dir: str, *, tag: str | None = None,
                   confidence_head: bool = False, save_expert0: bool = False,
                   expert_kind: str = "lora",
                   encoder: torch.nn.Module | None = None,
                   encoder_trainable_names: list[str] | None = None,
                   encoder_finetune_stages: int = 0) -> dict[str, str | None]:
    """Save MoE checkpoints in the format consumed by ``model.py``.

    ``expert_kind="lora"`` (default): write only the LoRA adapter params
    (``moe_adapter.pth``) + router (+ optional confidence head). The frozen base
    weights live in the LoRAMoELinear buffers and are NOT duplicated — inference
    reloads them from ``molnextr_best.pth``. Tiny checkpoints (~MB, not 3×32 MB).
    ``expert_kind="full"``: legacy 3-full-decoder-copy path.
    """
    save_dir = output_dir if tag is None else os.path.join(output_dir, "checkpoints", tag)
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = None
    selected_encoder_names = list(encoder_trainable_names or [])
    if encoder is not None and selected_encoder_names:
        encoder_state = encoder.state_dict()
        missing = [name for name in selected_encoder_names if name not in encoder_state]
        if missing:
            raise KeyError(
                f"trainable encoder parameters missing from state_dict: {missing[:5]}"
            )
        encoder_path = os.path.join(save_dir, "moe_encoder.pth")
        torch.save(
            {
                "encoder": {
                    name: encoder_state[name].detach().cpu()
                    for name in selected_encoder_names
                },
                "trainable_parameter_names": selected_encoder_names,
                "finetune_stages": int(encoder_finetune_stages),
            },
            encoder_path,
        )
    router_path = os.path.join(save_dir, "moe_router.pt")
    torch.save(
        {"router": model_for_loss.router.state_dict(),
         "feature_dim": model_args.encoder_dim,
         "num_experts": int(model_for_loss.num_experts),
         "router_kind": str(getattr(model_for_loss, "router_kind", "mean_mlp")),
         "token_fusion_mode": str(getattr(model_for_loss, "token_fusion_mode", "fixed")),
         "token_fusion_initial_sidecar_weight": float(
             getattr(model_for_loss, "token_fusion_initial_sidecar_weight", 0.2)
         ),
         "token_fusion_dispatch": str(
             getattr(model_for_loss, "token_fusion_dispatch", "soft")
         ),
         "token_fusion_hard_threshold": float(
             getattr(model_for_loss, "token_fusion_hard_threshold", 0.5)
         ),
         "specialist_ownership_scope": str(
             getattr(model_for_loss, "specialist_ownership_scope", "full")
         ),
         "attachment_set_enabled": bool(
             getattr(model_for_loss, "attachment_set_enabled", False)
         ),
         "attachment_set_feature_mode": str(
             getattr(model_for_loss, "attachment_set_feature_mode", "single_scale")
         ),
         "attachment_set_heatmap_logit_scale": float(
             getattr(model_for_loss, "attachment_set_heatmap_logit_scale", 4.0)
         ),
         "attachment_set_heatmap_prior_precision": float(
             getattr(model_for_loss, "attachment_set_heatmap_prior_precision", 8.0)
         ),
         "attachment_set_heads": (
             model_for_loss.attachment_set_heads.state_dict()
             if len(getattr(model_for_loss, "attachment_set_heads", ())) else None
         ),
         "fragment_terminal_action_head_enabled": bool(
             getattr(
                 model_for_loss,
                 "fragment_terminal_action_head_enabled",
                 False,
             )
         ),
         "fragment_terminal_action_head_hidden_dim": int(
             getattr(
                 model_for_loss,
                 "fragment_terminal_action_head_hidden_dim",
                 128,
             )
         ),
         "fragment_structured_terminal_edge_enabled": bool(
             getattr(
                 model_for_loss,
                 "fragment_structured_terminal_edge_enabled",
                 False,
             )
         ),
         "fragment_terminal_action_head": (
             model_for_loss.fragment_terminal_action_head.state_dict()
             if getattr(
                 model_for_loss,
                 "fragment_terminal_action_head",
                 None,
             ) is not None
             else None
         ),
         "decouple_fusion_policy_optimization": bool(
             getattr(model_for_loss, "decouple_fusion_policy_optimization", True)
         ),
         "one_sided_fusion_oracle": bool(
             getattr(model_for_loss, "one_sided_fusion_oracle", True)
         ),
         "token_fusion_gates": (
             model_for_loss.token_fusion_gates.state_dict()
             if len(getattr(model_for_loss, "token_fusion_gates", ())) else None
         )},
        router_path,
    )
    confidence_path = None
    if confidence_head and model_for_loss.confidence_head is not None:
        confidence_path = os.path.join(save_dir, "moe_confidence.pt")
        torch.save({"confidence_head": model_for_loss.confidence_head.state_dict()}, confidence_path)

    if expert_kind == "lora":
        from utils.MolNexTR.moe import lora_state_dict
        adapter_path = os.path.join(save_dir, "moe_adapter.pth")
        torch.save(
            {"decoder": lora_state_dict(model_for_loss._decoder),
             "lora_rank": int(model_for_loss.lora_rank),
             "lora_alpha": float(model_for_loss.lora_alpha),
             "num_experts": int(model_for_loss.num_experts),
             "num_adapter_experts": int(getattr(model_for_loss, "num_adapter_experts", 0))},
            adapter_path,
        )
        return {
            "adapter_path": adapter_path, "expert0_path": None,
            "expert1_path": None, "expert2_path": None,
            "router_path": router_path, "confidence_path": confidence_path,
            "regime_path": None, "encoder_path": encoder_path,
        }

    if expert_kind == "full_mixture":
        # expert0 is the frozen complete decoder (byte-identical base, reloaded
        # from molnextr_best.pth at inference) — skip unless debugging. In the
        # per-sidecar architecture, expert1=markush and expert2=fragment; legacy
        # collapsed runs may only have expert1.
        expert0_path = None
        if save_expert0:
            expert0_path = os.path.join(save_dir, "moe_expert0.pth")
            torch.save({"decoder": model_for_loss.experts[0].state_dict()}, expert0_path)
        expert1_path = os.path.join(save_dir, "moe_expert1.pth")
        torch.save({"decoder": model_for_loss.experts[1].state_dict()}, expert1_path)
        expert2_path = None
        if len(model_for_loss.experts) > 2:
            expert2_path = os.path.join(save_dir, "moe_expert2.pth")
            torch.save({"decoder": model_for_loss.experts[2].state_dict()}, expert2_path)
        return {
            "adapter_path": None, "expert0_path": expert0_path,
            "expert1_path": expert1_path, "expert2_path": expert2_path,
            "router_path": router_path, "confidence_path": confidence_path,
            "regime_path": None, "encoder_path": encoder_path,
        }

    # legacy full-decoder path
    expert0_path = None
    if save_expert0:
        expert0_path = os.path.join(save_dir, "moe_expert0.pth")
        torch.save({"decoder": model_for_loss.experts[0].state_dict()}, expert0_path)
    expert1_path = os.path.join(save_dir, "moe_expert1.pth")
    expert2_path = os.path.join(save_dir, "moe_expert2.pth")
    torch.save({"decoder": model_for_loss.experts[1].state_dict()}, expert1_path)
    torch.save({"decoder": model_for_loss.experts[2].state_dict()}, expert2_path)
    regime_path = None
    if getattr(model_for_loss, "regime_conditioning", False) and model_for_loss.regime_emb is not None:
        regime_path = os.path.join(save_dir, "moe_regime.pt")
        torch.save({"regime_emb": model_for_loss.regime_emb.state_dict()}, regime_path)
    return {
        "expert0_path": expert0_path, "expert1_path": expert1_path,
        "expert2_path": expert2_path, "router_path": router_path,
        "confidence_path": confidence_path, "regime_path": regime_path,
        "adapter_path": None, "encoder_path": encoder_path,
    }


def _progress(label: str, done: int, total: int, every: int) -> None:
    if every and every > 0 and (done == total or done % every == 0):
        print(f"{label}: {done}/{total}", flush=True)


def _sidecar_threshold_vector(sidecar_threshold, num_experts: int) -> np.ndarray:
    """Normalize the deployment threshold contract to one value per expert."""
    num_experts = max(1, int(num_experts))
    if isinstance(sidecar_threshold, dict):
        values = [float("inf")] + [1.01] * (num_experts - 1)
        for key, raw in sidecar_threshold.items():
            idx = int(key)
            if 0 <= idx < num_experts:
                values[idx] = float(raw)
    elif isinstance(sidecar_threshold, (list, tuple, np.ndarray)):
        raw_values = [float(value) for value in list(sidecar_threshold)]
        values = raw_values[:num_experts]
        values.extend([1.01] * (num_experts - len(values)))
    else:
        values = [float("inf")] + [float(sidecar_threshold)] * (num_experts - 1)
    values[0] = float("inf")
    return np.asarray(values, dtype=np.float64)


def _sidecar_acceptance_from_probs(probs: np.ndarray, sidecar_threshold):
    """Apply the exact inference rule: threshold is selected by argmax expert."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError(f"router probabilities must be (N, K), got {probs.shape}")
    thresholds = _sidecar_threshold_vector(sidecar_threshold, probs.shape[1])
    argmax = probs.argmax(axis=1)
    top = probs.max(axis=1)
    accepted = (argmax != 0) & (top >= thresholds[argmax])
    return accepted, argmax, top, thresholds


def calibrate_threshold(router, encoder, cal_df, tokenizer_args, device, target_recall,
                        progress_every: int = 0):
    """Legacy complete-p0 report for backward compatibility."""
    cal_complete = cal_df[cal_df["structure_type_label"] == 0].head(200)
    if len(cal_complete) == 0:
        return 0.9, {"note": "no complete calibration rows", "forced": [0, 0]}
    _labels, probability_matrix = _router_probs_for_df(
        router,
        encoder,
        cal_complete,
        tokenizer_args,
        device,
        progress_every=progress_every,
        scan_label="legacy complete-p0 calibration",
    )
    if probability_matrix.size == 0:
        return 0.9, {"note": "all calibration images unreadable", "forced": [0, 0]}
    probs = np.sort(probability_matrix[:, 0].astype(np.float64))
    keep = max(1, int(np.ceil(target_recall * len(probs))))
    threshold = float(min(max(probs[-keep], 0.5), 0.999))
    forced = int((probs >= threshold).sum())
    return threshold, {"complete_force_routed": [forced, int(len(probs))],
                       "min_complete_p0": float(probs[0]), "median_complete_p0": float(np.median(probs))}


def audit_sidecar_threshold(router, encoder, cal_df, tokenizer_args, device, sidecar_threshold,
                            progress_every: int = 0):
    """Measure complete leakage using the exact per-expert deployment rule."""
    cal_complete = cal_df[cal_df["structure_type_label"] == 0].head(512)
    if len(cal_complete) == 0:
        return {"note": "no complete calibration rows", "complete_rows": 0}
    _labels, probs = _router_probs_for_df(
        router,
        encoder,
        cal_complete,
        tokenizer_args,
        device,
        progress_every=progress_every,
        scan_label="sidecar threshold audit",
    )
    if probs.size == 0:
        return {"note": "all calibration images unreadable", "complete_rows": 0}
    accepted, argmax, _top, thresholds = _sidecar_acceptance_from_probs(
        probs,
        sidecar_threshold,
    )
    sidecar = argmax != 0
    values = probs[:, 1:].max(axis=1) if probs.shape[1] > 1 else np.zeros(len(probs))
    accepted_per_expert = {
        str(expert_idx): int((accepted & (argmax == expert_idx)).sum())
        for expert_idx in range(1, probs.shape[1])
    }
    n = int(len(probs))
    return {
        "complete_rows": int(n),
        "argmax_sidecar": int(sidecar.sum()),
        "accepted_sidecar": int(accepted.sum()),
        "accepted_sidecar_rate": float(accepted.sum() / max(n, 1)),
        "accepted_per_expert": accepted_per_expert,
        "max_sidecar_p99": float(np.quantile(values, 0.99)),
        "max_sidecar_p95": float(np.quantile(values, 0.95)),
        "max_sidecar_max": float(values.max()),
        "threshold": float(max(thresholds[1:], default=float("inf"))),
        "thresholds": [float(value) for value in thresholds],
    }


def _df_for_router_calibration(cal_df, max_per_label: int) -> "pd.DataFrame":
    if max_per_label is None or int(max_per_label) <= 0:
        return cal_df
    pieces = []
    for label in sorted(cal_df["structure_type_label"].dropna().unique()):
        pieces.append(cal_df[cal_df["structure_type_label"] == label].head(int(max_per_label)))
    if not pieces:
        return cal_df.head(0)
    return pd.concat(pieces, ignore_index=True)


def _router_probs_for_df(router, encoder, cal_df, tokenizer_args, device, max_per_label: int = 0,
                         progress_every: int = 0,
                         scan_label: str = "router threshold calibration"):
    """Return (labels, probs) for calibration rows readable by the MolNexTR transform."""
    from utils.MolNexTR.model import resize_small_image_to_long_edge
    _model_args, base = tokenizer_args
    df_eval = _df_for_router_calibration(cal_df, max_per_label)
    labels, probs = [], []
    bs = 8
    print(f"{scan_label}: scanning {len(df_eval)} rows", flush=True)
    router_was_training = bool(router.training)
    encoder_was_training = bool(encoder.training)
    router.eval()
    encoder.eval()
    try:
        for i in range(0, len(df_eval), bs):
            chunk = df_eval.iloc[i:i + bs]
            tensors, kept_labels = [], []
            for _, row in chunk.iterrows():
                import cv2
                img = cv2.imread(str(row["file_path"]))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize_small_image_to_long_edge(img, base.preprocess_long_edge)
                tensors.append(base.transform(image=img, keypoints=[])["image"])
                kept_labels.append(int(row["structure_type_label"]))
            if not tensors:
                continue
            stacked = torch.stack(tensors, 0).to(device)
            with torch.inference_mode():
                feats, _ = encoder(stacked)
                p = router(feats).softmax(-1).detach().cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(kept_labels)
            _progress(scan_label, min(i + bs, len(df_eval)),
                      len(df_eval), progress_every)
    finally:
        router.train(router_was_training)
        encoder.train(encoder_was_training)
    if not probs:
        return np.asarray([], dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def calibrate_sidecar_thresholds(
    router,
    encoder,
    cal_df,
    tokenizer_args,
    device,
    *,
    requested_threshold: float,
    margin: float,
    max_per_label: int,
    progress_every: int = 0,
):
    """Select per-sidecar thresholds with zero observed complete->sidecar accepts."""
    labels, probs = _router_probs_for_df(
        router, encoder, cal_df, tokenizer_args, device, max_per_label=max_per_label,
        progress_every=progress_every,
    )
    if probs.size == 0:
        raise RuntimeError(
            "router threshold calibration has no readable rows; fix the calibration split "
            "or file paths instead of using uncalibrated sidecar thresholds"
        )
    argmax = probs.argmax(axis=1)
    top = probs.max(axis=1)
    num_experts = probs.shape[1]
    thresholds = [1.01] + [float(requested_threshold)] * max(0, num_experts - 1)
    complete = labels == 0
    complete_max_per_expert = {}
    for expert_idx in range(1, num_experts):
        # max prob a COMPLETE molecule assigns to this sidecar when the router's
        # argmax is this sidecar (the only way a complete molecule can leak).
        # threshold > this => zero observed complete->sidecar leakage.
        risky = complete & (argmax == expert_idx)
        max_complete = float(probs[risky, expert_idx].max()) if bool(risky.any()) else 0.0
        complete_max_per_expert[expert_idx] = max_complete
        # Data-driven: just above the calibration split's complete-max, so we
        # accept every sidecar-confident sample the router can reliably identify
        # (the requested floor only protects against an under-trained router).
        calibrated = max(
            float(requested_threshold),
            float(np.nextafter(max_complete, np.inf)) + float(margin),
        )
        thresholds[expert_idx] = 1.01 if calibrated > 1.0 else float(calibrated)

    accepted = (argmax != 0) & (top >= np.asarray(thresholds, dtype=np.float32)[argmax])
    per_expert = {}
    for expert_idx in range(num_experts):
        name = EXPERT_NAMES[expert_idx] if expert_idx < len(EXPERT_NAMES) else f"expert{expert_idx}"
        pred = argmax == expert_idx
        accepted_k = accepted & pred
        true_k = labels == expert_idx
        per_expert[name] = {
            "threshold": float(thresholds[expert_idx]),
            "calibration_rows_true_label": int(true_k.sum()),
            "argmax_rows": int(pred.sum()),
            "accepted_rows": int(accepted_k.sum()),
            "accepted_true_label_rows": int((accepted_k & true_k).sum()),
            "accepted_complete_rows": int((accepted_k & complete).sum()),
            "true_label_accept_rate": float((accepted_k & true_k).sum() / max(int(true_k.sum()), 1)),
            "argmax_precision": float((pred & true_k).sum() / max(int(pred.sum()), 1)),
        }
    report = {
        "schema_version": "moe_router_sidecar_threshold_calibration_v1",
        "policy": {
            "default_route": "expert0",
            "complete_to_sidecar_max_observed": 0,
            "thresholds_selected_on_calibration_split": True,
            "accept_rule": "argmax != expert0 and top_prob >= sidecar_confidence_thresholds[argmax]",
        },
        "rows": int(len(labels)),
        "label_counts": {str(k): int((labels == k).sum()) for k in range(num_experts)},
        "requested_threshold": float(requested_threshold),
        "margin": float(margin),
        "thresholds": [float(x) for x in thresholds],
        "complete_to_sidecar_max_per_expert": {
            int(k): float(v) for k, v in complete_max_per_expert.items()
        },
        "complete_accepted_sidecar_rows": int((accepted & complete).sum()),
        "sidecar_accept_rows": int(accepted.sum()),
        "per_expert": per_expert,
    }
    return thresholds, report


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
    # persistence_mode is off on this box → first CUDA init in a fresh process
    # can transiently fail; retry a few times before falling back to CPU.
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
            # Domain-aware + size-aware oversampling: real patent fragments need
            # terminal wavy attachment evidence, while tiny fragments are rare
            # and easy for complete-molecule priors to over-generate.
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
            # Attachment-carbon-to-nitrogen (*CN) oversampling: real patents are
            # ~55% *CN amine substituents but synthetic data is only ~9%, so the
            # model over-predicts C-C and inserts a spurious carbon. Upweighting
            # *CN rows corrects this prior without regenerating data.
            cn_mask = np.zeros(len(df), dtype=bool)
            if args.fragment_oversample_cn and "fragment_backbone_smiles" in df.columns:
                bb = df["fragment_backbone_smiles"].fillna("").astype(str).to_numpy()
                # attachment carbon whose next heavy atom is N: backbone starts CN
                cn_mask = frag_mask & np.char.startswith(bb.astype("U"), "CN")
                w[cn_mask] = np.maximum(w[cn_mask], float(args.fragment_cn_weight))
            # Amide/ester oversampling: *C(=O)... backbone. The decoder frequently
            # rearranges *C(=O)N into *CC(=O)N (amide ghost carbon). Upweighting
            # corrects the carbonyl-attachment tokenization prior. RDKit
            # canonicalizes *C(=O) to backbone "O=C..." so match both forms.
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
        # Warm-start the specialist (expert1) from the frozen base, so fine-tuning
        # begins at the complete-molecule solution and only learns the markush/
        # fragment delta (anchored to complete on non-attachment tokens via KL).
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
        # Per-sidecar experts and sparse attachment relations are conditional.
        # The graph changes from batch to batch, so static_graph is invalid and
        # unused-parameter discovery is required for correct reduction.
        find_unused_parameters=True,
    ) if distributed else training_root
    unwrapped_training_root = train_model.module if distributed else train_model
    model_for_loss = (
        unwrapped_training_root.moe
        if joint_encoder_training
        else unwrapped_training_root
    )
    encoder_for_training = base.encoder

    # Separate pretrained decoder fine-tuning from randomly initialized set
    # prediction. Sharing the sidecar's low LR with the DETR-style heads leaves
    # cardinality/localization materially under-trained in bounded probes.
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
        # Separate expert1 (markush) params from expert2 (fragment) params
        # so markush can be frozen (lr=0) while fragment trains. This prevents
        # fragment training data from corrupting the markush decoder.
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
                # Equivalent to DDP.no_sync() for non-update micro-batches,
                # without wrapping the entire task/SCST loss block. The next
                # synchronized backward reduces the accumulated gradients.
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
                # Legacy frozen-encoder path remains available for checkpoint
                # reproduction. Production trains the final Swin stage inside
                # the DDP boundary below.
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
                # Real Tanimoto target: free-decode the mixture (no grad) and
                # compare to the gold molecule. This is the non-toy signal.
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
            # SCST / REINFORCE (Bottleneck #1: objective misalignment). Train the
            # free-running mixture to MAXIMIZE the assembled-SMILES Tanimoto (the
            # deploy metric) — the decoder has otherwise only ever seen teacher-
            # forced token CE. Self-critical greedy baseline; the KL-to-expert0
            # anchor (distill_complete_weight, above) prevents collapse. Complete
            # rows are forced_default -> frozen expert0, skipped.
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
                    # GRPO: sample G trajectories per row, group-normalized advantage.
                    # On near-saturated rows all samples score similarly -> std~0 ->
                    # advantage~0 -> near-zero gradient (no drift, unlike single-sample
                    # SCST). On rows with room, samples vary -> meaningful gradient.
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
                    # RAW advantage (r_i - group_mean), NOT std-normalized: std-norm
                    # would amplify the tiny reward differences on near-saturated rows
                    # (Tanimoto~0.9, std~0.01) back to unit scale -> drift heldout.
                    # Raw advantage stays small on saturated (~+-0.01 -> little drift)
                    # and large on rows with room (~+-0.3 -> real signal).
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
                # Recompute deployment-mode visual features, then detach them.
                # The on-policy auxiliary is allowed to update only Expert2;
                # teacher-forced graph losses remain responsible for the shared
                # visual encoder.
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
                                        # The terminal action is the complete
                                        # searched bonded graph, not the raw
                                        # forced-star edge argmax.
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

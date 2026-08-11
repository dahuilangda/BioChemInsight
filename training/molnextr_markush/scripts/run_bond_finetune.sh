#!/usr/bin/env bash
# Bond-focused fine-tune of the MolNexTR MoE decoder.
#
# Strengthens aromatic/multiple edge loss weights and uses the expanded
# training cache (complete + markush + real patent fragments).
#
# Usage:
#   bash training/molnextr_markush/scripts/run_bond_finetune.sh [EPOCHS] [GPUS]
#   default EPOCHS=1, GPUS=1
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/dahuilangda/miniconda3/envs/llm/bin/python}"
CKPT="${CKPT:-experiments/moe/production}"
CACHE="${CACHE:-experiments/moe/bond_finetune_cache.parquet}"
OUT="${OUT:-experiments/moe/bond_finetune_output}"
REAL_ROOT="${REAL_ROOT:-training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2}"
EPOCHS="${1:-1}"
GPUS="${2:-1}"

mkdir -p "${OUT}"

if [[ ! -f "${CACHE}" ]]; then
  echo "Building training cache..."
  "${PYTHON}" training/molnextr_markush/tools/build_expanded_bond_finetune_cache.py \
    --out-dir "$(dirname "${CACHE}")"
fi

echo "=== bond fine-tune ==="
echo "  resume: ${CKPT}  output: ${OUT}  epochs: ${EPOCHS}  GPUs: ${GPUS}"
echo "  start: $(date '+%H:%M:%S')"

ARGS=(
  training/molnextr_markush/tools/train_moe.py
  --per-label 0 --epochs "${EPOCHS}" --batch-size 8 --grad-accum 2
  --lr 1e-6 --encoder-lr 2e-7 --encoder-finetune-stages 2
  --router-lr 5e-6 --attachment-set-lr 5e-6
  --output-dir "${OUT}"
  --df-cache "${CACHE}" --reuse-df-cache
  --fragment-oversample-tiny
  --require-real-original-data
  --real-original-train-df "${REAL_ROOT}/train/data.parquet"
  --real-original-train-report "${REAL_ROOT}/train/report.json"
  --real-original-weight 12.0
  --sampling-focus-fraction 0.10 --auto-epoch-rows-cap 10000
  --expert-kind full_mixture --full-mixture-sidecar-mode per_sidecar
  --fragment-decoder-train-scope full --router-kind attention_pool
  --token-fusion-mode fixed --token-fusion-dispatch hard --token-fusion-hard-threshold 0.5
  --specialist-ownership-scope full --decouple-fusion-policy-optimization
  --one-sided-fusion-oracle --token-fusion-initial-sidecar-weight 0.2
  --token-fusion-supervision-weight 0.0 --token-fusion-attachment-target 0.95
  --token-fusion-backbone-target 0.15 --token-fusion-oracle-temperature 1.0
  --loss-reduction per_sample --mixture-complete-floor 0.0
  --mixture-token-ce-weight 0.0 --mixture-edge-ce-weight 0.0
  --edge-distill-weight 1.0 --distill-complete-weight 1.0
  --coordinate-missing-task-policy masked_sequence --sidecar-coordinate-context full
  --routing-strategy soft_mixture --sidecar-confidence-threshold 0.5
  --sidecar-threshold-margin 0.03 --router-margin-weight 0.05 --router-margin 2.0
  --attachment-set-enabled --attachment-set-hidden-dim 256 --attachment-set-num-queries 40
  --attachment-set-num-layers 3 --attachment-set-num-heads 8 --attachment-set-max-count 40
  --attachment-set-min-confidence 0.1 --attachment-set-max-anchor-distance 0.25
  --attachment-set-decode-mode direct_sidecar --attachment-set-feature-mode multiscale_pointer_heatmap
  --attachment-set-feature-levels 3 --attachment-set-max-feature-size 48
  --attachment-set-min-pointer-confidence 0.15 --attachment-set-loss-weight 1.0
  --attachment-set-point-loss-weight 5.0 --attachment-set-cardinality-loss-weight 1.0
  --attachment-set-relation-loss-weight 1.0 --attachment-set-anchor-loss-weight 5.0
  --attachment-set-pointer-loss-weight 3.0 --attachment-set-dummy-pointer-loss-weight 3.0
  --attachment-set-heatmap-loss-weight 2.0 --attachment-set-heatmap-cost-weight 2.0
  --attachment-set-heatmap-diversity-loss-weight 0.5
  --fragment-symbol-weight 2.0 --attachment-symbol-loss-weight 2.0
  --variable-identity-loss-weight 3.0 --attachment-cardinality-loss-weight 0.25
  --fragment-eos-weight 3.0 --fragment-terminal-dummy-margin-weight 1.0
  --fragment-terminal-dummy-margin 2.0 --phase2-sep
  --fragment-premature-eos-unlikelihood-weight 0.0 --fragment-grpo-weight 0.0
  --fragment-counterfactual-dpo-weight 0.0 --fragment-search-policy-weight 0.0
  --aromatic-edge-weight 4.0 --multiple-edge-weight 3.0 --edge-valence-loss-weight 1.0
  --sidecar-nonzero-edge-weight 4.0
  --resume-expert1 "${CKPT}/moe_expert1.pth" --resume-expert2 "${CKPT}/moe_expert2.pth"
  --resume-encoder "${CKPT}/moe_encoder.pth" --resume-router "${CKPT}/moe_router.pt"
)

if (( GPUS > 1 )); then
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${GPUS}" "${ARGS[@]}"
else
  "${PYTHON}" "${ARGS[@]}"
fi

echo "=== done $(date '+%H:%M:%S') ==="

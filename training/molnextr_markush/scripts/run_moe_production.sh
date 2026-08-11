#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "\nERROR: command failed at %s:%s: %s\n" "${BASH_SOURCE[0]}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

# Production orchestrator for MolNexTR MoE data generation, QC, training, and eval.
# It intentionally delegates chemistry/image generation to the checked Python tools.
# This bash file only controls sequencing, safety checks, and reports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/dahuilangda/miniconda3/envs/llm/bin/python}"
DATASET_ID="molnextr_moe_production_v1"
STAGE="qc"

ORDINARY_SHARDS="full"
MARKUSH_SHARDS="full"
FRAGMENT_SHARDS="full"

ORDINARY_ROWS=0
MARKUSH_ROWS=256
FRAGMENT_ROWS=256
SEED=2026064011
SKIP_EXISTING=1
GENERATE_JOBS=1
SOURCE_MIN_AGE_SECONDS=0
SOURCE_CALIBRATION_FRACTION=0.2
GENERATION_PIDS=()

OUTPUT_DIR="experiments/moe/molnextr_moe_production_v1"
DF_CACHE="experiments/moe/molnextr_moe_production_v1_train_df.parquet"
REPORT_DIR=""
REUSE_DF_CACHE=1
FROZEN_SOURCE_REPORT=""
TRAIN_EPOCHS=12
TRAIN_BATCH_SIZE=8
GRAD_ACCUM=2
TRAIN_LR="5e-5"
ENCODER_LR="1e-5"
ENCODER_FINETUNE_STAGES=1
ROUTER_LR="1e-4"
ATTACHMENT_SET_LR="2e-4"
DDP_GPUS=2
TRAIN_EXTRA_ARGS=()

EVAL_PER_BUCKET=150
EVAL_BATCH_SIZE=1
RUN_STANDARD_EVAL=1
RUN_REAL_MARKUSH_EVAL=1
RUN_REAL_HARD_EVAL=1
RUN_REAL_TASK_EVAL=1
REAL_HARD_EVAL_CSV="training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"
REAL_TASK_DIR="frontend/backend/data/tasks/aa7b5e81f5f64c6283f9dccbc82e484e"
REAL_ORIGINAL_ROOT="training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2"
REAL_MARKUSH_EVAL_PARQUET="${REAL_ORIGINAL_ROOT}/eval/data.parquet"

CLEAN_OLD_ARTIFACTS=0
CLEAN_FRAGMENT_SHARDS=""
CLEAN_MARKUSH_SHARDS=""
CLEAN_ORDINARY_SHARDS=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage <stage> [options]

Stages:
  generate      Generate ordinary/Markush/fragment shards and run per-shard gates.
  qc            Run strict source discovery, gate summaries, coverage, and stale-name scans.
  build-data    Build train/calibration dataframe only.
  train         Train the MoE model.
  eval          Evaluate on held-out, official patent, and BioChemInsight task crops.
  all           generate -> qc -> build-data -> train -> eval.

Important defaults:
  --dataset-id molnextr_moe_production_v1
  --ordinary-shards full     Uses every local MolGrapher parquet shard.
  --markush-shards full      Covers the complete Markush source candidate plan with per-bucket tail windows.
  --fragment-shards full     Covers the complete fragment seed table across all attachment buckets.
  --ordinary-rows 0          Full rows from each selected MolGrapher parquet.
  --markush-rows 256         Accepted rows per non-empty Markush R-count bucket window.
  --fragment-rows 256        Rows per fragment attachment bucket shard.
  --skip-existing 1
  --generate-jobs 1          Shard-level generation parallelism. Use 2-8 after a smoke pass.

Examples:
  # Generate and QC the full formal dataset, resumable.
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage generate

  # Train after data is ready.
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage train --ddp-gpus 2

  # Full run: generate data, QC, build df, train, standard+real eval.
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage all --ddp-gpus 2

  # Rebuild only fragment shards 21-48.
  bash training/molnextr_markush/scripts/run_moe_production.sh \
    --stage generate --ordinary-shards "" --markush-shards "" --fragment-shards 21-48 --skip-existing 0

  # Replace the bad Markush generation only; keeps raw/source data and the good fragment/ordinary shards.
  bash training/molnextr_markush/scripts/run_moe_production.sh \
    --stage generate --ordinary-shards "" --fragment-shards "" \
    --markush-shards full --clean-markush-shards full --skip-existing 0

Options:
  --stage VALUE
  --dataset-id VALUE
  --python PATH
  --ordinary-shards RANGE     full, range list like 0-4, or empty to skip.
  --markush-shards RANGE      full, range list like 0,2-5, or empty to skip.
  --fragment-shards RANGE     full, range list like 1-48, or empty to skip.
  --ordinary-rows N           0 means full selected parquet shard.
  --markush-rows N            Accepted rows per Markush R-count bucket.
  --fragment-rows N
  --seed N
  --skip-existing 0|1
  --generate-jobs N
  --source-min-age-seconds N
  --output-dir PATH
  --df-cache PATH
  --report-dir PATH
  --reuse-df-cache 0|1
  --frozen-source-report PATH  Revalidate and reuse a previously gated whole-shard split.
  --epochs N
  --batch-size N
  --grad-accum N
  --lr FLOAT
  --encoder-lr FLOAT
  --encoder-finetune-stages N
  --router-lr FLOAT
  --attachment-set-lr FLOAT
  --ddp-gpus N                1 uses plain python; >1 uses torch.distributed.run.
  --train-extra-arg ARG       Repeatable extra arg passed to train_moe.py.
  --eval-per-bucket N
  --eval-batch-size N
  --run-standard-eval 0|1
  --run-real-markush-eval 0|1
  --run-real-hard-eval 0|1
  --run-real-task-eval 0|1
  --real-markush-eval-parquet PATH
  --real-hard-eval-csv PATH
  --real-task-dir PATH
  --clean-old-artifacts       Remove known obsolete generated/runs artifacts, never raw/source/original.
  --clean-fragment-shards RANGE
  --clean-markush-shards RANGE   Use full to replace all generated Markush shards/contracts.
  --clean-ordinary-shards RANGE
  --dry-run                   Print commands without executing them.
  -h|--help
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_env() {
  local -a env_args=()
  while [[ "$#" -gt 0 && "$1" == *=* ]]; do
    env_args+=("$1")
    shift
  done
  printf '+'
  printf ' %q' env "${env_args[@]}" "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    env "${env_args[@]}" "$@"
  fi
}

run_shard_command() {
  local -a cmd=("$@")
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

wait_for_generation_slot() {
  local max_jobs="$1"
  local failed=0
  while (( ${#GENERATION_PIDS[@]} >= max_jobs )); do
    local pid="${GENERATION_PIDS[0]}"
    if ! wait "${pid}"; then
      failed=1
    fi
    GENERATION_PIDS=("${GENERATION_PIDS[@]:1}")
    if (( failed != 0 )); then
      return 1
    fi
  done
}

enqueue_generation_command() {
  local max_jobs="$1"
  shift
  if (( max_jobs <= 1 )) || [[ "${DRY_RUN}" == "1" ]]; then
    run_shard_command "$@"
    return 0
  fi
  wait_for_generation_slot "${max_jobs}"
  run_shard_command "$@" &
  GENERATION_PIDS+=("$!")
}

wait_for_all_generation_jobs() {
  local failed=0
  local pid
  for pid in "${GENERATION_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  GENERATION_PIDS=()
  if (( failed != 0 )); then
    die "one or more parallel generation shard jobs failed"
  fi
}

parse_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) echo 1 ;;
    0|false|FALSE|no|NO|n|N) echo 0 ;;
    *) die "invalid boolean: ${1:-}" ;;
  esac
}

parse_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ && "${1}" -ge 1 ]] || die "invalid positive integer: ${1:-}"
  echo "$1"
}

range_to_list() {
  local spec="${1:-}"
  [[ -z "${spec}" ]] && return 0
  [[ "${spec}" == "full" ]] && die "internal error: range_to_list received unresolved full spec"
  local part start end i
  IFS=',' read -ra _parts <<< "${spec}"
  for part in "${_parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == *-* ]]; then
      start="${part%%-*}"
      end="${part##*-}"
      [[ "${start}" =~ ^[0-9]+$ && "${end}" =~ ^[0-9]+$ ]] || die "invalid range: ${part}"
      (( start <= end )) || die "invalid descending range: ${part}"
      for ((i=start; i<=end; i++)); do
        printf '%s\n' "${i}"
      done
    else
      [[ "${part}" =~ ^[0-9]+$ ]] || die "invalid shard index: ${part}"
      printf '%s\n' "${part}"
    fi
  done | awk '!seen[$0]++'
}

source_inventory_json() {
  "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import json
from collections import Counter

import pyarrow.parquet as pq

root = Path("training/molnextr_markush")
molgrapher_paths = sorted((root / "data/raw/molgrapher_synthetic_300k/data").glob("*.parquet"))
molgrapher = []
for path in molgrapher_paths:
    pf = pq.ParquetFile(path)
    molgrapher.append({"path": str(path), "rows": int(pf.metadata.num_rows), "row_groups": int(pf.num_row_groups)})

markush_plan = root / "data/generated/pose_factory/molnextr_moe_production_v1_markush_source_anchor_plan/candidate_plan.csv"
markush_buckets = Counter()
markush_collections = Counter()
markush_subsets = Counter()
if markush_plan.exists():
    with markush_plan.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            markush_buckets[str(row.get("annotation_r_bucket") or "")] += 1
            markush_collections[str(row.get("source_collection") or "")] += 1
            markush_subsets[str(row.get("subset") or "")] += 1

fragment_seeds = root / "data/generated/pose_factory/molnextr_moe_production_v1_fragment_source_backbone_seeds/seeds.csv"
fragment_rows = 0
if fragment_seeds.exists():
    with fragment_seeds.open(newline="", encoding="utf-8") as handle:
        fragment_rows = sum(1 for _ in csv.DictReader(handle))

mg1 = root / "data/raw/markushgrapher-synthetic-training-source/markushgrapher-synthetic-training"
mg1_train_arrows = sorted((mg1 / "train").glob("data-*-of-00050.arrow")) if (mg1 / "train").exists() else []
mg1_test_arrows = sorted((mg1 / "test").glob("data-*-of-00006.arrow")) if (mg1 / "test").exists() else []
mg2_parquets = sorted((root / "data/raw/markushgrapher2").glob("*/*.parquet"))

print(json.dumps({
    "schema_version": "molnextr_moe_source_inventory_v1",
    "molgrapher_parquets": molgrapher,
    "molgrapher_parquet_count": len(molgrapher),
    "molgrapher_rows": sum(item["rows"] for item in molgrapher),
    "markush_candidate_plan": str(markush_plan),
    "markush_candidate_plan_rows_by_bucket": dict(sorted(markush_buckets.items())),
    "markush_candidate_plan_rows_by_source_collection": dict(sorted(markush_collections.items())),
    "markush_candidate_plan_rows_by_subset": dict(sorted(markush_subsets.items())),
    "markush_candidate_plan_rows": sum(markush_buckets.values()),
    "fragment_seed_csv": str(fragment_seeds),
    "fragment_seed_rows": int(fragment_rows),
    "markushgrapher1_train_arrow_count": len(mg1_train_arrows),
    "markushgrapher1_test_arrow_count": len(mg1_test_arrows),
    "markushgrapher2_parquet_count": len(mg2_parquets),
}, sort_keys=True))
PY
}

write_source_inventory() {
  local output="${REPORT_DIR}/source_inventory.json"
  local inventory
  inventory="$(source_inventory_json)"
  printf '%s\n' "${inventory}" > "${output}"
  printf '%s\n' "${inventory}"
  log "source inventory -> ${output}"
}

full_ordinary_shard_count() {
  source_inventory_json | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["molgrapher_parquet_count"])'
}

full_fragment_shard_count() {
  source_inventory_json | FRAGMENT_ROWS="${FRAGMENT_ROWS}" "${PYTHON}" -c '
import json
import math
import os
import sys
from training.molnextr_markush.src.molnextr_dataset_toolkit.config import FRAGMENT_TARGET_BUCKETS
data = json.load(sys.stdin)
rows = int(data["fragment_seed_rows"])
per_shard = int(os.environ["FRAGMENT_ROWS"])
if per_shard <= 0:
    raise SystemExit("FRAGMENT_ROWS must be positive for full fragment generation")
bucket_count = len(FRAGMENT_TARGET_BUCKETS)
print(max(bucket_count, bucket_count * math.ceil(rows / (bucket_count * per_shard))))
'
}

resolved_shard_list() {
  local branch="$1"
  local spec="$2"
  local count i
  [[ -z "${spec}" ]] && return 0
  if [[ "${spec}" != "full" ]]; then
    range_to_list "${spec}"
    return 0
  fi
  case "${branch}" in
    ordinary) count="$(full_ordinary_shard_count)" ;;
    fragment) count="$(full_fragment_shard_count)" ;;
    markush)
      # Markush full uses per-bucket source windows instead of numeric shard expansion.
      echo "full"
      return 0
      ;;
    *) die "unsupported branch for full shard resolution: ${branch}" ;;
  esac
  [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]] || die "no ${branch} source shards discovered"
  for ((i=0; i<count; i++)); do
    printf '%s\n' "${i}"
  done
}

ordinary_source_plan_line() {
  local shard="$1"
  local rows="$2"
  SHARD="${shard}" ROWS="${rows}" "${PYTHON}" - <<'PY'
from pathlib import Path
import os

paths = sorted(Path("training/molnextr_markush/data/raw/molgrapher_synthetic_300k/data").glob("*.parquet"))
if not paths:
    raise SystemExit("no MolGrapher parquet files found")
shard = int(os.environ["SHARD"])
rows = int(os.environ["ROWS"])
path = paths[shard % len(paths)]
start = 0 if rows <= 0 else (shard // len(paths)) * rows
print(f"{path}\t{start}")
PY
}

markush_full_plan_tsv() {
  MARKUSH_ROWS="${MARKUSH_ROWS}" "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import math
import os
from collections import Counter

from training.molnextr_markush.src.molnextr_dataset_toolkit.config import MARKUSH_R_COUNT_BUCKETS

rows_per_bucket = int(os.environ["MARKUSH_ROWS"])
if rows_per_bucket <= 0:
    raise SystemExit("MARKUSH_ROWS must be positive for full Markush generation")
candidate_multiplier = 4
candidate_chunk = rows_per_bucket * candidate_multiplier
plan = Path("training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1_markush_source_anchor_plan/candidate_plan.csv")
if not plan.exists():
    raise SystemExit(f"missing Markush candidate plan: {plan}")
counts = Counter()
with plan.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        bucket = str(row.get("annotation_r_bucket") or "")
        if bucket in MARKUSH_R_COUNT_BUCKETS:
            counts[bucket] += 1
if not counts:
    raise SystemExit("Markush candidate plan has no usable annotation_r_bucket rows")
shard_count = max(math.ceil(counts[bucket] / candidate_chunk) for bucket in MARKUSH_R_COUNT_BUCKETS)
for shard in range(shard_count):
    windows = []
    targets = []
    nonempty = False
    for bucket in MARKUSH_R_COUNT_BUCKETS:
        start = shard * candidate_chunk
        end = min((shard + 1) * candidate_chunk, int(counts[bucket]))
        if end > start:
            nonempty = True
            target = min(rows_per_bucket, end - start)
        else:
            start = int(counts[bucket])
            end = int(counts[bucket])
            target = 0
        windows.append(f"{bucket}={start}:{end}")
        targets.append(f"{bucket}={target}")
    if nonempty:
        print(f"{shard}\t{','.join(targets)}\t{','.join(windows)}")
PY
}

skip_flag() {
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    printf '%s' "--skip-existing"
  fi
}

require_repo_state() {
  [[ -d "training/molnextr_markush" ]] || die "must run inside BioChemInsight repo"
  [[ -x "${PYTHON}" ]] || die "python executable is not executable: ${PYTHON}"
  [[ -f "training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py" ]] || die "missing dataset pipeline"
  [[ -f "training/molnextr_markush/tools/train_moe.py" ]] || die "missing train_moe.py"
  [[ -f "evaluation/eval_moe.py" ]] || die "missing evaluation/eval_moe.py"
  [[ -f "models/molnextr_best.pth" ]] || die "missing base checkpoint: models/molnextr_best.pth"
}

init_paths() {
  if [[ -z "${REPORT_DIR}" ]]; then
    REPORT_DIR="${OUTPUT_DIR}/production_reports"
  fi
  mkdir -p "${REPORT_DIR}" "${OUTPUT_DIR}" "$(dirname "${DF_CACHE}")"
}

check_source_data() {
  log "checking raw/source/eval data availability"
  run_env \
    "REPORT_DIR=${REPORT_DIR}" \
    "REUSE_DF_CACHE=${REUSE_DF_CACHE}" \
    "DF_CACHE=${DF_CACHE}" \
    "FROZEN_SOURCE_REPORT=${FROZEN_SOURCE_REPORT}" \
    "PHASE2_SEP=${PHASE2_SEP:-0}" \
    "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import json
from collections import Counter
import pyarrow.parquet as pq

checks = []

def add(name: str, path: Path, ok: bool, **extra):
    rec = {"name": name, "path": str(path), "exists": path.exists(), "ok": bool(ok)}
    rec.update(extra)
    checks.append(rec)

mg1 = Path("training/molnextr_markush/data/raw/markushgrapher-synthetic-training-source/markushgrapher-synthetic-training")
train = mg1 / "train"
test = mg1 / "test"
train_arrows = sorted(train.glob("data-*-of-00050.arrow")) if train.exists() else []
test_arrows = sorted(test.glob("data-*-of-00006.arrow")) if test.exists() else []
add("markushgrapher_synthetic_train_50_arrow", train, len(train_arrows) == 50, count=len(train_arrows), expected=50)
add("markushgrapher_synthetic_test_6_arrow", test, len(test_arrows) == 6, count=len(test_arrows), expected=6)
add("markushgrapher_synthetic_dataset_dict", mg1 / "dataset_dict.json", (mg1 / "dataset_dict.json").exists())
mg2_parquets = sorted(Path("training/molnextr_markush/data/raw/markushgrapher2").glob("*/*.parquet"))
add("markushgrapher2_raw", Path("training/molnextr_markush/data/raw/markushgrapher2"), len(mg2_parquets) > 0, parquet_count=len(mg2_parquets))
markush_plan = Path("training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1_markush_source_anchor_plan/candidate_plan.csv")
markush_collections = Counter()
if markush_plan.exists():
    with markush_plan.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            markush_collections[str(row.get("source_collection") or "")] += 1
add(
    "markush_candidate_plan_uses_mg1_and_mg2_sources",
    markush_plan,
    markush_collections.get("markushgrapher_synthetic_training_source", 0) > 0
    and markush_collections.get("markushgrapher2", 0) > 0,
    source_collection_counts=dict(sorted(markush_collections.items())),
    policy="production Markush generation must include the large MG1 arrow source and MG2 parquet source",
)
molgrapher_data = Path("training/molnextr_markush/data/raw/molgrapher_synthetic_300k/data")
molgrapher_parquets = sorted(molgrapher_data.glob("*.parquet")) if molgrapher_data.exists() else []
molgrapher_rows = 0
for parquet in molgrapher_parquets:
    molgrapher_rows += int(pq.ParquetFile(parquet).metadata.num_rows)
add(
    "molgrapher_train_parquets_all_available",
    molgrapher_data,
    len(molgrapher_parquets) > 0 and molgrapher_rows > 0,
    count=len(molgrapher_parquets),
    rows=molgrapher_rows,
    policy="use every local parquet in full production mode; do not hard-code train-00000 only",
)
for name, rel in [
    ("fragment_source_seeds", "training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1_fragment_source_backbone_seeds/seeds.csv"),
    ("rgreco_fragment_eval", "training/molnextr_markush/data/rgreco_fragment_eval/eval.csv"),
    ("real_wavy_hard_eval", "training/molnextr_markush/data/real_wavy_hard_eval/eval.csv"),
    ("literature_eval", "training/molnextr_markush/data/literature_eval/eval.csv"),
    ("original_eval", "training/molnextr_markush/data/original_eval/eval.csv"),
]:
    p = Path(rel)
    add(name, p, p.exists())

failures = [rec for rec in checks if not rec["ok"]]
out = Path(__import__("os").environ["REPORT_DIR"]) / "source_data_check.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"schema_version": "molnextr_moe_source_data_check_v1", "checks": checks}, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2, ensure_ascii=False))
print(f"source data check -> {out}")
if failures:
    import os
    frozen_report = os.environ.get("FROZEN_SOURCE_REPORT", "").strip()
    frozen_cache = os.environ.get("DF_CACHE", "").strip()
    allowed_frozen_failures = {
        "markush_candidate_plan_uses_mg1_and_mg2_sources",
        "fragment_source_seeds",
    }
    failure_names = {str(item["name"]) for item in failures}
    if (
        os.environ.get("REUSE_DF_CACHE") == "1"
        and frozen_report
        and frozen_cache
        and failure_names <= allowed_frozen_failures
    ):
        import pandas as pd
        from training.molnextr_markush.tools.train_moe import (
            load_frozen_source_partition,
            validate_dataframe_contract,
        )
        load_frozen_source_partition(
            frozen_report,
            production_source_root=(
                "training/molnextr_markush/data/generated/pose_factory/"
                "molnextr_moe_production_v1"
            ),
        )
        frozen_frame = pd.read_parquet(frozen_cache)
        validate_dataframe_contract(
            frozen_frame,
            context="frozen production training dataframe",
            native_attachment_extension=os.environ.get("PHASE2_SEP") == "1",
        )
        print(
            "generation-plan files are absent, but the frozen dataframe and "
            "previously gated whole-shard partition were fully revalidated"
        )
    else:
        raise SystemExit("source data check failed")
PY
  write_source_inventory
}

clean_known_old_artifacts() {
  [[ "${CLEAN_OLD_ARTIFACTS}" == "1" ]] || return 0
  log "cleaning known obsolete generated/run artifacts; raw/source/original data are not touched"
  run rm -rf \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards000_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards001_004_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards005_014_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards015_024_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards025_034_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards035_044_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards045_054_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards055_064_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_mg1_topup_strict_v1_schedule_shards065_069_summary.json \
    training/molnextr_markush/data/generated/pose_factory/markush_capacity_strict_v1_schedule.json \
    training/molnextr_markush/data/generated/pose_factory/markush_anchorfilter_batch_s039_summary.json \
    training/molnextr_markush/data/generated/pose_factory/fragment_strict_orientation_topup13_052_057_source_leak.json \
    training/molnextr_markush/data/generated/pose_factory/markush_v4_anchorfilter_dataset_correctness_review.json \
    training/molnextr_markush/data/generated/pose_factory/markush_v4_dataset_correctness_and_assembly_readiness_audit.json \
    training/molnextr_markush/runs/sidecar_contract \
    training/molnextr_markush/runs/measured
}

clean_selected_shards() {
  local branch="$1"
  local spec="$2"
  local shard
  [[ -z "${spec}" ]] && return 0
  log "cleaning selected ${branch} shards: ${spec}"
  if [[ "${spec}" == "full" ]]; then
    case "${branch}" in
      ordinary|markush|fragment) ;;
      *) die "unsupported branch for full clean: ${branch}" ;;
    esac
    run rm -rf \
      "training/molnextr_markush/data/generated/pose_factory/${DATASET_ID}/${branch}" \
      "training/molnextr_markush/runs/${DATASET_ID}_contracts/${branch}"
    return 0
  fi
  while IFS= read -r shard; do
    [[ -z "${shard}" ]] && continue
    printf -v shard_dir "s%03d" "${shard}"
    run rm -rf \
      "training/molnextr_markush/data/generated/pose_factory/${DATASET_ID}/${branch}/${shard_dir}" \
      "training/molnextr_markush/runs/${DATASET_ID}_contracts/${branch}/${shard_dir}"
  done < <(range_to_list "${spec}")
}

generate_branch() {
  local branch="$1"
  local spec="$2"
  local rows="$3"
  local shard extra_skip plan_line parquet_path source_start markush_targets markush_windows
  [[ -z "${spec}" ]] && return 0
  log "generating ${branch} shards: ${spec}"
  GENERATION_PIDS=()
  if [[ "${branch}" == "markush" && "${spec}" == "full" ]]; then
    while IFS=$'\t' read -r shard markush_targets markush_windows; do
      [[ -z "${shard}" ]] && continue
      extra_skip="$(skip_flag)"
      enqueue_generation_command "${GENERATE_JOBS}" "${PYTHON}" training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py \
        --dataset-id "${DATASET_ID}" \
        --branch markush \
        --start-shard "${shard}" \
        --shards 1 \
        --markush-rows "${rows}" \
        --markush-candidate-multiplier 4 \
        --markush-renderer-seed-retries 3 \
        --markush-bucket-targets "${markush_targets}" \
        --markush-bucket-window-overrides "${markush_windows}" \
        --execute \
        ${extra_skip:+"${extra_skip}"}
    done < <(markush_full_plan_tsv)
    wait_for_all_generation_jobs
    return 0
  fi
  while IFS= read -r shard; do
    [[ -z "${shard}" ]] && continue
    extra_skip="$(skip_flag)"
    case "${branch}" in
      ordinary)
        plan_line="$(ordinary_source_plan_line "${shard}" "${rows}")"
        parquet_path="${plan_line%%$'\t'*}"
        source_start="${plan_line##*$'\t'}"
        enqueue_generation_command "${GENERATE_JOBS}" "${PYTHON}" training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py \
          --dataset-id "${DATASET_ID}" \
          --branch ordinary \
          --start-shard "${shard}" \
          --shards 1 \
          --ordinary-rows "${rows}" \
          --ordinary-parquet "${parquet_path}" \
          --ordinary-source-start-row "${source_start}" \
          --execute \
          ${extra_skip:+"${extra_skip}"}
        ;;
      markush)
        enqueue_generation_command "${GENERATE_JOBS}" "${PYTHON}" training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py \
          --dataset-id "${DATASET_ID}" \
          --branch markush \
          --start-shard "${shard}" \
          --shards 1 \
          --markush-rows "${rows}" \
          --markush-renderer-seed-retries 3 \
          --execute \
          ${extra_skip:+"${extra_skip}"}
        ;;
      fragment)
        enqueue_generation_command "${GENERATE_JOBS}" "${PYTHON}" training/molnextr_markush/tools/run_molnextr_dataset_pipeline.py \
          --dataset-id "${DATASET_ID}" \
          --branch fragment \
          --start-shard "${shard}" \
          --shards 1 \
          --fragment-rows "${rows}" \
          --seed "${SEED}" \
          --execute \
          ${extra_skip:+"${extra_skip}"}
        ;;
      *)
        die "unsupported branch: ${branch}"
        ;;
    esac
  done < <(resolved_shard_list "${branch}" "${spec}")
  wait_for_all_generation_jobs
}

generate_data() {
  check_source_data
  clean_known_old_artifacts
  clean_selected_shards ordinary "${CLEAN_ORDINARY_SHARDS}"
  clean_selected_shards markush "${CLEAN_MARKUSH_SHARDS}"
  clean_selected_shards fragment "${CLEAN_FRAGMENT_SHARDS}"
  generate_branch ordinary "${ORDINARY_SHARDS}" "${ORDINARY_ROWS}"
  generate_branch markush "${MARKUSH_SHARDS}" "${MARKUSH_ROWS}"
  generate_branch fragment "${FRAGMENT_SHARDS}" "${FRAGMENT_ROWS}"
}

qc_report() {
  log "running strict production source discovery and gate summaries"
  run_env \
    "DATASET_ID=${DATASET_ID}" \
    "SOURCE_MIN_AGE_SECONDS=${SOURCE_MIN_AGE_SECONDS}" \
    "SOURCE_CALIBRATION_FRACTION=${SOURCE_CALIBRATION_FRACTION}" \
    "REPORT_DIR=${REPORT_DIR}" \
    "FROZEN_SOURCE_REPORT=${FROZEN_SOURCE_REPORT}" \
    "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import json
import os
import sys
from collections import Counter

from training.molnextr_markush.src.moe_sources import (
    PRODUCTION_RELATIVE_ROOT,
    discover_available_production_pose_factory,
)
from training.molnextr_markush.tools.train_moe import load_frozen_source_partition

DATASET_ID = os.environ["DATASET_ID"]
SOURCE_MIN_AGE_SECONDS = float(os.environ["SOURCE_MIN_AGE_SECONDS"])
SOURCE_CALIBRATION_FRACTION = float(os.environ["SOURCE_CALIBRATION_FRACTION"])
REPORT_DIR = Path(os.environ["REPORT_DIR"])
root = Path("training/molnextr_markush") / PRODUCTION_RELATIVE_ROOT
frozen_report = os.environ.get("FROZEN_SOURCE_REPORT", "").strip()
if frozen_report:
    train, cal, report = load_frozen_source_partition(
        frozen_report,
        production_source_root=root,
    )
else:
    train, cal, report = discover_available_production_pose_factory(
        root,
        min_age_seconds=SOURCE_MIN_AGE_SECONDS,
        calibration_fraction=SOURCE_CALIBRATION_FRACTION,
        seed=20260630,
    )
print("\n[source discovery]", flush=True)
source_discovery_console = {}
for label_name, label_report in report["labels"].items():
    source_discovery_console[label_name] = {
        key: label_report.get(key)
        for key in [
            "label",
            "pattern",
            "candidate_csvs",
            "stable_csvs",
            "train_csvs",
            "calibration_csvs",
        ]
        if key in label_report
    }
    source_discovery_console[label_name]["skipped_csvs"] = len(label_report.get("skipped_csvs") or [])
print(json.dumps({"train": {k: len(v) for k, v in train.items()}, "cal": {k: len(v) for k, v in cal.items()}}, indent=2), flush=True)
print(json.dumps(source_discovery_console, indent=2, ensure_ascii=False), flush=True)

contract_root = Path("training/molnextr_markush/runs") / f"{DATASET_ID}_contracts"
gate_specs = {
    "complete": ("ordinary/s*", [("validation.json", "trainable"), ("ordinary_molnextr_quality_contract.json", "passed")]),
    "markush": ("markush/s*", [
        ("accepted_validation.json", "trainable"),
        ("accepted_formal_nonlinear_warp_contract.json", "passed"),
        ("accepted_pose_alignment.json", "passed"),
        ("markush_substitution_anchor_contract.json", "passed"),
    ]),
    "fragment": ("fragment/s*", [
        ("validation.json", "trainable"),
        ("attachment_visual_contract.json", "passed"),
        ("fragment_formal_nonlinear_warp_contract.json", "passed"),
    ]),
}
gate_summary = {}
gate_counts = {}
failures = []
for branch, (globpat, gates) in gate_specs.items():
    rows = []
    for d in sorted(contract_root.glob(globpat)):
        rec = {"shard": d.name}
        ok = True
        for fn, key in gates:
            p = d / fn
            if not p.exists():
                rec[fn] = "missing"
                ok = False
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            val = data.get(key)
            rec[fn] = val
            rec[fn + "_row_count"] = data.get("row_count")
            rec[fn + "_blockers"] = data.get("blockers", [])
            if val is not True:
                ok = False
        rec["all_passed"] = ok
        if not ok:
            failures.append({"branch": branch, **rec})
        rows.append(rec)
    gate_summary[branch] = rows
    gate_counts[branch] = {
        "shards": len(rows),
        "passed": sum(1 for row in rows if row.get("all_passed") is True),
        "failed": sum(1 for row in rows if row.get("all_passed") is not True),
    }
print("\n[gate summary]", flush=True)
print(json.dumps({"counts": gate_counts, "failures": failures}, indent=2, ensure_ascii=False), flush=True)

csv.field_size_limit(sys.maxsize)
fragment_root = root / "fragment"
fragment_rows = 0
fragment_shards = set()
fragment_counter_keys = [
    "attachment_render_mode",
    "attachment_anchor",
    "attachment_anchor_depiction_mode",
    "attachment_anchor_label_is_visible_text",
    "attachment_direction",
    "attachment_render_geometry",
    "render_style",
    "semantic_family",
]
fragment_counters = {key: Counter() for key in fragment_counter_keys}
for csv_path in sorted(fragment_root.glob("s*/attachment_fragment_positive.csv")):
    shard = csv_path.parent.name
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            q = json.loads(row.get("render_quality") or "{}")
            fragment_rows += 1
            fragment_shards.add(shard)
            for key in fragment_counter_keys:
                fragment_counters[key][str((q.get(key) if q.get(key) not in (None, "") else row.get(key, "")))] += 1
print("\n[fragment coverage]", flush=True)
print("rows", fragment_rows, "shards", len(fragment_shards), flush=True)
for key in fragment_counter_keys:
    counter = fragment_counters[key]
    print("\n" + key)
    for name, count in sorted(counter.items()):
        print(name, count)
    sys.stdout.flush()

markush_root = root / "markush"
markush_counts = Counter()
markush_token_issues = Counter()
markush_rows = 0
for csv_path in sorted(markush_root.glob("s*/accepted_candidate/markush_layout_positive.csv")):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            markush_rows += 1
            q = json.loads(row.get("render_quality") or "{}")
            rq = q.get("markush") or {}
            graph = q.get("graph_consistency") if isinstance(q.get("graph_consistency"), dict) else {}
            atom_coordinates = q.get("atom_coordinates") if isinstance(q.get("atom_coordinates"), list) else []
            ocr_cells = rq.get("ocr_cells") if isinstance(rq.get("ocr_cells"), list) else []
            markush_counts["rows_with_ocr_cells"] += int(bool(rq.get("ocr_cells")))
            markush_counts["rows_with_pose_mapping"] += int(bool(q.get("pose_mapping")))
            markush_counts["rows_with_document_realism"] += int(bool(q.get("document_realism")))
            if graph.get("visible_star_token_indices"):
                markush_token_issues["rows_with_star_atom_coordinate_tokens"] += 1
            if any(str(atom.get("token") or "").strip() == "*" for atom in atom_coordinates if isinstance(atom, dict)):
                markush_token_issues["rows_with_literal_star_tokens"] += 1
            if any(str(cell.get("text") or "").strip() == "*" for cell in ocr_cells if isinstance(cell, dict)):
                markush_token_issues["rows_with_literal_star_ocr_cells"] += 1
print("\n[markush coverage]", flush=True)
print("rows", markush_rows, dict(markush_counts), flush=True)
print("token_issues", dict(markush_token_issues), flush=True)

ordinary_root = root / "ordinary"
ordinary_rows = 0
ordinary_sources = Counter()
ordinary_render_styles = Counter()
ordinary_paper_profiles = Counter()
ordinary_document_operations = Counter()
for csv_path in sorted(ordinary_root.glob("s*/ordinary_negative.csv")):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ordinary_rows += 1
            ordinary_sources[str(row.get("source_arrow") or "")] += 1
            q = json.loads(row.get("render_quality") or "{}")
            ordinary_render_styles[str(q.get("render_style") or "missing")] += 1
            domain = q.get("ordinary_document_domain_policy") if isinstance(q.get("ordinary_document_domain_policy"), dict) else {}
            ordinary_paper_profiles[str(domain.get("paper_profile") or "missing")] += 1
            operations = domain.get("operations") if isinstance(domain.get("operations"), list) else []
            ordinary_document_operations.update(str(item) for item in operations)
print("\n[ordinary coverage]", flush=True)
print("rows", ordinary_rows, "source_arrow", dict(sorted(ordinary_sources.items())), flush=True)
print("render_style", dict(sorted(ordinary_render_styles.items())), flush=True)
print("paper_profile", dict(sorted(ordinary_paper_profiles.items())), flush=True)
print("document_operations", dict(sorted(ordinary_document_operations.items())), flush=True)

if failures:
    print("\nFAIL: at least one production gate failed")
    print(json.dumps(failures, indent=2, ensure_ascii=False))
    raise SystemExit(2)
if markush_token_issues:
    print("\nFAIL: Markush accepted data still contains literal '*' variable tokens/cells")
    print(json.dumps(dict(markush_token_issues), indent=2, ensure_ascii=False))
    raise SystemExit(2)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "qc_summary.json").write_text(
    json.dumps(
        {
            "schema_version": "molnextr_moe_production_qc_summary_v1",
            "source_discovery": report["labels"],
            "source_counts": {"train": {k: len(v) for k, v in train.items()}, "cal": {k: len(v) for k, v in cal.items()}},
            "gates": gate_summary,
            "fragment_rows": fragment_rows,
            "markush_rows": markush_rows,
            "markush_token_issues": dict(markush_token_issues),
            "ordinary_rows": ordinary_rows,
            "ordinary_render_style_counts": dict(sorted(ordinary_render_styles.items())),
            "ordinary_paper_profile_counts": dict(sorted(ordinary_paper_profiles.items())),
            "ordinary_document_operation_counts": dict(sorted(ordinary_document_operations.items())),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"\nqc summary -> {REPORT_DIR / 'qc_summary.json'}")
PY

  log "checking obsolete names are absent from production code/data entry points"
  run_env \
    "REPORT_DIR=${REPORT_DIR}" \
    "SELF_SCRIPT=training/molnextr_markush/scripts/run_moe_production.sh" \
    "${PYTHON}" - <<'PY'
from pathlib import Path
import json
import os
patterns = [
    "markush_cdk_pose_probe",
    "formal_sim_v1_full_source_v3",
    "markush_full_source_anchor_plan_v4_formal",
    "fragment_full_source_backbone_seeds_probe",
    "tmp_wavy",
    "tight_smoke",
    "probe_molnextr_attachment_capability",
    "plan_fragment_stratified_expansion",
    "run_markush_cdk_generation_schedule",
    "build_markush_cdk_generation_schedule",
    "probe_cdk_pose_mapping",
    "probe_markush_failure_repairability",
    "audit_markush_v4_dataset_correctness",
    "allow-legacy-until-cleanup",
    "architecture_candidate_split",
    "fragment_architecture_candidate_repaired_aggregate",
    "molgrapher_synthetic_ordinary_validation27383",
    "frozen_base_sidecar_contract",
    "check_raw_training_source_readiness",
    "check_roadmap_constraints",
]
roots = [
    Path("training/molnextr_markush/src"),
    Path("training/molnextr_markush/tools"),
    Path("training/molnextr_markush/configs"),
    Path("README.md"),
    Path("README_zh.md"),
    Path("training/molnextr_markush/README.md"),
    Path("training/molnextr_markush/scripts/command.txt"),
    Path("experiments/moe/molnextr_moe_production_v1/production_reports"),
]
path_only_roots = [
    Path("training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1"),
    Path("training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1_markush_source_anchor_plan"),
    Path("training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1_fragment_source_backbone_seeds"),
    Path("training/molnextr_markush/runs/molnextr_moe_production_v1_contracts"),
]
text_suffixes = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
max_text_bytes = 8 * 1024 * 1024
excluded = {Path(os.environ["SELF_SCRIPT"]).resolve()}
bad = []
def scan_path_text(path):
    try:
        data = path.read_bytes()
    except Exception:
        return
    if len(data) > max_text_bytes:
        return
    for pattern in patterns:
        count = data.count(pattern.encode())
        if count:
            bad.append({"pattern": pattern, "count": count, "path": str(path)})

for root in roots:
    if not root.exists():
        continue
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for path in files:
        if any(part == "__pycache__" for part in path.parts):
            continue
        if path.resolve() in excluded:
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        scan_path_text(path)
for root in path_only_roots:
    if not root.exists():
        continue
    for path in [root, *root.rglob("*")]:
        path_text = str(path)
        if "literal_star_backups" in path.parts:
            continue
        for pattern in patterns:
            if pattern in path_text:
                bad.append({"pattern": pattern, "count": 1, "path": path_text, "scope": "path"})
if bad:
    print(json.dumps(bad[:100], indent=2, ensure_ascii=False))
    raise SystemExit("obsolete names remain; do not train until cleaned")
out = Path(os.environ["REPORT_DIR"]) / "obsolete_name_scan.json"
out.write_text(json.dumps({"schema_version": "obsolete_name_scan_v1", "bad": bad}, indent=2), encoding="utf-8")
print("obsolete-name scan passed")
print(f"obsolete-name scan -> {out}")
PY
}

build_data() {
  check_source_data
  if [[ ! -f "${REAL_ORIGINAL_ROOT}/train/data.parquet" || ! -f "${REAL_ORIGINAL_ROOT}/train/report.json" ]]; then
    build_real_original_data
  else
    log "real-original source data already present, skipping slow rebuild"
  fi
  log "building MoE dataframe only"
  mkdir -p "$(dirname "${DF_CACHE}")" "${OUTPUT_DIR}"
  local build_args=(
    training/molnextr_markush/tools/train_moe.py
    --build-data-only
    --per-label 0
    --source-min-age-seconds "${SOURCE_MIN_AGE_SECONDS}"
    --source-calibration-fraction "${SOURCE_CALIBRATION_FRACTION}"
    --output-dir "${OUTPUT_DIR}"
    --df-cache "${DF_CACHE}"
  )
  if [[ "${REUSE_DF_CACHE}" == "1" ]]; then
    build_args+=(--reuse-df-cache)
  fi
  if [[ -n "${FROZEN_SOURCE_REPORT}" ]]; then
    build_args+=(--frozen-source-report "${FROZEN_SOURCE_REPORT}")
  fi
  run "${PYTHON}" "${build_args[@]}"
}

build_real_original_data() {
  log "building pose-verified rows from original patent images"
  run "${PYTHON}" training/molnextr_markush/tools/build_real_markushgrapher_ocsr.py \
    --split all \
    --output-root "${REAL_ORIGINAL_ROOT}" \
    --skip-existing
}

train_model() {
  check_source_data
  if [[ ! -f "${REAL_ORIGINAL_ROOT}/train/data.parquet" || ! -f "${REAL_ORIGINAL_ROOT}/train/report.json" ]]; then
    build_real_original_data
  fi
  log "training MoE"
  mkdir -p "${OUTPUT_DIR}" "$(dirname "${DF_CACHE}")"
  local auto_epoch_rows_cap=35000
  local fragment_decoder_scope="last_cross_output_edge"
  local fragment_symbol_weight="1.25"
  local attachment_symbol_loss_weight="1.0"
  local variable_identity_loss_weight="2.0"
  local fragment_attachment_margin_weight="0.0"
  if [[ "${PHASE2_SEP:-0}" == "1" ]]; then
    # Native v5 must train the <sep> embedding and causal self-attention, not
    # only the output/cross-attention heads. 40k rows/label with a 90% coverage
    # stream traverses all 391,516 fragment rows within 12 epochs.
    auto_epoch_rows_cap=40000
    fragment_decoder_scope="full"
    fragment_symbol_weight="2.0"
    attachment_symbol_loss_weight="2.0"
    variable_identity_loss_weight="3.0"
    fragment_attachment_margin_weight="1.0"
  fi
  local common_args=(
    training/molnextr_markush/tools/train_moe.py
    --per-label 0
    --epochs "${TRAIN_EPOCHS}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --lr "${TRAIN_LR}"
    --encoder-lr "${ENCODER_LR}"
    --encoder-finetune-stages "${ENCODER_FINETUNE_STAGES}"
    --router-lr "${ROUTER_LR}"
    --attachment-set-lr "${ATTACHMENT_SET_LR}"
    --source-min-age-seconds "${SOURCE_MIN_AGE_SECONDS}"
    --source-calibration-fraction "${SOURCE_CALIBRATION_FRACTION}"
    --output-dir "${OUTPUT_DIR}"
    --df-cache "${DF_CACHE}"
    --fragment-oversample-tiny
    --require-real-original-data
    --real-original-train-df "${REAL_ORIGINAL_ROOT}/train/data.parquet"
    --real-original-train-report "${REAL_ORIGINAL_ROOT}/train/report.json"
    --real-original-weight 12.0
    --sampling-focus-fraction 0.10
    --auto-epoch-rows-cap "${auto_epoch_rows_cap}"
    --routing-strategy soft_mixture
    --sidecar-confidence-threshold 0.5
    --sidecar-threshold-margin 0.03
    --router-margin-weight 0.05
    --router-margin 2.0
    --expert-kind full_mixture
    --full-mixture-sidecar-mode per_sidecar
    --fragment-decoder-train-scope "${fragment_decoder_scope}"
    --router-kind attention_pool
    --token-fusion-mode fixed
    --token-fusion-dispatch hard
    --token-fusion-hard-threshold 0.5
    --specialist-ownership-scope full
    --decouple-fusion-policy-optimization
    --one-sided-fusion-oracle
    --token-fusion-initial-sidecar-weight 0.2
    --token-fusion-supervision-weight 0.0
    --token-fusion-attachment-target 0.95
    --token-fusion-backbone-target 0.15
    --token-fusion-oracle-temperature 1.0
    --loss-reduction per_sample
    --mixture-complete-floor 0.0
    --mixture-token-ce-weight 0.0
    --mixture-edge-ce-weight 0.0
    --edge-distill-weight 1.0
    --distill-complete-weight 1.0
    --coordinate-missing-task-policy masked_sequence
    --sidecar-coordinate-context full
    --attachment-set-enabled
    --attachment-set-hidden-dim 256
    --attachment-set-num-queries 40
    --attachment-set-num-layers 3
    --attachment-set-num-heads 8
    --attachment-set-max-count 40
    --attachment-set-min-confidence 0.1
    --attachment-set-max-anchor-distance 0.25
    --attachment-set-decode-mode direct_sidecar
    --attachment-set-feature-mode multiscale_pointer_heatmap
    --attachment-set-feature-levels 3
    --attachment-set-max-feature-size 48
    --attachment-set-min-pointer-confidence 0.15
    --attachment-set-loss-weight 1.0
    --attachment-set-point-loss-weight 5.0
    --attachment-set-cardinality-loss-weight 1.0
    --attachment-set-relation-loss-weight 1.0
    --attachment-set-anchor-loss-weight 5.0
    --attachment-set-pointer-loss-weight 3.0
    --attachment-set-dummy-pointer-loss-weight 3.0
    --attachment-set-heatmap-loss-weight 2.0
    --attachment-set-heatmap-cost-weight 2.0
    --attachment-set-heatmap-diversity-loss-weight 0.5
    --fragment-symbol-weight "${fragment_symbol_weight}"
    --attachment-symbol-loss-weight "${attachment_symbol_loss_weight}"
    --variable-identity-loss-weight "${variable_identity_loss_weight}"
    --attachment-cardinality-loss-weight 0.25
    --fragment-eos-weight 3.0
    --fragment-terminal-dummy-margin-weight "${fragment_attachment_margin_weight}"
    --fragment-terminal-dummy-margin 2.0
    --fragment-premature-eos-unlikelihood-weight 0.0
    --fragment-grpo-weight 0.0
    --fragment-grpo-every 20
    --fragment-grpo-max-rows 2
    --fragment-grpo-group-size 4
    --fragment-grpo-temperature 0.8
    --fragment-grpo-top-p 0.95
    --fragment-grpo-reference-kl-weight 0.0
    --fragment-counterfactual-dpo-weight 0.0
    --fragment-counterfactual-dpo-beta 0.5
    --fragment-counterfactual-edge-weight 0.0
    --fragment-counterfactual-reward-margin 0.02
    --fragment-search-policy-weight 0.0
    --fragment-search-edge-weight 0.0
    --fragment-search-policy-margin 2.0
  )
  # --- Experimental terminal/search machinery: OFF by default (production-safe). ---
  # These were verified HARMFUL or inert in the 2026-07-17 audit:
  #  * FragmentTerminalActionHead corrupts the AR trajectory (head-disabled diagnostic:
  #    tanimoto 0.162 -> 0.627 when removed); applied via adjust_logits mid-rollout
  #    on a free-running path it was never trained against (exposure bias).
  #  * structured_terminal_edge + search_policy were ON only in the regressing
  #    direct_graph probes; the best model (direct_graph_100step_control_v1) had all
  #    three OFF. Set ENABLE_FRAGMENT_TERMINAL_HEAD=1 / ENABLE_FRAGMENT_SEARCH_POLICY=1
  #    to opt back in for ablation only.
  if [[ "${ENABLE_FRAGMENT_TERMINAL_HEAD:-0}" == "1" ]]; then
    common_args+=(
      --fragment-terminal-action-head-enabled
      --fragment-terminal-action-head-loss-weight 1.0
      --fragment-terminal-action-head-lr 1e-3
      --fragment-terminal-action-head-grad-clip 1.0
      --fragment-structured-terminal-edge-enabled
    )
  fi
  if [[ "${ENABLE_FRAGMENT_SEARCH_POLICY:-0}" == "1" ]]; then
    common_args+=(--fragment-search-policy-weight 1.0)
  fi
  # Phase 2 (MolParser <sep>): fragment rows get a <sep>[anchor:*] suffix on the
  # dummy-free backbone. Off unless PHASE2_SEP=1.
  if [[ "${PHASE2_SEP:-0}" == "1" ]]; then
    common_args+=(--phase2-sep)
  fi
  if [[ -n "${FROZEN_SOURCE_REPORT}" ]]; then
    common_args+=(--frozen-source-report "${FROZEN_SOURCE_REPORT}")
  fi
  common_args+=("${TRAIN_EXTRA_ARGS[@]}")
  if [[ "${REUSE_DF_CACHE}" == "1" ]]; then
    common_args+=(--reuse-df-cache)
  fi
  if (( DDP_GPUS > 1 )); then
    run "${PYTHON}" -m torch.distributed.run --nproc_per_node="${DDP_GPUS}" "${common_args[@]}"
  else
    run "${PYTHON}" "${common_args[@]}"
  fi
}

eval_model() {
  local moe_config="${OUTPUT_DIR}/moe_config.json"
  [[ -f "${moe_config}" ]] || die "missing MoE config: ${moe_config}; run train first"
  mkdir -p "${OUTPUT_DIR}/eval"

  if [[ "${RUN_STANDARD_EVAL}" == "1" ]]; then
    log "running standard held-out complete/markush/fragment eval"
    run "${PYTHON}" evaluation/eval_moe.py \
      --moe-config "${moe_config}" \
      --per-bucket "${EVAL_PER_BUCKET}" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --out "${OUTPUT_DIR}/eval/heldout_eval_report.json" \
    --source-report "${OUTPUT_DIR}/moe_source_report.json" \
      --source-min-age-seconds "${SOURCE_MIN_AGE_SECONDS}" \
      --source-calibration-fraction "${SOURCE_CALIBRATION_FRACTION}"
  fi

  if [[ "${RUN_REAL_MARKUSH_EVAL}" == "1" ]]; then
    [[ -f "${REAL_MARKUSH_EVAL_PARQUET}" ]] || die \
      "missing official MarkushGrapher eval parquet: ${REAL_MARKUSH_EVAL_PARQUET}"
    log "running official original-patent MarkushGrapher test eval"
    run "${PYTHON}" evaluation/eval_real_markushgrapher.py \
      --moe-config "${moe_config}" \
      --eval-parquet "${REAL_MARKUSH_EVAL_PARQUET}" \
      --out "${OUTPUT_DIR}/eval/real_markushgrapher_eval_report.json" \
      --batch-size "${EVAL_BATCH_SIZE}"
  fi

  if [[ "${RUN_REAL_HARD_EVAL}" == "1" ]]; then
    [[ -f "${REAL_HARD_EVAL_CSV}" ]] || die "missing real hard eval csv: ${REAL_HARD_EVAL_CSV}"
    log "running real Markush/wavy hard eval"
    run "${PYTHON}" evaluation/eval_real_wavy_hard_eval.py \
      --moe-config "${moe_config}" \
      --eval-csv "${REAL_HARD_EVAL_CSV}" \
      --out "${OUTPUT_DIR}/eval/real_wavy_hard_eval_report.json" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --min-wavy-star-rate 0.50 \
      --min-wavy-ready-rate 0.50 \
      --min-wavy-strict-ready-rate 0.50 \
      --min-markush-ready-rate 0.50
  fi

  if [[ "${RUN_REAL_TASK_EVAL}" == "1" ]]; then
    [[ -d "${REAL_TASK_DIR}" ]] || die "missing real task dir: ${REAL_TASK_DIR}"
    [[ -f "${REAL_HARD_EVAL_CSV}" ]] || die \
      "missing annotated BioChemInsight task crop csv: ${REAL_HARD_EVAL_CSV}"
    log "running strict BioChemInsight task crop eval"
    run "${PYTHON}" evaluation/eval_real_wavy_hard_eval.py \
      --moe-config "${moe_config}" \
      --eval-csv "${REAL_HARD_EVAL_CSV}" \
      --out "${OUTPUT_DIR}/eval/biocheminsight_task_eval_report.json" \
      --batch-size "${EVAL_BATCH_SIZE}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dataset-id) DATASET_ID="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --ordinary-shards) ORDINARY_SHARDS="$2"; shift 2 ;;
    --markush-shards) MARKUSH_SHARDS="$2"; shift 2 ;;
    --fragment-shards) FRAGMENT_SHARDS="$2"; shift 2 ;;
    --ordinary-rows) ORDINARY_ROWS="$2"; shift 2 ;;
    --markush-rows) MARKUSH_ROWS="$2"; shift 2 ;;
    --fragment-rows) FRAGMENT_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING="$(parse_bool "$2")"; shift 2 ;;
    --generate-jobs) GENERATE_JOBS="$(parse_positive_int "$2")"; shift 2 ;;
    --source-min-age-seconds) SOURCE_MIN_AGE_SECONDS="$2"; shift 2 ;;
    --source-calibration-fraction) SOURCE_CALIBRATION_FRACTION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --df-cache) DF_CACHE="$2"; shift 2 ;;
    --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --lr) TRAIN_LR="$2"; shift 2 ;;
    --encoder-lr) ENCODER_LR="$2"; shift 2 ;;
    --encoder-finetune-stages) ENCODER_FINETUNE_STAGES="$2"; shift 2 ;;
    --router-lr) ROUTER_LR="$2"; shift 2 ;;
    --attachment-set-lr) ATTACHMENT_SET_LR="$2"; shift 2 ;;
    --ddp-gpus) DDP_GPUS="$2"; shift 2 ;;
    --train-extra-arg) TRAIN_EXTRA_ARGS+=("$2"); shift 2 ;;
    --eval-per-bucket) EVAL_PER_BUCKET="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --run-standard-eval) RUN_STANDARD_EVAL="$(parse_bool "$2")"; shift 2 ;;
    --run-real-markush-eval) RUN_REAL_MARKUSH_EVAL="$(parse_bool "$2")"; shift 2 ;;
    --run-real-hard-eval) RUN_REAL_HARD_EVAL="$(parse_bool "$2")"; shift 2 ;;
    --run-real-task-eval) RUN_REAL_TASK_EVAL="$(parse_bool "$2")"; shift 2 ;;
    --real-markush-eval-parquet) REAL_MARKUSH_EVAL_PARQUET="$2"; shift 2 ;;
    --real-hard-eval-csv) REAL_HARD_EVAL_CSV="$2"; shift 2 ;;
    --real-task-dir) REAL_TASK_DIR="$2"; shift 2 ;;
    --report-dir) REPORT_DIR="$2"; shift 2 ;;
    --reuse-df-cache) REUSE_DF_CACHE="$(parse_bool "$2")"; shift 2 ;;
    --frozen-source-report) FROZEN_SOURCE_REPORT="$2"; shift 2 ;;
    --clean-old-artifacts) CLEAN_OLD_ARTIFACTS=1; shift ;;
    --clean-fragment-shards) CLEAN_FRAGMENT_SHARDS="$2"; shift 2 ;;
    --clean-markush-shards) CLEAN_MARKUSH_SHARDS="$2"; shift 2 ;;
    --clean-ordinary-shards) CLEAN_ORDINARY_SHARDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "${STAGE}" in
  generate|qc|build-data|train|eval|all) ;;
  *) die "invalid --stage: ${STAGE}" ;;
esac

require_repo_state
init_paths

log "stage=${STAGE} dataset=${DATASET_ID} output=${OUTPUT_DIR}"
case "${STAGE}" in
  generate)
    generate_data
    qc_report
    ;;
  qc)
    check_source_data
    qc_report
    ;;
  build-data)
    qc_report
    build_data
    ;;
  train)
    qc_report
    build_data
    train_model
    ;;
  eval)
    eval_model
    ;;
  all)
    generate_data
    qc_report
    build_data
    train_model
    eval_model
    ;;
esac

log "done"

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PUBLIC_SCALE_EVIDENCE = [
    {
        "project": "MolScribe",
        "source": "https://arxiv.org/abs/2205.14311 and https://github.com/thomas0809/MolScribe",
        "evidence": (
            "The paper reports 200K molecules for experiments, 1M molecules for the final model, "
            "and a separate 5K PubChem validation set; the official code distributes a 1m680k "
            "checkpoint and multi-GPU training configuration."
        ),
    },
    {
        "project": "MolParser",
        "source": "https://arxiv.org/abs/2411.11098",
        "evidence": (
            "MolParser builds MolParser-7M and uses large synthetic data plus active-learning "
            "samples cropped from real patents and scientific literature."
        ),
    },
    {
        "project": "MolGrapher",
        "source": "https://arxiv.org/abs/2308.12234",
        "evidence": (
            "MolGrapher introduces a synthetic data generation pipeline and USPTO-30K as a "
            "large-scale real-image benchmark."
        ),
    },
    {
        "project": "MolRecBench-Wild",
        "source": "https://arxiv.org/abs/2605.05832",
        "evidence": (
            "MolRecBench-Wild uses 5,029 real structures from 820 recent chemistry papers and "
            "shows that real-world OCSR evaluation remains hard even for modern models."
        ),
    },
    {
        "project": "IMG2SMI",
        "source": "https://arxiv.org/abs/2204.09950",
        "evidence": (
            "IMG2SMI reports an 81 million molecule-scale dataset for chemical image-to-SMILES "
            "training, illustrating that final OCSR-scale generalization claims commonly use "
            "data far beyond tens of thousands of rows."
        ),
    },
]


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def manifest_branch(manifest: dict[str, Any], branch: str) -> dict[str, Any]:
    value = manifest.get(branch)
    return value if isinstance(value, dict) else {}


def bucket_counts(branch: dict[str, Any]) -> dict[str, int]:
    raw = branch.get("bucket_sizes")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int_value(value) for key, value in raw.items()}


def min_bucket_count(counts: dict[str, int]) -> int:
    positive = [int(value) for value in counts.values() if int(value) > 0]
    return min(positive) if positive else 0


def summarize_branch(
    *,
    name: str,
    branch: dict[str, Any],
    min_architecture_positive_rows: int,
    min_architecture_train_rows: int,
    min_architecture_calibration_rows: int,
    min_formal_positive_rows: int,
    min_formal_train_rows: int,
    min_formal_calibration_rows: int,
    min_architecture_bucket_rows: int,
    min_formal_bucket_rows: int,
    min_source_groups_for_architecture: int,
    min_source_groups_for_formal: int,
) -> dict[str, Any]:
    input_rows = int_value(branch.get("input_rows"))
    train_rows = int_value(branch.get("train_rows"))
    calibration_rows = int_value(branch.get("calibration_rows"))
    source_groups = int_value(branch.get("source_groups"))
    calibration_source_groups = int_value(branch.get("calibration_source_groups"))
    overlap = int_value(branch.get("source_group_overlap"))
    counts = bucket_counts(branch)
    bucket_min = min_bucket_count(counts)
    bucket_count = int_value(branch.get("bucket_count")) or len(counts)

    measured_blockers: list[str] = []
    architecture_blockers: list[str] = []
    formal_blockers: list[str] = []
    deployment_blockers: list[str] = []

    if input_rows <= 0 or train_rows <= 0 or calibration_rows <= 0:
        measured_blockers.append(f"{name} split has empty input/train/calibration rows")
    if overlap != 0:
        measured_blockers.append(f"{name} source_group_overlap is {overlap}; required 0")
        architecture_blockers.append(f"{name} source_group_overlap is {overlap}; required 0")
        formal_blockers.append(f"{name} source_group_overlap is {overlap}; required 0")

    if input_rows < min_architecture_positive_rows:
        architecture_blockers.append(
            f"{name} positives are {input_rows}; required >= {min_architecture_positive_rows} "
            "for architecture comparison"
        )
    if train_rows < min_architecture_train_rows:
        architecture_blockers.append(
            f"{name} train positives are {train_rows}; required >= {min_architecture_train_rows} "
            "for architecture comparison"
        )
    if calibration_rows < min_architecture_calibration_rows:
        architecture_blockers.append(
            f"{name} calibration positives are {calibration_rows}; required >= "
            f"{min_architecture_calibration_rows} for architecture comparison"
        )
    if source_groups < min_source_groups_for_architecture:
        architecture_blockers.append(
            f"{name} source groups are {source_groups}; required >= {min_source_groups_for_architecture} "
            "for architecture comparison"
        )
    if counts and bucket_min < min_architecture_bucket_rows:
        architecture_blockers.append(
            f"{name} minimum nonempty bucket count is {bucket_min}; required >= "
            f"{min_architecture_bucket_rows} for architecture comparison"
        )

    if input_rows < min_formal_positive_rows:
        formal_blockers.append(
            f"{name} positives are {input_rows}; required >= {min_formal_positive_rows} "
            "for formal branch training candidate"
        )
    if train_rows < min_formal_train_rows:
        formal_blockers.append(
            f"{name} train positives are {train_rows}; required >= {min_formal_train_rows} "
            "for formal branch training candidate"
        )
    if calibration_rows < min_formal_calibration_rows:
        formal_blockers.append(
            f"{name} calibration positives are {calibration_rows}; required >= "
            f"{min_formal_calibration_rows} for formal branch training candidate"
        )
    if source_groups < min_source_groups_for_formal:
        formal_blockers.append(
            f"{name} source groups are {source_groups}; required >= {min_source_groups_for_formal} "
            "for formal branch training candidate"
        )
    if counts and bucket_min < min_formal_bucket_rows:
        formal_blockers.append(
            f"{name} minimum nonempty bucket count is {bucket_min}; required >= "
            f"{min_formal_bucket_rows} for formal branch training candidate"
        )

    deployment_blockers.extend(formal_blockers)
    deployment_blockers.append(
        f"{name} has no independent real benchmark evidence attached to this scale report"
    )

    if formal_blockers:
        classification = "architecture_or_measured_gate_only" if not architecture_blockers else "measured_gate_or_probe_only"
    else:
        classification = "formal_training_candidate"
    if deployment_blockers:
        deployment_classification = "not_deployment_certification_candidate"
    else:
        deployment_classification = "deployment_certification_candidate"

    return {
        "kind": name,
        "input_rows": int(input_rows),
        "train_rows": int(train_rows),
        "calibration_rows": int(calibration_rows),
        "source_groups": int(source_groups),
        "calibration_source_groups": int(calibration_source_groups),
        "source_group_overlap": int(overlap),
        "bucket_count": int(bucket_count),
        "bucket_sizes": counts,
        "min_nonempty_bucket_rows": int(bucket_min),
        "measured_gate_allowed": not measured_blockers,
        "architecture_comparison_allowed": not architecture_blockers,
        "formal_training_candidate": not formal_blockers,
        "deployment_certification_candidate": not deployment_blockers,
        "classification": classification,
        "deployment_classification": deployment_classification,
        "measured_blockers": measured_blockers,
        "architecture_blockers": architecture_blockers,
        "formal_blockers": formal_blockers,
        "deployment_blockers": deployment_blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify whether a sidecar split is large enough for measured gates, architecture comparison, formal training, and deployment claims."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-architecture-positive-rows", type=int, default=30000)
    parser.add_argument("--min-architecture-train-rows", type=int, default=20000)
    parser.add_argument("--min-architecture-calibration-rows", type=int, default=10000)
    parser.add_argument("--min-architecture-bucket-rows", type=int, default=1000)
    parser.add_argument("--min-architecture-source-groups", type=int, default=20000)
    parser.add_argument("--min-formal-positive-rows", type=int, default=100000)
    parser.add_argument("--min-formal-train-rows", type=int, default=80000)
    parser.add_argument("--min-formal-calibration-rows", type=int, default=20000)
    parser.add_argument("--min-formal-bucket-rows", type=int, default=2500)
    parser.add_argument("--min-formal-source-groups", type=int, default=80000)
    parser.add_argument("--min-final-ocsr-scale-rows", type=int, default=1000000)
    parser.add_argument("--min-real-benchmark-rows-for-deployment", type=int, default=5000)
    args = parser.parse_args()

    manifest = load_json(args.split_manifest)
    branches = {
        "fragment": summarize_branch(
            name="fragment",
            branch=manifest_branch(manifest, "fragment"),
            min_architecture_positive_rows=args.min_architecture_positive_rows,
            min_architecture_train_rows=args.min_architecture_train_rows,
            min_architecture_calibration_rows=args.min_architecture_calibration_rows,
            min_formal_positive_rows=args.min_formal_positive_rows,
            min_formal_train_rows=args.min_formal_train_rows,
            min_formal_calibration_rows=args.min_formal_calibration_rows,
            min_architecture_bucket_rows=args.min_architecture_bucket_rows,
            min_formal_bucket_rows=args.min_formal_bucket_rows,
            min_source_groups_for_architecture=args.min_architecture_source_groups,
            min_source_groups_for_formal=args.min_formal_source_groups,
        ),
        "markush": summarize_branch(
            name="markush",
            branch=manifest_branch(manifest, "markush"),
            min_architecture_positive_rows=args.min_architecture_positive_rows,
            min_architecture_train_rows=args.min_architecture_train_rows,
            min_architecture_calibration_rows=args.min_architecture_calibration_rows,
            min_formal_positive_rows=args.min_formal_positive_rows,
            min_formal_train_rows=args.min_formal_train_rows,
            min_formal_calibration_rows=args.min_formal_calibration_rows,
            min_architecture_bucket_rows=args.min_architecture_bucket_rows,
            min_formal_bucket_rows=args.min_formal_bucket_rows,
            min_source_groups_for_architecture=args.min_architecture_source_groups,
            min_source_groups_for_formal=args.min_formal_source_groups,
        ),
    }
    ordinary = manifest_branch(manifest, "ordinary")
    ordinary_summary = {
        "input_rows": int_value(ordinary.get("input_rows")),
        "train_rows": int_value(ordinary.get("train_rows")),
        "calibration_rows": int_value(ordinary.get("calibration_rows")),
        "source_groups": int_value(ordinary.get("source_groups")),
    }

    total_branch_positive_rows = sum(int(branch["input_rows"]) for branch in branches.values())
    formal_scale_blockers = []
    architecture_blockers = []
    deployment_blockers = []
    for name, branch in branches.items():
        architecture_blockers.extend(branch["architecture_blockers"])
        formal_scale_blockers.extend(branch["formal_blockers"])
        deployment_blockers.extend(branch["deployment_blockers"])
        if not branch["measured_gate_allowed"]:
            formal_scale_blockers.extend(branch["measured_blockers"])

    if total_branch_positive_rows < int(args.min_final_ocsr_scale_rows):
        deployment_blockers.append(
            "total fragment+markush positives are "
            f"{total_branch_positive_rows}; required >= {args.min_final_ocsr_scale_rows} "
            "before claiming OCSR-scale final generalization"
        )

    report = {
        "schema_version": "training_scale_adequacy_v1",
        "split_manifest": str(args.split_manifest),
        "policy": {
            "small_data_is_debug_probe_or_measured_gate_only": True,
            "architecture_comparison_requires_large_stratified_source_disjoint_data": True,
            "formal_branch_training_requires_100k_scale_by_default": True,
            "deployment_certification_requires_large_scale_plus_independent_real_benchmark": True,
            "do_not_claim_generalization_from_thousands_of_rows": True,
            "does_not_replace_schema_visual_source_leak_coverage_taxonomy_attachment_confidence_runtime_model_scale_router_gates": True,
        },
        "thresholds": {
            "min_architecture_positive_rows": int(args.min_architecture_positive_rows),
            "min_architecture_train_rows": int(args.min_architecture_train_rows),
            "min_architecture_calibration_rows": int(args.min_architecture_calibration_rows),
            "min_architecture_bucket_rows": int(args.min_architecture_bucket_rows),
            "min_architecture_source_groups": int(args.min_architecture_source_groups),
            "min_formal_positive_rows": int(args.min_formal_positive_rows),
            "min_formal_train_rows": int(args.min_formal_train_rows),
            "min_formal_calibration_rows": int(args.min_formal_calibration_rows),
            "min_formal_bucket_rows": int(args.min_formal_bucket_rows),
            "min_formal_source_groups": int(args.min_formal_source_groups),
            "min_final_ocsr_scale_rows": int(args.min_final_ocsr_scale_rows),
            "min_real_benchmark_rows_for_deployment": int(args.min_real_benchmark_rows_for_deployment),
        },
        "public_scale_evidence": PUBLIC_SCALE_EVIDENCE,
        "branches": branches,
        "ordinary": ordinary_summary,
        "architecture_comparison_allowed": not architecture_blockers,
        "formal_training_start_allowed": not formal_scale_blockers,
        "deployment_certification_allowed": not deployment_blockers,
        "architecture_blockers": architecture_blockers,
        "formal_scale_blockers": formal_scale_blockers,
        "deployment_blockers": deployment_blockers,
        "decision": {
            "current_data_can_debug_code_paths": True,
            "current_data_can_run_measured_gates": all(branch["measured_gate_allowed"] for branch in branches.values()),
            "current_data_can_compare_architectures": not architecture_blockers,
            "current_data_can_start_formal_branch_training": not formal_scale_blockers,
            "current_data_can_support_final_generalization_claim": not deployment_blockers,
        },
        "recommended_next_steps": [
            "Treat current fragment data as probe/measured-gate data until it reaches at least 100k stratified positives.",
            "Use the existing tens-of-thousands Markush split for measured gates and direction finding, not final generalization claims.",
            "Build larger fragment and Markush corpora with source-disjoint stratification, long-tail bucket coverage, hard negatives, and independent real benchmarks.",
            "After scale adequacy passes, rerun schema, visual, source-leak, coverage, taxonomy, attachment-role, confidence, runtime, model-scale, and router gates before formal training.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if formal_scale_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

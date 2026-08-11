from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ALLOWED_GENERATED_DIRS: set[str] = set()

ALLOWED_SPLIT_DIRS: set[str] = set()

ALLOWED_SMOKE_DIRS: set[str] = set()

ALLOWED_MARKUSH_SMOKE_DIRS: set[str] = set()

ALLOWED_CONTRACT_DIRS = {
    "accepted_markush_layout",
    "review_accepted_markush_layout_current",
}

ALLOWED_CONTRACT_DIR_PATTERNS: list[re.Pattern[str]] = []

FORBIDDEN_FILES = {
    "training/molnextr_markush/tools/build_pose_factory_fragment_v4_shard.py",
    "training/molnextr_markush/tools/build_pose_factory_rdkit_shard.py",
    "training/molnextr_markush/tools/build_endpoint_anchor_label_audit.py",
    "training/molnextr_markush/tools/build_endpoint_anchor_review_queue.py",
    "training/molnextr_markush/tools/split_endpoint_anchor_labels.py",
    "training/molnextr_markush/tools/validate_endpoint_anchor_review.py",
}

FORBIDDEN_NAME_SNIPPETS = [
    "rdkit_attachment_fragment_v3",
    "rdkit_attachment_fragment_v4",
    "rdkit_attachment_fragment_v5",
    "rdkit_attachment_fragment_v6",
    "rdkit_attachment_fragment_v7",
    "rdkit_attachment_fragment_v8",
    "rdkit_attachment_fragment_v9",
    "rdkit_attachment_fragment_v10",
    "rdkit_attachment_fragment_v11",
    "document_cut_wavy_fragment_v2",
    "indigo_ordinary_v1_probe",
    "sidecar_v6_formal",
    "sidecar_v9f_formal",
    "v6_cpu",
    "v6_gated",
    "v9f_cpu",
    "gate_should_fail",
    "current_router_policy",
    "endpoint_anchor_manual",
    "endpoint_anchor_split",
]


def child_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_dir())


def ensure_legacy_only(path: Path) -> list[str]:
    if not path.exists():
        return []
    if not path.is_dir():
        return [f"legacy path is not a directory: {path}"]
    return []


def forbidden_snippet_hits(root: Path) -> list[str]:
    hits: list[str] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        text = str(path)
        if any(snippet in text for snippet in FORBIDDEN_NAME_SNIPPETS):
            hits.append(text)
    return sorted(hits)


def missing_required_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not Path(path).exists()]


def formal_sim_required_paths(root: Path, dataset_id: str, contract_root: Path) -> list[str]:
    dataset_root = root / "data/generated/pose_factory" / dataset_id
    split_root = dataset_root / "splits/source_disjoint_candidate"
    return [
        str(dataset_root / "aggregate/markush_layout_positive.csv"),
        str(dataset_root / "aggregate/attachment_fragment_positive.csv"),
        str(dataset_root / "aggregate/ordinary_negative.csv"),
        str(split_root / "formal_sidecar_split_manifest.json"),
        str(split_root / "markush_train.csv"),
        str(split_root / "markush_calibration.csv"),
        str(split_root / "fragment_train.csv"),
        str(split_root / "fragment_calibration.csv"),
        str(split_root / "ordinary_train.csv"),
        str(split_root / "ordinary_calibration.csv"),
        str(contract_root / "aggregate/markush_csv.validation.json"),
        str(contract_root / "aggregate/fragment_csv.validation.json"),
        str(contract_root / "aggregate/ordinary_csv.validation.json"),
        str(contract_root / "aggregate/markush_pose_alignment.json"),
        str(contract_root / "aggregate/markush_substitution_anchor_contract.json"),
        str(contract_root / "formal_sim_lineage_audit.json"),
        str(contract_root / "attachment_role_contract.json"),
        str(contract_root / "source_leak.json"),
        str(contract_root / "coverage.json"),
        str(contract_root / "fragment_taxonomy_alignment.json"),
        str(contract_root / "markush_assistant_visual_review.json"),
        str(contract_root / "markush_visual_risk_report.json"),
        str(contract_root / "splits/split_image_preservation.json"),
        str(contract_root / "splits/split_pose_preservation_contract.json"),
        str(contract_root / "splits/router_complete_path_contract.json"),
        str(contract_root / "markush_capacity_maximized_scale.json"),
        str(contract_root / "markush_standards_coverage.json"),
        str(contract_root / "gpu_runtime_evidence.json"),
        str(contract_root / "molnextr_dataset_toolkit_contract_check.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that the MolNexTR sidecar training workspace contains only the current accepted data/gate artifacts.")
    parser.add_argument("--root", default="training/molnextr_markush")
    parser.add_argument("--dataset-id", default="molnextr_moe_production_v1")
    parser.add_argument("--contract-root", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    dataset_id = str(args.dataset_id or "").strip()
    contract_root_arg = Path(args.contract_root) if args.contract_root else Path()
    active_contract_root = contract_root_arg if str(contract_root_arg) else root / "runs/sidecar_contract"
    generated_root = root / "data/generated/pose_factory"
    split_root = root / "data/splits"
    smoke_root = root / "runs/sidecar_smoke"
    markush_smoke_root = root / "runs/markush_sidecar_smoke"
    contract_root = root / "runs/sidecar_contract"
    legacy_split_root = root / "data/legacy_splits"
    legacy_run_root = root / "runs/legacy_runs"

    generated_dirs = {path.name for path in child_dirs(generated_root)}
    split_dirs = {path.name for path in child_dirs(split_root)}
    smoke_dirs = {path.name for path in child_dirs(smoke_root)}
    markush_smoke_dirs = {path.name for path in child_dirs(markush_smoke_root)}
    contract_dirs = {path.name for path in child_dirs(contract_root)}

    if not dataset_id:
        raise SystemExit("--dataset-id is required for production workspace checks")
    required_paths = formal_sim_required_paths(root, dataset_id, active_contract_root)

    blockers: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "generated_dirs": sorted(generated_dirs),
        "split_dirs": sorted(split_dirs),
        "smoke_dirs": sorted(smoke_dirs),
        "markush_smoke_dirs": sorted(markush_smoke_dirs),
        "contract_dirs": sorted(contract_dirs),
        "dataset_id": dataset_id,
        "active_contract_root": str(active_contract_root) if dataset_id else str(contract_root),
    }

    allowed_generated_dirs = set(ALLOWED_GENERATED_DIRS)
    allowed_generated_dirs.add(dataset_id)
    allowed_generated_dirs.add(f"{dataset_id}_fragment_source_backbone_seeds")
    allowed_generated_dirs.add(f"{dataset_id}_markush_source_anchor_plan")
    unexpected_generated = sorted(generated_dirs - allowed_generated_dirs)
    unexpected_splits = sorted(split_dirs - ALLOWED_SPLIT_DIRS)
    unexpected_smoke = sorted(smoke_dirs - ALLOWED_SMOKE_DIRS)
    unexpected_markush_smoke = sorted(markush_smoke_dirs - ALLOWED_MARKUSH_SMOKE_DIRS)
    unexpected_contract_dirs = sorted(
        name
        for name in contract_dirs - ALLOWED_CONTRACT_DIRS
        if not any(pattern.fullmatch(name) for pattern in ALLOWED_CONTRACT_DIR_PATTERNS)
    )
    forbidden_existing_files = sorted(path for path in FORBIDDEN_FILES if Path(path).exists())
    pycache_dirs = sorted(str(path) for path in root.rglob("__pycache__") if path.is_dir())
    snippet_hits = forbidden_snippet_hits(root / "data") + forbidden_snippet_hits(root / "runs")
    missing_paths = missing_required_paths(required_paths)

    legacy_findings: list[str] = []
    if unexpected_generated:
        legacy_findings.append(f"unexpected generated pose_factory directories: {unexpected_generated}")
    if unexpected_splits:
        legacy_findings.append(f"unexpected split directories: {unexpected_splits}")
    if unexpected_smoke:
        legacy_findings.append(f"unexpected sidecar smoke directories: {unexpected_smoke}")
    if unexpected_markush_smoke:
        legacy_findings.append(f"unexpected Markush sidecar smoke directories: {unexpected_markush_smoke}")
    if unexpected_contract_dirs:
        legacy_findings.append(f"unexpected sidecar contract subdirectories: {unexpected_contract_dirs}")
    if pycache_dirs:
        legacy_findings.append(f"python cache directories exist: {pycache_dirs}")
    if snippet_hits:
        legacy_findings.append(f"forbidden legacy data/run paths exist: {snippet_hits[:50]}")

    blockers.extend(legacy_findings)

    if forbidden_existing_files:
        blockers.append(f"forbidden legacy files exist: {forbidden_existing_files}")
    if missing_paths:
        blockers.append(f"required current accepted paths are missing: {missing_paths}")
    blockers.extend(ensure_legacy_only(legacy_split_root))
    blockers.extend(ensure_legacy_only(legacy_run_root))

    report = {
        "schema_version": "clean_training_workspace_v1",
        "root": str(root),
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "details": details,
        "required_paths": required_paths,
        "policy": {
            "only_current_generated_shards_allowed": True,
            "only_current_split_dirs_allowed": True,
            "only_current_or_documented_fragment_expert_evidence_runs_allowed": True,
            "only_current_or_documented_markush_expert_evidence_runs_allowed": True,
            "raw_data_is_protected_and_not_scanned_as_generated_output": True,
            "legacy_splits_are_archived_outside_active_split_root": True,
            "legacy_runs_are_archived_outside_active_run_roots": True,
            "old_generator_entrypoints_forbidden": True,
            "training_entrypoint_preflight_gate_is_current_required_artifact": True,
            "formal_sim_dataset_mode": True,
            "legacy_artifacts_are_blockers": True,
            "raw_data_is_not_deleted_or_required_to_be_cleaned": True,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

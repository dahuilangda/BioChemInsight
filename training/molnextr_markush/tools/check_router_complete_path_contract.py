from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import markush_variable_count


SPLIT_ROLES = {
    "fragment_train_csv": "fragment_positive",
    "fragment_calibration_csv": "fragment_positive",
    "markush_train_csv": "markush_positive",
    "markush_calibration_csv": "markush_positive",
    "ordinary_train_csv": "router_negative_complete_molecule",
    "ordinary_calibration_csv": "router_negative_complete_molecule",
}


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(value: Any) -> bool:
    return str(value or "").strip() != ""


def bool_quality(quality: dict[str, Any], *keys: str) -> bool:
    value: Any = quality
    for key in keys:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value is True


def inspect_row(row: dict[str, str], *, role: str) -> list[str]:
    blockers: list[str] = []
    quality = parse_json_object(str(row.get("render_quality") or "{}"))
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    structure_type = str(quality.get("structure_type") or "").strip()
    bucket = str(row.get("structure_type_bucket") or "").strip()
    label = str(row.get("structure_type_label") or "").strip()
    if role == "fragment_positive":
        if bucket != "attachment_fragment":
            blockers.append(f"fragment positive has structure_type_bucket={bucket!r}")
        if structure_type and structure_type != "attachment_fragment":
            blockers.append(f"fragment positive has render_quality.structure_type={structure_type!r}")
        for field in ["endpoint_x", "endpoint_y", "attachment_anchor", "attachment_render_mode"]:
            if not nonempty(row.get(field)):
                blockers.append(f"fragment positive missing {field}")
        if graph.get("single_dummy_atom") is not True:
            blockers.append("fragment positive graph_consistency.single_dummy_atom is not true")
        if graph.get("anchor_dummy_bond_present") is not True:
            blockers.append("fragment positive graph_consistency.anchor_dummy_bond_present is not true")
        if not bool_quality(quality, "quality_gates", "atom_coordinates_present"):
            blockers.append("fragment positive missing atom-coordinate quality gate")
        if "attachment_endpoint" not in quality:
            blockers.append("fragment positive missing render_quality.attachment_endpoint")
        if "attachment_anchor" not in quality:
            blockers.append("fragment positive missing render_quality.attachment_anchor")
    elif role == "markush_positive":
        if bucket != "markush_layout":
            blockers.append(f"Markush positive has structure_type_bucket={bucket!r}")
        if structure_type and structure_type != "markush_layout":
            blockers.append(f"Markush positive has render_quality.structure_type={structure_type!r}")
        markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
        if not isinstance(markush.get("ocr_cells"), list) or not markush.get("ocr_cells"):
            blockers.append("Markush positive missing render_quality.markush.ocr_cells")
        if markush_variable_count(row) <= 0:
            blockers.append("Markush positive has no annotation-matched OCR anchor labels")
        if not bool_quality(quality, "quality_gates", "atom_coordinates_present"):
            blockers.append("Markush positive missing atom-coordinate quality gate")
        if not isinstance(quality.get("atom_coordinates"), list) or not quality.get("atom_coordinates"):
            blockers.append("Markush positive missing atom_coordinates")
    elif role == "router_negative_complete_molecule":
        if bucket != "ordinary_structure":
            blockers.append(f"ordinary router negative has structure_type_bucket={bucket!r}")
        if label and label != "complete_molecule":
            blockers.append(f"ordinary router negative has structure_type_label={label!r}")
        if structure_type and structure_type != "complete_compound":
            blockers.append(f"ordinary router negative has render_quality.structure_type={structure_type!r}")
        if nonempty(row.get("endpoint_x")) or nonempty(row.get("endpoint_y")):
            blockers.append("ordinary router negative unexpectedly has endpoint coordinates")
        if nonempty(row.get("attachment_render_mode")):
            blockers.append("ordinary router negative unexpectedly has attachment_render_mode")
        if graph.get("single_dummy_atom") is True:
            blockers.append("ordinary router negative unexpectedly has single_dummy_atom=true")
        if graph.get("anchor_dummy_bond_present") is True:
            blockers.append("ordinary router negative unexpectedly has anchor_dummy_bond_present=true")
        if not bool_quality(quality, "quality_gates", "atom_coordinates_present"):
            blockers.append("ordinary router negative missing atom-coordinate quality gate")
    else:
        blockers.append(f"unknown role {role!r}")
    return blockers


def inspect_csv(path: Path, *, role: str, max_examples: int) -> dict[str, Any]:
    blockers: list[str] = []
    row_count = 0
    fieldnames: list[str] = []
    role_counts: Counter[str] = Counter()
    example_issues: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row_index, row in enumerate(reader):
            row_count += 1
            role_counts[str(row.get("structure_type_bucket") or "missing")] += 1
            row_blockers = inspect_row(row, role=role)
            if row_blockers:
                blockers.extend(row_blockers)
                if len(example_issues) < max_examples:
                    example_issues.append(
                        {
                            "row_index": row_index,
                            "source_id": row.get("source_id"),
                            "file_path": row.get("file_path"),
                            "issues": row_blockers,
                        }
                    )
    if row_count <= 0:
        blockers.append(f"{path} has no rows")
    return {
        "path": str(path),
        "role": role,
        "row_count": row_count,
        "fieldnames": fieldnames,
        "structure_type_bucket_counts": dict(sorted(role_counts.items())),
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers[:max_examples],
        "example_issues": example_issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit router/complete-path separation for MolNexTR Markush sidecar splits. "
            "This does not accept data and does not start expert training."
        )
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--molnextr-checkpoint", default="models/molnextr_best.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.split_manifest)
    manifest = load_json(manifest_path)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    policy = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
    purpose = manifest.get("purpose") if isinstance(manifest.get("purpose"), dict) else {}
    blockers: list[str] = []
    warnings: list[str] = []

    if policy.get("complete_molecules_remain_on_original_molnextr_path") is not True:
        blockers.append("split manifest policy.complete_molecules_remain_on_original_molnextr_path is not true")
    if policy.get("fragment_and_markush_rows_are_expert_routed_only") is not True:
        blockers.append("split manifest policy.fragment_and_markush_rows_are_expert_routed_only is not true")
    complete_purpose = str(purpose.get("complete_molecule_path") or "")
    if "bypass sidecar" not in complete_purpose and "original" not in complete_purpose:
        blockers.append("split manifest purpose.complete_molecule_path does not describe original-path bypass")

    checkpoint = Path(args.molnextr_checkpoint)
    checkpoint_report: dict[str, Any] = {
        "path": str(checkpoint),
        "exists": checkpoint.exists(),
        "sha256": "",
    }
    if not checkpoint.exists():
        blockers.append(f"MolNexTR checkpoint does not exist: {checkpoint}")
    else:
        checkpoint_report["sha256"] = file_sha256(checkpoint)

    reports: dict[str, Any] = {}
    for output_key, role in SPLIT_ROLES.items():
        path_text = str(outputs.get(output_key) or "")
        if not path_text:
            blockers.append(f"split manifest missing outputs.{output_key}")
            continue
        path = Path(path_text)
        if not path.exists():
            blockers.append(f"split output does not exist for {output_key}: {path}")
            continue
        report = inspect_csv(path, role=role, max_examples=int(args.max_examples))
        reports[output_key] = report
        if report["passed"] is not True:
            blockers.append(f"{output_key} failed router role audit")

    ordinary_rows = sum(
        int(report.get("row_count") or 0)
        for key, report in reports.items()
        if SPLIT_ROLES.get(key) == "router_negative_complete_molecule"
    )
    fragment_rows = sum(
        int(report.get("row_count") or 0)
        for key, report in reports.items()
        if SPLIT_ROLES.get(key) == "fragment_positive"
    )
    markush_rows = sum(
        int(report.get("row_count") or 0)
        for key, report in reports.items()
        if SPLIT_ROLES.get(key) == "markush_positive"
    )
    if ordinary_rows <= 0:
        warnings.append("no ordinary complete-molecule router negatives were observed")
    if fragment_rows <= 0:
        blockers.append("no fragment expert positive rows were observed")
    if markush_rows <= 0:
        blockers.append("no Markush expert positive rows were observed")

    report = {
        "schema_version": "router_complete_path_contract_v1",
        "split_manifest": str(manifest_path),
        "split_stage": manifest.get("stage"),
        "split_accepted_for_training": manifest.get("accepted_for_training") is True,
        "molnextr_checkpoint": checkpoint_report,
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "row_counts": {
            "fragment_expert_positive": fragment_rows,
            "markush_expert_positive": markush_rows,
            "router_negative_complete_molecule": ordinary_rows,
        },
        "reports": reports,
        "policy": {
            "does_not_accept_training_data": True,
            "does_not_start_expert_training": True,
            "complete_molecule_runtime_path": "direct_original_molnextr",
            "complete_molecule_checkpoint": "models/molnextr_best.pth",
            "ordinary_complete_rows_are_router_rejection_negatives_only": True,
            "ordinary_complete_rows_must_not_be_expert_positive_targets": True,
            "fragment_rows_require_explicit_fragment_router_branch": True,
            "markush_rows_require_explicit_markush_router_branch": True,
            "fragment_and_markush_positive_rows_must_not_mix_branches": True,
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

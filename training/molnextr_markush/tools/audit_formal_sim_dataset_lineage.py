from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (
    FORMAL_NONLINEAR_WARP_POLICY,
    FORMAL_SIM_DATASET_FAMILY,
    FORMAL_SIM_LINEAGE_SCHEMA_VERSION,
    FORMAL_SIM_POLICY_SCHEMA_VERSION,
    FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
    fragment_formal_nonlinear_contract_passed,
    formal_nonlinear_pose_contract_passed,
    parse_json_object,
)


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def has_attachment_metadata(row: dict[str, str], quality: dict[str, Any]) -> bool:
    for key in set(row) | set(quality):
        if "attach" not in key.lower() and "endpoint" not in key.lower() and "anchor" not in key.lower():
            continue
        value = quality.get(key) if key in quality else row.get(key)
        if isinstance(value, (dict, list)):
            if value:
                return True
        elif str(value or "").strip():
            return True
    return False


def row_issues(row: dict[str, str], *, expected_branch: str) -> list[str]:
    quality = parse_json_object(row.get("render_quality"))
    if not quality:
        return ["render_quality_invalid"]
    issues: list[str] = []
    lineage = quality.get("dataset_lineage") if isinstance(quality.get("dataset_lineage"), dict) else {}
    policy = quality.get("simulation_policy") if isinstance(quality.get("simulation_policy"), dict) else {}
    if lineage.get("schema_version") != FORMAL_SIM_LINEAGE_SCHEMA_VERSION:
        issues.append("dataset_lineage_schema_mismatch")
    if lineage.get("dataset_family") != FORMAL_SIM_DATASET_FAMILY:
        issues.append("dataset_lineage_family_mismatch")
    if policy.get("schema_version") != FORMAL_SIM_POLICY_SCHEMA_VERSION:
        issues.append("simulation_policy_schema_mismatch")
    if policy.get("dataset_family") != FORMAL_SIM_DATASET_FAMILY:
        issues.append("simulation_policy_family_mismatch")
    if str(lineage.get("branch") or "") != expected_branch:
        issues.append("dataset_lineage_branch_mismatch")
    if str(policy.get("branch") or "") != expected_branch:
        issues.append("simulation_policy_branch_mismatch")
    if not str(lineage.get("parent_source_group") or "").strip():
        issues.append("parent_source_group_missing")
    if policy.get("research_only") is not False:
        issues.append("simulation_policy_research_only_not_false")
    if policy.get("debug_only") is not False:
        issues.append("simulation_policy_debug_only_not_false")
    if policy.get("formal_capable") is not True:
        issues.append("simulation_policy_not_formal_capable")
    if not isinstance(policy.get("allowed_operations"), list):
        issues.append("simulation_policy_allowed_operations_not_list")

    pose_alignment = quality.get("pose_alignment") if isinstance(quality.get("pose_alignment"), dict) else {}
    if pose_alignment.get("image_to_graph_orientation_alignment") is not True:
        issues.append("pose_alignment_not_true")
    if pose_alignment.get("synchronized_after_augmentation") is not True:
        issues.append("pose_alignment_not_synchronized_after_augmentation")

    if expected_branch == "markush_layout_positive":
        if quality.get("structure_type") != "markush_layout":
            issues.append("markush_structure_type_mismatch")
        if pose_alignment.get("coordinate_mutation_after_render") is True:
            if not formal_nonlinear_pose_contract_passed(quality):
                issues.append("markush_formal_nonlinear_contract_not_passed")
            warp = quality.get("nonlinear_document_warp") if isinstance(quality.get("nonlinear_document_warp"), dict) else {}
            if warp.get("policy") != FORMAL_NONLINEAR_WARP_POLICY:
                issues.append("markush_formal_nonlinear_policy_mismatch")
        elif pose_alignment.get("coordinate_mutation_after_render") is not False:
            issues.append("markush_coordinate_mutation_invalid")
    elif expected_branch == "fragment_attachment_positive":
        if quality.get("structure_type") != "attachment_fragment":
            issues.append("fragment_structure_type_mismatch")
        if pose_alignment.get("coordinate_mutation_after_render") is True:
            if not fragment_formal_nonlinear_contract_passed(quality):
                issues.append("fragment_formal_nonlinear_contract_not_passed")
            warp = (
                quality.get("fragment_nonlinear_document_warp")
                if isinstance(quality.get("fragment_nonlinear_document_warp"), dict)
                else {}
            )
            if warp.get("policy") != FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY:
                issues.append("fragment_formal_nonlinear_policy_mismatch")
        elif pose_alignment.get("coordinate_mutation_after_render") is not False:
            issues.append("fragment_coordinate_mutation_invalid")
        graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
        if graph.get("single_dummy_atom") is not True:
            issues.append("fragment_not_single_dummy_atom")
        if graph.get("anchor_dummy_bond_present") is not True:
            issues.append("fragment_missing_anchor_dummy_bond")
        endpoint = quality.get("attachment_endpoint") if isinstance(quality.get("attachment_endpoint"), dict) else {}
        anchor = quality.get("attachment_anchor_coord") if isinstance(quality.get("attachment_anchor_coord"), dict) else {}
        if not endpoint or not anchor:
            issues.append("fragment_missing_endpoint_or_anchor_coord")
    elif expected_branch == "ordinary_complete_router_negative":
        if quality.get("structure_type") != "complete_compound":
            issues.append("ordinary_structure_type_mismatch")
        if pose_alignment.get("coordinate_mutation_after_render") is not False:
            issues.append("ordinary_coordinate_mutation_not_allowed")
        graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
        if graph.get("dummy_atom_indices") not in ([], None):
            issues.append("ordinary_dummy_atoms_present")
        if has_attachment_metadata(row, quality):
            issues.append("ordinary_attachment_metadata_present")
        ordinary_policy = (
            quality.get("ordinary_document_domain_policy")
            if isinstance(quality.get("ordinary_document_domain_policy"), dict)
            else {}
        )
        if ordinary_policy.get("geometry_mutation_allowed") is not False:
            issues.append("ordinary_geometry_mutation_policy_missing")
    else:
        issues.append("unknown_expected_branch")
    return sorted(set(issues))


def audit_csv(path: Path, *, branch: str, max_examples: int) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    row_count = 0
    for index, row in enumerate(iter_rows(path), start=2):
        row_count += 1
        issues = row_issues(row, expected_branch=branch)
        if issues:
            issue_counts.update(issues)
            if len(examples) < max_examples:
                examples.append({"row_index": index, "source_id": row.get("source_id"), "issues": issues})
    return {
        "csv": str(path),
        "branch": branch,
        "row_count": row_count,
        "passed": row_count > 0 and not issue_counts,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "streaming_audit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal-sim lineage and branch-specific data contracts.")
    parser.add_argument("--markush-csv", action="append", default=[])
    parser.add_argument("--fragment-csv", action="append", default=[])
    parser.add_argument("--ordinary-csv", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    reports = []
    for path in args.markush_csv:
        reports.append(audit_csv(Path(path), branch="markush_layout_positive", max_examples=int(args.max_examples)))
    for path in args.fragment_csv:
        reports.append(audit_csv(Path(path), branch="fragment_attachment_positive", max_examples=int(args.max_examples)))
    for path in args.ordinary_csv:
        reports.append(audit_csv(Path(path), branch="ordinary_complete_router_negative", max_examples=int(args.max_examples)))

    blockers = []
    if not reports:
        blockers.append("no CSV inputs were provided")
    for report in reports:
        if report.get("passed") is not True:
            blockers.append(f"{report.get('branch')} formal-sim lineage failed for {report.get('csv')}")

    output = {
        "schema_version": "formal_sim_dataset_lineage_audit_v1",
        "dataset_family": FORMAL_SIM_DATASET_FAMILY,
        "passed": not blockers,
        "blockers": blockers,
        "reports": reports,
        "policy": {
            "new_dataset_lineage_required": True,
            "old_accepted_rows_must_not_be_reused_without_regeneration": True,
            "ordinary_router_negatives_must_remain_attachment_free": True,
            "fragment_nonlinear_requires_fragment_specific_contract_before_acceptance": True,
            "fragment_formal_contract_policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

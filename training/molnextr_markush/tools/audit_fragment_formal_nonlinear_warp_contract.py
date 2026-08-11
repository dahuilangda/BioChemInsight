from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (  # noqa: E402
    FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
    fragment_formal_nonlinear_contract_passed,
)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def valid_point(point: Any) -> bool:
    if isinstance(point, dict):
        x = as_float(point.get("x"))
        y = as_float(point.get("y"))
    elif isinstance(point, list) and len(point) == 2:
        x = as_float(point[0])
        y = as_float(point[1])
    else:
        return False
    return x is not None and y is not None and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def row_issues(row: dict[str, str]) -> list[str]:
    quality = parse_quality(row)
    if not quality:
        return ["render_quality_json_invalid"]
    issues: list[str] = []
    if not fragment_formal_nonlinear_contract_passed(quality):
        issues.append("fragment_formal_nonlinear_contract_not_passed")

    warp = (
        quality.get("fragment_nonlinear_document_warp")
        if isinstance(quality.get("fragment_nonlinear_document_warp"), dict)
        else {}
    )
    if warp.get("policy") != FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY:
        issues.append("fragment_nonlinear_policy_mismatch")
    if warp.get("research_only") is True or warp.get("debug_only") is True:
        issues.append("fragment_nonlinear_mislabeled_research_or_debug")
    if warp.get("formal_training_allowed") is not True:
        issues.append("fragment_nonlinear_not_allowed_by_contract")
    if int(warp.get("atom_outside_count") or 0) != 0:
        issues.append("fragment_atom_outside_count_nonzero")
    if int(warp.get("mark_anchor_outside_count") or 0) != 0:
        issues.append("fragment_mark_anchor_outside_count_nonzero")
    if warp.get("blank") is True or warp.get("dense") is True:
        issues.append("fragment_warped_image_blank_or_dense")

    pose = (
        quality.get("fragment_nonlinear_pose_preservation")
        if isinstance(quality.get("fragment_nonlinear_pose_preservation"), dict)
        else {}
    )
    if pose.get("passed") is not True:
        issues.append("fragment_nonlinear_pose_preservation_failed")
    for key in [
        "endpoint_matches_dummy_atom",
        "anchor_coord_matches_anchor_atom",
        "anchor_label_matches_atom_token",
        "single_dummy_atom",
        "anchor_dummy_bond_present",
        "connector_samples_valid",
        "terminal_mark_samples_valid",
    ]:
        if pose.get(key) is not True:
            issues.append(f"fragment_{key}_failed")

    atoms = quality.get("atom_coordinates")
    if not isinstance(atoms, list) or not atoms:
        issues.append("missing_atom_coordinates")
    else:
        for atom in atoms:
            if not isinstance(atom, dict) or not valid_point(atom):
                issues.append("invalid_warped_atom_coordinate")
                break

    endpoint = quality.get("attachment_endpoint") if isinstance(quality.get("attachment_endpoint"), dict) else {}
    anchor = quality.get("attachment_anchor_coord") if isinstance(quality.get("attachment_anchor_coord"), dict) else {}
    if not valid_point(endpoint):
        issues.append("invalid_attachment_endpoint")
    if not valid_point(anchor):
        issues.append("invalid_attachment_anchor_coord")

    connector = quality.get("attachment_connector") if isinstance(quality.get("attachment_connector"), dict) else {}
    samples = connector.get("sampled_points_normalized") if isinstance(connector.get("sampled_points_normalized"), list) else []
    if len(samples) < 3 or not all(valid_point(point) for point in samples):
        issues.append("invalid_connector_samples")

    mode = str(quality.get("attachment_render_mode") or "")
    mark_geometry = (
        quality.get("wavy_geometry")
        if mode == "wavy" and isinstance(quality.get("wavy_geometry"), dict)
        else quality.get("fragment_mark_geometry")
        if isinstance(quality.get("fragment_mark_geometry"), dict)
        else {}
    )
    if mark_geometry.get("nonlinear_warp_synchronized") is not True:
        issues.append("terminal_mark_geometry_not_synchronized")
    if mode in {"cut", "wavy"}:
        mark_samples = (
            mark_geometry.get("terminal_mark_sampled_points_normalized")
            if isinstance(mark_geometry.get("terminal_mark_sampled_points_normalized"), list)
            else []
        )
        if len(mark_samples) < 3 or not all(valid_point(point) for point in mark_samples):
            issues.append("invalid_terminal_mark_samples")
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal-capable nonlinear fragment document warp contracts.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    issue_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        issues = row_issues(row)
        if issues:
            issue_counts.update(issues)
            if len(examples) < int(args.max_examples):
                examples.append({"row_index": index, "source_id": row.get("source_id"), "issues": issues})

    report = {
        "schema_version": "fragment_formal_nonlinear_document_warp_contract_audit_v1",
        "csv": str(csv_path),
        "row_count": len(rows),
        "passed": len(rows) > 0 and not issue_counts,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "policy": {
            "formal_policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
            "research_only": False,
            "requires_synchronized_image_atom_endpoint_connector_terminal_mark_geometry": True,
            "does_not_replace_visual_attachment_role_source_leak_router_model_scale_or_readiness_gates": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

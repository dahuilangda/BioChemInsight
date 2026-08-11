from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageStat
import numpy as np


csv.field_size_limit(sys.maxsize)


def load_render_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("render_quality") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def bool_gate(quality: dict[str, Any], key: str) -> bool:
    gates = quality.get("quality_gates")
    if not isinstance(gates, dict):
        return False
    return gates.get(key) is True


def image_stats(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            extrema = gray.getextrema()
            arr = np.asarray(gray)
            ink_fraction = float((arr < 245).mean())
            purewhite_fraction = float((arr >= 254).mean())
            return {
                "readable": True,
                "width": int(width),
                "height": int(height),
                "mean_luma": mean,
                "ink_fraction": float(ink_fraction),
                "purewhite_fraction": purewhite_fraction,
                "blank": bool(ink_fraction < 0.002),
                "dense": bool(ink_fraction > 0.55),
                "extrema": list(extrema),
            }
    except OSError as exc:
        return {"readable": False, "error": str(exc)}


def audit_row(row: dict[str, str], csv_path: Path) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    quality = load_render_quality(row)
    image_path = row_image_path(row, csv_path)
    stats: dict[str, Any] = {}
    if not image_path.exists():
        issues.append("image_missing")
    else:
        stats = image_stats(image_path)
        if stats.get("readable") is not True:
            issues.append("image_unreadable")
        else:
            width = int(stats.get("width") or 0)
            height = int(stats.get("height") or 0)
            if min(width, height) < 80:
                issues.append("image_too_small")
            if max(width, height) > 2048:
                issues.append("image_unexpectedly_large")
            if stats.get("blank"):
                issues.append("image_blank")
            if stats.get("dense"):
                issues.append("image_too_dense")

    if quality.get("structure_type") != "complete_compound":
        issues.append("not_complete_compound")
    if str(row.get("structure_type_bucket") or "") != "ordinary_structure":
        issues.append("not_ordinary_bucket")
    smiles = str(row.get("SMILES") or row.get("smiles") or "")
    if "*" in smiles or " R" in smiles or "[R" in smiles:
        issues.append("ordinary_contains_attachment_or_rgroup_token")
    if not quality.get("atom_coordinates"):
        issues.append("atom_coordinates_missing")
    if not quality.get("bonds"):
        issues.append("bonds_missing")
    if quality.get("coord_policy") != "molgrapher_keypoints_normalized_to_generated_document_context_image":
        issues.append("coord_policy_mismatch")
    if quality.get("render_style") != "moldepictor_synthetic_patent_context":
        issues.append("render_style_mismatch")

    simulation = quality.get("simulation_policy") if isinstance(quality.get("simulation_policy"), dict) else {}
    if simulation.get("formal_capable") is not True:
        issues.append("simulation_not_formal_capable")
    if simulation.get("image_synchronized") is not True:
        issues.append("simulation_image_not_synchronized")
    if simulation.get("atom_coordinates_synchronized") is not True:
        issues.append("simulation_atom_coords_not_synchronized")

    domain = quality.get("ordinary_document_domain_policy")
    if not isinstance(domain, dict):
        issues.append("ordinary_domain_policy_missing")
    else:
        if domain.get("schema_version") != "ordinary_document_domain_policy_v2":
            issues.append("ordinary_domain_policy_schema_mismatch")
        if domain.get("enabled") is not True:
            issues.append("ordinary_document_context_not_enabled")
        if domain.get("geometry_mutation_allowed") is not False:
            issues.append("geometry_mutation_not_forbidden")
        if domain.get("graph_topology_mutation_allowed") is not False:
            issues.append("graph_topology_mutation_not_forbidden")
        if domain.get("attachment_like_rows_allowed") is not False:
            issues.append("attachment_like_rows_not_forbidden")
        if domain.get("complete_path") != "direct_original_molnextr":
            issues.append("complete_path_policy_mismatch")
        operations = domain.get("operations") if isinstance(domain.get("operations"), list) else []
        required_operations = {
            "moldepictor_source_ink_only_composite",
            "alpha_antialiased_transparent_white_structure_composite",
            "diverse_patent_paper_before_structure_composite",
        }
        missing = sorted(required_operations - set(str(item) for item in operations))
        for operation in missing:
            issues.append(f"ordinary_domain_operation_missing:{operation}")
        profile = str(domain.get("paper_profile") or "")
        if profile not in {"clean_white", "white_scan", "gray_scan", "aged_scan"}:
            issues.append("ordinary_paper_profile_missing_or_invalid")
        if domain.get("transparent_white_structure_composite") != "alpha_antialiased":
            issues.append("ordinary_transparent_white_composite_missing")
        if domain.get("atom_coordinates_synchronized") is not True:
            issues.append("ordinary_atom_coordinates_not_synchronized")
        if domain.get("molnextr_input_quality_passed") is not True:
            issues.append("ordinary_molnextr_input_quality_failed")

    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    if graph.get("row_smiles_canonical_matches_mol") is not True:
        issues.append("smiles_graph_mismatch")
    if graph.get("dummy_atom_indices") not in ([], None):
        issues.append("ordinary_has_dummy_atoms")
    if graph.get("anchor_dummy_bond_present") is True:
        issues.append("ordinary_has_anchor_dummy_bond")

    required_gates = [
        "image_readable",
        "atom_coordinates_present",
        "external_backend_referenced",
        "smiles_graph_consistent",
        "image_to_graph_orientation_alignment",
        "molnextr_pose_synchronized_after_augmentation",
        "ordinary_molnextr_input_quality_passed",
        "ordinary_document_context_present",
    ]
    for key in required_gates:
        if not bool_gate(quality, key):
            issues.append(f"quality_gate_missing:{key}")
    return issues, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ordinary complete-molecule negatives for MolNexTR-ready quality.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--output-clean-csv", default="")
    parser.add_argument(
        "--filter-rejected-rows",
        action="store_true",
        help="Write only audit-passing rows to --output-clean-csv and pass if at least one row remains.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = 0
    kept_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    issue_counts: Counter[str] = Counter()
    render_styles: Counter[str] = Counter()
    paper_profiles: Counter[str] = Counter()
    document_operations: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    ink_fractions: list[float] = []
    purewhite_fractions: list[float] = []
    examples: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            quality = load_render_quality(row)
            render_styles[str(quality.get("render_style") or "missing")] += 1
            domain = quality.get("ordinary_document_domain_policy") if isinstance(quality.get("ordinary_document_domain_policy"), dict) else {}
            paper_profiles[str(domain.get("paper_profile") or "missing")] += 1
            operations = domain.get("operations") if isinstance(domain.get("operations"), list) else []
            document_operations.update(str(item) for item in operations)
            issues, stats = audit_row(row, csv_path)
            if stats.get("readable") is True:
                widths.append(int(stats.get("width") or 0))
                heights.append(int(stats.get("height") or 0))
                ink_fractions.append(float(stats.get("ink_fraction") or 0.0))
                purewhite_fractions.append(float(stats.get("purewhite_fraction") or 0.0))
            if issues:
                issue_counts.update(issues)
                if len(examples) < int(args.max_examples):
                    examples.append(
                        {
                            "source_id": row.get("source_id", ""),
                            "file_path": row.get("file_path", ""),
                            "issues": issues,
                            "image_stats": stats,
                        }
                    )
            else:
                kept_rows.append(row)

    rejected_rows = rows - len(kept_rows)
    output_clean_csv = str(args.output_clean_csv or "")
    if output_clean_csv:
        write_rows(Path(output_clean_csv), kept_rows, fieldnames)

    def numeric_summary(values: list[float | int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        values_f = [float(value) for value in values]
        return {
            "count": len(values_f),
            "min": min(values_f),
            "max": max(values_f),
            "mean": sum(values_f) / len(values_f),
        }

    report = {
        "schema_version": "ordinary_molnextr_quality_contract_v1",
        "csv": str(csv_path),
        "row_count": rows,
        "kept_rows": len(kept_rows),
        "rejected_rows": rejected_rows,
        "output_clean_csv": output_clean_csv,
        "filter_rejected_rows": bool(args.filter_rejected_rows),
        "passed": (len(kept_rows) > 0 if args.filter_rejected_rows else rows > 0 and not issue_counts),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "render_style_counts": dict(sorted(render_styles.items())),
        "paper_profile_counts": dict(sorted(paper_profiles.items())),
        "document_operation_counts": dict(sorted(document_operations.items())),
        "image_stats": {
            "width": numeric_summary(widths),
            "height": numeric_summary(heights),
            "ink_fraction": numeric_summary(ink_fractions),
            "purewhite_fraction": numeric_summary(purewhite_fractions),
        },
        "policy": {
            "complete_molecules_are_router_negatives_only": True,
            "must_preserve_molgrapher_source_graph_and_synchronize_generated_document_context_keypoints": True,
            "must_be_attachment_free": True,
            "must_be_molnextr_pose_synchronized": True,
            "uses_patent_literature_input_domain_without_geometry_or_topology_mutation": True,
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

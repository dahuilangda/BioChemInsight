from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import (  # noqa: E402
    annotation_r_labels,
    annotation_stable_labels,
    cxsmiles_dummy_labels,
    markush_layout_cells,
)
from training.molnextr_markush.src.pose_factory import image_basic_stats  # noqa: E402
from utils.markush_labels import is_fixed_substituent_label, is_markush_label, markush_label_category, normalize_label  # noqa: E402

csv.field_size_limit(sys.maxsize)


def parse_quality(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def load_json_object(path_text: str) -> dict[str, Any]:
    value = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path_text} must contain a JSON object")
    return value


def image_readability_evidence(path_text: str, *, expected_rows: int | None = None) -> dict[str, Any]:
    if not path_text:
        return {"provided": False, "passed": False, "blockers": ["image readability evidence was not provided"]}
    report = load_json_object(path_text)
    blockers: list[str] = []
    row_count = int(report.get("row_count") or 0)
    if expected_rows is not None and row_count != int(expected_rows):
        blockers.append(f"image readability evidence row_count {row_count} != expected {int(expected_rows)}")
    if report.get("schema_version") == "markush_assistant_visual_review_v1":
        if report.get("strict_machine_visual_acceptance_passed") is not True:
            blockers.append("assistant visual review strict_machine_visual_acceptance_passed is not true")
        if report.get("research_probe_visual_readability_passed") is not True:
            blockers.append("assistant visual review readability probe did not pass")
        if report.get("formal_background_realism_passed") is not True:
            blockers.append("assistant visual review background realism did not pass")
    else:
        image_quality = report.get("image_quality_counts") if isinstance(report.get("image_quality_counts"), dict) else {}
        readable = int(image_quality.get("readable") or 0)
        if report.get("trainable") is not True:
            blockers.append("schema validation report is not trainable")
        if int(report.get("invalid_rows") or 0) != 0:
            blockers.append("schema validation report has invalid rows")
        if row_count <= 0:
            blockers.append("schema validation report row_count is missing")
        if readable != row_count:
            blockers.append(f"schema validation readable rows {readable} != row_count {row_count}")
    return {
        "provided": True,
        "path": str(path_text),
        "schema_version": report.get("schema_version"),
        "row_count": row_count,
        "passed": not blockers,
        "blockers": blockers,
    }


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def row_image_path(row: dict[str, Any], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def bbox_contains_point(bbox: Any, x: float, y: float, *, margin: float) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    values = [as_float(item) for item in bbox]
    if any(value is None for value in values):
        return False
    x1, y1, x2, y2 = [float(value) for value in values]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def bbox_center_distance(bbox: Any, x: float, y: float) -> float | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [as_float(item) for item in bbox]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    return float(math.dist(((x1 + x2) * 0.5, (y1 + y2) * 0.5), (x, y)))


def neighbor_indices(bonds: Any) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    if not isinstance(bonds, list):
        return neighbors
    for bond in bonds:
        if not isinstance(bond, dict):
            continue
        try:
            begin = int(bond.get("begin_atom_index"))
            end = int(bond.get("end_atom_index"))
        except (TypeError, ValueError):
            continue
        neighbors[begin].add(end)
        neighbors[end].add(begin)
    return neighbors


def validate_row(
    row: dict[str, Any],
    *,
    csv_path: Path,
    bbox_margin: float,
    require_real_atom_neighbor: bool,
    check_image: bool,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    cxsmiles = str(markush.get("cxsmiles") or row.get("SMILES") or row.get("smiles") or "")
    annotation = str(markush.get("annotation") or "")
    all_dummy_labels = cxsmiles_dummy_labels(cxsmiles)
    dummy_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if is_markush_label(label)
    }
    fixed_abbreviation_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if is_fixed_substituent_label(label)
    }
    non_variable_pseudo_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if not is_markush_label(label) and not is_fixed_substituent_label(label)
    }
    raw_r_labels = [normalize_label(item) for item in annotation_r_labels(annotation)]
    r_labels = [label for label in raw_r_labels if is_markush_label(label)]
    stable_labels = [normalize_label(item) for item in annotation_stable_labels(annotation)]
    expected_label_counts = Counter(dummy_labels.values())
    if not dummy_labels:
        issues.append("missing_cxsmiles_dummyLabel_atomProp")
    if r_labels and Counter(r_labels) != expected_label_counts:
        issues.append("annotation_r_labels_do_not_match_dummy_labels")
    stable_labels_match = not stable_labels or Counter(stable_labels) == expected_label_counts

    atom_coordinates = quality.get("atom_coordinates") if isinstance(quality.get("atom_coordinates"), list) else []
    atoms_by_index: dict[int, dict[str, Any]] = {}
    for atom in atom_coordinates:
        if not isinstance(atom, dict):
            continue
        try:
            atoms_by_index[int(atom.get("atom_index"))] = atom
        except (TypeError, ValueError):
            continue

    graph_dummy_indices = {
        int(item)
        for item in (graph.get("dummy_atom_indices") if isinstance(graph.get("dummy_atom_indices"), list) else [])
        if str(item).strip().lstrip("-").isdigit()
    }
    if not set(dummy_labels).issubset(graph_dummy_indices):
        issues.append("dummyLabel_atom_indices_do_not_match_graph_dummy_atom_indices")

    raw_cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
    try:
        variable_cell_count = int(markush.get("variable_anchor_count") or 0)
    except (TypeError, ValueError):
        variable_cell_count = len(markush_layout_cells(row))
    cells_by_atom_and_text: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in raw_cells:
        if not isinstance(cell, dict) or "atom_index" not in cell:
            continue
        try:
            atom_index = int(cell.get("atom_index"))
        except (TypeError, ValueError):
            continue
        text = normalize_label(str(cell.get("text") or ""))
        cells_by_atom_and_text[(atom_index, text)].append(cell)

    neighbors = neighbor_indices(quality.get("bonds"))
    bbox_distances: list[float] = []
    dummy_only_neighbor_count = 0
    for atom_index, label in sorted(dummy_labels.items()):
        atom = atoms_by_index.get(atom_index)
        if not isinstance(atom, dict):
            issues.append("dummy_atom_missing_atom_coordinate")
            continue
        token = normalize_label(str(atom.get("token") or ""))
        if token == "*":
            issues.append("dummy_atom_coordinate_token_is_star_instead_of_markush_label")
        if token != normalize_label(label):
            issues.append("dummy_atom_coordinate_token_mismatch")
        x = as_float(atom.get("x"))
        y = as_float(atom.get("y"))
        if x is None or y is None:
            issues.append("dummy_atom_coordinate_invalid")
            continue
        matches = cells_by_atom_and_text.get((atom_index, normalize_label(label)), [])
        if len(matches) != 1:
            issues.append("dummy_atom_does_not_have_exactly_one_matching_variable_ocr_cell")
        else:
            cell = matches[0]
            if not bbox_contains_point(cell.get("bbox"), float(x), float(y), margin=float(bbox_margin)):
                issues.append("dummy_atom_coordinate_outside_matching_variable_ocr_bbox")
            distance = bbox_center_distance(cell.get("bbox"), float(x), float(y))
            if distance is not None:
                bbox_distances.append(distance)
        row_neighbors = neighbors.get(atom_index, set())
        if not row_neighbors:
            issues.append("dummy_atom_has_no_graph_bond")
        elif require_real_atom_neighbor:
            has_real_neighbor = any(
                not is_markush_label(normalize_label(str(atoms_by_index.get(neighbor, {}).get("token") or "")))
                and normalize_label(str(atoms_by_index.get(neighbor, {}).get("token") or "")) != "*"
                for neighbor in row_neighbors
            )
            if not has_real_neighbor:
                dummy_only_neighbor_count += 1
                issues.append("dummy_atom_has_no_real_atom_neighbor")

    for cell in raw_cells:
        if not isinstance(cell, dict) or "atom_index" not in cell:
            continue
        text = normalize_label(str(cell.get("text") or ""))
        if text not in expected_label_counts:
            continue
        try:
            atom_index = int(cell.get("atom_index"))
        except (TypeError, ValueError):
            issues.append("variable_ocr_cell_has_invalid_atom_index")
            continue
        if atom_index not in dummy_labels:
            issues.append("variable_ocr_cell_atom_index_is_not_dummy")
        elif normalize_label(dummy_labels[atom_index]) != text:
            issues.append("variable_ocr_cell_text_mismatches_dummyLabel")

    if markush.get("variable_anchor_count_source") == "cxsmiles_dummyLabel_atomProp":
        if int(markush.get("r_tag_count") or -1) != len(dummy_labels):
            issues.append("markush_dummyLabel_anchor_count_metadata_mismatch")
    else:
        if int(markush.get("r_tag_count") or -1) != len(dummy_labels):
            issues.append("markush_r_tag_count_does_not_match_dummyLabel_count")
    if variable_cell_count != len(dummy_labels):
        issues.append("training_variable_cell_count_does_not_match_dummyLabel_count")

    image_stats: dict[str, Any] = {}
    if check_image:
        image_path = row_image_path(row, csv_path)
        image_stats = image_basic_stats(image_path) if image_path else {"readable": False, "error": "missing_image_path"}
        if image_stats.get("readable") is not True:
            issues.append("image_unreadable_for_substitution_contract")
        elif image_stats.get("blank") or image_stats.get("dense"):
            issues.append("image_not_visually_trainable_for_substitution_contract")

    metrics = {
        "source_id": str(row.get("source_id") or ""),
        "dummy_label_count": int(len(dummy_labels)),
        "all_pseudo_atom_label_count": int(len(all_dummy_labels)),
        "fixed_abbreviation_pseudo_atom_count": int(len(fixed_abbreviation_labels)),
        "non_variable_pseudo_atom_count": int(len(non_variable_pseudo_labels)),
        "pseudo_atom_label_categories": {
            str(index): markush_label_category(label) for index, label in sorted(all_dummy_labels.items())
        },
        "graph_dummy_count": int(len(graph_dummy_indices)),
        "raw_ocr_cell_count": int(len(raw_cells)),
        "training_variable_cell_count": int(variable_cell_count),
        "stable_label_count": int(len(stable_labels)),
        "annotation_r_label_count": int(len(raw_r_labels)),
        "annotation_variable_r_label_count": int(len(r_labels)),
        "dummy_only_neighbor_count": int(dummy_only_neighbor_count),
        "bbox_center_distance_max": max(bbox_distances) if bbox_distances else None,
        "stable_labels_match_dummy_labels": bool(stable_labels_match),
        "image_stats": image_stats,
    }
    return sorted(set(issues)), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Markush dummy/R-label substitution anchors against OCR cells, atom coordinates, and graph bonds."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--bbox-margin", type=float, default=0.015)
    parser.add_argument("--allow-dummy-only-neighbor", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--image-readability-report", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    prevalidated_readability = image_readability_evidence(str(args.image_readability_report or ""))
    use_readability_evidence = bool(args.image_readability_report) and prevalidated_readability.get("passed") is True
    issue_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric: dict[str, list[float]] = defaultdict(list)
    passed_rows = 0
    row_count = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            row_count += 1
            issues, metrics = validate_row(
                row,
                csv_path=csv_path,
                bbox_margin=float(args.bbox_margin),
                require_real_atom_neighbor=not bool(args.allow_dummy_only_neighbor),
                check_image=not bool(args.skip_image_check) and not use_readability_evidence,
            )
            if issues:
                issue_counts.update(issues)
                for issue in issues:
                    if len(examples[issue]) < int(args.max_examples):
                        examples[issue].append(
                            {
                                "row_index": row_index,
                                "source_id": metrics["source_id"],
                                "metrics": {key: value for key, value in metrics.items() if key != "image_stats"},
                            }
                        )
            else:
                passed_rows += 1
            for key in ["dummy_label_count", "training_variable_cell_count", "bbox_center_distance_max"]:
                value = metrics.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric[key].append(float(value))
            if metrics.get("stable_labels_match_dummy_labels") is False:
                issue_counts.update(["stable_definition_label_mismatch_diagnostic"])

    failed_rows = int(row_count - passed_rows)
    readability = (
        image_readability_evidence(str(args.image_readability_report), expected_rows=row_count)
        if args.image_readability_report
        else {"provided": False, "passed": False, "blockers": []}
    )
    if args.image_readability_report and readability.get("passed") is not True:
        issue_counts.update(["image_readability_evidence_failed"])
    blockers = []
    if failed_rows:
        blockers.append("one or more Markush substitution anchor contract issues were found")
    if args.image_readability_report and readability.get("passed") is not True:
        blockers.extend(str(item) for item in readability.get("blockers") or ["image readability evidence failed"])
    summary = {
        "schema_version": "markush_substitution_anchor_contract_v1",
        "csv": str(csv_path),
        "row_count": int(row_count),
        "passed_rows": int(passed_rows),
        "failed_rows": failed_rows,
        "passed": not blockers,
        "blockers": blockers,
        "policy": {
            "dummyLabel_or_atomLabel_atomProp_is_primary_variable_source": True,
            "rdkit_cxsmiles_dummy_or_atom_label_parse_precedes_regex_fallback": True,
            "variable_anchor_labels_require_is_markush_label": True,
            "fixed_abbreviation_pseudo_atoms_are_not_markush_variable_anchors": True,
            "generic_class_labels_are_markush_variables_when_dummy_anchor_closed": True,
            "ar_label_is_aryl_markush_variable_in_dummy_label_context_not_argon_atom": True,
            "non_variable_pseudo_atoms_are_diagnostic_only": True,
            "annotation_r_labels_must_match_dummy_labels_when_present": True,
            "annotation_r_label_match_filters_to_markush_variables": True,
            "fixed_and_non_variable_annotation_r_labels_are_diagnostic_only": True,
            "stable_definition_labels_are_diagnostic_not_anchor_hard_fail": True,
            "each_dummy_label_requires_exactly_one_matching_ocr_cell": True,
            "dummy_atom_coordinate_must_lie_inside_matching_ocr_bbox_with_margin": float(args.bbox_margin),
            "atom_coordinate_token_must_equal_the_cxsmiles_markush_dummy_label": True,
            "star_dummy_tokens_are_rejected_for_trainable_markush_variables": True,
            "dummy_atom_must_have_graph_bond": True,
            "dummy_atom_must_have_real_atom_neighbor": not bool(args.allow_dummy_only_neighbor),
            "variable_anchor_count_source_required_for_new_shards": "cxsmiles_dummyLabel_atomProp",
            "fallback_cxsmiles_atom_label_block_supported": True,
            "non_variable_element_ocr_cells_are_audited_separately_not_treated_as_markush_variables": True,
            "image_readability_evidence_consumed": bool(args.image_readability_report),
            "image_readability_report_must_match_row_count_when_provided": True,
            "image_readability_report_does_not_replace_dummy_ocr_coordinate_bond_neighbor_checks": True,
        },
        "image_readability_evidence": readability,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": {key: value for key, value in sorted(examples.items())},
        "metrics": {
            key: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
            }
            for key, values in sorted(numeric.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

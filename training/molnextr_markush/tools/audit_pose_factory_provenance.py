from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FRAGMENT_TYPES = {"attachment_fragment", "wavy_fragment", "cut_fragment"}
MARKUSH_TYPES = {"markush_layout"}
REQUIRED_RENDER_PARAMETERS = {
    "seed",
    "font_name",
    "font_size",
    "stroke_ratio",
    "symbol_margin_ratio",
}


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return bool(str(value).strip())


def image_path_for_row(row: dict[str, Any], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def point_in_unit_square(point: Any) -> bool:
    if not isinstance(point, dict):
        return False
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
    except (TypeError, ValueError):
        return False
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def normalized_structure_type(row: dict[str, Any], quality: dict[str, Any]) -> str:
    return str(
        quality.get("structure_type")
        or row.get("structure_type")
        or row.get("structure_type_bucket")
        or row.get("structure_type_label")
        or ""
    ).strip()


def normalize_markush_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def markush_labels_from_annotation(annotation: str) -> list[str]:
    labels = []
    for raw in re.findall(r"<r>(.*?)</r>", annotation or ""):
        text = normalize_markush_label(raw)
        if text:
            labels.append(text)
    return labels


def markush_ocr_texts(markush: dict[str, Any]) -> set[str]:
    cells = markush.get("ocr_cells")
    if not isinstance(cells, list):
        return set()
    texts = set()
    for cell in cells:
        if isinstance(cell, dict):
            text = normalize_markush_label(cell.get("text"))
            if text:
                texts.add(text)
    return texts


def markush_dummy_indices(quality: dict[str, Any]) -> set[int]:
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    indices = graph.get("dummy_atom_indices")
    if isinstance(indices, list):
        out = set()
        for value in indices:
            try:
                out.add(int(value))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    atoms = quality.get("atom_coordinates")
    out = set()
    if isinstance(atoms, list):
        for atom in atoms:
            if not isinstance(atom, dict) or str(atom.get("token") or "") != "*":
                continue
            try:
                out.add(int(atom.get("atom_index")))
            except (TypeError, ValueError):
                continue
    return out


def markush_visible_anchor_labels(markush: dict[str, Any], quality: dict[str, Any]) -> list[str]:
    cells = markush.get("ocr_cells")
    if not isinstance(cells, list):
        return []
    dummy_indices = markush_dummy_indices(quality)
    labels: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        try:
            atom_index = int(cell.get("atom_index"))
        except (TypeError, ValueError):
            continue
        if atom_index not in dummy_indices:
            continue
        text = normalize_markush_label(cell.get("text"))
        if text:
            labels.append(text)
    return labels


def add_issue(
    issues: list[dict[str, Any]],
    *,
    csv_path: Path,
    row_index: int,
    source_id: str,
    code: str,
    message: str,
) -> None:
    issues.append(
        {
            "csv": str(csv_path),
            "row_index": int(row_index),
            "source_id": source_id,
            "severity": "error",
            "code": code,
            "message": message,
        }
    )


def audit_row(csv_path: Path, row_index: int, row: dict[str, Any], issues: list[dict[str, Any]], counters: dict[str, Counter[str]]) -> None:
    quality = parse_json_object(row.get("render_quality"))
    source_id = str(row.get("source_id") or quality.get("source_record_id") or "").strip()
    structure_type = normalized_structure_type(row, quality)
    counters["structure_type"].update([structure_type or "missing"])

    image_path = image_path_for_row(row, csv_path)
    if not image_path or not image_path.exists() or not image_path.is_file():
        add_issue(
            issues,
            csv_path=csv_path,
            row_index=row_index,
            source_id=source_id,
            code="missing_image_file",
            message=f"image path is missing or unreadable: {image_path}",
        )
    else:
        counters["image_files"].update(["present"])

    atom_coordinates = quality.get("atom_coordinates")
    if not isinstance(atom_coordinates, list) or not atom_coordinates:
        add_issue(
            issues,
            csv_path=csv_path,
            row_index=row_index,
            source_id=source_id,
            code="missing_atom_coordinates",
            message="render_quality.atom_coordinates must be present and non-empty",
        )
    else:
        counters["atom_coordinates"].update(["present"])
        for atom in atom_coordinates:
            if not isinstance(atom, dict) or not {"atom_index", "x", "y"} <= set(atom):
                add_issue(
                    issues,
                    csv_path=csv_path,
                    row_index=row_index,
                    source_id=source_id,
                    code="invalid_atom_coordinate",
                    message="every atom coordinate must include atom_index, x, and y",
                )
                break

    bonds = quality.get("bonds")
    if not isinstance(bonds, list) or not bonds:
        add_issue(
            issues,
            csv_path=csv_path,
            row_index=row_index,
            source_id=source_id,
            code="missing_graph_bonds",
            message="render_quality.bonds must be present and non-empty",
        )
    else:
        counters["bonds"].update(["present"])
        for bond in bonds:
            if not isinstance(bond, dict) or not {"begin_atom_index", "end_atom_index"} <= set(bond):
                add_issue(
                    issues,
                    csv_path=csv_path,
                    row_index=row_index,
                    source_id=source_id,
                    code="invalid_graph_bond",
                    message="every bond must include begin_atom_index and end_atom_index",
                )
                break

    graph = quality.get("graph_consistency")
    if not isinstance(graph, dict) or graph.get("row_smiles_canonical_matches_mol") is not True:
        add_issue(
            issues,
            csv_path=csv_path,
            row_index=row_index,
            source_id=source_id,
            code="missing_or_failed_graph_consistency",
            message="graph_consistency must be present and row_smiles_canonical_matches_mol must be true",
        )
    else:
        counters["graph_consistency"].update(["present_and_matching"])

    render_provenance = quality.get("render_provenance")
    if not isinstance(render_provenance, dict):
        add_issue(
            issues,
            csv_path=csv_path,
            row_index=row_index,
            source_id=source_id,
            code="missing_render_provenance",
            message="render_quality.render_provenance must be present",
        )
    else:
        counters["render_provenance"].update(["present"])
        parameters = render_provenance.get("parameters")
        if not isinstance(parameters, dict) or not REQUIRED_RENDER_PARAMETERS <= set(parameters):
            missing = sorted(REQUIRED_RENDER_PARAMETERS - set(parameters if isinstance(parameters, dict) else {}))
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="incomplete_render_parameters",
                message=f"render_provenance.parameters missing required keys: {missing}",
            )
        if not is_nonempty(render_provenance.get("java_seed")):
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_render_seed",
                message="render_provenance.java_seed must be preserved",
            )

    for key in ["candidate_plan_index", "layout_seed", "source_document_key", "source_file", "source_record_id"]:
        if not is_nonempty(quality.get(key)):
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code=f"missing_{key}",
                message=f"render_quality.{key} must be preserved",
            )
        else:
            counters[key].update(["present"])

    if structure_type in MARKUSH_TYPES:
        markush = quality.get("markush")
        if not isinstance(markush, dict):
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_markush_block",
                message="Markush rows require render_quality.markush",
            )
            return
        counters["markush_rows"].update(["present"])
        cells = markush.get("ocr_cells")
        if not isinstance(cells, list) or not cells:
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_markush_ocr_cells",
                message="Markush rows require OCR cells",
            )
        else:
            counters["markush_ocr_cells"].update(["present"])
        labels = markush_visible_anchor_labels(markush, quality)
        label_source = "dummy_ocr_cells"
        if not labels:
            labels = markush_labels_from_annotation(str(markush.get("annotation") or ""))
            label_source = "annotation_r_tags"
        if not labels:
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_markush_anchor_labels",
                message="Markush rows must preserve visible dummy/R anchor labels in OCR cells or legacy <r> annotation labels",
            )
        else:
            counters["markush_anchor_labels"].update([f"present:{label_source}"])
            ocr_texts = markush_ocr_texts(markush)
            missing_labels = sorted(set(labels) - ocr_texts)
            if missing_labels:
                add_issue(
                    issues,
                    csv_path=csv_path,
                    row_index=row_index,
                    source_id=source_id,
                    code="markush_anchor_labels_not_in_ocr",
                    message=f"Markush anchor labels not found in OCR cells: {missing_labels[:10]}",
                )
        if int(markush.get("r_tag_count") or 0) <= 0:
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_markush_r_tag_count",
                message="Markush rows must preserve positive r_tag_count",
            )

    if structure_type in FRAGMENT_TYPES:
        counters["fragment_rows"].update(["present"])
        if not point_in_unit_square(quality.get("attachment_endpoint")):
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_or_invalid_attachment_endpoint",
                message="fragment rows require normalized attachment_endpoint coordinates",
            )
        else:
            counters["attachment_endpoint"].update(["present"])
        if not is_nonempty(quality.get("attachment_anchor")):
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_attachment_anchor",
                message="fragment rows require attachment_anchor",
            )
        else:
            counters["attachment_anchor"].update(["present"])
        if str(quality.get("attachment_direction") or "") not in {"left", "right", "top", "bottom"}:
            add_issue(
                issues,
                csv_path=csv_path,
                row_index=row_index,
                source_id=source_id,
                code="missing_attachment_direction",
                message="fragment rows require attachment_direction side",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated pose-factory rows for provenance and supervision preservation.")
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-issues", type=int, default=200)
    args = parser.parse_args()

    issues: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {
        "structure_type": Counter(),
        "image_files": Counter(),
        "atom_coordinates": Counter(),
        "bonds": Counter(),
        "graph_consistency": Counter(),
        "render_provenance": Counter(),
        "candidate_plan_index": Counter(),
        "layout_seed": Counter(),
        "source_document_key": Counter(),
        "source_file": Counter(),
        "source_record_id": Counter(),
        "markush_rows": Counter(),
        "markush_ocr_cells": Counter(),
        "markush_anchor_labels": Counter(),
        "fragment_rows": Counter(),
        "attachment_endpoint": Counter(),
        "attachment_anchor": Counter(),
    }
    row_count = 0
    csv_paths = [Path(path) for path in args.csv]
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=1):
                row_count += 1
                audit_row(csv_path, row_index, row, issues, counters)

    report = {
        "schema_version": "pose_factory_provenance_audit_v1",
        "csv": [str(path) for path in csv_paths],
        "row_count": int(row_count),
        "passed": row_count > 0 and not issues,
        "issue_count": len(issues),
        "issues": issues[: int(args.max_issues)],
        "coverage": {name: dict(counter.most_common()) for name, counter in counters.items()},
        "policy": {
            "image_graph_atom_coordinates_and_render_provenance_required": True,
            "markush_rows_require_ocr_cells_and_visible_dummy_anchor_labels": True,
            "legacy_annotation_r_labels_are_fallback_only": True,
            "fragment_rows_require_attachment_endpoint_and_anchor": True,
            "complete_molecule_rows_are_not_routed_here": True,
            "acceptance_decision": "audit_only_not_training_acceptance",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

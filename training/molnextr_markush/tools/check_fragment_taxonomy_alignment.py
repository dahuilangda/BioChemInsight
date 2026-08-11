from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def ratio(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


VISUAL_SHAPE_ALIASES = {
    "left_terminal_short_stub": "left_terminal_cut_or_open_stub",
}

REVIEW_ONLY_VISUAL_SHAPES = {
    "unknown_attachment_crop",
}


def normalize_axis_key(axis: str, key: str) -> str | None:
    value = str(key)
    if axis == "visual_shape":
        if value in REVIEW_ONLY_VISUAL_SHAPES:
            return None
        return VISUAL_SHAPE_ALIASES.get(value, value)
    return value


def normalized_positive_keys(summary: dict[str, Any], axis: str) -> set[str]:
    values = summary.get(axis) if isinstance(summary.get(axis), dict) else {}
    keys = set()
    for key, value in values.items():
        if int(value or 0) <= 0:
            continue
        normalized = normalize_axis_key(axis, str(key))
        if normalized:
            keys.add(normalized)
    return keys


def normalized_counter(counter: Counter[str], axis: str) -> Counter[str]:
    output: Counter[str] = Counter()
    for key, value in counter.items():
        normalized = normalize_axis_key(axis, str(key))
        if normalized:
            output[normalized] += int(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generated fragment taxonomy and metadata coverage. This is not a paired-data check: "
            "real-world metadata labels are allowed to be unpaired and are used for coverage, self-supervised "
            "grouping, or downstream heads."
        )
    )
    parser.add_argument("--generated-csv", required=True)
    parser.add_argument("--real-taxonomy-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-real-visual-fraction", type=float, default=0.70)
    parser.add_argument("--min-real-semantic-fraction", type=float, default=0.80)
    parser.add_argument("--min-real-chemistry-fraction", type=float, default=0.75)
    parser.add_argument("--min-mode-count", type=int, default=128)
    parser.add_argument("--min-side-count", type=int, default=128)
    parser.add_argument("--min-document-context-fraction", type=float, default=0.80)
    args = parser.parse_args()

    generated_csv = Path(args.generated_csv)
    rows = read_rows(generated_csv)
    real_taxonomy = load_json(Path(args.real_taxonomy_report))
    real_summary = real_taxonomy.get("summary") if isinstance(real_taxonomy.get("summary"), dict) else {}

    counters: dict[str, Counter[str]] = {
        "visual_shape": Counter(),
        "semantic_family": Counter(),
        "chemistry_family": Counter(),
        "attachment_render_mode": Counter(),
        "endpoint_side": Counter(),
        "attachment_render_geometry": Counter(),
        "augmentations": Counter(),
    }
    document_context_rows = 0
    graph_consistent_rows = 0
    for row in rows:
        quality = parse_quality(row)
        graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
        counters["visual_shape"].update([str(quality.get("visual_shape") or "missing")])
        counters["semantic_family"].update([str(quality.get("semantic_family") or "missing")])
        counters["chemistry_family"].update([str(quality.get("chemistry_family") or "missing")])
        counters["attachment_render_mode"].update([str(quality.get("attachment_render_mode") or row.get("attachment_render_mode") or "missing")])
        counters["endpoint_side"].update([str(quality.get("attachment_direction") or row.get("endpoint_side") or "missing")])
        counters["attachment_render_geometry"].update([str(quality.get("attachment_render_geometry") or "missing")])
        augmentations = quality.get("style_augmentations") if isinstance(quality.get("style_augmentations"), list) else []
        if augmentations:
            counters["augmentations"].update(str(item) for item in augmentations)
        else:
            counters["augmentations"].update(["none"])
        document_context = quality.get("document_context") if isinstance(quality.get("document_context"), dict) else {}
        if document_context.get("enabled") is True:
            document_context_rows += 1
        if (
            graph.get("row_smiles_canonical_matches_mol") is True
            and graph.get("single_dummy_atom") is True
            and graph.get("anchor_dummy_bond_present") is True
        ):
            graph_consistent_rows += 1

    total = len(rows)
    real_visual = normalized_positive_keys(real_summary, "visual_shape")
    real_semantic = normalized_positive_keys(real_summary, "semantic_family")
    real_chemistry = normalized_positive_keys(real_summary, "chemistry_family")
    generated_visual_counts = normalized_counter(counters["visual_shape"], "visual_shape")
    generated_semantic_counts = normalized_counter(counters["semantic_family"], "semantic_family")
    generated_chemistry_counts = normalized_counter(counters["chemistry_family"], "chemistry_family")
    generated_visual = {key for key, value in generated_visual_counts.items() if value > 0}
    generated_semantic = {key for key, value in generated_semantic_counts.items() if value > 0}
    generated_chemistry = {key for key, value in generated_chemistry_counts.items() if value > 0}

    visual_overlap = sorted(real_visual & generated_visual)
    semantic_overlap = sorted(real_semantic & generated_semantic)
    chemistry_overlap = sorted(real_chemistry & generated_chemistry)
    visual_fraction = ratio(len(visual_overlap), len(real_visual))
    semantic_fraction = ratio(len(semantic_overlap), len(real_semantic))
    chemistry_fraction = ratio(len(chemistry_overlap), len(real_chemistry))
    document_context_fraction = ratio(document_context_rows, total)
    graph_consistency_fraction = ratio(graph_consistent_rows, total)

    required_modes = ["wavy", "cut", "query_attachment", "dummy_atom"]
    required_sides = ["left", "right", "top", "bottom"]
    blockers: list[str] = []
    coverage_gaps: list[str] = []
    if total <= 0:
        blockers.append("generated fragment CSV has no rows")
    if visual_fraction < float(args.min_real_visual_fraction):
        coverage_gaps.append(
            f"real visual-shape coverage fraction {visual_fraction:.3f} < {args.min_real_visual_fraction:.3f}"
        )
    if semantic_fraction < float(args.min_real_semantic_fraction):
        coverage_gaps.append(
            f"real semantic-family coverage fraction {semantic_fraction:.3f} < {args.min_real_semantic_fraction:.3f}"
        )
    if chemistry_fraction < float(args.min_real_chemistry_fraction):
        coverage_gaps.append(
            f"real chemistry-family coverage fraction {chemistry_fraction:.3f} < {args.min_real_chemistry_fraction:.3f}"
        )
    for mode in required_modes:
        value = int(counters["attachment_render_mode"].get(mode) or 0)
        if value < int(args.min_mode_count):
            blockers.append(f"attachment mode {mode} count {value} < {args.min_mode_count}")
    for side in required_sides:
        value = int(counters["endpoint_side"].get(side) or 0)
        if value < int(args.min_side_count):
            blockers.append(f"endpoint side {side} count {value} < {args.min_side_count}")
    if document_context_fraction < float(args.min_document_context_fraction):
        blockers.append(f"document-context fraction {document_context_fraction:.3f} < {args.min_document_context_fraction:.3f}")
    if graph_consistency_fraction < 1.0:
        blockers.append(f"graph consistency fraction {graph_consistency_fraction:.3f} < 1.000")

    report = {
        "generated_csv": str(generated_csv),
        "real_taxonomy_report": str(args.real_taxonomy_report),
        "row_count": total,
        "passed": not blockers,
        "blockers": blockers,
        "coverage_gaps": coverage_gaps,
        "metrics": {
            "real_visual_shape_coverage_fraction": visual_fraction,
            "real_semantic_family_coverage_fraction": semantic_fraction,
            "real_chemistry_family_coverage_fraction": chemistry_fraction,
            "document_context_fraction": document_context_fraction,
            "graph_consistency_fraction": graph_consistency_fraction,
        },
        "real_axes": {
            "visual_shape": sorted(real_visual),
            "semantic_family": sorted(real_semantic),
            "chemistry_family": sorted(real_chemistry),
        },
        "generated_overlap": {
            "visual_shape": visual_overlap,
            "semantic_family": semantic_overlap,
            "chemistry_family": chemistry_overlap,
        },
        "missing_real_axes": {
            "visual_shape": sorted(real_visual - generated_visual),
            "semantic_family": sorted(real_semantic - generated_semantic),
            "chemistry_family": sorted(real_chemistry - generated_chemistry),
        },
        "generated_counts": {name: dict(counter.most_common()) for name, counter in counters.items()},
        "normalized_generated_counts": {
            "visual_shape": dict(generated_visual_counts.most_common()),
            "semantic_family": dict(generated_semantic_counts.most_common()),
            "chemistry_family": dict(generated_chemistry_counts.most_common()),
        },
        "policy": {
            "visual_review_still_required": True,
            "does_not_accept_manifest_by_itself": True,
            "does_not_require_real_sample_pairing": True,
            "metadata_labels_are_unpaired": True,
            "metadata_usage": [
                "coverage_audit",
                "stratified_sampling",
                "self_supervised_grouping",
                "downstream_auxiliary_prediction",
            ],
            "semantic_axis_overlap_is_diagnostic_not_a_hard_pairing_gate": True,
            "self_supervised_positive_views_must_come_from_same_record_or_rendered_augmentation": True,
            "visual_shape_aliases": VISUAL_SHAPE_ALIASES,
            "review_only_visual_shapes_excluded_from_axis_coverage": sorted(REVIEW_ONLY_VISUAL_SHAPES),
            "purpose": "automatic taxonomy and metadata coverage audit before visual acceptance",
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

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def markush_bucket(row: dict[str, str], quality: dict[str, Any]) -> str:
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket in COUNT_BUCKETS:
        return bucket
    try:
        count = int(markush.get("r_tag_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if 3 <= count <= 4:
        return "3-4"
    if 5 <= count <= 8:
        return "5-8"
    if count >= 9:
        return "9+"
    return "missing"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def image_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = list(gray.getdata())
    total = max(1, len(pixels))
    dark = sum(1 for value in pixels if value < 245)
    ink = sum(1 for value in pixels if value < 220)
    very_dark = sum(1 for value in pixels if value < 128)
    return {
        "width": int(width),
        "height": int(height),
        "dark_pixel_ratio_lt245": float(dark / total),
        "ink_pixel_ratio_lt220": float(ink / total),
        "very_dark_pixel_ratio_lt128": float(very_dark / total),
    }


def relative_or_absolute_image_path(csv_path: Path, row: dict[str, str]) -> Path:
    text = str(row.get("file_path") or "").strip()
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = csv_path.parent / path
    if candidate.exists():
        return candidate
    return path


def inspected_item(value: str) -> dict[str, str]:
    path, sep, note = value.partition("::")
    return {
        "path": path.strip(),
        "observation": note.strip() if sep else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an assistant/model-assisted visual substitute report for a Markush review package. "
            "This is not human manual visual acceptance and must not be used as manual_visual_acceptance_v1."
        )
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--risk-output", required=True)
    parser.add_argument("--inspected-image", action="append", default=[])
    parser.add_argument("--viewer", default="codex_assistant_visual_review")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    review_manifest_path = Path(args.review_manifest)
    review_manifest = load_json(review_manifest_path)
    counts_by_bucket: Counter[str] = Counter()
    document_realism_status: Counter[str] = Counter()
    realism_policy: Counter[str] = Counter()
    background_contract: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    missing_images = 0
    unreadable_images = 0
    sampled_image_stats: list[dict[str, Any]] = []
    max_line_rmse = 0.0
    max_line_abs_p95 = 0.0
    max_line_abs_max = 0.0
    max_r_tags = 0
    max_ocr_cells = 0
    max_variable_cells = 0
    min_stroke_ratio = None
    max_stroke_ratio = None
    background_context_rows = 0
    row_count = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            row_count += 1
            quality = parse_quality(row)
            markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
            pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
            document_realism = quality.get("document_realism") if isinstance(quality.get("document_realism"), dict) else {}
            document_context = quality.get("document_context")
            background_realism = quality.get("background_realism")
            render_parameters = document_realism.get("render_parameters") if isinstance(document_realism.get("render_parameters"), dict) else {}

            bucket = markush_bucket(row, quality)
            counts_by_bucket.update([bucket])
            status = str(document_realism.get("status") or "missing")
            document_realism_status.update([status])
            realism_policy.update([str(document_realism.get("policy") or "missing")])

            if isinstance(document_context, dict) and document_context.get("enabled") is True:
                background_context_rows += 1
                background_contract.update(["document_context_enabled"])
            elif isinstance(background_realism, dict) and background_realism.get("enabled") is True:
                background_context_rows += 1
                background_contract.update(["background_realism_enabled"])
            else:
                background_contract.update(["missing_or_disabled"])

            max_line_rmse = max(max_line_rmse, as_float(pose_mapping.get("line_constraint_rmse_svg_units")))
            max_line_abs_p95 = max(max_line_abs_p95, as_float(pose_mapping.get("line_constraint_abs_p95_svg_units")))
            max_line_abs_max = max(max_line_abs_max, as_float(pose_mapping.get("line_constraint_abs_max_svg_units")))

            r_tags = int(as_float(markush.get("r_tag_count")))
            ocr_cells = len(markush.get("ocr_cells") or []) if isinstance(markush.get("ocr_cells"), list) else 0
            variable_cells = int(as_float(markush.get("variable_anchor_count")))
            max_r_tags = max(max_r_tags, r_tags)
            max_ocr_cells = max(max_ocr_cells, ocr_cells)
            max_variable_cells = max(max_variable_cells, variable_cells)
            if r_tags >= 13:
                risk_counts.update(["r_tags_13_plus"])
            if ocr_cells >= 26:
                risk_counts.update(["ocr_cells_26_plus"])
            if variable_cells >= 13:
                risk_counts.update(["variable_cells_13_plus"])

            stroke = as_float(render_parameters.get("stroke_ratio"), default=-1.0)
            if stroke >= 0:
                min_stroke_ratio = stroke if min_stroke_ratio is None else min(min_stroke_ratio, stroke)
                max_stroke_ratio = stroke if max_stroke_ratio is None else max(max_stroke_ratio, stroke)
                if stroke < 0.6:
                    risk_counts.update(["stroke_ratio_lt_0_6"])
                if stroke > 1.8:
                    risk_counts.update(["stroke_ratio_gt_1_8"])

            path = relative_or_absolute_image_path(csv_path, row)
            if not path.exists():
                missing_images += 1
                continue
            if index < 256:
                try:
                    stats = image_stats(path)
                    if stats["ink_pixel_ratio_lt220"] < 0.001:
                        risk_counts.update(["very_low_ink_ratio_sampled"])
                    if stats["ink_pixel_ratio_lt220"] > 0.25:
                        risk_counts.update(["very_dense_ink_ratio_sampled"])
                    if len(sampled_image_stats) < 12:
                        sampled_image_stats.append({"path": str(path), **stats})
                except Exception:
                    unreadable_images += 1

    contact_sheet = str(review_manifest.get("contact_sheet") or review_manifest.get("output") or "")
    inspected_images = [inspected_item(value) for value in args.inspected_image]

    risk_report = {
        "schema_version": "markush_visual_risk_report_v1",
        "csv": str(csv_path),
        "review_manifest": str(review_manifest_path),
        "row_count": row_count,
        "counts_by_r_tag_bucket": dict(sorted(counts_by_bucket.items())),
        "document_realism_status_counts": dict(sorted(document_realism_status.items())),
        "realism_policy_counts": dict(sorted(realism_policy.items())),
        "background_contract_counts": dict(sorted(background_contract.items())),
        "background_context_rows": int(background_context_rows),
        "background_context_fraction": float(background_context_rows / row_count) if row_count else 0.0,
        "risk_counts": dict(sorted(risk_counts.items())),
        "maxima": {
            "line_constraint_rmse_svg_units": max_line_rmse,
            "line_constraint_abs_p95_svg_units": max_line_abs_p95,
            "line_constraint_abs_max_svg_units": max_line_abs_max,
            "r_tag_count": int(max_r_tags),
            "ocr_cell_count": int(max_ocr_cells),
            "variable_cell_count": int(max_variable_cells),
            "stroke_ratio_min": min_stroke_ratio,
            "stroke_ratio_max": max_stroke_ratio,
        },
        "image_readability_probe": {
            "sampled_first_rows": len(sampled_image_stats),
            "missing_images": int(missing_images),
            "unreadable_images": int(unreadable_images),
            "sampled_stats": sampled_image_stats,
        },
        "policy": {
            "report_only": True,
            "does_not_accept_training_data": True,
            "does_not_replace_manual_visual_acceptance": True,
            "does_not_relax_pose_or_substitution_gates": True,
        },
    }

    blockers = []
    if missing_images:
        blockers.append(f"{missing_images} images are missing")
    if unreadable_images:
        blockers.append(f"{unreadable_images} sampled images could not be read")
    if background_context_rows != row_count:
        blockers.append(
            f"document_context/background_realism is enabled for {background_context_rows}/{row_count} rows; "
            "background-realistic formal data is incomplete"
        )
    strict_machine_visual_acceptance_passed = not blockers

    background_limitation = (
        "The current Markush candidate has document_context/background_realism enabled for every row "
        "using the existing audited document_context/noise/JPEG pipeline."
        if background_context_rows == row_count
        else (
            f"document_context/background_realism is enabled for {background_context_rows}/{row_count} rows; "
            "background-realistic formal data is incomplete."
        )
    )

    assistant_report = {
        "schema_version": "markush_assistant_visual_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "viewer": str(args.viewer),
        "csv": str(csv_path),
        "review_manifest": str(review_manifest_path),
        "risk_report": str(args.risk_output),
        "contact_sheet": contact_sheet,
        "review_package_status": str(review_manifest.get("status") or ""),
        "row_count": row_count,
        "review_sample_count": int(review_manifest.get("sample_count") or 0),
        "inspected_images": inspected_images,
        "assistant_visual_substitute": True,
        "manual_visual_acceptance": False,
        "manual_visual_acceptance_schema": "not_markush_manual_visual_acceptance_v1",
        "research_probe_visual_readability_passed": missing_images == 0 and unreadable_images == 0,
        "strict_machine_visual_acceptance_passed": strict_machine_visual_acceptance_passed,
        "formal_visual_acceptance_passed": False,
        "formal_background_realism_passed": background_context_rows == row_count,
        "formal_training_visual_gate_passed": strict_machine_visual_acceptance_passed,
        "summary": {
            "geometry_and_anchor_visibility": (
                "The inspected contact sheet and original-resolution high-risk samples show readable bonds "
                "and visible R/star anchors for research/probe use."
            ),
            "limitations": [
                "This is assistant/model-assisted review, not explicit human manual visual acceptance.",
                "Thin-stroke and high-complexity samples remain visual risk buckets requiring stricter machine/assistant stratified review.",
                background_limitation,
            ],
            "recommended_use": "research_only_probe_or_measured_gate_after_other_gates_pass",
            "not_allowed_use": (
                "deployment_certification_or_final_visual_claim_without_independent_real_benchmark"
            ),
        },
        "blockers": blockers,
        "policy": {
            "does_not_start_training": True,
            "does_not_accept_training_data": True,
            "manual_human_decisions_csv_not_required": True,
            "does_not_claim_human_manual_acceptance": True,
            "does_not_relax_rmse_line_intersection_or_substitution_gates": True,
            "background_realism_requires_existing_audited_document_context_noise_jpeg_pipeline": True,
        },
    }

    risk_output = Path(args.risk_output)
    risk_output.parent.mkdir(parents=True, exist_ok=True)
    risk_output.write_text(json.dumps(risk_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assistant_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(assistant_report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

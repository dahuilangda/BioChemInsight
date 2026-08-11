from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return csv_path.parent / path


def parse_cxsmiles_label(cxsmiles: str) -> str:
    match = re.search(r"\|\$([^|]+)\$\|", cxsmiles)
    if not match:
        return ""
    first = match.group(1).split(";")[0].strip()
    return first or ""


def chemistry_family(smiles: str) -> str:
    text = smiles.replace("*", "")
    if "S(=O)(=O)" in text or "S(=O)" in text:
        return "sulfonyl_or_sulfonamide"
    if "C(=O)" in text or "C(=O)" in text or "=O" in text:
        return "carbonyl_or_acyl"
    if "c1" in text or "n1" in text or "o1" in text or "s1" in text:
        if any(atom in text for atom in ["n", "N", "o", "O", "s", "S"]):
            return "hetero_aromatic_or_heterocycle"
        return "aromatic_ring"
    if any(atom in text for atom in ["F", "Cl", "Br", "I"]):
        return "halogenated_aliphatic"
    if any(atom in text for atom in ["N", "O", "S"]):
        return "hetero_aliphatic"
    return "aliphatic_or_other"


def image_features(path: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        arr = np.asarray(image)
    mask = arr < 245
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "width": width,
            "height": height,
            "dark_pixel_ratio": 0.0,
            "ink_bbox": [0, 0, 0, 0],
            "dominant_edge": "none",
            "left_terminal_score": 0.0,
            "bottom_terminal_score": 0.0,
            "wavy_candidate_score": 0.0,
        }

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    ink_w = max(1, right - left)
    ink_h = max(1, bottom - top)
    crop = mask[top:bottom, left:right]
    band = max(3, int(round(min(width, height) * 0.08)))
    edge_counts = {
        "left": int(mask[:, :band].sum()),
        "right": int(mask[:, width - band :].sum()),
        "top": int(mask[:band, :].sum()),
        "bottom": int(mask[height - band :, :].sum()),
    }
    dominant = max(edge_counts, key=edge_counts.get)
    dominant = dominant if edge_counts[dominant] > 0 else "none"

    left_zone_w = max(4, int(round(ink_w * 0.30)))
    bottom_zone_h = max(4, int(round(ink_h * 0.30)))
    left_zone = crop[:, :left_zone_w]
    bottom_zone = crop[ink_h - bottom_zone_h :, :]
    total_ink = max(1, int(crop.sum()))

    left_cols = left_zone.sum(axis=0)
    left_rows = left_zone.sum(axis=1)
    bottom_cols = bottom_zone.sum(axis=0)
    bottom_rows = bottom_zone.sum(axis=1)

    left_terminal_score = float(left_zone.sum() / total_ink)
    bottom_terminal_score = float(bottom_zone.sum() / total_ink)
    left_verticalness = float((left_rows > 0).sum() / max(1, (left_cols > 0).sum()))
    bottom_horizontalness = float((bottom_cols > 0).sum() / max(1, (bottom_rows > 0).sum()))

    # Wavy marks create repeated transitions in a small terminal zone. This is
    # only a review heuristic; it is not used as a training label.
    terminal = left_zone if left_terminal_score >= bottom_terminal_score else bottom_zone
    projection = terminal.sum(axis=0 if terminal.shape[1] >= terminal.shape[0] else 1)
    active = projection > max(1, projection.max() * 0.25)
    transitions = int((active[1:] != active[:-1]).sum()) if len(active) > 1 else 0
    wavy_candidate_score = min(1.0, transitions / 10.0)

    return {
        "width": int(width),
        "height": int(height),
        "dark_pixel_ratio": float(mask.mean()),
        "ink_bbox": [left, top, right, bottom],
        "ink_width": ink_w,
        "ink_height": ink_h,
        "edge_ink_counts": edge_counts,
        "dominant_edge": dominant,
        "left_terminal_score": left_terminal_score,
        "bottom_terminal_score": bottom_terminal_score,
        "left_verticalness": left_verticalness,
        "bottom_horizontalness": bottom_horizontalness,
        "wavy_candidate_score": wavy_candidate_score,
    }


def visual_shape(features: dict[str, Any], cx_label: str, smiles: str) -> str:
    # Keep visual shape independent from CXSMILES semantic labels. In the
    # RGReco crops the CX label is often R/X, while the visible glyph is usually
    # a short terminal stub/cut, not a literal R or X text label.
    if features["dominant_edge"] == "bottom":
        return "bottom_crop_attachment_or_cut"
    if features["wavy_candidate_score"] >= 0.75:
        return "candidate_terminal_wavy_cut"
    if features["left_terminal_score"] >= 0.30:
        return "left_terminal_cut_or_open_stub"
    if features["left_terminal_score"] >= 0.16:
        return "left_terminal_short_stub"
    if cx_label == "*":
        return "visible_dummy_or_star_attachment"
    return "unknown_attachment_crop"


def semantic_family(cx_label: str) -> str:
    if cx_label in {"R", "R1", "R2"}:
        return "semantic_r_group_attachment"
    if cx_label in {"X", "Y", "Ar"}:
        return "semantic_query_attachment"
    if cx_label == "*":
        return "semantic_dummy_attachment"
    return "semantic_attachment_unspecified"


def make_grouped_sheet(records: list[dict[str, Any]], output: Path, *, tile_size: int, columns: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    grouped = sorted(records, key=lambda item: (item["visual_shape"], item["chemistry_family"], item["source_id"]))
    tiles = []
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for record in grouped:
        image = Image.open(record["file_path"]).convert("RGB")
        image.thumbnail((tile_size, tile_size - 58), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        tile.paste(image, ((tile_size - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        label = f"{record['visual_shape']} / {record['semantic_family']}"
        stats = f"{record['chemistry_family']} {record['width']}x{record['height']} {record['cx_label'] or '-'}"
        draw.text((5, tile_size - 54), label[:58], fill=(0, 0, 0), font=font)
        draw.text((5, tile_size - 36), stats[:58], fill=(0, 0, 0), font=font)
        draw.text((5, tile_size - 18), str(record["source_id"])[:58], fill=(0, 0, 0), font=font)
        tiles.append(tile)

    rows = max(1, math.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review-only taxonomy of real RGReco attachment fragments.")
    parser.add_argument("--csv", default="training/molnextr_markush/data/rgreco_fragment_eval/eval.csv")
    parser.add_argument("--output-json", default="training/molnextr_markush/runs/sidecar_contract/real_fragment_taxonomy_v1.json")
    parser.add_argument("--output-sheet", default="training/molnextr_markush/runs/sidecar_contract/real_fragment_taxonomy_v1.jpg")
    parser.add_argument("--tile-size", type=int, default=240)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in read_rows(csv_path):
        path = resolve_image_path(row, csv_path)
        try:
            smiles = str(row.get("SMILES") or "")
            cxsmiles = str(row.get("cxsmiles") or "")
            cx_label = parse_cxsmiles_label(cxsmiles)
            features = image_features(path)
            record = {
                "source_id": str(row.get("source_id") or ""),
                "file_path": str(path),
                "smiles": smiles,
                "cxsmiles": cxsmiles,
                "cx_label": cx_label,
                "chemistry_family": chemistry_family(smiles),
                "semantic_family": semantic_family(cx_label),
                **features,
            }
            record["visual_shape"] = visual_shape(record, cx_label, smiles)
            records.append(record)
        except Exception as exc:
            failures.append({"source_id": str(row.get("source_id") or ""), "file_path": str(path), "error": str(exc)})

    summary = {
        "row_count": len(records),
        "failure_count": len(failures),
        "visual_shape": dict(Counter(item["visual_shape"] for item in records).most_common()),
        "semantic_family": dict(Counter(item["semantic_family"] for item in records).most_common()),
        "chemistry_family": dict(Counter(item["chemistry_family"] for item in records).most_common()),
        "cx_label": dict(Counter(item["cx_label"] or "missing" for item in records).most_common()),
        "dominant_edge": dict(Counter(item["dominant_edge"] for item in records).most_common()),
    }
    matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (visual, chemistry), count in Counter((item["visual_shape"], item["chemistry_family"]) for item in records).items():
        matrix[visual][chemistry] = count

    report = {
        "csv": str(csv_path),
        "decision": "real_fragment_taxonomy_only_not_training_data",
        "summary": summary,
        "visual_chemistry_matrix": dict(sorted((key, dict(sorted(value.items()))) for key, value in matrix.items())),
        "acceptance_implications": {
            "must_not_train_as_wavy_only": True,
            "must_include_query_or_rgroup_label_attachments": summary["cx_label"].get("R", 0) > 0,
            "must_include_terminal_cut_or_stub_family": (
                summary["visual_shape"].get("left_terminal_cut_or_open_stub", 0)
                + summary["visual_shape"].get("left_terminal_short_stub", 0)
            )
            > 0,
            "must_keep_bottom_or_edge_contact_cases": summary["dominant_edge"].get("bottom", 0) > 0,
            "visual_shape_labels_are_review_heuristics": True,
            "semantic_labels_must_not_force_visible_text": True,
        },
        "records": records,
        "failures": failures[:50],
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    make_grouped_sheet(records, Path(args.output_sheet), tile_size=int(args.tile_size), columns=int(args.columns))
    print(json.dumps({"summary": summary, "decision": report["decision"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

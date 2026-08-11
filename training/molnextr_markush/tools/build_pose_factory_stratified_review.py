from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def image_path_for(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def candidate_plan_index(row: dict[str, str], quality: dict[str, Any]) -> str:
    value = quality.get("candidate_plan_index")
    if value is None or str(value).strip() == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def row_identity(row: dict[str, str], quality: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or "").strip(),
        str(quality.get("source_record_id") or "").strip(),
        candidate_plan_index(row, quality),
    )


def recovered_identities(path: str) -> set[tuple[str, str, str]]:
    if not path:
        return set()
    csv_path = Path(path)
    identities: set[tuple[str, str, str]] = set()
    for row in read_rows(csv_path):
        quality = parse_quality(row)
        identities.add(row_identity(row, quality))
    return identities


def add_reservoir_sample(
    samples: list[tuple[int, dict[str, str], dict[str, Any]]],
    item: tuple[int, dict[str, str], dict[str, Any]],
    *,
    seen_count: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(samples) < limit:
        samples.append(item)
        return
    replace_index = rng.randrange(seen_count)
    if replace_index < limit:
        samples[replace_index] = item


def bucket_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 12:
        return "6-12"
    if value <= 25:
        return "13-25"
    return "26+"


def rmse_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= 0.002:
        return "0-0.002"
    if value <= 0.005:
        return "0.002-0.005"
    if value <= 0.02:
        return "0.005-0.02"
    if value <= 0.05:
        return "0.02-0.05"
    return ">0.05"


def numeric_bucket(value: Any, *, low: float, high: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if number < low:
        return f"<{low:g}"
    if number > high:
        return f">{high:g}"
    return f"{low:g}-{high:g}"


def fragment_strata(row: dict[str, str], quality: dict[str, Any], *, source_label: str = "") -> list[str]:
    values = [
        f"mode:{quality.get('attachment_render_mode') or row.get('attachment_render_mode') or 'missing'}",
        f"visual:{quality.get('visual_shape') or 'missing'}",
        f"chemistry:{quality.get('chemistry_family') or 'missing'}",
        f"side:{quality.get('attachment_direction') or row.get('endpoint_side') or 'missing'}",
    ]
    augmentations = quality.get("style_augmentations") if isinstance(quality.get("style_augmentations"), list) else []
    for item in augmentations:
        values.append(f"augmentation:{item}")
    if source_label:
        values.append(f"source:{source_label}")
    return values


def markush_strata(row: dict[str, str], quality: dict[str, Any], *, source_label: str = "") -> list[str]:
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
    realism = quality.get("document_realism") if isinstance(quality.get("document_realism"), dict) else {}
    realism_parameters = realism.get("render_parameters") if isinstance(realism.get("render_parameters"), dict) else {}
    cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
    variable_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and str(cell.get("text") or "") not in {"", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
    ]
    try:
        rmse = float(pose_mapping.get("line_constraint_rmse_svg_units"))
    except (TypeError, ValueError):
        rmse = None
    values = [
        f"cell_count:{bucket_count(len(cells))}",
        f"variable_cells:{bucket_count(len(variable_cells))}",
        f"r_tags:{bucket_count(int(markush.get('r_tag_count') or 0))}",
        f"dummy_count:{bucket_count(int(markush.get('dummy_count') or 0))}",
        f"line_rmse:{rmse_bucket(rmse)}",
        f"backend:{quality.get('backend') or 'missing'}",
        f"realism_policy:{realism.get('policy') or 'missing'}",
        f"realism_status:{realism.get('status') or 'missing'}",
        f"realism_source:{realism.get('source_dataset') or quality.get('source_dataset') or 'missing'}",
        f"font:{realism_parameters.get('font_name') or 'missing'}",
        f"font_size:{numeric_bucket(realism_parameters.get('font_size'), low=8.0, high=14.0)}",
        f"stroke_ratio:{numeric_bucket(realism_parameters.get('stroke_ratio'), low=0.6, high=1.8)}",
        f"carbon_symbols:{realism_parameters.get('render_carbon_symbols', 'missing')}",
    ]
    if source_label:
        values.append(f"source:{source_label}")
    return values


def make_tile(sample: dict[str, str], image_root: Path, tile_size: int) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_root / sample["review_image"]).convert("RGB")
    image.thumbnail((tile_size, tile_size - 58), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    tile.paste(image, ((tile_size - image.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((5, tile_size - 54), sample["stratum"][:54], fill=(0, 0, 0), font=font)
    draw.text((5, tile_size - 36), sample["summary"][:54], fill=(0, 0, 0), font=font)
    draw.text((5, tile_size - 18), sample["source_id"][:54], fill=(0, 0, 0), font=font)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified visual review assets for pose-factory candidates.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--kind", choices=["fragment", "markush"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument(
        "--recovered-csv",
        default="",
        help="Optional recovered-row CSV. Matching rows get source:recovered stratum coverage in the review package.",
    )
    args = parser.parse_args()

    from PIL import Image

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    recovered = recovered_identities(str(args.recovered_csv or ""))
    rng = random.Random(int(args.seed))
    strata_samples: dict[str, list[tuple[int, dict[str, str], dict[str, Any]]]] = defaultdict(list)
    strata_counts: dict[str, int] = defaultdict(int)
    source_label_counts: dict[str, int] = defaultdict(int)
    input_row_count = 0
    for index, row in enumerate(iter_rows(csv_path), start=2):
        input_row_count += 1
        quality = parse_quality(row)
        source_label = "recovered" if row_identity(row, quality) in recovered else "base" if recovered else ""
        if source_label:
            source_label_counts[source_label] += 1
        keys = (
            fragment_strata(row, quality, source_label=source_label)
            if args.kind == "fragment"
            else markush_strata(row, quality, source_label=source_label)
        )
        for key in keys:
            strata_counts[key] += 1
            add_reservoir_sample(
                strata_samples[key],
                (index, dict(row), quality),
                seen_count=int(strata_counts[key]),
                limit=int(args.samples_per_stratum),
                rng=rng,
            )

    samples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for stratum in sorted(strata_samples):
        candidates = list(strata_samples[stratum])
        rng.shuffle(candidates)
        for row_index, row, quality in candidates:
            source_id = str(row.get("source_id") or f"row{row_index}")
            image_path = image_path_for(row, csv_path)
            target_name = f"{len(samples):05d}_{image_path.name}"
            target_path = image_dir / target_name
            if not target_path.exists():
                shutil.copy2(image_path, target_path)
            if args.kind == "fragment":
                summary = " ".join(
                    [
                        str(quality.get("attachment_render_mode") or row.get("attachment_render_mode") or ""),
                        str(quality.get("visual_shape") or ""),
                        str(quality.get("chemistry_family") or ""),
                        str(quality.get("attachment_direction") or row.get("endpoint_side") or ""),
                    ]
                )
            else:
                markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
                pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
                cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
                summary = (
                    f"cells={len(cells)} r={markush.get('r_tag_count', 0)} "
                    f"line_rmse={pose_mapping.get('line_constraint_rmse_svg_units', '')} "
                    f"line_p95={pose_mapping.get('line_constraint_abs_p95_svg_units', '')}"
                )
            key = (stratum, source_id)
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                {
                    "stratum": stratum,
                    "row_index": str(row_index),
                    "source_id": source_id,
                    "review_image": str(Path("images") / target_name),
                    "source_image": str(image_path),
                    "summary": summary,
                }
            )

    samples_csv = output_dir / "samples.csv"
    fieldnames = ["stratum", "row_index", "source_id", "review_image", "source_image", "summary"]
    with samples_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    tiles = [make_tile(sample, output_dir, int(args.tile_size)) for sample in samples]
    columns = max(1, int(args.columns))
    sheet_row_count = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * int(args.tile_size), sheet_row_count * int(args.tile_size)), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * int(args.tile_size), (index // columns) * int(args.tile_size)))
    sheet_path = output_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    html_rows = []
    for sample in samples:
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(sample['stratum'])}</td>"
            f"<td>{html.escape(sample['row_index'])}</td>"
            f"<td>{html.escape(sample['source_id'])}</td>"
            f"<td>{html.escape(sample['summary'])}</td>"
            f"<td><img src='{html.escape(sample['review_image'])}' style='max-width:220px;max-height:180px'></td>"
            "</tr>"
        )
    html_path = output_dir / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Pose Factory Review</title>"
        "<style>body{font-family:sans-serif}td{border:1px solid #ddd;padding:4px;vertical-align:top}"
        "table{border-collapse:collapse}</style>"
        f"<h1>{html.escape(args.kind)} stratified review</h1>"
        f"<p>CSV: {html.escape(str(csv_path))}</p>"
        "<table><thead><tr><th>stratum</th><th>row</th><th>source</th><th>summary</th><th>image</th></tr></thead><tbody>"
        + "\n".join(html_rows)
        + "</tbody></table>",
        encoding="utf-8",
    )

    report = {
        "csv": str(csv_path),
        "kind": args.kind,
        "output_dir": str(output_dir),
        "row_count": input_row_count,
        "sample_count": len(samples),
        "unique_sampled_source_ids": len({sample["source_id"] for sample in samples}),
        "stratum_count": len(strata_counts),
        "streaming_reservoir_sampling": True,
        "recovered_csv": str(args.recovered_csv or ""),
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "samples_csv": str(samples_csv),
        "contact_sheet": str(sheet_path),
        "html": str(html_path),
        "strata": {key: int(value) for key, value in sorted(strata_counts.items())},
        "status": "visual_review_package_only_does_not_accept_training_data",
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

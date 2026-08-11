from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def iter_parquet_rows(raw_root: Path, subset_globs: list[str], limit: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for subset_glob in subset_globs:
        for path in sorted(raw_root.glob(subset_glob)):
            schema = set(pq.read_schema(path).names)
            columns = [
                name
                for name in ["id", "page_image_path", "annotation", "cxsmiles_dataset", "cxsmiles", "cxsmiles_opt", "cells", "page_image"]
                if name in schema
            ]
            table = pq.read_table(path, columns=columns)
            for item in table.to_pylist():
                annotation = str(item.get("annotation") or "")
                cxsmiles = str(item.get("cxsmiles") or item.get("cxsmiles_dataset") or item.get("cxsmiles_opt") or "")
                cells = item.get("cells")
                cell_count = len(cells) if hasattr(cells, "__len__") else 0
                if "<r>" not in annotation and "*" not in cxsmiles and cell_count == 0:
                    continue
                item["_source_file"] = str(path)
                item["_subset"] = path.parent.name
                rows.append(item)
                if len(rows) >= limit:
                    return rows
    return rows


def image_from_page_image(value: Any):
    from PIL import Image

    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError("page_image bytes missing")


def normalize_cell(cell: Any) -> dict[str, Any] | None:
    if not isinstance(cell, dict):
        return None
    bbox = cell.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None
    try:
        box = [float(value) for value in bbox]
    except Exception:
        return None
    return {"bbox": box, "text": str(cell.get("text") or "")}


def parse_markush_labels(annotation: str, cxsmiles: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    r_tags = re.findall(r"<r>(.*?)</r>", annotation)
    cell_texts = [str(cell.get("text") or "") for cell in cells]
    cx_labels = []
    match = re.search(r"\|\$([^|]+)\$\|", cxsmiles)
    if match:
        cx_labels = [part.strip() for part in match.group(1).split(";") if part.strip()]
    labels = [label for label in r_tags + cell_texts + cx_labels if label]
    return {
        "r_tag_count": len(r_tags),
        "dummy_count": cxsmiles.count("*"),
        "cell_count": len(cells),
        "labels": labels,
        "label_count": len(labels),
        "has_bracket_or_parenthesis": any(char in annotation for char in ["<", ">", "(", ")", "[", "]"]),
    }


def draw_boxed_image(image: Any, cells: list[dict[str, Any]], max_side: int) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(box, outline=(220, 0, 0), width=max(2, width // 350))
        text = str(cell.get("text") or "")
        if text:
            draw.text((box[0], max(0, box[1] - 12)), text[:12], fill=(0, 0, 180), font=font)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def make_sheet(records: list[dict[str, Any]], output: Path, *, tile_size: int, columns: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tiles = []
    for record in records:
        image = Image.open(record["boxed_image_path"]).convert("RGB")
        image.thumbnail((tile_size, tile_size - 56), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        tile.paste(image, ((tile_size - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        label = f"{record['subset']} cells={record['cell_count']} dummy={record['dummy_count']} labels={record['label_count']}"
        draw.text((5, tile_size - 52), label[:60], fill=(0, 0, 0), font=font)
        draw.text((5, tile_size - 34), str(record["source_id"])[:60], fill=(0, 0, 0), font=font)
        draw.text((5, tile_size - 16), str(record["top_labels"])[:60], fill=(0, 0, 0), font=font)
        tiles.append(tile)

    rows = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review-only Markush layout/OCR target profile from protected MG2 raw data.")
    parser.add_argument("--raw-root", default="training/molnextr_markush/data/raw/markushgrapher2")
    parser.add_argument("--subset-glob", action="append", default=["ip5-markush/*.parquet", "m2s/*.parquet", "uspto-markush/*.parquet"])
    parser.add_argument("--output-dir", default="training/molnextr_markush/runs/sidecar_contract/markush_layout_target_profile_v1")
    parser.add_argument("--max-rows", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=260)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images_boxed"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = iter_parquet_rows(Path(args.raw_root), list(args.subset_glob), int(args.max_rows))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        source_id = f"mg2:{row.get('_subset')}:{row.get('id')}"
        try:
            cells = [cell for cell in (normalize_cell(cell) for cell in (row.get("cells") or [])) if cell]
            cxsmiles = str(row.get("cxsmiles") or row.get("cxsmiles_dataset") or row.get("cxsmiles_opt") or "")
            annotation = str(row.get("annotation") or "")
            labels = parse_markush_labels(annotation, cxsmiles, cells)
            image = image_from_page_image(row.get("page_image"))
            boxed = draw_boxed_image(image, cells, max_side=512)
            image_path = images_dir / f"{index:04d}.jpg"
            boxed.save(image_path, quality=90)
            records.append(
                {
                    "source_id": source_id,
                    "subset": str(row.get("_subset") or ""),
                    "source_file": str(row.get("_source_file") or ""),
                    "page_image_path": str(row.get("page_image_path") or ""),
                    "boxed_image_path": str(image_path),
                    "cxsmiles": cxsmiles,
                    "annotation": annotation,
                    "cells": cells,
                    "top_labels": labels["labels"][:12],
                    **{key: value for key, value in labels.items() if key != "labels"},
                    "image_width": image.width,
                    "image_height": image.height,
                }
            )
        except Exception as exc:
            failures.append({"source_id": source_id, "error": str(exc)})

    summary = {
        "row_count": len(records),
        "failure_count": len(failures),
        "subset": dict(Counter(record["subset"] for record in records).most_common()),
        "cell_count_bucket": dict(Counter("0" if record["cell_count"] == 0 else "1-5" if record["cell_count"] <= 5 else "6-15" if record["cell_count"] <= 15 else "16+" for record in records).most_common()),
        "dummy_count_bucket": dict(Counter("0" if record["dummy_count"] == 0 else "1-3" if record["dummy_count"] <= 3 else "4-8" if record["dummy_count"] <= 8 else "9+" for record in records).most_common()),
        "r_tag_count_bucket": dict(Counter("0" if record["r_tag_count"] == 0 else "1-3" if record["r_tag_count"] <= 3 else "4-8" if record["r_tag_count"] <= 8 else "9+" for record in records).most_common()),
    }
    report = {
        "decision": "markush_layout_target_profile_only_not_training_data",
        "raw_root": str(args.raw_root),
        "subset_glob": list(args.subset_glob),
        "summary": summary,
        "records": records,
        "failures": failures[:50],
        "training_policy": {
            "not_pose_trainable": True,
            "reason": "MG2 page images and OCR cells are valid Markush layout targets, but atom coordinates in page-image space are not yet proven for MolNexTR pose supervision.",
            "next_required_step": "Build CDK/MarkushGenerator pose-mapped Markush rows or a robust page-to-graph coordinate mapping before declaring trainable markush_layout rows.",
        },
    }
    report_path = output_dir / "profile.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    make_sheet(records, output_dir / "review_sheet.jpg", tile_size=int(args.tile_size), columns=int(args.columns))
    print(json.dumps({"summary": summary, "decision": report["decision"], "report": str(report_path)}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

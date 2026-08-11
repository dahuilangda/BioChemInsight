from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_path_for(row: dict[str, str], csv_path: Path) -> Path:
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


def edge_contact(mask: Any) -> dict[str, Any]:
    import numpy as np

    height, width = mask.shape
    band = max(1, int(round(min(height, width) * 0.08)))
    counts = {
        "left": int(mask[:, :band].sum()),
        "right": int(mask[:, width - band :].sum()),
        "top": int(mask[:band, :].sum()),
        "bottom": int(mask[height - band :, :].sum()),
    }
    total = max(1, int(mask.sum()))
    ratios = {key: value / total for key, value in counts.items()}
    side = max(counts, key=counts.get)
    return {
        "edge_band_px": band,
        "edge_ink_counts": counts,
        "edge_ink_ratios": ratios,
        "dominant_edge": side if counts[side] > 0 else "none",
    }


def classify_mark(row: dict[str, str]) -> str:
    smiles = str(row.get("SMILES") or row.get("cxsmiles") or "")
    if "*" in smiles:
        return "dummy_attachment_real_crop"
    return "unknown_fragment"


def bucket(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value <= cut:
            return label
    return labels[-1]


def image_stats(path: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        arr = np.asarray(image)
    mask = arr < 245
    dark_ratio = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs) == 0:
        ink_bbox = [0, 0, 0, 0]
        bbox_area_ratio = 0.0
    else:
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        ink_bbox = [left, top, right, bottom]
        bbox_area_ratio = ((right - left) * (bottom - top)) / max(1, width * height)
    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": width / max(1, height),
        "dark_pixel_ratio": dark_ratio,
        "ink_bbox": ink_bbox,
        "ink_bbox_area_ratio": bbox_area_ratio,
        **edge_contact(mask),
    }


def make_sheet(records: list[dict[str, Any]], output: Path, *, tile_size: int, columns: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for record in records:
        image = Image.open(record["file_path"]).convert("RGB")
        image.thumbnail((tile_size, tile_size - 48), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        x = (tile_size - image.width) // 2
        tile.paste(image, (x, 0))
        draw = ImageDraw.Draw(tile)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        label = f"{record['dominant_edge']} {record['width']}x{record['height']} ink={record['dark_pixel_ratio']:.3f}"
        draw.text((5, tile_size - 43), label[:54], fill=(0, 0, 0), font=font)
        draw.text((5, tile_size - 24), str(record["source_id"])[:54], fill=(0, 0, 0), font=font)
        tiles.append(tile)

    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, max(1, rows) * tile_size), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile real held-out fragment crops as visual targets for synthetic renderer reset.")
    parser.add_argument("--csv", default="training/molnextr_markush/data/rgreco_fragment_eval/eval.csv")
    parser.add_argument("--output-json", default="training/molnextr_markush/runs/sidecar_contract/real_fragment_target_profile_v1.json")
    parser.add_argument("--output-sheet", default="training/molnextr_markush/runs/sidecar_contract/real_fragment_target_profile_v1.jpg")
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    records = []
    failures = []
    for row in read_rows(csv_path):
        path = image_path_for(row, csv_path)
        try:
            stats = image_stats(path)
            records.append(
                {
                    "source_id": str(row.get("source_id") or ""),
                    "file_path": str(path),
                    "smiles": str(row.get("SMILES") or ""),
                    "cxsmiles": str(row.get("cxsmiles") or ""),
                    "mark_family": classify_mark(row),
                    **stats,
                    "width_bucket": bucket(stats["width"], [64, 128, 256], ["tiny", "small", "medium", "large"]),
                    "height_bucket": bucket(stats["height"], [64, 128, 256], ["tiny", "small", "medium", "large"]),
                    "ink_bucket": bucket(stats["dark_pixel_ratio"], [0.03, 0.08, 0.16], ["light", "normal", "dense", "very_dense"]),
                }
            )
        except Exception as exc:
            failures.append({"source_id": str(row.get("source_id") or ""), "file_path": str(path), "error": str(exc)})

    counters = {
        "dominant_edge": Counter(record["dominant_edge"] for record in records),
        "mark_family": Counter(record["mark_family"] for record in records),
        "width_bucket": Counter(record["width_bucket"] for record in records),
        "height_bucket": Counter(record["height_bucket"] for record in records),
        "ink_bucket": Counter(record["ink_bucket"] for record in records),
        "star_count": Counter(str(record["smiles"]).count("*") for record in records),
    }
    report = {
        "csv": str(csv_path),
        "row_count": len(records),
        "failure_count": len(failures),
        "failures": failures[:50],
        "summary": {key: dict(counter.most_common()) for key, counter in counters.items()},
        "records": records,
        "decision": "visual_target_profile_only_not_training_data",
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    make_sheet(records, Path(args.output_sheet), tile_size=int(args.tile_size), columns=int(args.columns))
    print(json.dumps({key: report[key] for key in ["csv", "row_count", "failure_count", "summary", "decision"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

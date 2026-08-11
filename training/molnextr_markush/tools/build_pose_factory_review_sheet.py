from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(sys.maxsize)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def point_from_quality(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, list) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def draw_marker(draw: Any, x: float, y: float, color: tuple[int, int, int], radius: int = 5) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=2)


def image_path_for(row: dict[str, str], csv_path: Path) -> Path:
    path = Path(str(row.get("file_path") or ""))
    return path if path.is_absolute() else csv_path.parent / path


def make_tile(row: dict[str, str], csv_path: Path, tile_size: int, *, draw_markers: bool) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    image_path = image_path_for(row, csv_path)
    image = Image.open(image_path).convert("RGB")
    quality = parse_quality(row)
    endpoint = point_from_quality(quality.get("attachment_endpoint"))
    anchor = point_from_quality(quality.get("attachment_anchor_coord"))

    image.thumbnail((tile_size, tile_size - 52), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    x_offset = (tile_size - image.width) // 2
    y_offset = 0
    tile.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(tile)

    if draw_markers and endpoint:
        draw_marker(draw, x_offset + endpoint[0] * image.width, y_offset + endpoint[1] * image.height, (220, 0, 0))
    if draw_markers and anchor:
        draw_marker(draw, x_offset + anchor[0] * image.width, y_offset + anchor[1] * image.height, (0, 80, 220), radius=4)

    mark_geometry = quality.get("fragment_mark_geometry") if isinstance(quality.get("fragment_mark_geometry"), dict) else {}
    junction = str(mark_geometry.get("wavy_connector_intersection_style") or "")
    label = f"{row.get('endpoint_side','')} {junction} {quality.get('render_style','')}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, tile_size - 45), label[:42], fill=(0, 0, 0), font=font)
    draw.text((6, tile_size - 25), str(row.get("source_id") or "")[:42], fill=(0, 0, 0), font=font)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual review sheet for pose-factory generated rows.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--hide-markers", action="store_true")
    parser.add_argument("--wavy-junction-style", default="", choices=["", "center_cross", "side_touch"])
    args = parser.parse_args()

    from PIL import Image

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    if args.wavy_junction_style:
        filtered_rows = []
        for row in rows:
            quality = parse_quality(row)
            mark_geometry = quality.get("fragment_mark_geometry") if isinstance(quality.get("fragment_mark_geometry"), dict) else {}
            if str(mark_geometry.get("wavy_connector_intersection_style") or "") == args.wavy_junction_style:
                filtered_rows.append(row)
        rows = filtered_rows
    rng = random.Random(int(args.seed))
    if len(rows) > int(args.max_samples):
        rows = rng.sample(rows, int(args.max_samples))

    tiles = [make_tile(row, csv_path, int(args.tile_size), draw_markers=not args.hide_markers) for row in rows]
    columns = max(1, int(args.columns))
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * int(args.tile_size), rows_count * int(args.tile_size)), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * int(args.tile_size)
        y = (index // columns) * int(args.tile_size)
        sheet.paste(tile, (x, y))

    output = Path(args.output) if args.output else csv_path.with_name("review_sheet.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    report = {
        "csv": str(csv_path),
        "output": str(output),
        "sample_count": len(tiles),
        "marker_legend": {
            "red": "attachment_endpoint",
            "blue": "attachment_anchor_coord",
        },
        "status": "visual_review_required_before_formal_training",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

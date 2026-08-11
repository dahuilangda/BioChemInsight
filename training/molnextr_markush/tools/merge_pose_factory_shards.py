from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def counts_for_structure_type(structure_type: str, row_count: int) -> dict[str, int]:
    if structure_type == "attachment_fragment":
        return {"attachment_fragment": row_count}
    if structure_type == "complete_compound":
        return {"complete_compound": row_count}
    if structure_type == "markush_layout":
        return {"markush_layout": row_count}
    return {structure_type or "rows": row_count}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pose-factory CSV shards into a self-contained shard directory.")
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--source-arrow", default="")
    parser.add_argument("--structure-type", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or Path(args.csv[0]).name
    output_csv = output_dir / output_name

    merged: list[dict[str, str]] = []
    fieldnames: list[str] = []
    source_counts: dict[str, int] = {}
    copied_images = 0
    for csv_text in args.csv:
        csv_path = Path(csv_text)
        rows = read_rows(csv_path)
        if rows and not fieldnames:
            fieldnames = list(rows[0].keys())
        source_counts[str(csv_path)] = len(rows)
        for row in rows:
            source_image = row_image_path(row, csv_path)
            if source_image.exists():
                target_image = image_dir / f"{len(merged):07d}_{source_image.name}"
                shutil.copy2(source_image, target_image)
                row = dict(row)
                row["file_path"] = str(Path("images") / target_image.name)
                if "image_path" in row:
                    row["image_path"] = row["file_path"]
                copied_images += 1
            merged.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    manifest = {
        "csv": str(output_csv),
        "input_csv": args.csv,
        "row_count": len(merged),
        "copied_images": copied_images,
        "source_counts": source_counts,
        "source_arrow": args.source_arrow,
        "structure_type": args.structure_type,
        "counts": counts_for_structure_type(str(args.structure_type or ""), len(merged)),
        "status": "merged_candidate_requires_validation_and_review",
        "accepted": False,
        "rejected": False,
        "acceptance": {
            "accepted": False,
            "rejected": False,
            "visual_review_passed": False,
            "source_leak_check_passed": False,
            "real_fragment_taxonomy_alignment_passed": False,
            "pose_mapping_review_passed": False,
            "reason": "Merged candidate; acceptance must be set only after validation, leak, taxonomy/pose, and visual review pass.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

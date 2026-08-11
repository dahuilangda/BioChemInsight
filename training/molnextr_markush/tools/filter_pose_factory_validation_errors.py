from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Remove rows listed as errors in a pose-factory validation report.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    bad_source_ids = {
        str(issue.get("source_id") or "")
        for issue in validation.get("issues", [])
        if issue.get("severity") == "error" and issue.get("source_id")
    }
    rows = read_rows(input_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    kept = []
    rejected = []
    for row in rows:
        if str(row.get("source_id") or "") in bad_source_ids:
            rejected.append(row)
            continue
        kept.append(dict(row))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in kept:
        source_path = row_image_path(row, input_csv)
        relative = Path(str(row.get("file_path") or ""))
        if source_path.exists() and not relative.is_absolute():
            target = output_csv.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != target.resolve():
                shutil.copy2(source_path, target)
                copied += 1

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "validation": str(args.validation),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "copied_images": copied,
        "rejected_source_ids": [str(row.get("source_id") or "") for row in rejected],
        "status": "filtered_validation_errors_requires_revalidation",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

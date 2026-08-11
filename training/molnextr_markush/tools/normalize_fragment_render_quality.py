from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def chemistry_family(smiles: str) -> str:
    text = smiles.replace("*", "")
    if "S(=O)(=O)" in text or "S(=O)" in text:
        return "sulfonyl_or_sulfonamide"
    if "C(=O)" in text or "=O" in text:
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize fragment render_quality metadata without changing images or coordinates.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    fieldnames = list(rows[0].keys()) if rows else []
    changed = 0
    chemistry_counts: dict[str, int] = {}
    for row in rows:
        quality = parse_quality(row)
        if not quality:
            continue
        smiles = str(row.get("SMILES") or row.get("smiles") or "")
        family = str(quality.get("chemistry_family") or "").strip() or chemistry_family(smiles)
        chemistry_counts[family] = chemistry_counts.get(family, 0) + 1
        if quality.get("chemistry_family") != family:
            quality["chemistry_family"] = family
            row["render_quality"] = json.dumps(quality, sort_keys=True)
            changed += 1

    output = Path(args.output) if args.output else csv_path
    write_rows(output, rows, fieldnames)
    report = {
        "csv": str(csv_path),
        "output": str(output),
        "row_count": len(rows),
        "changed_rows": changed,
        "chemistry_family_counts": chemistry_counts,
        "status": "metadata_normalized_images_and_coordinates_unchanged",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

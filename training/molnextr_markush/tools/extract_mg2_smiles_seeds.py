from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


def clean_smiles(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split(" |", 1)[0].strip()
    text = re.sub(r"\[[0-9]+\*\]", "*", text)
    return text


def import_rdkit():
    try:
        from rdkit import Chem
    except Exception:
        return None
    return Chem


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SMILES/CXSMILES seeds from protected MarkushGrapher-2 raw parquet files.")
    parser.add_argument("--raw-root", default="training/molnextr_markush/data/raw/markushgrapher2")
    parser.add_argument("--subset-glob", default="uspto-mol-m-54k/*.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--allow-dummy", action="store_true")
    parser.add_argument("--require-rdkit-parse", action="store_true")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    Chem = import_rdkit()

    raw_root = Path(args.raw_root)
    paths = sorted(Path(path) for path in glob.glob(str(raw_root / args.subset_glob)))
    if not paths:
        raise SystemExit(f"no parquet files matched {raw_root / args.subset_glob}")

    rows = []
    seen = set()
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        columns = [name for name in ["id", "cxsmiles", "cxsmiles_dataset", "cxsmiles_opt", "image_name", "page_image_path", "__filename"] if name in schema_names]
        table = pq.read_table(path, columns=columns)
        for item in table.to_pylist():
            candidates = [
                clean_smiles(item.get("cxsmiles")),
                clean_smiles(item.get("cxsmiles_dataset")),
                clean_smiles(item.get("cxsmiles_opt")),
            ]
            smiles = next((candidate for candidate in candidates if candidate), "")
            if not smiles:
                continue
            if not args.allow_dummy and "*" in smiles:
                continue
            if args.require_rdkit_parse:
                if Chem is None:
                    raise SystemExit("RDKit is required for --require-rdkit-parse")
                if Chem.MolFromSmiles(smiles) is None:
                    continue
            key = smiles
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_id": f"mg2:{path.parent.name}:{item.get('id')}",
                    "source_file": str(path),
                    "image_name": str(item.get("image_name") or item.get("page_image_path") or item.get("__filename") or ""),
                    "SMILES": smiles,
                    "smiles": smiles,
                    "has_dummy": "true" if "*" in smiles else "false",
                }
            )
            if 0 < int(args.max_rows) <= len(rows):
                break
        if 0 < int(args.max_rows) <= len(rows):
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source_id", "source_file", "image_name", "SMILES", "smiles", "has_dummy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "raw_root": str(raw_root),
        "subset_glob": args.subset_glob,
        "output": str(output),
        "row_count": len(rows),
        "allow_dummy": bool(args.allow_dummy),
        "require_rdkit_parse": bool(args.require_rdkit_parse),
        "source_files": [str(path) for path in paths],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)


def import_rdkit():
    from rdkit import RDLogger
    from rdkit import Chem

    RDLogger.DisableLog("rdApp.*")
    return Chem


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=8192)
def canonical_smiles(smiles: str, *, strip_dummy: bool) -> str:
    Chem = import_rdkit()
    text = str(smiles or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    if strip_dummy:
        editable = Chem.RWMol(mol)
        for atom_index in sorted(
            [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0],
            reverse=True,
        ):
            editable.RemoveAtom(atom_index)
        mol = editable.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return ""
    return Chem.MolToSmiles(mol, canonical=True)


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(row: dict[str, str]) -> str:
    quality = parse_quality(row)
    dataset = str(quality.get("source_dataset") or row.get("source_arrow") or "").strip()
    record = str(quality.get("source_record_id") or row.get("source_id") or "").strip()
    return f"{dataset}:{record}" if dataset or record else ""


def collect_generated(csv_paths: list[Path]) -> dict[str, Any]:
    row_count = 0
    for csv_path in csv_paths:
        for row in iter_rows(csv_path):
            row_count += 1
            image_path = row_image_path(row, csv_path)
            item = {
                "csv": str(csv_path),
                "source_id": str(row.get("source_id") or ""),
                "source_key": source_key(row),
                "smiles": str(row.get("SMILES") or row.get("smiles") or ""),
                "canonical_smiles": canonical_smiles(str(row.get("SMILES") or row.get("smiles") or ""), strip_dummy=False),
                "canonical_backbone_smiles": canonical_smiles(
                    str(row.get("SMILES") or row.get("smiles") or ""),
                    strip_dummy=True,
                ),
                "file_path": str(image_path),
                "image_sha1": sha1_file(image_path) if image_path.exists() else "",
            }
            yield item


def collect_generated_counters(csv_paths: list[Path]) -> dict[str, Any]:
    row_count = 0
    source_keys: Counter[str] = Counter()
    canonical_smiles_counter: Counter[str] = Counter()
    canonical_backbone_counter: Counter[str] = Counter()
    image_sha1: Counter[str] = Counter()
    for item in collect_generated(csv_paths):
        row_count += 1
        if item["source_key"]:
            source_keys[item["source_key"]] += 1
        if item["canonical_smiles"]:
            canonical_smiles_counter[item["canonical_smiles"]] += 1
        if item["canonical_backbone_smiles"]:
            canonical_backbone_counter[item["canonical_backbone_smiles"]] += 1
        if item["image_sha1"]:
            image_sha1[item["image_sha1"]] += 1
    return {
        "row_count": row_count,
        "source_keys": source_keys,
        "canonical_smiles": canonical_smiles_counter,
        "canonical_backbone_smiles": canonical_backbone_counter,
        "image_sha1": image_sha1,
    }


def collect_heldout(csv_paths: list[Path]) -> dict[str, Any]:
    row_count = 0
    source_ids: Counter[str] = Counter()
    source_gold: Counter[str] = Counter()
    canonical_smiles_counter: Counter[str] = Counter()
    canonical_backbone_counter: Counter[str] = Counter()
    image_sha1: Counter[str] = Counter()
    for csv_path in csv_paths:
        for row in iter_rows(csv_path):
            row_count += 1
            image_path = row_image_path(row, csv_path)
            smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "")
            source_id = str(row.get("source_id") or "")
            source_gold_value = str(row.get("source_gold") or "")
            full = canonical_smiles(smiles, strip_dummy=False)
            backbone = canonical_smiles(smiles, strip_dummy=True)
            image_digest = sha1_file(image_path) if image_path.exists() else ""
            if source_id:
                source_ids[source_id] += 1
            if source_gold_value:
                source_gold[source_gold_value] += 1
            if full:
                canonical_smiles_counter[full] += 1
            if backbone:
                canonical_backbone_counter[backbone] += 1
            if image_digest:
                image_sha1[image_digest] += 1
    return {
        "row_count": row_count,
        "source_ids": source_ids,
        "source_gold": source_gold,
        "canonical_smiles": canonical_smiles_counter,
        "canonical_backbone_smiles": canonical_backbone_counter,
        "image_sha1": image_sha1,
    }


def overlapping_values(left: Counter[str], right: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    overlaps = []
    for key in sorted(set(left) & set(right)):
        overlaps.append({"value": key, "generated_count": int(left[key]), "heldout_count": int(right[key])})
        if len(overlaps) >= limit:
            break
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated pose-factory shards for source leakage into held-out eval sets.")
    parser.add_argument("--generated-csv", action="append", required=True)
    parser.add_argument("--heldout-csv", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-backbone-overlap", action="store_true")
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    generated = collect_generated_counters([Path(path) for path in args.generated_csv])
    heldout = collect_heldout([Path(path) for path in args.heldout_csv])
    limit = int(args.max_examples)

    source_overlaps = overlapping_values(generated["source_keys"], heldout["source_ids"], limit=limit)
    image_overlaps = overlapping_values(generated["image_sha1"], heldout["image_sha1"], limit=limit)
    exact_smiles_overlaps = overlapping_values(generated["canonical_smiles"], heldout["canonical_smiles"], limit=limit)
    backbone_overlaps = overlapping_values(
        generated["canonical_backbone_smiles"],
        heldout["canonical_backbone_smiles"],
        limit=limit,
    )

    blockers = []
    if source_overlaps:
        blockers.append("generated source ids overlap held-out source ids")
    if image_overlaps:
        blockers.append("generated image hashes overlap held-out images")
    if exact_smiles_overlaps:
        blockers.append("generated canonical SMILES overlap held-out canonical SMILES")
    if backbone_overlaps and not args.allow_backbone_overlap:
        blockers.append("generated backbone canonical SMILES overlap held-out backbone SMILES")

    report = {
        "generated_csv": args.generated_csv,
        "heldout_csv": args.heldout_csv,
        "generated_rows": int(generated["row_count"]),
        "heldout_rows": int(heldout["row_count"]),
        "allow_backbone_overlap": bool(args.allow_backbone_overlap),
        "streaming_audit": True,
        "source_id_overlaps": source_overlaps,
        "image_sha1_overlaps": image_overlaps,
        "canonical_smiles_overlaps": exact_smiles_overlaps,
        "canonical_backbone_smiles_overlaps": backbone_overlaps,
        "passed": not blockers,
        "blockers": blockers,
        "policy": {
            "source_id": "must not overlap held-out source_id",
            "image_sha1": "must not duplicate held-out image bytes",
            "canonical_smiles": "must not duplicate held-out full canonical SMILES",
            "canonical_backbone_smiles": "blocked by default because fragment dummy removal can reveal held-out fragment backbones",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

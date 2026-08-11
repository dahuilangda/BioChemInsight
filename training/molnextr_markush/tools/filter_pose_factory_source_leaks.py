from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(sys.maxsize)


def import_rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def csv_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def write_header(path: Path, fieldnames: list[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer._filter_pose_factory_handle = handle  # type: ignore[attr-defined]
    return writer


def copy_row_images(rows: list[dict[str, str]], input_csv: Path, output_csv: Path) -> int:
    copied = 0
    for row in rows:
        source_path = row_image_path(row, input_csv)
        if not source_path.exists():
            continue
        raw_output = str(row.get("file_path") or row.get("image_path") or "").strip()
        if not raw_output:
            continue
        relative_output = Path(raw_output)
        if relative_output.is_absolute():
            continue
        target_path = output_csv.parent / relative_output
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
            copied += 1
    return copied


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
        dummy_indices = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        for atom_index in sorted(dummy_indices, reverse=True):
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


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def infer_structure_type(rows: list[dict[str, str]]) -> str:
    for row in rows:
        quality = parse_quality(row)
        structure_type = str(quality.get("structure_type") or row.get("structure_type") or row.get("structure_type_bucket") or "").strip()
        if structure_type:
            return structure_type
    return ""


def counts_for_structure_type(structure_type: str, row_count: int) -> dict[str, int]:
    if structure_type == "attachment_fragment":
        return {"attachment_fragment": row_count}
    if structure_type == "complete_compound":
        return {"complete_compound": row_count}
    if structure_type == "markush_layout":
        return {"markush_layout": row_count}
    return {structure_type or "rows": row_count}


def source_key(row: dict[str, str]) -> str:
    quality = parse_quality(row)
    dataset = str(quality.get("source_dataset") or row.get("source_arrow") or "").strip()
    record = str(quality.get("source_record_id") or row.get("source_id") or "").strip()
    return f"{dataset}:{record}" if dataset or record else ""


def collect_block_keys(heldout_csvs: list[Path]) -> dict[str, set[str]]:
    keys = {
        "source_ids": set(),
        "image_sha1": set(),
        "canonical_smiles": set(),
        "canonical_backbone_smiles": set(),
    }
    for csv_path in heldout_csvs:
        for row in read_rows(csv_path):
            smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "")
            source_id = str(row.get("source_id") or "").strip()
            source_gold = str(row.get("source_gold") or "").strip()
            if source_id:
                keys["source_ids"].add(source_id)
            if source_gold:
                keys["source_ids"].add(source_gold)
            full = canonical_smiles(smiles, strip_dummy=False)
            backbone = canonical_smiles(smiles, strip_dummy=True)
            if full:
                keys["canonical_smiles"].add(full)
            if backbone:
                keys["canonical_backbone_smiles"].add(backbone)
            image_path = row_image_path(row, csv_path)
            if image_path.exists():
                keys["image_sha1"].add(sha1_file(image_path))
    return keys


def row_block_reasons(row: dict[str, str], csv_path: Path, keys: dict[str, set[str]], *, allow_backbone_overlap: bool) -> list[str]:
    reasons: list[str] = []
    smiles = str(row.get("SMILES") or row.get("smiles") or "")
    full = canonical_smiles(smiles, strip_dummy=False)
    backbone = canonical_smiles(smiles, strip_dummy=True)
    source = source_key(row)
    image_path = row_image_path(row, csv_path)
    image_sha1 = sha1_file(image_path) if image_path.exists() else ""
    if source and source in keys["source_ids"]:
        reasons.append("source_id_overlap")
    if image_sha1 and image_sha1 in keys["image_sha1"]:
        reasons.append("image_sha1_overlap")
    if full and full in keys["canonical_smiles"]:
        reasons.append("canonical_smiles_overlap")
    if backbone and backbone in keys["canonical_backbone_smiles"] and not allow_backbone_overlap:
        reasons.append("canonical_backbone_smiles_overlap")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated pose-factory CSV rows that overlap held-out eval sets.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--heldout-csv", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--allow-backbone-overlap", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    fieldnames = csv_fieldnames(input_csv)
    block_keys = collect_block_keys([Path(path) for path in args.heldout_csv])
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    kept_rows = 0
    copied_images = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in iter_rows(input_csv):
            input_rows += 1
            reasons = row_block_reasons(row, input_csv, block_keys, allow_backbone_overlap=bool(args.allow_backbone_overlap))
            if reasons:
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if len(rejected) < 80:
                    rejected.append({"source_id": row.get("source_id") or "", "reasons": reasons})
                continue
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            kept_rows += 1
            source_path = row_image_path(row, input_csv)
            if source_path.exists():
                raw_output = str(row.get("file_path") or row.get("image_path") or "").strip()
                if raw_output:
                    relative_output = Path(raw_output)
                    if not relative_output.is_absolute():
                        target_path = output_csv.parent / relative_output
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        if source_path.resolve() != target_path.resolve():
                            shutil.copy2(source_path, target_path)
                            copied_images += 1
    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "heldout_csv": args.heldout_csv,
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "rejected_rows": input_rows - kept_rows,
        "copied_images": copied_images,
        "reason_counts": reason_counts,
        "rejected_examples": rejected,
        "allow_backbone_overlap": bool(args.allow_backbone_overlap),
        "streaming_filter": True,
        "status": "filtered_requires_validation_source_leak_visual_review_before_acceptance",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.manifest:
        structure_type = ""
        for kept_row in iter_rows(output_csv):
            structure_type = infer_structure_type([kept_row])
            break
        manifest = {
            "source_csv": str(input_csv),
            "csv": str(output_csv),
            "row_count": kept_rows,
            "rejected_rows": input_rows - kept_rows,
            "filter_report": str(report_path),
            "structure_type": structure_type,
            "status": "candidate_filtered_requires_validation_visual_review_and_leak_check",
            "accepted": False,
            "rejected": False,
            "counts": counts_for_structure_type(structure_type, kept_rows),
            "acceptance": {
                "accepted": False,
                "rejected": False,
                "visual_review_passed": False,
                "source_leak_check_passed": False,
                "pose_mapping_review_passed": False,
                "real_fragment_taxonomy_alignment_passed": False,
                "reason": "Filtered candidate; requires validation, source-leak check, taxonomy/pose checks, and manual review.",
            },
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

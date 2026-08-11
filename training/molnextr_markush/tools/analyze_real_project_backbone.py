from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_json_dict(value: Any) -> dict[str, Any]:
    try:
        if value is None or str(value).strip() == "":
            return {}
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def mol_from_smiles(smiles: str, *, sanitize: bool = True) -> Chem.Mol | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text, sanitize=sanitize)
    if mol is not None:
        return mol
    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def strip_dummy_atoms(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    dummy_indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetAtomicNum() == 0 or "*" in atom.GetSymbol()
    ]
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    stripped = editable.GetMol()
    if stripped.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        try:
            stripped.UpdatePropertyCache(strict=False)
        except Exception:
            return None
    return stripped


def largest_fragment(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return mol
    largest = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        largest.UpdatePropertyCache(strict=False)
    return largest


def canonical_smiles(mol: Chem.Mol | None, *, ignore_stereo: bool = False) -> str:
    if mol is None:
        return ""
    working = Chem.Mol(mol)
    if ignore_stereo:
        Chem.RemoveStereochemistry(working)
    try:
        return Chem.MolToSmiles(working, canonical=True, isomericSmiles=not ignore_stereo)
    except Exception:
        return ""


def element_counts(mol: Chem.Mol | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counts[atom.GetSymbol()] += 1
    return counts


def count_similarity(gold: Counter[str], pred: Counter[str]) -> float:
    total = sum((gold | pred).values())
    if total == 0:
        return 0.0
    return sum((gold & pred).values()) / total


def fingerprint_similarity(gold: Chem.Mol | None, pred: Chem.Mol | None) -> float | None:
    if gold is None or pred is None:
        return None
    try:
        gold_fp = AllChem.GetMorganFingerprintAsBitVect(gold, 2, nBits=2048)
        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(gold_fp, pred_fp))
    except Exception:
        return None


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    old_smiles = str(row.get("old_smiles") or row.get("SMILES") or row.get("target") or "")
    new_smiles = str(row.get("new_smiles") or row.get("predicted_smiles") or "")
    expected_attachment = int(row.get("expected_attachment_count") or 0)
    structure_type = str(row.get("structure_type") or row.get("STRUCTURE_TYPE") or "")
    if not structure_type and expected_attachment > 0:
        structure_type = "fragment"
    render_quality = parse_json_dict(row.get("render_quality"))
    attachment_render_mode = str(render_quality.get("attachment_render_mode") or "")
    gold_mol = strip_dummy_atoms(mol_from_smiles(old_smiles))
    pred_mol = strip_dummy_atoms(mol_from_smiles(new_smiles))
    pred_largest = largest_fragment(pred_mol)

    gold_canon = canonical_smiles(gold_mol)
    pred_canon = canonical_smiles(pred_mol)
    pred_largest_canon = canonical_smiles(pred_largest)
    gold_no_stereo = canonical_smiles(gold_mol, ignore_stereo=True)
    pred_no_stereo = canonical_smiles(pred_mol, ignore_stereo=True)
    pred_largest_no_stereo = canonical_smiles(pred_largest, ignore_stereo=True)

    gold_counts = element_counts(gold_mol)
    pred_counts = element_counts(pred_mol)
    pred_largest_counts = element_counts(pred_largest)
    return {
        "row_index": int(row.get("row_index") or 0),
        "structure_type": structure_type,
        "source_arrow": str(row.get("source_arrow") or ""),
        "attachment_render_mode": attachment_render_mode,
        "old_smiles": old_smiles,
        "new_smiles": new_smiles,
        "gold_backbone": gold_canon,
        "pred_backbone": pred_canon,
        "pred_largest_backbone": pred_largest_canon,
        "gold_backbone_no_stereo": gold_no_stereo,
        "pred_backbone_no_stereo": pred_no_stereo,
        "pred_largest_backbone_no_stereo": pred_largest_no_stereo,
        "target_valid": bool(gold_canon),
        "pred_valid_after_strip": bool(pred_canon),
        "pred_multi_component_after_strip": "." in pred_canon,
        "backbone_exact": bool(gold_canon and gold_canon == pred_canon),
        "backbone_exact_no_stereo": bool(gold_no_stereo and gold_no_stereo == pred_no_stereo),
        "largest_component_exact": bool(gold_canon and gold_canon == pred_largest_canon),
        "largest_component_exact_no_stereo": bool(
            gold_no_stereo and gold_no_stereo == pred_largest_no_stereo
        ),
        "element_count_similarity": count_similarity(gold_counts, pred_counts),
        "largest_element_count_similarity": count_similarity(gold_counts, pred_largest_counts),
        "fingerprint_similarity": fingerprint_similarity(gold_mol, pred_mol),
        "largest_fingerprint_similarity": fingerprint_similarity(gold_mol, pred_largest),
        "old_attachment_count": int(row.get("old_attachment_count") or 0),
        "new_attachment_count": int(
            row.get("new_attachment_count") or row.get("predicted_attachment_count") or 0
        ),
        "expected_attachment_count": expected_attachment,
        "valid_smiles": row.get("valid_smiles"),
        "multi_component": row.get("multi_component"),
        "quality_issues": row.get("quality_issues") or [],
        "filter_reason": str(row.get("filter_reason") or row.get("STRUCTURE_FILTER_REASON") or ""),
        "segment_file": str(row.get("segment_file") or row.get("SEGMENT_FILE") or ""),
        "image_file": str(row.get("image_file") or row.get("IMAGE_FILE") or ""),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fragment_rows = [row for row in rows if row["structure_type"] == "fragment"]
    markush_rows = [row for row in rows if row["structure_type"] == "markush"]
    wavy_rows = [
        row
        for row in fragment_rows
        if row.get("attachment_render_mode") == "wavy"
        or "wavy" in str(row.get("filter_reason") or "").lower()
    ]

    def metrics(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        if not bucket:
            return {
                "rows": 0,
                "pred_valid_after_strip_rate": 0.0,
                "backbone_exact_rate": 0.0,
                "backbone_exact_no_stereo_rate": 0.0,
                "largest_component_exact_rate": 0.0,
                "largest_component_exact_no_stereo_rate": 0.0,
                "multi_component_after_strip_rate": 0.0,
                "mean_element_count_similarity": 0.0,
                "mean_largest_element_count_similarity": 0.0,
                "mean_fingerprint_similarity": 0.0,
                "mean_largest_fingerprint_similarity": 0.0,
                "attachment_exact_rate": 0.0,
            }
        fp_values = [
            float(row["fingerprint_similarity"])
            for row in bucket
            if row.get("fingerprint_similarity") is not None
        ]
        largest_fp_values = [
            float(row["largest_fingerprint_similarity"])
            for row in bucket
            if row.get("largest_fingerprint_similarity") is not None
        ]
        return {
            "rows": len(bucket),
            "pred_valid_after_strip_rate": mean(
                [1.0 if row["pred_valid_after_strip"] else 0.0 for row in bucket]
            ),
            "backbone_exact_rate": mean([1.0 if row["backbone_exact"] else 0.0 for row in bucket]),
            "backbone_exact_no_stereo_rate": mean(
                [1.0 if row["backbone_exact_no_stereo"] else 0.0 for row in bucket]
            ),
            "largest_component_exact_rate": mean(
                [1.0 if row["largest_component_exact"] else 0.0 for row in bucket]
            ),
            "largest_component_exact_no_stereo_rate": mean(
                [1.0 if row["largest_component_exact_no_stereo"] else 0.0 for row in bucket]
            ),
            "multi_component_after_strip_rate": mean(
                [1.0 if row["pred_multi_component_after_strip"] else 0.0 for row in bucket]
            ),
            "mean_element_count_similarity": mean(
                [float(row["element_count_similarity"]) for row in bucket]
            ),
            "mean_largest_element_count_similarity": mean(
                [float(row["largest_element_count_similarity"]) for row in bucket]
            ),
            "mean_fingerprint_similarity": mean(fp_values),
            "mean_largest_fingerprint_similarity": mean(largest_fp_values),
            "attachment_exact_rate": mean(
                [
                    1.0
                    if int(row["new_attachment_count"]) == int(row["expected_attachment_count"])
                    else 0.0
                    for row in bucket
                ]
            ),
        }

    examples = {
        "fragment_exact": [row for row in fragment_rows if row["backbone_exact_no_stereo"]][:5],
        "fragment_backbone_wrong": [
            row
            for row in sorted(
                fragment_rows,
                key=lambda item: (
                    float(item["largest_element_count_similarity"]),
                    float(item["largest_fingerprint_similarity"] or 0.0),
                ),
            )
            if not row["backbone_exact_no_stereo"]
        ][:8],
        "fragment_attachment_hit_backbone_wrong": [
            row
            for row in fragment_rows
            if int(row["new_attachment_count"]) > 0 and not row["backbone_exact_no_stereo"]
        ][:8],
    }
    return {
        "rows": len(rows),
        "type_counts": {"fragment": len(fragment_rows), "markush": len(markush_rows)},
        "fragment": metrics(fragment_rows),
        "wavy_fragment": metrics(wavy_rows),
        "markush": metrics(markush_rows),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="real_project_eval.results.jsonl or segment CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", default="")
    args = parser.parse_args()

    rows = [analyze_row(row) for row in read_rows(Path(args.input))]
    report = {
        "input": args.input,
        **summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

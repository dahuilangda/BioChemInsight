#!/usr/bin/env python
"""Build the real-USPTO complete-compound training shard (label=0).

Source: MolScribe's uspto_mol train CSV (real patent depiction crops, ~680k
rows, images + MOL-derived SMILES on disk). The complete bucket of the MoE
training mix has no real-domain rows at all; this tool fills it.

Rows are kept only when the SMILES parses and sanitizes (the CSV carries
~25% bracket-token OCR garbage), carries no R-group/wildcard (those belong to
the markush/fragment buckets), fits the chartok vocabulary and the 480-token
budget, and is not on the six-patent contamination blacklist shared with the
real markushgrapher eval split. Coordinates are left empty on purpose: the
CSV's node_coords are MOL depiction coordinates, not image pixels, so rows
train symbol-only (mask_ratio=1) — the same contract real markush rows use
when pose recovery is unavailable.

Output: contract-v3 parquet + report.json under
training/molnextr_markush/data/generated/real_uspto_complete/.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_CONTRACT_VERSION = "molnextr_moe_graph_contract_v3"
SOURCE_ARROW = "real_uspto_complete_v1"
GENERATOR_VERSION = "real_uspto_complete_shard_v1"

# chartok_coords budget: SOS + per-char SMILES + 2 coord tokens per atom + EOS.
MAX_TOKEN_BUDGET = 470

# Patents whose grants appear in both this CSV pool and the official
# markushgrapher eval split (scan report, contamination section).
EVAL_BLACKLIST_PATENTS = frozenset(
    {"6344453", "6444415", "6495573", "6521785", "6538013", "6995262"}
)

PATENT_RE = re.compile(r"US(\d{7,8})-")
BOND_TYPE_CODES = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}

_vocab_cache: set[str] | None = None


def _char_vocab() -> set[str]:
    global _vocab_cache
    if _vocab_cache is None:
        vocab_path = ROOT / "utils" / "MolNexTR" / "vocab" / "vocab_chars.json"
        with open(vocab_path) as handle:
            _vocab_cache = set(json.load(handle))
    return _vocab_cache


def _edges_from_mol(mol) -> list[list[int]]:
    from rdkit import Chem
    edges = []
    for bond in mol.GetBonds():
        code = BOND_TYPE_CODES.get(str(bond.GetBondType()))
        if code is None:
            raise ValueError(f"unsupported bond type {bond.GetBondType()}")
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.GetBondType() == Chem.BondType.AROMATIC and code == 4:
            # edges matrix is symmetric for undirected classes; store once.
            if begin > end:
                begin, end = end, begin
        edges.append([begin, end, code])
    return edges


def process_chunk(args):
    """Filter + convert a slice of CSV rows. Returns (records, counters)."""
    rows, raw_root = args
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    vocab = _char_vocab()
    counters = {
        "total": 0, "missing_image": 0, "blacklisted": 0, "markish_or_wildcard": 0,
        "empty": 0, "rdkit_failed": 0, "vocab_out": 0, "too_long": 0, "kept": 0,
    }
    records = []
    for row in rows:
        counters["total"] += 1
        smiles = (row.get("SMILES") or "").strip()
        rel_path = (row.get("file_path") or "").strip()
        if not smiles or not rel_path:
            counters["empty"] += 1
            continue
        if "*" in smiles or "[R" in smiles.upper().replace("[R", "[R"):
            counters["markish_or_wildcard"] += 1
            continue
        abs_path = str((raw_root / rel_path).resolve())
        if not os.path.exists(abs_path):
            counters["missing_image"] += 1
            continue
        patent = PATENT_RE.search(rel_path)
        if patent and patent.group(1) in EVAL_BLACKLIST_PATENTS:
            counters["blacklisted"] += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            counters["rdkit_failed"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            counters["rdkit_failed"] += 1
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if not canonical or any(ch not in vocab for ch in canonical):
            counters["vocab_out"] += 1
            continue
        n_atoms = mol.GetNumAtoms()
        if 2 + len(canonical) + 2 * n_atoms > MAX_TOKEN_BUDGET:
            counters["too_long"] += 1
            continue
        mol2 = Chem.MolFromSmiles(canonical)
        if mol2 is None or mol2.GetNumAtoms() != n_atoms:
            counters["rdkit_failed"] += 1
            continue
        try:
            edges = _edges_from_mol(mol2)
        except Exception:
            counters["rdkit_failed"] += 1
            continue
        records.append({
            "file_path": abs_path,
            "SMILES": canonical,
            "edges": json.dumps(edges),
            "structure_type_label": 0,
            "structure_type_bucket": "ordinary_structure",
            "data_contract_version": DATA_CONTRACT_VERSION,
            "image_domain": "real_original",
            "source_arrow": SOURCE_ARROW,
            "node_coords": "",
            "node_coords_space": "",
            "coordinate_targets_available": False,
            "coordinate_pose_verified": False,
            "attachment_points": "[]",
            "attachment_points_complete": False,
            "attachment_point_dummy_indices": "[]",
            "source_dataset": "molscribe_uspto_mol",
            "source_image_name": rel_path.split("/")[-1],
            "atom_count": n_atoms,
            "dummy_count": 0,
        })
        counters["kept"] += 1
    return records, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="uspto_mol train CSV (paths relative to --raw-root)")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="directory the CSV file_path values are relative to")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N CSV rows (0 = all)")
    args = parser.parse_args()

    csv.field_size_limit(10 ** 9)
    rows = []
    with open(args.csv, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} CSV rows from {args.csv}")

    chunk = 2000
    chunks = [(rows[i:i + chunk], args.raw_root) for i in range(0, len(rows), chunk)]
    total_counters = {}
    records = []
    t0 = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        for i, (recs, counters) in enumerate(pool.imap_unordered(process_chunk, chunks)):
            records.extend(recs)
            for key, value in counters.items():
                total_counters[key] = total_counters.get(key, 0) + value
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(chunks)} chunks, kept={total_counters['kept']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(records)
    out_parquet = args.out_dir / "data.parquet"
    df.to_parquet(out_parquet, index=False)

    report = {
        "schema_version": "real_uspto_complete_v1",
        "generator_version": GENERATOR_VERSION,
        "source_csv": str(args.csv),
        "rows_total": len(rows),
        "rows_kept": len(df),
        "filter_counters": total_counters,
        "contract": {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "structure_type_label": 0,
            "coordinate_supervision": "symbol_only_mask_ratio_1",
            "stereo_policy": "isomeric canonical SMILES retained (@ and /\\ kept)",
            "edge_classes": "1/2/3/4 from RDKit graph in SMILES atom order",
            "contamination_blacklist": sorted(EVAL_BLACKLIST_PATENTS),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.out_dir / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("rows_total", "rows_kept", "filter_counters", "elapsed_seconds")},
                     indent=2))
    print(f"wrote {out_parquet}")


if __name__ == "__main__":
    main()

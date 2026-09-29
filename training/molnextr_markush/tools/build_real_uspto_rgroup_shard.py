#!/usr/bin/env python
"""Build the real-USPTO R-group training shard (label=1, markush).

Source: MolScribe's uspto_mol train CSV (real patent depiction crops, ~680k
rows). The rows targeted here are the ones the label=0 complete-compound shard
deliberately skipped: SMILES that still contain bracket variable labels
("[R1]", "[X]", "[Ar]", ...) predicted by MolScribe on real patent R-group
depiction crops. They are converted into numbered-dummy markush graphs:

  [R12] -> [12*]   [R1] -> [1*]   [R] -> *
  [X] [Y] [Z] [Q] [A] [E] [Ar] [Het] [Hal] [R'] -> *   (unnumbered, unreliable)
  extensions (subscript-letter / numbered variable letters, same semantics):
  [Ra] [Rb] ... -> *      [X1] [Z2] [A3] ... -> *

Rows whose bracket tokens still hold unmapped variable-shaped text after the
replacement pass (e.g. "[RaH]", "[RO2C]", "[Resin]", "[R(2)]", punctuation
garbage) are dropped and counted: these are OCR-noisy labels we cannot ground
in a dummy graph reliably.

Everything downstream mirrors build_real_uspto_complete_shard.py: RDKit parse
+ sanitize + canonical(isomeric) rewrite, chartok vocab + token budget, edges
from the canonical graph in SMILES atom order, contract-v3 columns, and the
markushgrapher-eval patent blacklist. structure_type_bucket reuses
src/curriculum.py's assign_bucket (markush_fragment / multi_label_scaffold /
variable_atom_or_query_like / single_variable_scaffold_like); after RDKit
canonicalization every wildcard is written either as a bare '*' or as '[n*]',
both of which curriculum.is_markush_token already recognizes, so the profile
is markush-aware for every kept row. Multi-component rows ('[1*]C.[2*]C.'
substituent-list depictions) are kept, matching the existing label=1
real_markushgrapher_ocsr_v2 parquet.

Output: contract-v3 parquet + report.json under
training/molnextr_markush/data/generated/real_uspto_rgroup/.
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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_CONTRACT_VERSION = "molnextr_moe_graph_contract_v3"
SOURCE_ARROW = "real_uspto_rgroup_v1"
GENERATOR_VERSION = "real_uspto_rgroup_shard_v1"
SOURCE_DATASET = "molscribe_uspto_mol"

# chartok_coords budget: SOS + per-char SMILES + 2 coord tokens per atom + EOS.
MAX_TOKEN_BUDGET = 470
MAX_DUMMY_COUNT = 40
MIN_ATOM_COUNT = 3

# Task spec replacement rules (applied to the SMILES text before RDKit).
NUMBERED_R_RE = re.compile(r"\[R(\d{1,3})\]")            # [R1]/[R12] -> [1*]/[12*]
PRIME_R_RE = re.compile(r"\[R'+\]")                      # [R']/[R''] -> *
LETTERED_R_RE = re.compile(r"\[R([a-z])\]")              # [Ra]/[Rb] -> * (subscript letter)
NUMBERED_LETTER_RE = re.compile(r"\[([XYZQAE])(\d{1,2})\]")  # [X1]/[Z2] -> *
PLAIN_VARIABLE_SYMBOLS = ("Ar", "Het", "Hal", "R", "X", "Y", "Z", "Q", "A", "E")

# Residual check: bracket tokens still starting with a variable-letter prefix
# after replacement are unmappable OCR noise (RaH, RO2C, Resin, R(2), ...).
BRACKET_TOKEN_RE = re.compile(r"\[[^\]]*\]")
VARIABLE_PREFIX_RE = re.compile(r"^(?:R|X|Y|Z|Q|A|E)")

PATENT_RE = re.compile(r"US(\d{7,8})-")
# Eval-side extraction tolerates names like "US10899730B2.pdf_16_1.png" while
# refusing publication numbers ("US20100009286A1" -> no 7-8 digit boundary).
EVAL_PATENT_RE = re.compile(r"US(\d{7,8})(?!\d)")

BOND_TYPE_CODES = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}

_vocab_cache: set[str] | None = None
_curriculum_cache = None


def _char_vocab() -> set[str]:
    global _vocab_cache
    if _vocab_cache is None:
        vocab_path = ROOT / "utils" / "MolNexTR" / "vocab" / "vocab_chars.json"
        with open(vocab_path) as handle:
            _vocab_cache = set(json.load(handle))
    return _vocab_cache


def _curriculum():
    global _curriculum_cache
    if _curriculum_cache is None:
        from training.molnextr_markush.src import curriculum

        _curriculum_cache = curriculum
    return _curriculum_cache


def normalize_variable_tokens(smiles: str) -> tuple[str, Counter]:
    """Rewrite bracket variable labels into dummy atoms. Returns (smiles, stats)."""
    stats: Counter = Counter()
    out = smiles

    def _count_numbered_r(match: re.Match) -> str:
        stats["numbered_R_to_numbered_dummy"] += 1
        return f"[{match.group(1)}*]"

    out = NUMBERED_R_RE.sub(_count_numbered_r, out)

    def _count_prime(_match: re.Match) -> str:
        stats["prime_R_to_dummy"] += 1
        return "*"

    out = PRIME_R_RE.sub(_count_prime, out)

    def _count_lettered_r(_match: re.Match) -> str:
        stats["lettered_R_to_dummy"] += 1
        return "*"

    out = LETTERED_R_RE.sub(_count_lettered_r, out)

    def _count_numbered_letter(_match: re.Match) -> str:
        stats["numbered_variable_letter_to_dummy"] += 1
        return "*"

    out = NUMBERED_LETTER_RE.sub(_count_numbered_letter, out)

    for symbol in PLAIN_VARIABLE_SYMBOLS:
        token = f"[{symbol}]"
        if token in out:
            stats[f"plain_{symbol}_to_dummy"] += out.count(token)
            out = out.replace(token, "*")
    return out, stats


def residual_variable_tokens(smiles: str) -> list[str]:
    """Bracket tokens that still look like unmapped variable labels."""
    residual = []
    for token in BRACKET_TOKEN_RE.findall(smiles):
        inner = token[1:-1]
        if VARIABLE_PREFIX_RE.match(inner):
            residual.append(inner)
    return residual


def _edges_from_mol(mol) -> list[list[int]]:
    edges = []
    for bond in mol.GetBonds():
        code = BOND_TYPE_CODES.get(str(bond.GetBondType()))
        if code is None:
            raise ValueError(f"unsupported bond type {bond.GetBondType()}")
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin > end:
            begin, end = end, begin
        edges.append([begin, end, code])
    return edges


def _profile_from_mol(mol, edges: list[list[int]]) -> dict:
    """Dummy-aware version of curriculum.profile_row.

    curriculum.profile_row detects markush tokens through
    utils.markush_labels.is_markush_label, whose RGROUP_SYMBOLS table only
    recognizes numbered dummies up to [12*]; patent R-numbers routinely go
    higher (R14..R41), which would profile as ordinary_structure. Here the
    markush atom set is taken straight from the RDKit graph (atomic number 0),
    so every wildcard counts, and the dict shape stays compatible with
    curriculum.assign_bucket.
    """
    from collections import Counter as _Counter

    markush_indices = {atom.GetIdx() for atom in mol.GetAtoms()
                       if atom.GetAtomicNum() == 0}
    markush_degrees = _Counter()
    for begin, end, _bond_type in edges:
        begin, end = int(begin), int(end)
        if begin in markush_indices:
            markush_degrees[begin] += 1
        if end in markush_indices:
            markush_degrees[end] += 1
    return {
        "atom_count": mol.GetNumAtoms(),
        "markush_atom_count": len(markush_indices),
        "single_neighbor_markush_count": sum(
            1 for value in markush_degrees.values() if value == 1
        ),
        "multi_neighbor_markush_count": sum(
            1 for value in markush_degrees.values() if value > 1
        ),
    }


def backbone_smiles(smiles: str) -> str | None:
    """Canonical SMILES of the dummy-stripped graph (coarse overlap check)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for idx in reversed(range(rw.GetNumAtoms())):
        if rw.GetAtomWithIdx(idx).GetAtomicNum() == 0:
            rw.RemoveAtom(idx)
    if rw.GetNumAtoms() == 0:
        return ""
    trimmed = rw.GetMol()
    try:
        Chem.SanitizeMol(trimmed)
    except Exception:
        pass
    return Chem.MolToSmiles(trimmed, isomericSmiles=True, canonical=True)


def process_chunk(args):
    """Filter + convert a slice of CSV rows. Returns (records, counters)."""
    rows, raw_root, blacklist = args
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    vocab = _char_vocab()
    curriculum = _curriculum()
    counters = {
        "total": 0,
        "missing_image": 0,
        "blacklisted": 0,
        "residual_variable_token": 0,
        "rdkit_failed": 0,
        "no_dummy": 0,
        "too_many_dummies": 0,
        "too_few_atoms": 0,
        "vocab_out": 0,
        "too_long": 0,
        "bucket_fallback": 0,
        "kept": 0,
        "kept_multicomponent": 0,
        "kept_singlecomponent": 0,
    }
    token_stats: Counter = Counter()
    residual_stats: Counter = Counter()
    bucket_stats: Counter = Counter()
    records = []
    seen_paths = set()
    for row in rows:
        counters["total"] += 1
        smiles = (row.get("SMILES") or "").strip()
        rel_path = (row.get("file_path") or "").strip()
        if not smiles or not rel_path:
            counters["empty_row"] = counters.get("empty_row", 0) + 1
            continue
        abs_path = str((raw_root / rel_path).resolve())
        if abs_path in seen_paths:
            counters["duplicate_file_path"] = counters.get("duplicate_file_path", 0) + 1
            continue
        if not os.path.exists(abs_path):
            counters["missing_image"] += 1
            continue
        patent = PATENT_RE.search(rel_path)
        if patent and int(patent.group(1)) in blacklist:
            counters["blacklisted"] += 1
            continue

        replaced, stats = normalize_variable_tokens(smiles)
        token_stats.update(stats)
        residual = residual_variable_tokens(replaced)
        if residual:
            counters["residual_variable_token"] += 1
            for token in sorted(set(residual)):
                residual_stats[token] += 1
            continue

        mol = Chem.MolFromSmiles(replaced)
        if mol is None:
            counters["rdkit_failed"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            counters["rdkit_failed"] += 1
            continue
        dummy_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
        if dummy_count < 1:
            counters["no_dummy"] += 1
            continue
        if dummy_count > MAX_DUMMY_COUNT:
            counters["too_many_dummies"] += 1
            continue
        n_atoms = mol.GetNumAtoms()
        if n_atoms < MIN_ATOM_COUNT:
            counters["too_few_atoms"] += 1
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if not canonical:
            counters["rdkit_failed"] += 1
            continue
        if any(ch not in vocab for ch in canonical):
            counters["vocab_out"] += 1
            continue
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

        profile = _profile_from_mol(mol2, edges)
        bucket = curriculum.assign_bucket(profile)
        if bucket == "ordinary_structure":
            # Cannot happen: every kept row has >=1 dummy (atomic number 0)
            # and the profile above counts them all. Defensive fallback keeps
            # label=1 semantics if that invariant is ever broken.
            counters["bucket_fallback"] += 1
            bucket = "single_variable_scaffold_like"
        bucket_stats[bucket] += 1

        if "." in canonical:
            counters["kept_multicomponent"] += 1
        else:
            counters["kept_singlecomponent"] += 1
        records.append({
            "file_path": abs_path,
            "SMILES": canonical,
            "edges": json.dumps(edges),
            "structure_type_label": 1,
            "structure_type_bucket": bucket,
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
            "source_dataset": SOURCE_DATASET,
            "source_image_name": rel_path.split("/")[-1],
            "atom_count": int(n_atoms),
            "dummy_count": int(dummy_count),
        })
        seen_paths.add(abs_path)
        counters["kept"] += 1
    return records, counters, token_stats, residual_stats, bucket_stats


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return round(ordered[low] * (1 - frac) + ordered[high] * frac, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_680k.csv",
        help="uspto_mol train CSV (paths relative to --raw-root)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "training/molnextr_markush/data/raw/molscribe",
        help="directory the CSV file_path values are relative to",
    )
    parser.add_argument(
        "--eval-parquet",
        type=Path,
        default=ROOT / "training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2/eval/data.parquet",
        help="markushgrapher eval parquet used for the contamination blacklist",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "training/molnextr_markush/data/generated/real_uspto_rgroup",
    )
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N CSV rows (0 = all)")
    parser.add_argument("--overlap-sample", type=int, default=4000,
                        help="rows sampled for the eval backbone overlap check")
    args = parser.parse_args()

    # --- contamination blacklist + eval reference sets ---------------------
    import pandas as pd
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    blacklist: set[int] = set()
    eval_smiles: set[str] = set()
    eval_backbones: set[str] = set()
    if args.eval_parquet.exists():
        eval_df = pd.read_parquet(args.eval_parquet)
        for name in eval_df["source_image_name"].astype(str):
            for patent in EVAL_PATENT_RE.findall(name):
                blacklist.add(int(patent))
        for smi in eval_df["SMILES"].astype(str):
            smi = smi.split(" ")[0].strip()
            if not smi:
                continue
            eval_smiles.add(smi)
            bb = backbone_smiles(smi)
            if bb:
                eval_backbones.add(bb)
    print(f"eval blacklist patents: {len(blacklist)}; "
          f"eval canonical smiles: {len(eval_smiles)}; "
          f"eval backbones: {len(eval_backbones)}")

    # --- load target rows ---------------------------------------------------
    csv.field_size_limit(10 ** 9)
    rows = []
    rows_scanned = 0
    with open(args.csv, newline="") as handle:
        for row in csv.DictReader(handle):
            rows_scanned += 1
            if "[R" in (row.get("SMILES") or ""):
                rows.append(row)
            if args.limit and rows_scanned >= args.limit:
                break
    print(f"scanned {rows_scanned} CSV rows, {len(rows)} target rows with '[R' "
          f"from {args.csv}")

    chunk = 2000
    chunks = [(rows[i:i + chunk], args.raw_root, blacklist)
              for i in range(0, len(rows), chunk)]
    total_counters: dict[str, int] = {}
    token_stats: Counter = Counter()
    residual_stats: Counter = Counter()
    bucket_stats: Counter = Counter()
    records: list[dict] = []
    t0 = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        for i, (recs, counters, toks, resid, buckets) in enumerate(
            pool.imap_unordered(process_chunk, chunks)
        ):
            records.extend(recs)
            for key, value in counters.items():
                total_counters[key] = total_counters.get(key, 0) + value
            token_stats.update(toks)
            residual_stats.update(resid)
            bucket_stats.update(buckets)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  {i + 1}/{len(chunks)} chunks, kept={total_counters['kept']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # Deterministic order (stable reruns): by file_path then SMILES.
    records.sort(key=lambda r: (r["file_path"], r["SMILES"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=[
        "file_path", "SMILES", "edges", "structure_type_label",
        "structure_type_bucket", "data_contract_version", "image_domain",
        "source_arrow", "node_coords", "node_coords_space",
        "coordinate_targets_available", "coordinate_pose_verified",
        "attachment_points", "attachment_points_complete",
        "attachment_point_dummy_indices", "source_dataset",
        "source_image_name", "atom_count", "dummy_count",
    ])
    df = df.astype({
        "structure_type_label": "int64",
        "atom_count": "int64",
        "dummy_count": "int64",
        "coordinate_targets_available": "bool",
        "coordinate_pose_verified": "bool",
        "attachment_points_complete": "bool",
    })
    out_parquet = args.out_dir / "data.parquet"
    df.to_parquet(out_parquet, index=False)

    # --- distributions + overlap spot check --------------------------------
    dummies = df["dummy_count"].astype(int).tolist()
    atoms = df["atom_count"].astype(int).tolist()
    dummy_hist = {str(k): int(v) for k, v in sorted(Counter(dummies).items())}

    overlap_rows = min(args.overlap_sample, len(df))
    sample = df.sample(n=overlap_rows, random_state=0)["SMILES"].tolist() if overlap_rows else []
    backbones = [backbone_smiles(smi) for smi in sample]
    backbones_valid = [bb for bb in backbones if bb]
    bb_overlap = sum(1 for bb in backbones_valid if bb in eval_backbones)
    exact_overlap = (
        sum(1 for smi in df["SMILES"] if smi in eval_smiles) if len(df) else 0
    )

    drop_reasons = {
        key: value for key, value in sorted(total_counters.items())
        if key not in {"total", "kept", "kept_multicomponent", "kept_singlecomponent"}
    }
    report = {
        "schema_version": "real_uspto_rgroup_v1",
        "generator_version": GENERATOR_VERSION,
        "source_csv": str(args.csv),
        "rows_scanned_in_csv": rows_scanned,
        "rows_target_with_R": len(rows),
        "rows_kept": len(df),
        "filter_counters": dict(sorted(total_counters.items())),
        "drop_reasons": drop_reasons,
        "top_residual_variable_tokens": residual_stats.most_common(30),
        "token_replacement_counters": dict(sorted(token_stats.items())),
        "label_distribution": {"1": int(len(df))} if len(df) else {},
        "bucket_distribution": dict(sorted(bucket_stats.items())),
        "component_distribution": {
            "single_component": total_counters.get("kept_singlecomponent", 0),
            "multi_component": total_counters.get("kept_multicomponent", 0),
        },
        "dummy_count_distribution": {
            "p50": _percentile(dummies, 0.50),
            "p90": _percentile(dummies, 0.90),
            "max": int(max(dummies)) if dummies else 0,
            "min": int(min(dummies)) if dummies else 0,
            "mean": round(sum(dummies) / len(dummies), 3) if dummies else 0.0,
            "histogram": dummy_hist,
        },
        "atom_count_distribution": {
            "p50": _percentile(atoms, 0.50),
            "p90": _percentile(atoms, 0.90),
            "max": int(max(atoms)) if atoms else 0,
            "mean": round(sum(atoms) / len(atoms), 3) if atoms else 0.0,
        },
        "contamination": {
            "eval_blacklist_patents": sorted(str(p) for p in blacklist),
            "eval_blacklist_size": len(blacklist),
            "rows_dropped_blacklist": total_counters.get("blacklisted", 0),
            "exact_canonical_smiles_overlap_with_eval": int(exact_overlap),
            "backbone_overlap_check": {
                "sampled": overlap_rows,
                "valid_backbones": len(backbones_valid),
                "overlapping": bb_overlap,
                "overlap_rate": round(bb_overlap / len(backbones_valid), 6)
                if backbones_valid else 0.0,
            },
        },
        "contract": {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "structure_type_label": 1,
            "coordinate_supervision": "symbol_only_mask_ratio_1",
            "stereo_policy": "isomeric canonical SMILES retained (@ and /\\ kept)",
            "edge_classes": "1/2/3/4 from RDKit graph of the canonical SMILES",
            "bucket_rule": "src/curriculum.py assign_bucket on the canonical graph",
            "multi_component_policy": "kept (substituent-list depictions)",
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.out_dir / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)

    summary_keys = ("rows_scanned_in_csv", "rows_target_with_R", "rows_kept",
                    "filter_counters", "bucket_distribution",
                    "component_distribution")
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    print("dummy distribution:", json.dumps(
        {k: report["dummy_count_distribution"][k]
         for k in ("p50", "p90", "max", "mean")}))
    print("contamination:", json.dumps({
        "rows_dropped_blacklist": report["contamination"]["rows_dropped_blacklist"],
        "exact_canonical_overlap": exact_overlap,
        "backbone_overlap": report["contamination"]["backbone_overlap_check"],
    }))
    print(f"wrote {out_parquet} ({len(df)} rows) and report.json "
          f"in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

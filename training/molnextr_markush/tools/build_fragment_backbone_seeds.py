from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["source_id", "SMILES", "smiles", "fragment_seed_method", "parent_smiles", "parent_source_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SOURCE_SMILES_COLUMNS = (
    "SMILES",
    "smiles",
    "cxsmiles",
    "cxsmiles_dataset",
    "cxsmiles_opt",
    "mol",
)
SOURCE_ID_COLUMNS = ("source_id", "id", "image_name", "page_image_path", "__filename")


def clean_source_molecule_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Keep the molecular graph portion; CXSMILES annotations are useful source
    # metadata but RDKit seed decomposition should start from the base graph.
    text = text.split(" |", 1)[0].strip()
    text = re.sub(r"\[[0-9]+\*\]", "*", text)
    return text


def best_source_molecule(row: dict[str, Any]) -> tuple[str, str]:
    for column in SOURCE_SMILES_COLUMNS:
        value = clean_source_molecule_text(row.get(column))
        if value:
            return value, column
    return "", ""


def source_rows_from_csv(path: Path, *, max_rows: int) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(read_csv_rows(path)):
        smiles, column = best_source_molecule(row)
        if not smiles:
            continue
        source_id = next((str(row.get(name) or "").strip() for name in SOURCE_ID_COLUMNS if row.get(name)), "")
        source_id = source_id or f"{path.name}:{index}"
        rows.append({"source_id": source_id, "smiles": smiles, "source_column": column})
        if 0 < int(max_rows) <= len(rows):
            break
    return rows


def source_rows_from_parquet(path: Path, *, max_rows: int) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    schema = set(pq.read_schema(path).names)
    columns = [name for name in [*SOURCE_ID_COLUMNS, *SOURCE_SMILES_COLUMNS] if name in schema]
    if not any(name in schema for name in SOURCE_SMILES_COLUMNS):
        raise ValueError(f"{path} has none of molecule columns {SOURCE_SMILES_COLUMNS}")
    rows = []
    table = pq.read_table(path, columns=columns)
    for index, row in enumerate(table.to_pylist()):
        smiles, column = best_source_molecule(row)
        if not smiles:
            continue
        source_id = next((str(row.get(name) or "").strip() for name in SOURCE_ID_COLUMNS if row.get(name)), "")
        source_id = source_id or f"{path.name}:{index}"
        rows.append({"source_id": source_id, "smiles": smiles, "source_column": column})
        if 0 < int(max_rows) <= len(rows):
            break
    return rows


def source_rows_from_arrow(path: Path, *, max_rows: int) -> list[dict[str, str]]:
    import pyarrow as pa

    def append_batch_rows(batch: Any, *, offset: int, rows: list[dict[str, str]]) -> bool:
        for index, row in enumerate(batch.to_pylist()):
            smiles, column = best_source_molecule(row)
            if not smiles:
                continue
            source_id = next((str(row.get(name) or "").strip() for name in SOURCE_ID_COLUMNS if row.get(name)), "")
            source_id = source_id or f"{path.name}:{offset + index}"
            rows.append({"source_id": source_id, "smiles": smiles, "source_column": column})
            if 0 < int(max_rows) <= len(rows):
                return True
        return False

    with path.open("rb") as handle:
        try:
            reader = pa.ipc.open_file(handle)
            schema = set(reader.schema.names)
            columns = [name for name in [*SOURCE_ID_COLUMNS, *SOURCE_SMILES_COLUMNS] if name in schema]
            if not any(name in schema for name in SOURCE_SMILES_COLUMNS):
                raise ValueError(f"{path} has none of molecule columns {SOURCE_SMILES_COLUMNS}")
            column_indices = [reader.schema.get_field_index(name) for name in columns]
            rows: list[dict[str, str]] = []
            offset = 0
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index).select(column_indices)
                if append_batch_rows(batch, offset=offset, rows=rows):
                    break
                offset += batch.num_rows
            return rows
        except Exception:
            handle.seek(0)
            reader = pa.ipc.open_stream(handle)
            schema = set(reader.schema.names)
            columns = [name for name in [*SOURCE_ID_COLUMNS, *SOURCE_SMILES_COLUMNS] if name in schema]
            if not any(name in schema for name in SOURCE_SMILES_COLUMNS):
                raise ValueError(f"{path} has none of molecule columns {SOURCE_SMILES_COLUMNS}")
            column_indices = [reader.schema.get_field_index(name) for name in columns]
            rows = []
            offset = 0
            for batch in reader:
                selected = batch.select(column_indices)
                if append_batch_rows(selected, offset=offset, rows=rows):
                    break
                offset += batch.num_rows
            return rows


def source_rows_from_path(path: Path, *, max_rows: int) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        return source_rows_from_parquet(path, max_rows=max_rows)
    if path.suffix == ".arrow":
        return source_rows_from_arrow(path, max_rows=max_rows)
    return source_rows_from_csv(path, max_rows=max_rows)


def unsupported_fragment_seed_source_reason(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/markushgrapher2/ip5-markush/" in text:
        return "unsupported_markush_only_source_for_fragment_backbone_seed_decomposition"
    return ""


def canonical_smiles(mol: Any) -> str:
    from rdkit import Chem

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
            atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def strip_attachment_dummies(value: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value or ""), sanitize=False)
    if mol is None:
        return ""
    editable = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(int(atom_index))
    stripped = editable.GetMol()
    return canonical_smiles(stripped)


def strip_attachment_dummies_fast(value: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value or ""), sanitize=False)
    if mol is None:
        return ""
    editable = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummy_indices:
        return canonical_smiles(mol)
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(int(atom_index))
    stripped = editable.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        try:
            Chem.SanitizeMol(
                stripped,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            return ""
    return Chem.MolToSmiles(stripped, canonical=True)


def disable_rdkit_parse_noise() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def normalize_attachment_dummies(value: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value or ""), sanitize=False)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            return ""
    return Chem.MolToSmiles(mol, canonical=True)


def largest_fragment_smiles(value: str) -> str:
    from rdkit import Chem

    normalized = normalize_attachment_dummies(str(value or ""))
    mol = Chem.MolFromSmiles(normalized or str(value or ""))
    if mol is None and ("M  END" in str(value or "") or "V3000" in str(value or "") or "V2000" in str(value or "")):
        mol = Chem.MolFromMolBlock(str(value or ""), sanitize=True, removeHs=False)
    if mol is None:
        return ""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return ""
    largest = max(frags, key=lambda item: item.GetNumHeavyAtoms())
    return canonical_smiles(largest)


def fragment_candidates(
    parent_smiles: str,
    *,
    include_brics_recap: bool = True,
    include_parent: bool = False,
) -> list[tuple[str, str]]:
    from rdkit import Chem
    from rdkit.Chem import BRICS, Recap

    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return []
    candidates: list[tuple[str, str]] = []

    if include_parent:
        candidates.append(("parent_direct", parent_smiles))

    if include_brics_recap:
        try:
            brics_values = list(BRICS.BRICSDecompose(mol, returnMols=False))
        except Exception:
            brics_values = []
        for value in brics_values:
            try:
                smiles = largest_fragment_smiles(normalize_attachment_dummies(str(value)))
            except Exception:
                continue
            if smiles:
                candidates.append(("brics", smiles))

        try:
            hierarchy = Recap.RecapDecompose(mol)
            for value in hierarchy.GetLeaves().keys():
                try:
                    smiles = largest_fragment_smiles(normalize_attachment_dummies(str(value)))
                except Exception:
                    continue
                if smiles:
                    candidates.append(("recap", smiles))
        except Exception:
            pass

    rotatable_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        rotatable_bonds.append(int(bond.GetIdx()))
    for bond_idx in rotatable_bonds[:8]:
        try:
            cut = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
            for frag in Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=False):
                smiles = canonical_smiles(frag)
                if smiles:
                    candidates.append(("single_bond_cut", smiles))
        except Exception:
            continue
    return candidates


def valid_fragment_seed(smiles: str, *, min_atoms: int, max_atoms: int) -> bool:
    from rdkit import Chem
    from training.molnextr_markush.tools.pose_factory_rdkit_utils import can_accept_attachment_dummy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    atom_count = mol.GetNumHeavyAtoms()
    if atom_count < int(min_atoms) or atom_count > int(max_atoms):
        return False
    if "." in smiles:
        return False
    if smiles.count("*") > 1:
        return False
    allowed = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "*"}
    if not all(atom.GetSymbol() in allowed for atom in mol.GetAtoms()):
        return False
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            return False
    return any(can_accept_attachment_dummy(mol, int(atom.GetIdx())) for atom in mol.GetAtoms())


def has_anchor_symbol(smiles: str, allowed_symbols: set[str]) -> bool:
    if not allowed_symbols:
        return True
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in allowed_symbols:
            return True
    return False


def stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


class SourceRowTimeout(TimeoutError):
    pass


def _raise_source_row_timeout(_signum: int, _frame: Any) -> None:
    raise SourceRowTimeout("source row processing timed out")


def process_source_row(
    *,
    path: Path,
    source: dict[str, str],
    min_atoms: int,
    max_atoms: int,
    anchor_symbols: set[str],
    seen: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[str]]:
    new_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    method_counts: Counter[str] = Counter()
    parent, parent_policy = source_parent_for_decomposition(source["smiles"])
    if not parent:
        failures.append({"input": str(path), "source_id": source["source_id"], "smiles": source["smiles"], "error": parent_policy})
        return new_rows, failures, method_counts
    dummy_parent_policy = parent_policy == "dummy_parent_direct_and_single_bond_cut_only"
    try:
        candidates = fragment_candidates(
            parent,
            include_brics_recap=not dummy_parent_policy,
            include_parent=dummy_parent_policy,
        )
    except Exception as exc:
        failures.append(
            {
                "input": str(path),
                "source_id": source["source_id"],
                "smiles": source["smiles"],
                "error": f"fragment_candidate_generation_failed:{type(exc).__name__}:{exc}",
            }
        )
        return new_rows, failures, method_counts
    for method, candidate in candidates:
        backbone = strip_attachment_dummies(candidate)
        if not backbone:
            continue
        if not valid_fragment_seed(backbone, min_atoms=int(min_atoms), max_atoms=int(max_atoms)):
            continue
        if anchor_symbols and not has_anchor_symbol(backbone, anchor_symbols):
            continue
        dedupe_key = backbone
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        method_counts.update([method])
        seed_id = f"rdkit_fragment_seed:{stable_id(source['source_id'], method, candidate)}"
        new_rows.append(
            {
                "source_id": seed_id,
                "SMILES": backbone,
                "smiles": backbone,
                "fragment_seed_method": method,
                "parent_smiles": parent,
                "parent_source_id": source["source_id"],
            }
        )
    return new_rows, failures, method_counts


def source_parent_for_decomposition(value: str) -> tuple[str, str]:
    """Return a dummy-free parent graph for RDKit decomposition."""
    from rdkit import Chem

    raw = clean_source_molecule_text(value)
    mol = Chem.MolFromSmiles(raw, sanitize=False)
    if mol is None:
        return "", "invalid_parent_smiles"
    parent = strip_attachment_dummies_fast(raw) if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()) else canonical_smiles(mol)
    if not parent:
        return "", "invalid_parent_smiles"
    parent = largest_fragment_smiles(parent)
    if not parent:
        return "", "parent_attachment_dummies_not_strippable_for_fragment_seed_decomposition"
    policy = "dummy_parent_direct_and_single_bond_cut_only" if "*" in raw else "direct_parent"
    return parent, policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fragment backbone seed SMILES from ordinary molecules using RDKit BRICS/RECAP/bond cuts."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="CSV, parquet, or Arrow file with SMILES/smiles/cxsmiles/cxsmiles_opt/mol.",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Glob for CSV/parquet/Arrow molecule source files. Relative globs are resolved from repo root.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-source-rows-per-input", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--source-row-timeout-seconds", type=int, default=5)
    parser.add_argument(
        "--anchor-symbols",
        default="",
        help="Optional comma-separated atom symbols required to be present in the seed backbone, e.g. N or C,N.",
    )
    args = parser.parse_args()
    input_paths = [Path(path) for path in args.input]
    for pattern in args.input_glob:
        text = str(pattern)
        matches = glob.glob(text if Path(text).is_absolute() else str(ROOT / text), recursive=True)
        input_paths.extend(Path(path) for path in sorted(matches))
    input_paths = list(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("at least one --input or --input-glob source is required")
    disable_rdkit_parse_noise()
    anchor_symbols = {item.strip() for item in str(args.anchor_symbols or "").split(",") if item.strip()}

    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    method_counts: Counter[str] = Counter()
    seen: set[str] = set()
    source_count = 0

    source_column_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()
    source_file_row_counts: Counter[str] = Counter()
    for path in input_paths:
        try:
            source_rows = source_rows_from_path(path, max_rows=int(args.max_source_rows_per_input))
        except Exception as exc:
            failures.append({"input": str(path), "source_id": "", "smiles": "", "error": str(exc)})
            continue
        print(
            json.dumps(
                {
                    "event": "fragment_seed_source_loaded",
                    "input": str(path),
                    "source_rows": len(source_rows),
                    "source_rows_scanned_so_far": source_count + len(source_rows),
                    "seed_rows_so_far": len(rows),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        unsupported_reason = unsupported_fragment_seed_source_reason(path)
        if unsupported_reason:
            for source in source_rows:
                source_count += 1
                source_column_counts.update([str(source.get("source_column") or "missing")])
                source_file_counts.update([str(path)])
                source_file_row_counts.update([str(path)])
                failures.append(
                    {
                        "input": str(path),
                        "source_id": source["source_id"],
                        "smiles": source["smiles"],
                        "error": unsupported_reason,
                    }
                )
            continue
        for source_index, source in enumerate(source_rows, start=1):
            source_count += 1
            source_column_counts.update([str(source.get("source_column") or "missing")])
            source_file_counts.update([str(path)])
            source_file_row_counts.update([str(path)])
            if source_index == 1 or source_index % 500 == 0 or source_index == len(source_rows):
                print(
                    json.dumps(
                        {
                            "event": "fragment_seed_source_progress",
                            "input": str(path),
                            "source_index": int(source_index),
                            "source_rows": int(len(source_rows)),
                            "seed_rows_so_far": int(len(rows)),
                            "failure_count_so_far": int(len(failures)),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            old_handler = signal.signal(signal.SIGALRM, _raise_source_row_timeout)
            signal.alarm(max(0, int(args.source_row_timeout_seconds)))
            try:
                new_rows, new_failures, new_method_counts = process_source_row(
                    path=path,
                    source=source,
                    min_atoms=int(args.min_atoms),
                    max_atoms=int(args.max_atoms),
                    anchor_symbols=anchor_symbols,
                    seen=seen,
                )
            except SourceRowTimeout:
                failures.append(
                    {
                        "input": str(path),
                        "source_id": source["source_id"],
                        "smiles": source["smiles"],
                        "error": f"source_row_processing_timeout:{int(args.source_row_timeout_seconds)}s",
                    }
                )
                continue
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            failures.extend(new_failures)
            method_counts.update(new_method_counts)
            rows.extend(new_rows)
            if 0 < int(args.max_seeds) <= len(rows):
                break
            if 0 < int(args.max_seeds) <= len(rows):
                break
        if 0 < int(args.max_seeds) <= len(rows):
            break

    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    report = {
        "schema_version": "fragment_backbone_seed_report_v1",
        "inputs": [str(item) for item in input_paths],
        "input_globs": [str(item) for item in args.input_glob],
        "output_csv": str(output_csv),
        "source_rows_scanned": int(source_count),
        "seed_rows": int(len(rows)),
        "anchor_symbols": sorted(anchor_symbols),
        "method_counts": dict(sorted(method_counts.items())),
        "source_column_counts": dict(sorted(source_column_counts.items())),
        "source_file_counts": dict(sorted(source_file_counts.items())),
        "source_file_row_counts": dict(sorted(source_file_row_counts.items())),
        "failure_count": int(len(failures)),
        "failures": failures[:50],
        "policy": {
            "seeds_are_not_trainable_rows": True,
            "source_parent_attachment_dummies_stripped_before_decomposition": True,
            "dummy_parent_uses_direct_parent_and_single_bond_cut_only": True,
            "source_row_processing_timeout_seconds": int(args.source_row_timeout_seconds),
            "fragment_renderer_must_preserve_molnextr_input_contract": True,
            "realistic_document_photo_style_rendering_required_downstream": True,
            "source_leak_and_visual_gates_required_before_training": True,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

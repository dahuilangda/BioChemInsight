"""Audit fragment SMILES/coordinate/edge atom-index alignment in a MoE parquet."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.molnextr_markush.src.moe_dataset import (
    FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
    FRAGMENT_DECODER_LINEARIZATION_METHOD,
    MOE_DATA_CONTRACT_VERSION,
    _bond_type_index,
    _dummy_atom_indices,
    fragment_attachment_contract,
)
from utils.MolNexTR.tokenization import atomwise_tokenizer


def atom_count(smiles: str) -> int:
    return sum(
        1
        for token in atomwise_tokenizer(str(smiles or ""))
        if token.isalpha() or token.startswith("[") or token == "*"
    )


def audit(df_path: Path, previous_df_path: Path | None = None) -> dict:
    frame = pd.read_parquet(df_path)
    fragments = frame.loc[frame["structure_type_label"].astype(int).eq(2)].copy()
    issues = []
    chemical_reordered = 0
    decoder_reordered = 0
    mapping_digest = hashlib.sha256()
    source_dummy_positions: Counter[int] = Counter()
    chemical_dummy_positions: Counter[int] = Counter()
    decoder_dummy_positions: Counter[int] = Counter()
    invalid_rows = 0
    decoder_targets_changed = 0
    canonical_backbone_prefix_rows = 0
    chemistry_cache = {}

    required = {
        "atom_index_alignment_verified",
        "atom_index_alignment_method",
        "smiles_to_source_atom_indices",
        "source_dummy_atom_index",
        "smiles_dummy_atom_index",
        "atom_index_reordered",
        "decoder_smiles",
        "fragment_linearization_verified",
        "fragment_linearization_method",
        "fragment_backbone_smiles",
        "fragment_decoder_backbone_prefix",
        "decoder_to_smiles_atom_indices",
        "decoder_to_source_atom_indices",
        "chemical_smiles_dummy_atom_index",
        "decoder_dummy_atom_index",
        "fragment_dummy_is_final_token",
        "fragment_backbone_prefix_exact",
        "decoder_atom_index_reordered",
        "data_contract_version",
    }
    missing = sorted(required - set(fragments.columns))
    if missing:
        return {
            "schema_version": "molnextr_fragment_graph_linearization_audit_v2",
            "passed": False,
            "dataframe": str(df_path),
            "fragment_rows": int(len(fragments)),
            "issues": [f"missing_columns:{','.join(missing)}"],
        }

    for row_index, row in fragments.iterrows():
        row_issues = []
        smiles = str(row["SMILES"])
        decoder_smiles = str(row["decoder_smiles"])
        chemical_count = atom_count(smiles)
        decoder_count = atom_count(decoder_smiles)
        try:
            chemical_to_source = [int(value) for value in ast.literal_eval(
                str(row["smiles_to_source_atom_indices"])
            )]
            decoder_to_chemical = [int(value) for value in ast.literal_eval(
                str(row["decoder_to_smiles_atom_indices"])
            )]
            decoder_to_source = [int(value) for value in ast.literal_eval(
                str(row["decoder_to_source_atom_indices"])
            )]
            coords = ast.literal_eval(str(row["node_coords"]))
            edges = ast.literal_eval(str(row["edges"]))
        except (SyntaxError, TypeError, ValueError) as exc:
            row_issues.append(f"unparseable_alignment_payload:{exc}")
            chemical_to_source, decoder_to_chemical = [], []
            decoder_to_source, coords, edges = [], [], []
        chemical_dummy_indices = _dummy_atom_indices(smiles)
        decoder_dummy_indices = _dummy_atom_indices(decoder_smiles)
        source_dummy = int(row["source_dummy_atom_index"])
        chemical_dummy = int(row["chemical_smiles_dummy_atom_index"])
        decoder_dummy = int(row["decoder_dummy_atom_index"])
        if bool(row["atom_index_alignment_verified"]) is not True:
            row_issues.append("alignment_not_verified")
        if str(row["atom_index_alignment_method"]) != FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD:
            row_issues.append("alignment_method_mismatch")
        if str(row["data_contract_version"]) != MOE_DATA_CONTRACT_VERSION:
            row_issues.append("data_contract_version_mismatch")
        if bool(row["fragment_linearization_verified"]) is not True:
            row_issues.append("linearization_not_verified")
        if str(row["fragment_linearization_method"]) != FRAGMENT_DECODER_LINEARIZATION_METHOD:
            row_issues.append("linearization_method_mismatch")
        expected = list(range(chemical_count))
        for name, mapping in (
            ("chemical_to_source", chemical_to_source),
            ("decoder_to_chemical", decoder_to_chemical),
            ("decoder_to_source", decoder_to_source),
        ):
            if len(mapping) != chemical_count or sorted(mapping) != expected:
                row_issues.append(f"{name}_mapping_is_not_atom_permutation")
        if decoder_count != chemical_count:
            row_issues.append("decoder_atom_count_mismatch")
        if len(coords) != decoder_count:
            row_issues.append("coordinate_count_mismatch")
        if chemical_dummy_indices != [chemical_dummy]:
            row_issues.append("chemical_smiles_dummy_index_mismatch")
        if int(row["smiles_dummy_atom_index"]) != chemical_dummy:
            row_issues.append("legacy_chemical_dummy_index_mismatch")
        if decoder_dummy_indices != [decoder_dummy]:
            row_issues.append("decoder_dummy_index_mismatch")
        if decoder_dummy != decoder_count - 1:
            row_issues.append("decoder_dummy_is_not_last_atom")
        decoder_tokens = atomwise_tokenizer(decoder_smiles)
        if not decoder_tokens or decoder_tokens[-1] != "*":
            row_issues.append("decoder_dummy_is_not_final_lexical_token")
        if bool(row["fragment_dummy_is_final_token"]) is not True:
            row_issues.append("dummy_final_token_flag_is_false")
        if not (0 <= chemical_dummy < len(chemical_to_source)) or (
            chemical_to_source and chemical_to_source[chemical_dummy] != source_dummy
        ):
            row_issues.append("source_dummy_mapping_mismatch")
        if not (0 <= decoder_dummy < len(decoder_to_chemical)) or (
            decoder_to_chemical and decoder_to_chemical[decoder_dummy] != chemical_dummy
        ):
            row_issues.append("decoder_chemical_dummy_mapping_mismatch")
        if not (0 <= decoder_dummy < len(decoder_to_source)) or (
            decoder_to_source and decoder_to_source[decoder_dummy] != source_dummy
        ):
            row_issues.append("decoder_source_dummy_mapping_mismatch")
        if chemical_to_source and decoder_to_chemical and decoder_to_source:
            composed = [chemical_to_source[index] for index in decoder_to_chemical]
            if composed != decoder_to_source:
                row_issues.append("decoder_source_mapping_composition_mismatch")
        prefix = str(row["fragment_decoder_backbone_prefix"])
        if decoder_smiles != f"{prefix}*":
            row_issues.append("decoder_target_is_not_prefix_plus_dummy")
        if bool(row["fragment_backbone_prefix_exact"]) is not True:
            row_issues.append("backbone_prefix_flag_is_false")
        else:
            canonical_backbone_prefix_rows += 1
        attachment_ok, attachment_reason = fragment_attachment_contract(
            decoder_smiles,
            edges,
        )
        if not attachment_ok:
            row_issues.append(f"fragment_contract:{attachment_reason}")
        if bool(row["atom_index_reordered"]) != (chemical_to_source != expected):
            row_issues.append("chemical_reordered_flag_mismatch")
        if bool(row["decoder_atom_index_reordered"]) != (
            decoder_to_chemical != expected
        ):
            row_issues.append("decoder_reordered_flag_mismatch")
        if chemical_to_source != expected:
            chemical_reordered += 1
        if decoder_to_chemical != expected:
            decoder_reordered += 1
        decoder_targets_changed += int(decoder_smiles != smiles)

        chemistry = chemistry_cache.get(smiles)
        if chemistry is None:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                chemistry = None
            else:
                chemical_atom_tokens = [
                    token
                    for token in atomwise_tokenizer(smiles)
                    if token.isalpha() or token.startswith("[") or token == "*"
                ]
                editable = Chem.RWMol(molecule)
                editable.RemoveAtom(chemical_dummy)
                backbone = editable.GetMol()
                Chem.SanitizeMol(backbone)
                canonical_backbone = Chem.MolToSmiles(
                    backbone,
                    canonical=True,
                    isomericSmiles=True,
                )
                chemical_edges = {
                    (
                        min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                        max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
                        _bond_type_index(bond),
                    )
                    for bond in molecule.GetBonds()
                }
                chemistry = (chemical_atom_tokens, canonical_backbone, chemical_edges)
            chemistry_cache[smiles] = chemistry
        if chemistry is None:
            row_issues.append("chemical_smiles_not_parseable")
        else:
            chemical_atom_tokens, canonical_backbone, chemical_edges = chemistry
            if str(row["fragment_backbone_smiles"]) != canonical_backbone:
                row_issues.append("canonical_backbone_mismatch")
            decoder_atom_tokens = [
                token
                for token in decoder_tokens
                if token.isalpha() or token.startswith("[") or token == "*"
            ]
            if decoder_to_chemical and len(decoder_atom_tokens) == chemical_count:
                expected_tokens = [
                    chemical_atom_tokens[index]
                    for index in decoder_to_chemical
                ]
                if decoder_atom_tokens != expected_tokens:
                    row_issues.append("decoder_atom_tokens_do_not_preserve_chemical_tokens")
            try:
                remapped_edges = {
                    (
                        min(decoder_to_chemical[int(begin)], decoder_to_chemical[int(end)]),
                        max(decoder_to_chemical[int(begin)], decoder_to_chemical[int(end)]),
                        int(bond_type),
                    )
                    for begin, end, bond_type in edges
                }
            except (IndexError, TypeError, ValueError):
                remapped_edges = set()
            if remapped_edges != chemical_edges:
                row_issues.append("decoder_edges_do_not_reconstruct_chemical_graph")
        source_dummy_positions[source_dummy] += 1
        chemical_dummy_positions[chemical_dummy] += 1
        decoder_dummy_positions[decoder_dummy] += 1
        mapping_digest.update(str(row["file_path"]).encode("utf-8"))
        mapping_digest.update(b"\0")
        mapping_digest.update(repr(decoder_to_source).encode("ascii"))
        mapping_digest.update(b"\n")
        if row_issues and len(issues) < 20:
            issues.append({
                "row_index": int(row_index),
                "file_path": str(row["file_path"]),
                "issues": row_issues,
            })
        if row_issues:
            invalid_rows += 1

    comparison = None
    if previous_df_path is not None and previous_df_path.exists():
        previous = pd.read_parquet(
            previous_df_path,
            columns=["file_path", "SMILES", "node_coords", "edges", "structure_type_label"],
        )
        previous = previous.loc[previous["structure_type_label"].astype(int).eq(2)]
        merged = previous.merge(
            fragments[["file_path", "SMILES", "decoder_smiles", "node_coords", "edges"]],
            on="file_path",
            how="outer",
            suffixes=("_previous", "_aligned"),
            indicator=True,
        )
        shared = merged.loc[merged["_merge"].eq("both")]
        comparison = {
            "previous_dataframe": str(previous_df_path),
            "previous_fragment_rows": int(len(previous)),
            "aligned_fragment_rows": int(len(fragments)),
            "recovered_fragment_rows": int((merged["_merge"] == "right_only").sum()),
            "missing_previous_fragment_rows": int((merged["_merge"] == "left_only").sum()),
            "shared_fragment_rows": int(len(shared)),
            "shared_smiles_changed": int(
                shared["SMILES_previous"].ne(shared["SMILES_aligned"]).sum()
            ),
            "shared_decoder_target_changed_from_previous_smiles": int(
                shared["SMILES_previous"].ne(shared["decoder_smiles"]).sum()
            ),
            "shared_coordinate_order_changed": int(
                shared["node_coords_previous"].ne(shared["node_coords_aligned"]).sum()
            ),
            "shared_edge_indices_changed": int(
                shared["edges_previous"].ne(shared["edges_aligned"]).sum()
            ),
        }

    return {
        "schema_version": "molnextr_fragment_graph_linearization_audit_v2",
        "passed": invalid_rows == 0 and len(fragments) > 0,
        "dataframe": str(df_path),
        "total_rows": int(len(frame)),
        "label_counts": {
            str(key): int(value)
            for key, value in frame["structure_type_label"].value_counts().items()
        },
        "fragment_rows": int(len(fragments)),
        "verified_rows": int(
            fragments["atom_index_alignment_verified"].fillna(False).astype(bool).sum()
        ),
        "chemical_alignment_reordered_rows": int(chemical_reordered),
        "decoder_reordered_rows": int(decoder_reordered),
        "decoder_targets_changed_rows": int(decoder_targets_changed),
        "canonical_backbone_prefix_rows": int(canonical_backbone_prefix_rows),
        "invalid_rows": int(invalid_rows),
        "alignment_method": FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
        "linearization_method": FRAGMENT_DECODER_LINEARIZATION_METHOD,
        "mapping_sha256": mapping_digest.hexdigest(),
        "source_dummy_position_counts": {
            str(key): int(value) for key, value in sorted(source_dummy_positions.items())
        },
        "chemical_smiles_dummy_position_counts": {
            str(key): int(value) for key, value in sorted(chemical_dummy_positions.items())
        },
        "decoder_dummy_position_counts": {
            str(key): int(value) for key, value in sorted(decoder_dummy_positions.items())
        },
        "comparison": comparison,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=Path, required=True)
    parser.add_argument("--previous-df", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.df, args.previous_df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("passed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

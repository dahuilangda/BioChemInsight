"""Compile an atom-aligned MoE parquet into backbone-first decoder targets."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.molnextr_markush.src.moe_dataset import (
    FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
    MOE_DATA_CONTRACT_VERSION,
    fragment_attachment_contract,
    linearize_fragment_decoder_target,
)


def _parse_int_list(value) -> list[int]:
    return [int(item) for item in ast.literal_eval(str(value))]


def build(input_path: Path, output_path: Path, progress_every: int) -> dict:
    frame = pd.read_parquet(input_path)
    fragment_mask = frame["structure_type_label"].astype(int).eq(2)
    fragments = frame.loc[fragment_mask]
    if not len(fragments):
        raise ValueError("input dataframe has no fragment rows")
    if not fragments["atom_index_alignment_verified"].fillna(False).astype(bool).all():
        raise ValueError("input dataframe contains unverified fragment atom alignment")
    methods = set(
        fragments["atom_index_alignment_method"].fillna("").astype(str).unique()
    )
    if methods != {FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD}:
        raise ValueError(f"unexpected fragment alignment methods: {sorted(methods)}")

    node_coords = frame["node_coords"].astype(str).tolist()
    edge_values = frame["edges"].astype(str).tolist()
    decoder_smiles = frame["SMILES"].astype(str).tolist()
    fragment_verified = [False] * len(frame)
    fragment_method = ["not_required_for_non_fragment"] * len(frame)
    backbone_smiles = [""] * len(frame)
    decoder_backbone_prefix = [""] * len(frame)
    decoder_to_chemical_values = [repr([])] * len(frame)
    decoder_to_source_values = [repr([])] * len(frame)
    chemical_dummy_values = [-1] * len(frame)
    decoder_dummy_values = [-1] * len(frame)
    dummy_final_values = [False] * len(frame)
    prefix_exact_values = [False] * len(frame)
    decoder_reordered_values = [False] * len(frame)

    cache: dict[str, dict] = {}
    method_counts: Counter[str] = Counter()
    mapping_digest = hashlib.sha256()
    started = time.time()
    for processed, (position, row) in enumerate(fragments.iterrows(), start=1):
        smiles = str(row["SMILES"])
        coords = ast.literal_eval(str(row["node_coords"]))
        edges = ast.literal_eval(str(row["edges"]))
        chemical_to_source = _parse_int_list(row["smiles_to_source_atom_indices"])
        cached = cache.get(smiles)
        if cached is None:
            target, converted_coords, converted_edges, report = (
                linearize_fragment_decoder_target(
                    smiles,
                    coords,
                    edges,
                    chemical_to_source,
                )
            )
            cached = {
                "target": target,
                "report": {
                    key: value
                    for key, value in report.items()
                    if key != "decoder_to_source_atom_indices"
                },
            }
            cache[smiles] = cached
        else:
            report = dict(cached["report"])
            decoder_to_chemical = [
                int(value) for value in report["decoder_to_smiles_atom_indices"]
            ]
            chemical_to_decoder = {
                chemical: decoder
                for decoder, chemical in enumerate(decoder_to_chemical)
            }
            converted_coords = [coords[index] for index in decoder_to_chemical]
            converted_edges = sorted(
                [
                    [
                        chemical_to_decoder[int(begin)],
                        chemical_to_decoder[int(end)],
                        int(bond_type),
                    ]
                    for begin, end, bond_type in edges
                ],
                key=lambda edge: (
                    min(edge[0], edge[1]),
                    max(edge[0], edge[1]),
                    edge[2],
                ),
            )
            report["decoder_to_source_atom_indices"] = [
                chemical_to_source[index] for index in decoder_to_chemical
            ]
            target = str(cached["target"])
            attachment_ok, attachment_reason = fragment_attachment_contract(
                target,
                converted_edges,
            )
            if not attachment_ok:
                raise RuntimeError(
                    f"cached fragment linearization failed for {smiles!r}: "
                    f"{attachment_reason}"
                )

        node_coords[position] = repr(converted_coords)
        edge_values[position] = repr(converted_edges)
        decoder_smiles[position] = target
        fragment_verified[position] = bool(report["verified"])
        fragment_method[position] = str(report["method"])
        backbone_smiles[position] = str(report["backbone_smiles"])
        decoder_backbone_prefix[position] = str(report["decoder_backbone_prefix"])
        decoder_to_chemical_values[position] = repr(
            report["decoder_to_smiles_atom_indices"]
        )
        decoder_to_source_values[position] = repr(
            report["decoder_to_source_atom_indices"]
        )
        chemical_dummy_values[position] = int(
            report["chemical_smiles_dummy_atom_index"]
        )
        decoder_dummy_values[position] = int(report["decoder_dummy_atom_index"])
        dummy_final_values[position] = bool(report["dummy_is_final_token"])
        prefix_exact_values[position] = bool(report["backbone_prefix_exact"])
        decoder_reordered_values[position] = bool(
            report["reordered_from_chemical_smiles"]
        )
        method_counts[str(report["method"])] += 1
        mapping_digest.update(str(row["file_path"]).encode("utf-8"))
        mapping_digest.update(b"\0")
        mapping_digest.update(decoder_to_source_values[position].encode("ascii"))
        mapping_digest.update(b"\n")
        if progress_every > 0 and processed % progress_every == 0:
            print(
                f"compiled={processed}/{len(fragments)} "
                f"unique_targets={len(cache)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    frame["node_coords"] = node_coords
    frame["edges"] = edge_values
    frame["decoder_smiles"] = decoder_smiles
    frame["fragment_linearization_verified"] = fragment_verified
    frame["fragment_linearization_method"] = fragment_method
    frame["fragment_backbone_smiles"] = backbone_smiles
    frame["fragment_decoder_backbone_prefix"] = decoder_backbone_prefix
    frame["decoder_to_smiles_atom_indices"] = decoder_to_chemical_values
    frame["decoder_to_source_atom_indices"] = decoder_to_source_values
    frame["chemical_smiles_dummy_atom_index"] = chemical_dummy_values
    frame["decoder_dummy_atom_index"] = decoder_dummy_values
    frame["fragment_dummy_is_final_token"] = dummy_final_values
    frame["fragment_backbone_prefix_exact"] = prefix_exact_values
    frame["decoder_atom_index_reordered"] = decoder_reordered_values
    frame["data_contract_version"] = MOE_DATA_CONTRACT_VERSION

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "schema_version": "molnextr_fragment_decoder_linearization_build_v1",
        "passed": True,
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": int(len(frame)),
        "fragment_rows": int(len(fragments)),
        "unique_fragment_smiles": int(len(cache)),
        "linearization_methods": dict(method_counts),
        "decoder_mapping_sha256": mapping_digest.hexdigest(),
        "data_contract_version": MOE_DATA_CONTRACT_VERSION,
        "elapsed_seconds": float(time.time() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    report = build(args.input, args.output, args.progress_every)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

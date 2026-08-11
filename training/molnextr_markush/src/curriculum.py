from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from utils.markush_labels import is_markush_label
from training.molnextr_markush.src.labels import atom_tokens


def is_markush_token(token: str) -> bool:
    text = str(token or "").strip()
    if text == "*" or text.endswith("*"):
        return True
    return is_markush_label(text)


def profile_row(row: pd.Series) -> dict:
    tokens = atom_tokens(str(row.get("SMILES") or ""))
    markush_indices = [idx for idx, token in enumerate(tokens) if is_markush_token(token)]
    edges = json.loads(str(row.get("edges") or "[]"))
    markush_degrees = Counter()
    markush_bond_types = Counter()
    for begin, end, bond_type in edges:
        begin = int(begin)
        end = int(end)
        if begin in markush_indices:
            markush_degrees[begin] += 1
            markush_bond_types[int(bond_type)] += 1
        if end in markush_indices:
            markush_degrees[end] += 1
            markush_bond_types[int(bond_type)] += 1
    return {
        "atom_count": len(tokens),
        "markush_atom_count": len(markush_indices),
        "single_neighbor_markush_count": sum(1 for value in markush_degrees.values() if value == 1),
        "multi_neighbor_markush_count": sum(1 for value in markush_degrees.values() if value > 1),
        "markush_bond_types": dict(markush_bond_types),
    }


def assign_bucket(profile: dict) -> str:
    markush_count = int(profile["markush_atom_count"])
    atom_count = int(profile["atom_count"])
    single_neighbor = int(profile["single_neighbor_markush_count"])
    multi_neighbor = int(profile["multi_neighbor_markush_count"])
    if markush_count == 0:
        return "ordinary_structure"
    if atom_count <= 12 and single_neighbor >= 1:
        return "markush_fragment"
    if markush_count >= 2 and atom_count >= 12:
        return "multi_label_scaffold"
    if multi_neighbor:
        return "variable_atom_or_query_like"
    return "single_variable_scaffold_like"

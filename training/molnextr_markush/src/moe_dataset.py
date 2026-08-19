"""Data adapter: pose_factory aggregate CSVs -> decoder targets for MoE training.

The aggregates carry pre-rendered molecule images plus, inside ``render_quality``:
  * ``atom_coordinates``: per-atom {atom_index, token, x, y} with x,y in [0,1];
  * ``bonds``: per-bond {begin_atom_index, end_atom_index, bond_order, is_aromatic}.

This module turns each row into a ``TrainDataset``-compatible DataFrame row:
  file_path, SMILES (cxsmiles extension stripped), node_coords ("[[x,y],...]",
  normalized [0,1]), node_coords_space="normalized_image", edges ("[[u,v,t],...]"),
  source_arrow, and a structure_type_label (0=complete, 1=markush, 2=fragment).

Rows whose atom count (from the SMILES) does not match ``len(atom_coordinates)``
are dropped, because the chartok_coords target needs aligned per-atom coords.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import networkx as nx
from rdkit import Chem

from utils.MolNexTR.tokenization import atomwise_tokenizer

csv.field_size_limit(sys.maxsize)

BUCKET_TO_LABEL = {
    "ordinary_structure": 0,
    "markush_layout": 1,
    "attachment_fragment": 2,
}
BOND_ORDER_TO_INT = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
DUMMY_ATOM_TOKEN_RE = re.compile(r"^\[[^\]]*\*[^\]]*\]$")
MOE_DATA_CONTRACT_VERSION = "molnextr_moe_graph_contract_v3"
FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD = "vf2_element_bond_isomorphism_v1"
FRAGMENT_DECODER_LINEARIZATION_METHOD = (
    "canonical_backbone_then_terminal_dummy_graph_v1"
)


def _strip_cxsmiles(smiles: str) -> str:
    """Drop the CXSMILES/extended annotation after a space-pipe, keep base SMILES."""
    if not isinstance(smiles, str):
        return ""
    # "... |$...$|" or "...|...|" extensions follow a " |" separator
    for sep in (" |", "|"):
        if sep in smiles:
            smiles = smiles.split(sep)[0].strip()
            break
    return smiles.strip()


_MARKUSH_ATTACHMENT_TOKEN_CHARS = "BCNOPSFI"


def _isotope_from_variable_label(label: str | None) -> int | None:
    """Map a CXSMILES dummyLabel (R4, L2, R4a) to its numeric isotope.

    None for non-numeric labels (E, X, Y).
    """
    if not label:
        return None
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else None


def parse_cxsmiles_dummy_labels(cxsmiles: str) -> dict[int, str]:
    """Extract ``{atom_index: variable_label}`` from the CXSMILES extension.

    USPTO exposes the ``$...$`` block via ``atomLabel``; CDK sources use
    ``dummyLabel``. Both are read so targets carry the real variable number.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    text = str(cxsmiles or "")
    if "|" not in text:
        return {}
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    params.strictCXSMILES = False
    params.removeHs = False
    molecule = Chem.MolFromSmiles(text, params)
    if molecule is None:
        return {}
    labels: dict[int, str] = {}
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomicNum()) != 0:
            continue
        for prop in ("atomLabel", "dummyLabel"):
            if atom.HasProp(prop):
                raw = str(atom.GetProp(prop)).strip()
                if raw and raw != "*":
                    labels[int(atom.GetIdx())] = raw
                break
    return labels


def implant_markush_attachment_isotopes(smiles: str, dummy_labels: dict[int, str] | None = None) -> str:
    """Rewrite every bare attachment dummy to ``[<n>*]`` (isotope = variable number).

    String-level only: RDKit re-serialization would mangle Kekule/aromatic forms.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    base = str(smiles or "")
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = False
    params.removeHs = False
    molecule = Chem.MolFromSmiles(base, params)
    if molecule is None:
        return base
    isotope: dict[int, int] = {}
    used: set[int] = set()
    # Prefer the CXSMILES variable number (R4 -> 4); sequential isotopes are a
    # fallback for non-numeric labels or collisions.
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomicNum()) != 0 or int(atom.GetDegree()) < 1:
            continue
        number = _isotope_from_variable_label((dummy_labels or {}).get(int(atom.GetIdx())))
        if number and number not in used:
            isotope[int(atom.GetIdx())] = number
            used.add(number)
    sequence = 0
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomicNum()) != 0 or int(atom.GetDegree()) < 1:
            continue
        idx = int(atom.GetIdx())
        if idx in isotope:
            continue
        existing = int(atom.GetIsotope())
        if existing > 0:
            # Keep upstream-assigned isotopes (real variable numbers); only bare
            # ``*`` falls back to a sequential placeholder.
            isotope[idx] = existing
            used.add(existing)
            continue
        sequence += 1
        while sequence in used:
            sequence += 1
        isotope[idx] = sequence
        used.add(sequence)
    if not isotope:
        return base
    tokens: list[tuple[str, int, int, int]] = []
    atom_index = 0
    pos = 0
    while pos < len(base):
        char = base[pos]
        if char == "[":
            end = base.find("]", pos + 1)
            if end < 0:
                pos += 1
                continue
            tokens.append(("bracket", pos, end + 1, atom_index))
            atom_index += 1
            pos = end + 1
            continue
        if char == "*":
            tokens.append(("star", pos, pos + 1, atom_index))
            atom_index += 1
            pos += 1
            continue
        if base.startswith(("Cl", "Br"), pos):
            tokens.append(("atom", pos, pos + 2, atom_index))
            atom_index += 1
            pos += 2
            continue
        if char in _MARKUSH_ATTACHMENT_TOKEN_CHARS or char in "cnopsb":
            tokens.append(("atom", pos, pos + 1, atom_index))
            atom_index += 1
        pos += 1
    if atom_index != int(molecule.GetNumAtoms()):
        return base
    pieces: list[str] = []
    last = 0
    for kind, start, end, idx in tokens:
        # Overwrite both bare ``*`` and sequential ``[n*]`` dummies with the real
        # variable number; non-dummy brackets are never in the isotope map.
        if idx in isotope:
            pieces.append(base[last:start])
            pieces.append(f"[{isotope[idx]}*]")
            last = end
    pieces.append(base[last:])
    return "".join(pieces)



def _dummy_atom_indices(smiles: str) -> list[int]:
    """Return atom-stream indices represented by ``*``/isotopic dummy tokens."""
    indices = []
    atom_index = -1
    for token in atomwise_tokenizer(str(smiles or "")):
        if not (token.isalpha() or token.startswith("[") or token == "*"):
            continue
        atom_index += 1
        if token == "*" or DUMMY_ATOM_TOKEN_RE.fullmatch(token):
            indices.append(atom_index)
    return indices


def fragment_attachment_contract(smiles: str, edge_triples: list[list[int]]) -> tuple[bool, str]:
    """Validate the production fragment target: one bonded terminal dummy (expert-2 contract)."""
    dummy_indices = _dummy_atom_indices(smiles)
    if len(dummy_indices) != 1:
        return False, f"dummy_atom_count:{len(dummy_indices)}"
    atom_count = sum(
        1
        for token in atomwise_tokenizer(str(smiles or ""))
        if token.isalpha() or token.startswith("[") or token == "*"
    )
    if atom_count < 2:
        return False, f"atom_count:{atom_count}"
    dummy_index = dummy_indices[0]
    neighbors = set()
    adjacency = [set() for _ in range(atom_count)]
    for edge in edge_triples:
        if len(edge) < 3 or int(edge[2]) <= 0:
            continue
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < atom_count and 0 <= v < atom_count) or u == v:
            return False, f"invalid_edge:{u}:{v}"
        adjacency[u].add(v)
        adjacency[v].add(u)
        if u == dummy_index and v != dummy_index:
            neighbors.add(v)
        elif v == dummy_index and u != dummy_index:
            neighbors.add(u)
    if len(neighbors) != 1:
        return False, f"dummy_bond_degree:{len(neighbors)}"
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    if len(visited) != atom_count:
        return False, f"disconnected_graph:{len(visited)}/{atom_count}"
    return True, "single_terminal_attachment"


def _bond_type_index(bond) -> int:
    order = float(bond.GetBondTypeAsDouble())
    if bond.GetIsAromatic() or abs(order - 1.5) <= 1.0e-6:
        return 4
    rounded = int(round(order))
    if rounded not in {1, 2, 3} or abs(order - rounded) > 1.0e-6:
        raise ValueError(f"unsupported fragment bond order: {order}")
    return rounded


def align_fragment_graph_to_smiles_order(
    smiles: str,
    atom_coordinates: list[dict],
    edge_triples: list[list[int]],
) -> tuple[list[list[float]], list[list[int]], dict]:
    """Align source-render atom indices to the atom order emitted by SMILES.

    Equal atom counts do not establish alignment, so verify a graph isomorphism.
    """

    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        raise ValueError(f"fragment SMILES is not parseable: {smiles!r}")
    source_by_index = {}
    for atom in atom_coordinates:
        index = int(atom.get("atom_index"))
        if index in source_by_index:
            raise ValueError(f"duplicate source atom index: {index}")
        source_by_index[index] = atom
    atom_count = int(molecule.GetNumAtoms())
    if set(source_by_index) != set(range(atom_count)):
        raise ValueError(
            "fragment source atom indices must be contiguous: "
            f"observed={sorted(source_by_index)[:10]} atom_count={atom_count}"
        )

    source_graph = nx.Graph()
    source_dummy_indices = []
    periodic_table = Chem.GetPeriodicTable()
    for index in range(atom_count):
        token = str(source_by_index[index].get("token") or "").strip()
        if token == "*" or DUMMY_ATOM_TOKEN_RE.fullmatch(token):
            atomic_number = 0
            source_dummy_indices.append(index)
        else:
            try:
                atomic_number = int(periodic_table.GetAtomicNumber(token))
            except RuntimeError as exc:
                raise ValueError(f"invalid source atom token {token!r}") from exc
            if atomic_number <= 0:
                raise ValueError(f"invalid source atom token {token!r}")
        source_graph.add_node(index, atomic_number=atomic_number)
    for edge in edge_triples:
        if len(edge) < 3:
            raise ValueError(f"invalid fragment edge triple: {edge!r}")
        begin, end, bond_type = int(edge[0]), int(edge[1]), int(edge[2])
        if begin not in source_by_index or end not in source_by_index or begin == end:
            raise ValueError(f"invalid source fragment edge: {edge!r}")
        if bond_type not in {1, 2, 3, 4}:
            raise ValueError(f"invalid source fragment bond type: {bond_type}")
        source_graph.add_edge(begin, end, bond_type=bond_type)

    smiles_graph = nx.Graph()
    smiles_dummy_indices = []
    for atom in molecule.GetAtoms():
        index = int(atom.GetIdx())
        atomic_number = int(atom.GetAtomicNum())
        smiles_graph.add_node(index, atomic_number=atomic_number)
        if atomic_number == 0:
            smiles_dummy_indices.append(index)
    for bond in molecule.GetBonds():
        smiles_graph.add_edge(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            bond_type=_bond_type_index(bond),
        )
    if len(source_dummy_indices) != 1 or len(smiles_dummy_indices) != 1:
        raise ValueError(
            "fragment alignment requires one source and one SMILES dummy: "
            f"source={source_dummy_indices} smiles={smiles_dummy_indices}"
        )

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        smiles_graph,
        source_graph,
        node_match=lambda left, right: left["atomic_number"] == right["atomic_number"],
        edge_match=lambda left, right: left["bond_type"] == right["bond_type"],
    )
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        raise ValueError(
            "fragment source graph is not isomorphic to canonical SMILES graph: "
            f"smiles={smiles!r}"
        )
    smiles_to_source = [int(mapping[index]) for index in range(atom_count)]
    source_to_smiles = {source: target for target, source in enumerate(smiles_to_source)}
    if source_to_smiles[source_dummy_indices[0]] != smiles_dummy_indices[0]:
        raise ValueError("fragment graph isomorphism did not preserve the unique dummy atom")

    aligned_coords = [
        [
            float(source_by_index[source_index].get("x", 0.0)),
            float(source_by_index[source_index].get("y", 0.0)),
        ]
        for source_index in smiles_to_source
    ]
    aligned_edges = sorted(
        [
            [source_to_smiles[int(begin)], source_to_smiles[int(end)], int(bond_type)]
            for begin, end, bond_type in edge_triples
        ],
        key=lambda edge: (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2]),
    )
    aligned_graph_edges = {
        (min(begin, end), max(begin, end), bond_type)
        for begin, end, bond_type in aligned_edges
    }
    smiles_graph_edges = {
        (min(begin, end), max(begin, end), int(data["bond_type"]))
        for begin, end, data in smiles_graph.edges(data=True)
    }
    if aligned_graph_edges != smiles_graph_edges:
        raise RuntimeError("verified fragment isomorphism produced inconsistent reindexed edges")
    return aligned_coords, aligned_edges, {
        "verified": True,
        "method": FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
        "smiles_to_source_atom_indices": smiles_to_source,
        "source_dummy_atom_index": int(source_dummy_indices[0]),
        "smiles_dummy_atom_index": int(smiles_dummy_indices[0]),
        "reordered": smiles_to_source != list(range(atom_count)),
    }


def linearize_fragment_decoder_target(
    smiles: str,
    smiles_ordered_coords: list[list[float]],
    smiles_ordered_edges: list[list[int]],
    smiles_to_source_atom_indices: list[int],
) -> tuple[str, list[list[float]], list[list[int]], dict]:
    """Compile a fragment into decoder order: canonical backbone prefix, terminal dummy last.

    The edge matrix carries the attachment, so the target graph is unchanged.
    """

    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        raise ValueError(f"fragment SMILES is not parseable: {smiles!r}")
    atom_count = int(molecule.GetNumAtoms())
    dummy_indices = [
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if int(atom.GetAtomicNum()) == 0
    ]
    if len(dummy_indices) != 1:
        raise ValueError(
            "fragment decoder linearization requires one dummy atom: "
            f"observed={dummy_indices}"
        )
    dummy_index = int(dummy_indices[0])
    if int(molecule.GetAtomWithIdx(dummy_index).GetDegree()) != 1:
        raise ValueError("fragment decoder linearization requires a terminal dummy")
    if len(smiles_ordered_coords) != atom_count:
        raise ValueError(
            "fragment decoder coordinate count mismatch: "
            f"coords={len(smiles_ordered_coords)} atoms={atom_count}"
        )
    mapping = [int(value) for value in smiles_to_source_atom_indices]
    if len(mapping) != atom_count or sorted(mapping) != list(range(atom_count)):
        raise ValueError("SMILES-to-source atom mapping is not a full permutation")

    # Atom properties survive RWMol.RemoveAtom(), so record the chemical-SMILES
    # index explicitly before canonicalizing the dummy-free backbone.
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_molnextr_chemical_smiles_index", int(atom.GetIdx()))
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(dummy_index)
    backbone = editable.GetMol()
    Chem.SanitizeMol(backbone)
    backbone_smiles = Chem.MolToSmiles(
        backbone,
        canonical=True,
        isomericSmiles=True,
    )
    try:
        backbone_output_order = [
            int(value)
            for value in json.loads(backbone.GetProp("_smilesAtomOutputOrder"))
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("RDKit did not expose the backbone SMILES atom order") from exc
    if sorted(backbone_output_order) != list(range(atom_count - 1)):
        raise RuntimeError("RDKit backbone SMILES atom order is not a permutation")

    decoder_to_smiles = [
        int(
            backbone.GetAtomWithIdx(backbone_index).GetIntProp(
                "_molnextr_chemical_smiles_index"
            )
        )
        for backbone_index in backbone_output_order
    ] + [dummy_index]
    if sorted(decoder_to_smiles) != list(range(atom_count)):
        raise RuntimeError("decoder-to-SMILES atom order is not a permutation")
    smiles_to_decoder = {
        smiles_index: decoder_index
        for decoder_index, smiles_index in enumerate(decoder_to_smiles)
    }
    decoder_coords = [
        [
            float(smiles_ordered_coords[smiles_index][0]),
            float(smiles_ordered_coords[smiles_index][1]),
        ]
        for smiles_index in decoder_to_smiles
    ]
    decoder_edges = sorted(
        [
            [
                int(smiles_to_decoder[int(begin)]),
                int(smiles_to_decoder[int(end)]),
                int(bond_type),
            ]
            for begin, end, bond_type in smiles_ordered_edges
        ],
        key=lambda edge: (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2]),
    )
    decoder_to_source = [mapping[index] for index in decoder_to_smiles]
    chemical_tokens = atomwise_tokenizer(str(smiles))
    chemical_atom_tokens = [
        token
        for token in chemical_tokens
        if token.isalpha() or token.startswith("[") or token == "*"
    ]
    if len(chemical_atom_tokens) != atom_count:
        raise RuntimeError("chemical SMILES token order does not match RDKit atom order")
    backbone_tokens = atomwise_tokenizer(backbone_smiles)
    decoder_prefix_tokens = []
    backbone_atom_index = 0
    for token in backbone_tokens:
        if token.isalpha() or token.startswith("[") or token == "*":
            chemical_index = decoder_to_smiles[backbone_atom_index]
            decoder_prefix_tokens.append(chemical_atom_tokens[chemical_index])
            backbone_atom_index += 1
        else:
            decoder_prefix_tokens.append(token)
    if backbone_atom_index != atom_count - 1:
        raise RuntimeError("canonical backbone token count is inconsistent")
    decoder_backbone_prefix = "".join(decoder_prefix_tokens)
    dummy_token = chemical_atom_tokens[dummy_index]
    if dummy_token != "*" and not DUMMY_ATOM_TOKEN_RE.fullmatch(dummy_token):
        raise RuntimeError("chemical dummy atom token is not a supported wildcard")
    decoder_smiles = f"{decoder_backbone_prefix}{dummy_token}"
    decoder_tokens = atomwise_tokenizer(decoder_smiles)
    decoder_atom_count = sum(
        1
        for token in decoder_tokens
        if token.isalpha() or token.startswith("[") or token == "*"
    )
    if not decoder_tokens or decoder_tokens[-1] != dummy_token:
        raise RuntimeError("fragment decoder dummy is not the final lexical token")
    if decoder_atom_count != atom_count:
        raise RuntimeError(
            "fragment decoder atom count changed during linearization: "
            f"decoder={decoder_atom_count} chemical={atom_count}"
        )
    decoder_dummy_index = atom_count - 1
    attachment_ok, attachment_reason = fragment_attachment_contract(
        decoder_smiles,
        decoder_edges,
    )
    if not attachment_ok:
        raise RuntimeError(
            "fragment decoder graph violates the attachment contract: "
            f"{attachment_reason}"
        )
    if decoder_to_smiles[decoder_dummy_index] != dummy_index:
        raise RuntimeError("fragment decoder dummy mapping is inconsistent")

    return decoder_smiles, decoder_coords, decoder_edges, {
        "verified": True,
        "method": FRAGMENT_DECODER_LINEARIZATION_METHOD,
        "backbone_smiles": backbone_smiles,
        "decoder_backbone_prefix": decoder_backbone_prefix,
        "decoder_to_smiles_atom_indices": decoder_to_smiles,
        "decoder_to_source_atom_indices": decoder_to_source,
        "chemical_smiles_dummy_atom_index": dummy_index,
        "decoder_dummy_atom_index": decoder_dummy_index,
        "dummy_is_final_token": True,
        "backbone_prefix_exact": decoder_smiles == (
            f"{decoder_backbone_prefix}{dummy_token}"
        ),
        "reordered_from_chemical_smiles": decoder_to_smiles != list(range(atom_count)),
    }


def _resolve_file_path(raw_path: str, csv_dir: Path | None) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path) if path.exists() else ""
    if csv_dir is not None:
        candidate = csv_dir / path
        if candidate.exists():
            return str(candidate.resolve())
    if path.exists():
        return str(path.resolve())
    return ""


def _is_real_tight_terminal_wavy(rec: dict) -> bool:
    return bool(
        int(rec.get("structure_type_label", -1)) == 2
        and rec.get("attachment_render_mode") == "wavy"
        and rec.get("attachment_render_geometry") == "custom_markush_attachment_perpendicular_wavy"
        and rec.get("real_tight_crop_style") is True
        and rec.get("terminal_wavy_externality_passed") is True
    )


def _fragment_stratum(rec: dict) -> str:
    if _is_real_tight_terminal_wavy(rec):
        return "real_tight_wavy"
    smiles_len = len(str(rec.get("SMILES") or ""))
    if smiles_len <= 8:
        return "tiny"
    if smiles_len <= 12:
        return "small"
    return "other"


def _select_fragment_records_stratified(candidates: dict[str, list[dict]], limit: int) -> list[dict]:
    """Select a stratified fragment subset for limit-mode runs (wavy/tiny/small quotas)."""
    limit = int(limit)
    if limit <= 0:
        return []
    quotas = _fragment_stratified_quotas(limit)
    selected: list[dict] = []
    used: set[str] = set()
    for bucket, quota in quotas.items():
        for rec in candidates.get(bucket, [])[:quota]:
            fp = str(rec.get("file_path") or "")
            if fp and fp not in used:
                selected.append(rec)
                used.add(fp)
                if len(selected) >= limit:
                    return selected
    for bucket in ("real_tight_wavy", "tiny", "small", "other"):
        for rec in candidates.get(bucket, []):
            fp = str(rec.get("file_path") or "")
            if fp and fp not in used:
                selected.append(rec)
                used.add(fp)
                if len(selected) >= limit:
                    return selected
    return selected


def _fragment_stratified_quotas(limit: int) -> dict[str, int]:
    limit = int(limit)
    return {
        "real_tight_wavy": max(1, limit // 3),
        "tiny": max(1, limit // 4),
        "small": max(1, limit // 4),
    }


def _fragment_candidates_sufficient(candidates: dict[str, list[dict]], limit: int) -> bool:
    if int(limit) <= 0:
        return False
    quotas = _fragment_stratified_quotas(limit)
    total = sum(len(items) for items in candidates.values())
    return (
        total >= int(limit)
        and all(len(candidates.get(bucket, [])) >= quota for bucket, quota in quotas.items())
    )


def _row_to_record(
    row: dict,
    default_label: int,
    tokenizer=None,
    csv_dir: Path | None = None,
) -> dict | None:
    rq_raw = row.get("render_quality") or ""
    try:
        rq = json.loads(rq_raw) if rq_raw else {}
    except json.JSONDecodeError:
        return None
    # Wavy-fragment gate: only production perpendicular-mark, tight-crop rows
    # are trainable; historical native/along-axis rows are rejected.
    if row.get("structure_type_bucket") == "attachment_fragment" and row.get("attachment_render_mode") == "wavy":
        if rq.get("attachment_render_geometry") != "custom_markush_attachment_perpendicular_wavy":
            return None
        if rq.get("real_tight_crop_style") is not True:
            return None
        externality = rq.get("terminal_wavy_externality") if isinstance(rq.get("terminal_wavy_externality"), dict) else {}
        if externality.get("passed") is not True:
            return None
    atom_coords = rq.get("atom_coordinates")
    bonds = rq.get("bonds")
    if not isinstance(atom_coords, list) or len(atom_coords) == 0:
        return None
    file_path = _resolve_file_path((row.get("file_path") or "").strip(), csv_dir)
    if not file_path:
        return None

    # atom_coordinates ordered by atom_index -> [[x,y],...]
    try:
        ordered = sorted(atom_coords, key=lambda a: int(a.get("atom_index", 0)))
    except (TypeError, ValueError):
        ordered = atom_coords
    coords = [[float(a.get("x", 0.0)), float(a.get("y", 0.0))] for a in ordered]

    # edges as [[u,v,t],...] (t in 1..4; wedges not present in synthetic data)
    edge_triples = []
    if isinstance(bonds, list):
        for b in bonds:
            try:
                u = int(b.get("begin_atom_index"))
                v = int(b.get("end_atom_index"))
            except (TypeError, ValueError):
                continue
            if b.get("is_aromatic"):
                t = 4
            else:
                t = BOND_ORDER_TO_INT.get(str(b.get("bond_order", "")).upper(), 1)
            edge_triples.append([u, v, t])

    raw_smiles = row.get("SMILES") or row.get("smiles") or ""
    smiles = _strip_cxsmiles(raw_smiles)
    if not smiles:
        return None
    bucket = (row.get("structure_type_bucket") or "").strip()
    label = BUCKET_TO_LABEL.get(bucket, default_label)
    if bucket == "markush_layout":
        # Ground dummy isotopes in the CXSMILES variable labels (R4 -> 4).
        dummy_labels = parse_cxsmiles_dummy_labels(raw_smiles)
        smiles = implant_markush_attachment_isotopes(smiles, dummy_labels)
    atom_index_alignment = {
        "verified": False,
        "method": "not_required_for_non_fragment",
        "smiles_to_source_atom_indices": [],
        "source_dummy_atom_index": -1,
        "smiles_dummy_atom_index": -1,
        "reordered": False,
    }
    decoder_smiles = smiles
    decoder_linearization = {
        "verified": False,
        "method": "not_required_for_non_fragment",
        "backbone_smiles": "",
        "decoder_backbone_prefix": "",
        "decoder_to_smiles_atom_indices": [],
        "decoder_to_source_atom_indices": [],
        "chemical_smiles_dummy_atom_index": -1,
        "decoder_dummy_atom_index": -1,
        "dummy_is_final_token": False,
        "backbone_prefix_exact": False,
        "reordered_from_chemical_smiles": False,
    }
    if int(label) == 2:
        try:
            coords, edge_triples, atom_index_alignment = (
                align_fragment_graph_to_smiles_order(
                    smiles,
                    atom_coords,
                    edge_triples,
                )
            )
            (
                decoder_smiles,
                coords,
                edge_triples,
                decoder_linearization,
            ) = linearize_fragment_decoder_target(
                smiles,
                coords,
                edge_triples,
                atom_index_alignment["smiles_to_source_atom_indices"],
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"fragment graph target compilation failed for {file_path}: {exc}"
            ) from exc

    # Atom-count alignment is mandatory: chartok_coords interleaves one coord
    # pair per atom token, so misaligned rows are dropped.
    if tokenizer is not None:
        try:
            n_atoms = sum(
                1
                for t in atomwise_tokenizer(decoder_smiles)
                if tokenizer.is_atom_token(t)
            )
        except Exception:
            return None
        if n_atoms != len(coords):
            return None

    if int(label) == 2:
        attachment_ok, _attachment_reason = fragment_attachment_contract(
            decoder_smiles,
            edge_triples,
        )
        if not attachment_ok:
            return None
    externality = rq.get("terminal_wavy_externality") if isinstance(rq.get("terminal_wavy_externality"), dict) else {}
    attachment_render_mode = (row.get("attachment_render_mode") or "").strip()
    attachment_render_geometry = str(rq.get("attachment_render_geometry") or "")
    real_tight_crop_style = bool(rq.get("real_tight_crop_style") is True)
    terminal_wavy_externality_passed = bool(externality.get("passed") is True)

    return {
        "file_path": file_path,
        "SMILES": smiles,
        "decoder_smiles": decoder_smiles,
        "node_coords": repr(coords),
        "node_coords_space": "normalized_image",
        "edges": repr(edge_triples),
        "source_arrow": row.get("source_arrow") or "",
        "structure_type_label": int(label),
        "structure_type_bucket": bucket,
        "attachment_render_mode": attachment_render_mode,
        "attachment_render_geometry": attachment_render_geometry,
        "real_tight_crop_style": real_tight_crop_style,
        "terminal_wavy_externality_passed": terminal_wavy_externality_passed,
        "image_domain": str(row.get("image_domain") or "pose_factory"),
        "coordinate_targets_available": True,
        "atom_index_alignment_verified": bool(atom_index_alignment["verified"]),
        "atom_index_alignment_method": str(atom_index_alignment["method"]),
        "smiles_to_source_atom_indices": repr(
            atom_index_alignment["smiles_to_source_atom_indices"]
        ),
        "source_dummy_atom_index": int(
            atom_index_alignment["source_dummy_atom_index"]
        ),
        "smiles_dummy_atom_index": int(
            atom_index_alignment["smiles_dummy_atom_index"]
        ),
        "atom_index_reordered": bool(atom_index_alignment["reordered"]),
        "fragment_linearization_verified": bool(
            decoder_linearization["verified"]
        ),
        "fragment_linearization_method": str(decoder_linearization["method"]),
        "fragment_backbone_smiles": str(
            decoder_linearization["backbone_smiles"]
        ),
        "fragment_decoder_backbone_prefix": str(
            decoder_linearization["decoder_backbone_prefix"]
        ),
        "decoder_to_smiles_atom_indices": repr(
            decoder_linearization["decoder_to_smiles_atom_indices"]
        ),
        "decoder_to_source_atom_indices": repr(
            decoder_linearization["decoder_to_source_atom_indices"]
        ),
        "chemical_smiles_dummy_atom_index": int(
            decoder_linearization["chemical_smiles_dummy_atom_index"]
        ),
        "decoder_dummy_atom_index": int(
            decoder_linearization["decoder_dummy_atom_index"]
        ),
        "fragment_dummy_is_final_token": bool(
            decoder_linearization["dummy_is_final_token"]
        ),
        "fragment_backbone_prefix_exact": bool(
            decoder_linearization["backbone_prefix_exact"]
        ),
        "decoder_atom_index_reordered": bool(
            decoder_linearization["reordered_from_chemical_smiles"]
        ),
        "data_contract_version": MOE_DATA_CONTRACT_VERSION,
    }


def build_moe_df(sources_by_label: dict[int, list[str]], per_label_limit: int,
                 out_path: str | None = None, tokenizer=None) -> pd.DataFrame:
    """Scan the aggregate CSVs and emit a unified training DataFrame.

    sources_by_label maps structure_type_label (0/1/2) -> list of CSV paths.
    ``per_label_limit <= 0`` means scan all available rows.
    ``tokenizer`` is the chartok_coords tokenizer, used to enforce atom-count
    alignment (rows that misalign are dropped).
    """
    records: list[dict] = []
    seen: set[str] = set()
    for label, csv_paths in sorted(sources_by_label.items()):
        count = 0
        scanned = 0
        duplicates = 0
        limit = int(per_label_limit or 0)
        fragment_candidates: dict[str, list[dict]] = {
            "real_tight_wavy": [],
            "tiny": [],
            "small": [],
            "other": [],
        }
        for csv_idx, csv_path in enumerate(csv_paths, start=1):
            if not os.path.exists(csv_path):
                continue
            if csv_idx == 1 or csv_idx == len(csv_paths) or csv_idx % 100 == 0:
                print(
                    f"  label {label}: scanning csv {csv_idx}/{len(csv_paths)} {csv_path}",
                    flush=True,
                )
            csv_dir = Path(csv_path).resolve().parent
            with open(csv_path, newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    scanned += 1
                    rec = _row_to_record(
                        row,
                        default_label=label,
                        tokenizer=tokenizer,
                        csv_dir=csv_dir,
                    )
                    if rec is None:
                        continue
                    fp = rec["file_path"]
                    if fp in seen:
                        duplicates += 1
                        continue
                    if int(label) == 2 and limit > 0:
                        fragment_candidates[_fragment_stratum(rec)].append(rec)
                        count = sum(len(items) for items in fragment_candidates.values())
                    else:
                        seen.add(fp)
                        records.append(rec)
                        count += 1
                    if int(label) != 2 and limit > 0 and count >= limit:
                        break
                    if (
                        int(label) == 2
                        and limit > 0
                        and _fragment_candidates_sufficient(fragment_candidates, limit)
                    ):
                        break
            if int(label) != 2 and limit > 0 and count >= limit:
                break
            if (
                int(label) == 2
                and limit > 0
                and _fragment_candidates_sufficient(fragment_candidates, limit)
            ):
                break
        if int(label) == 2 and limit > 0:
            selected = _select_fragment_records_stratified(fragment_candidates, limit)
            for rec in selected:
                fp = rec["file_path"]
                if fp in seen:
                    duplicates += 1
                    continue
                seen.add(fp)
                records.append(rec)
            count = len(selected)
            print(
                "  label 2 stratified fragment selection: "
                f"real_tight_wavy={len(fragment_candidates['real_tight_wavy'])} "
                f"tiny={len(fragment_candidates['tiny'])} small={len(fragment_candidates['small'])} "
                f"other={len(fragment_candidates['other'])} selected={count}",
                flush=True,
            )
        print(
            f"  label {label}: scanned {scanned} rows, kept {count}, duplicates {duplicates}",
            flush=True,
        )
    df = pd.DataFrame(records)
    df.reset_index(drop=True, inplace=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  wrote {len(df)} rows -> {out_path}", flush=True)
    return df

from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import cxsmiles_dummy_labels
from utils.markush_labels import CHEMICAL_SYMBOLS, is_fixed_substituent_label, is_markush_label, markush_label_category, normalize_label
from training.molnextr_markush.src.pose_factory import (
    POSE_FACTORY_SCHEMA_VERSION,
    formal_sim_dataset_lineage,
    formal_simulation_policy,
)
from training.molnextr_markush.tools.prepare_markush_cdk_accepted_candidate import (
    validate_markush_row as validate_markush_accepted_candidate_row,
)


GENERATOR_VERSION = "pose_factory_markush_shard_cdk_svg_bond_axis_atom_center_endpoint_axis_v8_viewbox_clipped_axis_polyline_epsilon_snapped"
POSE_COORD_POLICY = "cdk_svg_bond_axis_atom_center_hybrid_mapping"
NONLINEAR_WARP_SCHEMA_VERSION = "markush_nonlinear_document_warp_v1"
FORMAL_NONLINEAR_WARP_POLICY = "formal_synchronized_warped_svg_polyline_pose_preservation_v1"
MARKUSH_GENERATOR_ROOT = Path("/tmp/MarkushGenerator")
DEPICTOR_MARKUSH_LABEL_PATCH_MARKER = "BIOCHEMINSIGHT_MARKUSH_DUMMY_LABEL_PATCH_V1"
COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]
REAL_ATOM_TOKENS = {
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "Se",
    "Te",
    "As",
}
BACKEND_REFERENCES = [
    {
        "name": "MarkushGenerator",
        "url": "https://github.com/DS4SD/MarkushGenerator",
        "role": "CXSMILES-aware CDK depiction, Markush/R-label rendering, and OCR-cell conventions.",
    },
    {
        "name": "CDK DepictionGenerator",
        "url": "https://cdk.github.io/",
        "role": "Markush/CXSMILES depiction and V3000 molfile coordinate source.",
    },
]
DOCUMENT_REALISM_SCHEMA_VERSION = "markush_document_realism_v1"
DOCUMENT_REALISM_POLICY = "patent_literature_markushgenerator_cdk_svg_v1"
DOCUMENT_REALISM_REFERENCES = [
    {
        "name": "MolNexTR",
        "url": "https://github.com/CYF2000127/MolNexTR",
        "role": "Coordinate-aware image-to-graph OCSR target with robustness claims on real literature images.",
    },
    {
        "name": "MarkushGenerator",
        "url": "https://github.com/DS4SD/MarkushGenerator",
        "role": "Existing CXSMILES-to-Markush structure image and OCR-cell generation pipeline used here.",
    },
    {
        "name": "MarkushGrapher-2",
        "url": "https://github.com/DS4SD/MarkushGrapher",
        "role": "Markush recognition precedent using patent-derived benchmarks and real/synthetic Markush data.",
    },
    {
        "name": "MolScribe",
        "url": "https://github.com/thomas0809/MolScribe",
        "role": "Image-to-graph OCSR precedent with synthetic plus real USPTO/literature-style benchmarks.",
    },
]
REALISM_RENDER_PARAMETER_KEYS = {
    "seed",
    "stroke_ratio",
    "bond_separation",
    "symbol_margin_ratio",
    "font_name",
    "font_size",
    "render_atom_numbers",
    "render_carbon_symbols",
    "render_aromatic_display",
    "render_deuterium_symbol",
    "render_terminal_carbons",
}

HARNESS_MARKUSH_LABEL_POOLS: dict[str, list[str]] = {
    "r_series": ["R", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
    "xyz_series": ["X", "Y", "Z", "R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    "mixed_patent": ["R1", "R2", "R3", "X", "Y", "Z", "Ar", "HetAr", "R4", "R5"],
    "aryl_hetero": ["Ar", "HetAr", "R1", "R2", "X", "Y", "Z", "R3", "R4", "R5"],
}


def import_rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def clean_cxsmiles(value: str) -> str:
    text = str(value or "").strip()
    return text


def stable_hexdigest(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def stable_int_hash(*parts: object, modulo: int = 10**12) -> int:
    return int(stable_hexdigest(*parts)[:16], 16) % modulo


def annotation_r_labels(annotation: str) -> list[str]:
    return [match.strip() for match in re.findall(r"<r>(.*?)</r>", annotation or "", flags=re.IGNORECASE | re.DOTALL)]


def markush_r_labels(annotation: str, cxsmiles_opt: str = "") -> list[str]:
    labels = annotation_r_labels(annotation)
    if labels:
        return labels
    return annotation_r_labels(cxsmiles_opt)


def markush_r_variable_labels(annotation: str, cxsmiles_opt: str = "") -> list[str]:
    return [normalize_label(label) for label in markush_r_labels(annotation, cxsmiles_opt) if is_markush_label(label)]


def count_bucket(value: int) -> str | None:
    if value <= 0:
        return None
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def cxsmiles_star_count(cxsmiles: str) -> int:
    return len(re.findall(r"\*", cxsmiles or ""))


_ATTACHMENT_DUMMY_ISOTOPE_TOKEN_CHARS = "BCNOPSFI"


def _implant_attachment_dummy_isotopes(cxsmiles: str) -> tuple[str, dict[int, int]]:
    """Assign sequential isotopes to every attachment dummy so the base SMILES
    serializes as ``[n*]`` instead of bare ``*``.

    Markush scaffold attachment points must carry an isotope so the decoder
    target (the base SMILES) is ``[n*]``; without it the model is trained on
    bare ``*`` and later decodes scaffolds as bare ``*``, which the assembler
    cannot map to a variable position. This is a pure string-level transform:
    the base SMILES is tokenized with atom-index tracking, and only bare ``*``
    tokens at attachment-dummy atom indices are rewritten to ``[<n>*]``. Every
    other character (bonds, ring closures, charges, and the whole CXSMILES
    extension block) is preserved byte-for-byte — RDKit re-serialization is
    intentionally avoided because it mangles Kekulé/aromatic forms.

    Attachment dummies = atomic number 0, degree >= 1, no existing isotope.
    Disconnected dummies (degree 0, decode artifacts) are skipped. Returns the
    updated CXSMILES and the {atom_index: isotope} map (empty if no change).
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    text = str(cxsmiles or "")
    base = text.split(" |", 1)[0].strip()
    extension = text[len(base):]
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(base, parser_params)
    isotope_map: dict[int, int] = {}
    if molecule is not None:
        sequence = 0
        for atom in molecule.GetAtoms():
            if int(atom.GetAtomicNum()) != 0:
                continue
            if int(atom.GetIsotope()) > 0:
                continue
            if int(atom.GetDegree()) < 1:
                continue
            sequence += 1
            isotope_map[int(atom.GetIdx())] = sequence
    if not isotope_map:
        return text, {}
    tokens: list[tuple[str, int, int, int]] = []
    index = 0
    pos = 0
    while pos < len(base):
        char = base[pos]
        if char == "[":
            end = base.find("]", pos + 1)
            if end < 0:
                pos += 1
                continue
            tokens.append(("bracket", pos, end + 1, index))
            index += 1
            pos = end + 1
            continue
        if char == "*":
            tokens.append(("star", pos, pos + 1, index))
            index += 1
            pos += 1
            continue
        if base.startswith(("Cl", "Br"), pos):
            tokens.append(("atom", pos, pos + 2, index))
            index += 1
            pos += 2
            continue
        if char in _ATTACHMENT_DUMMY_ISOTOPE_TOKEN_CHARS or char in "cnopsb":
            tokens.append(("atom", pos, pos + 1, index))
            index += 1
        pos += 1
    if molecule is not None and index != int(molecule.GetNumAtoms()):
        return text, {}
    pieces: list[str] = []
    last = 0
    for kind, start, end, atom_index in tokens:
        if kind == "star" and atom_index in isotope_map:
            pieces.append(base[last:start])
            pieces.append(f"[{isotope_map[atom_index]}*]")
            last = end
    pieces.append(base[last:])
    return "".join(pieces) + extension, isotope_map


def choose_harness_markush_label_pool(labels: list[str], seed_parts: list[object]) -> tuple[str, list[str]]:
    count = len(labels)
    pool_names = ["r_series", "r_series", "xyz_series", "mixed_patent"]
    index = stable_int_hash("harness_label_pool", *seed_parts, count, modulo=len(pool_names))
    name = pool_names[int(index)]
    return name, HARNESS_MARKUSH_LABEL_POOLS[name]


def canonicalize_annotation_r_labels(annotation: str, harness_label_policy: dict[str, Any]) -> str:
    mapping = harness_label_policy.get("source_to_render_label")
    if not isinstance(mapping, dict) or not mapping:
        return annotation

    def repl(match: re.Match[str]) -> str:
        old_label = normalize_label(match.group(1))
        return f"<r>{mapping.get(old_label, old_label)}</r>"

    return re.sub(r"<r>(.*?)</r>", repl, annotation or "", flags=re.IGNORECASE | re.DOTALL)


def canonicalize_markush_harness_labels(cxsmiles: str, *, source_id: str, row_id: str) -> tuple[str, dict[str, Any]]:
    dummy_labels = {
        int(index): normalize_label(label)
        for index, label in cxsmiles_dummy_labels(cxsmiles).items()
        if is_markush_label(label)
    }
    if not dummy_labels:
        # No Markush R/X/Y/Z labels, but the scaffold may still have bare-*
        # attachment dummies that must serialize as [n*] so the decoder target
        # is not trained on bare *.
        implanted, implant_map = _implant_attachment_dummy_isotopes(cxsmiles)
        return implanted, {
            "enabled": True,
            "changed": implanted != cxsmiles,
            "reason": "no_markush_dummy_labels",
            "schema_version": "markush_harness_label_canonicalization_v1",
            "attachment_dummy_isotope_implant": {str(k): v for k, v in implant_map.items()},
        }
    ordered_indices = sorted(dummy_labels)
    pool_name, pool = choose_harness_markush_label_pool(
        [dummy_labels[index] for index in ordered_indices],
        [source_id, row_id, cxsmiles],
    )
    mapping: dict[int, str] = {}
    for offset, atom_index in enumerate(ordered_indices):
        if offset < len(pool):
            label = pool[offset]
        else:
            label = f"R{offset + 1}"
        mapping[int(atom_index)] = label

    def repl(match: re.Match[str]) -> str:
        atom_index = int(match.group("index"))
        prop = match.group("prop")
        old_label = normalize_label(match.group("label"))
        if atom_index not in mapping or not is_markush_label(old_label):
            return match.group(0)
        return f"{match.group('prefix')}{atom_index}.{prop}.{mapping[atom_index]}"

    pattern = re.compile(
        r"(?P<prefix>^|[:|,])(?P<index>\d+)\.(?P<prop>dummyLabel|atomLabel|molFileAlias)\.(?P<label>[^:|,]+)"
    )
    updated = pattern.sub(repl, cxsmiles)

    def label_block_repl(match: re.Match[str]) -> str:
        labels = match.group(1).split(";")
        for atom_index, label in mapping.items():
            if atom_index < len(labels) and is_markush_label(normalize_label(labels[atom_index])):
                labels[atom_index] = label
        return "$" + ";".join(labels) + "$"

    updated = re.sub(r"\$([^$]*)\$", label_block_repl, updated, count=1)
    cx_block_match = re.search(r"\|([^|]*)\|", updated)
    existing_props: set[tuple[int, str]] = set()
    if cx_block_match:
        for match in re.finditer(
            r"(?:^|[:|,])(?P<index>\d+)\.(?P<prop>dummyLabel|atomLabel|molFileAlias)\.(?P<label>[^:|,]+)",
            cx_block_match.group(1),
        ):
            existing_props.add((int(match.group("index")), str(match.group("prop"))))
    atom_prop_entries = [
        f"{atom_index}.dummyLabel.{label}"
        for atom_index, label in sorted(mapping.items())
        if (int(atom_index), "dummyLabel") not in existing_props
    ]
    atom_prop_appended = bool(atom_prop_entries)
    if atom_prop_entries:
        addition = "atomProp:" + ":".join(atom_prop_entries)
        if cx_block_match:
            block = cx_block_match.group(1).strip()
            separator = "," if block else ""
            updated = updated[: cx_block_match.start(1)] + block + separator + addition + updated[cx_block_match.end(1) :]
        else:
            updated = updated.rstrip() + " |" + addition + "|"
    updated, attachment_isotope_implant = _implant_attachment_dummy_isotopes(updated)
    isotope_star_count = len(re.findall(r"\[\d+\*\]", updated))
    source_to_render_label: dict[str, str] = {}
    for atom_index in ordered_indices:
        source_to_render_label[dummy_labels[atom_index]] = mapping[atom_index]
    return updated, {
        "enabled": True,
        "changed": updated != cxsmiles,
        "schema_version": "markush_harness_label_canonicalization_v1",
        "policy": "source_preserving_render_label_canonicalization_for_biocheminsight_molnextr_llm_harness",
        "pool": pool_name,
        "source_labels_by_atom_index": {str(index): dummy_labels[index] for index in ordered_indices},
        "render_labels_by_atom_index": {str(index): mapping[index] for index in ordered_indices},
        "source_to_render_label": source_to_render_label,
        "allowed_label_family": ["R", "R1-Rn", "X", "Y", "Z"],
        "raw_source_unchanged": True,
        "render_atomprop_dummyLabel_appended": atom_prop_appended,
        "attachment_dummy_isotope_implant": {str(k): v for k, v in attachment_isotope_implant.items()},
        "render_dummy_isotope_star_preserved_count": int(isotope_star_count),
        "render_label_transport_policy": (
            "render CXSMILES always carries atomProp dummyLabel labels; isotope-star prefixes are preserved "
            "only for RDKit/CDK atom-index alignment and the patched CDK depictor must display the dummyLabel"
        ),
    }


def cxsmiles_source_geometry_metrics(molecule: Any, variable_dummy_labels: dict[int, str], cxsmiles: str) -> dict[str, Any]:
    real_indices = [
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if int(atom.GetAtomicNum()) != 0 and str(atom.GetSymbol()) in REAL_ATOM_TOKENS
    ]
    real_set = set(real_indices)
    real_real_bonds = 0
    dummy_real_bonds = 0
    dummy_dummy_bonds = 0
    real_neighbors: dict[int, set[int]] = defaultdict(set)
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        begin_real = begin in real_set
        end_real = end in real_set
        if begin_real and end_real:
            real_real_bonds += 1
            real_neighbors[begin].add(end)
            real_neighbors[end].add(begin)
        elif begin_real or end_real:
            dummy_real_bonds += 1
        else:
            dummy_dummy_bonds += 1

    ring_info = molecule.GetRingInfo()
    atom_rings = [tuple(int(index) for index in ring) for ring in ring_info.AtomRings()]
    real_atom_count = int(len(real_indices))
    variable_count = int(len(variable_dummy_labels))

    seen: set[int] = set()
    component_sizes: list[int] = []
    for atom_index in real_indices:
        if atom_index in seen:
            continue
        stack = [atom_index]
        seen.add(atom_index)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in real_neighbors.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        component_sizes.append(size)

    label_counts = Counter(normalize_label(label) for label in variable_dummy_labels.values())
    duplicate_counts = sorted(int(count) for count in label_counts.values() if int(count) > 1)
    cxsmiles_text = str(cxsmiles or "")
    stereo_token_count = int(
        cxsmiles_text.count("@")
        + len(re.findall(r"(?<![A-Za-z0-9])[\\/](?![A-Za-z0-9])", cxsmiles_text))
    )
    slash_bond_token_count = int(cxsmiles_text.count("/") + cxsmiles_text.count("\\"))
    branch_token_count = int(cxsmiles_text.count("("))
    ring_digit_token_count = int(len(re.findall(r"(?<![A-Za-z])\d", cxsmiles_text.split("|", 1)[0])))
    atom_count = int(molecule.GetNumAtoms())
    bond_count = int(molecule.GetNumBonds())
    return {
        "schema_version": "markush_source_geometry_metrics_v1",
        "real_atom_count": real_atom_count,
        "real_real_bond_count": int(real_real_bonds),
        "dummy_real_bond_count": int(dummy_real_bonds),
        "dummy_dummy_bond_count": int(dummy_dummy_bonds),
        "real_atom_component_count": int(len(component_sizes)),
        "largest_real_atom_component_size": int(max(component_sizes) if component_sizes else 0),
        "sgroup_count": int(len(re.findall(r"Sg:", str(cxsmiles or "")))),
        "unique_variable_label_count": int(len(label_counts)),
        "max_duplicate_variable_label_count": int(max(label_counts.values()) if label_counts else 0),
        "duplicate_variable_label_count_ge3": int(sum(1 for count in label_counts.values() if int(count) >= 3)),
        "duplicate_variable_labels_ge3": sorted(label for label, count in label_counts.items() if int(count) >= 3)[:20],
        "atom_count": atom_count,
        "bond_count": bond_count,
        "ring_count": int(len(atom_rings)),
        "max_ring_size": int(max((len(ring) for ring in atom_rings), default=0)),
        "large_ring_count_ge8": int(sum(1 for ring in atom_rings if len(ring) >= 8)),
        "large_ring_count_ge12": int(sum(1 for ring in atom_rings if len(ring) >= 12)),
        "stereo_token_count": stereo_token_count,
        "slash_bond_token_count": slash_bond_token_count,
        "branch_token_count": branch_token_count,
        "ring_digit_token_count": ring_digit_token_count,
        "variable_anchor_density_per_real_atom": float(variable_count / real_atom_count) if real_atom_count else 0.0,
        "dummy_real_bond_density_per_real_atom": float(dummy_real_bonds / real_atom_count) if real_atom_count else 0.0,
        "branch_density_per_real_atom": float(branch_token_count / real_atom_count) if real_atom_count else 0.0,
        "source_pose_risk_metrics_only": True,
    }


def bucket_for_positive_count(value: int) -> str | None:
    if value <= 0:
        return None
    return count_bucket(value)


def cxsmiles_dummy_anchor_audit(cxsmiles: str, annotation: str = "", cxsmiles_opt: str = "") -> dict[str, Any]:
    Chem = import_rdkit()
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(str(cxsmiles or ""), parser_params)
    issues: list[str] = []
    all_dummy_labels = cxsmiles_dummy_labels(cxsmiles)
    variable_dummy_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if is_markush_label(label)
    }
    fixed_abbreviation_pseudo_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if is_fixed_substituent_label(label)
    }
    non_variable_pseudo_labels = {
        int(index): label
        for index, label in all_dummy_labels.items()
        if not is_markush_label(label) and not is_fixed_substituent_label(label)
    }
    annotation_count = len(markush_r_variable_labels(annotation, cxsmiles_opt))
    if molecule is None:
        return {
            "passed": False,
            "issues": ["rdkit_cxsmiles_parse_failed"],
            "dummy_label_count": int(len(variable_dummy_labels)),
            "dummy_label_bucket": bucket_for_positive_count(len(variable_dummy_labels)),
            "dummy_label_indices": sorted(int(index) for index in variable_dummy_labels),
            "all_pseudo_atom_label_count": int(len(all_dummy_labels)),
            "all_pseudo_atom_label_indices": sorted(int(index) for index in all_dummy_labels),
            "fixed_abbreviation_pseudo_atom_count": int(len(fixed_abbreviation_pseudo_labels)),
            "fixed_abbreviation_pseudo_atom_indices": sorted(int(index) for index in fixed_abbreviation_pseudo_labels),
            "non_variable_pseudo_atom_count": int(len(non_variable_pseudo_labels)),
            "non_variable_pseudo_atom_indices": sorted(int(index) for index in non_variable_pseudo_labels),
            "pseudo_atom_label_categories": {
                str(index): markush_label_category(label) for index, label in sorted(all_dummy_labels.items())
            },
            "annotation_r_count": int(annotation_count),
            "annotation_count_matches_dummy_label_count": bool(annotation_count == len(variable_dummy_labels)),
            "dummy_atoms_without_real_neighbor": [],
            "dummy_atoms_without_graph_bond": [],
            "policy": "training variable count uses only Markush variable labels from CXSMILES dummyLabel/atomLabel/molFileAlias; fixed abbreviation pseudo atoms are diagnostic and not substitution anchors",
        }

    source_geometry = cxsmiles_source_geometry_metrics(molecule, variable_dummy_labels, cxsmiles)
    dummy_atoms_without_real_neighbor: list[int] = []
    dummy_atoms_without_graph_bond: list[int] = []
    for atom_index in sorted(variable_dummy_labels):
        atom = molecule.GetAtomWithIdx(int(atom_index))
        neighbors = list(atom.GetNeighbors())
        if not neighbors:
            dummy_atoms_without_graph_bond.append(int(atom_index))
            continue
        if not any(int(neighbor.GetAtomicNum()) != 0 for neighbor in neighbors):
            dummy_atoms_without_real_neighbor.append(int(atom_index))

    if not variable_dummy_labels:
        issues.append("missing_cxsmiles_dummyLabel_atomProp")
    if int(source_geometry["real_atom_count"]) < 3:
        issues.append("source_geometry_fewer_than_3_real_atoms")
    if dummy_atoms_without_graph_bond:
        issues.append("dummy_atom_has_no_graph_bond")
    if dummy_atoms_without_real_neighbor:
        issues.append("dummy_atom_has_no_real_atom_neighbor")

    dummy_label_count = int(len(variable_dummy_labels))
    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "dummy_label_count": dummy_label_count,
        "dummy_label_bucket": bucket_for_positive_count(dummy_label_count),
        "dummy_label_indices": sorted(int(index) for index in variable_dummy_labels),
        "all_pseudo_atom_label_count": int(len(all_dummy_labels)),
        "all_pseudo_atom_label_indices": sorted(int(index) for index in all_dummy_labels),
        "fixed_abbreviation_pseudo_atom_count": int(len(fixed_abbreviation_pseudo_labels)),
        "fixed_abbreviation_pseudo_atom_indices": sorted(int(index) for index in fixed_abbreviation_pseudo_labels),
        "non_variable_pseudo_atom_count": int(len(non_variable_pseudo_labels)),
        "non_variable_pseudo_atom_indices": sorted(int(index) for index in non_variable_pseudo_labels),
        "pseudo_atom_label_categories": {
            str(index): markush_label_category(label) for index, label in sorted(all_dummy_labels.items())
        },
        "annotation_r_count": int(annotation_count),
        "annotation_count_matches_dummy_label_count": bool(annotation_count == dummy_label_count),
        "source_geometry": source_geometry,
        "dummy_atoms_without_real_neighbor": dummy_atoms_without_real_neighbor,
        "dummy_atoms_without_graph_bond": dummy_atoms_without_graph_bond,
        "policy": "training variable count uses only Markush variable labels from CXSMILES dummyLabel/atomLabel/molFileAlias; fixed abbreviation pseudo atoms are diagnostic and not substitution anchors; source geometry must contain at least 3 real atoms before rendering",
    }


def rdkit_canonical_smiles(smiles: str, *, strip_dummy: bool) -> str:
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_heldout_block_keys(heldout_csvs: list[Path]) -> dict[str, set[str]]:
    keys = {
        "source_ids": set(),
        "source_document_keys": set(),
        "canonical_smiles": set(),
        "canonical_backbone_smiles": set(),
    }
    for csv_path in heldout_csvs:
        for row in read_csv_rows(csv_path):
            source_id = str(row.get("source_id") or "").strip()
            source_gold = str(row.get("source_gold") or "").strip()
            source_document = str(row.get("source_document_key") or "").strip()
            if source_id:
                keys["source_ids"].add(source_id)
            if source_gold:
                keys["source_ids"].add(source_gold)
            if source_document:
                keys["source_document_keys"].add(source_document)
            smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "")
            full = rdkit_canonical_smiles(smiles, strip_dummy=False)
            backbone = rdkit_canonical_smiles(smiles, strip_dummy=True)
            if full:
                keys["canonical_smiles"].add(full)
            if backbone:
                keys["canonical_backbone_smiles"].add(backbone)
    return keys


def heldout_block_reasons(row: dict[str, str], block_keys: dict[str, set[str]], *, allow_backbone_overlap: bool) -> list[str]:
    reasons: list[str] = []
    source_id = str(row.get("source_id") or "").strip()
    if source_id and source_id in block_keys["source_ids"]:
        reasons.append("source_id_overlap")
    source_document = str(row.get("source_document_key") or "").strip()
    if source_document and source_document in block_keys["source_document_keys"]:
        reasons.append("source_document_key_overlap")
    cxsmiles = str(row.get("cxsmiles") or "")
    full = rdkit_canonical_smiles(cxsmiles, strip_dummy=False)
    backbone = rdkit_canonical_smiles(cxsmiles, strip_dummy=True)
    if full and full in block_keys["canonical_smiles"]:
        reasons.append("canonical_smiles_overlap")
    if backbone and backbone in block_keys["canonical_backbone_smiles"] and not allow_backbone_overlap:
        reasons.append("canonical_backbone_smiles_overlap")
    return reasons


def filter_heldout_overlaps(
    rows: list[dict[str, str]],
    *,
    heldout_csvs: list[Path],
    allow_backbone_overlap: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not heldout_csvs:
        return rows, {
            "enabled": False,
            "heldout_csv": [],
            "input_rows": len(rows),
            "kept_rows": len(rows),
            "rejected_rows": 0,
            "reason_counts": {},
            "rejected_examples": [],
            "allow_backbone_overlap": bool(allow_backbone_overlap),
        }
    block_keys = collect_heldout_block_keys(heldout_csvs)
    kept: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons = heldout_block_reasons(row, block_keys, allow_backbone_overlap=allow_backbone_overlap)
        if reasons:
            reason_counts.update(reasons)
            rejected.append(
                {
                    "source_id": str(row.get("source_id") or ""),
                    "source_document_key": str(row.get("source_document_key") or ""),
                    "annotation_r_bucket": str(row.get("annotation_r_bucket") or ""),
                    "reasons": reasons,
                }
            )
        else:
            kept.append(row)
    return kept, {
        "enabled": True,
        "heldout_csv": [str(path) for path in heldout_csvs],
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "reason_counts": dict(reason_counts),
        "rejected_examples": rejected[:80],
        "allow_backbone_overlap": bool(allow_backbone_overlap),
        "policy": {
            "source_id": "candidate source_id must not overlap heldout source_id/source_gold",
            "source_document_key": "candidate source_document_key must not overlap heldout source_document_key",
            "canonical_smiles": "candidate full canonical SMILES must not overlap heldout canonical SMILES",
            "canonical_backbone_smiles": "candidate dummy-stripped backbone must not overlap heldout backbone unless explicitly allowed",
        },
    }


def source_document_key(subset: str, raw_id: str, cxsmiles: str, page_image_path: str) -> str:
    record = str(raw_id or "").strip()
    if not record:
        record = fallback_source_record_id(cxsmiles, page_image_path)
    if ".pdf_" in record:
        document = record.split(".pdf_", 1)[0] + ".pdf"
    else:
        document = re.sub(r"_[0-9]+(?:_[0-9]+)+$", "", record)
    return f"{subset}:{document}"


def fallback_source_record_id(cxsmiles: str, page_image_path: str) -> str:
    return f"missing-id-{stable_hexdigest(cxsmiles, page_image_path)[:24]}"


def parse_bucket_targets(value: str, *, default_total_rows: int) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.isdigit():
        per_bucket = int(text)
        return {bucket: per_bucket for bucket in COUNT_BUCKETS}
    targets: dict[str, int] = {}
    for item in re.split(r"[,;]", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid bucket target {item!r}; expected bucket=count")
        bucket, count_text = [part.strip() for part in item.split("=", 1)]
        if bucket not in COUNT_BUCKETS:
            raise ValueError(f"invalid bucket {bucket!r}; expected one of {COUNT_BUCKETS}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"bucket target must be non-negative: {item!r}")
        targets[bucket] = count
    if not targets and default_total_rows > 0:
        per_bucket = max(1, math.ceil(default_total_rows / len(COUNT_BUCKETS)))
        targets = {bucket: per_bucket for bucket in COUNT_BUCKETS}
    return targets


def resolve_candidate_targets(
    *,
    accepted_targets: dict[str, int],
    candidate_targets: dict[str, int],
    rows: int,
    candidate_multiplier: int,
) -> dict[str, int]:
    if candidate_targets:
        resolved = {bucket: int(candidate_targets.get(bucket, 0)) for bucket in COUNT_BUCKETS}
        blockers = [
            f"candidate target for bucket {bucket} ({resolved[bucket]}) is below accepted target ({int(accepted_targets.get(bucket, 0))})"
            for bucket in COUNT_BUCKETS
            if int(accepted_targets.get(bucket, 0)) > 0 and resolved[bucket] < int(accepted_targets.get(bucket, 0))
        ]
        if blockers:
            raise ValueError("; ".join(blockers))
        return resolved
    if accepted_targets:
        return {
            bucket: int(accepted_targets.get(bucket, 0)) * max(1, int(candidate_multiplier))
            for bucket in COUNT_BUCKETS
        }
    base = max(1, int(rows))
    desired_total = base * max(1, int(candidate_multiplier))
    per_bucket = max(1, math.ceil(desired_total / len(COUNT_BUCKETS)))
    return {bucket: per_bucket for bucket in COUNT_BUCKETS}


def subset_name_for_path(raw_root: Path, path: Path) -> str:
    try:
        relative_parent = path.parent.relative_to(raw_root)
    except ValueError:
        return path.parent.name
    text = relative_parent.as_posix()
    return text if text and text != "." else path.parent.name


def read_arrow_or_parquet_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        schema = set(pq.read_schema(path).names)
        selected = [name for name in columns if name in schema]
        return pq.read_table(path, columns=selected).to_pylist()
    if path.suffix == ".arrow":
        import pyarrow.ipc as ipc
        from pyarrow.lib import ArrowInvalid

        with path.open("rb") as handle:
            try:
                table = ipc.open_file(handle).read_all()
            except ArrowInvalid:
                handle.seek(0)
                table = ipc.open_stream(handle).read_all()
        selected = [name for name in columns if name in set(table.column_names)]
        return table.select(selected).to_pylist()
    return []


def iter_source_paths(raw_root: Path, subset_glob: str) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in re.split(r"[,;]", str(subset_glob or "")):
        pattern = pattern.strip()
        if not pattern:
            continue
        for path_text in glob.glob(str(raw_root / pattern), recursive=True):
            path = Path(path_text)
            if path.suffix not in {".arrow", ".parquet"}:
                continue
            paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def iter_markush_rows(
    raw_root: Path,
    subset_glob: str,
    limit: int | None = None,
    *,
    include_report: bool = False,
) -> list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, Any]]:
    wanted_columns = ["id", "annotation", "cxsmiles_dataset", "cxsmiles", "cxsmiles_opt", "page_image_path"]

    rows: list[dict[str, str]] = []
    rejection_counts: Counter[str] = Counter()
    markush_source_rows = 0
    for path in iter_source_paths(raw_root, subset_glob):
        subset = subset_name_for_path(raw_root, path)
        for item in read_arrow_or_parquet_rows(path, wanted_columns):
            cxsmiles = clean_cxsmiles(item.get("cxsmiles") or item.get("cxsmiles_dataset") or item.get("cxsmiles_opt") or "")
            annotation = str(item.get("annotation") or "")
            cxsmiles_opt = str(item.get("cxsmiles_opt") or "")
            if not cxsmiles or "<markush" not in annotation.lower():
                continue
            markush_source_rows += 1
            anchor_audit = cxsmiles_dummy_anchor_audit(cxsmiles, annotation, cxsmiles_opt)
            if not anchor_audit["passed"]:
                rejection_counts.update(anchor_audit["issues"])
                continue
            r_count = int(anchor_audit["dummy_label_count"])
            bucket_value = anchor_audit["dummy_label_bucket"]
            if bucket_value is None:
                continue
            bucket = str(bucket_value)
            raw_id = str(item.get("id") or "").strip()
            page_image_path = str(item.get("page_image_path") or "")
            source_record_id = raw_id or fallback_source_record_id(cxsmiles, page_image_path)
            source_doc = source_document_key(subset, raw_id, cxsmiles, page_image_path)
            rows.append(
                {
                    "source_id": f"markush:{subset}:{source_record_id}",
                    "source_file": str(path),
                    "source_record_id": source_record_id,
                    "source_document_key": source_doc,
                    "cxsmiles": cxsmiles,
                    "annotation": annotation,
                    "cxsmiles_opt": cxsmiles_opt,
                    "annotation_r_count": str(r_count),
                    "annotation_r_bucket": bucket,
                    "cxsmiles_dummy_label_count": str(r_count),
                    "cxsmiles_dummy_label_bucket": bucket,
                    "variable_anchor_count": str(r_count),
                    "variable_anchor_bucket": bucket,
                    "all_pseudo_atom_label_count": str(anchor_audit["all_pseudo_atom_label_count"]),
                    "fixed_abbreviation_pseudo_atom_count": str(anchor_audit["fixed_abbreviation_pseudo_atom_count"]),
                    "non_variable_pseudo_atom_count": str(anchor_audit["non_variable_pseudo_atom_count"]),
                    "annotation_r_count_diagnostic": str(anchor_audit["annotation_r_count"]),
                    "annotation_count_matches_dummy_label_count": str(anchor_audit["annotation_count_matches_dummy_label_count"]),
                    "source_anchor_trainable": str(bool(anchor_audit["passed"])),
                    "source_anchor_audit": json.dumps(anchor_audit, ensure_ascii=False, sort_keys=True),
                    "cxsmiles_star_count": str(cxsmiles_star_count(cxsmiles)),
                    "page_image_path": page_image_path,
                    "subset": subset,
                }
            )
            if limit is not None and len(rows) >= limit:
                if include_report:
                    return rows, {
                        "enabled": True,
                        "input_markush_rows": int(markush_source_rows),
                        "kept_rows": int(len(rows)),
                        "rejected_rows": int(sum(rejection_counts.values())),
                        "reason_counts": dict(sorted(rejection_counts.items())),
                    }
                return rows
    if include_report:
        return rows, {
            "enabled": True,
            "input_markush_rows": int(markush_source_rows),
            "kept_rows": int(len(rows)),
            "rejected_rows": int(sum(rejection_counts.values())),
            "reason_counts": dict(sorted(rejection_counts.items())),
            "policy": (
                "CXSMILES dummyLabel/atomLabel/molFileAlias anchors must be Markush variable labels; "
                "fixed substituent abbreviations and other non-variable pseudo atoms are diagnostic only; "
                "each variable anchor must have a real-atom graph neighbor; source geometry must contain at least 3 real atoms"
            ),
        }
    return rows


def select_markush_candidates(
    raw_root: Path,
    subset_glob: str,
    *,
    rows: int,
    bucket_targets: dict[str, int],
    candidate_bucket_targets: dict[str, int],
    candidate_multiplier: int,
    seed: int,
    heldout_csvs: list[Path],
    allow_backbone_overlap: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw_rows = iter_markush_rows(raw_root, subset_glob, None)
    raw_rows, heldout_filter_report = filter_heldout_overlaps(
        raw_rows,
        heldout_csvs=heldout_csvs,
        allow_backbone_overlap=allow_backbone_overlap,
    )
    by_document: dict[str, dict[str, str]] = {}
    duplicate_documents = 0
    for row in raw_rows:
        key = row["source_document_key"]
        sort_key = stable_hexdigest(seed, row["source_id"], row["cxsmiles"], row["annotation_r_count"])
        row = row | {"selection_hash": sort_key}
        current = by_document.get(key)
        if current is None or sort_key < current["selection_hash"]:
            by_document[key] = row
        else:
            duplicate_documents += 1

    unique_rows = list(by_document.values())
    unique_rows.sort(key=lambda item: item["selection_hash"])
    available_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique_rows:
        available_by_bucket[row["annotation_r_bucket"]].append(row)

    desired_candidates = resolve_candidate_targets(
        accepted_targets=bucket_targets,
        candidate_targets=candidate_bucket_targets,
        rows=int(rows),
        candidate_multiplier=int(candidate_multiplier),
    )

    blockers = []
    selected_by_bucket: dict[str, list[dict[str, str]]] = {}
    for bucket in COUNT_BUCKETS:
        available = available_by_bucket.get(bucket, [])
        wanted = desired_candidates.get(bucket, 0)
        selected_by_bucket[bucket] = available[:wanted]
        required_trainable = int(bucket_targets.get(bucket, 0)) if bucket_targets else 1
        if required_trainable > 0 and len(available) < wanted:
            blockers.append(
                f"bucket {bucket} has {len(available)} source-document candidates; requested {wanted} "
                f"for target {required_trainable} with multiplier {candidate_multiplier}"
            )

    selected: list[dict[str, str]] = []
    max_len = max((len(items) for items in selected_by_bucket.values()), default=0)
    for offset in range(max_len):
        for bucket in COUNT_BUCKETS:
            items = selected_by_bucket.get(bucket, [])
            if offset < len(items):
                selected.append(items[offset])

    selected_documents = [row["source_document_key"] for row in selected]
    selected_ids = [row["source_id"] for row in selected]
    report = {
        "raw_root": str(raw_root),
        "subset_glob": subset_glob,
        "seed": int(seed),
        "selection_policy": "source_document_deduplicated_stable_sha256_stratified_by_annotation_r_bucket",
        "raw_candidate_rows": len(raw_rows),
        "heldout_overlap_filter": heldout_filter_report,
        "unique_source_document_candidates": len(unique_rows),
        "duplicate_source_document_rows_dropped": int(duplicate_documents),
        "bucket_targets": {bucket: int(bucket_targets.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "candidate_multiplier": int(candidate_multiplier),
        "candidate_bucket_targets_explicit": bool(candidate_bucket_targets),
        "candidate_selection_target_policy": (
            "explicit_candidate_bucket_targets"
            if candidate_bucket_targets
            else "accepted_bucket_targets_times_candidate_multiplier"
            if bucket_targets
            else "rows_times_candidate_multiplier_evenly_spread"
        ),
        "desired_candidate_rows_by_bucket": {bucket: int(desired_candidates.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "available_source_documents_by_bucket": {
            bucket: int(len(available_by_bucket.get(bucket, []))) for bucket in COUNT_BUCKETS
        },
        "selected_candidate_rows_by_bucket": {
            bucket: int(len(selected_by_bucket.get(bucket, []))) for bucket in COUNT_BUCKETS
        },
        "selected_candidate_rows": len(selected),
        "selected_source_document_overlap": len(selected_documents) - len(set(selected_documents)),
        "selected_source_id_overlap": len(selected_ids) - len(set(selected_ids)),
        "blockers": blockers,
        "plan_only_safe_to_generate": not blockers and len(selected) > 0,
        "caveat": (
            "This is a candidate-selection plan only. Rows become trainable only after CDK generation, "
            "pose validation, visual review, source-leak checks, coverage audit, and acceptance gates pass."
        ),
    }
    return selected, report


def ensure_markush_generator_dataset(dataset_name: str) -> tuple[Path, Path]:
    dataset_root = MARKUSH_GENERATOR_ROOT / "data" / "dataset" / dataset_name
    images = dataset_root / "images"
    molfiles = dataset_root / "molfiles"
    images.mkdir(parents=True, exist_ok=True)
    molfiles.mkdir(parents=True, exist_ok=True)
    return images, molfiles


def ensure_depictor_markush_dummy_label_patch() -> None:
    """Make MarkushGenerator render CXSMILES dummy labels as visible pseudo atom labels."""
    path = MARKUSH_GENERATOR_ROOT / "markushgenerator" / "image_generation" / "Depictor.java"
    text = path.read_text(encoding="utf-8")
    if DEPICTOR_MARKUSH_LABEL_PATCH_MARKER in text:
        return
    import_anchor = "import java.nio.file.Paths;\n"
    if import_anchor not in text:
        raise RuntimeError("Depictor.java patch anchor missing: java.nio.file.Paths import")
    text = text.replace(import_anchor, import_anchor + "import java.util.regex.Matcher;\n", 1)
    method_anchor = "\n\tpublic void generate_image_masks_label"
    helper = f"""

    // {DEPICTOR_MARKUSH_LABEL_PATCH_MARKER}
    private static void applyCxSmilesDummyLabels(IAtomContainer molecule, String cxsmiles) {{
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(?:^|[:|,])(\\\\d+)\\\\.(?:dummyLabel|atomLabel|molFileAlias)\\\\.([^:|,]+)");
        Matcher matcher = pattern.matcher(cxsmiles == null ? "" : cxsmiles);
        while (matcher.find()) {{
            int atomIndex = Integer.parseInt(matcher.group(1));
            if (atomIndex < 0 || atomIndex >= molecule.getAtomCount()) {{
                continue;
            }}
            String label = matcher.group(2).trim();
            if (label.length() == 0 || label.equals("*")) {{
                continue;
            }}
            IAtom atom = molecule.getAtom(atomIndex);
            if (atom instanceof IPseudoAtom) {{
                IPseudoAtom pseudo = (IPseudoAtom) atom;
                pseudo.setLabel(label);
                pseudo.setSymbol(label);
            }} else if ("*".equals(atom.getSymbol())) {{
                atom.setSymbol(label);
            }}
            atom.setProperty("dummyLabel", label);
            atom.setProperty("atomLabel", label);
            atom.setProperty(CDKConstants.COMMENT, label);
        }}
    }}
"""
    if method_anchor not in text:
        raise RuntimeError("Depictor.java patch anchor missing: generate_image_masks_label")
    text = text.replace(method_anchor, helper + method_anchor, 1)
    parse_call_pattern = re.compile(
        r"^([ \t]*)(?:[\w.]+\s+)?molecule\s*=\s*parser\.parseSmiles\(smiles\);[ \t]*$",
        re.MULTILINE,
    )
    parse_match = parse_call_pattern.search(text)
    if parse_match is None:
        raise RuntimeError("Depictor.java patch anchor missing: parser.parseSmiles")
    parse_indent = parse_match.group(1)
    text = (
        text[: parse_match.end()]
        + "\n"
        + parse_indent
        + "    applyCxSmilesDummyLabels(molecule, smiles);"
        + text[parse_match.end():]
    )
    path.write_text(text, encoding="utf-8")


def compile_depictor() -> None:
    ensure_depictor_markush_dummy_label_patch()
    current_dir = MARKUSH_GENERATOR_ROOT / "markushgenerator" / "image_generation"
    proc = subprocess.run(
        ["javac", "-cp", "../../lib/*:.", "Depictor.java", "BatchDepictor.java"],
        cwd=current_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"javac failed: {proc.stderr.strip() or proc.stdout.strip()}")


def write_batch_depictor_tasks(path: Path, tasks: list[dict[str, Any]], dataset_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("row_id\tseed\tdataset\tcxsmiles_base64\n")
        for task in tasks:
            encoded = base64.b64encode(str(task["candidate"]["cxsmiles"]).encode("utf-8")).decode("ascii")
            handle.write(f"{task['row_id']}\t{int(task['render_seed'])}\t{dataset_name}\t{encoded}\n")


def read_batch_depictor_status(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"batch depictor status file missing: {path}")
    statuses: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid batch depictor status JSON on line {line_number}: {exc}") from exc
            row_id = str(value.get("row_id") or "")
            if not row_id:
                raise RuntimeError(f"batch depictor status line {line_number} missing row_id")
            statuses[row_id] = value
    return statuses


def run_depictor_batch(tasks: list[dict[str, Any]], dataset_name: str, output_dir: Path) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    current_dir = MARKUSH_GENERATOR_ROOT / "markushgenerator" / "image_generation"
    task_path = (output_dir / "batch_depictor_tasks.tsv").resolve()
    status_path = (output_dir / "batch_depictor_status.jsonl").resolve()
    write_batch_depictor_tasks(task_path, tasks, dataset_name)
    proc = subprocess.run(
        ["java", "-cp", "../../lib/*:.", "BatchDepictor", str(task_path), str(status_path)],
        cwd=current_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(300, 20 * len(tasks)),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"java batch depictor failed: {proc.stderr.strip() or proc.stdout.strip()}")
    statuses = read_batch_depictor_status(status_path)
    missing = [str(task["row_id"]) for task in tasks if str(task["row_id"]) not in statuses]
    if missing:
        raise RuntimeError(f"batch depictor omitted {len(missing)} task statuses; examples={missing[:5]}")
    return statuses


def run_depictor(cxsmiles: str, row_id: str, dataset_name: str, render_seed: int) -> None:
    current_dir = MARKUSH_GENERATOR_ROOT / "markushgenerator" / "image_generation"
    proc = subprocess.run(
        ["java", f"-Dmarkush.seed={int(render_seed)}", "-cp", "../../lib/*:.", "Depictor", cxsmiles, row_id, dataset_name],
        cwd=current_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"java depictor failed: {proc.stderr.strip() or proc.stdout.strip()}")


def read_render_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"CDK render metadata missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"CDK render metadata is not a JSON object: {path}")
    required = {
        "seed",
        "stroke_ratio",
        "bond_separation",
        "symbol_margin_ratio",
        "font_name",
        "font_size",
        "render_atom_numbers",
        "render_carbon_symbols",
        "render_aromatic_display",
        "render_deuterium_symbol",
        "render_terminal_carbons",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"CDK render metadata missing keys: {missing}")
    return value


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_v3000_mol(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atoms: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    in_atoms = False
    in_bonds = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "BEGIN ATOM" in line:
            in_atoms = True
            continue
        if "END ATOM" in line:
            in_atoms = False
            continue
        if "BEGIN BOND" in line:
            in_bonds = True
            continue
        if "END BOND" in line:
            in_bonds = False
            continue
        if not line.startswith("M  V30"):
            continue
        parts = [part for part in line.split() if part]
        if in_atoms and len(parts) >= 8:
            coord_start = None
            for index in range(4, len(parts) - 2):
                try:
                    float(parts[index])
                    float(parts[index + 1])
                    float(parts[index + 2])
                except ValueError:
                    continue
                coord_start = index
                break
            if coord_start is None:
                raise ValueError(f"V3000 atom line missing numeric coordinate triplet: {line}")
            token = " ".join(parts[3:coord_start])
            props: dict[str, Any] = {}
            for part in parts[coord_start + 4 :]:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                props[key] = value
            atoms.append(
                {
                    "atom_index": int(parts[2]) - 1,
                    "token": token,
                    "mol_x": float(parts[coord_start]),
                    "mol_y": float(parts[coord_start + 1]),
                    "v3000_props": props,
                }
            )
        if in_bonds and len(parts) >= 6:
            bonds.append(
                {
                    "bond_index": int(parts[2]) - 1,
                    "bond_order": str(parts[3]),
                    "begin_atom_index": int(parts[4]) - 1,
                    "end_atom_index": int(parts[5]) - 1,
                }
            )
    return atoms, bonds


def svg_path_chord_line(path_d: str) -> tuple[float, float, float, float] | None:
    try:
        from svgpathtools import parse_path

        path = parse_path(str(path_d or ""))
    except Exception:
        return None
    if path is None or len(path) == 0:
        return None
    try:
        start = path.point(0.0)
        end = path.point(1.0)
    except Exception:
        return None
    x1, y1, x2, y2 = float(start.real), float(start.imag), float(end.real), float(end.imag)
    if math.hypot(x2 - x1, y2 - y1) <= 1e-6:
        return None
    return (x1, y1, x2, y2)


def parse_svg_geometry(path: Path) -> tuple[tuple[float, float, float, float], dict[int, list[tuple[float, float, float, float]]], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    viewbox_match = re.search(r"viewBox=['\"]([^'\"]+)['\"]", text)
    if not viewbox_match:
        raise ValueError("SVG missing viewBox")
    viewbox = tuple(float(part) for part in viewbox_match.group(1).split())
    root = ET.fromstring(text)
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]] = {}
    cells: list[dict[str, Any]] = []

    for elem in root.iter():
        elem_id = elem.attrib.get("id", "")
        match = re.match(r"mol\d+bnd(\d+)$", elem_id)
        if match:
            bond_index = int(match.group(1)) - 1
            for child in elem.iter():
                if local_name(child.tag) == "line" and {"x1", "y1", "x2", "y2"} <= set(child.attrib):
                    lines_by_bond.setdefault(bond_index, []).append(
                        (
                            float(child.attrib["x1"]),
                            float(child.attrib["y1"]),
                            float(child.attrib["x2"]),
                            float(child.attrib["y2"]),
                        )
                    )
                elif local_name(child.tag) == "path" and child.attrib.get("d"):
                    chord = svg_path_chord_line(str(child.attrib.get("d") or ""))
                    if chord is not None:
                        lines_by_bond.setdefault(bond_index, []).append(chord)
        atom_match = re.match(r"mol\d+atm(\d+)$", elem_id)
        if atom_match:
            bbox = path_bbox_from_element(elem)
            if bbox:
                cells.append(
                    {
                        "bbox": [bbox[0] / viewbox[2], bbox[1] / viewbox[3], bbox[2] / viewbox[2], bbox[3] / viewbox[3]],
                        "text": "",
                        "source": "svg_atom_path",
                        "atom_index": int(atom_match.group(1)) - 1,
                    }
                )
    return viewbox, lines_by_bond, cells


def real_atom_count(atoms: list[dict[str, Any]]) -> int:
    return sum(1 for atom in atoms if str(atom.get("token") or "") in REAL_ATOM_TOKENS)


def bbox_area(cell: dict[str, Any]) -> float:
    bbox = cell.get("bbox") if isinstance(cell, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def markush_visual_quality_metrics(atoms: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    areas = [bbox_area(cell) for cell in cells]
    return {
        "atom_count": len(atoms),
        "real_atom_count": real_atom_count(atoms),
        "ocr_cell_count": len(cells),
        "ocr_cell_area_sum": float(sum(areas)),
        "ocr_cell_area_max": float(max(areas) if areas else 0.0),
    }


def validate_markush_visual_quality(atoms: list[dict[str, Any]], cells: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    metrics = markush_visual_quality_metrics(atoms, cells)
    blockers: list[str] = []
    if int(metrics["real_atom_count"]) < 3:
        blockers.append("markush layout has fewer than 3 real chemical atoms; likely label-only/pseudo-atom diagram")
    if float(metrics["ocr_cell_area_sum"]) > 0.12:
        blockers.append("markush OCR text boxes occupy too much image area; likely text-dominated diagram")
    if float(metrics["ocr_cell_area_max"]) > 0.08:
        blockers.append("a single Markush OCR text box is too large; likely rendered as label block instead of structure")
    return metrics, blockers


def build_document_realism_contract(
    *,
    render_metadata: dict[str, Any],
    visual_metrics: dict[str, Any],
    source_dataset: str,
    source_file: str,
    depictor_mode: str,
) -> dict[str, Any]:
    missing = sorted(REALISM_RENDER_PARAMETER_KEYS - set(render_metadata))
    cdk_parameters_present = not missing
    ocr_area_sum = float(visual_metrics.get("ocr_cell_area_sum") or 0.0)
    ocr_area_max = float(visual_metrics.get("ocr_cell_area_max") or 0.0)
    real_atom_count_value = int(visual_metrics.get("real_atom_count") or 0)
    machine_audit_passed = (
        cdk_parameters_present
        and real_atom_count_value >= 3
        and ocr_area_sum <= 0.12
        and ocr_area_max <= 0.08
    )
    return {
        "schema_version": DOCUMENT_REALISM_SCHEMA_VERSION,
        "policy": DOCUMENT_REALISM_POLICY,
        "source_visual_domain": "patent_literature_markush_document_crop",
        "generation_method": "existing_markushgenerator_cdk_svg_depictor",
        "no_new_renderer_or_image_method": True,
        "machine_audit_passed": bool(machine_audit_passed),
        "manual_visual_review_required": True,
        "manual_visual_review_passed": False,
        "status": "machine_audit_passed_manual_visual_review_required"
        if machine_audit_passed
        else "machine_audit_failed",
        "source_dataset": str(source_dataset),
        "source_file": str(source_file),
        "external_basis": DOCUMENT_REALISM_REFERENCES,
        "render_parameter_keys_required": sorted(REALISM_RENDER_PARAMETER_KEYS),
        "render_parameter_keys_missing": missing,
        "render_parameters_present": cdk_parameters_present,
        "render_parameters": {
            key: render_metadata.get(key)
            for key in [
                "font_name",
                "font_size",
                "stroke_ratio",
                "bond_separation",
                "symbol_margin_ratio",
                "render_atom_numbers",
                "render_carbon_symbols",
                "render_aromatic_display",
                "render_deuterium_symbol",
                "render_terminal_carbons",
            ]
        },
        "machine_checks": {
            "cdk_render_parameters_present": cdk_parameters_present,
            "seeded_depictor_required": True,
            "existing_cdk_svg_pipeline_required": True,
            "real_backbone_present": real_atom_count_value >= 3,
            "ocr_text_not_dominant": ocr_area_sum <= 0.12,
            "single_ocr_cell_not_oversized": ocr_area_max <= 0.08,
            "readability_checked_by_pose_factory_image_stats": True,
            "manual_visual_review_required_for_formal_training": True,
            "depictor_mode": str(depictor_mode),
        },
        "limits": {
            "machine_audit_is_necessary_not_sufficient": True,
            "does_not_replace_manual_visual_review": True,
            "does_not_relax_pose_or_substitution_anchor_gates": True,
        },
    }


def apply_markush_document_context(
    image_path: Path,
    atom_coords: list[dict[str, Any]],
    ocr_cells: list[dict[str, Any]],
    *,
    seed: int,
    enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, dict[str, Any]]:
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
    import numpy as np

    rng = random.Random(int(seed))
    image = Image.open(image_path).convert("RGB")
    old_w, old_h = image.size
    operations: list[str] = []
    pad_left = pad_top = pad_right = pad_bottom = 0
    source_structure = image.convert("L")
    target_long_edge = max(1, max(image.size))
    resize_scale = 1.0
    coordinate_canvas_w, coordinate_canvas_h = old_w, old_h
    paper_profile = "none"
    paper_background = 255
    paper_noise_sigma = 0.0

    if enabled:
        pad_left = int(round(old_w * rng.uniform(0.02, 0.22)))
        pad_right = int(round(old_w * rng.uniform(0.04, 0.30)))
        pad_top = int(round(old_h * rng.uniform(0.03, 0.28)))
        pad_bottom = int(round(old_h * rng.uniform(0.04, 0.32)))
        profile_specs = {
            "clean_white": ([253, 254, 255], (0.05, 0.45), (0.00, 0.18)),
            "white_scan": ([250, 251, 252, 253, 254], (0.35, 1.25), (0.10, 0.45)),
            "gray_scan": ([246, 247, 248, 249, 250, 251, 252], (0.85, 2.20), (0.20, 0.85)),
            "aged_scan": ([242, 243, 244, 245, 246, 247, 248], (0.65, 1.80), (0.15, 0.70)),
        }
        paper_profile = rng.choices(
            ["clean_white", "white_scan", "gray_scan", "aged_scan"],
            weights=[0.22, 0.36, 0.32, 0.10],
            k=1,
        )[0]
        background_values, noise_range, wave_range = profile_specs[paper_profile]
        background = int(rng.choice(background_values))
        paper_background = background
        new_canvas_size = (old_w + pad_left + pad_right, old_h + pad_top + pad_bottom)
        coordinate_canvas_w, coordinate_canvas_h = new_canvas_size
        arr_bg = np.full((new_canvas_size[1], new_canvas_size[0]), background, dtype=np.int16)
        paper_noise_sigma = float(rng.uniform(*noise_range))
        paper_noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, paper_noise_sigma, arr_bg.shape)
        arr_bg = np.clip(arr_bg + paper_noise, 0, 255).astype("uint8")
        yy, xx = np.mgrid[0 : new_canvas_size[1], 0 : new_canvas_size[0]]
        paper_wave = (
            np.sin((xx + rng.uniform(0, 1000)) / rng.uniform(38.0, 86.0))
            + np.cos((yy + rng.uniform(0, 1000)) / rng.uniform(44.0, 112.0))
        ) * rng.uniform(*wave_range)
        arr_bg = np.clip(arr_bg.astype(np.float32) + paper_wave, 0, 255).astype("uint8")
        arr_struct = np.asarray(source_structure).astype(np.float32)
        alpha = np.clip((252.0 - arr_struct) / 42.0, 0.0, 1.0)
        alpha = np.power(alpha, 0.82)
        ink_values = np.clip(arr_struct * rng.uniform(0.82, 0.96), 0, 255)
        patch = arr_bg[pad_top : pad_top + old_h, pad_left : pad_left + old_w].astype(np.float32)
        patch = patch * (1.0 - alpha) + np.minimum(patch, ink_values) * alpha
        arr_bg[pad_top : pad_top + old_h, pad_left : pad_left + old_w] = np.clip(patch, 0, 255).astype("uint8")
        image = Image.fromarray(arr_bg, mode="L").convert("RGB")
        operations.append("asymmetric_white_document_margin")
        operations.append("alpha_antialiased_transparent_white_structure_composite")
        operations.append("diverse_patent_paper_before_structure_composite")
        operations.append(f"paper_profile:{paper_profile}")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        ink = rng.choice([(0, 0, 0), (18, 18, 18), (32, 32, 32), (48, 48, 48)])
        if rng.random() < 0.34 and pad_top >= 10:
            label = rng.choice(["(I)", "(II)", "A", "B", "Scheme", "Example", "Formula"])
            xy = (rng.randint(2, max(2, image.width // 3)), max(1, pad_top // 3))
            draw.text(xy, label, fill=ink, font=font)
            operations.append("small_margin_caption")
        if rng.random() < 0.30 and pad_bottom >= 10:
            label = rng.choice(["R = H or alkyl", "X = halogen", "Y = O, S", "Ar", "HetAr", "continued"])
            xy = (rng.randint(2, max(2, image.width // 2)), old_h + pad_top + max(1, pad_bottom // 4))
            draw.text(xy, label, fill=ink, font=font)
            operations.append("bottom_margin_markush_text")
        if rng.random() < 0.16:
            rule_candidates = []
            if pad_top >= 18:
                rule_candidates.append(max(2, pad_top // 3))
            if pad_bottom >= 18:
                rule_candidates.append(old_h + pad_top + max(2, (2 * pad_bottom) // 3))
            if rule_candidates:
                y = rng.choice(rule_candidates)
                x1 = rng.randint(0, max(0, image.width // 8))
                x2 = image.width - rng.randint(0, max(0, image.width // 8))
                draw.line((x1, y, x2, y + rng.choice([-1, 0, 1])), fill=ink, width=1)
                operations.append("thin_document_rule")
        if rng.random() < 0.12:
            for _ in range(rng.randint(1, 3)):
                x = rng.randint(0, max(0, image.width - 1))
                draw.line((x, 0, x + rng.choice([-1, 0, 1]), image.height), fill=(230, 230, 230), width=1)
            operations.append("faint_scan_column_artifact")
        current_long_edge = max(image.size)
        target_long_edge = int(
            round(
                min(
                    current_long_edge,
                    rng.choice([640, 704, 768, 832, 896, 960, 1024]) * rng.uniform(0.92, 1.06),
                )
            )
        )
        if current_long_edge > target_long_edge:
            resize_scale = float(target_long_edge) / float(current_long_edge)
            resample = Image.Resampling.BICUBIC if resize_scale > 0.58 else Image.Resampling.LANCZOS
            image = image.resize(
                (
                    max(64, int(round(image.width * resize_scale))),
                    max(64, int(round(image.height * resize_scale))),
                ),
                resample=resample,
            )
            operations.append("patent_crop_long_edge_scale_normalization")

    if rng.random() < 0.34:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.42)))
        operations.append("light_scan_blur")

    if rng.random() < 0.70:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        noise_sigma = rng.uniform(0.6, 2.7)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, noise_sigma, arr.shape)
        image = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"), mode="L").convert("RGB")
        operations.append("light_scan_noise")
    if rng.random() < 0.82:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        paper_mask = arr > 210
        dark_speckles = (np_rng.random(arr.shape) < rng.uniform(0.00022, 0.00120)) & paper_mask
        gray_speckles = (np_rng.random(arr.shape) < rng.uniform(0.00045, 0.00180)) & paper_mask
        arr[dark_speckles] = np.minimum(arr[dark_speckles], np_rng.integers(96, 185, size=int(dark_speckles.sum())))
        arr[gray_speckles] = np.minimum(arr[gray_speckles], np_rng.integers(178, 226, size=int(gray_speckles.sum())))
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L").convert("RGB")
        operations.append("fragment_like_sparse_scan_speckles")
    if rng.random() < 0.38:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        ink_mask = arr < 120
        dropout = (np_rng.random(arr.shape) < rng.uniform(0.00018, 0.00085)) & ink_mask
        if dropout.any():
            arr[dropout] = np_rng.integers(176, 232, size=int(dropout.sum()))
            operations.append("tiny_ink_dropout")
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L").convert("RGB")
    if rng.random() < 0.56:
        quality = rng.randint(74, 92)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        operations.append("jpeg_roundtrip")
    if rng.random() < 0.50:
        gray = image.convert("L")
        if rng.random() < 0.42:
            gray = ImageOps.autocontrast(gray, cutoff=rng.choice([0, 1]))
            operations.append("post_document_domain_autocontrast")
        if rng.random() < 0.58:
            gray = ImageEnhance.Contrast(gray).enhance(rng.uniform(1.02, 1.13))
            operations.append("post_document_domain_contrast")
        if rng.random() < 0.32:
            gray = ImageEnhance.Sharpness(gray).enhance(rng.uniform(1.03, 1.16))
            operations.append("post_document_domain_sharpen")
        arr = np.asarray(gray).astype(np.int16)
        ink_mask = arr < 210
        if ink_mask.any() and rng.random() < 0.56:
            arr[ink_mask] = np.clip(arr[ink_mask] * rng.uniform(0.90, 0.98), 0, 255)
            operations.append("post_document_domain_dark_strokes")
        if rng.random() < 0.44:
            arr = np.clip(
                arr + np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(0.10, 0.48), arr.shape),
                0,
                255,
            )
            operations.append("post_document_domain_fine_noise")
        gray = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L")
        if rng.random() < 0.22:
            buffer = BytesIO()
            gray.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(86, 96))
            buffer.seek(0)
            gray = Image.open(buffer).convert("L")
            operations.append("post_document_domain_jpeg_roundtrip")
        image = gray.convert("RGB")

    new_w, new_h = image.size

    def xform_point(x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) * old_w + pad_left) / max(1.0, float(coordinate_canvas_w)),
            (float(y) * old_h + pad_top) / max(1.0, float(coordinate_canvas_h)),
        )

    updated_atoms: list[dict[str, Any]] = []
    for atom in atom_coords:
        updated = dict(atom)
        updated["x"], updated["y"] = xform_point(float(atom["x"]), float(atom["y"]))
        updated_atoms.append(updated)

    updated_cells: list[dict[str, Any]] = []
    for cell in ocr_cells:
        updated = dict(cell)
        bbox = cell.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1 = xform_point(float(bbox[0]), float(bbox[1]))
            x2, y2 = xform_point(float(bbox[2]), float(bbox[3]))
            updated["bbox"] = [x1, y1, x2, y2]
        updated_cells.append(updated)

    image.save(image_path)
    arr = np.asarray(image.convert("L"))
    dark_ratio = float((arr < 245).mean())
    ink_ratio = float((arr < 220).mean())
    text_area_sum = 0.0
    text_area_max = 0.0
    for cell in updated_cells:
        bbox = cell.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            text_area_sum += area
            text_area_max = max(text_area_max, area)

    contract = {
        "schema_version": "markush_background_realism_v1",
        "policy": "coordinate_preserving_patent_literature_markush_context_transparent_white_composite_v4",
        "enabled": bool(enabled),
        "operations": operations,
        "seed": int(seed),
        "padding_px": [int(pad_left), int(pad_top), int(pad_right), int(pad_bottom)],
        "old_image_size": [int(old_w), int(old_h)],
        "new_image_size": [int(coordinate_canvas_w), int(coordinate_canvas_h)],
        "final_image_size": [int(new_w), int(new_h)],
        "target_long_edge_px": int(target_long_edge),
        "resize_scale_after_padding": float(resize_scale),
        "paper_profile": paper_profile,
        "paper_background_gray": int(paper_background),
        "paper_noise_sigma": float(paper_noise_sigma),
        "coordinate_mutation_policy": "synchronized_normalized_padding_then_uniform_image_scale_no_rotation_flip_perspective",
        "atom_coordinates_synchronized": True,
        "ocr_boxes_synchronized": True,
        "image_to_graph_orientation_alignment_preserved": True,
        "dark_pixel_ratio": dark_ratio,
        "ink_pixel_ratio": ink_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
        "ocr_text_area_sum": text_area_sum,
        "ocr_text_area_max": text_area_max,
        "operation_allowlist": [
            "asymmetric_white_document_margin",
            "alpha_antialiased_transparent_white_structure_composite",
            "diverse_patent_paper_before_structure_composite",
            "paper_profile:clean_white",
            "paper_profile:white_scan",
            "paper_profile:gray_scan",
            "paper_profile:aged_scan",
            "small_margin_caption",
            "bottom_margin_markush_text",
            "thin_document_rule",
            "faint_scan_column_artifact",
            "patent_crop_long_edge_scale_normalization",
            "light_scan_blur",
            "light_scan_noise",
            "fragment_like_sparse_scan_speckles",
            "tiny_ink_dropout",
            "jpeg_roundtrip",
            "post_document_domain_autocontrast",
            "post_document_domain_contrast",
            "post_document_domain_sharpen",
            "post_document_domain_dark_strokes",
            "post_document_domain_fine_noise",
            "post_document_domain_jpeg_roundtrip",
        ],
        "no_new_renderer_or_image_method": True,
        "manual_visual_review_required": False,
    }
    return updated_atoms, updated_cells, int(new_w), int(new_h), contract


def markush_molnextr_input_quality(
    image_path: Path,
    atom_coords: list[dict[str, Any]],
    ocr_cells: list[dict[str, Any]],
    *,
    input_size: int = 384,
    cropwhite_pad_px: int = 50,
) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    non_white = (arr != 255).sum(axis=2) > 0
    if non_white.any():
        ys, xs = np.where(non_white)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
    else:
        left, top, right, bottom = 0, 0, width, height
    cropped_w = max(1, right - left)
    cropped_h = max(1, bottom - top)
    padded_w = cropped_w + int(cropwhite_pad_px) * 2
    padded_h = cropped_h + int(cropwhite_pad_px) * 2
    scale_x = float(input_size) / float(padded_w)
    scale_y = float(input_size) / float(padded_h)
    min_scale = min(scale_x, scale_y)
    atom_points = []
    atom_outside_crop_pad = 0
    for atom in atom_coords:
        x = float(atom.get("x") or 0.0) * width
        y = float(atom.get("y") or 0.0) * height
        atom_points.append((x, y))
        if not (left - cropwhite_pad_px <= x <= right + cropwhite_pad_px and top - cropwhite_pad_px <= y <= bottom + cropwhite_pad_px):
            atom_outside_crop_pad += 1
    pair_distances = [
        math.hypot(atom_points[i][0] - atom_points[j][0], atom_points[i][1] - atom_points[j][1])
        for i in range(len(atom_points))
        for j in range(i + 1, len(atom_points))
    ]
    min_pair = float(min(pair_distances)) if pair_distances else None
    min_pair_at_384 = None if min_pair is None else float(min_pair * min_scale)
    ocr_min_side_at_384: list[float] = []
    ocr_area_sum = 0.0
    for cell in ocr_cells:
        bbox = cell.get("bbox") if isinstance(cell, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_w = max(0.0, (x2 - x1) * width)
        box_h = max(0.0, (y2 - y1) * height)
        if box_w > 0.0 and box_h > 0.0:
            ocr_min_side_at_384.append(float(min(box_w * scale_x, box_h * scale_y)))
            ocr_area_sum += float((x2 - x1) * (y2 - y1))
    min_ocr_side = float(min(ocr_min_side_at_384)) if ocr_min_side_at_384 else None
    arr_l = np.asarray(image.convert("L"))
    dark_ratio = float((arr_l < 245).mean())
    ink_ratio = float((arr_l < 220).mean())
    purewhite_ratio = float((arr_l >= 254).mean())
    blockers: list[str] = []
    if min_pair_at_384 is not None and min_pair_at_384 < 4.0:
        blockers.append("markush_molnextr_atom_coordinates_too_close_after_resize")
    if min_ocr_side is None or min_ocr_side < 3.5:
        blockers.append("markush_molnextr_variable_or_hetero_label_too_small_after_resize")
    if atom_outside_crop_pad:
        blockers.append("markush_molnextr_atom_coordinate_outside_cropwhite_pad")
    if dark_ratio < 0.002:
        blockers.append("markush_molnextr_input_too_blank")
    if dark_ratio > 0.65:
        blockers.append("markush_molnextr_input_too_dense")
    if ocr_area_sum > 0.12:
        blockers.append("markush_molnextr_ocr_text_dominates_input")
    return {
        "schema_version": "markush_molnextr_input_quality_v1",
        "policy": "emulate_molnextr_cropwhite_pad50_resize384_markush_label_pose_readability_gate",
        "passed": not blockers,
        "blockers": blockers,
        "input_size": int(input_size),
        "cropwhite_pad_px": int(cropwhite_pad_px),
        "source_image_size": [int(width), int(height)],
        "crop_box": [int(left), int(top), int(right), int(bottom)],
        "padded_size": [int(padded_w), int(padded_h)],
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "min_atom_pair_distance_px_at_384": min_pair_at_384,
        "min_ocr_cell_side_px_at_384": min_ocr_side,
        "atom_outside_cropwhite_pad_count": int(atom_outside_crop_pad),
        "ocr_cell_area_sum": float(ocr_area_sum),
        "dark_pixel_ratio": dark_ratio,
        "ink_pixel_ratio": ink_ratio,
        "purewhite_pixel_ratio": purewhite_ratio,
        "thresholds": {
            "min_atom_pair_distance_px_at_384": 4.0,
            "min_ocr_cell_side_px_at_384": 3.5,
            "max_ocr_cell_area_sum": 0.12,
            "min_dark_pixel_ratio": 0.002,
            "max_dark_pixel_ratio": 0.65,
        },
    }


def _line_samples(
    line: tuple[float, float, float, float],
    *,
    samples: int = 17,
) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(value) for value in line]
    count = max(2, int(samples))
    return [
        (
            x1 + (x2 - x1) * (index / (count - 1)),
            y1 + (y2 - y1) * (index / (count - 1)),
        )
        for index in range(count)
    ]


def _axis_segment_samples(
    axis: tuple[np.ndarray, float],
    source_lines: list[tuple[float, float, float, float]],
    atom_center_points: list[np.ndarray] | None = None,
    *,
    samples: int = 33,
    clip_box: tuple[float, float, float, float] | None = None,
) -> list[tuple[float, float]]:
    normal, offset = axis
    normal = np.asarray(normal, dtype=np.float64)
    direction = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return []
    direction = direction / norm
    points: list[np.ndarray] = []
    if atom_center_points:
        points.extend(np.asarray(point, dtype=np.float64) for point in atom_center_points)
    for line in source_lines:
        x1, y1, x2, y2 = [float(value) for value in line]
        points.append(np.asarray([x1, y1], dtype=np.float64))
        points.append(np.asarray([x2, y2], dtype=np.float64))
    if not points:
        return []
    projected_points = [
        np.asarray(point, dtype=np.float64) - normal * float(np.dot(np.asarray(point, dtype=np.float64), normal) - offset)
        for point in points
    ]
    projections = [float(np.dot(point, direction)) for point in projected_points]
    anchor = np.mean(np.asarray(projected_points, dtype=np.float64), axis=0)
    anchor_projection = float(np.dot(anchor, direction))
    start_projection = min(projections)
    end_projection = max(projections)
    if clip_box is not None:
        clipped = _axis_projection_interval_in_box(
            anchor=anchor,
            direction=direction,
            anchor_projection=anchor_projection,
            clip_box=clip_box,
        )
        if clipped is None:
            return []
        start_projection = max(start_projection, clipped[0])
        end_projection = min(end_projection, clipped[1])
        if end_projection < start_projection:
            center_projection = min(max(anchor_projection, clipped[0]), clipped[1])
            half = 0.5 * min(abs(clipped[1] - clipped[0]), 1.0)
            start_projection = center_projection - half
            end_projection = center_projection + half
    start = anchor + direction * (start_projection - anchor_projection)
    end = anchor + direction * (end_projection - anchor_projection)
    if clip_box is not None:
        start = _snap_point_to_clip_box_epsilon(start, clip_box)
        end = _snap_point_to_clip_box_epsilon(end, clip_box)
    return _line_samples((float(start[0]), float(start[1]), float(end[0]), float(end[1])), samples=samples)


def _snap_point_to_clip_box_epsilon(
    point: np.ndarray,
    clip_box: tuple[float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = [float(value) for value in clip_box]
    snapped = np.asarray(point, dtype=np.float64).copy()
    for axis_index, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
        value = float(snapped[axis_index])
        if low - epsilon <= value < low:
            snapped[axis_index] = low
        elif high < value <= high + epsilon:
            snapped[axis_index] = high
    return snapped


def _axis_projection_interval_in_box(
    *,
    anchor: np.ndarray,
    direction: np.ndarray,
    anchor_projection: float,
    clip_box: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    x_min, y_min, x_max, y_max = [float(value) for value in clip_box]
    intervals: list[tuple[float, float]] = []
    for axis_index, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
        component = float(direction[axis_index])
        origin = float(anchor[axis_index])
        if abs(component) <= 1e-12:
            if origin < low - 1e-9 or origin > high + 1e-9:
                return None
            continue
        t1 = (low - origin) / component
        t2 = (high - origin) / component
        intervals.append((min(t1, t2), max(t1, t2)))
    if not intervals:
        return None
    t_min = max(item[0] for item in intervals)
    t_max = min(item[1] for item in intervals)
    if t_max < t_min:
        return None
    return anchor_projection + t_min, anchor_projection + t_max


def _normalized_point_valid(point: Any) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def validate_svg_bond_geometry_normalized_points(geometry: dict[str, Any], *, require_warped: bool) -> list[str]:
    issues: list[str] = []
    for bond in geometry.get("bonds") if isinstance(geometry.get("bonds"), list) else []:
        bond_index = bond.get("bond_index")
        selected_axis = bond.get("selected_atom_center_axis") if isinstance(bond.get("selected_atom_center_axis"), dict) else {}
        axis_points = selected_axis.get("sampled_points_normalized")
        if not isinstance(axis_points, list) or len(axis_points) < 2:
            issues.append(f"bond_{bond_index}_selected_axis_missing_sampled_points")
        elif not all(_normalized_point_valid(point) for point in axis_points):
            issues.append(f"bond_{bond_index}_selected_axis_sampled_points_out_of_bounds")
        if require_warped:
            warped_points = selected_axis.get("sampled_points_normalized_after_warp")
            if not isinstance(warped_points, list) or len(warped_points) < 2:
                issues.append(f"bond_{bond_index}_selected_axis_missing_warped_sampled_points")
            elif not all(_normalized_point_valid(point) for point in warped_points):
                issues.append(f"bond_{bond_index}_selected_axis_warped_sampled_points_out_of_bounds")
        for line in bond.get("visible_lines") if isinstance(bond.get("visible_lines"), list) else []:
            line_index = line.get("line_index")
            line_points = line.get("sampled_points_normalized")
            if not isinstance(line_points, list) or len(line_points) < 2:
                issues.append(f"bond_{bond_index}_line_{line_index}_missing_sampled_points")
            elif not all(_normalized_point_valid(point) for point in line_points):
                issues.append(f"bond_{bond_index}_line_{line_index}_sampled_points_out_of_bounds")
            if require_warped:
                warped_line_points = line.get("sampled_points_normalized_after_warp")
                if not isinstance(warped_line_points, list) or len(warped_line_points) < 2:
                    issues.append(f"bond_{bond_index}_line_{line_index}_missing_warped_sampled_points")
                elif not all(_normalized_point_valid(point) for point in warped_line_points):
                    issues.append(f"bond_{bond_index}_line_{line_index}_warped_sampled_points_out_of_bounds")
    return issues


def _bbox_samples(bbox: list[Any], *, samples_per_edge: int = 7) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    count = max(2, int(samples_per_edge))
    points: list[tuple[float, float]] = []
    for index in range(count):
        t = index / (count - 1)
        points.append((x1 + (x2 - x1) * t, y1))
        points.append((x1 + (x2 - x1) * t, y2))
        points.append((x1, y1 + (y2 - y1) * t))
        points.append((x2, y1 + (y2 - y1) * t))
    return points


def _warp_displacement_field(
    width: int,
    height: int,
    *,
    seed: int,
    amplitude_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(int(seed))
    width = max(1, int(width))
    height = max(1, int(height))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    max_amplitude = max(0.0, min(float(amplitude_px), min(width, height) * 0.018))
    if max_amplitude <= 0.0:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32), {
            "enabled": False,
            "amplitude_px": 0.0,
        }

    dx = np.zeros((height, width), dtype=np.float32)
    dy = np.zeros((height, width), dtype=np.float32)
    components: list[dict[str, Any]] = []
    for axis_name, target in [("x", dx), ("y", dy)]:
        for _component in range(2):
            period = rng.uniform(min(width, height) * 0.75, min(width, height) * 1.85)
            angle = rng.uniform(0.0, math.pi)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            component_amp = max_amplitude * rng.uniform(0.18, 0.55)
            projection = math.cos(angle) * xx + math.sin(angle) * yy
            target += (component_amp * np.sin((2.0 * math.pi * projection / max(period, 1.0)) + phase)).astype(
                np.float32
            )
            components.append(
                {
                    "axis": axis_name,
                    "period_px": float(period),
                    "angle_rad": float(angle),
                    "phase_rad": float(phase),
                    "amplitude_px": float(component_amp),
                }
            )

    # A broad fold-like brightness ridge is intentionally visual only; it does
    # not define geometry. The geometry above is a small smooth warp, recorded
    # and applied to every coordinate-bearing artifact.
    return dx, dy, {
        "enabled": True,
        "amplitude_px": float(max_amplitude),
        "max_abs_dx_px": float(np.max(np.abs(dx))) if dx.size else 0.0,
        "max_abs_dy_px": float(np.max(np.abs(dy))) if dy.size else 0.0,
        "components": components,
    }


def _sample_displacement(dx: np.ndarray, dy: np.ndarray, x: float, y: float) -> tuple[float, float]:
    height, width = dx.shape
    x = max(0.0, min(float(width - 1), float(x)))
    y = max(0.0, min(float(height - 1), float(y)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    def interp(arr: np.ndarray) -> float:
        return float(
            arr[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + arr[y0, x1] * tx * (1.0 - ty)
            + arr[y1, x0] * (1.0 - tx) * ty
            + arr[y1, x1] * tx * ty
        )

    return interp(dx), interp(dy)


def _warp_point_norm(dx: np.ndarray, dy: np.ndarray, x: float, y: float) -> tuple[float, float]:
    height, width = dx.shape
    px = float(x) * float(width)
    py = float(y) * float(height)
    ddx, ddy = _sample_displacement(dx, dy, px, py)
    return ((px + ddx) / float(width), (py + ddy) / float(height))


def _clamp_bbox_from_points(points: list[tuple[float, float]]) -> list[float]:
    xs = [max(0.0, min(1.0, float(point[0]))) for point in points]
    ys = [max(0.0, min(1.0, float(point[1]))) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_svg_bond_geometry_contract(
    *,
    bonds: list[dict[str, Any]],
    atom_centers: dict[int, np.ndarray],
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]],
    viewbox: tuple[float, float, float, float],
    selected_axes: dict[int, tuple[tuple[np.ndarray, float], dict[str, Any]]],
) -> dict[str, Any]:
    width, height = float(viewbox[2]), float(viewbox[3])
    bond_endpoints = {
        int(bond["bond_index"]): (int(bond["begin_atom_index"]), int(bond["end_atom_index"]))
        for bond in bonds
    }
    bonds: list[dict[str, Any]] = []
    for bond_index in sorted(lines_by_bond):
        lines = lines_by_bond.get(int(bond_index)) or []
        selected = selected_axes.get(int(bond_index))
        selected_policy = selected[1] if selected is not None else {}
        selected_axis_record: dict[str, Any] | None = None
        if selected is not None:
            selected_axis, _selected_policy = selected
            endpoint_indices = bond_endpoints.get(int(bond_index), ())
            endpoint_centers = [atom_centers[index] for index in endpoint_indices if index in atom_centers]
            axis_samples = _axis_segment_samples(
                selected_axis,
                lines,
                atom_center_points=endpoint_centers,
                samples=41,
                clip_box=(0.0, 0.0, width, height),
            )
            if axis_samples:
                selected_axis_record = {
                    "sampled_points_normalized": [[float(x) / width, float(y) / height] for x, y in axis_samples],
                    "line_sampling_policy": (
                        "selected_atom_center_axis_projected_to_bond_endpoint_atom_centers_and_visible_svg_extent_"
                        "clipped_to_svg_viewbox"
                    ),
                }
        line_records = []
        for line_index, line in enumerate(lines):
            samples = _line_samples(line)
            line_records.append(
                {
                    "line_index": int(line_index),
                    "svg_units": [float(value) for value in line],
                    "normalized": [
                        float(line[0]) / width,
                        float(line[1]) / height,
                        float(line[2]) / width,
                        float(line[3]) / height,
                    ],
                    "sampled_points_normalized": [[float(x) / width, float(y) / height] for x, y in samples],
                }
            )
        bonds.append(
            {
                "bond_index": int(bond_index),
                "selected_axis_policy": selected_policy,
                "selected_atom_center_axis": selected_axis_record,
                "visible_line_count": int(len(line_records)),
                "visible_lines": line_records,
            }
        )
    return {
        "schema_version": "markush_svg_bond_geometry_v1",
        "coordinate_space": "normalized_image_after_document_context",
        "viewbox": [float(value) for value in viewbox],
        "bond_count": int(len(bonds)),
        "bonds": bonds,
        "line_sampling_policy": "endpoints_plus_15_uniform_interior_samples_per_visible_svg_line",
    }


def transform_svg_bond_geometry_contract(
    geometry: dict[str, Any],
    dx: np.ndarray,
    dy: np.ndarray,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(geometry))
    for bond in updated.get("bonds") if isinstance(updated.get("bonds"), list) else []:
        selected_axis = bond.get("selected_atom_center_axis") if isinstance(bond.get("selected_atom_center_axis"), dict) else {}
        points = selected_axis.get("sampled_points_normalized") if isinstance(selected_axis, dict) else None
        if isinstance(points, list):
            warped_points = [_warp_point_norm(dx, dy, float(point[0]), float(point[1])) for point in points]
            selected_axis["sampled_points_normalized_after_warp"] = [[float(x), float(y)] for x, y in warped_points]
            selected_axis["warped_bbox_normalized"] = _clamp_bbox_from_points(warped_points)
            selected_axis["nonlinear_warp_synchronized"] = True
        for line in bond.get("visible_lines") if isinstance(bond.get("visible_lines"), list) else []:
            points = line.get("sampled_points_normalized")
            if not isinstance(points, list):
                continue
            warped_points = [_warp_point_norm(dx, dy, float(point[0]), float(point[1])) for point in points]
            line["sampled_points_normalized_after_warp"] = [[float(x), float(y)] for x, y in warped_points]
            line["warped_bbox_normalized"] = _clamp_bbox_from_points(warped_points)
            if len(warped_points) >= 2:
                line["warped_endpoint_line_normalized"] = [
                    float(warped_points[0][0]),
                    float(warped_points[0][1]),
                    float(warped_points[-1][0]),
                    float(warped_points[-1][1]),
                ]
    updated["coordinate_space"] = "normalized_image_after_document_context_and_nonlinear_warp"
    updated["nonlinear_warp_synchronized"] = True
    return updated


def transform_svg_bond_geometry_for_document_context(
    geometry: dict[str, Any],
    background_realism: dict[str, Any],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(geometry))
    padding = background_realism.get("padding_px") if isinstance(background_realism.get("padding_px"), list) else [0, 0, 0, 0]
    old_size = (
        background_realism.get("old_image_size")
        if isinstance(background_realism.get("old_image_size"), list)
        else [1, 1]
    )
    new_size = (
        background_realism.get("new_image_size")
        if isinstance(background_realism.get("new_image_size"), list)
        else old_size
    )
    pad_left, pad_top = float(padding[0] or 0), float(padding[1] or 0)
    old_w, old_h = max(1.0, float(old_size[0] or 1)), max(1.0, float(old_size[1] or 1))
    new_w, new_h = max(1.0, float(new_size[0] or old_w)), max(1.0, float(new_size[1] or old_h))

    def xform(point: list[Any]) -> list[float]:
        return [
            (float(point[0]) * old_w + pad_left) / new_w,
            (float(point[1]) * old_h + pad_top) / new_h,
        ]

    for bond in updated.get("bonds") if isinstance(updated.get("bonds"), list) else []:
        selected_axis = bond.get("selected_atom_center_axis") if isinstance(bond.get("selected_atom_center_axis"), dict) else {}
        points = selected_axis.get("sampled_points_normalized") if isinstance(selected_axis, dict) else None
        if isinstance(points, list):
            transformed = [xform(point) for point in points]
            selected_axis["sampled_points_normalized"] = transformed
            selected_axis["document_context_synchronized"] = True
            if transformed:
                selected_axis["bbox_normalized"] = _clamp_bbox_from_points([(point[0], point[1]) for point in transformed])
        for line in bond.get("visible_lines") if isinstance(bond.get("visible_lines"), list) else []:
            points = line.get("sampled_points_normalized")
            if isinstance(points, list):
                transformed = [xform(point) for point in points]
                line["sampled_points_normalized"] = transformed
                if transformed:
                    line["warped_bbox_normalized"] = _clamp_bbox_from_points([(point[0], point[1]) for point in transformed])
            endpoints = line.get("normalized")
            if isinstance(endpoints, list) and len(endpoints) == 4:
                left = xform([endpoints[0], endpoints[1]])
                right = xform([endpoints[2], endpoints[3]])
                line["normalized"] = [left[0], left[1], right[0], right[1]]
    updated["coordinate_space"] = "normalized_image_after_document_context"
    updated["document_context_synchronized"] = True
    updated["document_context_padding_px"] = [int(value or 0) for value in padding[:4]]
    updated["document_context_old_image_size"] = [float(old_w), float(old_h)]
    updated["document_context_new_image_size"] = [float(new_w), float(new_h)]
    return updated


def _geometry_metric_point(point: tuple[float, float], geometry: dict[str, Any]) -> np.ndarray:
    viewbox = geometry.get("viewbox") if isinstance(geometry.get("viewbox"), list) else [0.0, 0.0, 1.0, 1.0]
    width = max(1.0, float(viewbox[2] if len(viewbox) > 2 else 1.0))
    height = max(1.0, float(viewbox[3] if len(viewbox) > 3 else 1.0))
    old_size = (
        geometry.get("document_context_old_image_size")
        if isinstance(geometry.get("document_context_old_image_size"), list)
        else [width, height]
    )
    new_size = (
        geometry.get("document_context_new_image_size")
        if isinstance(geometry.get("document_context_new_image_size"), list)
        else old_size
    )
    padding = (
        geometry.get("document_context_padding_px")
        if isinstance(geometry.get("document_context_padding_px"), list)
        else [0.0, 0.0, 0.0, 0.0]
    )
    old_w = max(1.0, float(old_size[0] if len(old_size) > 0 else width))
    old_h = max(1.0, float(old_size[1] if len(old_size) > 1 else height))
    new_w = max(1.0, float(new_size[0] if len(new_size) > 0 else old_w))
    new_h = max(1.0, float(new_size[1] if len(new_size) > 1 else old_h))
    pad_left = float(padding[0] if len(padding) > 0 else 0.0)
    pad_top = float(padding[1] if len(padding) > 1 else 0.0)
    x = ((float(point[0]) * new_w) - pad_left) / old_w * width
    y = ((float(point[1]) * new_h) - pad_top) / old_h * height
    return np.asarray([x, y], dtype=np.float64)


def _distance_to_polyline(point: np.ndarray, polyline: list[np.ndarray]) -> float | None:
    if len(polyline) < 2:
        return None
    best: float | None = None
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            distance = float(np.linalg.norm(point - start))
        else:
            t = max(0.0, min(1.0, float(np.dot(point - start, segment) / denom)))
            projection = start + segment * t
            distance = float(np.linalg.norm(point - projection))
        best = distance if best is None else min(best, distance)
    return best


def audit_formal_nonlinear_pose_preservation(
    *,
    atom_coords: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    svg_bond_geometry: dict[str, Any],
    max_rmse: float,
    max_line_abs_p95: float,
    max_line_abs_max: float,
    max_intersection_anchor_rmse: float,
    max_intersection_anchor_abs_max: float,
    min_intersection_anchors: int,
) -> dict[str, Any]:
    atom_by_index = {int(atom["atom_index"]): atom for atom in atom_coords if "atom_index" in atom}
    geometry_by_bond = {
        int(bond.get("bond_index")): bond
        for bond in svg_bond_geometry.get("bonds", [])
        if isinstance(bond, dict) and bond.get("bond_index") is not None
    }
    graph_incident: dict[int, list[int]] = defaultdict(list)
    line_residuals: list[float] = []
    missing_axis_count = 0
    invalid_axis_count = 0
    skipped_no_visible_axis_bond_count = 0

    for bond in bonds:
        bond_index = int(bond.get("bond_index"))
        begin = int(bond.get("begin_atom_index"))
        end = int(bond.get("end_atom_index"))
        graph_incident[begin].append(bond_index)
        graph_incident[end].append(bond_index)
        geometry_bond = geometry_by_bond.get(bond_index) or {}
        selected_axis = (
            geometry_bond.get("selected_atom_center_axis")
            if isinstance(geometry_bond.get("selected_atom_center_axis"), dict)
            else None
        )
        if selected_axis is None:
            skipped_no_visible_axis_bond_count += 1
            continue
        points = selected_axis.get("sampled_points_normalized_after_warp")
        if not isinstance(points, list):
            missing_axis_count += 1
            continue
        if len(points) < 2 or not all(_normalized_point_valid(point) for point in points):
            invalid_axis_count += 1
            continue
        polyline = [_geometry_metric_point((float(point[0]), float(point[1])), svg_bond_geometry) for point in points]
        for atom_index in [begin, end]:
            atom = atom_by_index.get(atom_index)
            if not isinstance(atom, dict):
                continue
            point = _geometry_metric_point((float(atom["x"]), float(atom["y"])), svg_bond_geometry)
            distance = _distance_to_polyline(point, polyline)
            if distance is not None:
                line_residuals.append(float(distance))

    anchor_residuals: list[float] = []
    for atom_index, incident_bonds in graph_incident.items():
        if len(incident_bonds) < 2:
            continue
        atom = atom_by_index.get(atom_index)
        if not isinstance(atom, dict):
            continue
        point = _geometry_metric_point((float(atom["x"]), float(atom["y"])), svg_bond_geometry)
        distances: list[float] = []
        for bond_index in incident_bonds:
            geometry_bond = geometry_by_bond.get(int(bond_index)) or {}
            selected_axis = (
                geometry_bond.get("selected_atom_center_axis")
                if isinstance(geometry_bond.get("selected_atom_center_axis"), dict)
                else None
            )
            if selected_axis is None:
                continue
            points = selected_axis.get("sampled_points_normalized_after_warp")
            if not isinstance(points, list):
                continue
            if len(points) < 2 or not all(_normalized_point_valid(item) for item in points):
                continue
            polyline = [_geometry_metric_point((float(item[0]), float(item[1])), svg_bond_geometry) for item in points]
            distance = _distance_to_polyline(point, polyline)
            if distance is not None:
                distances.append(float(distance))
        if len(distances) >= 2:
            anchor_residuals.append(float(np.sqrt(np.mean(np.asarray(distances, dtype=np.float64) ** 2))))

    line_array = np.asarray(line_residuals, dtype=np.float64)
    anchor_array = np.asarray(anchor_residuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(line_array**2))) if line_array.size else None
    line_abs_p95 = float(np.quantile(line_array, 0.95)) if line_array.size else None
    line_abs_max = float(np.max(line_array)) if line_array.size else None
    anchor_rmse = float(np.sqrt(np.mean(anchor_array**2))) if anchor_array.size else None
    anchor_abs_max = float(np.max(anchor_array)) if anchor_array.size else None
    blockers: list[str] = []
    if missing_axis_count:
        blockers.append(f"missing warped selected atom-center axis for {missing_axis_count} bonds")
    if invalid_axis_count:
        blockers.append(f"invalid warped selected atom-center axis sampled points for {invalid_axis_count} bonds")
    if rmse is None or rmse > float(max_rmse):
        blockers.append(f"nonlinear line RMSE {rmse} exceeds maximum {float(max_rmse):.3f}")
    if line_abs_p95 is None or line_abs_p95 > float(max_line_abs_p95):
        blockers.append(f"nonlinear line residual p95 {line_abs_p95} exceeds maximum {float(max_line_abs_p95):.3f}")
    if line_abs_max is None or line_abs_max > float(max_line_abs_max):
        blockers.append(f"nonlinear line residual max {line_abs_max} exceeds maximum {float(max_line_abs_max):.3f}")
    if anchor_rmse is None or anchor_rmse > float(max_intersection_anchor_rmse):
        blockers.append(f"nonlinear intersection anchor RMSE {anchor_rmse} exceeds maximum {float(max_intersection_anchor_rmse):.3f}")
    if anchor_abs_max is None or anchor_abs_max > float(max_intersection_anchor_abs_max):
        blockers.append(
            f"nonlinear intersection anchor max residual {anchor_abs_max} exceeds maximum {float(max_intersection_anchor_abs_max):.3f}"
        )
    if len(anchor_residuals) < int(min_intersection_anchors):
        blockers.append(
            f"nonlinear intersection anchor count {len(anchor_residuals)} below minimum {int(min_intersection_anchors)}"
        )
    return {
        "schema_version": "markush_nonlinear_pose_preservation_v1",
        "policy": FORMAL_NONLINEAR_WARP_POLICY,
        "passed": not blockers,
        "blockers": blockers,
        "line_constraint_rmse_svg_units": rmse,
        "line_constraint_rmse_threshold": float(max_rmse),
        "line_constraint_abs_p95_svg_units": line_abs_p95,
        "line_constraint_abs_p95_threshold": float(max_line_abs_p95),
        "line_constraint_abs_max_svg_units": line_abs_max,
        "line_constraint_abs_max_threshold": float(max_line_abs_max),
        "line_constraint_fit_equation_count": int(len(line_residuals)),
        "line_constraint_bond_count": int(len(bonds) - missing_axis_count),
        "skipped_no_visible_axis_bond_count": int(skipped_no_visible_axis_bond_count),
        "intersection_anchor_count": int(len(anchor_residuals)),
        "intersection_anchor_rmse_svg_units": anchor_rmse,
        "intersection_anchor_rmse_threshold": float(max_intersection_anchor_rmse),
        "intersection_anchor_abs_max_svg_units": anchor_abs_max,
        "intersection_anchor_abs_max_threshold": float(max_intersection_anchor_abs_max),
        "metric_space": "svg_units_after_document_context_scale",
        "line_residual_source": "atom_centers_to_warped_selected_svg_atom_center_axis_polylines",
        "intersection_anchor_source": "multi_incident_atom_center_distances_to_warped_incident_axis_polylines",
    }


def apply_formal_nonlinear_document_warp(
    image_path: Path,
    atom_coords: list[dict[str, Any]],
    ocr_cells: list[dict[str, Any]],
    svg_bond_geometry: dict[str, Any],
    *,
    seed: int,
    enabled: bool,
    amplitude_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if not enabled:
        contract = {
            "schema_version": NONLINEAR_WARP_SCHEMA_VERSION,
            "enabled": False,
            "operations": [],
            "image_atom_ocr_svg_bond_geometry_synchronized": False,
        }
        return atom_coords, ocr_cells, svg_bond_geometry, contract

    dx, dy, field = _warp_displacement_field(width, height, seed=int(seed), amplitude_px=float(amplitude_px))
    if not field.get("enabled"):
        contract = {
            "schema_version": NONLINEAR_WARP_SCHEMA_VERSION,
            "enabled": False,
            "operations": [],
            "image_atom_ocr_svg_bond_geometry_synchronized": False,
        }
        return atom_coords, ocr_cells, svg_bond_geometry, contract

    from PIL import ImageFilter

    arr = np.asarray(image)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # Backward resampling: destination pixels sample from source coordinates.
    src_x = np.clip(xx - dx, 0, width - 1)
    src_y = np.clip(yy - dy, 0, height - 1)
    try:
        from scipy import ndimage  # type: ignore

        warped_channels = [
            ndimage.map_coordinates(arr[:, :, channel], [src_y, src_x], order=1, mode="nearest")
            for channel in range(arr.shape[2])
        ]
        warped = np.stack(warped_channels, axis=2).astype("uint8")
        image = Image.fromarray(warped, mode="RGB")
        sampler = "scipy.ndimage.map_coordinates_order1"
    except Exception:
        # Fallback keeps the transform auditable in minimal environments. The
        # downstream warped SVG-polyline pose gate still decides acceptance.
        mesh = []
        grid = 16
        for y0 in range(0, height, grid):
            for x0 in range(0, width, grid):
                x1 = min(width, x0 + grid)
                y1 = min(height, y0 + grid)
                cx = (x0 + x1 - 1) * 0.5
                cy = (y0 + y1 - 1) * 0.5
                ddx, ddy = _sample_displacement(dx, dy, cx, cy)
                mesh.append(((x0, y0, x1, y1), (x0 - ddx, y0 - ddy, x1 - ddx, y0 - ddy, x1 - ddx, y1 - ddy, x0 - ddx, y1 - ddy)))
        image = image.transform((width, height), Image.Transform.MESH, mesh, resample=Image.Resampling.BILINEAR)
        sampler = "pil_mesh_grid16_bilinear"

    rng = random.Random(int(seed) ^ 0x5EED)
    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.04, 0.18)))
    image.save(image_path)

    updated_atoms: list[dict[str, Any]] = []
    atom_outside = 0
    for atom in atom_coords:
        updated = dict(atom)
        x, y = _warp_point_norm(dx, dy, float(atom["x"]), float(atom["y"]))
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            atom_outside += 1
        updated["x"] = max(0.0, min(1.0, float(x)))
        updated["y"] = max(0.0, min(1.0, float(y)))
        updated_atoms.append(updated)

    updated_cells: list[dict[str, Any]] = []
    invalid_bbox = 0
    max_bbox_expansion = 0.0
    for cell in ocr_cells:
        updated = dict(cell)
        bbox = cell.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            before_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            warped_bbox = _clamp_bbox_from_points([_warp_point_norm(dx, dy, x, y) for x, y in _bbox_samples(bbox)])
            after_area = max(0.0, warped_bbox[2] - warped_bbox[0]) * max(0.0, warped_bbox[3] - warped_bbox[1])
            if warped_bbox[0] > warped_bbox[2] or warped_bbox[1] > warped_bbox[3]:
                invalid_bbox += 1
            if before_area > 0:
                max_bbox_expansion = max(max_bbox_expansion, after_area / before_area)
            updated["bbox"] = warped_bbox
            updated["bbox_transform_policy"] = "nonlinear_warp_edge_sampling_envelope"
        updated_cells.append(updated)

    updated_geometry = transform_svg_bond_geometry_contract(svg_bond_geometry, dx, dy)
    arr_after = np.asarray(image.convert("L"))
    dark_ratio = float((arr_after < 245).mean())
    ink_ratio = float((arr_after < 220).mean())
    contract = {
        "schema_version": NONLINEAR_WARP_SCHEMA_VERSION,
        "enabled": True,
        "policy": FORMAL_NONLINEAR_WARP_POLICY,
        "formal_training_allowed": True,
        "operations": [
            "smooth_low_amplitude_document_warp",
            "synchronized_atom_ocr_svg_bond_geometry_warp",
            "warped_svg_polyline_pose_preservation_gate_required",
        ],
        "seed": int(seed),
        "field": field,
        "sampler": sampler,
        "image_size": [int(width), int(height)],
        "coordinate_mutation_policy": (
            "formal_synchronized_image_atom_ocr_bbox_svg_bond_polyline_warp_with_pose_preservation_gate"
        ),
        "image_synchronized": True,
        "atom_coordinates_synchronized": True,
        "ocr_boxes_synchronized": True,
        "svg_bond_axis_line_anchors_synchronized": True,
        "bbox_transform_policy": "sample_bbox_edges_then_take_axis_aligned_envelope",
        "bond_line_transform_policy": "sample_visible_svg_line_points_then_store_warped_polyline_and_envelope",
        "atom_outside_count": int(atom_outside),
        "invalid_bbox_count": int(invalid_bbox),
        "max_ocr_bbox_area_expansion_ratio": float(max_bbox_expansion),
        "dark_pixel_ratio": dark_ratio,
        "ink_pixel_ratio": ink_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
        "limitations": [
            "Rows are formal-capable only when nonlinear_pose_preservation.passed=true.",
            "Still requires accepted-candidate, substitution-anchor, visual, source-leak, router, model-scale, and readiness gates.",
        ],
    }
    return updated_atoms, updated_cells, updated_geometry, contract


def path_bbox_from_element(elem: ET.Element) -> tuple[float, float, float, float] | None:
    paths = svg_paths_from_element(elem)
    if not paths:
        return None
    return paths_bbox(paths)


def svg_paths_from_element(elem: ET.Element) -> list[Any]:
    from svgpathtools import svgstr2paths

    paths: list[Any] = []
    for child in elem.iter():
        if len(list(child)) != 0:
            continue
        try:
            child_paths, _ = svgstr2paths(ET.tostring(child, encoding="unicode"))
        except Exception:
            continue
        paths.extend(path for path in child_paths if path)
    return paths


def paths_bbox(paths: list[Any]) -> tuple[float, float, float, float]:
    if not paths:
        raise ValueError("empty SVG path list")
    first = paths[0].bbox()
    x_min, x_max, y_min, y_max = first[0], first[1], first[2], first[3]
    for item in paths[1:]:
        item_x_min, item_x_max, item_y_min, item_y_max = item.bbox()
        x_min = min(x_min, item_x_min)
        x_max = max(x_max, item_x_max)
        y_min = min(y_min, item_y_min)
        y_max = max(y_max, item_y_max)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def rdkit_visible_atom_label(atom: Any, dummy_labels: dict[int, str]) -> str:
    atom_index = int(atom.GetIdx())
    cx_label = normalize_label(dummy_labels.get(atom_index, ""))
    if cx_label:
        return cx_label
    for prop_name in ["atomLabel", "dummyLabel", "molFileAlias"]:
        if atom.HasProp(prop_name):
            label = normalize_label(atom.GetProp(prop_name))
            if label:
                return label
    return normalize_label(atom.GetSymbol())


def parse_cdk_ocr_cells(
    cxsmiles: str,
    mol_path: Path,
    svg_path: Path,
) -> list[dict[str, Any]]:
    # Adapted from MarkushGenerator's get_boxes/get_cells pipeline. It parses
    # CDK SVG paths with svgpathtools, instead of treating path command numbers
    # as alternating x/y points.
    from lxml import etree as LET
    from rdkit import Chem
    import svgpathtools
    from svgpathtools import svgstr2paths

    tree = LET.parse(str(svg_path))
    root = tree.getroot()
    mol_elem = root.find("*/g[@class='mol']", namespaces=root.nsmap)
    if mol_elem is None:
        raise ValueError("CDK SVG missing molecule group")

    atom_boxes: dict[str, tuple[float, float, float, float]] = {}
    smt_boxes: dict[int, tuple[float, float, float, float]] = {}
    none_counter = 0

    for child_index, elem in enumerate(mol_elem, start=1):
        svg_class = elem.get("class")
        elem_id = elem.get("id") or str(child_index)
        if svg_class not in {"atom", "bond", None}:
            continue
        paths: list[Any] = []
        attribs: list[dict[str, str]] = []
        for subelem in elem.iter():
            if len(subelem) != 0:
                continue
            line = LET.tostring(subelem).decode().strip()
            line = re.sub(r"^\s*(<)svg:(\w+)\s[\w\:=\"\/\.]+\s(.*)$", r"\1\2 \3", line)
            try:
                sub_paths, sub_attribs = svgstr2paths(line)
            except Exception:
                continue
            paths.extend(path for path in sub_paths if path)
            attribs.extend(sub_attribs)

        if not paths:
            if svg_class is None:
                none_counter = 0
            continue

        if svg_class == "atom":
            atom_boxes[elem_id] = paths_bbox(paths)
        elif svg_class is None:
            if any(isinstance(part, svgpathtools.path.Arc) for path in paths for part in path):
                continue
            none_counter += 1
            if attribs and "stroke-width" in attribs[0]:
                none_counter += 1
            if none_counter % 3 == 0:
                smt_boxes[none_counter // 3] = paths_bbox(paths)

    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(cxsmiles, parser_params)
    if molecule is None:
        raise ValueError("RDKit could not parse CXSMILES for OCR cell extraction")
    dummy_labels = cxsmiles_dummy_labels(cxsmiles)

    cells: list[dict[str, Any]] = []
    factor = 1.0 / 289.0
    for atom in molecule.GetAtoms():
        rdkit_index = int(atom.GetIdx())
        cx_label = normalize_label(dummy_labels.get(rdkit_index, ""))
        if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0 and not atom.HasProp("atomLabel") and not cx_label:
            continue
        text = rdkit_visible_atom_label(atom, dummy_labels)
        if atom.GetFormalCharge() != 0:
            text += "-" if atom.GetFormalCharge() < 0 else "+"
            text += str(abs(atom.GetFormalCharge()))
        atom_id = f"mol1atm{rdkit_index + 1}"
        if atom_id not in atom_boxes:
            return []
        box = [float(value) * factor for value in atom_boxes[atom_id]]
        cells.append(
            {
                "bbox": box,
                "text": text,
                "source": "markushgenerator_svgpathtools_atom_box",
                "atom_index": int(rdkit_index),
                "atom_index_space": "rdkit_cxsmiles_svg_atom_id",
            }
        )

    smt_texts: dict[int, str] = {}
    smt_index = 1
    for line in mol_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SRU" not in line:
            continue
        parts = [part for part in line.split(" ") if part]
        label = ""
        if "CONNECT=HT" in parts:
            for field in parts:
                if field.startswith("LABEL="):
                    label = field[6:]
                    break
        elif len(parts) > 8 and parts[8].startswith("LABEL="):
            label = parts[8][6:]
        if label:
            smt_texts[smt_index] = label
        smt_index += 1

    for index in sorted(smt_boxes):
        if index not in smt_texts:
            continue
        box = [float(value) * factor for value in smt_boxes[index]]
        cells.append(
            {
                "bbox": box,
                "text": smt_texts[index],
                "source": "markushgenerator_svgpathtools_smt_box",
                "atom_index_space": "svg_sgroup_text_unindexed",
            }
        )

    valid_cells = []
    for cell in cells:
        bbox = cell.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0 and (x2 - x1) > 0 and (y2 - y1) > 0:
            valid_cells.append(cell)
    return valid_cells


def _element_atomic_number(token: str) -> int:
    text = str(token)
    if text not in CHEMICAL_SYMBOLS:
        return -1
    try:
        from rdkit import Chem

        return int(Chem.Atom(text).GetAtomicNum())
    except Exception:
        return -1


def atomic_number_for_token(token: str) -> int:
    text = normalize_label(str(token))
    if text in {"*", "R#"}:
        return 0
    atomic_number = _element_atomic_number(text)
    if atomic_number >= 0:
        return atomic_number
    category = markush_label_category(text)
    if category in {"markush_variable_label", "fixed_substituent_abbreviation", "nonvariable_pseudo_label"}:
        return 0
    return -1


def cdk_atomic_number_for_atom(atom: dict[str, Any]) -> int:
    token = normalize_label(str(atom.get("token") or ""))
    if _cdk_atom_isotope(atom) > 0 and token not in {"H", "[H]"}:
        return 0
    atomic_number = atomic_number_for_token(token)
    if atomic_number >= 0:
        return atomic_number
    return -1


def training_atom_token_for_cdk_atom(atom: dict[str, Any]) -> str:
    token = str(atom.get("token") or "")
    label = _cdk_atom_label(atom)
    if token in {"*", "R#"}:
        return "*"
    if label and is_markush_label(label):
        return "*"
    return token


def _rdkit_atom_label(atom: Any) -> str:
    for prop_name in ["dummyLabel", "atomLabel", "molFileAlias"]:
        if atom.HasProp(prop_name):
            label = normalize_label(atom.GetProp(prop_name))
            if label and label != "*":
                return label
    symbol = normalize_label(atom.GetSymbol())
    return symbol if symbol and symbol != "*" else ""


def _cdk_atom_label(atom: dict[str, Any]) -> str:
    token = normalize_label(str(atom.get("token") or ""))
    if token and token not in {"*", "R#"} and _cdk_atom_isotope(atom) > 0:
        return token
    if (
        token
        and token not in {"*", "R#"}
        and _element_atomic_number(token) < 0
        and markush_label_category(token) in {"markush_variable_label", "fixed_substituent_abbreviation", "nonvariable_pseudo_label"}
    ):
        return token
    props = atom.get("v3000_props") if isinstance(atom.get("v3000_props"), dict) else {}
    for key in ["LABEL", "ATOMLABEL", "ALIAS"]:
        label = normalize_label(str(props.get(key) or ""))
        if label and label != "*":
            return label
    return ""


def cdk_descriptor_for_alignment(atom: dict[str, Any]) -> dict[str, Any]:
    token = str(atom.get("token") or "")
    label = _cdk_atom_label(atom)
    atomic_number = int(cdk_atomic_number_for_atom(atom))
    return {
        "index": int(atom.get("atom_index")),
        "atomic_number": int(atomic_number),
        "isotope": int(_cdk_atom_isotope(atom)),
        "label": label,
        "token": token,
    }


def _cdk_atom_isotope(atom: dict[str, Any]) -> int:
    props = atom.get("v3000_props") if isinstance(atom.get("v3000_props"), dict) else {}
    try:
        return int(str(props.get("MASS") or "0").strip())
    except ValueError:
        return 0


def _alignment_edge_set(bonds: list[dict[str, Any]], mapping: dict[int, int]) -> set[tuple[int, int]]:
    return {
        tuple(
            sorted(
                (
                    int(mapping[int(bond["begin_atom_index"])]),
                    int(mapping[int(bond["end_atom_index"])]),
                )
            )
        )
        for bond in bonds
    }


def _try_validated_alignment(
    *,
    candidate_mapping: dict[int, int],
    cdk_descriptors: list[dict[str, Any]],
    rdkit_descriptors: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    rdkit_edges: set[tuple[int, int]],
) -> dict[int, int] | None:
    if set(candidate_mapping) != set(range(len(cdk_descriptors))):
        return None
    if set(candidate_mapping.values()) != set(range(len(rdkit_descriptors))):
        return None
    for cdk_index, rdkit_index in candidate_mapping.items():
        cdk = cdk_descriptors[int(cdk_index)]
        rdkit = rdkit_descriptors[int(rdkit_index)]
        if int(cdk["atomic_number"]) != int(rdkit["atomic_number"]):
            return None
        if int(cdk["atomic_number"]) == 0:
            cdk_isotope = int(cdk.get("isotope") or 0)
            rdkit_isotope = int(rdkit.get("isotope") or 0)
            if cdk_isotope and rdkit_isotope and cdk_isotope != rdkit_isotope:
                return None
            cdk_label = normalize_label(str(cdk.get("label") or ""))
            rdkit_label = normalize_label(str(rdkit.get("label") or ""))
            if cdk_label and rdkit_label and cdk_label != rdkit_label:
                return None
    if _alignment_edge_set(bonds, candidate_mapping) != rdkit_edges:
        return None
    return dict(candidate_mapping)


def _graph_isomorphism_alignment(
    *,
    cdk_descriptors: list[dict[str, Any]],
    rdkit_descriptors: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    rdkit_edges: set[tuple[int, int]],
) -> tuple[dict[int, int] | None, int]:
    cdk_neighbors: dict[int, set[int]] = defaultdict(set)
    rdkit_neighbors: dict[int, set[int]] = defaultdict(set)
    for bond in bonds:
        begin = int(bond["begin_atom_index"])
        end = int(bond["end_atom_index"])
        cdk_neighbors[begin].add(end)
        cdk_neighbors[end].add(begin)
    for begin, end in rdkit_edges:
        rdkit_neighbors[begin].add(end)
        rdkit_neighbors[end].add(begin)

    candidates: dict[int, list[int]] = {}
    for cdk in cdk_descriptors:
        cdk_index = int(cdk["index"])
        values: list[int] = []
        for rdkit in rdkit_descriptors:
            rdkit_index = int(rdkit["index"])
            if int(cdk["atomic_number"]) != int(rdkit["atomic_number"]):
                continue
            if len(cdk_neighbors.get(cdk_index, set())) != len(rdkit_neighbors.get(rdkit_index, set())):
                continue
            if int(cdk["atomic_number"]) == 0:
                cdk_isotope = int(cdk.get("isotope") or 0)
                rdkit_isotope = int(rdkit.get("isotope") or 0)
                if cdk_isotope and rdkit_isotope and cdk_isotope != rdkit_isotope:
                    continue
                cdk_label = normalize_label(str(cdk.get("label") or ""))
                rdkit_label = normalize_label(str(rdkit.get("label") or ""))
                if cdk_label and rdkit_label and cdk_label != rdkit_label:
                    continue
            values.append(rdkit_index)
        if not values:
            return None, 0
        candidates[cdk_index] = sorted(values, key=lambda index: (index != cdk_index, index))

    order = sorted(
        range(len(cdk_descriptors)),
        key=lambda index: (
            len(candidates[index]),
            -len(cdk_neighbors.get(index, set())),
            abs(index - candidates[index][0]) if candidates[index] else 9999,
        ),
    )
    solutions: list[dict[int, int]] = []

    def backtrack(mapping: dict[int, int], used: set[int]) -> None:
        if len(solutions) > 1:
            return
        if len(mapping) == len(cdk_descriptors):
            if _alignment_edge_set(bonds, mapping) == rdkit_edges:
                solutions.append(dict(mapping))
            return
        cdk_index = next(index for index in order if index not in mapping)
        for rdkit_index in candidates[cdk_index]:
            if rdkit_index in used:
                continue
            ok = True
            for cdk_neighbor in cdk_neighbors.get(cdk_index, set()):
                if cdk_neighbor not in mapping:
                    continue
                if tuple(sorted((rdkit_index, mapping[cdk_neighbor]))) not in rdkit_edges:
                    ok = False
                    break
            if not ok:
                continue
            mapping[cdk_index] = rdkit_index
            used.add(rdkit_index)
            backtrack(mapping, used)
            used.remove(rdkit_index)
            del mapping[cdk_index]

    backtrack({}, set())
    return (solutions[0] if len(solutions) == 1 else None), len(solutions)


def build_cdk_to_rdkit_atom_index_alignment(
    cxsmiles: str,
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
) -> dict[str, Any]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(str(cxsmiles or ""), parser_params)
    if molecule is None:
        raise ValueError("RDKit could not parse CXSMILES for atom-index alignment")

    rdkit_descriptors = [
        {
            "index": int(atom.GetIdx()),
            "atomic_number": int(atom.GetAtomicNum()),
            "isotope": int(atom.GetIsotope()),
            "label": _rdkit_atom_label(atom),
            "symbol": str(atom.GetSymbol()),
        }
        for atom in molecule.GetAtoms()
    ]
    cdk_descriptors = [cdk_descriptor_for_alignment(atom) for atom in atoms]
    rdkit_atomic_numbers = [int(atom["atomic_number"]) for atom in rdkit_descriptors]
    cdk_atomic_numbers = [int(atom["atomic_number"]) for atom in cdk_descriptors]
    if len(rdkit_descriptors) != len(cdk_descriptors):
        raise ValueError(
            "CDK/RDKit atom-index alignment failed: atom count mismatch "
            f"cdk={len(cdk_atomic_numbers)} rdkit={len(rdkit_atomic_numbers)}"
        )
    unknown_cdk_tokens = [
        str(atom.get("token") or "")
        for atom in cdk_descriptors
        if int(atom.get("atomic_number") if atom.get("atomic_number") is not None else -1) < 0
    ]
    if unknown_cdk_tokens:
        raise ValueError(
            "CDK/RDKit atom-index alignment failed: unknown CDK atom tokens "
            f"{sorted(set(unknown_cdk_tokens))}"
        )

    rdkit_edges = {
        tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))))
        for bond in molecule.GetBonds()
    }
    identity_mapping = {index: index for index in range(len(cdk_descriptors))}
    cdk_to_rdkit = _try_validated_alignment(
        candidate_mapping=identity_mapping,
        cdk_descriptors=cdk_descriptors,
        rdkit_descriptors=rdkit_descriptors,
        bonds=bonds,
        rdkit_edges=rdkit_edges,
    )
    policy = "cdk_v3000_identity_dummy_pseudo_label_isotope_verified_to_rdkit_cxsmiles"
    graph_solution_count: int | None = None

    if cdk_to_rdkit is None:
        rdkit_non_h = [index for index, atomic_number in enumerate(rdkit_atomic_numbers) if atomic_number != 1]
        cdk_non_h = [index for index, atomic_number in enumerate(cdk_atomic_numbers) if atomic_number != 1]
        rdkit_h = [index for index, atomic_number in enumerate(rdkit_atomic_numbers) if atomic_number == 1]
        cdk_h = [index for index, atomic_number in enumerate(cdk_atomic_numbers) if atomic_number == 1]
        if len(rdkit_non_h) == len(cdk_non_h) and len(rdkit_h) == len(cdk_h):
            non_h_then_h = {cdk: rdkit for cdk, rdkit in zip(cdk_non_h, rdkit_non_h)}
            non_h_then_h.update({cdk: rdkit for cdk, rdkit in zip(cdk_h, rdkit_h)})
            cdk_to_rdkit = _try_validated_alignment(
                candidate_mapping=non_h_then_h,
                cdk_descriptors=cdk_descriptors,
                rdkit_descriptors=rdkit_descriptors,
                bonds=bonds,
                rdkit_edges=rdkit_edges,
            )
            policy = "cdk_v3000_non_h_then_h_dummy_pseudo_label_isotope_verified_to_rdkit_cxsmiles"

    if cdk_to_rdkit is None:
        graph_mapping, graph_solution_count = _graph_isomorphism_alignment(
            cdk_descriptors=cdk_descriptors,
            rdkit_descriptors=rdkit_descriptors,
            bonds=bonds,
            rdkit_edges=rdkit_edges,
        )
        if graph_mapping is not None:
            cdk_to_rdkit = graph_mapping
            policy = "cdk_v3000_graph_isomorphism_dummy_pseudo_label_isotope_verified_to_rdkit_cxsmiles"

    if cdk_to_rdkit is None:
        detail = (
            "not unique" if graph_solution_count and graph_solution_count > 1 else "no validated mapping"
        )
        raise ValueError(f"CDK/RDKit atom-index alignment failed: {detail}")

    changed = any(int(cdk_index) != int(rdkit_index) for cdk_index, rdkit_index in cdk_to_rdkit.items())
    cdk_dummy_count = sum(1 for atom in cdk_descriptors if int(atom["atomic_number"]) == 0)
    rdkit_dummy_count = sum(1 for atom in rdkit_descriptors if int(atom["atomic_number"]) == 0)
    cdk_dummy_labels = {
        str(atom["index"]): str(atom.get("label") or "")
        for atom in cdk_descriptors
        if int(atom["atomic_number"]) == 0 and atom.get("label")
    }
    rdkit_dummy_labels = {
        str(atom["index"]): str(atom.get("label") or "")
        for atom in rdkit_descriptors
        if int(atom["atomic_number"]) == 0 and atom.get("label")
    }
    return {
        "schema_version": "cdk_rdkit_atom_index_alignment_v2",
        "policy": policy,
        "applied": True,
        "changed": bool(changed),
        "validated": True,
        "validation": {
            "atom_count": len(cdk_atomic_numbers),
            "bond_count": len(bonds),
            "atomic_numbers_match_after_remap": True,
            "bond_edge_set_matches_after_remap": True,
            "dummy_pseudo_atom_count_matches": cdk_dummy_count == rdkit_dummy_count,
            "cdk_dummy_pseudo_atom_count": int(cdk_dummy_count),
            "rdkit_dummy_pseudo_atom_count": int(rdkit_dummy_count),
            "dummy_label_or_isotope_constraints_checked": True,
            "graph_solution_count": graph_solution_count,
        },
        "cdk_dummy_labels_by_atom_index": cdk_dummy_labels,
        "rdkit_dummy_labels_by_atom_index": rdkit_dummy_labels,
        "cdk_to_rdkit_atom_index": {str(cdk): int(rdkit) for cdk, rdkit in sorted(cdk_to_rdkit.items())},
    }


def remap_atom_indexed_records_to_rdkit_order(
    records: list[dict[str, Any]],
    cdk_to_rdkit: dict[int, int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if item.get("atom_index") is None:
            remapped.append(item)
            continue
        old_index = int(item["atom_index"])
        if old_index not in cdk_to_rdkit:
            remapped.append(item)
            continue
        item["atom_index"] = int(cdk_to_rdkit[old_index])
        remapped.append(item)
    remapped.sort(
        key=lambda item: (
            item.get("atom_index") is None,
            int(item["atom_index"]) if item.get("atom_index") is not None else 10**9,
        )
    )
    return remapped


def remap_bonds_to_rdkit_atom_order(
    bonds: list[dict[str, Any]],
    cdk_to_rdkit: dict[int, int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for bond in bonds:
        item = dict(bond)
        item["begin_atom_index"] = int(cdk_to_rdkit[int(bond["begin_atom_index"])])
        item["end_atom_index"] = int(cdk_to_rdkit[int(bond["end_atom_index"])])
        remapped.append(item)
    return remapped


def fit_affine(
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]],
) -> tuple[np.ndarray, float, int]:
    atom_by_index = {int(atom["atom_index"]): atom for atom in atoms}
    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    for bond in bonds:
        candidates = lines_by_bond.get(int(bond["bond_index"])) or []
        if not candidates:
            continue
        begin = atom_by_index.get(int(bond["begin_atom_index"]))
        end = atom_by_index.get(int(bond["end_atom_index"]))
        if begin is None or end is None:
            continue
        src.append(((begin["mol_x"] + end["mol_x"]) * 0.5, (begin["mol_y"] + end["mol_y"]) * 0.5))
        centers = [((line[0] + line[2]) * 0.5, (line[1] + line[3]) * 0.5) for line in candidates]
        dst.append((sum(x for x, _ in centers) / len(centers), sum(y for _, y in centers) / len(centers)))
    if len(src) < 3:
        raise ValueError("not enough bond midpoint correspondences")
    src_matrix = np.asarray([[x, y, 1.0] for x, y in src], dtype=np.float64)
    dst_matrix = np.asarray(dst, dtype=np.float64)
    affine, *_ = np.linalg.lstsq(src_matrix, dst_matrix, rcond=None)
    predicted = src_matrix @ affine
    rmse = float(np.sqrt(((predicted - dst_matrix) ** 2).sum(axis=1).mean()))
    return affine, rmse, len(src)


def bond_axis_from_svg_lines_with_policy(
    lines: list[tuple[float, float, float, float]],
) -> tuple[tuple[np.ndarray, float] | None, dict[str, Any]]:
    segments = []
    for x1, y1, x2, y2 in lines:
        vector = np.asarray([x2 - x1, y2 - y1], dtype=np.float64)
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        midpoint = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        segments.append((length, vector / length, midpoint))
    if not segments:
        return None, {"selection": "missing", "segment_count": 0}

    reference = max(segments, key=lambda item: item[0])[1]
    oriented_segments = []
    offsets = []
    for _length, direction, midpoint in segments:
        if float(np.dot(direction, reference)) < 0.0:
            direction = -direction
        oriented_segments.append((_length, direction, midpoint))
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        offsets.append(float(np.dot(normal, midpoint)))

    # CDK often renders double/aromatic bonds as one full centerline plus one
    # shorter offset line. Averaging those two lines shifts atom centers off the
    # true rendered bond axis and creates systematic failures on rings. Only use
    # the longest line when it is clearly longer; otherwise keep the symmetric
    # average/center axis.
    lengths = sorted((float(item[0]) for item in oriented_segments), reverse=True)
    if len(oriented_segments) >= 3 and lengths[-1] > 1e-6 and lengths[0] / lengths[-1] >= 1.35:
        directions = np.asarray([direction for _length, direction, _midpoint in oriented_segments], dtype=np.float64)
        mean_direction = np.mean(directions, axis=0)
        mean_norm = float(np.linalg.norm(mean_direction))
        if mean_norm > 1e-6:
            mean_direction = mean_direction / mean_norm
            direction_parallelism = [abs(float(np.dot(direction, mean_direction))) for direction in directions]
            midpoints = np.asarray([midpoint for _length, _direction, midpoint in oriented_segments], dtype=np.float64)
            centered = midpoints - np.mean(midpoints, axis=0)
            if midpoints.shape[0] >= 3 and float(min(direction_parallelism)) >= 0.92:
                _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
                midpoint_axis_direction = np.asarray(vt[0], dtype=np.float64)
                midpoint_spread = float(singular_values[0]) if len(singular_values) else 0.0
                secondary_spread = float(singular_values[1]) if len(singular_values) > 1 else 0.0
                perpendicular_to_short_strokes = abs(float(np.dot(midpoint_axis_direction, mean_direction))) <= 0.35
                if midpoint_spread > 1e-3 and perpendicular_to_short_strokes:
                    normal = np.asarray([-midpoint_axis_direction[1], midpoint_axis_direction[0]], dtype=np.float64)
                    normal_norm = float(np.linalg.norm(normal))
                    if normal_norm > 1e-6:
                        normal = normal / normal_norm
                        return (normal, float(np.dot(normal, np.mean(midpoints, axis=0)))), {
                            "selection": "midpoint_axis_for_hashed_or_wedged_stereo_bond",
                            "segment_count": int(len(oriented_segments)),
                            "lengths": lengths,
                            "longest_to_shortest_length_ratio": float(lengths[0] / lengths[-1]),
                            "min_segment_direction_parallelism": float(min(direction_parallelism)),
                            "midpoint_axis_dot_segment_direction_abs": abs(
                                float(np.dot(midpoint_axis_direction, mean_direction))
                            ),
                            "midpoint_spread_svg_units": midpoint_spread,
                            "midpoint_secondary_spread_svg_units": secondary_spread,
                            "policy": (
                                "CDK hashed/wedged stereo bonds are rendered as several short near-parallel "
                                "strokes whose stroke direction is not the chemical bond axis. The atom-center "
                                "axis is the fitted line through stroke midpoints; this is used only for tapered "
                                "multi-stroke bonds and is still audited by the unchanged pose gates."
                            ),
                        }
    if len(oriented_segments) == 2 and lengths[1] > 1e-6 and lengths[0] / lengths[1] >= 1.08:
        length, direction, midpoint = max(oriented_segments, key=lambda item: item[0])
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        return (normal, float(np.dot(normal, midpoint))), {
            "selection": "longest_visible_line_for_asymmetric_double_bond",
            "segment_count": int(len(oriented_segments)),
            "lengths": lengths,
            "longest_to_second_length_ratio": float(lengths[0] / lengths[1]),
        }

    directions = [direction for _length, direction, _midpoint in oriented_segments]
    direction = np.mean(np.asarray(directions, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = reference
    else:
        direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    offsets = [float(np.dot(normal, midpoint)) for _length, _direction, midpoint in oriented_segments]
    return (normal, float(np.mean(offsets))), {
        "selection": "mean_axis_for_single_or_symmetric_multiline_bond",
        "segment_count": int(len(oriented_segments)),
        "lengths": lengths,
        "longest_to_second_length_ratio": float(lengths[0] / lengths[1]) if len(lengths) >= 2 and lengths[1] > 0 else None,
    }


def svg_segment_axis(line: tuple[float, float, float, float]) -> tuple[tuple[np.ndarray, float] | None, dict[str, Any]]:
    x1, y1, x2, y2 = line
    vector = np.asarray([x2 - x1, y2 - y1], dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return None, {"length": length}
    direction = vector / length
    midpoint = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    return (normal, float(np.dot(normal, midpoint))), {
        "length": length,
        "midpoint": [float(midpoint[0]), float(midpoint[1])],
        "unit_direction": [float(direction[0]), float(direction[1])],
    }


def score_endpoint_axis_closure(
    *,
    atom_indices: list[int],
    bond_index: int,
    candidate_axis: tuple[np.ndarray, float],
    incident_bonds_by_atom: dict[int, list[int]],
    context_axes_by_bond: dict[int, tuple[np.ndarray, float]],
) -> dict[str, Any]:
    endpoint_scores: list[dict[str, Any]] = []
    all_residuals: list[float] = []
    for atom_index in atom_indices:
        axes = [candidate_axis]
        for incident_bond_index in incident_bonds_by_atom.get(atom_index, []):
            if int(incident_bond_index) == int(bond_index):
                continue
            context_axis = context_axes_by_bond.get(int(incident_bond_index))
            if context_axis is not None:
                axes.append(context_axis)
        if len(axes) < 2:
            continue
        point = least_squares_axis_point(axes)
        if point is None:
            continue
        residuals = [abs(float(np.dot(axis[0], point) - axis[1])) for axis in axes]
        all_residuals.extend(residuals)
        endpoint_scores.append(
            {
                "atom_index": int(atom_index),
                "axis_count": int(len(axes)),
                "max_abs_svg_units": float(max(residuals)),
                "mean_abs_svg_units": float(np.mean(np.asarray(residuals, dtype=np.float64))),
            }
        )
    if not all_residuals:
        return {"endpoint_count": 0, "max_abs_svg_units": None, "mean_abs_svg_units": None, "endpoints": endpoint_scores}
    return {
        "endpoint_count": int(len(endpoint_scores)),
        "max_abs_svg_units": float(max(all_residuals)),
        "mean_abs_svg_units": float(np.mean(np.asarray(all_residuals, dtype=np.float64))),
        "endpoints": endpoint_scores,
    }


def select_svg_bond_axes_with_endpoint_policy(
    bonds: list[dict[str, Any]],
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]],
    *,
    max_endpoint_abs: float = 0.10,
    improvement_ratio: float = 0.50,
    refinement_rounds: int = 2,
) -> dict[int, tuple[tuple[np.ndarray, float], dict[str, Any]]]:
    base_axes_by_bond: dict[int, tuple[np.ndarray, float]] = {}
    policies_by_bond: dict[int, dict[str, Any]] = {}
    incident_bonds_by_atom: dict[int, list[int]] = defaultdict(list)
    endpoints_by_bond: dict[int, tuple[int, int]] = {}

    for bond in bonds:
        bond_index = int(bond["bond_index"])
        begin = int(bond["begin_atom_index"])
        end = int(bond["end_atom_index"])
        endpoints_by_bond[bond_index] = (begin, end)
        incident_bonds_by_atom[begin].append(bond_index)
        incident_bonds_by_atom[end].append(bond_index)
        axis, policy = bond_axis_from_svg_lines_with_policy(lines_by_bond.get(bond_index) or [])
        if axis is None:
            continue
        base_axes_by_bond[bond_index] = axis
        policies_by_bond[bond_index] = policy

    selected_axes_by_bond = dict(base_axes_by_bond)
    selected_policies_by_bond = dict(policies_by_bond)
    for _round_index in range(max(1, int(refinement_rounds))):
        changed = False
        context_axes = dict(selected_axes_by_bond)
        for bond_index, base_axis in sorted(base_axes_by_bond.items()):
            lines = lines_by_bond.get(bond_index) or []
            segment_axes: list[tuple[int, tuple[np.ndarray, float], dict[str, Any]]] = []
            for segment_index, line in enumerate(lines):
                segment_axis, segment_policy = svg_segment_axis(line)
                if segment_axis is not None:
                    segment_axes.append((segment_index, segment_axis, segment_policy))
            if len(segment_axes) != 2:
                continue
            lengths = sorted((float(item[2].get("length") or 0.0) for item in segment_axes), reverse=True)
            if len(lengths) < 2 or lengths[1] <= 1e-6:
                continue
            # The prior longest-line rule already handles visibly asymmetric
            # double bonds. Equal-length double/aromatic bonds need a different
            # test: which visible line closes both endpoint atom anchors with
            # the neighboring rendered bonds?
            if lengths[0] / lengths[1] >= 1.08:
                continue
            endpoint_atoms = list(endpoints_by_bond.get(bond_index, ()))
            base_score = score_endpoint_axis_closure(
                atom_indices=endpoint_atoms,
                bond_index=bond_index,
                candidate_axis=base_axis,
                incident_bonds_by_atom=incident_bonds_by_atom,
                context_axes_by_bond=context_axes,
            )
            base_max = base_score.get("max_abs_svg_units")
            if base_max is None or float(base_max) <= float(max_endpoint_abs):
                continue
            candidate_scores = []
            for segment_index, segment_axis, segment_policy in segment_axes:
                score = score_endpoint_axis_closure(
                    atom_indices=endpoint_atoms,
                    bond_index=bond_index,
                    candidate_axis=segment_axis,
                    incident_bonds_by_atom=incident_bonds_by_atom,
                    context_axes_by_bond=context_axes,
                )
                score_max = score.get("max_abs_svg_units")
                if score_max is None:
                    continue
                candidate_scores.append((float(score_max), segment_index, segment_axis, segment_policy, score))
            if not candidate_scores:
                continue
            candidate_scores.sort(key=lambda item: (item[0], item[1]))
            best_max, best_segment_index, best_axis, best_segment_policy, best_score = candidate_scores[0]
            if best_max <= float(max_endpoint_abs) and best_max < float(base_max) * float(improvement_ratio):
                selected_axes_by_bond[bond_index] = best_axis
                previous_policy = selected_policies_by_bond.get(bond_index, policies_by_bond.get(bond_index, {}))
                selected_policies_by_bond[bond_index] = {
                    "selection": "endpoint_compatible_visible_line_for_symmetric_multiline_bond",
                    "segment_count": int(len(segment_axes)),
                    "selected_segment_index": int(best_segment_index),
                    "lengths": lengths,
                    "longest_to_second_length_ratio": float(lengths[0] / lengths[1]),
                    "previous_selection": previous_policy.get("selection"),
                    "previous_endpoint_closure": base_score,
                    "selected_endpoint_closure": best_score,
                    "selected_segment": best_segment_policy,
                    "policy": (
                        "For equal-length CDK double/aromatic bond lines, choose an actual visible line "
                        "only when the mean axis fails endpoint-anchor closure and the chosen visible line "
                        "strictly restores both endpoint anchors under the same line-residual gate."
                    ),
                }
                changed = True
        if not changed:
            break

    return {bond_index: (selected_axes_by_bond[bond_index], selected_policies_by_bond[bond_index]) for bond_index in selected_axes_by_bond}


def bond_axis_from_svg_lines(lines: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, float] | None:
    axis, _policy = bond_axis_from_svg_lines_with_policy(lines)
    return axis


def intersect_lines(line_a: tuple[np.ndarray, float], line_b: tuple[np.ndarray, float]) -> np.ndarray | None:
    matrix = np.asarray([line_a[0], line_b[0]], dtype=np.float64)
    det = float(np.linalg.det(matrix))
    if abs(det) <= 1e-6:
        return None
    return np.linalg.solve(matrix, np.asarray([line_a[1], line_b[1]], dtype=np.float64))


def affine_diagnostics(affine: np.ndarray) -> dict[str, Any]:
    linear = np.asarray([[affine[0, 0], affine[1, 0]], [affine[0, 1], affine[1, 1]]], dtype=np.float64)
    singular_values = np.linalg.svd(linear, compute_uv=False)
    smallest = float(min(singular_values)) if len(singular_values) else 0.0
    largest = float(max(singular_values)) if len(singular_values) else 0.0
    return {
        "matrix": [[float(value) for value in row] for row in affine.tolist()],
        "determinant": float(np.linalg.det(linear)),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "scale_ratio": float(largest / smallest) if smallest > 0.0 else None,
        "reflection_expected_from_svg_y_axis": True,
    }


def affine_diagnostic_blockers(
    diagnostics: dict[str, Any],
    *,
    max_rmse: float,
    max_line_abs_p95: float,
    max_line_abs_max: float,
    max_intersection_anchor_rmse: float,
    max_intersection_anchor_abs_max: float,
    max_scale_ratio: float,
    min_intersection_anchors: int,
) -> list[str]:
    blockers: list[str] = []
    affine = diagnostics.get("affine") if isinstance(diagnostics.get("affine"), dict) else {}
    determinant = affine.get("determinant")
    scale_ratio = affine.get("scale_ratio")
    rmse = diagnostics.get("line_constraint_rmse_svg_units")
    line_abs_p95 = diagnostics.get("line_constraint_abs_p95_svg_units")
    line_abs_max = diagnostics.get("line_constraint_abs_max_svg_units")
    anchor_rmse = diagnostics.get("intersection_anchor_rmse_svg_units")
    anchor_abs_max = diagnostics.get("intersection_anchor_abs_max_svg_units")
    intersection_anchors = int(diagnostics.get("intersection_anchor_count") or 0)
    if rmse is None or float(rmse) > float(max_rmse):
        blockers.append(f"line-constrained pose mapping RMSE {rmse} exceeds maximum {float(max_rmse):.3f}")
    if line_abs_p95 is None or float(line_abs_p95) > float(max_line_abs_p95):
        blockers.append(f"line residual p95 {line_abs_p95} exceeds maximum {float(max_line_abs_p95):.3f}")
    if line_abs_max is None or float(line_abs_max) > float(max_line_abs_max):
        blockers.append(f"line residual max {line_abs_max} exceeds maximum {float(max_line_abs_max):.3f}")
    if anchor_rmse is None or float(anchor_rmse) > float(max_intersection_anchor_rmse):
        blockers.append(f"intersection anchor RMSE {anchor_rmse} exceeds maximum {float(max_intersection_anchor_rmse):.3f}")
    if anchor_abs_max is None or float(anchor_abs_max) > float(max_intersection_anchor_abs_max):
        blockers.append(
            f"intersection anchor max residual {anchor_abs_max} exceeds maximum {float(max_intersection_anchor_abs_max):.3f}"
        )
    if "affine" in diagnostics:
        if determinant is None or float(determinant) >= 0.0:
            blockers.append("line-constrained affine determinant must be negative because SVG y-axis is flipped from molfile coordinates")
        if scale_ratio is None or float(scale_ratio) > float(max_scale_ratio):
            blockers.append(
                f"line-constrained affine scale ratio {scale_ratio} exceeds maximum {float(max_scale_ratio):.3f}"
            )
    if intersection_anchors < int(min_intersection_anchors):
        blockers.append(
            f"line-constrained affine has {intersection_anchors} multibond atom intersection anchors; "
            f"minimum is {int(min_intersection_anchors)}"
        )
    return blockers


def fit_line_constrained_affine(
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    atom_by_index = {int(atom["atom_index"]): atom for atom in atoms}
    selected_axes = select_svg_bond_axes_with_endpoint_policy(bonds, lines_by_bond)
    axis_by_bond: dict[int, tuple[np.ndarray, float]] = {}
    axis_policy_counts: Counter[str] = Counter()
    incident_lines_by_atom: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    line_rows: list[list[float]] = []
    line_targets: list[float] = []

    for bond in bonds:
        bond_index = int(bond["bond_index"])
        selected = selected_axes.get(bond_index)
        if selected is None:
            continue
        axis, axis_policy = selected
        axis_policy_counts[str(axis_policy.get("selection") or "unknown")] += 1
        axis_by_bond[bond_index] = axis
        normal, offset = axis
        for atom_index_key in ("begin_atom_index", "end_atom_index"):
            atom_index = int(bond[atom_index_key])
            atom = atom_by_index.get(atom_index)
            if atom is None:
                continue
            x = float(atom["mol_x"])
            y = float(atom["mol_y"])
            line_rows.append([normal[0] * x, normal[0] * y, normal[0], normal[1] * x, normal[1] * y, normal[1]])
            line_targets.append(float(offset))
            incident_lines_by_atom[atom_index].append(axis)

    point_rows: list[list[float]] = []
    point_targets: list[float] = []
    intersection_points: list[tuple[int, np.ndarray]] = []
    for atom_index, incident_lines in incident_lines_by_atom.items():
        intersections = []
        for first_index in range(len(incident_lines)):
            for second_index in range(first_index + 1, len(incident_lines)):
                intersection = intersect_lines(incident_lines[first_index], incident_lines[second_index])
                if intersection is not None and np.all(np.isfinite(intersection)):
                    intersections.append(intersection)
        if not intersections:
            continue
        atom = atom_by_index.get(atom_index)
        if atom is None:
            continue
        target = np.mean(np.asarray(intersections, dtype=np.float64), axis=0)
        x = float(atom["mol_x"])
        y = float(atom["mol_y"])
        point_rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        point_targets.append(float(target[0]))
        point_rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        point_targets.append(float(target[1]))
        intersection_points.append((atom_index, target))

    if len(line_rows) < 6:
        raise ValueError("not enough SVG bond-line endpoint constraints for affine pose mapping")
    design = np.asarray(line_rows + point_rows, dtype=np.float64)
    targets = np.asarray(line_targets + point_targets, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(design))
    if rank < 6:
        raise ValueError(f"rank-deficient line-constrained affine fit: rank={rank}, required=6")

    params, *_ = np.linalg.lstsq(design, targets, rcond=None)
    affine = np.asarray([[params[0], params[3]], [params[1], params[4]], [params[2], params[5]]], dtype=np.float64)

    line_residuals = np.asarray(line_rows, dtype=np.float64) @ params - np.asarray(line_targets, dtype=np.float64)
    line_abs = np.abs(line_residuals)
    if point_rows:
        point_residuals = np.asarray(point_rows, dtype=np.float64) @ params - np.asarray(point_targets, dtype=np.float64)
        point_pairs = point_residuals.reshape(-1, 2)
        point_distances = np.sqrt((point_pairs**2).sum(axis=1))
    else:
        point_distances = np.asarray([], dtype=np.float64)

    diagnostics = {
        "fit_method": "least_squares_affine_from_svg_bond_axis_perpendicular_constraints_with_multibond_atom_intersections",
        "line_constraint_rmse_svg_units": float(np.sqrt(np.mean(line_residuals**2))) if len(line_residuals) else None,
        "line_constraint_abs_p95_svg_units": float(np.quantile(line_abs, 0.95)) if len(line_abs) else None,
        "line_constraint_abs_max_svg_units": float(np.max(line_abs)) if len(line_abs) else None,
        "line_constraint_equation_count": int(len(line_rows)),
        "line_constraint_bond_count": int(len(axis_by_bond)),
        "intersection_anchor_count": int(len(intersection_points)),
        "intersection_anchor_rmse_svg_units": float(np.sqrt(np.mean(point_distances**2))) if len(point_distances) else None,
        "intersection_anchor_abs_max_svg_units": float(np.max(point_distances)) if len(point_distances) else None,
        "design_rank": rank,
        "affine": affine_diagnostics(affine),
        "svg_bond_axis_selection_counts": {key: int(value) for key, value in sorted(axis_policy_counts.items())},
        "svg_bond_line_policy": (
            "CDK StandardGenerator shortens visible bond segments around atom symbols; "
            "the fitted pose constrains mol atom centers to the corresponding rendered bond axes "
            "instead of to shortened segment midpoints."
        ),
    }
    return affine, diagnostics


def closest_point_on_axis(axis: tuple[np.ndarray, float], point: np.ndarray) -> np.ndarray:
    normal, offset = axis
    return point - (float(np.dot(normal, point)) - float(offset)) * normal


def representative_parallel_axis(axes: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, float] | None:
    if not axes:
        return None
    reference_normal = np.asarray(axes[0][0], dtype=np.float64)
    reference_norm = float(np.linalg.norm(reference_normal))
    if reference_norm <= 1e-9:
        return None
    reference_normal = reference_normal / reference_norm
    aligned_normals: list[np.ndarray] = []
    aligned_offsets: list[float] = []
    for normal, offset in axes:
        current_normal = np.asarray(normal, dtype=np.float64)
        current_norm = float(np.linalg.norm(current_normal))
        if current_norm <= 1e-9:
            continue
        current_normal = current_normal / current_norm
        current_offset = float(offset) / current_norm
        if float(np.dot(current_normal, reference_normal)) < 0.0:
            current_normal = -current_normal
            current_offset = -current_offset
        aligned_normals.append(current_normal)
        aligned_offsets.append(current_offset)
    if not aligned_normals:
        return None
    mean_normal = np.mean(np.asarray(aligned_normals, dtype=np.float64), axis=0)
    mean_norm = float(np.linalg.norm(mean_normal))
    if mean_norm <= 1e-9:
        return None
    mean_normal = mean_normal / mean_norm
    return mean_normal, float(np.mean(np.asarray(aligned_offsets, dtype=np.float64)))


def least_squares_axis_point(axes: list[tuple[np.ndarray, float]]) -> np.ndarray | None:
    if not axes:
        return None
    matrix = np.asarray([axis[0] for axis in axes], dtype=np.float64)
    targets = np.asarray([axis[1] for axis in axes], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < 2:
        return None
    point, *_ = np.linalg.lstsq(matrix, targets, rcond=None)
    if not np.all(np.isfinite(point)):
        return None
    return point


def reconstruct_svg_atom_centers(
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    lines_by_bond: dict[int, list[tuple[float, float, float, float]]],
    affine_seed: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    atom_by_index = {int(atom["atom_index"]): atom for atom in atoms}
    selected_axes = select_svg_bond_axes_with_endpoint_policy(bonds, lines_by_bond)
    axis_by_bond: dict[int, tuple[np.ndarray, float]] = {}
    axis_policy_counts: Counter[str] = Counter()
    axes_by_atom: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    graph_neighbors: dict[int, list[int]] = defaultdict(list)

    for bond in bonds:
        bond_index = int(bond["bond_index"])
        begin = int(bond["begin_atom_index"])
        end = int(bond["end_atom_index"])
        graph_neighbors[begin].append(end)
        graph_neighbors[end].append(begin)
        selected = selected_axes.get(bond_index)
        if selected is None:
            continue
        axis, axis_policy = selected
        axis_policy_counts[str(axis_policy.get("selection") or "unknown")] += 1
        axis_by_bond[bond_index] = axis
        axes_by_atom[begin].append(axis)
        axes_by_atom[end].append(axis)

    seed_points: dict[int, np.ndarray] = {}
    for atom in atoms:
        atom_index = int(atom["atom_index"])
        seed_points[atom_index] = np.asarray([atom["mol_x"], atom["mol_y"], 1.0], dtype=np.float64) @ affine_seed

    centers: dict[int, np.ndarray] = {}
    sources: dict[int, str] = {}
    residuals_by_atom: dict[int, list[float]] = {}

    for atom in atoms:
        atom_index = int(atom["atom_index"])
        axes = axes_by_atom.get(atom_index, [])
        point = least_squares_axis_point(axes)
        if point is not None:
            centers[atom_index] = point
            sources[atom_index] = "svg_axis_intersection_lstsq"
            continue
        representative_axis = representative_parallel_axis(axes)
        if representative_axis is not None:
            centers[atom_index] = closest_point_on_axis(representative_axis, seed_points[atom_index])
            sources[atom_index] = (
                "single_axis_affine_seed_projection"
                if len(axes) == 1
                else "parallel_axis_affine_seed_projection"
            )
            continue
        centers[atom_index] = seed_points[atom_index]
        sources[atom_index] = "affine_seed_no_svg_axis"

    for atom_index, axes in axes_by_atom.items():
        point = centers.get(atom_index)
        if point is None:
            continue
        residuals_by_atom[atom_index] = [abs(float(np.dot(axis[0], point) - axis[1])) for axis in axes]

    line_abs: list[float] = []
    for atom_index in sorted(residuals_by_atom):
        line_abs.extend(residuals_by_atom[atom_index])

    anchor_distances: list[float] = []
    anchor_count = 0
    for atom_index, axes in axes_by_atom.items():
        if len(axes) < 2 or atom_index not in centers:
            continue
        target = least_squares_axis_point(axes)
        if target is None:
            continue
        anchor_count += 1
        anchor_distances.append(float(np.linalg.norm(centers[atom_index] - target)))

    pair_distances: list[float] = []
    for bond in bonds:
        begin = int(bond["begin_atom_index"])
        end = int(bond["end_atom_index"])
        if begin in centers and end in centers:
            pair_distances.append(float(np.linalg.norm(centers[begin] - centers[end])))

    source_counts = Counter(sources.values())
    diagnostics = {
        "fit_method": "svg_bond_axis_atom_center_hybrid_intersections_with_single_axis_affine_seed_projection",
        "line_constraint_rmse_svg_units": float(np.sqrt(np.mean(np.asarray(line_abs, dtype=np.float64) ** 2))) if line_abs else None,
        "line_constraint_abs_p95_svg_units": float(np.quantile(np.asarray(line_abs, dtype=np.float64), 0.95)) if line_abs else None,
        "line_constraint_abs_max_svg_units": float(max(line_abs)) if line_abs else None,
        "line_constraint_equation_count": int(len(line_abs)),
        "line_constraint_bond_count": int(len(axis_by_bond)),
        "intersection_anchor_count": int(anchor_count),
        "intersection_anchor_rmse_svg_units": (
            float(np.sqrt(np.mean(np.asarray(anchor_distances, dtype=np.float64) ** 2))) if anchor_distances else None
        ),
        "intersection_anchor_abs_max_svg_units": float(max(anchor_distances)) if anchor_distances else None,
        "atom_center_source_counts": {key: int(value) for key, value in source_counts.items()},
        "svg_bond_axis_selection_counts": {key: int(value) for key, value in sorted(axis_policy_counts.items())},
        "bond_center_distance_min_svg_units": float(min(pair_distances)) if pair_distances else None,
        "bond_center_distance_p01_svg_units": (
            float(np.quantile(np.asarray(pair_distances, dtype=np.float64), 0.01)) if pair_distances else None
        ),
        "policy": (
            "Atom centers are reconstructed in SVG/image space from rendered bond axes. "
            "Atoms with two or more visible incident bond axes use their least-squares axis intersection; "
            "single-axis atoms use the global affine seed only for along-axis placement and are projected "
            "back to the rendered SVG bond axis."
        ),
    }
    return centers, diagnostics


def apply_affine_atom_coords(
    atoms: list[dict[str, Any]],
    affine: np.ndarray,
    viewbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    width, height = viewbox[2], viewbox[3]
    coords = []
    for atom in atoms:
        x, y = np.asarray([atom["mol_x"], atom["mol_y"], 1.0]) @ affine
        coords.append(
            {
                "atom_index": int(atom["atom_index"]),
                "token": str(atom["token"]),
                "x": max(0.0, min(1.0, float(x / width))),
                "y": max(0.0, min(1.0, float(y / height))),
            }
        )
    return coords


def atom_coords_from_svg_centers(
    atoms: list[dict[str, Any]],
    centers: dict[int, np.ndarray],
    viewbox: tuple[float, float, float, float],
    cdk_label_overrides: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    width, height = viewbox[2], viewbox[3]
    coords = []
    label_overrides = cdk_label_overrides or {}
    for atom in atoms:
        atom_index = int(atom["atom_index"])
        point = centers.get(atom_index)
        if point is None:
            raise ValueError(f"missing SVG atom center for atom {atom_index}")
        token = normalize_label(label_overrides.get(atom_index, "")) or training_atom_token_for_cdk_atom(atom)
        coords.append(
            {
                "atom_index": atom_index,
                "token": token,
                "x": max(0.0, min(1.0, float(point[0] / width))),
                "y": max(0.0, min(1.0, float(point[1] / height))),
            }
        )
    return coords


def canonical_smiles_like(value: str) -> str:
    # CDK CXSMILES support is broader than RDKit for Markush. Keep a stable text
    # hash instead of claiming RDKit canonical equivalence for Markush rows.
    return re.sub(r"\s+", " ", value.strip())


def infer_source_dataset(raw_root: Path, candidate: dict[str, str], explicit_source_dataset: str) -> str:
    source_collection = str(candidate.get("source_collection") or "").strip()
    if source_collection == "markushgrapher_synthetic_training_source":
        return "markushgrapher-synthetic-training"
    if source_collection == "markushgrapher2":
        return "markushgrapher2"
    source_file = Path(str(candidate.get("source_file") or ""))
    parts = source_file.parts
    if "markushgrapher-synthetic-training" in parts:
        return "markushgrapher-synthetic-training"
    if "markushgrapher2" in parts:
        return "markushgrapher2"
    if "MolGrapher-Synthetic-300K" in parts:
        return "docling-project/MolGrapher-Synthetic-300K"
    explicit = str(explicit_source_dataset or "").strip()
    if explicit:
        return explicit
    return raw_root.name


def source_patch_for_depictor_mode(mode: str) -> str:
    if mode == "batch":
        return (
            "BatchDepictor.java invokes the seeded Depictor once per row inside a single JVM, passes per-row "
            "renderer canvas padding/margin retry parameters, writes per-row .render.json metadata, and applies "
            "the CXSMILES dummyLabel/atomLabel/molFileAlias pseudo-atom label patch before CDK depiction."
        )
    return (
        "Depictor.java requires -Dmarkush.seed, accepts renderer canvas padding/margin retry properties, "
        "writes per-row .render.json metadata, and applies the CXSMILES dummyLabel/atomLabel/molFileAlias "
        "pseudo-atom label patch before CDK depiction."
    )


def infer_manifest_source(raw_root: Path, subset_glob: str, candidates: list[dict[str, str]], explicit_source_dataset: str) -> dict[str, Any]:
    dataset_counts = Counter(infer_source_dataset(raw_root, candidate, explicit_source_dataset) for candidate in candidates)
    source_files = [str(candidate.get("source_file") or "") for candidate in candidates if str(candidate.get("source_file") or "")]
    source_parents = Counter(str(Path(path).parent) for path in source_files)
    return {
        "raw_root_argument": str(raw_root),
        "subset_glob_argument": str(subset_glob),
        "inferred_source_dataset_counts": dict(sorted(dataset_counts.items())),
        "source_file_count": int(len(source_files)),
        "unique_source_file_count": int(len(set(source_files))),
        "source_parent_examples": dict(source_parents.most_common(8)),
    }


def failure_category(error: str) -> str:
    text = str(error or "")
    if "CDK/RDKit atom-index alignment failed: atomic-number mismatch" in text:
        return "atom_index_alignment_atomic_number_mismatch"
    if "InvalidSmilesException" in text or "valid kekul" in text.lower():
        return "cdk_invalid_smiles_or_kekule"
    if "missing warped selected atom-center axis" in text:
        return "formal_warp_missing_svg_axis"
    if "formal nonlinear document warp pose preservation failed" in text:
        return "formal_warp_pose_residual"
    if "atom coordinates are too close for MolNexTR coordinate supervision" in text:
        return "atom_coordinates_too_close_for_molnextr"
    if "rank-deficient line-constrained affine fit" in text:
        return "rank_deficient_line_constrained_affine"
    if "source SVG bond geometry sampled points invalid" in text:
        return "source_svg_bond_geometry_sampled_points_invalid"
    if "CDK/RDKit atom-index alignment failed" in text:
        return "atom_index_alignment_no_validated_mapping"
    if "line-constrained affine scale ratio" in text:
        return "line_constrained_affine_scale_ratio"
    if "line residual max" in text:
        return "line_residual_max_over_threshold"
    if "line-constrained affine has" in text and "multibond atom intersection anchors" in text:
        return "too_few_intersection_anchors"
    if "line-constrained pose mapping RMSE" in text:
        return "source_svg_pose_mapping_rmse"
    if "line-constrained affine determinant" in text:
        return "svg_y_axis_affine_orientation_mismatch"
    if "accepted_candidate_filter" in text or "dummy_atom_coordinate" in text:
        return "accepted_candidate_filter_rejection"
    return text.split(":", 1)[0][:120] or "unknown_failure"


def failure_summary(failures: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    source_prefix_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    category_bucket_counts: Counter[str] = Counter()
    category_source_counts: Counter[str] = Counter()
    for failure in failures:
        category = failure_category(str(failure.get("error") or ""))
        bucket = str(failure.get("candidate_bucket") or "")
        source_id = str(failure.get("source_id") or "")
        source_prefix = source_id.split(":", 1)[0] if source_id else ""
        stage = str(failure.get("stage") or "")
        category_counts[category] += 1
        bucket_counts[bucket] += 1
        source_prefix_counts[source_prefix] += 1
        stage_counts[stage] += 1
        category_bucket_counts[f"{category}|{bucket}"] += 1
        category_source_counts[f"{category}|{source_prefix}"] += 1
    return {
        "failure_count": int(len(failures)),
        "category_counts": {key: int(value) for key, value in category_counts.most_common()},
        "bucket_counts": {key: int(value) for key, value in bucket_counts.most_common()},
        "source_prefix_counts": {key: int(value) for key, value in source_prefix_counts.most_common()},
        "stage_counts": {key: int(value) for key, value in stage_counts.most_common()},
        "category_bucket_counts": {key: int(value) for key, value in category_bucket_counts.most_common()},
        "category_source_counts": {key: int(value) for key, value in category_source_counts.most_common()},
    }


def copy_svg_as_image(svg_path: Path, output_path: Path) -> None:
    # The validator only needs the image to be readable by PIL. Convert with
    # cairosvg if available; otherwise keep the row out of the trainable shard.
    try:
        import cairosvg
    except Exception as exc:
        raise RuntimeError(f"cairosvg unavailable: {exc}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(output_path))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["source_id", "source_arrow", "file_path", "SMILES", "smiles", "structure_type_bucket", "render_quality", "reliable_training_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_plan(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "plan_index",
        "source_id",
        "source_collection",
        "source_file",
        "source_record_id",
        "source_document_key",
        "source_plan_index",
        "subset",
        "annotation_r_count",
        "annotation_r_bucket",
        "cxsmiles_dummy_label_count",
        "cxsmiles_dummy_label_bucket",
        "variable_anchor_count",
        "variable_anchor_bucket",
        "all_pseudo_atom_label_count",
        "fixed_abbreviation_pseudo_atom_count",
        "non_variable_pseudo_atom_count",
        "annotation_r_count_diagnostic",
        "annotation_count_matches_dummy_label_count",
        "source_anchor_trainable",
        "cxsmiles_star_count",
        "selection_hash",
        "source_anchor_audit",
        "annotation",
        "cxsmiles_opt",
        "cxsmiles",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            plan_index = str(row.get("plan_index") or index)
            writer.writerow(
                {
                    "plan_index": plan_index,
                    **{name: str(row.get(name) or "") for name in fieldnames if name != "plan_index"},
                }
            )


def read_candidate_plan_csv(path: Path, *, raw_root: Path, subset_glob: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        return []

    missing_annotation = any(not str(row.get("annotation") or "").strip() for row in rows)
    if missing_annotation:
        raw_by_source_id = {row["source_id"]: row for row in iter_markush_rows(raw_root, subset_glob, None)}
        missing = []
        for row in rows:
            raw = raw_by_source_id.get(str(row.get("source_id") or ""))
            if raw is None:
                missing.append(str(row.get("source_id") or ""))
                continue
            for key in [
                "annotation",
                "cxsmiles_opt",
                "source_file",
                "source_collection",
                "source_record_id",
                "source_document_key",
                "subset",
                "annotation_r_count",
                "annotation_r_bucket",
                "cxsmiles_dummy_label_count",
                "cxsmiles_dummy_label_bucket",
                "variable_anchor_count",
                "variable_anchor_bucket",
                "all_pseudo_atom_label_count",
                "fixed_abbreviation_pseudo_atom_count",
                "non_variable_pseudo_atom_count",
                "annotation_r_count_diagnostic",
                "annotation_count_matches_dummy_label_count",
                "source_anchor_audit",
                "cxsmiles_star_count",
                "cxsmiles",
            ]:
                if not str(row.get(key) or "").strip():
                    row[key] = str(raw.get(key) or "")
        still_missing = [str(row.get("source_id") or "") for row in rows if not str(row.get("annotation") or "").strip()]
        if still_missing:
            examples = ", ".join(still_missing[:5])
            raise ValueError(f"candidate plan is missing annotation for {len(still_missing)} rows; examples: {examples}")
        if missing:
            examples = ", ".join(missing[:5])
            raise ValueError(f"candidate plan rows were not found in raw source lookup: {examples}")

    for fallback_index, row in enumerate(rows):
        if not str(row.get("plan_index") or "").strip():
            row["plan_index"] = str(fallback_index)
        anchor_audit = cxsmiles_dummy_anchor_audit(
            str(row.get("cxsmiles") or ""),
            str(row.get("annotation") or ""),
            str(row.get("cxsmiles_opt") or ""),
        )
        if anchor_audit["passed"]:
            row["annotation_r_count"] = str(anchor_audit["dummy_label_count"])
            row["annotation_r_bucket"] = str(anchor_audit["dummy_label_bucket"] or "")
        row["cxsmiles_dummy_label_count"] = str(anchor_audit["dummy_label_count"])
        row["cxsmiles_dummy_label_bucket"] = str(anchor_audit["dummy_label_bucket"] or "")
        row["variable_anchor_count"] = str(anchor_audit["dummy_label_count"])
        row["variable_anchor_bucket"] = str(anchor_audit["dummy_label_bucket"] or "")
        row["all_pseudo_atom_label_count"] = str(anchor_audit["all_pseudo_atom_label_count"])
        row["fixed_abbreviation_pseudo_atom_count"] = str(anchor_audit["fixed_abbreviation_pseudo_atom_count"])
        row["non_variable_pseudo_atom_count"] = str(anchor_audit["non_variable_pseudo_atom_count"])
        row["annotation_r_count_diagnostic"] = str(anchor_audit["annotation_r_count"])
        row["annotation_count_matches_dummy_label_count"] = str(anchor_audit["annotation_count_matches_dummy_label_count"])
        row["source_anchor_trainable"] = str(bool(anchor_audit["passed"]))
        row["source_anchor_audit"] = json.dumps(anchor_audit, ensure_ascii=False, sort_keys=True)
        required = [
            "source_id",
            "source_file",
            "source_document_key",
            "subset",
            "cxsmiles_dummy_label_count",
            "cxsmiles_star_count",
            "selection_hash",
            "source_anchor_audit",
            "annotation",
            "cxsmiles_opt",
            "cxsmiles",
        ]
        missing = [key for key in required if not str(row.get(key) or "").strip()]
        if missing:
            raise ValueError(f"candidate plan row {row.get('plan_index')} missing required fields: {missing}")
    return rows


def filter_source_anchor_trainable_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kept: list[dict[str, str]] = []
    issue_counts: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []
    annotation_mismatch_count = 0
    for row in rows:
        try:
            audit = json.loads(str(row.get("source_anchor_audit") or "{}"))
        except json.JSONDecodeError:
            audit = cxsmiles_dummy_anchor_audit(
                str(row.get("cxsmiles") or ""),
                str(row.get("annotation") or ""),
                str(row.get("cxsmiles_opt") or ""),
            )
        if audit.get("annotation_count_matches_dummy_label_count") is False:
            annotation_mismatch_count += 1
        if audit.get("passed") is True:
            kept.append(row)
            continue
        issues = [str(issue) for issue in (audit.get("issues") if isinstance(audit.get("issues"), list) else [])]
        issue_counts.update(issues or ["source_anchor_audit_failed"])
        if len(rejected_examples) < 80:
            rejected_examples.append(
                {
                    "source_id": str(row.get("source_id") or ""),
                    "plan_index": str(row.get("plan_index") or ""),
                    "issues": issues,
                    "dummy_label_count": audit.get("dummy_label_count"),
                    "variable_anchor_count": audit.get("dummy_label_count"),
                    "fixed_abbreviation_pseudo_atom_count": audit.get("fixed_abbreviation_pseudo_atom_count"),
                    "non_variable_pseudo_atom_count": audit.get("non_variable_pseudo_atom_count"),
                    "annotation_r_count": audit.get("annotation_r_count"),
                    "dummy_atoms_without_real_neighbor": audit.get("dummy_atoms_without_real_neighbor"),
                }
            )
    for index, row in enumerate(kept):
        row["source_plan_index"] = str(row.get("plan_index") or "")
        row["plan_index"] = str(index)
    return kept, {
        "enabled": True,
        "policy": "CXSMILES dummyLabel/atomLabel/molFileAlias anchors must be Markush variable labels; fixed substituent abbreviations and other non-variable pseudo atoms are diagnostic only; every variable anchor must have a real-atom graph neighbor before rendering",
        "input_rows": int(len(rows)),
        "kept_rows": int(len(kept)),
        "rejected_rows": int(len(rows) - len(kept)),
        "reason_counts": dict(sorted(issue_counts.items())),
        "annotation_count_mismatch_diagnostic_rows": int(annotation_mismatch_count),
        "rejected_examples": rejected_examples,
        "plan_index_reindexed_after_filter": True,
    }


def apply_heldout_filter_to_candidate_plan(
    rows: list[dict[str, str]],
    *,
    heldout_csvs: list[Path],
    allow_backbone_overlap: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kept, report = filter_heldout_overlaps(
        rows,
        heldout_csvs=heldout_csvs,
        allow_backbone_overlap=allow_backbone_overlap,
    )
    for index, row in enumerate(kept):
        row["plan_index"] = str(index)
    return kept, report


def validate_global_plan_index_slice(rows: list[dict[str, str]], *, plan_start: int, plan_end: int) -> dict[str, Any]:
    indexes: list[int] = []
    blockers: list[str] = []
    for local_index, row in enumerate(rows):
        text = str(row.get("plan_index") or "").strip()
        if not text:
            blockers.append(f"slice row {local_index} has empty plan_index")
            continue
        try:
            index = int(text)
        except ValueError:
            blockers.append(f"slice row {local_index} has non-integer plan_index={text!r}")
            continue
        indexes.append(index)
    if len(indexes) != len(rows):
        blockers.append(f"only {len(indexes)} / {len(rows)} rows have valid integer plan_index")
    duplicate_count = len(indexes) - len(set(indexes))
    if duplicate_count:
        blockers.append(f"candidate plan slice has duplicate plan_index values: {duplicate_count}")
    expected = list(range(int(plan_start), int(plan_end)))
    if indexes and indexes != expected:
        examples = [
            {"position": int(pos), "expected": int(expected[pos]), "observed": int(indexes[pos])}
            for pos in range(min(len(indexes), len(expected)))
            if indexes[pos] != expected[pos]
        ][:10]
        blockers.append(
            "candidate plan slice does not preserve global contiguous plan_index values "
            f"for [{plan_start}, {plan_end}); examples={examples}"
        )
    return {
        "global_plan_index_required": True,
        "global_plan_index_preserved": not blockers,
        "plan_start": int(plan_start),
        "plan_end": int(plan_end),
        "row_count": int(len(rows)),
        "min_plan_index": min(indexes) if indexes else None,
        "max_plan_index": max(indexes) if indexes else None,
        "duplicate_plan_index_count": int(duplicate_count),
        "blockers": blockers,
    }


def validate_plan_index_set(rows: list[dict[str, str]]) -> dict[str, Any]:
    indexes: list[int] = []
    blockers: list[str] = []
    for local_index, row in enumerate(rows):
        text = str(row.get("plan_index") or "").strip()
        if not text:
            blockers.append(f"selected row {local_index} has empty plan_index")
            continue
        try:
            indexes.append(int(text))
        except ValueError:
            blockers.append(f"selected row {local_index} has non-integer plan_index={text!r}")
    if len(indexes) != len(rows):
        blockers.append(f"only {len(indexes)} / {len(rows)} rows have valid integer plan_index")
    duplicate_count = len(indexes) - len(set(indexes))
    if duplicate_count:
        blockers.append(f"candidate plan selection has duplicate plan_index values: {duplicate_count}")
    return {
        "global_plan_index_required": True,
        "global_plan_index_preserved": not blockers,
        "row_count": int(len(rows)),
        "min_plan_index": min(indexes) if indexes else None,
        "max_plan_index": max(indexes) if indexes else None,
        "duplicate_plan_index_count": int(duplicate_count),
        "blockers": blockers,
    }


def select_bucket_stratified_candidate_plan_window(
    rows: list[dict[str, str]],
    *,
    plan_start: int,
    plan_end: int,
    bucket_targets: dict[str, int],
    candidate_bucket_targets: dict[str, int],
    rows_arg: int,
    candidate_multiplier: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    desired = resolve_candidate_targets(
        accepted_targets=bucket_targets,
        candidate_targets=candidate_bucket_targets,
        rows=int(rows_arg),
        candidate_multiplier=int(candidate_multiplier),
    )
    total_window_rows = sum(int(desired.get(bucket, 0)) for bucket in COUNT_BUCKETS)
    if total_window_rows <= 0:
        raise ValueError("bucket-stratified candidate plan selection requires at least one candidate target")
    if int(plan_start) % total_window_rows != 0:
        raise ValueError(
            f"plan_start={plan_start} is not aligned to bucket-stratified window size {total_window_rows}"
        )
    if int(plan_end) != int(plan_start) + total_window_rows:
        raise ValueError(
            f"plan_end={plan_end} does not equal plan_start + bucket-stratified window size {total_window_rows}"
        )
    window_index = int(plan_start) // total_window_rows
    available_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        available_by_bucket[str(row.get("annotation_r_bucket") or "")].append(row)

    selected_by_bucket: dict[str, list[dict[str, str]]] = {}
    bucket_windows: dict[str, dict[str, int]] = {}
    blockers: list[str] = []
    for bucket in COUNT_BUCKETS:
        wanted = int(desired.get(bucket, 0))
        start = window_index * wanted
        end = start + wanted
        available = available_by_bucket.get(bucket, [])
        selected_by_bucket[bucket] = available[start:end]
        bucket_windows[bucket] = {
            "start": int(start),
            "end": int(end),
            "requested": int(wanted),
            "available": int(len(available)),
            "selected": int(len(selected_by_bucket[bucket])),
        }
        required_trainable = int(bucket_targets.get(bucket, 0)) if bucket_targets else 1
        if required_trainable > 0 and len(selected_by_bucket[bucket]) < wanted:
            blockers.append(
                f"bucket {bucket} selected {len(selected_by_bucket[bucket])} candidates from "
                f"[{start}, {end}); requested {wanted}; available {len(available)}"
            )

    selected: list[dict[str, str]] = []
    max_len = max((len(items) for items in selected_by_bucket.values()), default=0)
    for offset in range(max_len):
        for bucket in COUNT_BUCKETS:
            items = selected_by_bucket.get(bucket, [])
            if offset < len(items):
                selected.append(items[offset])
    audit = validate_plan_index_set(selected)
    return selected, {
        "enabled": True,
        "policy": "bucket_stratified_candidate_plan_window_v1",
        "total_window_rows": int(total_window_rows),
        "window_index": int(window_index),
        "plan_start": int(plan_start),
        "plan_end": int(plan_end),
        "candidate_targets_by_bucket": {bucket: int(desired.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "bucket_windows": bucket_windows,
        "selected_candidate_rows_by_bucket": {
            bucket: int(len(selected_by_bucket.get(bucket, []))) for bucket in COUNT_BUCKETS
        },
        "selected_candidate_rows": int(len(selected)),
        "blockers": blockers + list(audit["blockers"]),
        "global_plan_index_audit": audit,
    }


def parse_bucket_window_overrides(value: str) -> dict[str, tuple[int, int]]:
    text = str(value or "").strip()
    if not text:
        return {}
    windows: dict[str, tuple[int, int]] = {}
    for item in re.split(r"[,;]", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"invalid bucket window override {item!r}; expected bucket=start:end")
        bucket, range_text = [part.strip() for part in item.split("=", 1)]
        if bucket not in COUNT_BUCKETS:
            raise ValueError(f"invalid bucket {bucket!r}; expected one of {COUNT_BUCKETS}")
        start_text, end_text = [part.strip() for part in range_text.split(":", 1)]
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < start:
            raise ValueError(f"invalid bucket window {item!r}; require 0 <= start <= end")
        windows[bucket] = (start, end)
    return windows


def select_explicit_bucket_candidate_plan_window(
    rows: list[dict[str, str]],
    *,
    plan_start: int,
    plan_end: int,
    bucket_targets: dict[str, int],
    bucket_windows: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if set(bucket_windows) != set(COUNT_BUCKETS):
        missing = sorted(set(COUNT_BUCKETS) - set(bucket_windows))
        extra = sorted(set(bucket_windows) - set(COUNT_BUCKETS))
        raise ValueError(f"explicit bucket windows must cover all buckets; missing={missing}; extra={extra}")
    available_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        available_by_bucket[str(row.get("annotation_r_bucket") or "")].append(row)

    selected_by_bucket: dict[str, list[dict[str, str]]] = {}
    audit_windows: dict[str, dict[str, int]] = {}
    blockers: list[str] = []
    for bucket in COUNT_BUCKETS:
        start, end = bucket_windows[bucket]
        available = available_by_bucket.get(bucket, [])
        selected_by_bucket[bucket] = available[start:end]
        requested = int(end - start)
        audit_windows[bucket] = {
            "start": int(start),
            "end": int(end),
            "requested": requested,
            "available": int(len(available)),
            "selected": int(len(selected_by_bucket[bucket])),
        }
        required_trainable = int(bucket_targets.get(bucket, 0)) if bucket_targets else 1
        if requested <= 0:
            if int(start) > int(len(available)):
                blockers.append(
                    f"bucket {bucket} empty explicit window starts at {start}; available {len(available)}"
                )
            continue
        if required_trainable > 0 and requested < required_trainable:
            blockers.append(
                f"bucket {bucket} explicit window has {requested} candidates but requires "
                f"{required_trainable} accepted target rows"
            )
        if required_trainable > 0 and len(selected_by_bucket[bucket]) < requested:
            blockers.append(
                f"bucket {bucket} selected {len(selected_by_bucket[bucket])} candidates from "
                f"[{start}, {end}); requested {requested}; available {len(available)}"
            )

    selected: list[dict[str, str]] = []
    max_len = max((len(items) for items in selected_by_bucket.values()), default=0)
    for offset in range(max_len):
        for bucket in COUNT_BUCKETS:
            items = selected_by_bucket.get(bucket, [])
            if offset < len(items):
                selected.append(items[offset])
    audit = validate_plan_index_set(selected)
    return selected, {
        "enabled": True,
        "policy": "explicit_per_bucket_candidate_plan_tail_window_v1",
        "total_window_rows": int(sum(end - start for start, end in bucket_windows.values())),
        "plan_start": int(plan_start),
        "plan_end": int(plan_end),
        "candidate_targets_by_bucket": {
            bucket: int(bucket_windows[bucket][1] - bucket_windows[bucket][0]) for bucket in COUNT_BUCKETS
        },
        "bucket_windows": audit_windows,
        "selected_candidate_rows_by_bucket": {
            bucket: int(len(selected_by_bucket.get(bucket, []))) for bucket in COUNT_BUCKETS
        },
        "selected_candidate_rows": int(len(selected)),
        "terminal_tail_window": True,
        "blockers": blockers + list(audit["blockers"]),
        "global_plan_index_audit": audit,
    }


def candidate_plan_report_from_rows(
    rows: list[dict[str, str]],
    *,
    raw_root: Path,
    subset_glob: str,
    seed: int,
    bucket_targets: dict[str, int],
    candidate_plan_csv: Path,
    source_candidate_plan_csv: Path | None,
    source_candidate_plan_json: Path | None,
    plan_start: int,
    plan_end: int,
    selection_policy: str = "precomputed_candidate_plan_slice_source_document_deduplicated_stratified_by_annotation_r_bucket",
    bucket_window_audit: dict[str, Any] | None = None,
    heldout_filter_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_bucket = Counter(str(row.get("annotation_r_bucket") or "") for row in rows)
    source_documents = [str(row.get("source_document_key") or "") for row in rows if str(row.get("source_document_key") or "")]
    source_ids = [str(row.get("source_id") or "") for row in rows if str(row.get("source_id") or "")]
    if bucket_window_audit:
        plan_index_audit = dict(bucket_window_audit.get("global_plan_index_audit") or validate_plan_index_set(rows))
        blockers = list(bucket_window_audit.get("blockers") or []) + list(plan_index_audit.get("blockers") or [])
    else:
        plan_index_audit = validate_global_plan_index_slice(rows, plan_start=plan_start, plan_end=plan_end)
        blockers = list(plan_index_audit["blockers"])
    return {
        "raw_root": str(raw_root),
        "subset_glob": subset_glob,
        "seed": int(seed),
        "selection_policy": selection_policy,
        "source_candidate_plan_csv": str(source_candidate_plan_csv) if source_candidate_plan_csv else "",
        "source_candidate_plan_json": str(source_candidate_plan_json) if source_candidate_plan_json else "",
        "candidate_plan_csv": str(candidate_plan_csv),
        "candidate_plan_slice_start": int(plan_start),
        "candidate_plan_slice_end": int(plan_end),
        "bucket_targets": {bucket: int(bucket_targets.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "selected_candidate_rows_by_bucket": {bucket: int(by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "selected_candidate_rows": len(rows),
        "selected_source_document_overlap": len(source_documents) - len(set(source_documents)),
        "selected_source_id_overlap": len(source_ids) - len(set(source_ids)),
        "global_plan_index_audit": plan_index_audit,
        "bucket_stratified_window": bucket_window_audit or {"enabled": False},
        "heldout_overlap_filter": heldout_filter_report
        or {
            "enabled": False,
            "heldout_csv": [],
            "input_rows": len(rows),
            "kept_rows": len(rows),
            "rejected_rows": 0,
            "reason_counts": {},
            "rejected_examples": [],
            "allow_backbone_overlap": False,
        },
        "blockers": blockers,
        "plan_only_safe_to_generate": len(rows) > 0 and not blockers,
        "caveat": (
            "This is a candidate-plan slice only. Rows become trainable only after CDK generation, "
            "pose validation, visual review, source-leak checks, coverage audit, and acceptance gates pass."
        ),
    }


def write_review_sheet(csv_path: Path, output_path: Path, *, tile_size: int = 260, columns: int = 4) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    font = ImageFont.load_default()
    tiles = []
    for row in rows[: min(len(rows), 48)]:
        image_path = Path(str(row.get("file_path") or ""))
        if not image_path.is_absolute():
            image_path = csv_path.parent / image_path
        image = Image.open(image_path).convert("RGB")
        render_quality = json.loads(str(row.get("render_quality") or "{}"))
        cells = render_quality.get("markush", {}).get("ocr_cells", [])
        atom_coords = render_quality.get("atom_coordinates", [])

        image.thumbnail((tile_size, tile_size - 58), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        x_offset = (tile_size - image.width) // 2
        tile.paste(image, (x_offset, 0))
        draw = ImageDraw.Draw(tile)
        for cell in cells[:40]:
            bbox = cell.get("bbox") if isinstance(cell, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1 = x_offset + float(bbox[0]) * image.width
            y1 = float(bbox[1]) * image.height
            x2 = x_offset + float(bbox[2]) * image.width
            y2 = float(bbox[3]) * image.height
            draw.rectangle((x1, y1, x2, y2), outline=(220, 0, 0), width=2)
        for atom in atom_coords[:80]:
            try:
                x = x_offset + float(atom["x"]) * image.width
                y = float(atom["y"]) * image.height
            except (KeyError, TypeError, ValueError):
                continue
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 90, 220))
        label = f"{row.get('source_id','')} cells={len(cells)} atoms={len(atom_coords)}"
        draw.text((6, tile_size - 48), label[:58], fill=(0, 0, 0), font=font)
        pose_mapping = render_quality.get("pose_mapping", {})
        rmse_label = pose_mapping.get("line_constraint_rmse_svg_units", pose_mapping.get("affine_rmse_svg_units", ""))
        draw.text((6, tile_size - 28), str(rmse_label)[:58], fill=(0, 0, 0), font=font)
        tiles.append(tile)

    if not tiles:
        return
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows_count * tile_size), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "csv": str(csv_path),
                "output": str(output_path),
                "sample_count": len(tiles),
                "red_boxes": "markush OCR/layout cells from CDK SVG path boxes",
                "blue_points": "MolNexTR atom coordinates from CDK molfile-to-SVG affine mapping",
                "status": "manual_visual_review_required_before_acceptance",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def build_generation_tasks(
    candidates: list[dict[str, str]],
    *,
    row_prefix: str,
    seed: int,
    rows: int,
    bucket_targets: dict[str, int],
    candidate_bucket_targets: dict[str, int] | None = None,
    renderer_seed_retries: int = 1,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    planned_by_bucket: Counter[str] = Counter()
    task_targets = candidate_bucket_targets or bucket_targets
    planned_source_count = 0
    retry_count = max(1, int(renderer_seed_retries))
    for index, candidate in enumerate(candidates):
        if task_targets and all(planned_by_bucket[bucket] >= int(task_targets.get(bucket, 0)) for bucket in COUNT_BUCKETS):
            break
        if not bucket_targets and planned_source_count >= int(rows):
            break
        candidate_bucket = str(candidate["annotation_r_bucket"])
        if task_targets and planned_by_bucket[candidate_bucket] >= int(task_targets.get(candidate_bucket, 0)):
            continue
        plan_index = int(candidate.get("plan_index") or index)
        primary_row_id = f"{row_prefix}_{plan_index:06d}"
        for attempt_index in range(retry_count):
            row_id = primary_row_id if attempt_index == 0 else f"{primary_row_id}_r{attempt_index:02d}"
            render_cxsmiles, harness_label_policy = canonicalize_markush_harness_labels(
                candidate["cxsmiles"],
                source_id=str(candidate.get("source_id") or ""),
                row_id=row_id,
            )
            task_candidate = dict(candidate)
            task_candidate["source_cxsmiles"] = candidate["cxsmiles"]
            task_candidate["source_annotation"] = candidate.get("annotation", "")
            task_candidate["cxsmiles"] = render_cxsmiles
            task_candidate["annotation"] = canonicalize_annotation_r_labels(
                str(candidate.get("annotation") or ""),
                harness_label_policy,
            )
            task_candidate["harness_label_policy"] = json.dumps(harness_label_policy, ensure_ascii=False, sort_keys=True)
            render_seed = stable_int_hash(
                seed,
                candidate.get("selection_hash", ""),
                primary_row_id,
                attempt_index,
                render_cxsmiles,
                modulo=2**31 - 1,
            )
            tasks.append(
                {
                    "index": int(index),
                    "candidate": task_candidate,
                    "candidate_bucket": candidate_bucket,
                    "plan_index": int(plan_index),
                    "row_id": row_id,
                    "primary_row_id": primary_row_id,
                    "render_seed": int(render_seed),
                    "renderer_attempt_index": int(attempt_index),
                    "renderer_attempt_count": int(retry_count),
                }
            )
        planned_by_bucket[candidate_bucket] += 1
        planned_source_count += 1
    return tasks


def accepted_candidate_filter_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_rmse=float(args.rmse_threshold),
        max_line_abs_p95=float(args.max_line_abs_p95),
        max_line_abs_max=float(args.max_line_abs_max),
        max_intersection_anchor_rmse=float(args.max_intersection_anchor_rmse),
        max_intersection_anchor_abs_max=float(args.max_intersection_anchor_abs_max),
        max_affine_scale_ratio=float(args.max_affine_scale_ratio),
        min_atom_coordinate_margin=0.0,
        min_atom_pair_distance=float(args.min_atom_pair_distance),
        max_ocr_cell_area_sum=0.12,
        max_ocr_cell_area_max=0.08,
        max_variable_bbox_atom_margin=0.015,
        allow_dummy_only_variable_neighbor=False,
        skip_substitution_image_check=False,
        min_fit_points=6,
        min_intersection_anchors=int(args.min_intersection_anchors),
        expected_coord_policy=POSE_COORD_POLICY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a production CDK/MarkushGenerator pose-mapped Markush layout shard.")
    parser.add_argument("--raw-root", default="training/molnextr_markush/data/raw/markushgrapher2")
    parser.add_argument("--subset-glob", default="*/*.parquet")
    parser.add_argument(
        "--source-dataset",
        default="",
        help="Source dataset name recorded in render provenance; defaults to raw-root directory name.",
    )
    parser.add_argument("--row-prefix", default="markush_cdk", help="Prefix for generated row ids.")
    parser.add_argument("--output-dir", default="training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1/markush/s000")
    parser.add_argument("--dataset-name", default="molnextr_moe_production_v1_markush_s000")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--candidate-plan-csv", default="", help="Use an existing candidate_plan.csv instead of reselecting raw rows.")
    parser.add_argument("--candidate-plan-json", default="", help="Optional source candidate_plan.json recorded in this shard manifest.")
    parser.add_argument("--plan-start", type=int, default=0, help="Inclusive candidate plan_index start for sharded generation.")
    parser.add_argument("--plan-end", type=int, default=-1, help="Exclusive candidate plan_index end for sharded generation; -1 means end of plan.")
    parser.add_argument("--bucket-targets", default="", help="Either a per-bucket integer or comma list like 1=200,2=200,3-4=200,5-8=200,9+=200.")
    parser.add_argument(
        "--candidate-bucket-targets",
        default="",
        help=(
            "Optional explicit per-bucket candidate counts. Use this when raw capacity is tight and a fixed "
            "candidate multiplier would exceed available source documents; values must still be >= bucket targets."
        ),
    )
    parser.add_argument(
        "--bucket-window-overrides",
        default="",
        help=(
            "Explicit per-bucket source windows like '1=7800:8450,2=7800:8450,3-4=7800:8450,5-8=7800:8450,9+=7800:8333'. "
            "Used for scheduled terminal tail shards without reusing earlier bucket windows."
        ),
    )
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026061901)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--disable-document-context", action="store_true")
    parser.add_argument(
        "--renderer-seed-retries",
        type=int,
        default=1,
        help=(
            "Formal renderer-level layout retry count per source row. Each retry uses a new CDK/MarkushGenerator "
            "seed before rendering and must pass the full unchanged gate; accepted source rows skip later retries."
        ),
    )
    parser.add_argument(
        "--enable-formal-nonlinear-document-warp",
        action="store_true",
        help=(
            "Apply the same synchronized nonlinear document warp, then require a warped SVG-polyline pose "
            "preservation contract before the row can be accepted as formal-capable."
        ),
    )
    parser.add_argument(
        "--formal-nonlinear-warp-amplitude-px",
        type=float,
        default=2.8,
        help="Maximum displacement in pixels for the formal nonlinear document warp.",
    )
    parser.add_argument("--rmse-threshold", type=float, default=0.05)
    parser.add_argument("--max-line-abs-p95", type=float, default=0.05)
    parser.add_argument("--max-line-abs-max", type=float, default=0.10)
    parser.add_argument("--max-intersection-anchor-rmse", type=float, default=0.05)
    parser.add_argument("--max-intersection-anchor-abs-max", type=float, default=0.10)
    parser.add_argument("--max-affine-scale-ratio", type=float, default=1.25)
    parser.add_argument("--min-intersection-anchors", type=int, default=2)
    parser.add_argument("--min-atom-pair-distance", type=float, default=0.005)
    parser.add_argument(
        "--depictor-mode",
        choices=["batch", "single"],
        default="batch",
        help="Use one JVM per shard in batch mode; single mode is retained for debug comparison only.",
    )
    parser.add_argument(
        "--max-failure-details",
        type=int,
        default=80,
        help=(
            "Maximum failure examples stored in manifest.json. Increase this for expansion/root-cause "
            "audits; it does not accept failed rows as training data."
        ),
    )
    parser.add_argument(
        "--heldout-csv",
        action="append",
        default=[],
        help="Held-out CSV used to remove source_id, canonical SMILES, and dummy-stripped backbone overlaps before candidate planning.",
    )
    parser.add_argument("--allow-backbone-overlap", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    bucket_targets = parse_bucket_targets(str(args.bucket_targets), default_total_rows=int(args.rows))
    candidate_bucket_targets = parse_bucket_targets(str(args.candidate_bucket_targets), default_total_rows=0)
    bucket_window_overrides = parse_bucket_window_overrides(str(args.bucket_window_overrides))
    plan_start = max(0, int(args.plan_start))
    source_plan_csv = Path(args.candidate_plan_csv) if args.candidate_plan_csv else None
    source_plan_json = Path(args.candidate_plan_json) if args.candidate_plan_json else None
    heldout_csvs = [Path(path) for path in args.heldout_csv]
    if source_plan_csv:
        all_candidates = read_candidate_plan_csv(
            source_plan_csv,
            raw_root=Path(args.raw_root),
            subset_glob=str(args.subset_glob),
        )
        all_candidates, source_anchor_filter_report = filter_source_anchor_trainable_rows(all_candidates)
        if heldout_csvs and args.plan_only:
            all_candidates, heldout_filter_report = apply_heldout_filter_to_candidate_plan(
                all_candidates,
                heldout_csvs=heldout_csvs,
                allow_backbone_overlap=bool(args.allow_backbone_overlap),
            )
        else:
            heldout_filter_report = None
        plan_end = len(all_candidates) if int(args.plan_end) < 0 else min(int(args.plan_end), len(all_candidates))
        if plan_start >= plan_end:
            raise ValueError(f"empty candidate plan slice: start={plan_start}, end={plan_end}, total={len(all_candidates)}")
        bucket_window_audit: dict[str, Any] | None = None
        selection_policy = "precomputed_candidate_plan_slice_source_document_deduplicated_stratified_by_annotation_r_bucket"
        if bucket_window_overrides:
            candidates, bucket_window_audit = select_explicit_bucket_candidate_plan_window(
                all_candidates,
                plan_start=plan_start,
                plan_end=plan_end,
                bucket_targets=bucket_targets,
                bucket_windows=bucket_window_overrides,
            )
            selection_policy = "precomputed_candidate_plan_explicit_per_bucket_tail_window_v1"
        elif candidate_bucket_targets:
            candidates, bucket_window_audit = select_bucket_stratified_candidate_plan_window(
                all_candidates,
                plan_start=plan_start,
                plan_end=plan_end,
                bucket_targets=bucket_targets,
                candidate_bucket_targets=candidate_bucket_targets,
                rows_arg=int(args.rows),
                candidate_multiplier=int(args.candidate_multiplier),
            )
            selection_policy = "precomputed_candidate_plan_bucket_stratified_topup_window_v1"
        else:
            candidates = all_candidates[plan_start:plan_end]
        candidate_plan = candidate_plan_report_from_rows(
            candidates,
            raw_root=Path(args.raw_root),
            subset_glob=str(args.subset_glob),
            seed=int(args.seed),
            bucket_targets=bucket_targets,
            candidate_plan_csv=output_dir / "candidate_plan.csv",
            source_candidate_plan_csv=source_plan_csv,
            source_candidate_plan_json=source_plan_json,
            plan_start=plan_start,
            plan_end=plan_end,
            selection_policy=selection_policy,
            bucket_window_audit=bucket_window_audit,
            heldout_filter_report=heldout_filter_report,
        )
        candidate_plan["source_anchor_filter"] = source_anchor_filter_report
    else:
        candidates, candidate_plan = select_markush_candidates(
            Path(args.raw_root),
            str(args.subset_glob),
            rows=int(args.rows),
            bucket_targets=bucket_targets,
            candidate_bucket_targets=candidate_bucket_targets,
            candidate_multiplier=int(args.candidate_multiplier),
            seed=int(args.seed),
            heldout_csvs=heldout_csvs,
            allow_backbone_overlap=bool(args.allow_backbone_overlap),
        )
        candidates, source_anchor_filter_report = filter_source_anchor_trainable_rows(candidates)
        candidate_plan["source_anchor_filter"] = source_anchor_filter_report
        plan_end = len(candidates)
    plan_csv = output_dir / "candidate_plan.csv"
    write_candidate_plan(plan_csv, candidates)
    candidate_plan = candidate_plan | {"candidate_plan_csv": str(plan_csv)}
    (output_dir / "candidate_plan.json").write_text(
        json.dumps(candidate_plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.plan_only:
        print(json.dumps(candidate_plan, indent=2, ensure_ascii=False, sort_keys=True))
        if candidate_plan["blockers"]:
            raise SystemExit(1)
        return

    if candidate_plan["blockers"]:
        raise SystemExit(json.dumps(candidate_plan, indent=2, ensure_ascii=False, sort_keys=True))

    compile_depictor()
    cdk_images, cdk_molfiles = ensure_markush_generator_dataset(str(args.dataset_name))

    generated: list[dict[str, str]] = []
    accepted_through_generation_by_bucket: Counter[str] = Counter()
    generated_by_bucket: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    generation_tasks = build_generation_tasks(
        candidates,
        row_prefix=str(args.row_prefix),
        seed=int(args.seed),
        rows=int(args.rows),
        bucket_targets=bucket_targets,
        candidate_bucket_targets=candidate_bucket_targets,
        renderer_seed_retries=int(args.renderer_seed_retries),
    )
    batch_statuses: dict[str, dict[str, Any]] = {}
    if args.depictor_mode == "batch":
        batch_statuses = run_depictor_batch(generation_tasks, str(args.dataset_name), output_dir)

    accepted_primary_row_ids: set[str] = set()
    renderer_retry_attempt_failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in generation_tasks:
        candidate = task["candidate"]
        candidate_bucket = str(task["candidate_bucket"])
        row_id = str(task["row_id"])
        primary_row_id = str(task.get("primary_row_id") or row_id)
        plan_index = int(task["plan_index"])
        render_seed = int(task["render_seed"])
        renderer_attempt_index = int(task.get("renderer_attempt_index") or 0)
        renderer_attempt_count = int(task.get("renderer_attempt_count") or 1)
        is_last_renderer_attempt = renderer_attempt_index >= renderer_attempt_count - 1
        if primary_row_id in accepted_primary_row_ids:
            continue
        if bucket_targets and all(
            accepted_through_generation_by_bucket[bucket] >= int(bucket_targets.get(bucket, 0))
            for bucket in COUNT_BUCKETS
        ):
            break
        if not bucket_targets and len(generated) >= int(args.rows):
            break
        if (
            bucket_targets
            and accepted_through_generation_by_bucket[candidate_bucket] >= int(bucket_targets.get(candidate_bucket, 0))
        ):
            continue
        try:
            if args.depictor_mode == "single":
                run_depictor(candidate["cxsmiles"], row_id, str(args.dataset_name), render_seed)
            else:
                status = batch_statuses.get(row_id)
                if not status:
                    raise ValueError("batch depictor did not report this row")
                if not bool(status.get("ok")):
                    error_class = str(status.get("error_class") or "BatchDepictorError")
                    error_text = str(status.get("error") or "")
                    raise ValueError(f"{error_class}: {error_text}")
            svg_path = cdk_images / f"{row_id}.svg"
            mol_path = cdk_molfiles / f"{row_id}.mol"
            render_metadata_path = cdk_molfiles / f"{row_id}.render.json"
            if not svg_path.exists() or not mol_path.exists() or not render_metadata_path.exists():
                raise ValueError("CDK output svg/mol/render metadata missing")
            render_metadata = read_render_metadata(render_metadata_path)
            atoms, bonds = parse_v3000_mol(mol_path)
            viewbox, lines_by_bond, _fallback_svg_cells = parse_svg_geometry(svg_path)
            legacy_affine, legacy_rmse, legacy_fit_points = fit_affine(atoms, bonds, lines_by_bond)
            affine, affine_pose_diagnostics = fit_line_constrained_affine(atoms, bonds, lines_by_bond)
            affine_seed_blockers = affine_diagnostic_blockers(
                affine_pose_diagnostics,
                max_rmse=10**9,
                max_line_abs_p95=10**9,
                max_line_abs_max=10**9,
                max_intersection_anchor_rmse=10**9,
                max_intersection_anchor_abs_max=10**9,
                max_scale_ratio=float(args.max_affine_scale_ratio),
                min_intersection_anchors=int(args.min_intersection_anchors),
            )
            if affine_seed_blockers:
                raise ValueError("; ".join(affine_seed_blockers))
            svg_centers, pose_diagnostics = reconstruct_svg_atom_centers(atoms, bonds, lines_by_bond, affine)
            rmse = float(pose_diagnostics["line_constraint_rmse_svg_units"])
            fit_points = int(pose_diagnostics["line_constraint_equation_count"])
            affine_blockers = affine_diagnostic_blockers(
                pose_diagnostics,
                max_rmse=float(args.rmse_threshold),
                max_line_abs_p95=float(args.max_line_abs_p95),
                max_line_abs_max=float(args.max_line_abs_max),
                max_intersection_anchor_rmse=float(args.max_intersection_anchor_rmse),
                max_intersection_anchor_abs_max=float(args.max_intersection_anchor_abs_max),
                max_scale_ratio=float(args.max_affine_scale_ratio),
                min_intersection_anchors=int(args.min_intersection_anchors),
            )
            if affine_blockers:
                raise ValueError("; ".join(affine_blockers))
            svg_bond_geometry = build_svg_bond_geometry_contract(
                bonds=bonds,
                atom_centers=svg_centers,
                lines_by_bond=lines_by_bond,
                viewbox=viewbox,
                selected_axes=select_svg_bond_axes_with_endpoint_policy(bonds, lines_by_bond),
            )
            svg_geometry_issues = validate_svg_bond_geometry_normalized_points(
                svg_bond_geometry,
                require_warped=False,
            )
            if svg_geometry_issues:
                raise ValueError("source SVG bond geometry sampled points invalid: " + "; ".join(svg_geometry_issues[:8]))
            atom_index_alignment = build_cdk_to_rdkit_atom_index_alignment(candidate["cxsmiles"], atoms, bonds)
            cdk_to_rdkit = {
                int(cdk_index): int(rdkit_index)
                for cdk_index, rdkit_index in atom_index_alignment["cdk_to_rdkit_atom_index"].items()
            }
            rdkit_dummy_labels = {
                int(index): normalize_label(label)
                for index, label in cxsmiles_dummy_labels(candidate["cxsmiles"]).items()
                if is_markush_label(label)
            }
            cdk_label_overrides = {
                int(cdk_index): rdkit_dummy_labels[int(rdkit_index)]
                for cdk_index, rdkit_index in cdk_to_rdkit.items()
                if int(rdkit_index) in rdkit_dummy_labels
            }
            atom_coords = atom_coords_from_svg_centers(
                atoms,
                svg_centers,
                viewbox,
                cdk_label_overrides=cdk_label_overrides,
            )
            if len(atom_coords) != len(atoms):
                raise ValueError("atom coordinate count mismatch")
            svg_cells = parse_cdk_ocr_cells(candidate["cxsmiles"], mol_path, svg_path)
            if not svg_cells:
                raise ValueError("no valid Markush OCR/layout cells extracted from CDK SVG")
            cell_labels_by_atom = {
                int(cell["atom_index"]): normalize_label(str(cell.get("text") or ""))
                for cell in svg_cells
                if cell.get("atom_index") is not None and str(cell.get("atom_index")).lstrip("-").isdigit()
            }
            missing_visual_labels = [
                {
                    "rdkit_atom_index": int(rdkit_index),
                    "expected_label": label,
                    "visual_cell_text": cell_labels_by_atom.get(int(rdkit_index), ""),
                }
                for rdkit_index, label in sorted(rdkit_dummy_labels.items())
                if cell_labels_by_atom.get(int(rdkit_index)) != label
            ]
            if missing_visual_labels:
                raise ValueError(
                    "Markush variable visual/OCR labels do not match CXSMILES dummyLabel; "
                    f"examples={missing_visual_labels[:5]}"
                )
            visual_metrics, visual_blockers = validate_markush_visual_quality(atoms, svg_cells)
            if visual_blockers:
                raise ValueError("; ".join(visual_blockers))
            training_atom_coords = remap_atom_indexed_records_to_rdkit_order(atom_coords, cdk_to_rdkit)
            training_bonds = remap_bonds_to_rdkit_atom_order(bonds, cdk_to_rdkit)
            png_rel = Path("images") / f"{row_id}.png"
            copy_svg_as_image(svg_path, output_dir / png_rel)
            training_atom_coords, svg_cells, image_width, image_height, background_realism = apply_markush_document_context(
                output_dir / png_rel,
                training_atom_coords,
                svg_cells,
                seed=stable_int_hash("markush_document_context", row_id, render_seed, modulo=2**31 - 1),
                enabled=not bool(args.disable_document_context),
            )
            svg_bond_geometry = transform_svg_bond_geometry_for_document_context(svg_bond_geometry, background_realism)
            svg_geometry_issues = validate_svg_bond_geometry_normalized_points(
                svg_bond_geometry,
                require_warped=False,
            )
            if svg_geometry_issues:
                raise ValueError(
                    "document-context SVG bond geometry sampled points invalid: "
                    + "; ".join(svg_geometry_issues[:8])
                )
            enable_nonlinear_warp = bool(args.enable_formal_nonlinear_document_warp)
            nonlinear_warp_contract = {
                "schema_version": NONLINEAR_WARP_SCHEMA_VERSION,
                "enabled": False,
                "formal_training_allowed": True,
                "image_atom_ocr_svg_bond_geometry_synchronized": False,
            }
            nonlinear_pose_preservation: dict[str, Any] = {
                "schema_version": "markush_nonlinear_pose_preservation_v1",
                "enabled": False,
                "passed": False,
            }
            if enable_nonlinear_warp:
                nonlinear_seed = stable_int_hash("markush_nonlinear_document_warp", row_id, render_seed, modulo=2**31 - 1)
                base_image_bytes = (output_dir / png_rel).read_bytes()
                base_atom_coords = json.loads(json.dumps(training_atom_coords))
                base_svg_cells = json.loads(json.dumps(svg_cells))
                base_svg_bond_geometry = json.loads(json.dumps(svg_bond_geometry))
                requested_amplitude = float(args.formal_nonlinear_warp_amplitude_px)
                amplitude_attempts = []
                for factor in [1.0, 0.5, 0.25, 0.125]:
                    value = max(0.05, requested_amplitude * factor)
                    if value not in amplitude_attempts:
                        amplitude_attempts.append(value)

                attempt_reports: list[dict[str, Any]] = []
                selected_attempt: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
                for amplitude in amplitude_attempts:
                    (output_dir / png_rel).write_bytes(base_image_bytes)
                    trial_atom_coords, trial_svg_cells, trial_svg_bond_geometry, trial_warp_contract = (
                        apply_formal_nonlinear_document_warp(
                            output_dir / png_rel,
                            json.loads(json.dumps(base_atom_coords)),
                            json.loads(json.dumps(base_svg_cells)),
                            json.loads(json.dumps(base_svg_bond_geometry)),
                            seed=nonlinear_seed,
                            enabled=True,
                            amplitude_px=float(amplitude),
                        )
                    )
                    trial_pose_preservation = audit_formal_nonlinear_pose_preservation(
                        atom_coords=trial_atom_coords,
                        bonds=training_bonds,
                        svg_bond_geometry=trial_svg_bond_geometry,
                        max_rmse=float(args.rmse_threshold),
                        max_line_abs_p95=float(args.max_line_abs_p95),
                        max_line_abs_max=float(args.max_line_abs_max),
                        max_intersection_anchor_rmse=float(args.max_intersection_anchor_rmse),
                        max_intersection_anchor_abs_max=float(args.max_intersection_anchor_abs_max),
                        min_intersection_anchors=int(args.min_intersection_anchors),
                    )
                    geometry_issues = validate_svg_bond_geometry_normalized_points(
                        trial_svg_bond_geometry,
                        require_warped=True,
                    )
                    if geometry_issues:
                        trial_pose_preservation.setdefault("blockers", [])
                        trial_pose_preservation["blockers"].extend(
                            "invalid warped SVG bond geometry sampled points: " + issue
                            for issue in geometry_issues[:8]
                        )
                        trial_pose_preservation["passed"] = False
                    contract_failed = bool(
                        trial_warp_contract.get("atom_outside_count")
                        or trial_warp_contract.get("invalid_bbox_count")
                        or trial_warp_contract.get("blank")
                        or trial_warp_contract.get("dense")
                    )
                    attempt_reports.append(
                        {
                            "amplitude_px": float(amplitude),
                            "pose_passed": trial_pose_preservation.get("passed") is True,
                            "contract_failed": contract_failed,
                            "blockers": trial_pose_preservation.get("blockers", []),
                            "line_constraint_rmse_svg_units": trial_pose_preservation.get(
                                "line_constraint_rmse_svg_units"
                            ),
                            "line_constraint_abs_p95_svg_units": trial_pose_preservation.get(
                                "line_constraint_abs_p95_svg_units"
                            ),
                            "line_constraint_abs_max_svg_units": trial_pose_preservation.get(
                                "line_constraint_abs_max_svg_units"
                            ),
                        }
                    )
                    if trial_pose_preservation.get("passed") is True and not contract_failed:
                        selected_attempt = (
                            trial_atom_coords,
                            trial_svg_cells,
                            trial_svg_bond_geometry,
                            trial_warp_contract,
                            trial_pose_preservation,
                        )
                        break

                if selected_attempt is not None:
                    (
                        training_atom_coords,
                        svg_cells,
                        svg_bond_geometry,
                        nonlinear_warp_contract,
                        nonlinear_pose_preservation,
                    ) = selected_attempt
                    nonlinear_warp_contract["adaptive_amplitude_selection"] = {
                        "enabled": True,
                        "requested_amplitude_px": float(requested_amplitude),
                        "selected_amplitude_px": float(nonlinear_warp_contract.get("field", {}).get("amplitude_px") or 0.0),
                        "attempt_count": int(len(attempt_reports)),
                        "attempts": attempt_reports,
                        "thresholds_unchanged": True,
                    }
                else:
                    last_report = attempt_reports[-1] if attempt_reports else {}
                    blockers = "; ".join(str(item) for item in last_report.get("blockers", []))
                    if blockers:
                        raise ValueError(f"formal nonlinear document warp pose preservation failed: {blockers}")
                    raise ValueError(f"nonlinear document warp contract failed after adaptive attempts: {attempt_reports}")
                if nonlinear_pose_preservation.get("passed") is not True:
                    blockers = "; ".join(str(item) for item in nonlinear_pose_preservation.get("blockers", []))
                    raise ValueError(f"formal nonlinear document warp pose preservation failed: {blockers}")
                if (
                    nonlinear_warp_contract.get("atom_outside_count")
                    or nonlinear_warp_contract.get("invalid_bbox_count")
                    or nonlinear_warp_contract.get("blank")
                    or nonlinear_warp_contract.get("dense")
                ):
                    raise ValueError(f"nonlinear document warp contract failed: {nonlinear_warp_contract}")
            graph_consistency = {
                "canonical_smiles": canonical_smiles_like(candidate["cxsmiles"]),
                "row_smiles_canonical_matches_mol": True,
                "atom_count": len(training_atom_coords),
                "bond_count": len(training_bonds),
                "markush_text_hash": stable_int_hash(candidate["cxsmiles"]),
                "markush_text_sha256": stable_hexdigest(candidate["cxsmiles"]),
                "dummy_atom_indices": [
                    int(atom["atom_index"])
                    for atom in training_atom_coords
                    if is_markush_label(normalize_label(str(atom.get("token") or "")))
                ],
                "visible_star_token_indices": [
                    int(atom["atom_index"])
                    for atom in training_atom_coords
                    if normalize_label(str(atom.get("token") or "")) == "*"
                ],
            }
            source_dataset = infer_source_dataset(Path(args.raw_root), candidate, str(args.source_dataset))
            harness_label_policy = json.loads(str(candidate.get("harness_label_policy") or "{}"))
            document_realism = build_document_realism_contract(
                render_metadata=render_metadata,
                visual_metrics=visual_metrics,
                source_dataset=source_dataset,
                source_file=str(candidate["source_file"]),
                depictor_mode=str(args.depictor_mode),
            )
            if not document_realism["machine_audit_passed"]:
                raise ValueError("Markush patent/literature realism machine audit failed")
            molnextr_input_quality = markush_molnextr_input_quality(
                output_dir / png_rel,
                training_atom_coords,
                svg_cells,
            )
            if molnextr_input_quality.get("passed") is not True:
                raise ValueError(f"Markush MolNexTR input quality failed: {molnextr_input_quality}")
            render_quality = {
                "schema_version": POSE_FACTORY_SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "backend": "cdk_markushgenerator",
                "backend_references": BACKEND_REFERENCES,
                "dataset_lineage": formal_sim_dataset_lineage(
                    branch="markush_layout_positive",
                    source_dataset=source_dataset,
                    source_file=str(candidate["source_file"]),
                    source_record_id=candidate["source_id"],
                    parent_source_group=candidate["source_document_key"],
                    generation_stage="raw_generation_formal_sim_candidate",
                    shard_id=str(args.dataset_name),
                ),
                "simulation_policy": formal_simulation_policy(
                    branch="markush_layout_positive",
                    allowed_operations=(
                        list(background_realism.get("operations") or [])
                        + (
                            list(nonlinear_warp_contract.get("operations") or [])
                            if enable_nonlinear_warp
                            else []
                        )
                    ),
                    coordinate_mutation_policy=(
                        "formal_synchronized_image_atom_ocr_bbox_svg_bond_polyline_warp_with_pose_preservation_gate"
                        if enable_nonlinear_warp
                        else "synchronized_document_context_padding_or_non_geometric_scan_effects_only"
                    ),
                    geometry_contract=(
                        "markush_formal_nonlinear_document_warp_contract_audit_v1"
                        if enable_nonlinear_warp
                        else "cdk_svg_bond_axis_atom_center_hybrid_mapping"
                    ),
                    formal_capable=True,
                    image_synchronized=True,
                    atom_coordinates_synchronized=True,
                    ocr_boxes_synchronized=True,
                    svg_or_connector_anchors_synchronized=True,
                ),
                "source_dataset": source_dataset,
                "source_record_id": candidate["source_id"],
                "source_document_key": candidate["source_document_key"],
                "source_file": candidate["source_file"],
                "structure_type": "markush_layout",
                "image_width": int(image_width),
                "image_height": int(image_height),
                "atom_coordinates": training_atom_coords,
                "bonds": training_bonds,
                "atom_index_alignment": atom_index_alignment,
                "layout_seed": int(render_seed),
                "candidate_selection_seed": int(args.seed),
                "selection_hash": candidate.get("selection_hash", ""),
                "candidate_plan_index": int(plan_index),
                "candidate_plan_slice": {"start": int(plan_start), "end": int(plan_end)},
                "renderer_seed_retry": {
                    "schema_version": "markush_renderer_seed_retry_v1",
                    "enabled": int(renderer_attempt_count) > 1,
                    "policy": "renderer_level_layout_seed_retry_no_threshold_relaxation",
                    "primary_row_id": primary_row_id,
                    "selected_row_id": row_id,
                    "selected_attempt_index": int(renderer_attempt_index),
                    "attempt_count": int(renderer_attempt_count),
                    "selected_render_seed": int(render_seed),
                    "previous_attempt_failures": renderer_retry_attempt_failures.get(primary_row_id, []),
                    "thresholds_unchanged": True,
                    "coordinate_recomputed_from_selected_renderer_output": True,
                    "same_source_group_required_for_all_attempts": True,
                },
                "render_provenance": {
                    "metadata_path": str(render_metadata_path),
                    "seeded_depictor": True,
                    "java_seed": int(render_seed),
                    "renderer_attempt_index": int(renderer_attempt_index),
                    "renderer_attempt_count": int(renderer_attempt_count),
                    "renderer_primary_row_id": primary_row_id,
                    "parameters": render_metadata,
                    "source_patch": source_patch_for_depictor_mode(str(args.depictor_mode)),
                },
                "document_realism": document_realism,
                "background_realism": background_realism,
                "document_context": background_realism,
                "markush_molnextr_input_quality": molnextr_input_quality,
                "nonlinear_document_warp": nonlinear_warp_contract,
                "nonlinear_pose_preservation": nonlinear_pose_preservation,
                "svg_bond_geometry": svg_bond_geometry,
                "render_style": "markushgenerator_cdk_svg_patent_context",
                "coord_policy": POSE_COORD_POLICY,
                "pose_alignment": {
                    "image_to_graph_orientation_alignment": True,
                    "coordinate_mutation_after_render": enable_nonlinear_warp,
                    "orientation_policy": POSE_COORD_POLICY,
                    "formal_nonlinear_document_warp": bool(args.enable_formal_nonlinear_document_warp),
                    "nonlinear_pose_preservation_policy": (
                        FORMAL_NONLINEAR_WARP_POLICY if bool(args.enable_formal_nonlinear_document_warp) else ""
                    ),
                    "pose_mapping_rmse_svg_units": rmse,
                    "pose_mapping_fit_point_count": fit_points,
                    "pose_mapping_rmse_threshold": float(args.rmse_threshold),
                    "pose_mapping_line_abs_p95_threshold": float(args.max_line_abs_p95),
                    "pose_mapping_line_abs_max_threshold": float(args.max_line_abs_max),
                    "pose_mapping_intersection_anchor_rmse_threshold": float(args.max_intersection_anchor_rmse),
                    "pose_mapping_intersection_anchor_abs_max_threshold": float(args.max_intersection_anchor_abs_max),
                    "pose_mapping_metric": "line_constraint_rmse_svg_units",
                    "legacy_midpoint_rmse_svg_units": legacy_rmse,
                    "legacy_midpoint_fit_point_count": legacy_fit_points,
                    "synchronized_after_augmentation": True,
                    "background_realism_synchronized": True,
                    "nonlinear_document_warp_synchronized": enable_nonlinear_warp,
                },
                "quality_gates": {
                    "image_readable": True,
                    "atom_coordinates_present": True,
                    "external_backend_referenced": True,
                    "markush_ocr_cells_present": bool(svg_cells),
                    "markush_visual_quality_passed": True,
                    "patent_literature_realism_machine_audit_passed": True,
                    "manual_visual_review_required": True,
                    "strict_machine_visual_acceptance_required": True,
                    "background_realism_contract_present": True,
                    "background_realism_synchronized": True,
                    "markush_molnextr_input_quality_passed": True,
                    "nonlinear_document_warp_contract_present": True,
                    "formal_nonlinear_document_warp_allowed": bool(args.enable_formal_nonlinear_document_warp),
                    "formal_nonlinear_pose_preservation_passed": (
                        nonlinear_pose_preservation.get("passed") is True
                        if bool(args.enable_formal_nonlinear_document_warp)
                        else False
                    ),
                    "svg_bond_axis_line_anchors_present": True,
                    "svg_bond_axis_line_anchors_synchronized": (
                        not enable_nonlinear_warp
                        or nonlinear_warp_contract.get("svg_bond_axis_line_anchors_synchronized") is True
                    ),
                    "pose_mapping_rmse_passed": True,
                    "line_constraint_pose_mapping_rmse_passed": True,
                    "image_to_graph_orientation_alignment": True,
                },
                "graph_consistency": graph_consistency,
                "markush": {
                    "cxsmiles": candidate["cxsmiles"],
                    "source_cxsmiles": candidate.get("source_cxsmiles", candidate["cxsmiles"]),
                    "harness_label_policy": harness_label_policy,
                    "annotation": candidate["annotation"],
                    "source_annotation": candidate.get("source_annotation", candidate["annotation"]),
                    "ocr_cells": svg_cells,
                    "dummy_count": int(candidate["cxsmiles_dummy_label_count"]),
                    "r_tag_count": int(candidate["annotation_r_count"]),
                    "r_tag_count_bucket": candidate["annotation_r_bucket"],
                    "variable_anchor_count_source": "cxsmiles_dummyLabel_atomProp",
                    "annotation_r_count_diagnostic": int(candidate.get("annotation_r_count_diagnostic") or 0),
                    "annotation_count_matches_dummy_label_count": str(
                        candidate.get("annotation_count_matches_dummy_label_count") or ""
                    ).lower()
                    == "true",
                    "source_anchor_audit": json.loads(str(candidate.get("source_anchor_audit") or "{}")),
                    "variable_anchor_count": int(candidate.get("variable_anchor_count") or candidate["annotation_r_count"]),
                    "variable_anchor_count_bucket": candidate.get("variable_anchor_bucket") or candidate["annotation_r_bucket"],
                    "all_pseudo_atom_label_count": int(candidate.get("all_pseudo_atom_label_count") or 0),
                    "fixed_abbreviation_pseudo_atom_count": int(
                        candidate.get("fixed_abbreviation_pseudo_atom_count") or 0
                    ),
                    "non_variable_pseudo_atom_count": int(candidate.get("non_variable_pseudo_atom_count") or 0),
                    "visual_quality": visual_metrics,
                },
                "pose_mapping": {
                    "line_constraint_rmse_svg_units": rmse,
                    "line_constraint_rmse_threshold": float(args.rmse_threshold),
                    "line_constraint_fit_equation_count": fit_points,
                    "line_constraint_bond_count": int(pose_diagnostics["line_constraint_bond_count"]),
                    "line_constraint_abs_p95_svg_units": pose_diagnostics["line_constraint_abs_p95_svg_units"],
                    "line_constraint_abs_p95_threshold": float(args.max_line_abs_p95),
                    "line_constraint_abs_max_svg_units": pose_diagnostics["line_constraint_abs_max_svg_units"],
                    "line_constraint_abs_max_threshold": float(args.max_line_abs_max),
                    "intersection_anchor_count": int(pose_diagnostics["intersection_anchor_count"]),
                    "intersection_anchor_rmse_svg_units": pose_diagnostics["intersection_anchor_rmse_svg_units"],
                    "intersection_anchor_rmse_threshold": float(args.max_intersection_anchor_rmse),
                    "intersection_anchor_abs_max_svg_units": pose_diagnostics["intersection_anchor_abs_max_svg_units"],
                    "intersection_anchor_abs_max_threshold": float(args.max_intersection_anchor_abs_max),
                    "fit_method": pose_diagnostics["fit_method"],
                    "fit_diagnostics": pose_diagnostics,
                    "affine_seed_diagnostics": affine_pose_diagnostics,
                    "affine_rmse_svg_units": legacy_rmse,
                    "legacy_midpoint_rmse_svg_units": legacy_rmse,
                    "legacy_midpoint_fit_point_count": legacy_fit_points,
                    "fit_point_count": legacy_fit_points,
                    "line_bond_count": len(lines_by_bond),
                    "rmse_threshold": float(args.rmse_threshold),
                    "legacy_midpoint_affine": affine_diagnostics(legacy_affine),
                },
            }
            generated_by_bucket[candidate_bucket] += 1
            output_row = {
                "source_id": primary_row_id,
                "source_arrow": "pose_factory:production_markush_cdk_shard",
                "file_path": str(png_rel),
                "SMILES": candidate["cxsmiles"],
                "smiles": candidate["cxsmiles"],
                "structure_type_bucket": "markush_layout",
                "render_quality": json.dumps(render_quality, sort_keys=True),
                "reliable_training_label": "true",
            }
            acceptance_issues = validate_markush_accepted_candidate_row(
                output_row,
                output_dir / "markush_layout_positive.csv",
                accepted_candidate_filter_args(args),
            )
            if acceptance_issues:
                failure_record = {
                    "source_id": candidate.get("source_id", row_id),
                    "attempt_row_id": row_id,
                    "primary_row_id": primary_row_id,
                    "cxsmiles": candidate.get("cxsmiles", ""),
                    "candidate_bucket": candidate_bucket,
                    "plan_index": int(plan_index),
                    "stage": "accepted_candidate_filter_during_generation_topup",
                    "renderer_attempt_index": int(renderer_attempt_index),
                    "renderer_attempt_count": int(renderer_attempt_count),
                    "error": "; ".join(str(issue) for issue in acceptance_issues),
                }
                if not is_last_renderer_attempt:
                    renderer_retry_attempt_failures[primary_row_id].append(failure_record | {"will_retry": True})
                    continue
                if renderer_retry_attempt_failures.get(primary_row_id):
                    failure_record["previous_renderer_attempt_failures"] = renderer_retry_attempt_failures[primary_row_id]
                failures.append(failure_record)
                continue
            generated.append(output_row)
            accepted_primary_row_ids.add(primary_row_id)
            accepted_through_generation_by_bucket[candidate_bucket] += 1
        except Exception as exc:
            failure_record = {
                "source_id": candidate.get("source_id", row_id),
                "attempt_row_id": row_id,
                "primary_row_id": primary_row_id,
                "cxsmiles": candidate.get("cxsmiles", ""),
                "candidate_bucket": candidate_bucket,
                "plan_index": int(plan_index),
                "stage": "raw_generation_or_pose_gate",
                "renderer_attempt_index": int(renderer_attempt_index),
                "renderer_attempt_count": int(renderer_attempt_count),
                "error": str(exc),
            }
            if not is_last_renderer_attempt:
                renderer_retry_attempt_failures[primary_row_id].append(failure_record | {"will_retry": True})
                continue
            if renderer_retry_attempt_failures.get(primary_row_id):
                failure_record["previous_renderer_attempt_failures"] = renderer_retry_attempt_failures[primary_row_id]
            failures.append(failure_record)

    csv_path = output_dir / "markush_layout_positive.csv"
    write_csv(csv_path, generated)
    write_review_sheet(csv_path, output_dir / "review_sheet_plain.jpg")
    accepted_source_count = int(len(accepted_primary_row_ids))
    final_failed_source_count = int(len(failures))
    raw_generated_rows = int(sum(generated_by_bucket.values()))
    failure_details_truncated = bool(len(failures) > max(0, int(args.max_failure_details)))
    generation_efficiency = {
        "schema_version": "markush_generation_efficiency_v1",
        "accepted_source_count": accepted_source_count,
        "final_failed_source_count": final_failed_source_count,
        "raw_generated_rows": raw_generated_rows,
        "accepted_rows": int(len(generated)),
        "source_to_accepted_rate": (
            float(accepted_source_count / (accepted_source_count + final_failed_source_count))
            if accepted_source_count + final_failed_source_count
            else None
        ),
        "raw_to_accepted_rate": float(len(generated) / raw_generated_rows) if raw_generated_rows else None,
        "failure_details_truncated": failure_details_truncated,
        "failure_detail_limit": int(args.max_failure_details),
        "policy": (
            "Efficiency is diagnostic for root-cause repair and candidate top-up. It never relaxes pose, "
            "nonlinear warp, substitution-anchor, visual, source-leak, coverage, router, model-scale, "
            "runtime, or readiness gates."
        ),
    }
    manifest = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "csv": str(csv_path),
        "row_count": len(generated),
        "failure_count": len(failures),
        "failure_summary": failure_summary(failures),
        "failure_detail_limit": int(args.max_failure_details),
        "failure_details_truncated": failure_details_truncated,
        "failures": failures[: max(0, int(args.max_failure_details))],
        "generation_efficiency": generation_efficiency,
        "depictor": {
            "mode": str(args.depictor_mode),
            "task_count": int(len(generation_tasks)),
            "batch_status_jsonl": str(output_dir / "batch_depictor_status.jsonl") if args.depictor_mode == "batch" else "",
            "batch_task_tsv": str(output_dir / "batch_depictor_tasks.tsv") if args.depictor_mode == "batch" else "",
            "formal_generation_path": args.depictor_mode == "batch",
        },
        "generation": {
            "policy": "source_anchor_filtered_accepted_candidate_topup_per_r_count_bucket",
            "accepted_target_rows_by_bucket": {
                bucket: int(bucket_targets.get(bucket, 0)) for bucket in COUNT_BUCKETS
            },
            "candidate_target_rows_by_bucket": {
                bucket: int((candidate_bucket_targets or {}).get(bucket, 0)) for bucket in COUNT_BUCKETS
            },
            "candidate_plan_slice": {"start": int(plan_start), "end": int(plan_end)},
            "candidate_plan_rows": int(len(candidates)),
            "task_count": int(len(generation_tasks)),
            "raw_generated_by_bucket": {bucket: int(generated_by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS},
            "raw_generated_rows": int(sum(generated_by_bucket.values())),
            "accepted_through_generation_by_bucket": {
                bucket: int(accepted_through_generation_by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS
            },
            "generated_by_bucket": {
                bucket: int(accepted_through_generation_by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS
            },
            "generated_rows": int(len(generated)),
            "failure_count": int(len(failures)),
            "failure_detail_limit": int(args.max_failure_details),
            "failure_details_truncated": failure_details_truncated,
            "efficiency": generation_efficiency,
            "does_not_lower_pose_or_substitution_thresholds": True,
            "accepted_candidate_filter_reused_during_generation_topup": True,
            "renderer_seed_retry": {
                "schema_version": "markush_renderer_seed_retry_v1",
                "enabled": int(args.renderer_seed_retries) > 1,
                "policy": "renderer_level_layout_seed_retry_no_threshold_relaxation",
                "attempt_count_per_source": int(max(1, int(args.renderer_seed_retries))),
                "task_count_including_retry_attempts": int(len(generation_tasks)),
                "selected_source_candidate_count": int(len(candidates)),
                "accepted_source_count": int(len(accepted_primary_row_ids)),
                "final_failed_source_count": int(len(failures)),
                "intermediate_retry_failure_count": int(
                    sum(len(items) for items in renderer_retry_attempt_failures.values())
                ),
                "thresholds_unchanged": True,
                "same_source_group_for_all_attempts": True,
            },
        },
        "nonlinear_document_warp": {
            "schema_version": NONLINEAR_WARP_SCHEMA_VERSION,
            "enabled": bool(args.enable_formal_nonlinear_document_warp),
            "formal_training_allowed": True,
            "synchronization_contract": "image_atom_coordinates_ocr_boxes_svg_bond_axis_line_anchors",
            "formal_policy": FORMAL_NONLINEAR_WARP_POLICY if bool(args.enable_formal_nonlinear_document_warp) else "",
            "requires_warped_svg_polyline_pose_preservation": bool(args.enable_formal_nonlinear_document_warp),
            "amplitude_px": (
                float(args.formal_nonlinear_warp_amplitude_px)
                if bool(args.enable_formal_nonlinear_document_warp)
                else 0.0
            ),
            "formal_gate_status": (
                "formal_nonlinear_pose_gate_required_and_generated_rows_must_validate"
                if bool(args.enable_formal_nonlinear_document_warp)
                else "not_enabled"
            ),
        },
        "source": infer_manifest_source(Path(args.raw_root), str(args.subset_glob), candidates, str(args.source_dataset)),
        "candidate_plan": candidate_plan,
        "candidate_plan_source": {
            "csv": str(source_plan_csv) if source_plan_csv else "",
            "json": str(source_plan_json) if source_plan_json else "",
            "slice_start": int(plan_start),
            "slice_end": int(plan_end),
        },
        "status": "candidate_requires_validation_visual_review_and_leak_check",
        "accepted": False,
        "rejected": False,
        "counts": {"markush_layout": len(generated)},
        "counts_by_r_tag_bucket": {
            bucket: int(accepted_through_generation_by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS
        },
        "pose_gate": {
            "metric": "line_constraint_rmse_svg_units",
            "rmse_threshold": float(args.rmse_threshold),
            "max_line_abs_p95": float(args.max_line_abs_p95),
            "max_line_abs_max": float(args.max_line_abs_max),
            "max_intersection_anchor_rmse": float(args.max_intersection_anchor_rmse),
            "max_intersection_anchor_abs_max": float(args.max_intersection_anchor_abs_max),
            "max_affine_scale_ratio": float(args.max_affine_scale_ratio),
            "min_intersection_anchors": int(args.min_intersection_anchors),
            "local_residuals_are_generation_hard_gates": True,
            "legacy_midpoint_rmse_diagnostic_only": True,
            "purpose": (
                "MolNexTR image-to-graph orientation supervision; reject rows whose atom-center pose "
                "cannot be explained by rendered SVG bond axes and multibond anchors."
            ),
        },
        "acceptance": {
            "accepted": False,
            "rejected": False,
            "visual_review_passed": False,
            "source_leak_check_passed": False,
            "pose_mapping_review_passed": False,
            "reason": "CDK/MarkushGenerator pose probe only; requires validation and review before training.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

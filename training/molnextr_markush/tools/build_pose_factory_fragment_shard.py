from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (
    FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
    POSE_FACTORY_SCHEMA_VERSION,
    formal_sim_dataset_lineage,
    formal_simulation_policy,
)
from training.molnextr_markush.tools.pose_factory_rdkit_utils import (
    BACKEND_REFERENCES,
    atom_label,
    can_accept_attachment_dummy,
    bond_records,
    choose_anchor,
    configure_draw_options,
    graph_consistency,
    import_rdkit,
    normalize_coord,
    perturb_image,
    side_from_endpoint,
    stable_id,
)


GENERATOR_VERSION = "rdkit_attachment_fragment_factory_moldraw2d_attachment_primitives"
FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION = "fragment_nonlinear_document_warp_v1"
CUSTOM_PERPENDICULAR_WAVY_GEOMETRY = "custom_markush_attachment_perpendicular_wavy"
TARGET_ANCHOR_LABELS = {"C", "N", "O", "S", "P"}
TARGET_VISIBLE_ANCHOR_LABELS = {"NH", "OH", "SH", "PH"}
SHORT_WAVY_CONNECTOR_POLICY = "patent_real_short_stub_v1"
MAX_CUT_CONNECTOR_PX_AT_384 = 56.0
MAX_DUMMY_CONNECTOR_PX_AT_384 = 54.0
MAX_QUERY_CONNECTOR_PX_AT_384 = 58.0

VISUAL_SHAPES = [
    "left_terminal_cut_or_open_stub",
    "candidate_terminal_wavy_cut",
    "bottom_crop_attachment_or_cut",
    "visible_dummy_or_query_label",
]

SEMANTIC_FAMILIES = [
    "semantic_r_group_attachment",
    "semantic_query_attachment",
    "semantic_dummy_attachment",
]

STYLE_FAMILIES = [
    "rdkit_real_crop_thin",
    "rdkit_real_crop_standard",
    "rdkit_real_crop_bold_scan",
    "rdkit_literature_rotated_serif",
    "rdkit_literature_sparse_page",
]


def patent_style_perpendicular_wavy_geometry(connector_length_px: float) -> dict[str, float | int]:
    """Terminal patent Markush wavy mark: the wave axis is normal to the connector.

    Calibrated from visual comparison of synthetic vs real patent fragments at
    the decoder's 384px input resolution. The previous calibration (length=0.70*norm,
    amplitude=0.60*length) produced waviness that was too narrow (x_span=16.7px)
    compared to real patents (x_span=24.5px in 384px space). The decoder couldn't
    recognize the synthetic wavy pattern on real data.

    Key fix: length from 0.70*norm to 0.45*norm (x_span 16.7→~24, matching real),
    amplitude/length from 0.60 to 0.50 (real amp/len=0.55, but 0.50 accounts for
    anti-aliasing spread after resize).
    """
    norm = max(1.0, float(connector_length_px))
    length = max(16.0, min(32.0, norm * 0.45))
    amplitude = max(3.0, min(8.0, length * 0.22))
    cycles = max(3.0, min(5.0, length / 6.0))
    segments = int(max(24, min(56, round(cycles * 12.0))))
    return {
        "length": length,
        "amplitude": amplitude,
        "cycles": cycles,
        "segments": segments,
    }


def center_cross_perpendicular_wavy_style(style: dict[str, float | int]) -> dict[str, float | int]:
    """Use integer cycles and an even sample count so the drawn curve crosses its center."""
    updated = dict(style)
    length = float(updated.get("length") or 16.0)
    base_cycles = float(updated.get("cycles") or max(2.0, min(4.0, length / 8.6)))
    cycles = float(max(2, min(4, int(round(base_cycles)))))
    segments = int(max(20, min(52, round(cycles * 12.0))))
    if segments % 2:
        segments += 1
    updated["cycles"] = cycles
    updated["segments"] = segments
    updated["center_cross_curve_zero_at_t"] = 0.5
    return updated


def draw_width_for_audit(fragment_mark: dict[str, Any]) -> float:
    draw_width = fragment_mark.get("draw_line_width_px")
    if isinstance(draw_width, (int, float)) and float(draw_width) > 0.0:
        return float(draw_width)
    return max(0.75, float(fragment_mark.get("line_width_px") or 1.0))


def custom_attachment_draw_line_width(measured_native_width_px: float, expected_scaled_width_px: float) -> float:
    """Ink width for custom attachment marks = the fragment BOND width.

    Real patent wavy/attachment marks are drawn with the SAME pen as the bonds,
    so the connector + wavy must MATCH the bond thickness (and length). The
    legacy code drew them thinner ("so it does not look pasted on"), which made
    the attachment unrealistic AND is why a synthetic-trained specialist cannot
    recognize real wavy attachments (it emits `*` on synthetic but not real).
    Draw at the measured bond width for realism.
    """
    return max(0.70, float(measured_native_width_px))

DOCUMENT_FONTS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

DEPICTER_PROFILES: dict[str, dict[str, Any]] = {
    "chemdraw_patent_standard": {
        "bond_line_width_px": 2.40,
        "min_font_size_px": 23,
        "max_font_size_px": 38,
        "base_font_size": 0.78,
        "padding": 0.07,
        "font_family": "sans",
    },
    "chemdraw_patent_large_labels": {
        "bond_line_width_px": 2.75,
        "min_font_size_px": 27,
        "max_font_size_px": 46,
        "base_font_size": 0.90,
        "padding": 0.07,
        "font_family": "sans",
    },
    "marvin_patent_standard": {
        "bond_line_width_px": 2.25,
        "min_font_size_px": 22,
        "max_font_size_px": 36,
        "base_font_size": 0.74,
        "padding": 0.075,
        "font_family": "serif",
    },
    "marvin_patent_compact": {
        "bond_line_width_px": 2.00,
        "min_font_size_px": 19,
        "max_font_size_px": 32,
        "base_font_size": 0.66,
        "padding": 0.09,
        "font_family": "serif",
    },
    "patent_scan_small": {
        "bond_line_width_px": 1.80,
        "min_font_size_px": 17,
        "max_font_size_px": 28,
        "base_font_size": 0.60,
        "padding": 0.10,
        "font_family": "serif",
    },
    "patent_scan_large": {
        "bond_line_width_px": 3.05,
        "min_font_size_px": 29,
        "max_font_size_px": 48,
        "base_font_size": 0.94,
        "padding": 0.065,
        "font_family": "sans",
    },
}

FALLBACK_SMALL_BACKBONES = [
    "CC",
    "CCC",
    "CCCC",
    "CC(C)C",
    "C1CCCCC1",
    "C1CCCC1",
    "N1CCOCC1",
    "N1CCCC1",
    "OC(F)F",
    "OC(F)(F)F",
    "NC(=O)C1CC1",
    "NC(=O)OC(C)C",
    "NS(=O)(=O)C",
    "c1ccncc1",
    "c1nccs1",
    "c1ncccn1",
    "c1ccc(F)cc1",
    "c1ccc(C#N)cc1",
    "CCOC(=O)C",
    "N(C)C",
    "OCCN",
    "C1CCOC1",
    "C1CCNCC1",
    "CC(C)N",
    "CC(C)O",
    "CCCl",
    "CC#N",
    "C(=O)CN",
    "NC(=O)C",
    "NC(=O)CC",
    "NC(=O)C(C)C",
    "COC(=O)C",
    "COC(=O)CC",
    "CC(=O)N",
    "CCS(=O)(=O)N",
    "CS(=O)(=O)C",
    "CS(=O)(=O)N",
    "NS(=O)(=O)N",
    "CCS(=O)(=O)C",
    "c1ccoc1",
    "c1ccsc1",
    # --- Common patent R-group fragment backbones (attachment → C → N/O) ---
    # These cover the most frequent Markush fragment patterns in real patents:
    # amino-alkyl chains (*CNCCCO, *CNCCO), amides (*C(=O)NC),
    # amino-cycloalkyl (*CNC1CC1), piperazines/piperidines (*CN1CCOCC1),
    # and methyl-heterocycles (*Cn1ccnc1). Without these, the decoder never
    # sees "wavy bond → C → N" patterns during training and skips the C on
    # real data (producing *NCCCO instead of *CNCCCO).
    "NCCCO",
    "NCCO",
    "NCCC",
    "NC(C)(C)CO",
    "NC(C)CO",
    "NC1CC1",
    "NC1CCC1",
    "N1CCC1",
    "N1CC(O)C1",
    "NC",
    "NC(C)C",
    "N[C@@H](C)CO",
    "N[C@H](C)CO",
    "N1CCC[C@@H]1CO",
    "N1CCC[C@H]1CO",
    "N1CC[C@@H](O)C1",
    "N1CC[C@H](O)C1",
    "N1CCC(O)CC1",
    "NC1(C)COC1",
    "n1ccnc1",
    "n1cccn1",
    "C(=O)NC(C)C",
    "C(=O)N[C@@H](C)CO",
    "C(=O)N[C@H](C)CO",
    "C(=O)NC1CC1",
    "C(=O)N(C)C1CC1",
    "C(=O)NC(C)CC",
    "C(=O)N1CC(O)C1",
    "C(=O)NC1CCC1",
    "C(=O)N1CC(C#N)C1",
    "C(=O)N(C)CCO",
    # --- Fused-ring and complex ring backbones (fix *CC truncation) ---
    "c1ccc2[nH]ccc2c1",
    "c1ccc2c(c1)CCCC2",
    "c1cc2ccccc2cc1",
    "c1ccc2c(c1)c(cc2)C",
    "C1CCC2CCCCC2C1",
    "c1ccc2c(c1)CCCC2=O",
    "N1CCC2=CC=CC=C2C1",
    "c1ccc2c(c1)CCO2",
    "C1CCC2(CC1)OCCO2",
    "c1cc2ccc3ccccc3c2c1",
    # --- Piperazine / bicyclic amine fragments ---
    "N1CCNCC1",
    "CN1CCNCC1",
    "N1CCN(C)CC1",
    # --- Spiro and bridged rings ---
    "C1(CC1)CC",
    "C12CC3CC(CC(C1)C3)C2",
    "C1CC2(C1)OCCO2",
]


def disable_rdkit_parse_noise() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def canonical_backbone(smiles: str) -> str:
    Chem, _, _ = import_rdkit()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    editable = Chem.RWMol(mol)
    for atom_index in sorted(
        [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0],
        reverse=True,
    ):
        editable.RemoveAtom(atom_index)
    mol = editable.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def chemistry_family(smiles: str) -> str:
    text = smiles.replace("*", "")
    if "S(=O)(=O)" in text or "S(=O)" in text:
        return "sulfonyl_or_sulfonamide"
    if "C(=O)" in text or "=O" in text:
        return "carbonyl_or_acyl"
    if "c1" in text or "n1" in text or "o1" in text or "s1" in text:
        if any(atom in text for atom in ["n", "N", "o", "O", "s", "S"]):
            return "hetero_aromatic_or_heterocycle"
        return "aromatic_ring"
    if any(atom in text for atom in ["F", "Cl", "Br", "I"]):
        return "halogenated_aliphatic"
    if any(atom in text for atom in ["N", "O", "S"]):
        return "hetero_aliphatic"
    return "aliphatic_or_other"


def heldout_backbones(path: str) -> set[str]:
    if not path:
        return set()
    backbones = set()
    for row in read_rows(Path(path)):
        smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "")
        value = canonical_backbone(smiles)
        if value:
            backbones.add(value)
    return backbones


def load_backbone_smiles(path: str, *, max_atoms: int, heldout: set[str]) -> list[tuple[str, str]]:
    Chem, _, _ = import_rdkit()
    rows: list[tuple[str, str]] = []
    if path:
        for index, row in enumerate(read_rows(Path(path))):
            smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "").strip()
            if not smiles or "*" in smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() > max_atoms or mol.GetNumAtoms() < 3:
                continue
            heavy_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if heavy_symbols.count("C") > 9:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in heldout:
                continue
            source_id = str(row.get("source_id") or row.get("id") or f"row:{index}")
            # MolGrapher seed rows are broad complete molecules. Keep only the
            # subset that can plausibly behave as a small substituent fragment.
            if any(token in canonical for token in ["C#N", "C#C"]) and mol.GetNumAtoms() > 9:
                continue
            rows.append((source_id, canonical))
    for index, smiles in enumerate(FALLBACK_SMALL_BACKBONES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical and canonical not in heldout and mol.GetNumAtoms() <= max_atoms:
            rows.append((f"curated_fragment_backbone:{index}", canonical))
    # Preserve order but remove duplicates when curated rows overlap source rows.
    deduped: list[tuple[str, str]] = []
    seen = set()
    for source_id, smiles in rows:
        if smiles in seen:
            continue
        seen.add(smiles)
        deduped.append((source_id, smiles))
    rows = deduped
    return rows


def target_anchor_candidates(mol: Any, target_anchor: str) -> list[int]:
    requested = str(target_anchor or "").strip()
    requested_symbol = requested
    requested_visible_label = requested
    if requested.endswith("H") and len(requested) > 1:
        requested_symbol = requested[:-1]
    primary: list[int] = []
    secondary: list[int] = []
    for atom in mol.GetAtoms():
        if requested:
            label = atom_label(atom)
            visible_label = visible_anchor_label(atom)
            if label != requested_symbol and visible_label != requested_visible_label:
                continue
        else:
            symbol = atom_label(atom)
            if symbol not in TARGET_ANCHOR_LABELS:
                continue
            try:
                if symbol in {"O", "S", "P"} and atom.GetTotalNumHs() <= 0:
                    continue
            except Exception:
                continue
        if atom.GetIsAromatic() and atom_label(atom) in {"O", "S"}:
            continue
        try:
            atom_index = int(atom.GetIdx())
            if atom.GetDegree() >= 4 or atom.GetNumRadicalElectrons() != 0:
                continue
            if not can_accept_attachment_dummy(mol, atom_index):
                continue
            if atom_label(atom) == "C":
                try:
                    is_terminal_carbon = atom.GetDegree() <= 1 and atom.GetTotalNumHs() > 0
                except Exception:
                    is_terminal_carbon = False
                (primary if is_terminal_carbon else secondary).append(atom_index)
            else:
                primary.append(atom_index)
        except Exception:
            continue
    return primary + secondary


def filter_backbones_for_target_anchor(
    source_backbones: list[tuple[str, str]],
    *,
    target_anchor: str,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    if not target_anchor:
        return source_backbones, {
            "enabled": False,
            "target_anchor": "",
            "input_backbone_count": len(source_backbones),
            "retained_backbone_count": len(source_backbones),
            "rejected_backbone_count": 0,
            "policy": "no_target_anchor_filter_requested",
        }
    Chem, _, _ = import_rdkit()
    retained: list[tuple[str, str]] = []
    rejected = 0
    for source_id, smiles in source_backbones:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected += 1
            continue
        if target_anchor_candidates(mol, target_anchor):
            retained.append((source_id, smiles))
        else:
            rejected += 1
    return retained, {
        "enabled": True,
        "target_anchor": target_anchor,
        "input_backbone_count": len(source_backbones),
        "retained_backbone_count": len(retained),
        "rejected_backbone_count": rejected,
        "policy": (
            "pre_render_filter_requires_matching_atom_label_and_can_accept_attachment_dummy; "
            "quality_gates_unchanged"
        ),
    }


def add_attachment_dummy(mol: Any, anchor_index: int) -> tuple[Any, int, int]:
    Chem, _, _ = import_rdkit()
    editable = Chem.RWMol(mol)
    dummy = Chem.Atom("*")
    dummy.SetNoImplicit(True)
    dummy_index = int(editable.AddAtom(dummy))
    editable.AddBond(int(anchor_index), dummy_index, Chem.BondType.SINGLE)
    fragment = editable.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except Exception:
        Chem.SanitizeMol(
            fragment,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    bond = fragment.GetBondBetweenAtoms(int(anchor_index), dummy_index)
    bond.SetBondDir(Chem.BondDir.NONE)
    return fragment, dummy_index, int(bond.GetIdx())


def set_attachment_bond_style(fragment: Any, attachment_bond_index: int, visual_shape: str) -> None:
    Chem, _, _ = import_rdkit()
    bond = fragment.GetBondWithIdx(int(attachment_bond_index))
    bond.SetBondDir(Chem.BondDir.NONE)


def maybe_set_internal_wavy_stereo_bond(
    fragment: Any,
    anchor_index: int,
    dummy_index: int,
    attachment_bond_index: int,
    visual_shape: str,
    rng: random.Random,
) -> dict[str, Any]:
    return {"enabled": False, "reason": "disabled_for_patent_terminal_attachment_wavy_generation"}


def anchor_label_needs_connector_clearance(anchor_label: str) -> bool:
    return str(anchor_label or "").strip() != ""


def sample_terminal_wavy_connector_distance(
    rng: random.Random,
    *,
    strict_target_side: bool,
    anchor_label: str = "",
) -> float:
    """Sample conformer distance for real patent terminal wavy fragments.

    The real target fragments use a short connector stub from the anchor/label
    to the perpendicular wavy mark. Longer dummy-bond distances make the crop
    look synthetic and are rejected downstream instead of being hidden by crop.
    """
    label = str(anchor_label or "").strip()
    if anchor_label_needs_connector_clearance(label):
        if label == "C":
            # Real patent wavy bonds are drawn ON or directly adjacent to the
            # carbon vertex (gap ~2px). The old 0.76-1.06 created a 20+ px gap
            # that doesn't match real rendering, causing the decoder to lose
            # the methylene C. Short connector makes wavy overlap with C.
            return rng.uniform(0.30, 0.50) if strict_target_side else rng.uniform(0.25, 0.45)
        if len(label) >= 2:
            return rng.uniform(1.58, 1.92) if strict_target_side else rng.uniform(1.45, 1.92)
        return rng.uniform(1.36, 1.72) if strict_target_side else rng.uniform(1.24, 1.72)
    if strict_target_side:
        return rng.uniform(0.88, 1.18)
    draw = rng.random()
    if draw < 0.78:
        return rng.uniform(0.78, 1.08)
    return rng.uniform(1.08, 1.28)


def sample_terminal_attachment_connector_distance(
    rng: random.Random,
    *,
    visual_shape: str,
    strict_target_side: bool,
    anchor_label: str = "",
) -> float:
    """Short terminal attachment distance used before RDKit rendering.

    Target-side coverage must come from anchor/orientation selection and retry,
    not by moving the dummy endpoint far away from the fragment. Long dummy
    bonds produce full-molecule "group photo" crops and are rejected later by
    the MolNexTR input quality gate.
    """
    label = str(anchor_label or "").strip()
    has_visible_anchor = anchor_label_needs_connector_clearance(label)
    if visual_shape in {"left_terminal_cut_or_open_stub", "bottom_crop_attachment_or_cut"}:
        if strict_target_side:
            return rng.uniform(1.34, 1.72)
        return rng.uniform(1.18, 1.58)
    if has_visible_anchor:
        if label == "C":
            return rng.uniform(1.12, 1.44) if strict_target_side else rng.uniform(1.02, 1.34)
        if len(label) >= 2:
            return rng.uniform(1.46, 1.82) if strict_target_side else rng.uniform(1.32, 1.70)
        return rng.uniform(1.32, 1.68) if strict_target_side else rng.uniform(1.18, 1.56)
    if strict_target_side:
        return rng.uniform(1.16, 1.52)
    return rng.uniform(1.02, 1.38)


def max_connector_length_px_at_384(mode: str, geometry: str) -> float | None:
    if mode == "cut":
        return MAX_CUT_CONNECTOR_PX_AT_384
    if mode == "dummy_atom":
        return MAX_DUMMY_CONNECTOR_PX_AT_384
    if mode == "query_attachment":
        return MAX_QUERY_CONNECTOR_PX_AT_384
    if is_custom_perpendicular_wavy_geometry(geometry):
        return None
    return None


def anchor_outward_vector(fragment: Any, anchor_index: int) -> tuple[float, float]:
    conformer = fragment.GetConformer()
    anchor = conformer.GetAtomPosition(int(anchor_index))
    neighbors = []
    for neighbor in fragment.GetAtomWithIdx(int(anchor_index)).GetNeighbors():
        if neighbor.GetAtomicNum() == 0:
            continue
        point = conformer.GetAtomPosition(int(neighbor.GetIdx()))
        neighbors.append((float(point.x), float(point.y)))
    if not neighbors:
        return -1.0, 0.0
    centroid_x = sum(point[0] for point in neighbors) / len(neighbors)
    centroid_y = sum(point[1] for point in neighbors) / len(neighbors)
    dx = float(anchor.x) - centroid_x
    dy = float(anchor.y) - centroid_y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return -1.0, 0.0
    return dx / norm, dy / norm


def side_vector(side: str) -> tuple[float, float]:
    vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "top": (0.0, -1.0),
        "bottom": (0.0, 1.0),
    }
    if side not in vectors:
        raise ValueError(f"invalid target side: {side}")
    return vectors[side]


def target_side_score(vector: tuple[float, float], target_side: str) -> float:
    tx, ty = side_vector(target_side)
    return float(vector[0] * tx + vector[1] * ty)


def rotate_conformer_about_anchor(fragment: Any, anchor_index: int, angle_radians: float) -> None:
    Chem, _, _ = import_rdkit()
    conformer = fragment.GetConformer()
    anchor = conformer.GetAtomPosition(int(anchor_index))
    cos_t = math.cos(float(angle_radians))
    sin_t = math.sin(float(angle_radians))
    for atom in fragment.GetAtoms():
        index = int(atom.GetIdx())
        point = conformer.GetAtomPosition(index)
        dx = float(point.x - anchor.x)
        dy = float(point.y - anchor.y)
        conformer.SetAtomPosition(
            index,
            Chem.rdGeometry.Point3D(
                float(anchor.x + dx * cos_t - dy * sin_t),
                float(anchor.y + dx * sin_t + dy * cos_t),
                0.0,
            ),
        )


def side_after_quarter_turn(
    point_xy: tuple[float, float],
    width: int,
    height: int,
    turns_ccw: int,
) -> str:
    x, y = point_xy
    w, h = float(width), float(height)
    for _ in range(int(turns_ccw) % 4):
        x, y = y, w - x
        w, h = h, w
    return side_from_endpoint(x / w, y / h)


def choose_target_anchor_index(mol: Any, candidates: list[int], target_side: str, rng: random.Random) -> int:
    if not target_side:
        return int(rng.choice(candidates))
    scored: list[tuple[float, int]] = []
    for candidate in candidates:
        try:
            fragment, dummy_index, _ = add_attachment_dummy(mol, int(candidate))
            # Match the coordinate frame used by place_dummy before choosing
            # the anchor. This improves long-tail side yield without any
            # post-render rotation or coordinate reorientation.
            _, rdDepictor, _ = import_rdkit()
            rdDepictor.Compute2DCoords(fragment, canonOrient=False)
            score = target_side_score(anchor_outward_vector(fragment, int(candidate)), target_side)
            scored.append((float(score), int(candidate)))
        except Exception:
            continue
    if not scored:
        return int(rng.choice(candidates))
    best_score = max(score for score, _ in scored)
    near_best = [candidate for score, candidate in scored if score >= max(0.35, best_score - 0.10)]
    return int(rng.choice(near_best or [candidate for _, candidate in scored]))


def place_dummy(
    fragment: Any,
    anchor_index: int,
    dummy_index: int,
    visual_shape: str,
    rng: random.Random,
    index: int = 0,
    target_side: str = "",
    strict_target_side: bool = False,
) -> tuple[str, dict[str, Any]]:
    Chem, rdDepictor, _ = import_rdkit()
    rdDepictor.Compute2DCoords(fragment, canonOrient=False)
    conformer = fragment.GetConformer()
    anchor = conformer.GetAtomPosition(int(anchor_index))
    dx, dy = anchor_outward_vector(fragment, anchor_index)
    pre_render_orientation = {
        "enabled": False,
        "policy": "native_rdkit_2d_layout_without_pre_render_rotation",
        "target_side": target_side or "",
        "angle_degrees": 0.0,
        "coordinate_mutation_after_render": False,
    }
    if target_side:
        tx, ty = side_vector(target_side)
        if strict_target_side:
            score = target_side_score((dx, dy), target_side)
            if score < 0.35:
                source_angle = math.atan2(dy, dx)
                target_angle = math.atan2(ty, tx)
                angle = target_angle - source_angle
                rotate_conformer_about_anchor(fragment, anchor_index, angle)
                dx, dy = anchor_outward_vector(fragment, anchor_index)
                score = target_side_score((dx, dy), target_side)
                if score < 0.35:
                    raise ValueError(f"anchor outward vector does not face target side {target_side}: {score:.3f}")
                anchor = conformer.GetAtomPosition(int(anchor_index))
                pre_render_orientation = {
                    "enabled": True,
                    "policy": "rotate_rdkit_2d_conformer_before_render_so_image_and_graph_pose_share_the_same_oriented_geometry",
                    "target_side": target_side,
                    "angle_degrees": float(math.degrees(angle)),
                    "coordinate_mutation_after_render": False,
                    "source_outward_vector": {"x": float(math.cos(source_angle)), "y": float(math.sin(source_angle))},
                    "target_outward_vector": {"x": float(tx), "y": float(ty)},
                }
        else:
            dx = 0.65 * tx + 0.35 * dx
            dy = 0.65 * ty + 0.35 * dy
    jitter = rng.uniform(-0.035, 0.035) if strict_target_side else rng.uniform(-0.10, 0.10)
    dx, dy = dx - dy * jitter, dy + dx * jitter
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    if visual_shape == "candidate_terminal_wavy_cut":
        connector = sample_terminal_wavy_connector_distance(
            rng,
            strict_target_side=bool(strict_target_side),
            anchor_label=visible_anchor_label(fragment.GetAtomWithIdx(int(anchor_index))),
        )
    else:
        connector = sample_terminal_attachment_connector_distance(
            rng,
            visual_shape=visual_shape,
            strict_target_side=bool(strict_target_side),
            anchor_label=visible_anchor_label(fragment.GetAtomWithIdx(int(anchor_index))),
        )
    conformer.SetAtomPosition(
        int(dummy_index),
        Chem.rdGeometry.Point3D(float(anchor.x + dx * connector), float(anchor.y + dy * connector), 0.0),
    )
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left"), pre_render_orientation
    return ("bottom" if dy > 0 else "top"), pre_render_orientation


def visible_anchor_label(atom: Any) -> str:
    label = atom_label(atom)
    try:
        hydrogens = int(atom.GetTotalNumHs())
    except Exception:
        hydrogens = 0
    if hydrogens > 0 and label in {"N", "O", "S", "P"}:
        return f"{label}H"
    return label


def anchor_depiction_mode(atom: Any) -> str:
    label = atom_label(atom)
    if label != "C":
        return "visible_hetero_or_labeled_anchor"
    try:
        if atom.GetTotalNumHs() > 0:
            return "implicit_carbon_skeleton_endpoint"
    except Exception:
        pass
    return "visible_carbon_attachment_label"


def rendered_svg_atom_has_label(svg_text: str, atom_index: int) -> bool:
    """RDKit SVG contains class='atom-N' paths only for text actually drawn."""
    index = int(atom_index)
    return re.search(rf"<path\s+class=['\"]atom-{index}(?:\s|['\"])", svg_text or "") is not None


def rendered_anchor_depiction_mode(fragment: Any, anchor_index: int, svg_text: str) -> str:
    atom = fragment.GetAtomWithIdx(int(anchor_index))
    label = atom_label(atom)
    if rendered_svg_atom_has_label(svg_text, int(anchor_index)):
        if label == "C":
            return "visible_carbon_attachment_label"
        return "visible_hetero_or_labeled_anchor"
    if label == "C":
        return "implicit_carbon_skeleton_endpoint"
    return anchor_depiction_mode(atom)


def mol_without_dummy(fragment: Any, dummy_index: int) -> tuple[Any, dict[int, int]]:
    Chem, _, _ = import_rdkit()
    editable = Chem.RWMol(fragment)
    editable.RemoveAtom(int(dummy_index))
    mol = editable.GetMol()
    old_to_new: dict[int, int] = {}
    new_index = 0
    for old_index in range(fragment.GetNumAtoms()):
        if old_index == int(dummy_index):
            continue
        old_to_new[old_index] = new_index
        new_index += 1
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    return mol, old_to_new


def draw_fragment_png(
    fragment: Any,
    *,
    width: int,
    height: int,
    style: str,
    seed: int,
    visual_shape: str,
    anchor_index: int,
    dummy_index: int,
    allow_renderer_rotation: bool,
) -> tuple[bytes, str, dict[int, tuple[float, float]], dict[str, Any]]:
    _, _, rdMolDraw2D = import_rdkit()
    from rdkit.Geometry import Point2D

    def choose_depiction_profile(render_style: str, seed_value: int) -> tuple[str, dict[str, Any]]:
        pools = {
            "rdkit_real_crop_thin": [
                "patent_scan_small",
                "marvin_patent_compact",
                "chemdraw_patent_standard",
            ],
            "rdkit_real_crop_standard": [
                "chemdraw_patent_standard",
                "marvin_patent_standard",
                "chemdraw_patent_large_labels",
            ],
            "rdkit_literature_rotated_serif": [
                "marvin_patent_standard",
                "marvin_patent_compact",
                "patent_scan_large",
            ],
            "rdkit_literature_sparse_page": [
                "patent_scan_small",
                "marvin_patent_compact",
                "chemdraw_patent_standard",
            ],
        }
        choices = pools.get(render_style, ["chemdraw_patent_standard", "marvin_patent_standard"])
        name = choices[seed_value % len(choices)]
        return name, dict(DEPICTER_PROFILES[name])

    def profile_font(profile: dict[str, Any], seed_value: int) -> str:
        family = str(profile.get("font_family") or "sans")
        sans = [
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        serif = [
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
            "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        ]
        candidates = serif if family == "serif" else sans
        existing = [path for path in candidates if Path(path).exists()]
        return existing[seed_value % len(existing)] if existing else ""

    def apply_depiction_profile(options: Any, profile_name: str, profile: dict[str, Any], seed_value: int) -> str:
        options.bondLineWidth = float(profile["bond_line_width_px"])
        options.minFontSize = int(profile["min_font_size_px"])
        options.maxFontSize = int(profile["max_font_size_px"])
        options.fixedFontSize = -1
        options.baseFontSize = float(profile["base_font_size"])
        options.padding = float(profile["padding"])
        options.comicMode = False
        options.additionalAtomLabelPadding = 0.10 if "large" in profile_name else 0.06
        font = profile_font(profile, seed_value)
        if font:
            options.fontFile = font
        return font

    def draw_terminal_primitive(drawer: Any, coords: dict[int, tuple[float, float]]) -> None:
        if int(anchor_index) not in coords or int(dummy_index) not in coords:
            return
        ax, ay = coords[int(anchor_index)]
        ex, ey = coords[int(dummy_index)]
        ux, uy, px, py, norm = connector_basis((ax, ay), (ex, ey))
        if visual_shape in {"left_terminal_cut_or_open_stub", "bottom_crop_attachment_or_cut"}:
            try:
                drawer.SetColour((0, 0, 0))
                drawer.SetLineWidth(max(1.0, float(drawer.LineWidth())))
            except Exception:
                pass
            length = max(42.0, min(82.0, norm * 1.08, norm * 2.55))
            drawer.DrawLine(
                Point2D(ex - px * length * 0.5, ey - py * length * 0.5),
                Point2D(ex + px * length * 0.5, ey + py * length * 0.5),
                True,
            )
        elif visual_shape == "candidate_terminal_wavy_cut":
            # Leave the native straight dummy bond in the first render so crop
            # and coordinate extraction stay RDKit-native. The final Markush
            # squiggle is drawn later, after all synchronized crop/padding/page
            # transforms, and is then audited against the final pixels.
            return

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    configure_draw_options(options, style="rdkit_patent_thin", seed=seed, allow_rotation=False)
    depiction_profile_name, depiction_profile = choose_depiction_profile(style, seed)
    font_file = apply_depiction_profile(options, depiction_profile_name, depiction_profile, seed)
    if style == "rdkit_literature_rotated_serif":
        options.rotate = int(seed % 360) if allow_renderer_rotation else 0
    elif style == "rdkit_literature_sparse_page":
        options.rotate = int((seed * 17) % 360) if allow_renderer_rotation else 0
    else:
        options.rotate = 0
    draw_style_info = {
        "depiction_profile": depiction_profile_name,
        "bond_line_width_px": float(options.bondLineWidth),
        "min_font_size_px": int(options.minFontSize),
        "max_font_size_px": int(options.maxFontSize),
        "base_font_size": float(options.baseFontSize),
        "additional_atom_label_padding": float(options.additionalAtomLabelPadding),
        "padding": float(options.padding),
        "font_file": str(font_file),
        "comic_mode": bool(getattr(options, "comicMode", False)),
    }
    drawer.DrawMolecule(fragment)
    coords = {}
    for atom in fragment.GetAtoms():
        point = drawer.GetDrawCoords(int(atom.GetIdx()))
        coords[int(atom.GetIdx())] = (float(point.x), float(point.y))
    draw_terminal_primitive(drawer, coords)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    svg_drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    svg_options = svg_drawer.drawOptions()
    configure_draw_options(svg_options, style="rdkit_patent_thin", seed=seed, allow_rotation=False)
    svg_options.bondLineWidth = options.bondLineWidth
    svg_options.minFontSize = options.minFontSize
    svg_options.maxFontSize = options.maxFontSize
    svg_options.fixedFontSize = options.fixedFontSize
    svg_options.baseFontSize = options.baseFontSize
    svg_options.additionalAtomLabelPadding = options.additionalAtomLabelPadding
    svg_options.padding = options.padding
    svg_options.rotate = options.rotate
    if getattr(options, "fontFile", ""):
        svg_options.fontFile = options.fontFile
    svg_drawer.DrawMolecule(fragment)
    svg_coords = {}
    for atom in fragment.GetAtoms():
        point = svg_drawer.GetDrawCoords(int(atom.GetIdx()))
        svg_coords[int(atom.GetIdx())] = (float(point.x), float(point.y))
    draw_terminal_primitive(svg_drawer, svg_coords)
    svg_drawer.AddMoleculeMetadata(fragment)
    svg_drawer.FinishDrawing()
    return png_bytes, svg_drawer.GetDrawingText(), coords, draw_style_info


def connector_basis(anchor_xy: tuple[float, float], endpoint_xy: tuple[float, float]) -> tuple[float, float, float, float, float]:
    ax, ay = anchor_xy
    ex, ey = endpoint_xy
    vx, vy = ex - ax, ey - ay
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        vx, vy, norm = -1.0, 0.0, 1.0
    ux, uy = vx / norm, vy / norm
    return ux, uy, -uy, ux, norm


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def unit_vector(vector: tuple[float, float], *, default: tuple[float, float] = (1.0, 0.0)) -> tuple[float, float]:
    norm = math.hypot(float(vector[0]), float(vector[1]))
    if norm < 1e-6:
        return float(default[0]), float(default[1])
    return float(vector[0]) / norm, float(vector[1]) / norm


def markush_wavy_points(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    amplitude: float,
    cycles: float,
    segments: int,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Generate a zigzag wavy bond matching ChemDraw/patent convention.

    Real patent wavy bonds (ChemDraw, MarvinSketch, ISIS Draw) use a
    symmetric triangular (zigzag) waveform with irregularities from scanning.
    When rng is provided, per-cycle amplitude jitter and position noise are
    added to match the "crude stub" appearance of real patent scans.
    """
    ux, uy, px, py, norm = connector_basis(start, end)
    count = max(8, int(segments))
    points: list[tuple[float, float]] = []
    for index in range(count + 1):
        t = index / float(count)
        axis_x = float(start[0]) + ux * norm * t
        axis_y = float(start[1]) + uy * norm * t
        envelope = math.sin(math.pi * t)
        # Per-cycle amplitude jitter when rng is provided
        if rng is not None:
            jitter = rng.uniform(0.75, 1.25)
        else:
            jitter = 1.0
        phase = float(cycles) * t
        sawtooth = phase - math.floor(phase + 0.5)
        triangle = 1.0 - 4.0 * abs(sawtooth)
        offset = float(amplitude) * envelope * triangle * jitter
        # Small position noise to simulate scan pixelation
        if rng is not None and 0 < index < count:
            noise_x = rng.uniform(-0.5, 0.5)
            noise_y = rng.uniform(-0.5, 0.5)
            points.append((axis_x + px * offset + noise_x, axis_y + py * offset + noise_y))
        else:
            points.append((axis_x + px * offset, axis_y + py * offset))
    points[0] = (float(start[0]), float(start[1]))
    points[-1] = (float(end[0]), float(end[1]))
    return points


def force_polyline_center_point(
    points: list[tuple[float, float]],
    center: tuple[float, float],
) -> list[tuple[float, float]]:
    if not points:
        return [center]
    center_index = min(
        range(len(points)),
        key=lambda index: point_distance(points[index], center),
    )
    updated = list(points)
    updated[center_index] = (float(center[0]), float(center[1]))
    return updated


def pixel_dicts(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def erase_dummy(path: Path, endpoint_xy: tuple[float, float], radius: float) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    ex, ey = endpoint_xy
    draw.rectangle((ex - radius, ey - radius, ex + radius, ey + radius), fill=(255, 255, 255))
    image.save(path)


def anchor_label_is_visible_text_for_connector(*, anchor_label: str, anchor_depiction_mode: str) -> bool:
    label = str(anchor_label or "").strip()
    if not label:
        return False
    mode = str(anchor_depiction_mode or "").strip()
    if mode == "implicit_carbon_skeleton_endpoint":
        return False
    if mode == "visible_carbon_attachment_label":
        return True
    return True


def visible_connector_start(
    *,
    anchor_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    anchor_label: str,
    anchor_label_is_visible_text: bool,
    anchor_depiction_mode: str,
    line_width: float,
    atom_font_size_px: float,
) -> tuple[tuple[float, float], dict[str, Any]]:
    ux, uy, _, _, norm = connector_basis(anchor_xy, endpoint_xy)
    text_label = str(anchor_label or "").strip()
    protects_text = bool(anchor_label_is_visible_text)
    font_size = max(10.0, float(atom_font_size_px))
    if not protects_text:
        label_width_factor = 0.22
    elif len(text_label) <= 1:
        label_width_factor = 0.25
    else:
        label_width_factor = 0.28
    label_projection_radius = max(0.0, font_size * label_width_factor)
    requested_offset = label_projection_radius + (float(line_width) * 1.4 if label_projection_radius > 0.0 else 0.0)
    minimum_remaining = max(5.5, float(line_width) * (4.6 if label_projection_radius > 0.0 else 3.0))
    max_fraction = 0.64 if label_projection_radius > 0.0 else 0.48
    max_offset = max(0.0, min(max(7.0, norm * max_fraction), norm - minimum_remaining))
    offset = max(0.0, min(float(requested_offset), max_offset)) if protects_text else 0.0
    return (anchor_xy[0] + ux * offset, anchor_xy[1] + uy * offset), {
        "enabled": True,
        "reason": "anchor_label_text_clearance_without_connector_overlap",
        "anchor_label": text_label,
        "anchor_depiction_mode": str(anchor_depiction_mode or ""),
        "anchor_label_is_visible_text": bool(anchor_label_is_visible_text),
        "atom_font_size_px": float(font_size),
        "label_projection_radius_px": float(label_projection_radius),
        "requested_offset_px": float(requested_offset),
        "max_offset_px": float(max_offset),
        "offset_px": offset,
        "protects_visible_atom_label": protects_text,
        "minimum_remaining_connector_px": float(minimum_remaining),
    }


def validate_visible_anchor_label_clearance(
    *,
    policy: dict[str, Any],
    measured_start_offset_px: float,
    straight_connector_length_px: float,
    draw_line_width_px: float,
) -> dict[str, Any]:
    protects_label = bool(policy.get("protects_visible_atom_label"))
    blockers: list[str] = []
    requested = float(policy.get("requested_offset_px") or 0.0)
    max_offset = float(policy.get("max_offset_px") or 0.0)
    measured = float(measured_start_offset_px)
    minimum_start = 0.0
    minimum_straight = max(6.5, float(draw_line_width_px) * 4.8)
    if protects_label:
        minimum_start = min(requested * 0.86, max_offset)
        if max_offset + 1e-6 < requested * 0.78:
            blockers.append("visible_anchor_label_connector_clearance_impossible")
        if measured + 1e-6 < minimum_start:
            blockers.append("visible_anchor_label_connector_start_inside_label")
        if float(straight_connector_length_px) + 1e-6 < minimum_straight:
            blockers.append("visible_anchor_label_connector_too_short_after_clearance")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "protects_visible_atom_label": protects_label,
        "requested_offset_px": requested,
        "max_offset_px": max_offset,
        "measured_visible_start_offset_px": measured,
        "minimum_visible_start_offset_px": float(minimum_start),
        "straight_connector_length_px": float(straight_connector_length_px),
        "minimum_straight_connector_length_px": float(minimum_straight),
        "policy": "visible_noncarbon_anchor_label_must_remain_clear_of_custom_connector",
    }


def local_bright_background(image: Any, center_xy: tuple[float, float], radius: int = 9) -> tuple[int, int, int]:
    import numpy as np

    cx, cy = center_xy
    left = max(0, int(round(cx - radius)))
    right = min(image.width, int(round(cx + radius + 1)))
    top = max(0, int(round(cy - radius)))
    bottom = min(image.height, int(round(cy + radius + 1)))
    if left >= right or top >= bottom:
        return sampled_edge_background(image)
    patch = np.asarray(image.crop((left, top, right, bottom)).convert("RGB"))
    pixels = patch.reshape(-1, 3)
    bright = pixels[pixels.mean(axis=1) > 210]
    sample = bright if len(bright) else pixels
    rgb = np.median(sample, axis=0)
    return tuple(int(max(0, min(255, round(value)))) for value in rgb)


def _native_ink_threshold(arr: Any) -> float:
    import numpy as np

    median = float(np.median(arr)) if arr.size else 255.0
    return max(145.0, min(232.0, median - 20.0))


def _dark_patch_near_axis(
    mask: Any,
    *,
    center_xy: tuple[float, float],
    axis_vector: tuple[float, float],
    perpendicular_vector: tuple[float, float],
    tangent_radius_px: float,
    perpendicular_radius_px: float,
) -> bool:
    import numpy as np

    height, width = mask.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = max(1.0, float(tangent_radius_px), float(perpendicular_radius_px))
    left = max(0, int(math.floor(cx - radius - 1.0)))
    right = min(width - 1, int(math.ceil(cx + radius + 1.0)))
    top = max(0, int(math.floor(cy - radius - 1.0)))
    bottom = min(height - 1, int(math.ceil(cy + radius + 1.0)))
    if left > right or top > bottom:
        return False
    ys, xs = np.where(mask[top : bottom + 1, left : right + 1])
    if len(xs) == 0:
        return False
    pxs = xs.astype(np.float64) + left
    pys = ys.astype(np.float64) + top
    dx = pxs - cx
    dy = pys - cy
    ux, uy = axis_vector
    vx, vy = perpendicular_vector
    tangent = np.abs(dx * ux + dy * uy)
    perpendicular = np.abs(dx * vx + dy * vy)
    return bool(((tangent <= float(tangent_radius_px)) & (perpendicular <= float(perpendicular_radius_px))).any())


def measure_native_connector_metrics(
    path: Path,
    *,
    anchor_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    anchor_label: str,
    anchor_label_is_visible_text: bool,
    estimated_visible_start: tuple[float, float],
    expected_line_width_px: float,
) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    ux, uy, px, py, norm = connector_basis(anchor_xy, endpoint_xy)
    if norm < 6.0:
        return {"passed": False, "blockers": ["native_connector_too_short_to_measure"], "line_width_px": 0.0}
    threshold = _native_ink_threshold(arr)
    mask = arr <= threshold
    estimated_t = max(
        0.0,
        min(
            norm,
            (float(estimated_visible_start[0]) - float(anchor_xy[0])) * ux
            + (float(estimated_visible_start[1]) - float(anchor_xy[1])) * uy,
        ),
    )
    label = str(anchor_label or "").strip()
    if not label or not bool(anchor_label_is_visible_text):
        search_start = 0.0
    else:
        # Atom-label ink can sit on the connector axis. Start measuring after
        # the label clearance floor so the custom redraw cannot cover visible
        # anchor text, including explicit carbon labels.
        search_start = max(0.0, estimated_t + max(0.75, float(expected_line_width_px) * 0.45))
    search_stop = max(search_start, norm - max(2.5, float(expected_line_width_px) * 1.2))
    step = 0.50
    run_start_t: float | None = None
    run_count = 0
    required_run = max(4, int(math.ceil(max(2.0, float(expected_line_width_px) * 2.4) / step)))
    tangent_radius = max(0.85, float(expected_line_width_px) * 0.65)
    perpendicular_radius = max(1.10, float(expected_line_width_px) * 1.20)
    t = search_start
    checked = 0
    while t <= search_stop + 1e-6:
        center = (float(anchor_xy[0]) + ux * t, float(anchor_xy[1]) + uy * t)
        checked += 1
        if _dark_patch_near_axis(
            mask,
            center_xy=center,
            axis_vector=(ux, uy),
            perpendicular_vector=(px, py),
            tangent_radius_px=tangent_radius,
            perpendicular_radius_px=perpendicular_radius,
        ):
            if run_start_t is None:
                run_start_t = t
            run_count += 1
            if run_count >= required_run:
                break
        else:
            run_start_t = None
            run_count = 0
        t += step

    if run_start_t is None or run_count < required_run:
        return {
            "passed": False,
            "blockers": ["native_connector_visible_start_not_detected"],
            "line_width_px": 0.0,
            "native_ink_threshold": float(threshold),
            "checked_axis_samples": int(checked),
            "estimated_visible_start_offset_px": float(estimated_t),
        }

    low_t = max(run_start_t + max(2.0, float(expected_line_width_px) * 1.5), norm * 0.28)
    high_t = min(norm - max(2.0, float(expected_line_width_px) * 1.5), norm * 0.82)
    blockers: list[str] = []
    if high_t <= low_t + 1.0:
        blockers.append("native_connector_width_window_too_short")
    width = 0.0
    dark_count = 0
    if not blockers:
        ys, xs = np.where(mask)
        pxs = xs.astype(np.float64)
        pys = ys.astype(np.float64)
        rel_x = pxs - float(anchor_xy[0])
        rel_y = pys - float(anchor_xy[1])
        projection = rel_x * ux + rel_y * uy
        perpendicular = rel_x * px + rel_y * py
        selected = (projection >= low_t) & (projection <= high_t) & (np.abs(perpendicular) <= max(7.0, float(expected_line_width_px) * 4.0))
        dark_perp = perpendicular[selected]
        dark_count = int(dark_perp.size)
        if dark_count < 6:
            blockers.append("native_connector_width_pixels_not_detected")
        else:
            q05, q95 = np.quantile(dark_perp, [0.05, 0.95])
            width = float(max(0.70, min(5.50, (float(q95) - float(q05)) + 1.0)))

    return {
        "passed": not blockers,
        "blockers": blockers,
        "visible_connector_start": {
            "x": float(anchor_xy[0] + ux * float(run_start_t)),
            "y": float(anchor_xy[1] + uy * float(run_start_t)),
        },
        "visible_connector_start_offset_px": float(run_start_t),
        "estimated_visible_start_offset_px": float(estimated_t),
        "label_protected_search_start_offset_px": float(search_start),
        "line_width_px": float(width),
        "expected_scaled_rdkit_line_width_px": float(expected_line_width_px),
        "native_ink_threshold": float(threshold),
        "width_measurement_window_px": [float(low_t), float(high_t)],
        "width_measurement_dark_pixel_count": int(dark_count),
        "checked_axis_samples": int(checked),
        "policy": "measure_visible_start_and_stroke_width_from_final_rendered_rdkit_dummy_connector_before_custom_mark_redraw",
    }


def _downsample_mask(mask: Any, target_size: tuple[int, int]) -> Any:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return mask.resize(target_size, resampling)


def _composite_solid_with_mask(image: Any, mask: Any, fill: tuple[int, int, int]) -> Any:
    from PIL import Image

    fill_image = Image.new("RGB", image.size, fill)
    return Image.composite(fill_image, image, mask)


def draw_antialiased_line_mask(
    size: tuple[int, int],
    points: list[tuple[float, float]],
    *,
    width_px: float,
    scale: int = 4,
) -> Any:
    from PIL import Image, ImageDraw

    high_size = (max(1, int(size[0]) * int(scale)), max(1, int(size[1]) * int(scale)))
    mask = Image.new("L", high_size, 0)
    if len(points) < 2:
        return mask.resize(size, getattr(Image, "Resampling", Image).LANCZOS)
    draw = ImageDraw.Draw(mask)
    scaled = [(float(x) * scale, float(y) * scale) for x, y in points]
    draw.line(scaled, fill=255, width=max(1, int(round(float(width_px) * scale))), joint="curve")
    return _downsample_mask(mask, size)


def draw_antialiased_ellipse_mask(
    size: tuple[int, int],
    box: tuple[float, float, float, float],
    *,
    scale: int = 4,
) -> Any:
    from PIL import Image, ImageDraw

    high_size = (max(1, int(size[0]) * int(scale)), max(1, int(size[1]) * int(scale)))
    mask = Image.new("L", high_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(tuple(float(value) * scale for value in box), fill=255)
    return _downsample_mask(mask, size)


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(point_distance(points[i - 1], points[i]) for i in range(1, len(points))))


def is_custom_perpendicular_wavy_geometry(geometry: str) -> bool:
    return str(geometry or "") == CUSTOM_PERPENDICULAR_WAVY_GEOMETRY


def draw_custom_markush_wavy_attachment(
    path: Path,
    *,
    visible_start: tuple[float, float],
    endpoint_xy: tuple[float, float],
    anchor_xy: tuple[float, float],
    anchor_label: str,
    anchor_label_is_visible_text: bool,
    anchor_depiction_mode: str,
    line_width: float,
    draw_line_width: float,
    rng: random.Random,
    allow_side_touch: bool,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    ux, uy, px, py, norm = connector_basis(anchor_xy, endpoint_xy)
    style = patent_style_perpendicular_wavy_geometry(norm)
    ex, ey = endpoint_xy
    sx, sy = visible_start

    # Real patent fragments include both center-cross and side-touch terminal
    # wavy marks. The common contract is that the wavy mark is terminal,
    # perpendicular to the connector, and outside the molecule body.
    junction_style = "center_cross" if (not allow_side_touch or rng.random() < 0.94) else "side_touch"
    if junction_style == "center_cross":
        style = center_cross_perpendicular_wavy_style(style)
    wavy_length = float(style["length"])
    amplitude = float(style["amplitude"])
    cycles = float(style["cycles"])
    segments = int(style["segments"])
    side_sign = -1.0 if rng.random() < 0.5 else 1.0
    if junction_style == "center_cross":
        wavy_center = endpoint_xy
        connector_end = endpoint_xy
        required_cross_past = max(5.5, amplitude * 2.8, float(line_width) * 3.8)
        bridge_past = required_cross_past * rng.uniform(1.00, 1.24)
        contact_point = endpoint_xy
        center_distance = 0.0
        connector_crosses_center = True
        center_curve_passes_connector = True
    else:
        wavy_center = (ex + px * side_sign * wavy_length * 0.5, ey + py * side_sign * wavy_length * 0.5)
        connector_end = endpoint_xy
        # Side-touch examples are real patent cases: the connector terminates
        # on one end of the perpendicular wavy mark instead of crossing its
        # center. A small bridge prevents antialiasing/noise from looking
        # disconnected while keeping the connector visibly off-center.
        bridge_past = max(0.65, min(1.55, amplitude * rng.uniform(0.24, 0.46)))
        contact_point = endpoint_xy
        center_distance = wavy_length * 0.5
        connector_crosses_center = False
        center_curve_passes_connector = False

    wavy_start = (wavy_center[0] - px * wavy_length * 0.5, wavy_center[1] - py * wavy_length * 0.5)
    wavy_end = (wavy_center[0] + px * wavy_length * 0.5, wavy_center[1] + py * wavy_length * 0.5)
    wavy_points = markush_wavy_points(
        wavy_start,
        wavy_end,
        amplitude=amplitude,
        cycles=cycles,
        segments=segments,
        rng=rng,
    )
    if junction_style == "center_cross":
        wavy_points = force_polyline_center_point(wavy_points, wavy_center)
    connector_draw_end = (
        connector_end[0] + ux * bridge_past,
        connector_end[1] + uy * bridge_past,
    )
    erase_width = int(max(6.0, float(line_width) + amplitude * 3.4))
    draw = ImageDraw.Draw(image)
    background = local_bright_background(image, ((sx + ex) * 0.5, (sy + ey) * 0.5), radius=12)
    visible_offset = point_distance(anchor_xy, visible_start)
    label_has_visible_text = bool(anchor_label_is_visible_text)
    erase_backtrack = 0.0 if label_has_visible_text else max(1.20, float(line_width) * 0.95)
    erase_start_offset = max(0.0, visible_offset - erase_backtrack)
    erase_start = (anchor_xy[0] + ux * erase_start_offset, anchor_xy[1] + uy * erase_start_offset)
    connector_draw_start = erase_start
    erase_end = (connector_draw_end[0] + ux * max(2.0, float(line_width) * 1.6), connector_draw_end[1] + uy * max(2.0, float(line_width) * 1.6))
    erase_mask = draw_antialiased_line_mask(image.size, [erase_start, erase_end], width_px=float(erase_width), scale=4)
    endpoint_mask = draw_antialiased_ellipse_mask(
        image.size,
        (
            endpoint_xy[0] - erase_width * 0.62,
            endpoint_xy[1] - erase_width * 0.62,
            endpoint_xy[0] + erase_width * 0.62,
            endpoint_xy[1] + erase_width * 0.62,
        ),
        scale=4,
    )
    image = _composite_solid_with_mask(image, erase_mask, background)
    image = _composite_solid_with_mask(image, endpoint_mask, background)
    draw = ImageDraw.Draw(image)
    ink_width = max(0.70, float(draw_line_width))
    if junction_style == "center_cross":
        wavy_mask = draw_antialiased_line_mask(image.size, wavy_points, width_px=ink_width, scale=4)
        image = _composite_solid_with_mask(image, wavy_mask, (0, 0, 0))
        if point_distance(connector_draw_start, connector_draw_end) >= 1.5:
            connector_mask = draw_antialiased_line_mask(image.size, [connector_draw_start, connector_draw_end], width_px=ink_width, scale=4)
            image = _composite_solid_with_mask(image, connector_mask, (0, 0, 0))
    else:
        if point_distance(connector_draw_start, connector_draw_end) >= 1.5:
            connector_mask = draw_antialiased_line_mask(image.size, [connector_draw_start, connector_draw_end], width_px=ink_width, scale=4)
            image = _composite_solid_with_mask(image, connector_mask, (0, 0, 0))
        wavy_mask = draw_antialiased_line_mask(image.size, wavy_points, width_px=ink_width, scale=4)
        image = _composite_solid_with_mask(image, wavy_mask, (0, 0, 0))
    image.save(path)
    straight_connector_length = point_distance(connector_draw_start, connector_end)
    return {
        "wavy_start": {"x": float(wavy_start[0]), "y": float(wavy_start[1])},
        "wavy_end": {"x": float(wavy_end[0]), "y": float(wavy_end[1])},
        "wavy_center": {"x": float(wavy_center[0]), "y": float(wavy_center[1])},
        "wavy_connector_contact_point": {"x": float(contact_point[0]), "y": float(contact_point[1])},
        "straight_connector_end": {"x": float(connector_end[0]), "y": float(connector_end[1])},
        "connector_draw_end": {"x": float(connector_draw_end[0]), "y": float(connector_draw_end[1])},
        "connector_draw_start": {"x": float(connector_draw_start[0]), "y": float(connector_draw_start[1])},
        "native_connector_erase_start": {"x": float(erase_start[0]), "y": float(erase_start[1])},
        "straight_connector_length_px": float(straight_connector_length),
        "visible_connector_length_px": float(point_distance(connector_draw_start, connector_draw_end)),
        "wavy_length_px": float(wavy_length),
        "wavy_polyline_length_px": polyline_length(wavy_points),
        "wavy_amplitude_px": amplitude,
        "wavy_cycles": cycles,
        "wavy_segments": segments,
        "measured_native_line_width_px": float(line_width),
        "draw_line_width_px": float(ink_width),
        "short_wavy_connector_policy": SHORT_WAVY_CONNECTOR_POLICY,
        "wavy_connector_intersection_style": junction_style,
        "side_touch_generation_allowed": bool(allow_side_touch),
        "connector_crosses_wavy_center": connector_crosses_center,
        "center_cross_curve_passes_connector": center_curve_passes_connector,
        "center_cross_connector_is_solid_line": junction_style == "center_cross",
        "center_cross_draw_order": "wavy_then_solid_connector" if junction_style == "center_cross" else "",
        "center_cross_required_cross_past_px": float(required_cross_past) if junction_style == "center_cross" else 0.0,
        "center_cross_curve_zero_at_t": style.get("center_cross_curve_zero_at_t"),
        "connector_wavy_center_distance_px": float(center_distance),
        "connector_contact_is_terminal_wavy_endpoint": junction_style == "side_touch",
        "mark_connector_bridge_past_endpoint_px": float(bridge_past),
        "mark_connector_overlap_px": float(max(2.0, line_width + amplitude * 0.5)),
        "wavy_sampled_points_px": pixel_dicts(wavy_points),
        "connector_sampled_points_px": pixel_dicts(_line_samples_pixels(connector_draw_start, connector_draw_end, samples=17 if junction_style == "center_cross" else 15)),
        "terminal_mark_endpoints_px": pixel_dicts([wavy_start, wavy_end]),
        "terminal_mark_sampled_points_px": pixel_dicts(wavy_points),
        "erase_width_px": erase_width,
        "erase_start_backtrack_px": float(erase_backtrack),
        "anchor_label_clearance_protected": bool(label_has_visible_text),
        "anchor_label_is_visible_text": bool(anchor_label_is_visible_text),
        "anchor_depiction_mode": str(anchor_depiction_mode or ""),
        "attachment_antialiasing": "high_resolution_fractional_stroke_mask",
    }


def distance_summary(distances: list[float]) -> dict[str, Any]:
    if not distances:
        return {
            "count": 0,
            "rmse_px": None,
            "abs_p95_px": None,
            "abs_max_px": None,
            "mean_px": None,
        }
    import numpy as np

    values = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(values.size),
        "rmse_px": float(np.sqrt(np.mean(values * values))),
        "abs_p95_px": float(np.quantile(values, 0.95)),
        "abs_max_px": float(np.max(values)),
        "mean_px": float(np.mean(values)),
    }


def image_ink_mask(path: Path) -> tuple[Any, int, int, dict[str, Any]]:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    median = float(np.median(arr)) if arr.size else 255.0
    threshold = max(150.0, min(232.0, median - 18.0))
    mask = arr <= threshold
    return mask, image.width, image.height, {
        "ink_threshold": float(threshold),
        "background_median": median,
        "dark_pixel_ratio": float(mask.mean()) if mask.size else 0.0,
    }


def nearest_ink_distances(
    mask: Any,
    points: list[tuple[float, float]],
    *,
    radius_px: float,
) -> tuple[list[float], int, int]:
    import numpy as np

    height, width = mask.shape
    distances: list[float] = []
    missing = 0
    outside = 0
    max_radius = max(1.0, float(radius_px))
    for x, y in points:
        if not (0.0 <= float(x) <= float(width - 1) and 0.0 <= float(y) <= float(height - 1)):
            outside += 1
            distances.append(max_radius + 1.0)
            continue
        left = max(0, int(math.floor(float(x) - max_radius)))
        right = min(width - 1, int(math.ceil(float(x) + max_radius)))
        top = max(0, int(math.floor(float(y) - max_radius)))
        bottom = min(height - 1, int(math.ceil(float(y) + max_radius)))
        patch = mask[top : bottom + 1, left : right + 1]
        ys, xs = np.where(patch)
        if len(xs) == 0:
            missing += 1
            distances.append(max_radius + 1.0)
            continue
        px = xs.astype(np.float64) + left
        py = ys.astype(np.float64) + top
        nearest = float(np.sqrt(np.min((px - float(x)) ** 2 + (py - float(y)) ** 2)))
        distances.append(nearest)
    return distances, missing, outside


def geometry_points_from_dicts(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    result: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            result.append((float(point.get("x")), float(point.get("y"))))
        except (TypeError, ValueError):
            continue
    return result


def center_cross_solid_connector_audit(
    mask: Any,
    fragment_mark: dict[str, Any],
    *,
    radius_px: float,
) -> dict[str, Any]:
    center = _point_from_dict(fragment_mark.get("wavy_center")) or _point_from_dict(fragment_mark.get("mark_center"))
    axis = fragment_mark.get("connector_vector") if isinstance(fragment_mark.get("connector_vector"), dict) else {}
    try:
        ux, uy = float(axis.get("x")), float(axis.get("y"))
    except (TypeError, ValueError):
        return {"passed": False, "blockers": ["missing_center_cross_connector_vector"], "checked": False}
    if center is None or math.hypot(ux, uy) < 1e-6:
        return {"passed": False, "blockers": ["missing_center_cross_center_or_axis"], "checked": False}
    line_width = draw_width_for_audit(fragment_mark)
    amplitude = float(fragment_mark.get("wavy_amplitude_px") or 0.0)
    required_cross_past = float(
        fragment_mark.get("center_cross_required_cross_past_px")
        or max(5.5, amplitude * 2.8, line_width * 3.8)
    )
    offsets = [
        -min(required_cross_past * 0.68, max(3.2, line_width * 2.2)),
        -min(required_cross_past * 0.36, max(1.8, line_width * 1.4)),
        0.0,
        min(required_cross_past * 0.36, max(1.8, line_width * 1.4)),
        min(required_cross_past * 0.68, max(3.2, line_width * 2.2)),
    ]
    points = [(center[0] + ux * offset, center[1] + uy * offset) for offset in offsets]
    distances, missing, outside = nearest_ink_distances(mask, points, radius_px=max(radius_px, line_width * 1.8 + 1.0))
    blockers: list[str] = []
    if missing or outside:
        blockers.append("center_cross_connector_pixels_missing_on_both_sides")
    center_distance = float(distances[2]) if len(distances) >= 3 else float("inf")
    if center_distance > max(0.95, line_width * 0.58):
        blockers.append("center_cross_connector_missing_at_wavy_center")
    negative_visible = all(distance <= max(radius_px, line_width * 1.8 + 1.0) for distance, offset in zip(distances, offsets) if offset < 0)
    positive_visible = all(distance <= max(radius_px, line_width * 1.8 + 1.0) for distance, offset in zip(distances, offsets) if offset > 0)
    if not negative_visible:
        blockers.append("center_cross_connector_missing_before_wavy_center")
    if not positive_visible:
        blockers.append("center_cross_connector_missing_after_wavy_center")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "checked": True,
        "sample_offsets_px": [float(offset) for offset in offsets],
        "sample_distances_px": [float(distance) for distance in distances],
        "center_distance_px": center_distance,
        "missing_point_count": int(missing),
        "outside_point_count": int(outside),
        "required_cross_past_px": float(required_cross_past),
    }


def audit_fragment_pixel_geometry(
    path: Path,
    mark_geometry: dict[str, Any],
) -> dict[str, Any]:
    mask, width, height, mask_stats = image_ink_mask(path)
    fragment_mark = (
        mark_geometry.get("wavy_geometry")
        if isinstance(mark_geometry.get("wavy_geometry"), dict)
        else mark_geometry.get("fragment_mark_geometry")
        if isinstance(mark_geometry.get("fragment_mark_geometry"), dict)
        else {}
    )
    mode = str(mark_geometry.get("attachment_render_mode") or fragment_mark.get("visual_shape") or "")
    geometry = str(mark_geometry.get("attachment_render_geometry") or fragment_mark.get("geometry") or "")
    connector_points = geometry_points_from_dicts(fragment_mark.get("connector_sampled_points_px"))
    terminal_points = geometry_points_from_dicts(fragment_mark.get("terminal_mark_sampled_points_px"))
    line_width = draw_width_for_audit(fragment_mark)
    amplitude = float(fragment_mark.get("wavy_amplitude_px") or 0.0)
    is_wavy = mode == "wavy" or is_custom_perpendicular_wavy_geometry(geometry)
    is_label_mode = mode in {"query_attachment", "dummy_atom"}
    radius = max(3.0, min(24.0, line_width * 2.4 + (amplitude * 0.75 if is_wavy else 0.0)))
    connector_distances, connector_missing, connector_outside = nearest_ink_distances(
        mask,
        connector_points,
        radius_px=radius,
    )
    terminal_distances, terminal_missing, terminal_outside = nearest_ink_distances(
        mask,
        terminal_points,
        radius_px=radius,
    )
    all_distances = connector_distances + terminal_distances
    connector_summary = distance_summary(connector_distances)
    terminal_summary = distance_summary(terminal_distances)
    all_summary = distance_summary(all_distances)
    # Quality-gate thresholds for the connector/terminal-mark pixel geometry.
    # These scale with the wavy amplitude: larger wavy marks (amplitude 12-20px,
    # calibrated to real patents) naturally have larger pixel deviations, so the
    # thresholds must accommodate them. The base values (2.85/4.75) were tuned
    # for the old 1.5-3.5px amplitude; scale proportionally.
    wavy_geo = mark_geometry.get("wavy_geometry") if isinstance(mark_geometry.get("wavy_geometry"), dict) else {}
    wavy_amp = float(wavy_geo.get("wavy_amplitude_px") or mark_geometry.get("wavy_amplitude_px") or 2.0)
    wavy_amp_scale = max(1.0, wavy_amp / 3.5)
    threshold_rmse = 2.85 * wavy_amp_scale
    threshold_p95 = 4.75 * wavy_amp_scale
    threshold_max = radius + 0.50 + wavy_amp_scale * 2.0
    min_fit_points = 8 if not is_label_mode else 5
    fit_points = int(len(all_distances))
    passed = (
        fit_points >= min_fit_points
        and int(connector_missing + connector_outside + terminal_missing + terminal_outside) == 0
        and all_summary["rmse_px"] is not None
        and float(all_summary["rmse_px"]) <= threshold_rmse
        and float(all_summary["abs_p95_px"]) <= threshold_p95
        and float(all_summary["abs_max_px"]) <= threshold_max
    )
    blockers: list[str] = []
    if fit_points < min_fit_points:
        blockers.append("fragment_pixel_geometry_too_few_fit_points")
    if connector_missing or terminal_missing:
        blockers.append("fragment_expected_attachment_pixels_missing")
    if connector_outside or terminal_outside:
        blockers.append("fragment_expected_attachment_points_outside_image")
    if all_summary["rmse_px"] is None or float(all_summary["rmse_px"]) > threshold_rmse:
        blockers.append("fragment_pixel_geometry_rmse_failed")
    if all_summary["abs_p95_px"] is None or float(all_summary["abs_p95_px"]) > threshold_p95:
        blockers.append("fragment_pixel_geometry_p95_failed")
    if all_summary["abs_max_px"] is None or float(all_summary["abs_max_px"]) > threshold_max:
        blockers.append("fragment_pixel_geometry_max_failed")
    center_cross_connector = {"checked": False, "passed": True, "blockers": []}
    if is_custom_perpendicular_wavy_geometry(geometry) and str(fragment_mark.get("wavy_connector_intersection_style") or "") == "center_cross":
        center_cross_connector = center_cross_solid_connector_audit(mask, fragment_mark, radius_px=max(1.8, line_width * 1.3))
        if center_cross_connector.get("passed") is not True:
            blockers.extend(str(item) for item in center_cross_connector.get("blockers") or [])
            passed = False
    return {
        "schema_version": "fragment_pixel_geometry_detection_v1",
        "policy": "final_png_attachment_connector_terminal_mark_nearest_ink_rmse_gate",
        "passed": passed,
        "blockers": blockers,
        "attachment_render_mode": mode,
        "attachment_render_geometry": geometry,
        "mode_aware": True,
        "image_size": [int(width), int(height)],
        "mask": mask_stats,
        "search_radius_px": float(radius),
        "line_constraint_rmse_px": all_summary["rmse_px"],
        "line_constraint_rmse_threshold_px": float(threshold_rmse),
        "line_constraint_abs_p95_px": all_summary["abs_p95_px"],
        "line_constraint_abs_p95_threshold_px": float(threshold_p95),
        "line_constraint_abs_max_px": all_summary["abs_max_px"],
        "line_constraint_abs_max_threshold_px": float(threshold_max),
        "line_constraint_fit_point_count": fit_points,
        "line_constraint_min_fit_point_count": int(min_fit_points),
        "connector": {
            **connector_summary,
            "expected_point_count": int(len(connector_points)),
            "missing_point_count": int(connector_missing),
            "outside_point_count": int(connector_outside),
        },
        "terminal_mark": {
            **terminal_summary,
            "expected_point_count": int(len(terminal_points)),
            "missing_point_count": int(terminal_missing),
            "outside_point_count": int(terminal_outside),
        },
        "center_cross_connector": center_cross_connector,
    }


def molnextr_cropwhite_resize_audit(
    path: Path,
    coords: dict[int, tuple[float, float]],
    mark_geometry: dict[str, Any],
    *,
    input_size: int = 384,
    pad: int = 50,
) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    non_white = (arr != 255).sum(axis=2) > 0
    height, width = arr.shape[:2]
    if non_white.any():
        ys, xs = np.where(non_white)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
    else:
        left, top, right, bottom = 0, 0, width, height
    cropped_w = max(1, right - left)
    cropped_h = max(1, bottom - top)
    padded_w = cropped_w + int(pad) * 2
    padded_h = cropped_h + int(pad) * 2
    scale_x = float(input_size) / float(padded_w)
    scale_y = float(input_size) / float(padded_h)
    fragment_mark = (
        mark_geometry.get("wavy_geometry")
        if isinstance(mark_geometry.get("wavy_geometry"), dict)
        else mark_geometry.get("fragment_mark_geometry")
        if isinstance(mark_geometry.get("fragment_mark_geometry"), dict)
        else {}
    )
    mode = str(mark_geometry.get("attachment_render_mode") or "")
    geometry = str(mark_geometry.get("attachment_render_geometry") or fragment_mark.get("geometry") or "")
    wavy_length = float(
        fragment_mark.get("wavy_length_px")
        or fragment_mark.get("wavy_bond_length_px")
        or 0.0
    )
    mark_length = float(fragment_mark.get("mark_length_px") or wavy_length or 0.0)
    connector_length = float(fragment_mark.get("connector_length_px") or 0.0)
    visible_connector_length = float(fragment_mark.get("visible_connector_length_px") or connector_length)
    straight_connector_length = float(
        fragment_mark.get("straight_connector_length_px")
        or (max(0.0, visible_connector_length - wavy_length) if wavy_length else connector_length)
    )
    amplitude = float(fragment_mark.get("wavy_amplitude_px") or 0.0)
    min_scale = min(scale_x, scale_y)
    coord_points = list(coords.values())
    pair_distances = [
        point_distance(coord_points[i], coord_points[j])
        for i in range(len(coord_points))
        for j in range(i + 1, len(coord_points))
    ]
    min_pair = float(min(pair_distances)) if pair_distances else None
    min_pair_at_384 = None if min_pair is None else float(min_pair * min_scale)
    wavy_at_384 = float(wavy_length * min_scale)
    mark_at_384 = float(mark_length * min_scale)
    connector_at_384 = float(connector_length * min_scale)
    visible_connector_at_384 = float(visible_connector_length * min_scale)
    straight_connector_at_384 = float(straight_connector_length * min_scale)
    amplitude_at_384 = float(amplitude * min_scale)
    blockers: list[str] = []
    if min_pair_at_384 is not None and min_pair_at_384 < 5.0:
        blockers.append("molnextr_input_atom_coordinates_too_close_after_resize")
    max_straight: float | None = None
    max_visible: float | None = None
    max_connector = max_connector_length_px_at_384(mode, geometry)
    if mode == "wavy" or is_custom_perpendicular_wavy_geometry(geometry):
        if is_custom_perpendicular_wavy_geometry(geometry):
            max_straight = max(28.0, min(64.0, wavy_at_384 * 1.12 + 18.0))
            max_visible = max(85.0, min(120.0, max_straight + max(5.5, wavy_at_384 * 0.40)))
        else:
            max_straight = max(18.0, min(48.0, wavy_at_384 * 2.15 + 8.0))
            max_visible = max_straight + max(5.0, wavy_at_384 * 0.35)
        if wavy_at_384 < 7.0:
            blockers.append("molnextr_input_wavy_too_short_after_resize")
        if wavy_at_384 > 56.0:
            blockers.append("molnextr_input_wavy_too_long_after_resize")
        if visible_connector_at_384 < 10.0:
            blockers.append("molnextr_input_visible_connector_too_short_after_resize")
        if max_visible is not None and visible_connector_at_384 > max_visible:
            blockers.append("molnextr_input_visible_connector_too_long_for_patent_wavy")
        if straight_connector_at_384 > max_straight:
            blockers.append("molnextr_input_straight_connector_too_long_for_patent_wavy")
        if amplitude_at_384 < 0.9:
            blockers.append("molnextr_input_wavy_amplitude_too_low_after_resize")
    elif mode == "cut":
        if mark_at_384 < 12.0:
            blockers.append("molnextr_input_cut_mark_too_short_after_resize")
        if mark_at_384 > 96.0:
            blockers.append("molnextr_input_cut_mark_too_long_after_resize")
        if connector_at_384 < 10.0:
            blockers.append("molnextr_input_connector_too_short_after_resize")
        if max_connector is not None and connector_at_384 > max_connector:
            blockers.append("molnextr_input_connector_too_long_for_patent_fragment")
    else:
        if connector_at_384 < 8.0:
            blockers.append("molnextr_input_connector_too_short_after_resize")
        if max_connector is not None and connector_at_384 > max_connector:
            blockers.append("molnextr_input_connector_too_long_for_patent_fragment")
        if mark_at_384 < 4.0:
            blockers.append("molnextr_input_terminal_label_or_dummy_too_small_after_resize")
    passed = not blockers
    return {
        "schema_version": "fragment_molnextr_input_quality_v1",
        "policy": "emulate_cropwhite_pad50_resize384_attachment_visibility_gate_with_patent_short_wavy_connector_contract",
        "passed": passed,
        "blockers": blockers,
        "attachment_render_mode": mode,
        "attachment_render_geometry": geometry,
        "mode_aware": True,
        "input_size": int(input_size),
        "cropwhite_pad_px": int(pad),
        "source_image_size": [int(width), int(height)],
        "crop_box": [int(left), int(top), int(right), int(bottom)],
        "padded_size": [int(padded_w), int(padded_h)],
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "wavy_length_px_at_384": wavy_at_384,
        "mark_length_px_at_384": mark_at_384,
        "wavy_amplitude_px_at_384": amplitude_at_384,
        "connector_length_px_at_384": connector_at_384,
        "visible_connector_length_px_at_384": visible_connector_at_384,
        "straight_connector_length_px_at_384": straight_connector_at_384,
        "min_atom_pair_distance_px_at_384": min_pair_at_384,
        "thresholds": {
            "min_wavy_length_px_at_384": 7.0,
            "max_wavy_length_px_at_384": 56.0,
            "min_wavy_amplitude_px_at_384": 0.9,
            "min_visible_connector_length_px_at_384": 10.0,
            "max_visible_connector_length_px_at_384": max_visible,
            "max_connector_length_px_at_384": max_connector,
            "max_straight_connector_length_px_at_384": (
                max(28.0, min(48.0, wavy_at_384 * 1.12 + 18.0))
                if is_custom_perpendicular_wavy_geometry(geometry)
                else max(18.0, min(48.0, wavy_at_384 * 2.15 + 8.0))
                if mode == "wavy"
                else None
            ),
            "min_connector_length_px_at_384": 10.0 if mode == "cut" else 8.0,
            "min_cut_mark_length_px_at_384": 12.0,
            "max_cut_mark_length_px_at_384": 96.0,
            "min_terminal_label_or_dummy_size_px_at_384": 4.0,
            "min_atom_pair_distance_px_at_384": 5.0,
        },
    }


def build_fragment_visual_quality(
    *,
    pixel_geometry: dict[str, Any],
    molnextr_input_quality: dict[str, Any],
    document_context_info: dict[str, Any],
    image_stats: dict[str, Any],
    attachment_domain_stats: dict[str, Any],
) -> dict[str, Any]:
    passed = (
        pixel_geometry.get("passed") is True
        and molnextr_input_quality.get("passed") is True
        and image_stats.get("blank") is not True
        and image_stats.get("dense") is not True
        and attachment_domain_stats.get("blank") is not True
        and attachment_domain_stats.get("dense") is not True
    )
    blockers: list[str] = []
    for name, payload in [
        ("pixel_geometry", pixel_geometry),
        ("molnextr_input_quality", molnextr_input_quality),
    ]:
        if payload.get("passed") is not True:
            blockers.append(f"{name}_failed")
            blockers.extend(str(value) for value in payload.get("blockers") or [])
    if image_stats.get("blank") or attachment_domain_stats.get("blank"):
        blockers.append("fragment_image_blank")
    if image_stats.get("dense") or attachment_domain_stats.get("dense"):
        blockers.append("fragment_image_dense")
    return {
        "schema_version": "fragment_visual_quality_v1",
        "policy": "markush_style_fragment_visual_quality_with_pixel_rmse_and_molnextr_input_gate",
        "passed": passed,
        "blockers": sorted(set(blockers)),
        "machine_audit_passed": passed,
        "manual_visual_review_required": True,
        "manual_visual_review_passed": False,
        "document_context_enabled": bool(document_context_info.get("enabled")),
        "document_context_operations": list(document_context_info.get("operations") or []),
        "pixel_geometry_passed": pixel_geometry.get("passed") is True,
        "molnextr_input_quality_passed": molnextr_input_quality.get("passed") is True,
        "image_density_passed": not bool(image_stats.get("blank") or image_stats.get("dense")),
        "attachment_domain_density_passed": not bool(
            attachment_domain_stats.get("blank") or attachment_domain_stats.get("dense")
        ),
    }


def build_fragment_document_realism(
    *,
    render_style: str,
    mark_geometry: dict[str, Any],
    document_context_info: dict[str, Any],
    visual_quality: dict[str, Any],
) -> dict[str, Any]:
    fragment_mark = (
        mark_geometry.get("wavy_geometry")
        if isinstance(mark_geometry.get("wavy_geometry"), dict)
        else mark_geometry.get("fragment_mark_geometry")
        if isinstance(mark_geometry.get("fragment_mark_geometry"), dict)
        else {}
    )
    mode = str(mark_geometry.get("attachment_render_mode") or "")
    geometry = str(mark_geometry.get("attachment_render_geometry") or fragment_mark.get("geometry") or "")
    is_wavy = mode == "wavy" or is_custom_perpendicular_wavy_geometry(geometry)
    return {
        "schema_version": "fragment_document_realism_v1",
        "policy": "patent_literature_markush_fragment_document_crop_v1",
        "status": "machine_audited_requires_manual_visual_review",
        "source_visual_domain": "patent_literature_markush_fragment_crop",
        "generation_method": "rdkit_moldraw2d_graph_with_custom_markush_attachment_primitive_and_existing_document_context_noise_jpeg_pipeline",
        "render_style": str(render_style),
        "document_context_enabled": bool(document_context_info.get("enabled")),
        "document_context_operations": list(document_context_info.get("operations") or []),
        "attachment_render_mode": mode,
        "attachment_render_geometry": geometry,
        "custom_markush_attachment_primitive": bool(is_wavy and is_custom_perpendicular_wavy_geometry(geometry)),
        "nonterminal_anchor_allowed": True,
        "single_attachment_instance_per_row": True,
        "not_rdkit_stereo_wavy": bool(fragment_mark.get("not_rdkit_stereo_wavy")),
        "bond_semantics_contract": "dummy_star_atom_single_anchor_bond_visual_attachment_mark_only_no_rdkit_unknown_or_stereo_wavy",
        "machine_audit_passed": visual_quality.get("passed") is True,
        "manual_visual_review_required": True,
        "manual_visual_review_passed": False,
        "render_parameters": {
            "line_width_px": fragment_mark.get("line_width_px"),
            "wavy_length_px": fragment_mark.get("wavy_length_px") or fragment_mark.get("wavy_bond_length_px"),
            "wavy_amplitude_px": fragment_mark.get("wavy_amplitude_px"),
            "wavy_cycles": fragment_mark.get("wavy_cycles"),
            "connector_length_px": fragment_mark.get("connector_length_px"),
            "visible_connector_length_px": fragment_mark.get("visible_connector_length_px"),
            "straight_connector_length_px": fragment_mark.get("straight_connector_length_px"),
        },
        "external_basis": [
            {
                "name": "MarkushGenerator/CDK formal-sim document realism contract",
                "role": "Match Markush sidecar training realism gates: page context, scan/JPEG perturbation, synchronized geometry evidence.",
            },
            {
                "name": "MolNexTR coordinate-aware image-to-graph supervision",
                "role": "Final image must preserve atom coordinates, connector visibility, and attachment mark visibility after CropWhite+Resize.",
            },
        ],
    }


def audit_attachment_bond_semantics(
    bonds: list[dict[str, Any]],
    *,
    anchor_index: int,
    dummy_index: int,
    mode: str,
    geometry: str,
) -> dict[str, Any]:
    attachment_bonds = []
    stereo_like = []
    for bond in bonds:
        if not isinstance(bond, dict):
            continue
        begin = bond.get("begin_atom_index")
        end = bond.get("end_atom_index")
        try:
            endpoints = {int(begin), int(end)}
        except (TypeError, ValueError):
            endpoints = set()
        direction = str(bond.get("bond_dir") or "")
        if endpoints == {int(anchor_index), int(dummy_index)}:
            attachment_bonds.append(bond)
        if direction not in {"NONE", ""}:
            stereo_like.append(
                {
                    "begin_atom_index": begin,
                    "end_atom_index": end,
                    "bond_dir": direction,
                    "bond_order": bond.get("bond_order"),
                }
            )
    attachment_bond = attachment_bonds[0] if attachment_bonds else {}
    attachment_dir = str(attachment_bond.get("bond_dir") or "")
    attachment_order = str(attachment_bond.get("bond_order") or "")
    blockers: list[str] = []
    if len(attachment_bonds) != 1:
        blockers.append("attachment_anchor_dummy_bond_count_not_one")
    if attachment_order.upper() != "SINGLE":
        blockers.append("attachment_anchor_dummy_bond_not_single")
    if attachment_dir != "NONE":
        blockers.append("attachment_anchor_dummy_bond_has_rdkit_direction")
    attachment_stereo_like = attachment_dir not in {"NONE", ""}
    if mode == "wavy" and is_custom_perpendicular_wavy_geometry(geometry) and attachment_stereo_like:
        blockers.append("custom_wavy_attachment_bond_has_rdkit_stereo_or_unknown_dir")
    return {
        "schema_version": "fragment_attachment_bond_semantics_v1",
        "policy": "visual_wavy_is_png_primitive_only_graph_keeps_star_dummy_single_bond_with_bonddir_none",
        "passed": not blockers,
        "blockers": blockers,
        "attachment_render_mode": str(mode),
        "attachment_render_geometry": str(geometry),
        "attachment_bond": attachment_bond,
        "stereo_or_unknown_bond_dir_count": int(len(stereo_like)),
        "stereo_or_unknown_bonds": stereo_like[:12],
        "not_rdkit_stereo_wavy": bool(mode != "wavy" or (is_custom_perpendicular_wavy_geometry(geometry) and not attachment_stereo_like)),
    }


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    vx, vy = ex - sx, ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-8:
        return point_distance(point, start)
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / denom))
    return math.hypot(px - (sx + t * vx), py - (sy + t * vy))


def _point_from_dict(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError):
        return None


def audit_terminal_wavy_externality(
    *,
    coords: dict[int, tuple[float, float]],
    bonds: list[dict[str, Any]],
    mark_geometry: dict[str, Any],
    anchor_index: int,
    dummy_index: int,
) -> dict[str, Any]:
    fragment_mark = (
        mark_geometry.get("wavy_geometry")
        if isinstance(mark_geometry.get("wavy_geometry"), dict)
        else mark_geometry.get("fragment_mark_geometry")
        if isinstance(mark_geometry.get("fragment_mark_geometry"), dict)
        else {}
    )
    mode = str(mark_geometry.get("attachment_render_mode") or "")
    geometry = str(mark_geometry.get("attachment_render_geometry") or fragment_mark.get("geometry") or "")
    if mode != "wavy" or not is_custom_perpendicular_wavy_geometry(geometry):
        return {
            "schema_version": "terminal_wavy_externality_v1",
            "passed": True,
            "applicable": False,
            "blockers": [],
        }
    blockers: list[str] = []
    anchor_xy = coords.get(int(anchor_index))
    endpoint_xy = coords.get(int(dummy_index))
    if anchor_xy is None or endpoint_xy is None:
        return {
            "schema_version": "terminal_wavy_externality_v1",
            "passed": False,
            "applicable": True,
            "blockers": ["missing_anchor_or_dummy_pixel_coordinate"],
        }
    ux, uy, _px, _py, norm = connector_basis(anchor_xy, endpoint_xy)
    line_width = float(fragment_mark.get("line_width_px") or 1.0)
    amplitude = float(fragment_mark.get("wavy_amplitude_px") or 0.0)
    wavy_center = _point_from_dict(fragment_mark.get("wavy_center")) or _point_from_dict(fragment_mark.get("mark_center"))
    center_distance = point_distance(wavy_center, endpoint_xy) if wavy_center is not None else None
    junction_style = str(fragment_mark.get("wavy_connector_intersection_style") or "")
    if junction_style not in {"center_cross", "side_touch"}:
        blockers.append("invalid_wavy_connector_intersection_style")
    wavy_length = float(fragment_mark.get("wavy_length_px") or fragment_mark.get("wavy_bond_length_px") or 0.0)
    if junction_style == "center_cross":
        if fragment_mark.get("connector_crosses_wavy_center") is not True:
            blockers.append("connector_crosses_wavy_center_not_declared")
        if fragment_mark.get("center_cross_curve_passes_connector") is not True:
            blockers.append("center_cross_curve_passes_connector_not_declared")
        if fragment_mark.get("center_cross_connector_is_solid_line") is not True:
            blockers.append("center_cross_solid_connector_not_declared")
        if center_distance is None or center_distance > max(1.5, line_width + 0.8):
            blockers.append("wavy_center_not_on_connector_endpoint")
        if float(fragment_mark.get("mark_connector_bridge_past_endpoint_px") or 0.0) < 0.75:
            blockers.append("connector_does_not_visibly_pass_through_wavy_center")
    elif junction_style == "side_touch":
        if fragment_mark.get("connector_crosses_wavy_center") is True:
            blockers.append("side_touch_declares_center_cross")
        if fragment_mark.get("connector_contact_is_terminal_wavy_endpoint") is not True:
            blockers.append("side_touch_contact_endpoint_not_declared")
        expected = max(1.0, wavy_length * 0.5)
        if center_distance is None or abs(center_distance - expected) > max(2.5, amplitude * 1.2):
            blockers.append("side_touch_wavy_center_not_one_half_mark_from_endpoint")

    neighbor_points: list[tuple[float, float]] = []
    non_dummy_points = {idx: xy for idx, xy in coords.items() if int(idx) != int(dummy_index)}
    for bond in bonds:
        try:
            begin = int(bond.get("begin_atom_index"))
            end = int(bond.get("end_atom_index"))
        except (TypeError, ValueError):
            continue
        if begin == int(anchor_index) and end != int(dummy_index) and end in coords:
            neighbor_points.append(coords[end])
        elif end == int(anchor_index) and begin != int(dummy_index) and begin in coords:
            neighbor_points.append(coords[begin])
    if not neighbor_points:
        neighbor_points = [xy for idx, xy in non_dummy_points.items() if int(idx) != int(anchor_index)]
    if neighbor_points:
        centroid = (
            sum(point[0] for point in neighbor_points) / len(neighbor_points),
            sum(point[1] for point in neighbor_points) / len(neighbor_points),
        )
        outward = unit_vector((anchor_xy[0] - centroid[0], anchor_xy[1] - centroid[1]), default=(ux, uy))
        outward_dot = float(ux * outward[0] + uy * outward[1])
    else:
        outward_dot = 1.0
    if outward_dot < 0.25:
        blockers.append("terminal_wavy_points_toward_molecule_interior")

    body_projection_values: list[float] = []
    for idx, point in non_dummy_points.items():
        if int(idx) == int(anchor_index):
            continue
        body_projection_values.append((point[0] - anchor_xy[0]) * ux + (point[1] - anchor_xy[1]) * uy)
    max_body_projection = max(body_projection_values) if body_projection_values else -1e9
    projection_clearance = max(5.0, min(16.0, norm * 0.16))
    if max_body_projection > norm - projection_clearance:
        blockers.append("molecule_extends_to_terminal_wavy_side")

    terminal_points = []
    for point in fragment_mark.get("terminal_mark_sampled_points_px") or []:
        parsed = _point_from_dict(point)
        if parsed is not None:
            terminal_points.append(parsed)
    min_terminal_projection = None
    if terminal_points:
        projections = [(point[0] - anchor_xy[0]) * ux + (point[1] - anchor_xy[1]) * uy for point in terminal_points]
        min_terminal_projection = float(min(projections))
        allowed_backreach = max(7.0, amplitude * 2.6)
        if junction_style == "center_cross":
            allowed_backreach = max(allowed_backreach, float(fragment_mark.get("center_cross_required_cross_past_px") or 0.0) + line_width)
        if min_terminal_projection < norm - allowed_backreach:
            blockers.append("terminal_wavy_reaches_back_inside_connector")
        endpoint_distances = [point_distance(point, endpoint_xy) for point in terminal_points]
        min_terminal_to_endpoint_distance = float(min(endpoint_distances)) if endpoint_distances else None
        if junction_style == "center_cross" and (
            min_terminal_to_endpoint_distance is None
            or min_terminal_to_endpoint_distance > max(1.25, line_width * 0.75 + 0.75)
        ):
            blockers.append("center_cross_wavy_curve_not_on_connector_endpoint")
        if junction_style == "side_touch" and (
            min_terminal_to_endpoint_distance is None
            or min_terminal_to_endpoint_distance > max(1.75, line_width + 0.75)
        ):
            blockers.append("side_touch_wavy_endpoint_not_on_connector_endpoint")
    else:
        blockers.append("missing_terminal_wavy_points")
        min_terminal_to_endpoint_distance = None

    non_attachment_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for bond in bonds:
        try:
            begin = int(bond.get("begin_atom_index"))
            end = int(bond.get("end_atom_index"))
        except (TypeError, ValueError):
            continue
        if int(dummy_index) in {begin, end}:
            continue
        if begin in coords and end in coords:
            non_attachment_segments.append((coords[begin], coords[end]))
    min_wavy_to_bond_distance = None
    if terminal_points and non_attachment_segments:
        distances = [
            _point_segment_distance(point, segment_start, segment_end)
            for point in terminal_points
            for segment_start, segment_end in non_attachment_segments
        ]
        min_wavy_to_bond_distance = float(min(distances)) if distances else None
        if min_wavy_to_bond_distance is not None and min_wavy_to_bond_distance < max(2.4, line_width * 2.0):
            blockers.append("terminal_wavy_too_close_to_non_attachment_bond")
    atom_distances = [
        point_distance(point, atom_xy)
        for point in terminal_points
        for idx, atom_xy in non_dummy_points.items()
        if int(idx) != int(anchor_index)
    ]
    min_wavy_to_atom_distance = float(min(atom_distances)) if atom_distances else None
    if min_wavy_to_atom_distance is not None and min_wavy_to_atom_distance < max(3.2, line_width * 2.4):
        blockers.append("terminal_wavy_too_close_to_non_attachment_atom")

    return {
        "schema_version": "terminal_wavy_externality_v1",
        "passed": not blockers,
        "applicable": True,
        "blockers": blockers,
        "connector_crosses_wavy_center": fragment_mark.get("connector_crosses_wavy_center") is True,
        "wavy_connector_intersection_style": junction_style,
        "wavy_center_endpoint_distance_px": center_distance,
        "connector_bridge_past_endpoint_px": float(fragment_mark.get("mark_connector_bridge_past_endpoint_px") or 0.0),
        "outward_dot": float(outward_dot),
        "max_body_projection_px": float(max_body_projection) if body_projection_values else None,
        "connector_length_px": float(norm),
        "projection_clearance_px": float(projection_clearance),
        "min_terminal_projection_px": min_terminal_projection,
        "min_terminal_to_endpoint_distance_px": min_terminal_to_endpoint_distance,
        "min_wavy_to_non_attachment_bond_px": min_wavy_to_bond_distance,
        "min_wavy_to_non_attachment_atom_px": min_wavy_to_atom_distance,
        "policy": "terminal_wavy_center_cross_or_side_touch_connector_and_outside_non_dummy_molecule_body",
    }


def draw_realistic_attachment(
    path: Path,
    *,
    anchor_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    anchor_label: str,
    anchor_depiction_mode: str,
    visual_shape: str,
    semantic_family: str,
    render_style: str,
    native_terminal_label: str,
    index: int,
    rng: random.Random,
    native_bond_line_width_px: float,
    coordinate_scale: float,
    draw_style_info: dict[str, Any],
) -> dict[str, Any]:
    ux, uy, px, py, norm = connector_basis(anchor_xy, endpoint_xy)
    ax, ay = anchor_xy
    ex, ey = endpoint_xy
    if float(native_bond_line_width_px) <= 0.0:
        raise ValueError("native_bond_line_width_px_required_for_attachment_render")
    expected_scaled_line_width = max(0.70, float(native_bond_line_width_px) * max(0.05, float(coordinate_scale)))
    anchor_has_visible_text = anchor_label_is_visible_text_for_connector(
        anchor_label=anchor_label,
        anchor_depiction_mode=anchor_depiction_mode,
    )
    estimated_visible_start, visible_start_policy = visible_connector_start(
        anchor_xy=anchor_xy,
        endpoint_xy=endpoint_xy,
        anchor_label=anchor_label,
        anchor_label_is_visible_text=anchor_has_visible_text,
        anchor_depiction_mode=anchor_depiction_mode,
        line_width=expected_scaled_line_width,
        atom_font_size_px=float(draw_style_info.get("max_font_size_px") or draw_style_info.get("min_font_size_px") or 18.0),
    )
    native_metrics = measure_native_connector_metrics(
        path,
        anchor_xy=anchor_xy,
        endpoint_xy=endpoint_xy,
        anchor_label=anchor_label,
        anchor_label_is_visible_text=anchor_has_visible_text,
        estimated_visible_start=estimated_visible_start,
        expected_line_width_px=expected_scaled_line_width,
    )
    if native_metrics.get("passed") is not True:
        raise ValueError(f"native attachment connector measurement failed: {native_metrics}")
    visible_point = native_metrics.get("visible_connector_start") if isinstance(native_metrics.get("visible_connector_start"), dict) else {}
    visible_start = (float(visible_point["x"]), float(visible_point["y"]))
    visible_start_policy = {
        **visible_start_policy,
        "measurement_policy": "final_rendered_rdkit_connector_pixel_mask",
        "estimated_visible_start": {"x": float(estimated_visible_start[0]), "y": float(estimated_visible_start[1])},
        "measured_visible_start_offset_px": float(native_metrics.get("visible_connector_start_offset_px") or 0.0),
        "estimated_visible_start_offset_px": float(native_metrics.get("estimated_visible_start_offset_px") or 0.0),
        "native_connector_measurement_passed": True,
    }
    line_width = max(0.75, float(native_metrics["line_width_px"]))
    draw_line_width = custom_attachment_draw_line_width(
        measured_native_width_px=line_width,
        expected_scaled_width_px=expected_scaled_line_width,
    )
    line_width_source = "final_rendered_native_attachment_bond_width"
    sx, sy = visible_start

    base = {
        "connector_vector": {"x": ux, "y": uy},
        "mark_axis_vector": {"x": px, "y": py},
        "connector_length_px": norm,
        "visible_connector_length_px": point_distance((sx, sy), (ex, ey)),
        "straight_connector_length_px": point_distance((sx, sy), (ex, ey)),
        "visible_connector_start": {"x": sx, "y": sy},
        "visible_connector_start_policy": visible_start_policy,
        "mark_axis_dot_connector_abs": abs(px * ux + py * uy),
        "line_width_px": line_width,
        "measured_native_line_width_px": float(line_width),
        "draw_line_width_px": float(draw_line_width),
        "line_width_source": line_width_source,
        "line_width_matches_native_bonds": line_width_source == "final_rendered_native_attachment_bond_width",
        "line_width_expected_scaled_rdkit_px": float(expected_scaled_line_width),
        "line_width_measurement": native_metrics,
        "depiction_profile": str(draw_style_info.get("depiction_profile") or ""),
        "atom_font_size_range_px": [
            int(draw_style_info.get("min_font_size_px") or 0),
            int(draw_style_info.get("max_font_size_px") or 0),
        ],
        "atom_font_size_range_estimated_final_px": [
            float(draw_style_info.get("min_font_size_px") or 0.0) * max(0.05, float(coordinate_scale)),
            float(draw_style_info.get("max_font_size_px") or 0.0) * max(0.05, float(coordinate_scale)),
        ],
        "final_coordinate_scale_from_rdkit_canvas": float(coordinate_scale),
        "atom_base_font_size": float(draw_style_info.get("base_font_size") or 0.0),
        "atom_label_padding": float(draw_style_info.get("additional_atom_label_padding") or 0.0),
        "atom_font_file": str(draw_style_info.get("font_file") or ""),
        "visual_shape": visual_shape,
        "semantic_family": semantic_family,
        "connector_render_source": "rdkit_moldraw2d_native_bond",
        "anchor_depiction_mode": str(anchor_depiction_mode or ""),
        "anchor_label_is_visible_text": bool(anchor_has_visible_text),
    }

    if visual_shape in {"left_terminal_cut_or_open_stub", "bottom_crop_attachment_or_cut"}:
        mark_center_x = ex
        mark_center_y = ey
        length = max(42.0, min(82.0, norm * 1.08, norm * 2.55))
        return {
            "attachment_render_mode": "cut",
            "attachment_render_geometry": "rdkit_moldraw2d_terminal_perpendicular_cut_bar",
            "fragment_mark_geometry": {
                **base,
                "mark_length_px": length,
                "mark_length_to_connector_ratio": length / norm,
                "mark_count": 1,
                "mark_center": {"x": mark_center_x, "y": mark_center_y},
                "mark_connector_overlap_px": 0.0,
                "terminal_mark_render_source": "rdkit_moldraw2d_draw_line_rawcoords",
                "terminal_mark_endpoints_px": pixel_dicts(
                    [
                        (mark_center_x - px * length * 0.5, mark_center_y - py * length * 0.5),
                        (mark_center_x + px * length * 0.5, mark_center_y + py * length * 0.5),
                    ]
                ),
                "terminal_mark_sampled_points_px": pixel_dicts(
                    _line_samples_pixels(
                        (mark_center_x - px * length * 0.5, mark_center_y - py * length * 0.5),
                        (mark_center_x + px * length * 0.5, mark_center_y + py * length * 0.5),
                        samples=15,
                    )
                ),
                "connector_sampled_points_px": pixel_dicts(_line_samples_pixels((sx, sy), (ex, ey), samples=19)),
            },
        }

    if visual_shape == "candidate_terminal_wavy_cut":
        primitive = draw_custom_markush_wavy_attachment(
            path,
            visible_start=(sx, sy),
            endpoint_xy=endpoint_xy,
            anchor_xy=anchor_xy,
            anchor_label=anchor_label,
            anchor_label_is_visible_text=anchor_has_visible_text,
            anchor_depiction_mode=anchor_depiction_mode,
            line_width=line_width,
            draw_line_width=draw_line_width,
            rng=rng,
            allow_side_touch=False,
        )
        length = float(primitive["wavy_length_px"])
        amplitude = float(primitive["wavy_amplitude_px"])
        cycles = float(primitive["wavy_cycles"])
        visible_label_clearance = validate_visible_anchor_label_clearance(
            policy=visible_start_policy,
            measured_start_offset_px=float(native_metrics.get("visible_connector_start_offset_px") or 0.0),
            straight_connector_length_px=float(primitive["straight_connector_length_px"]),
            draw_line_width_px=float(draw_line_width),
        )
        if visible_label_clearance.get("passed") is not True:
            raise ValueError(f"visible anchor label clearance failed: {visible_label_clearance}")
        mark_center = {
            "x": float(primitive["wavy_center"]["x"]),
            "y": float(primitive["wavy_center"]["y"]),
        }
        wavy = {
            **base,
            **primitive,
            "wavy_axis_vector": {"x": px, "y": py},
            "wavy_bond_length_px": length,
            "wavy_amplitude_px": amplitude,
            "wavy_width_px": line_width,
            "wavy_cycles": cycles,
            "wavy_length_to_connector_ratio": length / norm,
            "mark_center": mark_center,
            "pixel_connection_contract": "custom_markush_connector_to_terminal_perpendicular_wavy",
            "wavy_cut_style": "patent_terminal_perpendicular_wavy_mark",
            "wavy_axis_dot_connector_abs": abs(px * ux + py * uy),
            "visible_anchor_label_clearance": visible_label_clearance,
            "geometry": CUSTOM_PERPENDICULAR_WAVY_GEOMETRY,
            "terminal_mark_render_source": "custom_moldraw2d_image_primitive_terminal_perpendicular_wavy",
            "terminal_label_render_source": "rdkit_moldraw2d_hidden_dummy_atom",
            "not_rdkit_stereo_wavy": True,
            "terminal_perpendicular_wavy": True,
        }
        return {
            "attachment_render_mode": "wavy",
            "attachment_render_geometry": CUSTOM_PERPENDICULAR_WAVY_GEOMETRY,
            "wavy_geometry": wavy,
            "fragment_mark_geometry": {
                **base,
                **primitive,
                "mark_length_px": length,
                "mark_length_to_connector_ratio": length / norm,
                "mark_center": mark_center,
                "pixel_connection_contract": "custom_markush_connector_to_terminal_perpendicular_wavy",
                "wavy_cut_style": "patent_terminal_perpendicular_wavy_mark",
                "terminal_mark_render_source": "custom_moldraw2d_image_primitive_terminal_perpendicular_wavy",
                "terminal_label_render_source": "rdkit_moldraw2d_hidden_dummy_atom",
                "geometry": CUSTOM_PERPENDICULAR_WAVY_GEOMETRY,
                "wavy_axis_vector": {"x": px, "y": py},
                "wavy_axis_dot_connector_abs": abs(px * ux + py * uy),
                "not_rdkit_stereo_wavy": True,
                "terminal_perpendicular_wavy": True,
            },
        }

    label = native_terminal_label or "*"
    return {
        "attachment_render_mode": "query_attachment" if label != "*" else "dummy_atom",
        "attachment_render_geometry": "straight_connector_with_terminal_query_label"
        if label != "*"
        else "straight_connector_with_terminal_dummy_atom",
            "fragment_mark_geometry": {
                **base,
                "mark_label": label,
                "mark_label_position": {"x": ex, "y": ey},
                "mark_length_px": max(8.0, min(18.0, norm * 0.36)),
                "mark_length_to_connector_ratio": max(8.0, min(18.0, norm * 0.36)) / norm,
                "terminal_mark_sampled_points_px": pixel_dicts(_line_samples_pixels((sx, sy), (ex, ey), samples=9)),
                "connector_sampled_points_px": pixel_dicts(_line_samples_pixels((sx, sy), (ex, ey), samples=19)),
                "terminal_label_render_source": "rdkit_moldraw2d_native_atom_label",
            },
        }


def crop_to_real_fragment(
    path: Path,
    coords: dict[int, tuple[float, float]],
    rng: random.Random,
    *,
    anchor_index: int | None = None,
    dummy_index: int | None = None,
    target_side: str = "",
    real_tight_crop_style: bool = False,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image.convert("L"))
    mask = arr < 245
    if not mask.any():
        return coords, image.width, image.height, {"cropped": False, "reason": "blank_image"}
    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    protected_box = None
    if anchor_index is not None and dummy_index is not None and anchor_index in coords and dummy_index in coords:
        ax, ay = coords[int(anchor_index)]
        ex, ey = coords[int(dummy_index)]
        ux, uy, px, py, norm = connector_basis((ax, ay), (ex, ey))
        if real_tight_crop_style:
            wavy_style = patent_style_perpendicular_wavy_geometry(norm)
            endpoint_margin = max(7.0, min(16.0, float(wavy_style["amplitude"]) * 2.8 + 4.0))
            mark_margin = max(20.0, min(44.0, float(wavy_style["length"]) * 1.18 + float(wavy_style["amplitude"]) * 2.4))
            crop_guard_px = 5.0
        else:
            endpoint_margin = max(42.0, norm * 0.40)
            mark_margin = max(28.0, norm * 0.35)
            crop_guard_px = 10.0
        candidates = [
            (ex, ey),
            (ex + ux * endpoint_margin, ey + uy * endpoint_margin),
            (ex + px * mark_margin, ey + py * mark_margin),
            (ex - px * mark_margin, ey - py * mark_margin),
        ]
        protected_left = min(point[0] for point in candidates) - crop_guard_px
        protected_right = max(point[0] for point in candidates) + crop_guard_px
        protected_top = min(point[1] for point in candidates) - crop_guard_px
        protected_bottom = max(point[1] for point in candidates) + crop_guard_px
        left = min(left, int(math.floor(protected_left)))
        right = max(right, int(math.ceil(protected_right)))
        top = min(top, int(math.floor(protected_top)))
        bottom = max(bottom, int(math.ceil(protected_bottom)))
        protected_box = [protected_left, protected_top, protected_right, protected_bottom]
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        pad_left_extra = max(0, -left)
        pad_top_extra = max(0, -top)
        pad_right_extra = max(0, right - image.width)
        pad_bottom_extra = max(0, bottom - image.height)
        canvas = Image.new(
            "RGB",
            (image.width + pad_left_extra + pad_right_extra, image.height + pad_top_extra + pad_bottom_extra),
            sampled_edge_background(image),
        )
        canvas.paste(image, (pad_left_extra, pad_top_extra))
        image = canvas
        coords = {idx: (x + pad_left_extra, y + pad_top_extra) for idx, (x, y) in coords.items()}
        left += pad_left_extra
        right += pad_left_extra
        top += pad_top_extra
        bottom += pad_top_extra
    ink_w, ink_h = max(1, right - left), max(1, bottom - top)
    if real_tight_crop_style:
        pad_left = int(round(rng.uniform(0.025, 0.065) * ink_w))
        pad_right = int(round(rng.uniform(0.015, 0.050) * ink_w))
        pad_top = int(round(rng.uniform(0.045, 0.095) * ink_h))
        pad_bottom = int(round(rng.uniform(0.025, 0.065) * ink_h))
    else:
        pad_left = int(round(rng.uniform(0.08, 0.20) * ink_w))
        pad_right = int(round(rng.uniform(0.10, 0.28) * ink_w))
        pad_top = int(round(rng.uniform(0.12, 0.28) * ink_h))
        pad_bottom = int(round(rng.uniform(0.10, 0.28) * ink_h))
    if target_side == "top":
        pad_top = int(round(rng.uniform(0.01, 0.04 if real_tight_crop_style else 0.06) * ink_h))
        pad_bottom = int(round(rng.uniform(0.08 if real_tight_crop_style else 0.22, 0.16 if real_tight_crop_style else 0.42) * ink_h))
    elif target_side == "bottom":
        pad_top = int(round(rng.uniform(0.08 if real_tight_crop_style else 0.22, 0.16 if real_tight_crop_style else 0.42) * ink_h))
        pad_bottom = int(round(rng.uniform(0.01, 0.04 if real_tight_crop_style else 0.06) * ink_h))
    elif target_side == "left":
        pad_left = int(round(rng.uniform(0.01, 0.04 if real_tight_crop_style else 0.06) * ink_w))
        pad_right = int(round(rng.uniform(0.08 if real_tight_crop_style else 0.22, 0.16 if real_tight_crop_style else 0.42) * ink_w))
    elif target_side == "right":
        pad_left = int(round(rng.uniform(0.08 if real_tight_crop_style else 0.22, 0.16 if real_tight_crop_style else 0.42) * ink_w))
        pad_right = int(round(rng.uniform(0.01, 0.04 if real_tight_crop_style else 0.06) * ink_w))
    left = max(0, left - pad_left)
    right = min(image.width, right + pad_right)
    top = max(0, top - pad_top)
    bottom = min(image.height, bottom + pad_bottom)
    cropped = image.crop((left, top, right, bottom))
    if real_tight_crop_style:
        max_side = rng.choice([104, 112, 120, 128, 136, 144, 152])
        min_short_side = rng.choice([48, 52, 56, 60])
        max_short_side = rng.choice([88, 96, 104])
        scale = max_side / max(cropped.width, cropped.height)
        projected_short = min(cropped.width, cropped.height) * scale
        if projected_short < min_short_side:
            scale = min_short_side / min(cropped.width, cropped.height)
        projected_short = min(cropped.width, cropped.height) * scale
        if projected_short > max_short_side:
            scale = max_short_side / min(cropped.width, cropped.height)
        scale = max(0.35, min(1.45, scale))
    else:
        max_side = rng.choice([144, 160, 176, 192])
        min_short_side = rng.choice([104, 112, 120])
        max_short_side = None
        scale = min(max_side / max(cropped.width, cropped.height), min_short_side / min(cropped.width, cropped.height))
        scale = max(0.45, min(1.35, scale))
    if abs(scale - 1.0) > 0.03:
        new_size = (max(1, int(round(cropped.width * scale))), max(1, int(round(cropped.height * scale))))
        cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    cropped.save(path)
    updated = {idx: ((x - left) * scale, (y - top) * scale) for idx, (x, y) in coords.items()}
    return updated, cropped.width, cropped.height, {
        "cropped": True,
        "crop_box": [left, top, right, bottom],
        "protected_attachment_box": protected_box,
        "scale": scale,
        "target_max_side": max_side,
        "target_min_short_side": min_short_side,
        "target_max_short_side": max_short_side,
        "padding_px": [pad_left, pad_top, pad_right, pad_bottom],
        "real_tight_crop_style": bool(real_tight_crop_style),
        "operations": ["real_tight_wavy_crop"] if real_tight_crop_style else ["real_fragment_crop"],
        "target_side_preservation": {
            "enabled": bool(target_side),
            "target_side": target_side,
            "policy": "bias_synchronized_crop_padding_to_preserve_requested_endpoint_side_without_post_render_rotation",
        },
    }


def sampled_edge_background(image: Any) -> tuple[int, int, int]:
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bright = border[border.mean(axis=1) > 180]
    sample = bright if len(bright) else border
    rgb = np.median(sample, axis=0)
    return tuple(int(max(0, min(255, round(value)))) for value in rgb)


def require_endpoint_pixel_length(
    path: Path,
    coords: dict[int, tuple[float, float]],
    anchor_index: int,
    dummy_index: int,
    *,
    min_length_px: float,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    ax, ay = coords[int(anchor_index)]
    ex, ey = coords[int(dummy_index)]
    _, _, _, _, norm = connector_basis((ax, ay), (ex, ey))
    if float(min_length_px) > 0.0 and norm < float(min_length_px):
        raise ValueError(f"connector_too_short_after_native_render:{norm:.2f}px<{float(min_length_px):.2f}px")
    return coords, image.width, image.height, {
        "enforced": float(min_length_px) > 0.0,
        "connector_length_px": norm,
        "minimum_connector_length_px": float(min_length_px),
        "coordinate_mutation_after_render": False,
        "policy": "reject_if_native_rendered_connector_is_too_short",
    }


def rotate_to_target_endpoint_side(
    path: Path,
    coords: dict[int, tuple[float, float]],
    *,
    dummy_index: int,
    target_side: str,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if not target_side:
        return coords, image.width, image.height, {"enabled": False, "reason": "no_target_side"}
    if int(dummy_index) not in coords:
        raise ValueError("target_side_rotation_missing_dummy_coordinate")
    if target_side not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"invalid target side for rotation: {target_side}")

    endpoint_xy = coords[int(dummy_index)]
    candidates = [
        (turns, side_after_quarter_turn(endpoint_xy, image.width, image.height, turns))
        for turns in range(4)
    ]
    matching = [turns for turns, side in candidates if side == target_side]
    if not matching:
        raise ValueError(f"target_side_rotation_unreachable:{target_side}:{candidates}")
    turns_ccw = int(matching[0])
    if turns_ccw == 0:
        return coords, image.width, image.height, {
            "enabled": True,
            "turns_ccw": 0,
            "target_side": target_side,
            "source_side": candidates[0][1],
            "coordinate_mutation_after_render": False,
            "policy": "already_on_target_side_after_synchronized_render_geometry",
        }

    old_width, old_height = image.width, image.height
    if turns_ccw == 1:
        rotated = image.transpose(Image.Transpose.ROTATE_90)

        def transform(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return y, float(old_width) - x

    elif turns_ccw == 2:
        rotated = image.transpose(Image.Transpose.ROTATE_180)

        def transform(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return float(old_width) - x, float(old_height) - y

    else:
        rotated = image.transpose(Image.Transpose.ROTATE_270)

        def transform(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return float(old_height) - y, x

    updated = {idx: transform(point) for idx, point in coords.items()}
    rotated.save(path)
    return updated, rotated.width, rotated.height, {
        "enabled": True,
        "turns_ccw": turns_ccw,
        "target_side": target_side,
        "source_side": candidates[0][1],
        "coordinate_mutation_after_render": False,
        "policy": "synchronized_image_and_pose_quarter_turn_for_target_endpoint_side",
    }


def ensure_min_canvas_side(
    path: Path,
    coords: dict[int, tuple[float, float]],
    *,
    min_side_px: int,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    pad_left = pad_right = pad_top = pad_bottom = 0
    if image.width < int(min_side_px):
        missing = int(min_side_px) - image.width
        pad_left = missing // 2
        pad_right = missing - pad_left
    if image.height < int(min_side_px):
        missing = int(min_side_px) - image.height
        pad_top = missing // 2
        pad_bottom = missing - pad_top
    if not (pad_left or pad_right or pad_top or pad_bottom):
        return coords, image.width, image.height, {"enabled": False, "min_side_px": int(min_side_px), "padding_px": [0, 0, 0, 0]}
    canvas = Image.new("RGB", (image.width + pad_left + pad_right, image.height + pad_top + pad_bottom), sampled_edge_background(image))
    canvas.paste(image, (pad_left, pad_top))
    canvas.save(path)
    updated = {idx: (x + pad_left, y + pad_top) for idx, (x, y) in coords.items()}
    return updated, canvas.width, canvas.height, {
        "enabled": True,
        "min_side_px": int(min_side_px),
        "padding_px": [pad_left, pad_top, pad_right, pad_bottom],
    }


def preserve_endpoint_side_with_canvas_padding(
    path: Path,
    coords: dict[int, tuple[float, float]],
    *,
    dummy_index: int,
    target_side: str,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if not target_side:
        return coords, image.width, image.height, {
            "enabled": False,
            "reason": "no_target_side",
            "coordinate_mutation_after_render": False,
        }
    if int(dummy_index) not in coords:
        raise ValueError("side_preserving_padding_missing_dummy_coordinate")
    if target_side not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"invalid target side for side-preserving padding: {target_side}")

    ex, ey = coords[int(dummy_index)]
    pad_left = pad_right = pad_top = pad_bottom = 0
    margin = 2.0
    max_axis_padding = max(96, int(round(max(image.width, image.height) * 6.0)))
    if target_side == "top":
        pad_left = pad_right = max(12, int(round(image.width * 0.25)))
        width_after = float(image.width + pad_left + pad_right)
        x_after = float(ex + pad_left)
        horizontal_margin = max(1e-3, min(x_after / width_after, 1.0 - x_after / width_after))
        target_pad = max(24.0, image.height - 2.0 * ey + margin, ey / horizontal_margin - image.height + margin)
        pad_bottom = int(math.ceil(target_pad))
    elif target_side == "bottom":
        pad_left = pad_right = max(12, int(round(image.width * 0.25)))
        width_after = float(image.width + pad_left + pad_right)
        x_after = float(ex + pad_left)
        horizontal_margin = max(1e-3, min(x_after / width_after, 1.0 - x_after / width_after))
        target_pad = max(
            24.0,
            2.0 * ey - image.height + margin,
            (image.height - ey) / horizontal_margin - image.height + margin,
        )
        pad_top = int(math.ceil(target_pad))
    elif target_side == "left":
        pad_top = pad_bottom = max(12, int(round(image.height * 0.25)))
        height_after = float(image.height + pad_top + pad_bottom)
        y_after = float(ey + pad_top)
        vertical_margin = max(1e-3, min(y_after / height_after, 1.0 - y_after / height_after))
        target_pad = max(24.0, image.width - 2.0 * ex + margin, ex / vertical_margin - image.width + margin)
        pad_right = int(math.ceil(target_pad))
    elif target_side == "right":
        pad_top = pad_bottom = max(12, int(round(image.height * 0.25)))
        height_after = float(image.height + pad_top + pad_bottom)
        y_after = float(ey + pad_top)
        vertical_margin = max(1e-3, min(y_after / height_after, 1.0 - y_after / height_after))
        target_pad = max(
            24.0,
            2.0 * ex - image.width + margin,
            (image.width - ex) / vertical_margin - image.width + margin,
        )
        pad_left = int(math.ceil(target_pad))
    if max(pad_left, pad_right, pad_top, pad_bottom) > max_axis_padding:
        raise ValueError(
            "side_preserving_padding_requires_excessive_canvas:"
            f"{target_side}:{[pad_left, pad_top, pad_right, pad_bottom]}>{max_axis_padding}"
        )

    canvas = Image.new(
        "RGB",
        (image.width + pad_left + pad_right, image.height + pad_top + pad_bottom),
        sampled_edge_background(image),
    )
    canvas.paste(image, (pad_left, pad_top))
    canvas.save(path)
    updated = {idx: (x + pad_left, y + pad_top) for idx, (x, y) in coords.items()}
    endpoint = normalize_coord(updated[int(dummy_index)][0], updated[int(dummy_index)][1], canvas.width, canvas.height)
    observed_side = side_from_endpoint(endpoint["x"], endpoint["y"])
    if observed_side != target_side:
        raise ValueError(f"side_preserving_padding_failed:{target_side}:{observed_side}")
    return updated, canvas.width, canvas.height, {
        "enabled": True,
        "target_side": target_side,
        "observed_side_after_padding": observed_side,
        "padding_px": [pad_left, pad_top, pad_right, pad_bottom],
        "endpoint_before_padding_px": [float(ex), float(ey)],
        "endpoint_after_padding_normalized": endpoint,
        "max_axis_padding_px": int(max_axis_padding),
        "operations": ["synchronized_side_preserving_canvas_padding"],
        "coordinate_mutation_after_render": False,
        "policy": (
            "endpoint_position_solved_synchronized_canvas_padding_only_to_preserve_requested_endpoint_side_"
            "without_rotation_flip_or_endpoint_coordinate_rewrite"
        ),
    }


PIXEL_POINT_KEYS = {
    "visible_connector_start",
    "estimated_visible_start",
    "mark_center",
    "mark_label_position",
    "wavy_start",
    "wavy_end",
    "wavy_center",
    "wavy_connector_contact_point",
    "straight_connector_end",
    "connector_draw_end",
    "connector_draw_start",
    "native_connector_erase_start",
}

PIXEL_POINT_LIST_KEYS = {
    "connector_sampled_points_px",
    "terminal_mark_endpoints_px",
    "terminal_mark_sampled_points_px",
    "wavy_sampled_points_px",
}


def _shift_point_dict(point: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    updated = dict(point)
    updated["x"] = float(updated.get("x") or 0.0) + float(dx)
    updated["y"] = float(updated.get("y") or 0.0) + float(dy)
    return updated


def _shift_mark_geometry_pixels(value: Any, dx: float, dy: float) -> Any:
    if isinstance(value, dict):
        updated: dict[str, Any] = {}
        for key, item in value.items():
            if key in PIXEL_POINT_KEYS and isinstance(item, dict):
                updated[key] = _shift_point_dict(item, dx, dy)
            elif key in PIXEL_POINT_LIST_KEYS and isinstance(item, list):
                shifted = []
                for point in item:
                    shifted.append(_shift_point_dict(point, dx, dy) if isinstance(point, dict) else point)
                updated[key] = shifted
            elif isinstance(item, dict):
                updated[key] = _shift_mark_geometry_pixels(item, dx, dy)
            elif isinstance(item, list):
                updated[key] = [
                    _shift_mark_geometry_pixels(element, dx, dy) if isinstance(element, dict) else element
                    for element in item
                ]
            else:
                updated[key] = item
        return updated
    return value


def _geometry_pixel_points(mark_geometry: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in PIXEL_POINT_KEYS and isinstance(item, dict):
                try:
                    points.append((float(item.get("x")), float(item.get("y"))))
                except (TypeError, ValueError):
                    pass
            elif key in PIXEL_POINT_LIST_KEYS and isinstance(item, list):
                for point in item:
                    if not isinstance(point, dict):
                        continue
                    try:
                        points.append((float(point.get("x")), float(point.get("y"))))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(item, dict):
                visit(item)

    visit(mark_geometry)
    return points


def _fragment_mark_for_canvas(mark_geometry: dict[str, Any]) -> dict[str, Any]:
    if isinstance(mark_geometry.get("wavy_geometry"), dict):
        return mark_geometry["wavy_geometry"]
    if isinstance(mark_geometry.get("fragment_mark_geometry"), dict):
        return mark_geometry["fragment_mark_geometry"]
    return {}


def ensure_attachment_geometry_canvas_padding(
    path: Path,
    coords: dict[int, tuple[float, float]],
    mark_geometry: dict[str, Any],
) -> tuple[dict[int, tuple[float, float]], dict[str, Any], int, int, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    points = _geometry_pixel_points(mark_geometry)
    fragment_mark = _fragment_mark_for_canvas(mark_geometry)
    line_width = float(fragment_mark.get("line_width_px") or 1.0)
    amplitude = float(fragment_mark.get("wavy_amplitude_px") or 0.0)
    margin = max(3.0, line_width * 2.6, amplitude * 1.4)
    if not points:
        return coords, mark_geometry, image.width, image.height, {
            "enabled": False,
            "reason": "no_attachment_geometry_points",
            "coordinate_mutation_after_render": False,
            "padding_px": [0, 0, 0, 0],
        }

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    pad_left = max(0, int(math.ceil(margin - min_x)))
    pad_top = max(0, int(math.ceil(margin - min_y)))
    pad_right = max(0, int(math.ceil(max_x + margin - float(image.width - 1))))
    pad_bottom = max(0, int(math.ceil(max_y + margin - float(image.height - 1))))
    if not (pad_left or pad_top or pad_right or pad_bottom):
        return coords, mark_geometry, image.width, image.height, {
            "enabled": False,
            "reason": "attachment_geometry_already_inside_canvas",
            "coordinate_mutation_after_render": False,
            "padding_px": [0, 0, 0, 0],
            "attachment_geometry_bounds_px": [float(min_x), float(min_y), float(max_x), float(max_y)],
            "required_margin_px": float(margin),
        }

    max_padding = max(96, int(round(max(image.width, image.height) * 0.75)))
    if max(pad_left, pad_top, pad_right, pad_bottom) > max_padding:
        raise ValueError(
            "attachment_geometry_canvas_padding_excessive:"
            f"{[pad_left, pad_top, pad_right, pad_bottom]}>{max_padding}"
        )
    canvas = Image.new(
        "RGB",
        (image.width + pad_left + pad_right, image.height + pad_top + pad_bottom),
        sampled_edge_background(image),
    )
    canvas.paste(image, (pad_left, pad_top))
    canvas.save(path)
    updated_coords = {idx: (x + pad_left, y + pad_top) for idx, (x, y) in coords.items()}
    updated_mark_geometry = _shift_mark_geometry_pixels(mark_geometry, float(pad_left), float(pad_top))
    return updated_coords, updated_mark_geometry, canvas.width, canvas.height, {
        "enabled": True,
        "padding_px": [int(pad_left), int(pad_top), int(pad_right), int(pad_bottom)],
        "attachment_geometry_bounds_before_padding_px": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "required_margin_px": float(margin),
        "operations": ["synchronized_final_attachment_geometry_canvas_padding"],
        "coordinate_mutation_after_render": True,
        "policy": (
            "after_custom_terminal_mark_redraw_expand_canvas_and_shift_atom_endpoint_connector_mark_pixels_"
            "before_pixel_audit_and_molnextr_cropwhite_resize"
        ),
    }


def match_attachment_domain(path: Path, render_style: str, rng: random.Random) -> dict[str, Any]:
    from io import BytesIO
    from PIL import Image, ImageFilter
    import numpy as np

    image = Image.open(path).convert("L")
    operations: list[str] = []
    if render_style == "rdkit_real_crop_bold_scan":
        radius = rng.uniform(0.16, 0.42)
    elif render_style in {"rdkit_real_crop_thin", "rdkit_literature_sparse_page"}:
        radius = rng.uniform(0.06, 0.22)
    else:
        radius = rng.uniform(0.08, 0.30)
    if radius > 0.05:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        operations.append("post_attachment_domain_blur")
    if rng.random() < 0.46 or render_style == "rdkit_real_crop_bold_scan":
        arr = np.asarray(image).astype(np.int16)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(0.3, 1.8), arr.shape)
        image = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"), mode="L")
        operations.append("post_attachment_shared_noise")
    if rng.random() < 0.42:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(78, 94))
        buffer.seek(0)
        image = Image.open(buffer).convert("L")
        operations.append("post_attachment_jpeg_roundtrip")
    image.convert("RGB").save(path)
    arr = np.asarray(image)
    dark_ratio = float((arr < 245).mean())
    return {
        "operations": operations,
        "dark_pixel_ratio": dark_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
    }


def match_real_tight_wavy_domain(path: Path, rng: random.Random) -> dict[str, Any]:
    from io import BytesIO
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np

    image = Image.open(path).convert("L")
    operations: list[str] = []
    image = ImageOps.autocontrast(image, cutoff=rng.choice([0, 1]))
    operations.append("real_tight_wavy_autocontrast")
    contrast = rng.uniform(1.35, 1.85)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    operations.append("real_tight_wavy_contrast")
    if rng.random() < 0.78:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(1.08, 1.35))
        operations.append("real_tight_wavy_sharpen")
    arr = np.asarray(image).astype(np.int16)
    darken = rng.uniform(0.74, 0.86)
    ink = arr < 246
    arr[ink] = np.clip(arr[ink] * darken, 0, 255)
    operations.append("real_tight_wavy_dark_strokes")
    # Higher noise to match real scan statistics (sigma 2-8 vs old 0.12-0.42)
    noise_sigma = rng.uniform(2.0, 8.0)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, noise_sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    operations.append(f"real_tight_wavy_scan_noise_sigma{noise_sigma:.1f}")
    image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L")
    # JPEG compression at realistic patent quality (40-75, not 90-97)
    if rng.random() < 0.80:
        jpeg_quality = rng.randint(40, 75)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("L")
        operations.append(f"real_tight_wavy_jpeg_q{jpeg_quality}")

    image.convert("RGB").save(path)
    arr_after = np.asarray(image)
    dark_ratio = float((arr_after < 245).mean())
    ink_ratio_220 = float((arr_after < 220).mean())
    return {
        "operations": operations,
        "dark_pixel_ratio": dark_ratio,
        "ink_ratio_lt220": ink_ratio_220,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
        "real_tight_wavy_domain": True,
        "target_ink_ratio_lt220": [0.09, 0.14],
    }


def add_document_context(
    path: Path,
    coords: dict[int, tuple[float, float]],
    rng: random.Random,
    *,
    enabled: bool,
    anchor_index: int | None = None,
    dummy_index: int | None = None,
    target_side: str = "",
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    import numpy as np

    image = Image.open(path).convert("RGB")
    operations: list[str] = []
    paper_profile = "none"
    paper_background = 255
    paper_noise_sigma = 0.0
    font = ImageFont.load_default()
    for font_path in rng.sample(DOCUMENT_FONTS, k=len(DOCUMENT_FONTS)):
        try:
            font = ImageFont.truetype(font_path, size=rng.choice([8, 9, 10, 11]))
            break
        except Exception:
            continue

    if enabled:
        base_w, base_h = image.size
        pad_left = int(round(base_w * rng.uniform(0.02, 0.24)))
        pad_right = int(round(base_w * rng.uniform(0.04, 0.32)))
        pad_top = int(round(base_h * rng.uniform(0.03, 0.30)))
        pad_bottom = int(round(base_h * rng.uniform(0.04, 0.34)))
        if target_side == "top":
            pad_top = int(round(base_h * rng.uniform(0.01, 0.05)))
            pad_bottom = int(round(base_h * rng.uniform(0.18, 0.36)))
        elif target_side == "bottom":
            pad_top = int(round(base_h * rng.uniform(0.18, 0.36)))
            pad_bottom = int(round(base_h * rng.uniform(0.01, 0.05)))
        elif target_side == "left":
            pad_left = int(round(base_w * rng.uniform(0.01, 0.05)))
            pad_right = int(round(base_w * rng.uniform(0.18, 0.36)))
        elif target_side == "right":
            pad_left = int(round(base_w * rng.uniform(0.18, 0.36)))
            pad_right = int(round(base_w * rng.uniform(0.01, 0.05)))
        profile_specs = {
            "clean_white": ([253, 254, 255], (0.05, 0.45), (0.00, 0.18)),
            "white_scan": ([250, 251, 252, 253, 254], (0.35, 1.25), (0.10, 0.45)),
            "gray_scan": ([246, 247, 248, 249, 250, 251, 252], (0.85, 2.20), (0.20, 0.85)),
            "aged_scan": ([242, 243, 244, 245, 246, 247, 248], (0.65, 1.80), (0.15, 0.70)),
        }
        paper_profile = rng.choices(
            ["clean_white", "white_scan", "gray_scan", "aged_scan"],
            weights=[0.24, 0.34, 0.32, 0.10],
            k=1,
        )[0]
        background_values, noise_range, wave_range = profile_specs[paper_profile]
        paper_background = int(rng.choice(background_values))
        paper_noise_sigma = float(rng.uniform(*noise_range))
        canvas_size = (base_w + pad_left + pad_right, base_h + pad_top + pad_bottom)
        arr_bg = np.full((canvas_size[1], canvas_size[0]), paper_background, dtype=np.int16)
        arr_bg = np.clip(
            arr_bg + np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, paper_noise_sigma, arr_bg.shape),
            0,
            255,
        ).astype("uint8")
        yy, xx = np.mgrid[0 : canvas_size[1], 0 : canvas_size[0]]
        paper_wave = (
            np.sin((xx + rng.uniform(0, 1000)) / rng.uniform(38.0, 86.0))
            + np.cos((yy + rng.uniform(0, 1000)) / rng.uniform(44.0, 112.0))
        ) * rng.uniform(*wave_range)
        arr_bg = np.clip(arr_bg.astype(np.float32) + paper_wave, 0, 255).astype("uint8")
        arr_struct = np.asarray(image.convert("L")).astype(np.float32)
        alpha = np.clip((252.0 - arr_struct) / 42.0, 0.0, 1.0)
        alpha = np.power(alpha, 0.82)
        ink_values = np.clip(arr_struct * rng.uniform(0.82, 0.96), 0, 255)
        patch = arr_bg[pad_top : pad_top + base_h, pad_left : pad_left + base_w].astype(np.float32)
        patch = patch * (1.0 - alpha) + np.minimum(patch, ink_values) * alpha
        arr_bg[pad_top : pad_top + base_h, pad_left : pad_left + base_w] = np.clip(patch, 0, 255).astype("uint8")
        image = Image.fromarray(arr_bg, mode="L").convert("RGB")
        coords = {idx: (x + pad_left, y + pad_top) for idx, (x, y) in coords.items()}
        operations.append("asymmetric_white_document_margin")
        operations.append("alpha_antialiased_transparent_white_structure_composite")
        operations.append("diverse_patent_paper_before_structure_composite")
        operations.append(f"paper_profile:{paper_profile}")

        draw = ImageDraw.Draw(image)
        protected_points = []
        if anchor_index is not None and dummy_index is not None and anchor_index in coords and dummy_index in coords:
            ax, ay = coords[int(anchor_index)]
            ex, ey = coords[int(dummy_index)]
            ux, uy, _, _, norm = connector_basis((ax, ay), (ex, ey))
            protected_points = [
                (ex, ey, 34.0),
                (ex + ux * max(18.0, norm * 0.35), ey + uy * max(18.0, norm * 0.35), 42.0),
            ]

        def near_protected_box(box: tuple[int, int, int, int]) -> bool:
            if not protected_points:
                return False
            left, top, right, bottom = box
            for px, py, radius in protected_points:
                closest_x = min(max(px, left), right)
                closest_y = min(max(py, top), bottom)
                if math.hypot(px - closest_x, py - closest_y) < radius:
                    return True
            return False

        ink = rng.choice([(0, 0, 0), (15, 15, 15), (30, 30, 30), (45, 45, 45)])
        if rng.random() < 0.32 and pad_top >= 10:
            label = rng.choice(["(I)", "(II)", "A", "B", "Scheme", "Ex."])
            xy = (rng.randint(2, max(2, image.width // 3)), max(1, pad_top // 3))
            bbox = draw.textbbox(xy, label, font=font)
            if not near_protected_box(bbox):
                draw.text(xy, label, fill=ink, font=font)
                operations.append("small_margin_caption")
        if rng.random() < 0.26 and pad_bottom >= 10:
            label = rng.choice(["R", "X", "n=1", "Ar", "continued", "or pharmaceutically acceptable salt"])
            xy = (rng.randint(2, max(2, image.width // 2)), base_h + pad_top + max(1, pad_bottom // 4))
            bbox = draw.textbbox(xy, label, font=font)
            if not near_protected_box(bbox):
                draw.text(xy, label, fill=ink, font=font)
                operations.append("bottom_margin_text")
        if rng.random() < 0.14:
            rule_candidates = []
            if pad_top >= 18:
                rule_candidates.append(max(2, pad_top // 3))
            if pad_bottom >= 18:
                rule_candidates.append(base_h + pad_top + max(2, (2 * pad_bottom) // 3))
            if rule_candidates:
                y = rng.choice(rule_candidates)
                x1 = rng.randint(0, max(0, image.width // 8))
                x2 = image.width - rng.randint(0, max(0, image.width // 8))
                y2 = y + rng.choice([-1, 0, 1])
                if not near_protected_box((min(x1, x2), min(y, y2) - 2, max(x1, x2), max(y, y2) + 2)):
                    draw.line((x1, y, x2, y2), fill=ink, width=1)
                    operations.append("thin_document_rule")

        if rng.random() < 0.10:
            for _ in range(rng.randint(1, 3)):
                x = rng.randint(0, max(0, image.width - 1))
                draw.line((x, 0, x + rng.choice([-1, 0, 1]), image.height), fill=(230, 230, 230), width=1)
            operations.append("faint_scan_column_artifact")

    if rng.random() < 0.34:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.45)))
        operations.append("light_scan_blur")
    if rng.random() < 0.38:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(0.6, 2.8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        image = Image.fromarray(arr, mode="L").convert("RGB")
        operations.append("light_scan_noise")
    if rng.random() < 0.30:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(74, 92))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        operations.append("jpeg_roundtrip")

    image.save(path)
    return coords, image.width, image.height, {
        "enabled": enabled,
        "operations": operations,
        "paper_profile": paper_profile,
        "paper_background_gray": int(paper_background),
        "paper_noise_sigma": float(paper_noise_sigma),
        "transparent_white_structure_composite": "alpha_antialiased",
        "target_side_preservation": {
            "enabled": bool(target_side),
            "target_side": target_side,
            "policy": "bias_synchronized_document_margin_to_preserve_requested_endpoint_side_without_post_render_rotation",
        },
    }


def rotate_page_context(
    path: Path,
    coords: dict[int, tuple[float, float]],
    rng: random.Random,
    *,
    enabled: bool,
) -> tuple[dict[int, tuple[float, float]], int, int, dict[str, Any]]:
    if not enabled or rng.random() >= 0.36:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return coords, image.width, image.height, {"rotated": False}

    from PIL import Image

    image = Image.open(path).convert("RGB")
    angle = rng.uniform(-5.0, 5.0)
    old_w, old_h = image.size
    rotated = image.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC)
    new_w, new_h = rotated.size
    cx, cy = old_w / 2.0, old_h / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0
    theta = math.radians(-angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    updated = {}
    for idx, (x, y) in coords.items():
        dx, dy = x - cx, y - cy
        updated[idx] = (dx * cos_t - dy * sin_t + ncx, dx * sin_t + dy * cos_t + ncy)
    rotated.save(path)
    return updated, new_w, new_h, {"rotated": True, "angle_degrees": angle, "expand": True}


def _warp_displacement_field(
    width: int,
    height: int,
    *,
    seed: int,
    amplitude_px: float,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    rng = random.Random(int(seed))
    width = max(1, int(width))
    height = max(1, int(height))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    max_amplitude = max(0.0, min(float(amplitude_px), min(width, height) * 0.016))
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
            period = rng.uniform(min(width, height) * 0.85, min(width, height) * 2.05)
            angle = rng.uniform(0.0, math.pi)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            component_amp = max_amplitude * rng.uniform(0.16, 0.48)
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
    return dx, dy, {
        "enabled": True,
        "amplitude_px": float(max_amplitude),
        "max_abs_dx_px": float(np.max(np.abs(dx))) if dx.size else 0.0,
        "max_abs_dy_px": float(np.max(np.abs(dy))) if dy.size else 0.0,
        "components": components,
    }


def _sample_displacement(dx: Any, dy: Any, x: float, y: float) -> tuple[float, float]:
    height, width = dx.shape
    x = max(0.0, min(float(width - 1), float(x)))
    y = max(0.0, min(float(height - 1), float(y)))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    def interp(arr: Any) -> float:
        return float(
            arr[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + arr[y0, x1] * tx * (1.0 - ty)
            + arr[y1, x0] * (1.0 - tx) * ty
            + arr[y1, x1] * tx * ty
        )

    return interp(dx), interp(dy)


def _warp_pixel_point(dx: Any, dy: Any, x: float, y: float) -> tuple[float, float]:
    ddx, ddy = _sample_displacement(dx, dy, float(x), float(y))
    return float(x) + ddx, float(y) + ddy


def _unit_vector_from_points(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float, float]:
    vx = float(end[0]) - float(start[0])
    vy = float(end[1]) - float(start[1])
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return 1.0, 0.0, 1.0
    return vx / norm, vy / norm, norm


def _pixel_point_dict(point: tuple[float, float]) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1])}


def _normalized_from_pixel(point: tuple[float, float], width: int, height: int) -> dict[str, float]:
    return normalize_coord(float(point[0]), float(point[1]), int(width), int(height))


def _normalize_sampled_geometry_points(geometry: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    updated = json.loads(json.dumps(geometry)) if isinstance(geometry, dict) else {}

    def normalize_points(source_key: str, target_key: str) -> None:
        points = updated.get(source_key) if isinstance(updated.get(source_key), list) else []
        normalized: list[list[float]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            norm = _normalized_from_pixel((float(point.get("x") or 0.0), float(point.get("y") or 0.0)), width, height)
            normalized.append([float(norm["x"]), float(norm["y"])])
        if normalized:
            updated[target_key] = normalized

    normalize_points("connector_sampled_points_px", "connector_sampled_points_normalized")
    normalize_points("terminal_mark_sampled_points_px", "terminal_mark_sampled_points_normalized")
    endpoints = updated.get("terminal_mark_endpoints_px") if isinstance(updated.get("terminal_mark_endpoints_px"), list) else []
    normalized_endpoints: list[list[float]] = []
    for point in endpoints:
        if isinstance(point, dict):
            norm = _normalized_from_pixel((float(point.get("x") or 0.0), float(point.get("y") or 0.0)), width, height)
            normalized_endpoints.append([float(norm["x"]), float(norm["y"])])
    if normalized_endpoints:
        updated["terminal_mark_endpoints_normalized"] = normalized_endpoints
    return updated


def _line_samples_pixels(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    samples: int = 17,
) -> list[tuple[float, float]]:
    count = max(2, int(samples))
    return [
        (
            float(start[0]) + (float(end[0]) - float(start[0])) * index / (count - 1),
            float(start[1]) + (float(end[1]) - float(start[1])) * index / (count - 1),
        )
        for index in range(count)
    ]


def _warp_mark_geometry(
    geometry: dict[str, Any],
    dx: Any,
    dy: Any,
    *,
    anchor_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
) -> tuple[dict[str, Any], int]:
    updated = json.loads(json.dumps(geometry)) if isinstance(geometry, dict) else {}
    warped_anchor = _warp_pixel_point(dx, dy, anchor_xy[0], anchor_xy[1])
    warped_endpoint = _warp_pixel_point(dx, dy, endpoint_xy[0], endpoint_xy[1])
    ux, uy, connector_length = _unit_vector_from_points(warped_anchor, warped_endpoint)
    px, py = -uy, ux
    outside = 0
    height, width = dx.shape

    def warp_point_field(container: dict[str, Any], key: str) -> tuple[float, float] | None:
        nonlocal outside
        point = container.get(key) if isinstance(container.get(key), dict) else {}
        if not point:
            return None
        x = float(point.get("x") or 0.0)
        y = float(point.get("y") or 0.0)
        wx, wy = _warp_pixel_point(dx, dy, x, y)
        if not (0.0 <= wx <= float(width - 1) and 0.0 <= wy <= float(height - 1)):
            outside += 1
        container[key] = _pixel_point_dict((wx, wy))
        return wx, wy

    mark_center = warp_point_field(updated, "mark_center")
    warp_point_field(updated, "mark_label_position")
    warp_point_field(updated, "visible_connector_start")
    updated["connector_vector"] = {"x": float(ux), "y": float(uy)}
    updated["mark_axis_vector"] = {"x": float(px), "y": float(py)}
    updated["connector_length_px"] = float(connector_length)
    updated["mark_axis_dot_connector_abs"] = float(abs(px * ux + py * uy))

    mark_length = float(
        updated.get("mark_length_px")
        or updated.get("wavy_cut_length_px")
        or updated.get("wavy_bond_length_px")
        or 0.0
    )
    if mark_length > 0.0:
        updated["mark_length_to_connector_ratio"] = float(mark_length / max(connector_length, 1e-6))
        if "wavy_cut_length_px" in updated or "wavy_bond_length_px" in updated:
            updated["wavy_length_to_connector_ratio"] = float(mark_length / max(connector_length, 1e-6))
    if "wavy_axis_vector" in updated:
        updated["wavy_axis_vector"] = {"x": float(px), "y": float(py)}
        updated["wavy_axis_dot_connector_abs"] = float(abs(px * ux + py * uy))

    if is_custom_perpendicular_wavy_geometry(str(updated.get("geometry") or "")) and mark_length > 0.0:
        visible = updated.get("visible_connector_start") if isinstance(updated.get("visible_connector_start"), dict) else {}
        try:
            visible_start = (float(visible.get("x")), float(visible.get("y")))
        except (TypeError, ValueError):
            visible_start = warped_anchor
        junction_style = str(updated.get("wavy_connector_intersection_style") or "center_cross")
        side_sign = 1.0
        existing_center = updated.get("wavy_center") if isinstance(updated.get("wavy_center"), dict) else {}
        try:
            old_center = (float(existing_center.get("x")), float(existing_center.get("y")))
            side_sign = 1.0 if ((old_center[0] - endpoint_xy[0]) * px + (old_center[1] - endpoint_xy[1]) * py) >= 0 else -1.0
        except (TypeError, ValueError):
            pass
        if junction_style == "side_touch":
            wavy_center = (
                warped_endpoint[0] + px * side_sign * mark_length * 0.5,
                warped_endpoint[1] + py * side_sign * mark_length * 0.5,
            )
        else:
            wavy_center = warped_endpoint
        wavy_start = (wavy_center[0] - px * mark_length * 0.5, wavy_center[1] - py * mark_length * 0.5)
        wavy_end = (wavy_center[0] + px * mark_length * 0.5, wavy_center[1] + py * mark_length * 0.5)
        amplitude = float(updated.get("wavy_amplitude_px") or 2.4)
        cycles = float(updated.get("wavy_cycles") or max(2.2, min(4.8, mark_length / 8.6)))
        segments = int(updated.get("wavy_segments") or max(18, min(52, round(cycles * 10.0))))
        if junction_style == "center_cross":
            center_style = center_cross_perpendicular_wavy_style(
                {
                    "length": mark_length,
                    "amplitude": amplitude,
                    "cycles": cycles,
                    "segments": segments,
                }
            )
            cycles = float(center_style["cycles"])
            segments = int(center_style["segments"])
            updated["wavy_cycles"] = cycles
            updated["wavy_segments"] = segments
            updated["center_cross_curve_zero_at_t"] = center_style.get("center_cross_curve_zero_at_t")
        wavy_points = markush_wavy_points(
            wavy_start,
            wavy_end,
            amplitude=amplitude,
            cycles=cycles,
            segments=segments,
            rng=None,
        )
        if junction_style == "center_cross":
            wavy_points = force_polyline_center_point(wavy_points, wavy_center)
        bridge_past = float(updated.get("mark_connector_bridge_past_endpoint_px") or 0.0)
        draw_start = updated.get("connector_draw_start") if isinstance(updated.get("connector_draw_start"), dict) else None
        if draw_start is not None:
            warped_draw_start = _warp_pixel_point(dx, dy, float(draw_start["x"]), float(draw_start["y"]))
        else:
            warped_draw_start = visible_start
        connector_draw_end = (warped_endpoint[0] + ux * bridge_past, warped_endpoint[1] + uy * bridge_past)
        connector_samples = _line_samples_pixels(warped_draw_start, connector_draw_end, samples=15)
        updated["mark_axis_vector"] = {"x": float(px), "y": float(py)}
        updated["mark_axis_dot_connector_abs"] = float(abs(px * ux + py * uy))
        updated["mark_center"] = _pixel_point_dict(wavy_center)
        updated["wavy_center"] = _pixel_point_dict(wavy_center)
        updated["wavy_start"] = _pixel_point_dict(wavy_start)
        updated["wavy_end"] = _pixel_point_dict(wavy_end)
        updated["wavy_connector_contact_point"] = _pixel_point_dict(warped_endpoint)
        updated["straight_connector_end"] = _pixel_point_dict(warped_endpoint)
        updated["connector_draw_end"] = _pixel_point_dict(connector_draw_end)
        updated["connector_draw_start"] = _pixel_point_dict(warped_draw_start)
        updated["connector_wavy_center_distance_px"] = float(point_distance(wavy_center, warped_endpoint))
        updated["connector_contact_is_terminal_wavy_endpoint"] = junction_style == "side_touch"
        updated["wavy_length_px"] = float(mark_length)
        updated["wavy_bond_length_px"] = float(mark_length)
        updated["wavy_polyline_length_px"] = polyline_length(wavy_points)
        updated["wavy_length_to_connector_ratio"] = float(mark_length / max(connector_length, 1e-6))
        updated["mark_length_to_connector_ratio"] = float(mark_length / max(connector_length, 1e-6))
        updated["wavy_axis_vector"] = {"x": float(px), "y": float(py)}
        updated["wavy_axis_dot_connector_abs"] = float(abs(px * ux + py * uy))
        updated["connector_sampled_points_px"] = pixel_dicts(connector_samples)
        updated["wavy_sampled_points_px"] = pixel_dicts(wavy_points)
        updated["terminal_mark_endpoints_px"] = pixel_dicts([wavy_start, wavy_end])
        updated["terminal_mark_sampled_points_px"] = pixel_dicts(wavy_points)
    else:
        connector_samples = _line_samples_pixels(warped_anchor, warped_endpoint, samples=19)
        updated["connector_sampled_points_px"] = [_pixel_point_dict(point) for point in connector_samples]
    if mark_center is not None and mark_length > 0.0 and not is_custom_perpendicular_wavy_geometry(str(updated.get("geometry") or "")):
        start = (mark_center[0] - px * mark_length * 0.5, mark_center[1] - py * mark_length * 0.5)
        end = (mark_center[0] + px * mark_length * 0.5, mark_center[1] + py * mark_length * 0.5)
        updated["terminal_mark_endpoints_px"] = [_pixel_point_dict(start), _pixel_point_dict(end)]
        updated["terminal_mark_sampled_points_px"] = [
            _pixel_point_dict(point) for point in _line_samples_pixels(start, end, samples=15)
        ]
    updated["nonlinear_warp_synchronized"] = True
    updated["geometry_recomputed_after_warp"] = True
    return updated, outside


def audit_fragment_formal_nonlinear_contract(
    *,
    atom_coordinates: list[dict[str, Any]],
    render_quality: dict[str, Any],
) -> dict[str, Any]:
    atom_by_index = {}
    for atom in atom_coordinates:
        if not isinstance(atom, dict):
            continue
        try:
            atom_by_index[int(atom.get("atom_index"))] = atom
        except (TypeError, ValueError):
            continue
    blockers: list[str] = []
    graph = render_quality.get("graph_consistency") if isinstance(render_quality.get("graph_consistency"), dict) else {}
    dummy_index_value = render_quality.get("attachment_dummy_index")
    if dummy_index_value is None:
        dummy_index_value = graph.get("dummy_index")
    anchor_index_value = render_quality.get("attachment_anchor_index")
    if anchor_index_value is None:
        anchor_index_value = graph.get("anchor_index")
    dummy_index = int(dummy_index_value if dummy_index_value is not None else -1)
    anchor_index = int(anchor_index_value if anchor_index_value is not None else -1)
    endpoint = render_quality.get("attachment_endpoint") if isinstance(render_quality.get("attachment_endpoint"), dict) else {}
    anchor_coord = render_quality.get("attachment_anchor_coord") if isinstance(render_quality.get("attachment_anchor_coord"), dict) else {}
    dummy_atom = atom_by_index.get(dummy_index)
    anchor_atom = atom_by_index.get(anchor_index)

    def close_point(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-6) -> bool:
        try:
            return abs(float(left.get("x")) - float(right.get("x"))) <= tolerance and abs(
                float(left.get("y")) - float(right.get("y"))
            ) <= tolerance
        except (TypeError, ValueError):
            return False

    endpoint_matches_dummy = bool(dummy_atom and close_point(endpoint, dummy_atom))
    anchor_coord_matches_anchor_atom = bool(anchor_atom and close_point(anchor_coord, anchor_atom))
    anchor_symbol = str(render_quality.get("attachment_anchor_symbol") or render_quality.get("attachment_anchor") or "")
    anchor_label_matches_atom_token = bool(anchor_atom and anchor_symbol == str(anchor_atom.get("token") or ""))
    single_dummy_atom = graph.get("single_dummy_atom") is True
    anchor_dummy_bond_present = graph.get("anchor_dummy_bond_present") is True
    connector = render_quality.get("attachment_connector") if isinstance(render_quality.get("attachment_connector"), dict) else {}
    connector_samples = connector.get("sampled_points_normalized") if isinstance(connector.get("sampled_points_normalized"), list) else []
    connector_samples_valid = len(connector_samples) >= 3 and all(
        isinstance(point, list)
        and len(point) == 2
        and 0.0 <= float(point[0]) <= 1.0
        and 0.0 <= float(point[1]) <= 1.0
        for point in connector_samples
    )
    mode = str(render_quality.get("attachment_render_mode") or "")
    mark_geometry = (
        render_quality.get("wavy_geometry")
        if mode == "wavy" and isinstance(render_quality.get("wavy_geometry"), dict)
        else render_quality.get("fragment_mark_geometry")
        if isinstance(render_quality.get("fragment_mark_geometry"), dict)
        else {}
    )
    terminal_samples = (
        mark_geometry.get("terminal_mark_sampled_points_normalized")
        if isinstance(mark_geometry.get("terminal_mark_sampled_points_normalized"), list)
        else []
    )
    if mode in {"cut", "wavy"}:
        terminal_mark_samples_valid = len(terminal_samples) >= 3 and all(
            isinstance(point, list)
            and len(point) == 2
            and 0.0 <= float(point[0]) <= 1.0
            and 0.0 <= float(point[1]) <= 1.0
            for point in terminal_samples
        )
    else:
        terminal_mark_samples_valid = bool(mark_geometry.get("nonlinear_warp_synchronized"))

    checks = {
        "endpoint_matches_dummy_atom": endpoint_matches_dummy,
        "anchor_coord_matches_anchor_atom": anchor_coord_matches_anchor_atom,
        "anchor_label_matches_atom_token": anchor_label_matches_atom_token,
        "single_dummy_atom": single_dummy_atom,
        "anchor_dummy_bond_present": anchor_dummy_bond_present,
        "connector_samples_valid": connector_samples_valid,
        "terminal_mark_samples_valid": terminal_mark_samples_valid,
    }
    blockers.extend([name for name, passed in checks.items() if not passed])
    endpoint_length_policy = (
        render_quality.get("endpoint_length_policy")
        if isinstance(render_quality.get("endpoint_length_policy"), dict)
        else {}
    )
    connector_length = float(endpoint_length_policy.get("connector_length_px") or 0.0)
    minimum_connector_length = float(endpoint_length_policy.get("minimum_connector_length_px") or 0.0)
    if minimum_connector_length > 0.0 and connector_length + 1e-6 < minimum_connector_length:
        blockers.append("connector_length_below_declared_minimum")

    return {
        "schema_version": "fragment_formal_nonlinear_document_warp_contract_v1",
        "policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
        "passed": not blockers,
        "blockers": blockers,
        **checks,
        "connector_length_px": float(connector_length),
        "minimum_connector_length_px": float(minimum_connector_length),
        "attachment_direction": str(render_quality.get("attachment_direction") or ""),
        "attachment_render_mode": mode,
        "metric_policy": "normalized_endpoint_dummy_anchor_closure_plus_warped_connector_and_terminal_mark_samples",
        "does_not_replace_visual_attachment_role_source_leak_router_model_scale_or_readiness_gates": True,
    }


def apply_fragment_formal_nonlinear_document_warp(
    path: Path,
    coords: dict[int, tuple[float, float]],
    mark_geometry: dict[str, Any],
    *,
    anchor_index: int,
    dummy_index: int,
    seed: int,
    enabled: bool,
    amplitude_px: float,
) -> tuple[dict[int, tuple[float, float]], dict[str, Any], int, int, dict[str, Any]]:
    from PIL import Image, ImageFilter
    import numpy as np

    image = Image.open(path).convert("RGB")
    width, height = image.size
    if not enabled:
        return coords, mark_geometry, width, height, {
            "schema_version": FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION,
            "enabled": False,
            "operations": [],
        }
    if int(anchor_index) not in coords or int(dummy_index) not in coords:
        raise ValueError("fragment_formal_nonlinear_warp_missing_anchor_or_dummy_coordinate")
    dx, dy, field = _warp_displacement_field(width, height, seed=int(seed), amplitude_px=float(amplitude_px))
    if not field.get("enabled"):
        return coords, mark_geometry, width, height, {
            "schema_version": FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION,
            "enabled": False,
            "operations": [],
        }

    arr = np.asarray(image)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
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
        mesh = []
        grid = 16
        for y0 in range(0, height, grid):
            for x0 in range(0, width, grid):
                x1 = min(width, x0 + grid)
                y1 = min(height, y0 + grid)
                cx = (x0 + x1 - 1) * 0.5
                cy = (y0 + y1 - 1) * 0.5
                ddx, ddy = _sample_displacement(dx, dy, cx, cy)
                mesh.append(
                    (
                        (x0, y0, x1, y1),
                        (x0 - ddx, y0 - ddy, x1 - ddx, y0 - ddy, x1 - ddx, y1 - ddy, x0 - ddx, y1 - ddy),
                    )
                )
        image = image.transform((width, height), Image.Transform.MESH, mesh, resample=Image.Resampling.BILINEAR)
        sampler = "pil_mesh_grid16_bilinear"

    rng = random.Random(int(seed) ^ 0x51F00D)
    if rng.random() < 0.72:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.03, 0.14)))
    image.save(path)

    updated_coords: dict[int, tuple[float, float]] = {}
    atom_outside = 0
    for index, (x, y) in coords.items():
        wx, wy = _warp_pixel_point(dx, dy, x, y)
        if not (0.0 <= wx <= float(width - 1) and 0.0 <= wy <= float(height - 1)):
            atom_outside += 1
        updated_coords[int(index)] = (max(0.0, min(float(width - 1), wx)), max(0.0, min(float(height - 1), wy)))

    updated_mark_geometry = json.loads(json.dumps(mark_geometry))
    mark_outside = 0
    if "fragment_mark_geometry" in updated_mark_geometry:
        updated_mark_geometry["fragment_mark_geometry"], outside = _warp_mark_geometry(
            updated_mark_geometry["fragment_mark_geometry"],
            dx,
            dy,
            anchor_xy=coords[int(anchor_index)],
            endpoint_xy=coords[int(dummy_index)],
        )
        mark_outside += outside
    if "wavy_geometry" in updated_mark_geometry:
        updated_mark_geometry["wavy_geometry"], outside = _warp_mark_geometry(
            updated_mark_geometry["wavy_geometry"],
            dx,
            dy,
            anchor_xy=coords[int(anchor_index)],
            endpoint_xy=coords[int(dummy_index)],
        )
        mark_outside += outside

    arr_after = np.asarray(image.convert("L"))
    dark_ratio = float((arr_after < 245).mean())
    ink_ratio = float((arr_after < 220).mean())
    contract = {
        "schema_version": FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION,
        "enabled": True,
        "policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
        "formal_training_allowed": True,
        "research_only": False,
        "debug_only": False,
        "operations": [
            "smooth_low_amplitude_fragment_document_warp",
            "synchronized_atom_endpoint_connector_terminal_mark_warp",
            "fragment_endpoint_connector_mark_contract_gate_required",
        ],
        "seed": int(seed),
        "field": field,
        "sampler": sampler,
        "image_size": [int(width), int(height)],
        "coordinate_mutation_policy": "formal_synchronized_image_atom_endpoint_connector_terminal_mark_warp",
        "image_synchronized": True,
        "atom_coordinates_synchronized": True,
        "endpoint_coordinates_synchronized": True,
        "connector_anchors_synchronized": True,
        "terminal_mark_geometry_synchronized": True,
        "query_or_dummy_terminal_label_synchronized": True,
        "atom_outside_count": int(atom_outside),
        "mark_anchor_outside_count": int(mark_outside),
        "dark_pixel_ratio": dark_ratio,
        "ink_pixel_ratio": ink_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
        "limitations": [
            "Rows are formal-capable only when fragment_nonlinear_pose_preservation.passed=true.",
            "Still requires schema, visual, attachment-role, source-leak, router, model-scale, runtime, and readiness gates.",
        ],
    }
    return updated_coords, updated_mark_geometry, width, height, contract


def choose_visual_shape(index: int, rng: random.Random, *, target_mode: str = "", target_side: str = "") -> str:
    if target_mode:
        mapping = {
            "cut": "left_terminal_cut_or_open_stub",
            "wavy": "candidate_terminal_wavy_cut",
            "dummy_atom": "visible_dummy_or_query_label",
            "query_attachment": "visible_dummy_or_query_label",
        }
        if target_mode not in mapping:
            raise ValueError(f"invalid target attachment mode: {target_mode}")
        if target_mode == "cut" and target_side == "bottom":
            return "bottom_crop_attachment_or_cut"
        return mapping[target_mode]
    # Keep positive rows visually explicit: unmarked short stubs are too often
    # indistinguishable from ordinary terminal bonds.
    weighted = (
        ["left_terminal_cut_or_open_stub"] * 30
        + ["bottom_crop_attachment_or_cut"] * 15
        + ["candidate_terminal_wavy_cut"] * 25
        + ["visible_dummy_or_query_label"] * 30
    )
    return weighted[index % len(weighted)] if rng.random() < 0.85 else rng.choice(weighted)


def choose_semantic_family(index: int, rng: random.Random) -> str:
    weighted = ["semantic_r_group_attachment"] * 58 + ["semantic_query_attachment"] * 22 + ["semantic_dummy_attachment"] * 20
    return weighted[index % len(weighted)] if rng.random() < 0.85 else rng.choice(weighted)


def terminal_query_label(index: int, visual_shape: str, *, target_mode: str = "") -> str:
    if visual_shape != "visible_dummy_or_query_label":
        return ""
    if target_mode == "dummy_atom":
        return "*"
    if target_mode == "query_attachment":
        return ["R", "X", "Q"][index % 3]
    label_cycle = ["R", "X", "*", "R", "X", "Q", "*", "R", "*"]
    return label_cycle[index % len(label_cycle)]


def target_visible_anchor_label(mol: Any, anchor_index: int, target_anchor: str) -> str:
    requested = str(target_anchor or "").strip()
    atom = mol.GetAtomWithIdx(int(anchor_index))
    if requested in TARGET_VISIBLE_ANCHOR_LABELS:
        return requested
    return visible_anchor_label(atom)


def generate_row(
    *,
    backbone_smiles: str,
    source_id: str,
    output_dir: Path,
    index: int,
    rng: random.Random,
    image_size: int,
    document_context: bool,
    target_side: str = "",
    target_anchor: str = "",
    target_mode: str = "",
    strict_orientation: bool = False,
    real_tight_crop_style: bool = False,
    enable_formal_nonlinear_document_warp: bool = False,
    formal_nonlinear_warp_amplitude_px: float = 1.4,
) -> dict[str, str]:
    Chem, _, _ = import_rdkit()
    mol = Chem.MolFromSmiles(backbone_smiles)
    if mol is None:
        raise ValueError(f"invalid backbone SMILES: {backbone_smiles}")
    if target_anchor:
        anchor_candidates = target_anchor_candidates(mol, target_anchor)
        if not anchor_candidates:
            raise ValueError(f"no target anchor atoms: {target_anchor}")
        anchor_index = choose_target_anchor_index(mol, anchor_candidates, target_side, rng)
    else:
        anchor_index = choose_anchor(mol, rng)
    requested_anchor_visible_label = target_visible_anchor_label(mol, anchor_index, target_anchor)
    fragment, dummy_index, attachment_bond_index = add_attachment_dummy(mol, anchor_index)
    visual_shape = choose_visual_shape(index, rng, target_mode=target_mode, target_side=target_side)
    semantic_family = choose_semantic_family(index, rng)
    native_terminal_label = terminal_query_label(index, visual_shape, target_mode=target_mode)
    if native_terminal_label and native_terminal_label != "*":
        fragment.GetAtomWithIdx(dummy_index).SetProp("atomLabel", native_terminal_label)
    elif not native_terminal_label:
        fragment.GetAtomWithIdx(dummy_index).SetProp("atomLabel", "")
    set_attachment_bond_style(fragment, attachment_bond_index, visual_shape)
    internal_wavy_stereo = maybe_set_internal_wavy_stereo_bond(
        fragment,
        anchor_index,
        dummy_index,
        attachment_bond_index,
        visual_shape,
        rng,
    )
    requested_side, pre_render_orientation_info = place_dummy(
        fragment,
        anchor_index,
        dummy_index,
        visual_shape,
        rng,
        index=index,
        target_side=target_side,
        strict_target_side=bool(strict_orientation and target_side),
    )
    if real_tight_crop_style:
        style = rng.choice(
            [
                "rdkit_real_crop_thin",
                "rdkit_real_crop_standard",
                "rdkit_literature_rotated_serif",
                "rdkit_literature_sparse_page",
            ]
        )
    else:
        style = rng.choice(STYLE_FAMILIES)
    seed = rng.randint(0, 2**31 - 1)
    # Small-resolution render + LANCZOS upscale to match real patent crop
    # anti-aliasing (real fragments are 80-170px then upscaled to 384 by the
    # pipeline; rendering at 256 native produces crisp lines that don't match).
    native_size = rng.randint(100, 160) if real_tight_crop_style else image_size
    render_width = native_size
    render_height = native_size
    png_bytes, svg_text, coords, draw_style_info = draw_fragment_png(
        fragment,
        width=render_width,
        height=render_height,
        style=style,
        seed=seed,
        visual_shape=visual_shape,
        anchor_index=anchor_index,
        dummy_index=dummy_index,
        allow_renderer_rotation=not bool(strict_orientation),
    )
    if anchor_index not in coords or dummy_index not in coords:
        raise ValueError("drawer missing anchor or dummy coordinates")

    # Upscale the rendered PNG to image_size with LANCZOS to simulate the
    # real pipeline's resize step (produces gray/pixelated anti-aliasing).
    if native_size != image_size:
        from PIL import Image as PILImage
        import io as _io
        pil_img = PILImage.open(_io.BytesIO(png_bytes))
        pil_img = pil_img.resize((image_size, image_size), PILImage.LANCZOS)
        buf = _io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        scale = image_size / native_size
        for k in coords:
            cx, cy = coords[k]
            coords[k] = (cx * scale, cy * scale)
    row_id = stable_id("fragment_attachment", source_id, backbone_smiles, index, seed, visual_shape, semantic_family, document_context)
    image_rel = Path("images") / f"{row_id}.png"
    image_path = output_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(png_bytes)
    anchor_depiction = rendered_anchor_depiction_mode(fragment, anchor_index, svg_text)
    coords, width, height, crop_info = crop_to_real_fragment(
        image_path,
        coords,
        rng,
        anchor_index=anchor_index,
        dummy_index=dummy_index,
        target_side=target_side if strict_orientation else "",
        real_tight_crop_style=bool(real_tight_crop_style),
    )
    coords, width, height, endpoint_length_info = require_endpoint_pixel_length(
        image_path,
        coords,
        anchor_index,
        dummy_index,
        min_length_px=(
            0.0
            if visual_shape == "candidate_terminal_wavy_cut"
            else (22.0 if strict_orientation or target_side else 0.0)
        ),
    )
    coords, width, height, side_preserving_padding_info = preserve_endpoint_side_with_canvas_padding(
        image_path,
        coords,
        dummy_index=dummy_index,
        target_side=target_side if strict_orientation else "",
    )
    if strict_orientation and target_side:
        endpoint_coord_pre_rotation = normalize_coord(coords[dummy_index][0], coords[dummy_index][1], width, height)
        if side_from_endpoint(endpoint_coord_pre_rotation["x"], endpoint_coord_pre_rotation["y"]) != target_side:
            raise ValueError("strict_orientation_target_side_requires_native_layout_match")
        target_side_rotation_info = {
            "enabled": True,
            "turns_ccw": 0,
            "target_side": target_side,
            "coordinate_mutation_after_render": False,
            "policy": "strict_orientation_rejects_post_render_target_side_rotation",
        }
    else:
        coords, width, height, target_side_rotation_info = rotate_to_target_endpoint_side(
            image_path,
            coords,
            dummy_index=dummy_index,
            target_side=target_side,
        )
    coords, width, height, min_canvas_info = ensure_min_canvas_side(
        image_path,
        coords,
        min_side_px=0 if real_tight_crop_style else 88,
    )
    coords, width, height, document_context_info = add_document_context(
        image_path,
        coords,
        rng,
        enabled=document_context and not bool(real_tight_crop_style),
        anchor_index=anchor_index,
        dummy_index=dummy_index,
        target_side=target_side if strict_orientation else "",
    )
    coords, width, height, page_rotation_info = rotate_page_context(
        image_path,
        coords,
        rng,
        enabled=document_context and not bool(strict_orientation),
    )
    pose_alignment_info = {
        "image_to_graph_orientation_alignment": True,
        "coordinate_mutation_after_render": False,
        "orientation_policy": "rdkit_drawcoords_then_real_fragment_crop_with_synchronized_coordinate_transform",
        "synchronized_after_augmentation": True,
        "strict_orientation": bool(strict_orientation),
        "renderer_rotation_allowed": not bool(strict_orientation),
        "page_rotation_allowed": not bool(strict_orientation),
        "page_rotation_applied": bool(page_rotation_info.get("rotated")),
        "page_rotation_angle_degrees": float(page_rotation_info.get("angle_degrees") or 0.0),
        "target_side_rotation_applied": bool(target_side_rotation_info.get("enabled")),
        "target_side_rotation_turns_ccw": int(target_side_rotation_info.get("turns_ccw") or 0),
        "target_side": target_side_rotation_info.get("target_side") or "",
    }
    mark_geometry = draw_realistic_attachment(
        image_path,
        anchor_xy=coords[anchor_index],
        endpoint_xy=coords[dummy_index],
        anchor_label=requested_anchor_visible_label,
        anchor_depiction_mode=anchor_depiction,
        visual_shape=visual_shape,
        semantic_family=semantic_family,
        render_style=style,
        native_terminal_label=native_terminal_label,
        index=index,
        rng=rng,
        native_bond_line_width_px=float(draw_style_info["bond_line_width_px"]),
        coordinate_scale=float(crop_info.get("scale") or 1.0),
        draw_style_info=draw_style_info,
    )
    coords, mark_geometry, width, height, attachment_canvas_padding_info = ensure_attachment_geometry_canvas_padding(
        image_path,
        coords,
        mark_geometry,
    )
    nonlinear_warp_contract = {
        "schema_version": FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION,
        "enabled": False,
        "operations": [],
    }
    nonlinear_pose_preservation = {
        "schema_version": "fragment_formal_nonlinear_document_warp_contract_v1",
        "policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY,
        "passed": False,
        "blockers": ["fragment_formal_nonlinear_document_warp_not_enabled"],
    }
    if enable_formal_nonlinear_document_warp:
        coords, mark_geometry, width, height, nonlinear_warp_contract = apply_fragment_formal_nonlinear_document_warp(
            image_path,
            coords,
            mark_geometry,
            anchor_index=anchor_index,
            dummy_index=dummy_index,
            seed=int(stable_id("fragment_formal_nonlinear_warp", row_id, seed), 16) % (2**31 - 1),
            enabled=True,
            amplitude_px=float(formal_nonlinear_warp_amplitude_px),
        )
        if nonlinear_warp_contract.get("atom_outside_count") or nonlinear_warp_contract.get("mark_anchor_outside_count"):
            raise ValueError(f"fragment nonlinear document warp contract failed: {nonlinear_warp_contract}")
        if nonlinear_warp_contract.get("blank") or nonlinear_warp_contract.get("dense"):
            raise ValueError(f"fragment nonlinear document warp image density failed: {nonlinear_warp_contract}")
        pose_alignment_info["coordinate_mutation_after_render"] = True
        pose_alignment_info["formal_nonlinear_document_warp"] = True
        pose_alignment_info["nonlinear_pose_preservation_policy"] = FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY
        pose_alignment_info["orientation_policy"] = (
            "rdkit_drawcoords_then_synchronized_fragment_endpoint_connector_terminal_mark_document_warp"
        )
        _, _, _, _, warped_connector_length = connector_basis(coords[anchor_index], coords[dummy_index])
        endpoint_length_info = dict(endpoint_length_info)
        endpoint_length_info["connector_length_px"] = float(warped_connector_length)
        endpoint_length_info["coordinate_mutation_after_render"] = True
        endpoint_length_info["policy"] = (
            "reject_native_connector_before_warp_then_recompute_length_after_synchronized_formal_nonlinear_warp"
        )
        if (
            float(endpoint_length_info.get("minimum_connector_length_px") or 0.0) > 0.0
            and warped_connector_length + 1e-6 < float(endpoint_length_info.get("minimum_connector_length_px") or 0.0)
        ):
            raise ValueError("connector_too_short_after_fragment_formal_nonlinear_warp")
    if real_tight_crop_style:
        image_stats = perturb_image(image_path, style, rng)
        attachment_domain_stats = match_real_tight_wavy_domain(image_path, rng)
    else:
        image_stats = perturb_image(image_path, "rdkit_scan_like" if style == "rdkit_real_crop_bold_scan" else style, rng)
        attachment_domain_stats = match_attachment_domain(image_path, style, rng)
    if image_stats.get("blank") or image_stats.get("dense") or attachment_domain_stats.get("blank") or attachment_domain_stats.get("dense"):
        raise ValueError("image_density_gate_failed")
    pixel_geometry_detection = audit_fragment_pixel_geometry(image_path, mark_geometry)
    molnextr_input_quality = molnextr_cropwhite_resize_audit(
        image_path,
        coords,
        mark_geometry,
    )
    fragment_visual_quality = build_fragment_visual_quality(
        pixel_geometry=pixel_geometry_detection,
        molnextr_input_quality=molnextr_input_quality,
        document_context_info=document_context_info,
        image_stats=image_stats,
        attachment_domain_stats=attachment_domain_stats,
    )
    fragment_document_realism = build_fragment_document_realism(
        render_style=style,
        mark_geometry=mark_geometry,
        document_context_info=document_context_info,
        visual_quality=fragment_visual_quality,
    )
    if pixel_geometry_detection.get("passed") is not True:
        raise ValueError(f"fragment pixel geometry detection failed: {pixel_geometry_detection}")
    if molnextr_input_quality.get("passed") is not True:
        raise ValueError(f"fragment MolNexTR input quality failed: {molnextr_input_quality}")
    if fragment_visual_quality.get("passed") is not True:
        raise ValueError(f"fragment visual quality failed: {fragment_visual_quality}")

    anchor_coord = normalize_coord(coords[anchor_index][0], coords[anchor_index][1], width, height)
    endpoint_coord = normalize_coord(coords[dummy_index][0], coords[dummy_index][1], width, height)
    endpoint_side = side_from_endpoint(endpoint_coord["x"], endpoint_coord["y"])
    atom_coordinates = []
    for atom in fragment.GetAtoms():
        atom_index = int(atom.GetIdx())
        point = normalize_coord(coords[atom_index][0], coords[atom_index][1], width, height)
        atom_coordinates.append({"atom_index": atom_index, "token": atom_label(atom), "x": point["x"], "y": point["y"]})
    row_smiles = Chem.MolToSmiles(fragment)
    consistency = graph_consistency(fragment, row_smiles, anchor_index, dummy_index)
    if consistency.get("row_smiles_canonical_matches_mol") is not True:
        raise ValueError("generated fragment SMILES does not round-trip to rendered graph")
    bonds = bond_records(fragment)
    attachment_bond_semantics = audit_attachment_bond_semantics(
        bonds,
        anchor_index=anchor_index,
        dummy_index=dummy_index,
        mode=str(mark_geometry["attachment_render_mode"]),
        geometry=str(mark_geometry["attachment_render_geometry"]),
    )
    if attachment_bond_semantics.get("passed") is not True:
        raise ValueError(f"fragment attachment bond semantics failed: {attachment_bond_semantics}")
    terminal_wavy_externality = audit_terminal_wavy_externality(
        coords=coords,
        bonds=bonds,
        mark_geometry=mark_geometry,
        anchor_index=anchor_index,
        dummy_index=dummy_index,
    )
    if terminal_wavy_externality.get("passed") is not True:
        raise ValueError(f"terminal wavy externality failed: {terminal_wavy_externality}")
    render_quality = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "backend": "rdkit",
        "backend_version": Chem.rdBase.rdkitVersion,
        "backend_references": BACKEND_REFERENCES,
        "dataset_lineage": formal_sim_dataset_lineage(
            branch="fragment_attachment_positive",
            source_dataset="small_backbone_fragment_seed",
            source_record_id=source_id,
            parent_source_group=canonical_backbone(backbone_smiles) or backbone_smiles,
            generation_stage="raw_generation_formal_sim_candidate",
            shard_id=output_dir.name,
        ),
        "simulation_policy": formal_simulation_policy(
            branch="fragment_attachment_positive",
            allowed_operations=(
                list(crop_info.get("operations") or [])
                + list(side_preserving_padding_info.get("operations") or [])
                + list(attachment_canvas_padding_info.get("operations") or [])
                + list(document_context_info.get("operations") or [])
                + (["small_affine_rotation"] if page_rotation_info.get("rotated") else [])
                + list(nonlinear_warp_contract.get("operations") or [])
                + list(image_stats.get("operations") or [])
                + list(attachment_domain_stats.get("operations") or [])
            ),
            coordinate_mutation_policy=(
                "formal_synchronized_image_atom_endpoint_connector_terminal_mark_warp"
                if enable_formal_nonlinear_document_warp
                else "synchronized_rdkit_fragment_crop_margin_and_final_attachment_geometry_canvas_padding_with_non_geometric_scan_effects_only"
            ),
            geometry_contract=(
                "fragment_formal_nonlinear_document_warp_contract_v1"
                if enable_formal_nonlinear_document_warp
                else "fragment_endpoint_anchor_connector_contract_v1"
            ),
            formal_capable=True,
            image_synchronized=True,
            atom_coordinates_synchronized=True,
            endpoint_coordinates_synchronized=True,
            svg_or_connector_anchors_synchronized=True,
        ),
        "source_dataset": "small_backbone_fragment_seed",
        "source_record_id": source_id,
        "structure_type": "attachment_fragment",
        "image_width": width,
        "image_height": height,
        "canvas_width": image_size,
        "canvas_height": image_size,
        "atom_coordinates": atom_coordinates,
        "bonds": bonds,
        "layout_seed": seed,
        "render_style": style,
        "native_draw_style": draw_style_info,
        "real_tight_crop_style": bool(real_tight_crop_style),
        "coord_policy": "rdkit_drawcoords_then_real_fragment_crop_with_synchronized_coordinate_transform",
        "svg_metadata_present": bool(svg_text),
        "crop_policy": crop_info,
        "document_realism": fragment_document_realism,
        "fragment_visual_quality": fragment_visual_quality,
        "fragment_pixel_geometry_detection": pixel_geometry_detection,
        "fragment_molnextr_input_quality": molnextr_input_quality,
        "fragment_attachment_bond_semantics": attachment_bond_semantics,
        "terminal_wavy_externality": terminal_wavy_externality,
        "document_context": document_context_info,
        "fragment_nonlinear_document_warp": nonlinear_warp_contract,
        "fragment_nonlinear_pose_preservation": nonlinear_pose_preservation,
        "page_rotation": page_rotation_info,
        "endpoint_length_policy": endpoint_length_info,
        "pre_render_orientation": pre_render_orientation_info,
        "side_preserving_padding": side_preserving_padding_info,
        "attachment_canvas_padding": attachment_canvas_padding_info,
        "target_side_rotation": target_side_rotation_info,
        "pose_alignment": pose_alignment_info,
        "min_canvas_policy": min_canvas_info,
        "quality_gates": {
            "image_readable": True,
            "blank": bool(image_stats["blank"]),
            "dense": bool(image_stats["dense"]),
            "endpoint_in_image": True,
            "atom_coordinates_present": True,
            "external_backend_referenced": True,
            "smiles_graph_consistent": True,
            "single_dummy_attachment": True,
            "real_fragment_taxonomy_guided": True,
            "molnextr_pose_synchronized_after_augmentation": True,
            "image_to_graph_orientation_alignment": True,
            "fragment_formal_nonlinear_document_warp_allowed": bool(enable_formal_nonlinear_document_warp),
            "fragment_formal_nonlinear_contract_passed": False,
            "fragment_connector_mark_anchors_synchronized": bool(enable_formal_nonlinear_document_warp),
            "fragment_visual_quality_passed": True,
            "fragment_document_realism_machine_audit_passed": True,
            "fragment_pixel_geometry_detection_passed": True,
            "fragment_molnextr_input_quality_passed": True,
            "fragment_attachment_bond_semantics_passed": True,
            "fragment_visual_wavy_not_rdkit_stereo": attachment_bond_semantics.get("not_rdkit_stereo_wavy") is True,
            "terminal_wavy_externality_passed": terminal_wavy_externality.get("passed") is True,
            "real_tight_wavy_domain": bool(real_tight_crop_style),
            "final_attachment_geometry_canvas_synchronized": True,
        },
        "graph_consistency": consistency,
        "attachment_anchor": atom_label(fragment.GetAtomWithIdx(anchor_index)),
        "attachment_anchor_visible_label": requested_anchor_visible_label,
        "attachment_anchor_symbol": atom_label(fragment.GetAtomWithIdx(anchor_index)),
        "attachment_anchor_depiction_mode": anchor_depiction,
        "attachment_anchor_label_is_visible_text": anchor_label_is_visible_text_for_connector(
            anchor_label=requested_anchor_visible_label,
            anchor_depiction_mode=anchor_depiction,
        ),
        "attachment_anchor_depiction_detection": {
            "method": "rdkit_svg_atom_label_path_presence",
            "svg_atom_label_path_present": rendered_svg_atom_has_label(svg_text, anchor_index),
            "policy": "implicit carbon skeleton connectors are continuous; explicit carbon labels receive the same connector clearance as hetero atom labels",
        },
        "attachment_anchor_index": anchor_index,
        "attachment_anchor_coord": anchor_coord,
        "attachment_dummy_index": dummy_index,
        "attachment_bond_index": attachment_bond_index,
        "attachment_connector": {"start": anchor_coord, "end": endpoint_coord},
        "attachment_endpoint": endpoint_coord,
        "attachment_direction": endpoint_side,
        "requested_attachment_direction": requested_side,
        "attachment_render_mode": mark_geometry["attachment_render_mode"],
        "attachment_render_geometry": mark_geometry["attachment_render_geometry"],
        "fragment_mark_geometry": mark_geometry.get("fragment_mark_geometry", {}),
        "internal_wavy_stereo": internal_wavy_stereo,
        "visual_shape": visual_shape,
        "semantic_family": semantic_family,
        "chemistry_family": chemistry_family(row_smiles),
        "endpoint_label_quality": 1.0,
        "style_augmentations": document_context_info["operations"]
        + (["small_affine_rotation"] if page_rotation_info.get("rotated") else [])
        + image_stats["operations"]
        + attachment_domain_stats["operations"],
        "attachment_domain_match": attachment_domain_stats,
    }
    if "wavy_geometry" in mark_geometry:
        render_quality["wavy_geometry"] = mark_geometry["wavy_geometry"]
    if isinstance(render_quality.get("fragment_mark_geometry"), dict):
        render_quality["fragment_mark_geometry"] = _normalize_sampled_geometry_points(
            render_quality["fragment_mark_geometry"],
            width=width,
            height=height,
        )
    if isinstance(render_quality.get("wavy_geometry"), dict):
        render_quality["wavy_geometry"] = _normalize_sampled_geometry_points(
            render_quality["wavy_geometry"],
            width=width,
            height=height,
        )
    if enable_formal_nonlinear_document_warp:
        connector_samples = render_quality["fragment_mark_geometry"].get("connector_sampled_points_normalized", [])
        if connector_samples:
            render_quality["attachment_connector"]["sampled_points_normalized"] = connector_samples
            render_quality["attachment_connector"]["sample_policy"] = "warped_anchor_to_dummy_connector_polyline_samples"
        nonlinear_pose_preservation = audit_fragment_formal_nonlinear_contract(
            atom_coordinates=atom_coordinates,
            render_quality=render_quality,
        )
        if nonlinear_pose_preservation.get("passed") is not True:
            raise ValueError(f"fragment formal nonlinear contract failed: {nonlinear_pose_preservation}")
        render_quality["fragment_nonlinear_pose_preservation"] = nonlinear_pose_preservation
        render_quality["quality_gates"]["fragment_formal_nonlinear_contract_passed"] = True
    return {
        "source_id": row_id,
        "source_arrow": "pose_factory:attachment_fragment_rdkit_moldraw2d_attachment_primitives",
        "file_path": str(image_rel),
        "SMILES": row_smiles,
        "smiles": row_smiles,
        "structure_type_bucket": "attachment_fragment",
        "attachment_render_mode": str(render_quality["attachment_render_mode"]),
        "endpoint_side": endpoint_side,
        "attachment_anchor": render_quality["attachment_anchor"],
        "image_width": str(width),
        "image_height": str(height),
        "endpoint_x": f"{endpoint_coord['x']:.8f}",
        "endpoint_y": f"{endpoint_coord['y']:.8f}",
        "render_quality": json.dumps(render_quality, sort_keys=True),
        "reliable_training_label": "true",
    }


def parse_target_bucket(value: str) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {"side": "", "anchor": "", "mode": ""}
    parts = text.split("|")
    if len(parts) != 4 or parts[0] != "fragment":
        raise ValueError(f"target bucket must look like fragment|side|anchor|mode, got {text!r}")
    side, anchor, mode = parts[1], parts[2], parts[3]
    if side not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"invalid target side: {side}")
    if anchor not in TARGET_ANCHOR_LABELS and anchor not in TARGET_VISIBLE_ANCHOR_LABELS:
        raise ValueError(f"invalid target anchor: {anchor}")
    if mode not in {"cut", "dummy_atom", "query_attachment", "wavy"}:
        raise ValueError(f"invalid target mode: {mode}")
    return {"side": side, "anchor": anchor, "mode": mode}


def resolve_target(args: argparse.Namespace) -> dict[str, str]:
    target = parse_target_bucket(args.target_bucket)
    loose_side = str(getattr(args, "target_side", "") or "").strip()
    loose_anchor = str(getattr(args, "target_anchor", "") or "").strip()
    loose_mode = str(getattr(args, "target_mode", "") or "").strip()
    if loose_side:
        if loose_side not in {"left", "right", "top", "bottom"}:
            raise ValueError(f"invalid target side: {loose_side}")
        if target["side"] and target["side"] != loose_side:
            raise ValueError("--target-bucket side conflicts with --target-side")
        target["side"] = loose_side
    if loose_anchor:
        if loose_anchor not in TARGET_ANCHOR_LABELS and loose_anchor not in TARGET_VISIBLE_ANCHOR_LABELS:
            raise ValueError(f"invalid target anchor: {loose_anchor}")
        if target["anchor"] and target["anchor"] != loose_anchor:
            raise ValueError("--target-bucket anchor conflicts with --target-anchor")
        target["anchor"] = loose_anchor
    if loose_mode:
        if loose_mode not in {"cut", "dummy_atom", "query_attachment", "wavy"}:
            raise ValueError(f"invalid target mode: {loose_mode}")
        if target["mode"] and target["mode"] != loose_mode:
            raise ValueError("--target-bucket mode conflicts with --target-mode")
        target["mode"] = loose_mode
    return target


def target_matches(row: dict[str, str], target: dict[str, str]) -> bool:
    if target.get("side") and str(row.get("endpoint_side") or "") != target["side"]:
        return False
    requested_anchor = str(target.get("anchor") or "")
    if requested_anchor:
        quality = parse_quality(row)
        observed_symbol = str(quality.get("attachment_anchor_symbol") or row.get("attachment_anchor") or "")
        observed_visible = str(quality.get("attachment_anchor_visible_label") or row.get("attachment_anchor") or "")
        if requested_anchor in TARGET_VISIBLE_ANCHOR_LABELS:
            if observed_visible != requested_anchor:
                return False
        elif observed_symbol != requested_anchor and observed_visible != requested_anchor:
            return False
    if target.get("mode") and str(row.get("attachment_render_mode") or "") != target["mode"]:
        return False
    return True


def row_bucket(row: dict[str, str]) -> str:
    return "|".join(
        [
            "fragment",
            str(row.get("endpoint_side") or ""),
            str((parse_quality(row).get("attachment_anchor_visible_label") if parse_quality(row) else "") or row.get("attachment_anchor") or ""),
            str(row.get("attachment_render_mode") or ""),
        ]
    )


def commit_attempt_image(row: dict[str, str], *, attempt_dir: Path, output_dir: Path) -> dict[str, str]:
    image_rel = Path(str(row.get("file_path") or ""))
    if image_rel.is_absolute() or ".." in image_rel.parts or not image_rel.parts:
        raise ValueError(f"invalid generated image path: {image_rel}")
    source_path = attempt_dir / image_rel
    if not source_path.exists():
        raise ValueError(f"generated attempt image missing: {source_path}")
    target_path = output_dir / image_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_id",
        "source_arrow",
        "file_path",
        "SMILES",
        "smiles",
        "structure_type_bucket",
        "attachment_render_mode",
        "endpoint_side",
        "attachment_anchor",
        "image_width",
        "image_height",
        "endpoint_x",
        "endpoint_y",
        "render_quality",
        "reliable_training_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pose-aware RDKit MolDraw2D attachment-fragment shard.")
    parser.add_argument("--input-smiles-csv", default="")
    parser.add_argument("--heldout-fragment-csv", default="training/molnextr_markush/data/rgreco_fragment_eval/eval.csv")
    parser.add_argument("--output-dir", default="training/molnextr_markush/data/generated/pose_factory/fragment_attachment_candidate")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026061906)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument(
        "--target-bucket",
        default="",
        help="Optional exact target bucket, e.g. fragment|bottom|N|dummy_atom. Rows not matching the generated bucket are rejected.",
    )
    parser.add_argument("--target-mode", default="", help="Optional loose attachment mode target: cut, dummy_atom, query_attachment, or wavy.")
    parser.add_argument("--target-side", default="", help="Optional loose endpoint side target: left, right, top, or bottom.")
    parser.add_argument("--target-anchor", default="", help="Optional loose anchor atom target: C, N, O, S, P, or visible NH/OH/SH/PH.")
    parser.add_argument(
        "--target-attempts-per-backbone",
        type=int,
        default=4,
        help="When --target-bucket is set, try this many deterministic layout variants per backbone before moving on.",
    )
    parser.add_argument(
        "--max-source-attempts",
        type=int,
        default=0,
        help="Maximum source-backbone attempts before stopping. Defaults to rows*5; raise for strict long-tail buckets.",
    )
    parser.add_argument("--disable-document-context", action="store_true")
    parser.add_argument(
        "--real-tight-crop-style",
        action="store_true",
        help=(
            "Generate real-task-like tight, dark perpendicular terminal-wavy crops. "
            "Allowed only with --target-mode wavy; coordinates and terminal geometry remain synchronized."
        ),
    )
    parser.add_argument(
        "--strict-orientation",
        action="store_true",
        help=(
            "Disable renderer/page rotations and reject target-side rows that would require post-render rotation. "
            "Cropping, document context, light noise, and JPEG remain synchronized and recorded."
        ),
    )
    parser.add_argument(
        "--enable-formal-nonlinear-document-warp",
        action="store_true",
        help=(
            "Apply a low-amplitude synchronized fragment document warp and require "
            "fragment_formal_nonlinear_document_warp_contract_v1. This requires --strict-orientation."
        ),
    )
    parser.add_argument(
        "--formal-nonlinear-warp-amplitude-px",
        type=float,
        default=1.4,
        help="Maximum displacement in pixels for the formal fragment nonlinear document warp.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into the deterministic source-backbone stream; useful for non-overlapping top-up shards.",
    )
    args = parser.parse_args()
    disable_rdkit_parse_noise()
    if args.enable_formal_nonlinear_document_warp and not args.strict_orientation:
        raise SystemExit("--enable-formal-nonlinear-document-warp requires --strict-orientation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    target = resolve_target(args)
    if args.real_tight_crop_style and target.get("mode") != "wavy":
        raise SystemExit("--real-tight-crop-style requires a wavy target mode or target bucket")
    heldout = heldout_backbones(args.heldout_fragment_csv)
    source_backbones = load_backbone_smiles(args.input_smiles_csv, max_atoms=int(args.max_atoms), heldout=heldout)
    original_source_backbone_count = len(source_backbones)
    source_backbones, target_anchor_filter_info = filter_backbones_for_target_anchor(
        source_backbones,
        target_anchor=target["anchor"],
    )
    if not source_backbones:
        raise SystemExit("no source backbones available after heldout and target-anchor filtering")

    generated: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    attempts = 0
    start_index = max(0, int(args.start_index))
    max_attempts = int(args.max_source_attempts) if int(args.max_source_attempts) > 0 else max(int(args.rows) * 5, int(args.rows) + 20)
    while len(generated) < int(args.rows) and attempts < max_attempts:
        stream_index = start_index + attempts
        source_id, smiles = source_backbones[stream_index % len(source_backbones)]
        try:
            target_attempts = max(1, int(args.target_attempts_per_backbone)) if args.target_bucket else 1
            accepted_row = None
            last_error = ""
            for local_attempt in range(target_attempts):
                attempt_rng = random.Random(rng.randint(0, 2**31 - 1))
                attempt_index = stream_index * target_attempts + local_attempt
                try:
                    with tempfile.TemporaryDirectory(prefix="fragment_attempt_", dir=str(output_dir)) as attempt_dir_name:
                        attempt_dir = Path(attempt_dir_name)
                        row = generate_row(
                            backbone_smiles=smiles,
                            source_id=source_id,
                            output_dir=attempt_dir,
                            index=attempt_index,
                            rng=attempt_rng,
                            image_size=int(args.image_size),
                            document_context=not bool(args.disable_document_context),
                            target_side=target["side"],
                            target_anchor=target["anchor"],
                            target_mode=target["mode"],
                            strict_orientation=bool(args.strict_orientation),
                            real_tight_crop_style=bool(args.real_tight_crop_style),
                            enable_formal_nonlinear_document_warp=bool(args.enable_formal_nonlinear_document_warp),
                            formal_nonlinear_warp_amplitude_px=float(args.formal_nonlinear_warp_amplitude_px),
                        )
                        if any(target.values()) and not target_matches(row, target):
                            last_error = f"target_bucket_mismatch:{row_bucket(row)}"
                            continue
                        accepted_row = commit_attempt_image(row, attempt_dir=attempt_dir, output_dir=output_dir)
                except Exception as exc:
                    last_error = str(exc)
                    continue
                break
            if accepted_row is None:
                failures.append({"source_id": source_id, "smiles": smiles, "error": last_error or "target_bucket_unmatched"})
            else:
                generated.append(accepted_row)
        except Exception as exc:
            failures.append({"source_id": source_id, "smiles": smiles, "error": str(exc)})
        attempts += 1

    csv_path = output_dir / "attachment_fragment_positive.csv"
    write_csv(csv_path, generated)
    mode_counts = Counter()
    side_counts = Counter()
    visual_counts = Counter()
    semantic_counts = Counter()
    for row in generated:
        quality = json.loads(row["render_quality"])
        mode_counts.update([str(row.get("attachment_render_mode") or "")])
        side_counts.update([str(row.get("endpoint_side") or "")])
        visual_counts.update([str(quality.get("visual_shape") or "")])
        semantic_counts.update([str(quality.get("semantic_family") or "")])
    manifest = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "csv": str(csv_path),
        "row_count": len(generated),
        "failure_count": len(failures),
        "failures": failures[:50],
        "fragment_mark_mode_counts": dict(sorted(mode_counts.items())),
        "endpoint_side_counts": dict(sorted(side_counts.items())),
        "visual_shape_counts": dict(sorted(visual_counts.items())),
        "semantic_family_counts": dict(sorted(semantic_counts.items())),
        "seed": int(args.seed),
        "start_index": start_index,
        "target_bucket": str(args.target_bucket or ""),
        "source": "curated_small_fragment_backbones" if not args.input_smiles_csv else args.input_smiles_csv,
        "source_backbone_count_before_target_anchor_filter": int(original_source_backbone_count),
        "source_backbone_count_after_target_anchor_filter": int(len(source_backbones)),
        "target_anchor_source_filter": target_anchor_filter_info,
        "heldout_fragment_csv": args.heldout_fragment_csv,
        "heldout_backbone_exclusion_count": len(heldout),
        "external_references": BACKEND_REFERENCES,
        "document_context_enabled": not bool(args.disable_document_context),
        "strict_orientation": bool(args.strict_orientation),
        "real_tight_crop_style": bool(args.real_tight_crop_style),
        "real_tight_crop_contract": {
            "enabled": bool(args.real_tight_crop_style),
            "target_mode_required": "wavy target mode or target bucket" if bool(args.real_tight_crop_style) else "",
            "geometry_required": CUSTOM_PERPENDICULAR_WAVY_GEOMETRY if bool(args.real_tight_crop_style) else "",
            "coordinate_policy": (
                "rdkit_drawcoords_then_synchronized_tight_crop_terminal_perpendicular_wavy_render_and_final_attachment_canvas_padding"
                if bool(args.real_tight_crop_style)
                else ""
            ),
            "target_real_task_domain": {
                "long_edge_p10_p90": [103, 150],
                "short_edge_p10_p90": [52, 95],
                "ink_ratio_lt220_p10_p90": [0.09, 0.14],
            }
            if bool(args.real_tight_crop_style)
            else {},
        },
        "formal_nonlinear_document_warp": {
            "enabled": bool(args.enable_formal_nonlinear_document_warp),
            "schema_version": FRAGMENT_NONLINEAR_WARP_SCHEMA_VERSION,
            "formal_policy": FRAGMENT_FORMAL_NONLINEAR_WARP_POLICY
            if bool(args.enable_formal_nonlinear_document_warp)
            else "",
            "amplitude_px": float(args.formal_nonlinear_warp_amplitude_px)
            if bool(args.enable_formal_nonlinear_document_warp)
            else 0.0,
            "requires_fragment_contract": bool(args.enable_formal_nonlinear_document_warp),
        },
        "status": "candidate_requires_validation_leak_check_taxonomy_alignment_and_visual_review",
        "accepted": False,
        "rejected": False,
        "acceptance": {
            "accepted": False,
            "rejected": False,
            "visual_review_passed": False,
            "source_leak_check_passed": False,
            "real_fragment_taxonomy_alignment_passed": False,
            "reason": "New document-context fragment candidate; acceptance must be set only after schema, leak, coverage, taxonomy, and visual review pass.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

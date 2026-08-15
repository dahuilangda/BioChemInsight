from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen, rdDepictor, rdMolAlign
    from rdkit.Geometry import Point3D
    _HAS_COORDGEN = True
except ImportError:  # pragma: no cover - RDKit is expected in the runtime image
    Chem = None
    Point3D = None
    _HAS_COORDGEN = False


@dataclass(frozen=True)
class AttachmentLayoutResult:
    fragment_mol: Any
    note: str
    score: float | None = None
    angle_degrees: float = 0.0


def has_conformer(mol) -> bool:
    return mol is not None and mol.GetNumConformers() > 0


def heavy_atom_composition(mol) -> dict:
    """Element-symbol -> count for heavy, non-dummy atoms (excludes H and the
    R/* dummy, atomic number 0).  Used to compare skeletons regardless of how
    attachment points are represented.
    """
    if mol is None or Chem is None:
        return {}
    counts: Counter = Counter()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0 or atomic_num == 1:
            continue
        counts[atom.GetSymbol()] += 1
    return dict(counts)


# Matches a V2000/V3000 counts line (" 39 44 ... 999 V2000").
_COUNTS_LINE_RE = re.compile(r"^\s*\d+\s+\d+\s")


def normalize_molblock_header(molblock: str) -> str:
    """Re-align a molblock's 3-line header before the counts line.

    MolNexTR/RDKit molblocks are sometimes stored or stripped such that the
    leading blank "molecule name" line is lost, which shifts every line up by
    one and makes RDKit fail to locate the counts line.  This finds the counts
    line and re-inserts header padding so the block parses reliably.
    """
    if not molblock:
        return molblock or ''
    lines = molblock.splitlines()
    for idx, line in enumerate(lines):
        if _COUNTS_LINE_RE.match(line) and "." not in line and (
            "V2000" in line or "V3000" in line or "999" in line
        ):
            header = lines[max(0, idx - 3):idx]
            while len(header) < 3:
                header.insert(0, "")
            return "\n".join(header + lines[idx:])
    return molblock


def smiles_molblock_consistent(smiles: str | None, molblock: str | None) -> bool:
    """Return True when *molblock* and *smiles* describe the same heavy skeleton.

    Compares the heavy (non-H, non-dummy) element composition. This detects
    cases where the image-derived molblock lost or gained atoms relative to
    the canonical SMILES (e.g. a CF3 group collapsed into an R placeholder).

    Fail-open: returns True when either input is missing or unparseable —
    "nothing to reconcile". Not a fail-closed output gate.
    """
    if not smiles or not molblock or Chem is None:
        return True
    try:
        mol_smiles = Chem.MolFromSmiles(smiles)
        mol_mb = Chem.MolFromMolBlock(
            normalize_molblock_header(molblock), sanitize=False, removeHs=False
        )
    except Exception:
        return True
    if mol_smiles is None or mol_mb is None:
        return True
    return heavy_atom_composition(mol_smiles) == heavy_atom_composition(mol_mb)


def mol_from_smiles_coordgen(smiles: str):
    """Build a molecule from *smiles* with a clean CoordGen 2D layout.

    Falls back to ``rdDepictor.Compute2DCoords`` when CoordGen is unavailable.
    Returns None if the SMILES cannot be parsed.
    """
    if not smiles or Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if _HAS_COORDGEN:
        try:
            rdCoordGen.AddCoords(mol, rdCoordGen.CoordGenParams())
        except Exception:
            try:
                rdDepictor.Compute2DCoords(mol)
            except Exception:
                pass
    else:
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
    return mol


def coord(mol, atom_idx: int):
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    return float(pos.x), float(pos.y), float(pos.z)


def set_coord(mol, atom_idx: int, value):
    mol.GetConformer().SetAtomPosition(atom_idx, Point3D(float(value[0]), float(value[1]), float(value[2])))


def sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def scale(v, factor: float):
    return v[0] * factor, v[1] * factor, v[2] * factor


def length(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalized_2d(v):
    vector_length = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if vector_length < 1e-6:
        return None
    return v[0] / vector_length, v[1] / vector_length, 0.0


def average_bond_length(mol, default=1.2) -> float:
    if not has_conformer(mol):
        return default
    lengths = []
    for bond in mol.GetBonds():
        begin = coord(mol, bond.GetBeginAtomIdx())
        end = coord(mol, bond.GetEndAtomIdx())
        bond_length = length(sub(end, begin))
        if bond_length > 1e-4:
            lengths.append(bond_length)
    return sum(lengths) / len(lengths) if lengths else default


def median_bond_length(mol, default=1.2) -> float:
    if not has_conformer(mol):
        return default
    lengths = []
    for bond in mol.GetBonds():
        begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        if begin_atom.GetAtomicNum() == 0 or end_atom.GetAtomicNum() == 0:
            continue
        bond_length = length(sub(coord(mol, bond.GetBeginAtomIdx()), coord(mol, bond.GetEndAtomIdx())))
        if bond_length > 1e-4:
            lengths.append(bond_length)
    if not lengths:
        return default
    lengths.sort()
    middle = len(lengths) // 2
    if len(lengths) % 2:
        return lengths[middle]
    return (lengths[middle - 1] + lengths[middle]) / 2.0


def copy_conformer_with_added_attachment_dummy(mol, normalized_mol, dummy_idx: int, anchor_idx: int):
    if not has_conformer(mol) or Point3D is None:
        return
    old_conf = mol.GetConformer()
    new_conf = Chem.Conformer(normalized_mol.GetNumAtoms())
    for atom_idx in range(mol.GetNumAtoms()):
        pos = old_conf.GetAtomPosition(atom_idx)
        new_conf.SetAtomPosition(atom_idx, Point3D(pos.x, pos.y, pos.z))

    anchor = coord(mol, anchor_idx)
    direction = (0.0, 0.0, 0.0)
    anchor_atom = mol.GetAtomWithIdx(anchor_idx)
    for neighbor in anchor_atom.GetNeighbors():
        neighbor_coord = coord(mol, neighbor.GetIdx())
        direction = add(direction, sub(anchor, neighbor_coord))
    direction = normalized_2d(direction) or (1.0, 0.0, 0.0)
    dummy_coord = add(anchor, scale(direction, average_bond_length(mol)))
    new_conf.SetAtomPosition(dummy_idx, Point3D(*dummy_coord))
    normalized_mol.RemoveAllConformers()
    normalized_mol.AddConformer(new_conf, assignId=True)


def _copy_mol(mol):
    return Chem.Mol(mol) if mol is not None else None


def _rotate_atoms_around_anchor(mol, atom_indices: list[int], anchor_idx: int, theta: float):
    if not has_conformer(mol):
        return
    anchor = coord(mol, anchor_idx)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for atom_idx in atom_indices:
        if atom_idx == anchor_idx:
            continue
        point = coord(mol, atom_idx)
        relative = sub(point, anchor)
        rotated = (
            relative[0] * cos_t - relative[1] * sin_t,
            relative[0] * sin_t + relative[1] * cos_t,
            relative[2],
        )
        set_coord(mol, atom_idx, add(anchor, rotated))


def _rotate_atoms_around_bond(mol, atom_indices: list[int], pivot_a: int, pivot_b: int, theta: float):
    if not has_conformer(mol):
        return
    point_a = coord(mol, pivot_a)
    point_b = coord(mol, pivot_b)
    axis = normalized_2d(sub(point_b, point_a))
    if axis is None:
        return
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    ux, uy, _ = axis
    for atom_idx in atom_indices:
        if atom_idx in {pivot_a, pivot_b}:
            continue
        point = coord(mol, atom_idx)
        relative = sub(point, point_b)
        parallel_len = relative[0] * ux + relative[1] * uy
        parallel = (parallel_len * ux, parallel_len * uy, 0.0)
        perpendicular = (relative[0] - parallel[0], relative[1] - parallel[1], relative[2])
        rotated_perpendicular = (
            perpendicular[0] * cos_t - perpendicular[1] * sin_t,
            perpendicular[0] * sin_t + perpendicular[1] * cos_t,
            perpendicular[2],
        )
        set_coord(mol, atom_idx, add(point_b, add(parallel, rotated_perpendicular)))


def _bbox_for_indices(mol, atom_indices: list[int]):
    coords = [coord(mol, atom_idx) for atom_idx in atom_indices]
    if not coords:
        return None
    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_overlap_area(box_a, box_b) -> float:
    if not box_a or not box_b:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def fragment_layout_collision_score(
    scaffold_mol,
    fragment_mol,
    scaffold_anchor: int,
    fragment_anchor: int,
) -> float:
    scaffold_indices = list(range(scaffold_mol.GetNumAtoms()))
    fragment_indices = list(range(fragment_mol.GetNumAtoms()))
    bond_length = max(average_bond_length(scaffold_mol), average_bond_length(fragment_mol), 1.0)
    min_allowed = 0.55 * bond_length
    close_penalty = 0.0
    for s_idx in scaffold_indices:
        s_point = coord(scaffold_mol, s_idx)
        for f_idx in fragment_indices:
            if s_idx == scaffold_anchor and f_idx == fragment_anchor:
                continue
            f_point = coord(fragment_mol, f_idx)
            distance = length(sub(s_point, f_point))
            if distance < min_allowed:
                close_penalty += ((min_allowed - distance) / min_allowed) ** 2

    scaffold_box = _bbox_for_indices(scaffold_mol, scaffold_indices)
    fragment_box = _bbox_for_indices(fragment_mol, fragment_indices)
    overlap_penalty = _bbox_overlap_area(scaffold_box, fragment_box) / max(bond_length * bond_length, 1e-6)

    scaffold_anchor_point = coord(scaffold_mol, scaffold_anchor)
    fragment_anchor_point = coord(fragment_mol, fragment_anchor)
    outward = normalized_2d(sub(fragment_anchor_point, scaffold_anchor_point)) or (1.0, 0.0, 0.0)
    mean_projection = 0.0
    for point in (coord(fragment_mol, idx) for idx in fragment_indices):
        vector = sub(point, scaffold_anchor_point)
        mean_projection += vector[0] * outward[0] + vector[1] * outward[1]
    mean_projection /= max(1, len(fragment_indices))
    inward_penalty = max(0.0, bond_length * 0.4 - mean_projection) / bond_length

    return close_penalty * 10.0 + overlap_penalty + inward_penalty * 3.0


def _pose_drift_score(reference_mol, candidate_mol, atom_indices: list[int], bond_length: float) -> float:
    if not atom_indices:
        return 0.0
    drift = 0.0
    for atom_idx in atom_indices:
        drift += length(sub(coord(reference_mol, atom_idx), coord(candidate_mol, atom_idx)))
    return drift / (len(atom_indices) * max(bond_length, 1e-6))


def _side_atoms_after_bond(mol, start_idx: int, blocked_idx: int) -> set[int]:
    visited = {blocked_idx}
    stack = [start_idx]
    side = set()
    while stack:
        atom_idx = stack.pop()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        side.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                stack.append(neighbor_idx)
    return side


def _rotatable_fragment_bonds(mol, fragment_anchor: int) -> list[tuple[int, int, list[int]]]:
    if mol is None:
        return []
    rotors = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(begin)
        end_atom = mol.GetAtomWithIdx(end)
        if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
            continue
        begin_side = _side_atoms_after_bond(mol, begin, end)
        end_side = _side_atoms_after_bond(mol, end, begin)
        if fragment_anchor in begin_side:
            mobile = end_side
            pivot_a, pivot_b = begin, end
        elif fragment_anchor in end_side:
            mobile = begin_side
            pivot_a, pivot_b = end, begin
        else:
            continue
        if len(mobile) < 2 or len(mobile) >= mol.GetNumAtoms() - 1:
            continue
        rotors.append((pivot_a, pivot_b, sorted(mobile)))
    rotors.sort(key=lambda item: (len(item[2]), item[0], item[1]))
    return rotors


def optimize_fragment_rigid_pose(scaffold_mol, fragment_mol, scaffold_anchor: int, fragment_anchor: int):
    if not (has_conformer(scaffold_mol) and has_conformer(fragment_mol)):
        return AttachmentLayoutResult(fragment_mol, "attachment_layout_collision_optimization_skipped")

    candidate_angles = [
        0.0,
        math.radians(15),
        math.radians(-15),
        math.radians(30),
        math.radians(-30),
        math.radians(45),
        math.radians(-45),
        math.radians(60),
        math.radians(-60),
        math.radians(75),
        math.radians(-75),
        math.radians(90),
        math.radians(-90),
        math.radians(120),
        math.radians(-120),
        math.radians(150),
        math.radians(-150),
        math.radians(180),
    ]
    best_mol = fragment_mol
    best_score = float("inf")
    best_angle = 0.0
    fragment_indices = list(range(fragment_mol.GetNumAtoms()))
    for angle in candidate_angles:
        candidate = _copy_mol(fragment_mol)
        _rotate_atoms_around_anchor(candidate, fragment_indices, fragment_anchor, angle)
        score = fragment_layout_collision_score(scaffold_mol, candidate, scaffold_anchor, fragment_anchor)
        score += abs(angle) * 0.015
        if score < best_score:
            best_score = score
            best_mol = candidate
            best_angle = angle

    angle_degrees = round(math.degrees(best_angle), 1)
    note = "layout_fragment_pose_optimized"
    if abs(best_angle) > 1e-6:
        note = f"{note}:{angle_degrees}deg"
    return AttachmentLayoutResult(best_mol, note, best_score, angle_degrees)


def layout_fragment_on_scaffold_attachment(
    *,
    fragment_mol,
    fragment_without_dummy,
    fragment_anchor: int,
    adjusted_fragment_anchor: int,
    fragment_dummy: int,
    scaffold_mol,
    scaffold_without_dummy,
    scaffold_anchor: int,
    adjusted_scaffold_anchor: int,
    scaffold_dummy: int,
) -> AttachmentLayoutResult:
    if not (
        has_conformer(fragment_mol)
        and has_conformer(fragment_without_dummy)
        and has_conformer(scaffold_mol)
        and has_conformer(scaffold_without_dummy)
    ):
        return AttachmentLayoutResult(fragment_without_dummy, "attachment_layout_skipped_missing_2d_coordinates")

    source_anchor = coord(fragment_mol, fragment_anchor)
    source_dummy = coord(fragment_mol, fragment_dummy)
    target_anchor = coord(scaffold_mol, scaffold_dummy)
    target_dummy = coord(scaffold_mol, scaffold_anchor)
    source_vec = sub(source_dummy, source_anchor)
    target_vec = sub(target_dummy, target_anchor)
    target_len = length(target_vec)
    source_len = length(source_vec)
    if source_len < 1e-6 or target_len < 1e-6:
        return AttachmentLayoutResult(fragment_without_dummy, "attachment_layout_skipped_degenerate_attachment_vector")

    source_angle = math.atan2(source_vec[1], source_vec[0])
    target_angle = math.atan2(target_vec[1], target_vec[0])
    theta = target_angle - source_angle
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    scaffold_scale = median_bond_length(scaffold_without_dummy, default=target_len)
    fragment_scale = median_bond_length(fragment_without_dummy, default=source_len)
    scale_factor = 1.0
    if scaffold_scale > 1e-6 and fragment_scale > 1e-6:
        scale_factor = max(0.15, min(5.0, scaffold_scale / fragment_scale))
    for old_idx in range(fragment_mol.GetNumAtoms()):
        if old_idx == fragment_dummy:
            continue
        new_idx = old_idx - 1 if old_idx > fragment_dummy else old_idx
        old = coord(fragment_mol, old_idx)
        relative = sub(old, source_anchor)
        relative = scale(relative, scale_factor)
        rotated = (
            relative[0] * cos_t - relative[1] * sin_t,
            relative[0] * sin_t + relative[1] * cos_t,
            relative[2],
        )
        set_coord(fragment_without_dummy, new_idx, add(target_anchor, rotated))

    expected_anchor = fragment_anchor - 1 if fragment_anchor > fragment_dummy else fragment_anchor
    if adjusted_fragment_anchor != expected_anchor:
        return AttachmentLayoutResult(fragment_without_dummy, "attachment_layout_anchor_index_mismatch")

    optimized = optimize_fragment_rigid_pose(
        scaffold_without_dummy,
        fragment_without_dummy,
        adjusted_scaffold_anchor,
        adjusted_fragment_anchor,
    )
    scale_note = f"layout_fragment_scale_normalized:{round(scale_factor, 3)}x"
    return AttachmentLayoutResult(
        optimized.fragment_mol,
        f"layout_fragment_anchor_to_scaffold_dummy;{scale_note};{optimized.note}",
        optimized.score,
        optimized.angle_degrees,
    )


def normalize_bond_lengths(mol, target: float = 1.5) -> int:
    """Scale all 2D coordinates so the median bond length equals *target*.

    Bonds involving dummy atoms (atomic number 0) are excluded from the
    median calculation.  Returns the number of bonds used (0 = no change).
    """
    if not has_conformer(mol):
        return 0
    lengths = []
    for bond in mol.GetBonds():
        b = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(b).GetAtomicNum() == 0 or mol.GetAtomWithIdx(e).GetAtomicNum() == 0:
            continue
        d = length(sub(coord(mol, b), coord(mol, e)))
        if d > 1e-4:
            lengths.append(d)
    if not lengths:
        return 0
    lengths.sort()
    mid = len(lengths) // 2
    median_bl = lengths[mid] if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) / 2.0
    if median_bl < 1e-4:
        return 0
    scale = target / median_bl
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x * scale, p.y * scale, p.z * scale))
    return len(lengths)


def _regenerate_with_coordgen(mol) -> bool:
    """Regenerate the 2D layout with CoordGen (Schrodinger), in place.

    Preserves stereochemistry: wedge bond directions (``BondDir``) are captured
    before regeneration and reapplied afterwards.  Dummy R-group atoms are kept.
    On any failure the molecule is left untouched and False is returned so the
    caller can fall back to the pose-preserved layout.
    """
    if not _HAS_COORDGEN or not has_conformer(mol):
        return False
    try:
        # Snapshot stereo: wedge bond directions survive CoordGen, but capture
        # them defensively so we can restore on any silent loss.
        wedge_dirs = []
        for bond in mol.GetBonds():
            d = bond.GetBondDir()
            if d is not None and d != Chem.BondDir.NONE:
                wedge_dirs.append((bond.GetIdx(), d))

        params = rdCoordGen.CoordGenParams()
        rdCoordGen.AddCoords(mol, params)

        # Restore any wedge directions that did not survive.
        if wedge_dirs:
            for bidx, d in wedge_dirs:
                bond = mol.GetBondWithIdx(bidx)
                if bond.GetBondDir() != d:
                    bond.SetBondDir(d)

        # Match the rest of the pipeline's scale (median bond length ~ 1.5 A).
        normalize_bond_lengths(mol)
        return has_conformer(mol)
    except Exception:
        return False


def optimize_2d_layout(mol, structure_type: str | None = None) -> str:
    """Type-driven 2D layout optimization.

    The strategy is deterministic and keyed to *structure_type* — there are no
    heuristics or quality gates:

    * ``fragment``: full clean depiction via CoordGen.  Fragments are standalone
      variable-group depictions, so a canonical clean layout is appropriate.
    * everything else (``complete_compound``, ``markush``, ``markush_assembled``):
      pose-preserving light normalization.  ``NormalizeDepiction(canonicalize=0)``
      applies a *uniform* scale to standard bond length and a translation to the
      centroid — it performs **no rotation and no internal geometry change** (see
      RDKit ``RDDepictor.cpp``), so the image-derived pose is preserved exactly.

    Returns a short audit note.
    """
    if not has_conformer(mol):
        return "skipped_no_conformer"

    if structure_type == "fragment":
        if _regenerate_with_coordgen(mol):
            return "full_layout_coordgen"
        normalize_bond_lengths(mol)
        return "fragment_full_layout_failed_light_normalize"

    # complete_compound / markush / markush_assembled: preserve pose.
    # NormalizeDepiction(canonicalize=0) only scales + recenters; no rotation,
    # no bond-angle or ring-shape change — the depiction keeps its original pose.
    try:
        rdDepictor.NormalizeDepiction(mol, confId=0, canonicalize=0)
        return "light_normalize_pose_preserved"
    except Exception:
        normalize_bond_lengths(mol)
        return "light_normalize_fallback_scale"


def refine_assembled_layout(mol) -> str:
    """2D refinement for an assembled Markush molecule.

    Assembled molecules are constructed (scaffold + grafted fragments) but their
    grafted pose is meaningful, so they get the same pose-preserving light
    normalization as complete/markush structures — never a full relayout.
    """
    return optimize_2d_layout(mol, structure_type="markush_assembled")

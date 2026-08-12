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

    Compares the heavy (non-H, non-dummy) element composition of the two.  This
    detects cases where the image-derived molblock lost or gained atoms relative
    to the canonical / VLM-corrected SMILES — the most common failure being a
    ``CF3`` group (C + 3F) collapsed into a single ``R`` placeholder in the
    molblock while the SMILES correctly retains ``C(F)(F)F``.

    The molblock header is normalized before parsing so that a stripped leading
    blank line (which would otherwise shift the header and silently fail the
    parse) does not mask a real divergence.

    Returns True (i.e. "no problem detected") when either input is missing or
    cannot be parsed, so callers can treat a non-False result as "keep the
    molblock as-is".
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


def _layout_collision_score_for_mol(mol, bond_length: float | None = None) -> float:
    """Score the overall compactness of a molecule's 2D layout.

    Penalises close non-bonded atom pairs (distance < 0.55 * bond_length)
    and uneven bond-length variance.  Lower is better.
    Dummy atoms (atomic number 0) are excluded.
    """
    if not has_conformer(mol):
        return float("inf")
    if bond_length is None:
        bond_length = max(average_bond_length(mol), 1.0)
    min_allowed = 0.55 * bond_length
    penalty = 0.0
    n_atoms = mol.GetNumAtoms()
    for i in range(n_atoms):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            continue
        pi = coord(mol, i)
        for j in range(i + 1, n_atoms):
            if mol.GetAtomWithIdx(j).GetAtomicNum() == 0:
                continue
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue  # bonded pair – skip
            pj = coord(mol, j)
            d = length(sub(pi, pj))
            if d < min_allowed:
                penalty += ((min_allowed - d) / min_allowed) ** 2
    return penalty


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


def _target_bond_length(bond) -> float:
    """Target 2D bond length based on bond order (Angstrom)."""
    bt = bond.GetBondTypeAsDouble()
    if bt >= 1.9:
        return 1.20  # triple
    if bt >= 1.5:
        return 1.34  # double
    if bond.GetIsAromatic():
        return 1.40  # aromatic
    return 1.50  # single


def _target_angle(mol, center_idx: int) -> float:
    """Target 2D bond angle (degrees) for the atom at *center_idx*.

    Rules (2D chemical drawing conventions):
    - Ring atoms follow ideal polygon angles (hexagon=120, pentagon=108,
      square=90, triangle=60).
    - sp2 (double bond / aromatic) atoms: 120 degrees.
    - sp3 chain atoms: 109.5 degrees.
    - Terminal / degree-2 ring atoms keep the ring polygon angle.
    """
    atom = mol.GetAtomWithIdx(center_idx)
    ri = mol.GetRingInfo()
    if ri and ri.NumAtomRings(center_idx) > 0:
        # Largest ring containing this atom determines the polygon angle
        ring_sizes = [len(r) for r in ri.AtomRings() if center_idx in r]
        if ring_sizes:
            largest = max(ring_sizes)
            if largest == 6:
                return 120.0
            if largest == 5:
                return 108.0
            if largest == 4:
                return 90.0
            if largest == 3:
                return 60.0
    has_double = any(b.GetBondTypeAsDouble() >= 1.5 for b in atom.GetBonds())
    if has_double or atom.GetIsAromatic():
        return 120.0
    return 109.5


def _place_atom(anchor_pos, parent_pos, target_bl: float, target_angle_deg: float,
                reference_dir, flip: int) -> tuple:
    """Place an atom at distance *target_bl* from *anchor_pos*, making a
    *target_angle_deg* angle at *anchor_pos* with the parent bond direction.

    *reference_dir* is the direction the previous bond points (from parent to
    anchor).  The new bond direction is rotated by (180 - target_angle) from
    the reference, choosing the side that keeps the new atom closest to its
    original position (flip = +1 or -1).
    """
    import math as _m
    ang = _m.radians(180.0 - target_angle_deg)
    cos_a = _m.cos(ang)
    sin_a = _m.sin(ang)
    rx = reference_dir[0] * cos_a - flip * reference_dir[1] * sin_a
    ry = reference_dir[0] * sin_a + flip * reference_dir[1] * cos_a
    return (anchor_pos[0] + rx * target_bl,
            anchor_pos[1] + ry * target_bl)


def _fix_local_angles(mol) -> int:
    """Fix grossly distorted bond angles (>150 deg or <60 deg at sp2/sp3
    centres) by rotating the offending neighbour group into the correct
    target angle.  Only touches atoms whose angle deviates significantly;
    the rest of the molecule keeps its pose.

    Returns the number of atoms that were adjusted.
    """
    if not has_conformer(mol):
        return 0
    import math as _m

    ri = mol.GetRingInfo()
    adjusted = 0
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    for _pass in range(2):  # two passes: first fix worst, then propagate
        for j in range(n):
            atom_j = mol.GetAtomWithIdx(j)
            if atom_j.GetAtomicNum() == 0:
                continue
            nbrs = [a.GetIdx() for a in atom_j.GetNeighbors()
                    if mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() != 0]
            if len(nbrs) < 2:
                continue

            # Determine target angle
            in_ring = ri and ri.NumAtomRings(j) > 0
            has_double = any(b.GetBondTypeAsDouble() >= 1.5 for b in atom_j.GetBonds())
            if in_ring:
                ring_sizes = [len(r) for r in ri.AtomRings() if j in r]
                largest = max(ring_sizes) if ring_sizes else 6
                if largest == 6: target = _m.radians(120)
                elif largest == 5: target = _m.radians(108)
                elif largest == 4: target = _m.radians(90)
                elif largest == 3: target = _m.radians(60)
                else: target = _m.radians(120)
            elif has_double or atom_j.GetIsAromatic():
                target = _m.radians(120)
            else:
                target = _m.radians(120)  # 2D drawing convention: 120 for all chain atoms

            jp = conf.GetAtomPosition(j)

            for idx_i in range(len(nbrs)):
                for idx_k in range(idx_i + 1, len(nbrs)):
                    i, k = nbrs[idx_i], nbrs[kk] if (kk := idx_k) is not None else nbrs[idx_k]
                    ip = conf.GetAtomPosition(i)
                    kp = conf.GetAtomPosition(k)
                    v1 = np.array([ip.x - jp.x, ip.y - jp.y])
                    v2 = np.array([kp.x - jp.x, kp.y - jp.y])
                    l1 = np.linalg.norm(v1)
                    l2 = np.linalg.norm(v2)
                    if l1 < 1e-6 or l2 < 1e-6:
                        continue
                    cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
                    angle = np.arccos(cos_a)
                    deviation = abs(angle - target)

                    # Only fix gross deviations (> 40 deg from target)
                    if deviation < _m.radians(40):
                        continue

                    # Determine which side (i or k) to rotate.
                    # Rotate the neighbour with fewer heavy connections
                    # (more likely to be a terminal group).
                    i_side = len([a for a in mol.GetAtomWithIdx(i).GetNeighbors()
                                  if a.GetIdx() != j and mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() != 0])
                    k_side = len([a for a in mol.GetAtomWithIdx(k).GetNeighbors()
                                  if a.GetIdx() != j and mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() != 0])

                    if i_side <= k_side:
                        move_idx, fixed_idx = i, k
                    else:
                        move_idx, fixed_idx = k, i

                    # Compute current angle and desired rotation
                    fixed_vec = np.array([conf.GetAtomPosition(fixed_idx).x - jp.x,
                                          conf.GetAtomPosition(fixed_idx).y - jp.y])
                    move_vec = np.array([conf.GetAtomPosition(move_idx).x - jp.x,
                                         conf.GetAtomPosition(move_idx).y - jp.y])
                    current_angle = _m.atan2(move_vec[1], move_vec[0])
                    fixed_angle = _m.atan2(fixed_vec[1], fixed_vec[0])

                    # Desired position: rotate move_idx so that the angle
                    # between fixed_vec and move_vec equals target.
                    # Two candidates: clockwise and counter-clockwise.
                    desired1 = fixed_angle + target
                    desired2 = fixed_angle - target

                    # Pick the one closest to current position
                    diff1 = abs((desired1 - current_angle + _m.pi) % (2 * _m.pi) - _m.pi)
                    diff2 = abs((desired2 - current_angle + _m.pi) % (2 * _m.pi) - _m.pi)
                    desired = desired1 if diff1 < diff2 else desired2

                    rotation = desired - current_angle
                    # Normalize to [-pi, pi]
                    rotation = (rotation + _m.pi) % (2 * _m.pi) - _m.pi

                    # Rotate the move_idx and all atoms on its side of j
                    side = _side_atoms_after_bond(mol, move_idx, j)
                    cos_r = _m.cos(rotation)
                    sin_r = _m.sin(rotation)
                    for s in side:
                        sp = conf.GetAtomPosition(s)
                        rx = sp.x - jp.x
                        ry = sp.y - jp.y
                        new_x = jp.x + rx * cos_r - ry * sin_r
                        new_y = jp.y + rx * sin_r + ry * cos_r
                        conf.SetAtomPosition(s, Point3D(new_x, new_y, 0.0))
                    adjusted += 1

    return adjusted


def _ring_irregularity(mol) -> float:
    """Mean coefficient-of-variation of ring edge lengths across all SSSR
    rings (0.0 = every ring is a perfect polygon).  Dummy-atom bonds are
    skipped.  Returns 0.0 for acyclic molecules.
    """
    if not has_conformer(mol):
        return 0.0
    ri = mol.GetRingInfo()
    if ri is None or not ri.AtomRings():
        return 0.0
    conf = mol.GetConformer()
    cvs = []
    for ring in ri.AtomRings():
        edges = []
        n = len(ring)
        for i in range(n):
            a = ring[i]
            b = ring[(i + 1) % n]
            if mol.GetAtomWithIdx(a).GetAtomicNum() == 0 or mol.GetAtomWithIdx(b).GetAtomicNum() == 0:
                continue
            pa = conf.GetAtomPosition(a)
            pb = conf.GetAtomPosition(b)
            edges.append(math.hypot(pa.x - pb.x, pa.y - pb.y))
        if len(edges) < 3:
            continue
        mean_edge = sum(edges) / len(edges)
        if mean_edge < 1e-6:
            continue
        variance = sum((e - mean_edge) ** 2 for e in edges) / len(edges)
        cvs.append(math.sqrt(variance) / mean_edge)
    return sum(cvs) / len(cvs) if cvs else 0.0


def _collision_score(mol) -> float:
    """Weighted collision score for non-bonded heavy-atom pairs closer than
    0.7 x median bond length.  Higher = more overlapping/unreadable.
    """
    if not has_conformer(mol):
        return 0.0
    bl = median_bond_length(mol)
    if bl < 1e-6:
        return 0.0
    min_allowed = 0.7 * bl
    conf = mol.GetConformer()
    penalty = 0.0
    n = mol.GetNumAtoms()
    for i in range(n):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            continue
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if mol.GetAtomWithIdx(j).GetAtomicNum() == 0:
                continue
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            pj = conf.GetAtomPosition(j)
            d = math.hypot(pi.x - pj.x, pi.y - pj.y)
            if d < min_allowed and d > 1e-6:
                penalty += ((min_allowed - d) / min_allowed) ** 2
    return penalty


def _angle_deviation_deg(mol) -> float:
    """Mean absolute bond-angle deviation (degrees) from the ideal 2D drawing
    angle at every atom centre with >= 2 heavy neighbours.  Ring atoms use the
    ideal polygon angle, sp2/aromatic use 120, sp3 chains use 109.5.
    """
    if not has_conformer(mol):
        return 0.0
    import math as _m
    ri = mol.GetRingInfo()
    conf = mol.GetConformer()
    deviations = []
    n = mol.GetNumAtoms()
    for j in range(n):
        atom_j = mol.GetAtomWithIdx(j)
        if atom_j.GetAtomicNum() == 0:
            continue
        nbrs = [a.GetIdx() for a in atom_j.GetNeighbors()
                if mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() != 0]
        if len(nbrs) < 2:
            continue
        in_ring = ri and ri.NumAtomRings(j) > 0
        if in_ring:
            ring_sizes = [len(r) for r in ri.AtomRings() if j in r]
            largest = max(ring_sizes) if ring_sizes else 6
            target = {3: 60.0, 4: 90.0, 5: 108.0, 6: 120.0}.get(largest, 120.0)
        elif any(b.GetBondTypeAsDouble() >= 1.5 for b in atom_j.GetBonds()) or atom_j.GetIsAromatic():
            target = 120.0
        else:
            target = 109.5
        jp = conf.GetAtomPosition(j)
        for ii in range(len(nbrs)):
            for kk in range(ii + 1, len(nbrs)):
                ip = conf.GetAtomPosition(nbrs[ii])
                kp = conf.GetAtomPosition(nbrs[kk])
                v1 = np.array([ip.x - jp.x, ip.y - jp.y])
                v2 = np.array([kp.x - jp.x, kp.y - jp.y])
                l1 = float(np.linalg.norm(v1))
                l2 = float(np.linalg.norm(v2))
                if l1 < 1e-6 or l2 < 1e-6:
                    continue
                cos_a = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1))
                angle_deg = _m.degrees(_m.acos(cos_a))
                deviations.append(abs(angle_deg - target))
    return sum(deviations) / len(deviations) if deviations else 0.0


# Thresholds beyond which a pose-preserved layout is considered too messy and
# gets regenerated with CoordGen.  Calibrated against the distorted/CoordGen
# comparison (distorted ring CV ~0.175 clearly exceeds 0.12).
_QUALITY_RING_IRREGULARITY = 0.12
_QUALITY_COLLISION = 1.0
_QUALITY_ANGLE_DEVIATION_DEG = 25.0


def layout_quality_score(mol) -> dict:
    """Score the geometric quality of a molecule's current 2D layout.

    Returns a dict with ``ring_irregularity``, ``collision``, ``angle_deviation_deg``
    and a boolean ``needs_regeneration`` that is True when any signal exceeds its
    threshold (distorted rings, overlapping atoms, or kinked chains).  Lower is
    better for every numeric field.
    """
    ring_irr = _ring_irregularity(mol)
    collision = _collision_score(mol)
    angle_dev = _angle_deviation_deg(mol)
    needs = (
        ring_irr > _QUALITY_RING_IRREGULARITY
        or collision > _QUALITY_COLLISION
        or angle_dev > _QUALITY_ANGLE_DEVIATION_DEG
    )
    return {
        "ring_irregularity": ring_irr,
        "collision": collision,
        "angle_deviation_deg": angle_dev,
        "needs_regeneration": needs,
    }


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


def cleanup_structure_pose(mol, smiles: str | None = None) -> str:
    """2D layout cleanup with a quality-gated CoordGen fallback.

    Step 1 is pose-preserving (operates on the existing conformer only — no
    rigid-body realignment), so the original pose from the patent image is kept
    whenever it is already clean:

    1. normalize_bond_lengths: uniform scale to 1.5 A median.
    2. _fix_local_angles: fix grossly distorted angles (>40 deg deviation) by
       rotating the smaller neighbour group into the target angle.
    3. NormalizeDepiction(canonicalize=0) + StraightenDepiction: refine bond
       lengths/angles and snap chains to zig-zag.

    Step 2 measures the resulting layout.  If it is still geometrically poor
    (distorted rings, overlapping atoms, or kinked chains) the coordinates are
    regenerated with CoordGen for a clean, readable depiction.  Stereochemistry
    and R-group dummy atoms are preserved either way.

    Returns a semicolon-joined audit note describing what was applied.
    """
    if not has_conformer(mol):
        return "pose_cleanup_skipped"
    if not _HAS_COORDGEN:
        # No CoordGen available — do the best we can with pose cleanup only.
        try:
            normalize_bond_lengths(mol)
            n_fixed = _fix_local_angles(mol)
            normalize_bond_lengths(mol)
            rdDepictor.NormalizeDepiction(mol, confId=0, canonicalize=0)
            rdDepictor.StraightenDepiction(mol, confId=0)
            return f"fix_angles:{n_fixed};normalize_straighten;no_coordgen"
        except Exception:
            normalize_bond_lengths(mol)
            return "bl_only"

    coordgen_fallback = True
    try:
        import constants as _constants
        coordgen_fallback = bool(getattr(_constants, "STRUCTURE_2D_LAYOUT_COORDGEN_FALLBACK", True))
    except Exception:
        coordgen_fallback = True

    try:
        normalize_bond_lengths(mol)
        n_fixed = _fix_local_angles(mol)
        normalize_bond_lengths(mol)
        rdDepictor.NormalizeDepiction(mol, confId=0, canonicalize=0)
        rdDepictor.StraightenDepiction(mol, confId=0)
        note = f"fix_angles:{n_fixed};normalize_straighten"
    except Exception:
        normalize_bond_lengths(mol)
        note = "bl_only"

    if not coordgen_fallback:
        return f"{note};coordgen_disabled"

    score = layout_quality_score(mol)
    if not score["needs_regeneration"]:
        return f"{note};pose_preserved_clean"

    reasons = []
    if score["ring_irregularity"] > _QUALITY_RING_IRREGULARITY:
        reasons.append(f"ring{round(score['ring_irregularity'], 3)}")
    if score["collision"] > _QUALITY_COLLISION:
        reasons.append(f"col{round(score['collision'], 2)}")
    if score["angle_deviation_deg"] > _QUALITY_ANGLE_DEVIATION_DEG:
        reasons.append(f"ang{round(score['angle_deviation_deg'], 1)}")
    reason_tag = ",".join(reasons) or "poor"

    if _regenerate_with_coordgen(mol):
        return f"{note};coordgen_regenerated:{reason_tag}"
    return f"{note};coordgen_failed:{reason_tag};kept_pose"


def refine_assembled_layout(mol) -> str:
    """Pose-preserving 2D refinement for an assembled Markush molecule.

    Delegates to *cleanup_structure_pose* — both decoded and assembled
    molecules use the same CoordGen + rigid-alignment pipeline.
    """
    return cleanup_structure_pose(mol)

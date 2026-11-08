"""Residue coordinates and connection geometry."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from .chemistry import _is_carbonyl_carbon, find_macroring, ordered_ring, residue_units


@dataclass
class ResidueFrame:
    atoms: np.ndarray
    local: np.ndarray
    n_atom: int
    c_atom: int
    backbone: list[int]
    span: float
    port_cosines: dict[tuple[int, int], float]
    backbone_template: str = "residue"


def _unit_tree_bonds(fragment: Chem.Mol, xy: np.ndarray, anchors: set[int]) -> np.ndarray:
    adjacency = [set() for _ in range(fragment.GetNumAtoms())]
    for bond in fragment.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].add(b)
        adjacency[b].add(a)
    xy = xy.copy()
    for bond in fragment.GetBonds():
        if bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ray = xy[b] - xy[a]
        length = float(np.linalg.norm(ray))
        if abs(length - 1) < 1e-8:
            continue
        if length < 1e-8:
            raise ValueError("collapsed_template_bond")
        moving, stack = set(), [b]
        while stack:
            atom = stack.pop()
            if atom in moving:
                continue
            moving.add(atom)
            stack.extend(
                neighbour for neighbour in adjacency[atom] - moving if {atom, neighbour} != {a, b}
            )
        if a in moving:
            raise ValueError("invalid_template_bridge")
        if moving & anchors:
            moving = set(range(fragment.GetNumAtoms())) - moving
            if moving & anchors:
                raise ValueError("inconsistent_template_anchors")
            shift = -ray * (1 / length - 1)
        else:
            shift = ray * (1 / length - 1)
        xy[sorted(moving)] += shift
    return xy


def peptide_frames(
    mol: Chem.Mol, *, backbone_template: Literal["auto", "residue", "cycle"] = "residue"
) -> tuple[list[ResidueFrame], list[int]]:
    """Build residue coordinates and connection geometry from the molecular graph."""
    if backbone_template not in ("auto", "residue", "cycle"):
        raise ValueError("unsupported_backbone_template")
    ring = find_macroring(mol)
    order = ordered_ring(mol, ring)
    units = None
    for candidate in (order, order[::-1]):
        trial = residue_units(mol, candidate)
        if trial and all(_is_carbonyl_carbon(mol, u["atoms"][-1]) for u in trial):
            units, order = trial, candidate
            break
    if not units or len(units) < 3:
        raise ValueError("not_a_head_to_tail_peptide")
    use_cycle = backbone_template == "cycle" or (backbone_template == "auto" and len(order) <= 15)
    cycle_positions = {}
    if use_cycle:
        radius = 1.5 / (2 * np.sin(np.pi / len(order)))
        cycle_positions = {
            atom: radius
            * np.array(
                [np.cos(2 * np.pi * index / len(order)), np.sin(2 * np.pi * index / len(order))]
            )
            for index, atom in enumerate(order)
        }
    cuts = []
    unit_of = {}
    for k, u in enumerate(units):
        for atom in u["atoms"]:
            unit_of[atom] = k
        a, b = u["atoms"][-1], units[(k + 1) % len(units)]["atoms"][0]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError("invalid_peptide_interface")
        cuts.append(bond.GetIdx())
    # A single bond joining two different residues (disulfide, thioether,
    # lactam bridge) cannot survive fragmentation, and it is not part of a
    # rigid frame: it becomes its own closure equality in the solver.
    # Sidechain atoms belong to no unit, so ownership is a multi-source
    # flood from every unit's backbone atoms; the nearest unit claims each
    # sidechain atom, and a bond across two owners is a crosslink.
    owner_of = {}
    flood = []
    for k, u in enumerate(units):
        for atom in u["atoms"]:
            owner_of[atom] = k
            flood.append(atom)
    head = 0
    while head < len(flood):
        atom = flood[head]
        head += 1
        for neighbour in mol.GetAtomWithIdx(atom).GetNeighbors():
            j = neighbour.GetIdx()
            if j not in owner_of:
                owner_of[j] = owner_of[atom]
                flood.append(j)
    crosslinks = []
    cut_set = set(cuts)
    for bond in mol.GetBonds():
        if bond.GetIdx() in cut_set:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if owner_of.get(a) is None or owner_of.get(b) is None:
            continue
        if owner_of[a] != owner_of[b]:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                raise ValueError("unsupported_crosslink_bond_order")
            crosslinks.append((a, b))
            cuts.append(bond.GetIdx())
    # FragmentOnBonds labels each cut's dummies with isotope bondIdx + 1,
    # which is how a fragment tells its amide ports from crosslink ports.
    cut_isotope = {bond_idx + 1: bond_idx for bond_idx in cuts}
    cut = Chem.FragmentOnBonds(mol, cuts, addDummies=True)
    mappings = []
    fragments = Chem.GetMolFrags(cut, asMols=True, fragsMolAtomMapping=mappings)
    original_n = mol.GetNumAtoms()
    by_atom = {}
    for fragment, mapping in zip(fragments, mappings, strict=True):
        for a in mapping:
            if a < original_n:
                by_atom[a] = (fragment, mapping)
    frames = []
    claimed = set()
    for u in units:
        n, c = u["atoms"][0], u["atoms"][-1]
        fragment, mapping = by_atom[n]
        support = [a for a in mapping if a < original_n]
        if (
            c not in support
            or claimed.intersection(support)
            or not set(u["atoms"]).issubset(support)
        ):
            raise ValueError("crosslinked_residue_support")
        coord_map = {}
        if use_cycle:
            from rdkit.Geometry import Point2D

            for index, atom in enumerate(mapping):
                if atom < original_n:
                    target = atom
                else:
                    neighbour = mapping[fragment.GetAtomWithIdx(index).GetNeighbors()[0].GetIdx()]
                    external = [
                        a.GetIdx()
                        for a in mol.GetAtomWithIdx(neighbour).GetNeighbors()
                        if a.GetIdx() not in support
                    ]
                    if len(external) != 1:
                        raise ValueError("invalid_cycle_template_port")
                    target = external[0]
                if target in cycle_positions:
                    coord_map[index] = Point2D(*cycle_positions[target])
        elif len(u["atoms"]) == 3:
            na, ca, co = [mapping.index(a) for a in u["atoms"]]
            if (
                not fragment.GetBondBetweenAtoms(na, ca).IsInRing()
                and not fragment.GetBondBetweenAtoms(ca, co).IsInRing()
            ):
                from rdkit.Geometry import Point2D

                coord_map = {
                    na: Point2D(-1.299038105676658, 0.0),
                    ca: Point2D(0.0, -0.75),
                    co: Point2D(1.299038105676658, 0.0),
                }
        rdDepictor.Compute2DCoords(fragment, canonOrient=False, forceRDKit=True, coordMap=coord_map)
        xyz = np.asarray(fragment.GetConformer().GetPositions())[:, :2]
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in fragment.GetBonds()]
        scale = 1.5 if use_cycle else np.median([np.linalg.norm(xyz[a] - xyz[b]) for a, b in bonds])
        xy = xyz / scale
        if use_cycle:
            xy = _unit_tree_bonds(fragment, xy, set(coord_map))
        ni, ci = mapping.index(n), mapping.index(c)
        centre = (xy[ni] + xy[ci]) / 2
        tangent = xy[ci] - xy[ni]
        span = float(np.linalg.norm(tangent))
        if span < 0.1:
            raise ValueError("degenerate_residue_chord")
        tangent /= span
        normal = np.array([-tangent[1], tangent[0]])
        local = (xy - centre) @ np.stack((tangent, normal), axis=1)
        ids = [mapping.index(a) for a in support]
        local = local[ids]

        side = [
            i
            for i, a in enumerate(support)
            if a not in u["atoms"] and mol.GetAtomWithIdx(a).GetAtomicNum() != 8
        ]
        score = float(local[side, 1].sum()) if side else float(local[:, 1].sum())
        if score > 0:
            local[:, 1] *= -1
        port_cosines = {}
        crosslink_cosines = {}
        cut_by_atom = {}
        for bond_idx in cuts:
            bond = mol.GetBondWithIdx(bond_idx)
            for end in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                if end in support:
                    cut_by_atom[end] = bond_idx
        for atom in sorted(cut_by_atom):
            ei = mapping.index(atom)
            # Every cut endpoint carries exactly one dummy in its fragment
            # (an atom with two inter-frame cuts is a rejected topology), so
            # connectivity identifies the port without relying on RDKit's
            # dummy isotope labelling, which varies by side and version
            dummy = [
                nb.GetIdx()
                for nb in fragment.GetAtomWithIdx(ei).GetNeighbors()
                if mapping[nb.GetIdx()] >= original_n
            ]
            if len(dummy) != 1:
                raise ValueError("invalid_residue_port")
            port = xy[dummy[0]] - xy[ei]
            table = port_cosines if atom in (n, c) else crosslink_cosines
            for nb in fragment.GetAtomWithIdx(ei).GetNeighbors():
                ni = nb.GetIdx()
                if mapping[ni] >= original_n:
                    continue
                v = xy[ni] - xy[ei]
                table[(atom, mapping[ni])] = float(
                    v @ port / (np.linalg.norm(v) * np.linalg.norm(port))
                )
        port_cosines.update(crosslink_cosines)
        frames.append(
            ResidueFrame(
                np.array(support),
                local,
                n,
                c,
                list(u["atoms"]),
                span,
                port_cosines,
                "cycle" if use_cycle else "residue",
            )
        )
        claimed.update(support)
    if claimed != set(range(original_n)):
        raise ValueError("disconnected_or_uncovered_atoms")
    return frames, order, crosslinks

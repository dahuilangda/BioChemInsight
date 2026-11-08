from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class GeometryLimits:
    min_bond_ratio: float = 0.75
    max_bond_ratio: float = 1.25
    min_nonbonded_ratio: float = 0.6
    min_atom_bond_ratio: float = 0.3


def geometry_report(molecule, limits=None):
    limits = GeometryLimits() if limits is None else limits
    points = np.asarray(molecule.GetConformer().GetPositions())[:, :2]
    if not np.isfinite(points).all():
        return {"accepted": False, "reasons": ["nonfinite_coordinates"]}
    bonds = np.asarray(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()]
    )
    if not len(bonds):
        return {"accepted": False, "reasons": ["no_bonds"]}
    starts, ends = points[bonds[:, 0]], points[bonds[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    unit = float(np.median(lengths))
    if unit < 1e-8 or lengths.min() < 1e-8:
        return {"accepted": False, "reasons": ["collapsed_bond"]}
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    excluded = np.eye(len(points), dtype=bool)
    excluded[bonds[:, 0], bonds[:, 1]] = True
    excluded[bonds[:, 1], bonds[:, 0]] = True
    nonbonded = distances[~excluded]
    min_nonbonded = float(nonbonded.min() / unit) if nonbonded.size else None
    displacement = points[:, None] - starts[None]
    projection = np.clip(
        (displacement * vectors[None]).sum(axis=-1) / lengths[None] ** 2, 0, 1
    )
    nearest = starts[None] + projection[..., None] * vectors[None]
    atom_bond = np.linalg.norm(points[:, None] - nearest, axis=-1)
    atom_bond[bonds[:, 0], np.arange(len(bonds))] = np.inf
    atom_bond[bonds[:, 1], np.arange(len(bonds))] = np.inf
    min_atom_bond = float(atom_bond.min() / unit)
    crossings = []

    def orientation(a, b, c):
        v, w = b - a, c - a
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]

    for index, (start, end) in enumerate(bonds):
        rest = np.arange(index + 1, len(bonds))
        independent = np.all((bonds[rest] != start) & (bonds[rest] != end), axis=1)
        rest = rest[independent]
        first = orientation(starts[index], ends[index], starts[rest])
        second = orientation(starts[index], ends[index], ends[rest])
        third = orientation(starts[rest], ends[rest], starts[index])
        fourth = orientation(starts[rest], ends[rest], ends[index])
        hit = (first * second < -1e-12) & (third * fourth < -1e-12)
        crossings.extend((index, int(other)) for other in rest[hit])
    reasons = []
    if (
        lengths.min() / unit < limits.min_bond_ratio
        or lengths.max() / unit > limits.max_bond_ratio
    ):
        reasons.append("bond_length_outside_limits")
    if min_nonbonded is not None and min_nonbonded < limits.min_nonbonded_ratio:
        reasons.append("nonbonded_atom_contact")
    if min_atom_bond < limits.min_atom_bond_ratio:
        reasons.append("atom_bond_contact")
    if crossings:
        reasons.append("bond_crossing")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "limits": asdict(limits),
        "median_bond_length": unit,
        "min_bond_ratio": float(lengths.min() / unit),
        "max_bond_ratio": float(lengths.max() / unit),
        "min_nonbonded_ratio": min_nonbonded,
        "min_atom_bond_ratio": min_atom_bond,
        "crossings": crossings,
    }


def stereo_roundtrip(molecule):
    block = Chem.MolToMolBlock(molecule, includeStereo=True)
    recovered = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
    return recovered is not None and Chem.MolToSmiles(recovered) == Chem.MolToSmiles(
        Chem.RemoveHs(molecule)
    )

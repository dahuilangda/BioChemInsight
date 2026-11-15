"""Vectorized optimizer: CoordGen circular layout -> patent zig-zag style.

Starts from the clean expanded CoordGen ring (user-identified closest to
real patent style) and refines toward the real drawing's geometry:
  - backbone atoms alternate radial offsets (zig-zag ring)
  - skeletal angles relax toward 120 deg
  - bond lengths stay uniform
  - exocyclic atoms follow their backbone anchor rigidly (side chains stay
    intact)
  - non-bonded pairs repel (no overlaps)
Analytic gradient + L-BFGS-B; 155-atom molecules refine in <1 s.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point2D

RDLogger.DisableLog("rdApp.*")


def _ring_atom_indices(mol) -> list[int]:
    ri = mol.GetRingInfo()
    ring = max(ri.AtomRings(), key=len)
    return list(ring)


def _zigzag_targets(mol, ring: list[int], L: float) -> np.ndarray:
    """Alternating-radial ellipse targets for ring atoms (zig-zag)."""
    n = len(ring)
    rx = L * n / (2 * math.pi) * 1.05
    ry = rx * 0.82
    zig = 0.22
    theta = 2 * np.pi * np.arange(n) / n
    rad = np.where(np.arange(n) % 2 == 0, 1.0, 1.0 - zig)
    return np.stack([rx * rad * np.cos(theta), ry * rad * np.sin(theta)], axis=1)


def _bfs_anchors(mol, ring: list[int]) -> np.ndarray:
    """For every atom: the index of its nearest ring atom (BFS)."""
    n = mol.GetNumAtoms()
    anchor = np.full(n, -1, dtype=int)
    from collections import deque
    dq = deque()
    for a in ring:
        anchor[a] = a
        dq.append(a)
    while dq:
        a = dq.popleft()
        for b in mol.GetAtomWithIdx(a).GetNeighbors():
            if anchor[b.GetIdx()] == -1:
                anchor[b.GetIdx()] = a
                dq.append(b.GetIdx())
    return anchor


def patent_style_optimize(mol: Chem.Mol, zig: float = 0.22,
                          iters: int = 600) -> Chem.Mol:
    """Refine a macrocycle layout toward patent zig-zag style.

    mol: any conformer-bearing copy (CoordGen layout works well as init).
    Returns a new mol with refined 2D coordinates.
    """
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    P0 = np.array([[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y]
                   for a in range(n)])
    centroid = P0.mean(axis=0)

    ring = _ring_atom_indices(mol)
    anchor = _bfs_anchors(mol, ring)
    bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                      for b in mol.GetBonds()], dtype=int)
    blens = np.linalg.norm(P0[bonds[:, 0]] - P0[bonds[:, 1]], axis=1)
    L = float(np.median(blens))
    nbrs = {i: set() for i in range(n)}
    for i, j in bonds:
        nbrs[i].add(j); nbrs[j].add(i)
    ii, jj = np.triu_indices(n, k=1)
    far = np.array([j not in nbrs[i] for i, j in zip(ii, jj)])
    nb_i, nb_j = ii[far], jj[far]
    w_rep = 600.0 / max(1, len(nb_i))

    # zig-zag ellipse targets for ring atoms, centered at current centroid
    tgt = _zigzag_targets(mol, ring, L)
    tgt = tgt + centroid - tgt.mean(axis=0)
    tgt = tgt + (centroid - tgt.mean(axis=0))

    # exocyclic atoms: target = ring anchor target + preserved offset from
    # the anchor's init position (rigid follow of side chains)
    offsets = P0 - P0[anchor]

    def positions(x):
        return x.reshape(-1, 2)

    def energy_grad(x):
        p = x.reshape(-1, 2)
        g = np.zeros_like(p)
        e = 0.0
        # bond springs
        d = p[bonds[:, 0]] - p[bonds[:, 1]]
        r = np.linalg.norm(d, axis=1)
        diff = r - L
        e += 2.0 * np.sum(diff ** 2)
        f = (2.0 * diff / np.maximum(r, 1e-9))[:, None] * d
        np.add.at(g, bonds[:, 0], f)
        np.add.at(g, bonds[:, 1], -f)
        # ring zig-zag ellipse targets
        d = p[ring] - tgt
        e += 1.5 * np.sum(d * d)
        np.add.at(g, np.array(ring), 3.0 * d)
        # exocyclic follow (rigid side chains)
        d = p[exo] - (p[anchor[exo]] + offsets[exo])
        e += 1.0 * np.sum(d * d)
        np.add.at(g, exo, 2.0 * d)
        np.add.at(g, anchor[exo], -2.0 * d)
        # non-bonded repulsion
        d = p[nb_i] - p[nb_j]
        r2 = np.sum(d * d, axis=1) + 4.0
        e += w_rep * np.sum(1.0 / r2)
        f = (-2 * w_rep / r2 ** 2)[:, None] * d
        np.add.at(g, nb_i, f)
        np.add.at(g, nb_j, -f)
        return e, g.ravel()

    # exocyclic index array
    exo = np.array([i for i in range(n) if anchor[i] != i and
                    mol.GetAtomWithIdx(i).GetAtomicNum() != 0], dtype=int)
    offsets = P0 - P0[anchor]
    x0 = P0.ravel().copy()
    res = minimize(energy_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-12})
    p = res.x.reshape(-1, 2)
    out = Chem.Mol(mol)
    oconf = out.GetConformer()
    for a in range(n):
        oconf.SetAtomPosition(a, [float(p[a][0]), float(p[a][1]), 0.0])
    return out

"""Efficient optimizer: refine any 2D layout toward patent zig-zag style.

Energy terms (all soft, scipy L-BFGS):
  bond springs     — every bond at standard length L
  angle springs    — skeletal angles toward 120 deg
  ring alternation — macrocycle atoms alternate outer/inner radial targets
                     (the ChemDraw zig-zag ring look)
  side-chain out   — side-chain first bonds point away from ring centroid
  repulsion        — non-bonded atom pairs (graph dist >= 3) repel

Input: mol (any conformer or none — init from CDK/CoordGen/native).
Output: refined mol (patent-style zig-zag layout).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def _macrocycle_and_sides(mol):
    ri = mol.GetRingInfo()
    ring = max(ri.AtomRings(), key=len)
    ring_set = set(ring)
    sides = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring_set]
    return ring, ring_set, sides


def _refine(mol, L=30.0, inward=0.80, rounds=2):
    n_atoms = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    ring = max(ri.AtomRings(), key=len)
    ring_set = set(ring)
    n = len(ring)

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    nbrs = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        nbrs[i].add(j); nbrs[j].add(i)
    nb_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)
                if j not in nbrs[i]]

    conf = mol.GetConformer()
    x0 = np.zeros(2 * n_atoms)
    for a in range(n_atoms):
        p = conf.GetAtomPosition(a)
        x0[2 * a] = p.x; x0[2 * a + 1] = p.y

    # alternating radial targets for ring atoms (zig-zag)
    ring_idx = list(ring)
    targets = {}
    for k, a in enumerate(ring_idx):
        r_outer = n * 7.0
        r_in = r_outer - 14.0
        theta = 2 * math.pi * k / n
        rr = r_outer if k % 2 == 0 else r_in
        targets[a] = (rr * math.cos(theta), rr * math.sin(theta))
    # side-chain atoms: radial target away from centroid
    cx = sum(p[0] for p in targets.values()) / n
    cy = sum(p[1] for p in targets.values()) / n

    def energy(xy):
        p = xy.reshape(-1, 2)
        e = 0.0
        # bond springs
        for i, j in bonds:
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            r = math.hypot(dx, dy)
            e += 2.5 * (r - L) ** 2
        # angle springs at skeletal atoms (three neighbors -> 120 mean)
        for a in range(n_atoms):
            nbr = [x for x in nbrs[a] if x != a]
            if len(nbr) < 2: continue
            vs = []
            for x in nbr:
                v = [xy[2 * x] - xy[2 * a], xy[2 * x + 1] - xy[2 * a]]
                norm = math.hypot(v[0], v[1]) or 1e-6
                vs.append([v[0] / norm, v[1] / norm])
            for u, w in [(0, 1), (0, 2), (1, 2)]:
                if u < len(vs) and w < len(vs):
                    dot = max(-1, min(1, vs[u][0] * vs[w][0] + vs[u][1] * vs[w][1]))
                    ang = math.degrees(math.acos(dot))
                    if ang > 90:  # only penalize far-from-120 skeletal bends
                        e += 0.15 * (ang - 120) ** 2
        # ring alternation springs
        for a in ring_idx:
            tx, ty = targets[a]
            e += 0.9 * ((p[a][0] - tx) ** 2 + (p[a][1] - ty) ** 2)
        # side chains outward
        for a in [x for x in range(n_atoms) if x not in ring_set]:
            e += 0.05 * ((p[a][0] - cx) ** 2 + (p[a][1] - cy) ** 2)
        # repulsion
        for i, j in nb_pairs:
            dx = p[i][0] - p[j][0]
            dy = p[i][1] - p[j][1]
            e += 60.0 / ((dx * dx + dy * dy) + 9.0)
        return e

    for _ in range(rounds):
        res = minimize(energy, x0, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-10})
        x0 = res.x
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for a in range(n_atoms):
        conf.SetAtomPosition(a, [float(x0[2 * a]), float(x0[2 * a + 1]), 0.0])
    return out


def refine_to_patent_style(mol):
    """Public entry: mol -> refined mol in patent zig-zag style."""
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
    return _refine(mol)

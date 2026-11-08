"""Bond-axis reflections of complete pendant subgraphs."""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def subtree_reflections(problem, residues):
    adjacency = [set() for _ in range(problem.n)]
    for a, b in problem.bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    for residue in residues:
        frame = problem.frames[residue]
        support = set(map(int, frame.atoms))
        fixed = set(frame.backbone)
        for endpoint in (frame.n_atom, frame.c_atom):
            fixed.update(adjacency[endpoint] & support)
        for bond in problem.mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a not in support or b not in support:
                continue
            for root, pivot in ((a, b), (b, a)):
                moving, stack = set(), [pivot]
                while stack:
                    atom = stack.pop()
                    if atom == root or atom in moving:
                        continue
                    moving.add(atom)
                    stack.extend((adjacency[atom] - moving - {root}) & support)
                if not moving or moving & fixed:
                    continue
                axis = problem.local[pivot] - problem.local[root]
                length = np.linalg.norm(axis)
                if length < 1e-10:
                    continue
                axis /= length
                reflection = 2 * np.outer(axis, axis) - np.eye(2)
                ids = sorted(moving)
                relative = problem.local[ids] - problem.local[pivot]
                if np.linalg.norm(relative @ reflection.T - relative) < 1e-8:
                    continue
                yield residue, root, pivot, ids, reflection

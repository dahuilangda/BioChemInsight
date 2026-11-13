"""Bond-axis reflections of complete pendant subgraphs."""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def frame_bridges(molecule, owners):
    adjacency = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if owners[a] == owners[b]:
            adjacency[a].append(b)
            adjacency[b].append(a)
    discovered = np.full(len(adjacency), -1, dtype=int)
    low = np.full(len(adjacency), -1, dtype=int)
    parent = np.full(len(adjacency), -1, dtype=int)
    bridges, clock = set(), 0
    for root in range(len(adjacency)):
        if discovered[root] >= 0:
            continue
        discovered[root] = low[root] = clock
        clock += 1
        stack = [(root, iter(adjacency[root]))]
        while stack:
            atom, neighbours = stack[-1]
            neighbour = next(neighbours, None)
            if neighbour is None:
                stack.pop()
                previous = parent[atom]
                if previous >= 0:
                    low[previous] = min(low[previous], low[atom])
                    if low[atom] > discovered[previous]:
                        bridges.add(tuple(sorted((int(previous), atom))))
            elif discovered[neighbour] < 0:
                parent[neighbour] = atom
                discovered[neighbour] = low[neighbour] = clock
                clock += 1
                stack.append((neighbour, iter(adjacency[neighbour])))
            elif neighbour != parent[atom]:
                low[atom] = min(low[atom], discovered[neighbour])
    return bridges


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
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (
                tuple(sorted((a, b))) not in problem.frame_bridges
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                continue
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

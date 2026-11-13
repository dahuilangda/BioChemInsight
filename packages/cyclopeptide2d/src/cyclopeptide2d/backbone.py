"""Amide-cycle discovery independent of a molecular ring basis."""

from collections import deque

import numpy as np
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def amide_cycle(molecule: Chem.Mol, *, search_limit: int = 10000) -> list[int] | None:
    carbonyls = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            for bond in atom.GetBonds()
        )
    }
    vertices = sorted(
        atom
        for atom in carbonyls
        if any(
            neighbour.GetAtomicNum() == 7
            for neighbour in molecule.GetAtomWithIdx(atom).GetNeighbors()
        )
    )
    if len(vertices) < 2:
        return None
    vertex_set = set(vertices)
    routes = {atom: [] for atom in vertices}
    for start in vertices:
        for nitrogen in molecule.GetAtomWithIdx(start).GetNeighbors():
            if nitrogen.GetAtomicNum() != 7:
                continue
            queue = deque(
                (atom.GetIdx(), (nitrogen.GetIdx(), atom.GetIdx()))
                for atom in nitrogen.GetNeighbors()
                if atom.GetIdx() != start and atom.GetAtomicNum() not in (1, 7)
            )
            visited = {start, nitrogen.GetIdx()}
            while queue:
                current, path = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                if current in carbonyls:
                    if current in vertex_set:
                        routes[start].append((current, path))
                    continue
                for neighbour in molecule.GetAtomWithIdx(current).GetNeighbors():
                    atom = neighbour.GetIdx()
                    if atom not in visited and neighbour.GetAtomicNum() not in (1, 7):
                        queue.append((atom, (*path, atom)))
    position = {atom: index for index, atom in enumerate(vertices)}
    row, column = [], []
    for atom in vertices:
        for target, _ in routes[atom]:
            row.append(position[atom])
            column.append(position[target])
    graph = csr_matrix((np.ones(len(row)), (row, column)), shape=(len(vertices), len(vertices)))
    _, labels = connected_components(graph, directed=True, connection="strong")
    components = {}
    for atom, label in zip(vertices, labels, strict=True):
        components.setdefault(int(label), set()).add(atom)
    best, best_key, states = None, (0, 0), 0
    for component in components.values():
        if len(component) < 2:
            continue
        local = {
            atom: [(target, path) for target, path in routes[atom] if target in component]
            for atom in component
        }
        starts = (
            [min(component)]
            if all(len(paths) == 1 for paths in local.values())
            else sorted(component)
        )
        for start in starts:
            stack = [(start, (start,), frozenset((start,)), 0)]
            while stack:
                atom, path, used, amides = stack.pop()
                states += 1
                if states > search_limit:
                    raise ValueError("backbone_cycle_search_budget_exceeded")
                for target, extension in local[atom]:
                    closing = target == start
                    extra = extension[:-1] if closing else extension
                    if target < start or used.intersection(extra):
                        continue
                    candidate = (*path, *extra)
                    if closing:
                        key = (amides + 1, -len(candidate))
                        if key > best_key:
                            best, best_key = candidate, key
                    elif target not in used:
                        stack.append((target, candidate, used.union(extra), amides + 1))
    if best is None:
        return None
    selected = set(best)
    for ring in molecule.GetRingInfo().AtomRings():
        if set(ring) == selected:
            return list(ring)
    return list(best)

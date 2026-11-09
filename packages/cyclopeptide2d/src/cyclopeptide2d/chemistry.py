"""Molecular graph decomposition and stereochemistry checks."""

from __future__ import annotations

from rdkit import Chem


def _adjacency(mol: Chem.Mol) -> list[set[int]]:
    n = mol.GetNumAtoms()
    adj = [set() for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _is_carbonyl_carbon(mol: Chem.Mol, idx: int) -> bool:
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() != 6:
        return False
    for b in a.GetBonds():
        other = b.GetOtherAtomIdx(idx)
        if (
            b.GetBondType() == Chem.BondType.DOUBLE
            and mol.GetAtomWithIdx(other).GetAtomicNum() == 8
        ):
            return True
    return False


def stereo_roundtrip_ok(mol: Chem.Mol) -> bool:
    """Check molecular identity and stereochemistry after MOL serialization."""
    try:
        mb = Chem.MolToMolBlock(mol, includeStereo=True)
        back = Chem.MolFromMolBlock(mb, sanitize=True)
        if back is None:
            return False
        return Chem.MolToSmiles(back) == Chem.MolToSmiles(mol)
    except (RuntimeError, ValueError):
        return False


def _is_amide_nitrogen(mol: Chem.Mol, idx: int) -> bool:
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() != 7:
        return False
    return any(
        b.GetBondType() == Chem.BondType.SINGLE and _is_carbonyl_carbon(mol, b.GetOtherAtomIdx(idx))
        for b in a.GetBonds()
    )


def peptide_macroring(mol: Chem.Mol) -> list[int] | None:
    """Resolve the principal amide cycle before consulting the ring basis."""
    from .backbone import amide_cycle

    return amide_cycle(mol)


def find_macroring(mol: Chem.Mol) -> list[int]:
    """The polypeptide ring: amide-chain walk first, largest SSSR ring fallback.

    The smallest-set-of-smallest-rings does not necessarily contain the
    backbone cycle of a crosslinked peptide, so the chemistry-defined walk
    runs first and the geometric heuristic only covers nonstandard rings."""
    walk = peptide_macroring(mol)
    if walk is not None:
        return walk
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        raise ValueError("molecule has no ring")
    best = sorted(
        rings,
        key=lambda r: (len(r), sum(1 for i in r if _is_carbonyl_carbon(mol, i))),
        reverse=True,
    )[0]
    return list(best)


def linear_backbone_order(mol: Chem.Mol) -> list[int] | None:
    """Resolve an unbranched alpha-peptide backbone through directed amide links."""
    residues = {}
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetAtomicNum() != 7:
            continue
        routes = []
        for alpha in nitrogen.GetNeighbors():
            if alpha.GetAtomicNum() != 6 or _is_carbonyl_carbon(mol, alpha.GetIdx()):
                continue
            for carbonyl in alpha.GetNeighbors():
                if _is_carbonyl_carbon(mol, carbonyl.GetIdx()):
                    routes.append((nitrogen.GetIdx(), alpha.GetIdx(), carbonyl.GetIdx()))
        if len(routes) == 1:
            residues[nitrogen.GetIdx()] = routes[0]
        elif routes:
            return None
        # N-methylated backbone nitrogens (sarcosine) reach this branch only
        # when their alpha route exists above; a nitrogen without any route
        # is a cap or sidechain amide and simply does not start a residue.
    if len(residues) < 2:
        return None
    following = {}
    incoming = {nitrogen: 0 for nitrogen in residues}
    for nitrogen, (_, _, carbonyl) in residues.items():
        neighbours = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
            if atom.GetIdx() in residues and atom.GetIdx() != nitrogen
        ]
        if len(neighbours) > 1:
            return None
        if neighbours:
            following[nitrogen] = neighbours[0]
            incoming[neighbours[0]] += 1
    starts = [nitrogen for nitrogen, count in incoming.items() if count == 0]
    if len(starts) != 1 or any(count > 1 for count in incoming.values()):
        return None
    current, visited, order = starts[0], set(), []
    while current not in visited:
        visited.add(current)
        order.extend(atom for atom in residues[current] if atom is not None)
        if current not in following:
            break
        current = following[current]
    if len(visited) != len(residues) or len(set(order)) != len(order):
        return None
    blocked = set(order[1:-1])
    reachable, stack = set(), [order[0]]
    while stack:
        atom = stack.pop()
        if atom in reachable:
            continue
        reachable.add(atom)
        stack.extend(
            neighbour.GetIdx()
            for neighbour in mol.GetAtomWithIdx(atom).GetNeighbors()
            if neighbour.GetIdx() not in blocked and neighbour.GetIdx() not in reachable
        )
    if order[-1] in reachable:
        return None
    return order


def ordered_ring(mol: Chem.Mol, atom_ids: list[int]) -> list[int]:
    """Return cycle atoms in bond order."""
    n = len(atom_ids)
    aset = set(atom_ids)
    cur, prev = atom_ids[0], None
    order = [cur]
    while len(order) < n:
        nxts = [
            b.GetOtherAtomIdx(cur)
            for b in mol.GetAtomWithIdx(cur).GetBonds()
            if b.GetOtherAtomIdx(cur) in aset and b.GetOtherAtomIdx(cur) != prev
        ]
        if not nxts:
            raise ValueError("ring walk failed")
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def residue_units(
    mol: Chem.Mol, order: list[int], chord_pairs: list[tuple] | None = None
) -> list[dict] | None:
    """Partition an ordered cycle at neutral amide nitrogens.

    Return residue atom indices and cycle positions, or None if segmentation fails."""
    adj = _adjacency(mol)
    pos = {a: k for k, a in enumerate(order)}
    m = len(order)
    cyc = {a: {x for x in adj[a] if x in pos and x != a} for a in order}

    def is_amide_n(idx: int) -> bool:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() != 7 or a.GetFormalCharge() != 0:
            return False
        return any(
            mol.GetAtomWithIdx(nb).GetAtomicNum() == 6 and _is_carbonyl_carbon(mol, nb)
            for nb in cyc[idx]
        )

    n_positions = [k for k, a in enumerate(order) if is_amide_n(a)]
    if not n_positions:
        return None
    units: list[dict] = []
    for j, nk in enumerate(n_positions):
        end = n_positions[(j + 1) % len(n_positions)]
        span = (end - nk) % m
        if span < 2:
            return None
        atoms = [order[(nk + t) % m] for t in range(span)]
        units.append({"atoms": atoms, "start": nk, "end": end})
    if sum(len(u["atoms"]) for u in units) != m:
        return None
    for u in units:
        u["chord_edges"] = []
        lo, hi = u["start"], u["start"] + len(u["atoms"])
        for ch in chord_pairs or []:
            for p in ch[:2]:
                if lo <= p < hi:
                    u["chord_edges"].append(p)
    return units

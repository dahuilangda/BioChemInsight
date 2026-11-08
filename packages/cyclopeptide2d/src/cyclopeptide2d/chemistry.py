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


def _amide_adjacency(mol: Chem.Mol):
    """Backbone carbon/nitrogen classification for the ring walk."""
    carbonyls = [a.GetIdx() for a in mol.GetAtoms() if _is_carbonyl_carbon(mol, a.GetIdx())]
    return carbonyls


def peptide_macroring(mol: Chem.Mol) -> list[int] | None:
    """Walk the ring C(=O) -> N -> CA -> C(=O) ... along backbone bonds.

    Sidechain amides are dead ends (a terminal CONH2 has no further
    backbone), and crosslinks never bend the ring, so the chemistry-defined
    walk is immune to them. Returns None when no closed backbone cycle
    exists."""
    def is_n(idx):
        return mol.GetAtomWithIdx(idx).GetAtomicNum() == 7

    def is_carbon(idx):
        return mol.GetAtomWithIdx(idx).GetAtomicNum() == 6

    def neighbours(idx, exclude=()):
        return [
            b.GetOtherAtomIdx(idx)
            for b in mol.GetAtomWithIdx(idx).GetBonds()
            if b.GetOtherAtomIdx(idx) not in exclude
        ]

    def walk_step(atom, prev):
        """Next backbone atom, or None at a dead end or an ambiguity."""
        if _is_carbonyl_carbon(mol, atom):
            nxts = [i for i in neighbours(atom, (prev,)) if is_n(i)]
            return nxts[0] if len(nxts) == 1 else None
        if is_n(atom):
            nxts = [i for i in neighbours(atom, (prev,)) if is_carbon(i)]
            return nxts[0] if len(nxts) == 1 else None
        # CA: continue into the backbone carbonyl — a sidechain amide
        # carbonyl dead-ends (its nitrogen has no further backbone carbon),
        # which filters Asn/Gln branching
        nxts = []
        for i in neighbours(atom, (prev,)):
            if not _is_carbonyl_carbon(mol, i):
                continue
            amide_nitrogens = [j for j in neighbours(i) if is_n(j) and j != atom]
            if any(
                any(is_carbon(j) for j in neighbours(n, (i,)))
                for n in amide_nitrogens
            ):
                nxts.append(i)
        return nxts[0] if len(nxts) == 1 else None

    for start in [a.GetIdx() for a in mol.GetAtoms() if _is_carbonyl_carbon(mol, a.GetIdx())]:
        amide_ns = [i for i in neighbours(start) if is_n(i)]
        if not amide_ns:
            continue  # terminal or sidechain carbonyl
        for second in amide_ns:
            order = [start, second]
            previous, current = start, second
            ok = False
            for _ in range(mol.GetNumAtoms() + 2):
                nxt = walk_step(current, previous)
                if nxt is None:
                    break
                if nxt == start:
                    ok = True
                    break
                order.append(nxt)
                previous, current = current, nxt
            if ok:
                return order
    return None


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

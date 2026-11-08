"""Chemical graph decomposition and stereo invariants."""
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
        if (b.GetBondType() == Chem.BondType.DOUBLE
                and mol.GetAtomWithIdx(other).GetAtomicNum() == 8):
            return True
    return False

def stereo_roundtrip_ok(mol: Chem.Mol) -> bool:
    """True when the drawn 2D coordinates reproduce the input stereo: the
    molblock writer derives every wedge from the chiral tags + placed
    coordinates; re-parsing re-derives the tags from the wedges.  Any
    coordinate arrangement that would print the wrong wedge/dash at a
    C alpha (e.g. after an inside flip) fails here."""
    try:
        mb = Chem.MolToMolBlock(mol, includeStereo=True)
        back = Chem.MolFromMolBlock(mb, sanitize=True)
        if back is None:
            return False
        return Chem.MolToSmiles(back) == Chem.MolToSmiles(mol)
    except Exception:
        return False

def find_macroring(mol: Chem.Mol) -> list[int]:
    """Atom cycle of the macroring: the largest ring; ties broken toward
    the ring carrying the most amide carbonyls (the peptide backbone ring,
    not a fused small ring)."""
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        raise ValueError("molecule has no ring")
    best = sorted(rings,
                  key=lambda r: (len(r),
                                 sum(1 for i in r
                                     if _is_carbonyl_carbon(mol, i))),
                  reverse=True)[0]
    return list(best)

def ordered_ring(mol: Chem.Mol, atom_ids: list[int]) -> list[int]:
    """Walk the cycle in bond order starting from atom_ids[0]."""
    n = len(atom_ids)
    aset = set(atom_ids)
    cur, prev = atom_ids[0], None
    order = [cur]
    while len(order) < n:
        nxts = [b.GetOtherAtomIdx(cur)
                for b in mol.GetAtomWithIdx(cur).GetBonds()
                if b.GetOtherAtomIdx(cur) in aset
                and b.GetOtherAtomIdx(cur) != prev]
        if not nxts:
            raise ValueError("ring walk failed")
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order

def residue_units(mol: Chem.Mol, order: list[int],
                  chord_pairs: list[tuple] | None = None) -> list[dict] | None:
    """Segment the drawn cycle into amino-acid residue units.

    A unit starts at a backbone amide N (neutral N on the cycle) and
    ends at the next carbonyl C on the cycle: the standard residue
    contributes [N, C_alpha, C(=O)] (glycine included), proline (or any
    residue whose side chain merges into the ring) contributes its
    extra ring carbons between N and C_alpha.  Returns None when the
    cycle does not parse as residues (non-peptide ring -- caller falls
    back to the geometric builders).

    Each unit: {"atoms": [cycle atom ids N..CO], "start": position,
    "chord_edges": [positions of chord endpoints inside the unit]}."""
    adj = _adjacency(mol)
    pos = {a: k for k, a in enumerate(order)}
    m = len(order)
    cyc = {a: {x for x in adj[a] if x in pos and x != a} for a in order}

    def is_amide_n(idx: int) -> bool:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() != 7 or a.GetFormalCharge() != 0:
            return False
        return any(mol.GetAtomWithIdx(nb).GetAtomicNum() == 6 and
                   _is_carbonyl_carbon(mol, nb) for nb in cyc[idx])

    n_positions = [k for k, a in enumerate(order) if is_amide_n(a)]
    if not n_positions:
        return None
    units: list[dict] = []
    for j, nk in enumerate(n_positions):
        end = n_positions[(j + 1) % len(n_positions)]
        span = (end - nk) % m
        if span < 2:
            return None                  # degenerate segmentation
        atoms = [order[(nk + t) % m] for t in range(span)]
        units.append({"atoms": atoms, "start": nk, "end": end})
    if sum(len(u["atoms"]) for u in units) != m:
        return None
    for u in units:
        u["chord_edges"] = []
        lo, hi = u["start"], u["start"] + len(u["atoms"])
        for ch in (chord_pairs or []):
            for p in ch[:2]:
                if lo <= p < hi:
                    u["chord_edges"].append(p)
    return units

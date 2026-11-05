"""Residue typing: decompose a decoded cyclic-peptide SMILES into its
ordered residue/unit codes, typed against the vocabulary.

Backbone model: the macrocycle ring path is segmented at every amide N
(a ring N bonded to an in-path carbonyl C). Each N→N interval is one unit
(single residues, multi-alpha units like lanthionine bridges or
pseudoproline rings — anything the vocabulary can build).

Typing: each unit's atoms (path span + side chains hanging off it) are
extracted as a submolecule, the unit amine N is anchored with atom map 1,
stereochemistry stripped, and the canonical SMILES compared against
precomputed signatures for every vocabulary code. Isomorphic units with the
same amine anchor produce identical signatures.

API:
  smiles_to_sequence(smiles) -> (codes, ok, detail)
    codes in cycle order starting from an arbitrary unit (caller rotates
    against an anchor list for comparison).
"""
from __future__ import annotations

import math
from collections import deque

from rdkit import Chem, RDLogger

from utils.sequence_parser import (
    EXOTIC_RESIDUE_FRAGMENTS,
    EXOTIC_RESIDUE_UNITS,
    _EXOTIC_RING_RESIDUES,
    _unit_fragment,
)

RDLogger.DisableLog("rdApp.*")

# codes enumerated in the same order as _unit_fragment's vocabulary walk;
# built lazily on first use
_SIGNATURES: dict[str, str] | None = None


def _strip_stereo(mol):
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    return mol


def reference_signatures() -> dict[str, str]:
    """code -> canonical unit signature, for every buildable code."""
    global _SIGNATURES
    if _SIGNATURES is None:
        sigs: dict[str, str] = {}
        vocab = dict(EXOTIC_RESIDUE_UNITS)
        vocab.update(EXOTIC_RESIDUE_FRAGMENTS)
        vocab.update(_EXOTIC_RING_RESIDUES)
        for code, frag in vocab.items():
            if isinstance(frag, tuple):
                s1, s2 = frag
                frag = f"N[C@]({s1})({s2})C(=O)"
            ref_mol = Chem.MolFromSmiles(frag)
            if ref_mol is None or ref_mol.GetNumAtoms() < 2:
                continue
            # terminal carbonyl C (last atom of every linear fragment):
            # aldehyde H in the standalone vs N-bonded in the macrocycle —
            # strip it so both sides match
            carb = ref_mol.GetAtomWithIdx(ref_mol.GetNumAtoms() - 2)
            if carb.GetSymbol() == "C":
                carb.SetNoImplicit(True)
                carb.SetNumExplicitHs(0)
            ref_mol = _strip_stereo(ref_mol)
            try:
                ref_mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            sig = Chem.MolToSmiles(ref_mol)
            if sig:
                sigs[code] = sig
        _SIGNATURES = sigs
    return _SIGNATURES


def _largest_ring_path(mol) -> list[int]:
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return []
    ring = max(ri.AtomRings(), key=len)
    return list(ring)


def _amide_n_indices(mol, path: set[int]) -> list[int]:
    """Ring N indices that are backbone amide nitrogens: bonded to an
    in-path carbonyl carbon."""
    out = []
    for i in path:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != "N":
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in path or nb.GetSymbol() != "C":
                continue
            for b in nb.GetBonds():
                if b.GetBondTypeAsDouble() == 2:
                    for nn in nb.GetNeighbors():
                        if nn.GetSymbol() == "O" and nn.GetIdx() not in path:
                            out.append(i)
                            break
                    else:
                        continue
                    break
            else:
                continue
            break
    return sorted(out)


def _extract_unit(mol, atoms: set[int]):
    """RWMol copy of the unit's atoms (same extraction both sides)."""
    rw = Chem.RWMol()
    idx_map = {}
    for a in sorted(atoms):
        src = mol.GetAtomWithIdx(a)
        copy = Chem.Atom(src.GetSymbol())
        copy.SetFormalCharge(src.GetFormalCharge())
        copy.SetNoImplicit(src.GetNoImplicit())
        copy.SetNumExplicitHs(src.GetTotalNumHs())
        idx_map[a] = rw.AddAtom(copy)
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in idx_map and c in idx_map:
            rw.AddBond(idx_map[a], idx_map[c], b.GetBondType())
    sub = rw.GetMol()
    sub = _strip_stereo(sub)
    try:
        sub.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return sub


def _unit_canonical(mol, atoms: set[int]) -> str:
    return Chem.MolToSmiles(_extract_unit(mol, atoms))


def smiles_to_sequence(smiles: str) -> tuple[list[str], bool, str]:
    """Type a cyclic-peptide SMILES into ordered vocabulary codes.

    Returns (codes, ok, detail). codes are in cycle order starting from an
    arbitrary unit; ok=False when the molecule has no typeable macrocycle
    or a unit matches no vocabulary code."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return [], False, "SMILES does not parse"
    Chem.RemoveStereochemistry(mol)
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    path = _largest_ring_path(mol)
    if len(path) < 9:
        return [], False, f"no large macrocycle ring (largest ring {len(path)} atoms)"
    path_set = set(path)
    ring_order = path + path  # doubled for circular walking
    cuts = _amide_n_indices(mol, path_set)
    if not cuts:
        return [], False, "no backbone amide junctions found on the macrocycle"
    # unit atoms: path span + side chains (BFS ancestry: every non-ring
    # atom inherits the ring-path position of its nearest ring ancestor)
    neighbors = {i: [] for i in range(mol.GetNumAtoms())}
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        neighbors[a].append(c)
        neighbors[c].append(a)
    pos_of = {a: k for k, a in enumerate(path)}
    origin = {a: pos_of[a] for a in path}
    order = deque(path)
    seen = set(path)
    while order:
        i = order.popleft()
        for j in neighbors[i]:
            if j not in seen:
                seen.add(j)
                origin[j] = origin[i]
                order.append(j)
    units: list[set[int]] = [set() for _ in cuts]
    ncuts = len(cuts)
    cut_pos = sorted(pos_of[c] for c in cuts)
    for atom, k in origin.items():
        ci = 0
        for c in range(ncuts):
            if k >= cut_pos[c]:
                ci = c
        units[ci].add(atom)
    sigs = reference_signatures()
    out: list[str] = []
    unknown = []
    start = None
    for ci in range(ncuts):
        atoms = units[ci]
        amine_n = path[cut_pos[ci]]
        sub = _unit_canonical(mol, atoms)
        match = None
        for code, ref in sigs.items():
            if ref == sub:
                match = code
                break
        if match is None:
            unknown.append(sub)
            out.append(f"?{ci}")
        else:
            out.append(match)
            if start is None:
                start = match
    if unknown:
        return out, False, f"{len(unknown)} unit(s) match no vocabulary code"
    return out, True, ""


def compare_with_anchor(smiles: str, anchor_codes: list[str]) -> dict:
    """Typed comparison: decoded residues vs sequence-anchor residues,
    rotated to the best alignment. Returns per-position agreement and the
    residue-level diff."""
    got, ok, detail = smiles_to_sequence(smiles)
    if not ok:
        return {"ok": False, "detail": detail}
    n, a = len(got), len(anchor_codes)
    best = None
    for shift in range(n):
        rot = [got[(shift + j) % n] for j in range(n)]
        matches = sum(1 for j in range(min(n, a)) if rot[j] == anchor_codes[j % a])
        score = matches - abs(n - a)
        if best is None or score > best[0]:
            best = (score, shift, matches)
    _, shift, matches = best
    aligned = [got[(shift + j) % got.__len__()] for j in range(len(got))]
    diffs = []
    for j in range(max(len(aligned), len(anchor_codes))):
        d = aligned[j % len(aligned)] if j < len(aligned) else "(missing)"
        r = anchor_codes[j % len(anchor_codes)] if anchor_codes else "(none)"
        if d != r:
            diffs.append({"position": j + 1, "decoded": d, "anchor": r})
    agree = len(aligned) == len(anchor_codes) and not diffs
    return {"ok": True, "decoded_residues": aligned, "anchor_residues": anchor_codes,
            "residue_count_match": len(aligned) == len(anchor_codes),
            "matched_positions": matches, "diffs": diffs,
            "agree": agree}

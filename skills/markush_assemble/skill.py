"""Atomic skill: Markush template -> enumerated concrete molecules.

Assembly path: a template (SMILES with atom-mapped dummies, or a V3000
molblock) carries attachment dummies ([*:1]); each label pairs with a
fragment's dummy; Chem.molzip performs the bond formation, RDKit
sanitization is the gate. Two correctness checks guard every member:

  pre  — each paired label appears exactly once per side, and both
         dummies have a single neighbour
  post — graded sanitization (valence vs kekulization reported separately)
         plus formula cross-validation when the patent declares one

Combination enumeration: fragments given per label as a list enumerate
every combination (library size = product of candidate counts, capped).
V3000 molblocks route through skills.markush_assemble.v3000, which
normalizes LINKNODE placement and enumerates via rdMolEnumerator.
"""
from __future__ import annotations

import itertools
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from skills.markush_assemble import v3000
from skills.registry import register

RDLogger.DisableLog("rdApp.*")

_MAX_LIBRARY = 20000


@register("markush_assemble", "chemistry",
          "Markush template + R-group fragments -> concrete molecules")
async def assemble(payload: dict) -> dict:
    template_smiles = payload.get("template_smiles")
    molblock = payload.get("molblock")
    fragment_sets = payload.get("fragments") or {}
    declared_formula = payload.get("declared_formula")
    activity_rows = payload.get("activity_rows") or []

    gaps: list[str] = []
    members: list[dict] = []

    if molblock:
        try:
            enumerated, notes = v3000.enumerate_molblock(molblock)
        except v3000.V3000ParseError as exc:
            return {"ok": False, "error": f"molblock does not parse: {exc}"}
        gaps.extend(notes)
    elif template_smiles:
        base = Chem.MolFromSmiles(template_smiles)
        if base is None:
            return {"ok": False, "error": "template SMILES does not parse"}
        enumerated = [base]
    else:
        return {"ok": False, "error": "no template given"}

    # fragment combinations apply to every enumerated variant
    for variant in enumerated:
        labels = _label_counts(variant)
        combos, unknown = _combinations(labels, fragment_sets)
        if combos is None:
            return {"ok": False,
                    "error": f"library exceeds {_MAX_LIBRARY} members; "
                             "narrow the fragment lists"}
        gaps.extend(f"fragment label '{k}' not present in template; ignored"
                    for k in unknown)
        for combo in combos:
            ok, err = _precheck(variant, combo)
            if not ok:
                gaps.append(err)
                continue
            merged = _stitch(variant, combo)
            if merged is None:
                gaps.append("molzip failed for one combination")
                continue
            member, err2 = _gate(merged, declared_formula)
            if member is None:
                gaps.append(err2)
                continue
            members.append(member)

    joined = _join_activity(members, activity_rows)
    return {"ok": bool(members), "members": members,
            "activity_join": joined, "gaps": gaps}


def _dummy_label(atom) -> str:
    """Attachment label of a dummy atom: atom map number first, then the
    V3000 dummyLabel (RDKit stores 'R'-symbol dummies that way)."""
    map_num = atom.GetAtomMapNum()
    if map_num:
        return str(map_num)
    if atom.HasProp("dummyLabel"):
        return atom.GetProp("dummyLabel")
    return f"?{atom.GetIdx()}"


def _label_counts(mol) -> dict[str, list[int]]:
    """label -> indices of the dummy (atomic-number-0) atoms carrying it."""
    labels: dict[str, list[int]] = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        labels.setdefault(_dummy_label(a), []).append(a.GetIdx())
    return labels


def _combinations(labels, fragment_sets) -> tuple[list[dict], list[str]] | None:
    unknown = sorted(k for k in fragment_sets if k not in labels)
    active = {k: fragment_sets[k] for k in fragment_sets if k in labels}
    if not active:
        return [{}], unknown
    total = 1
    for v in active.values():
        total *= len(v)
    if total > _MAX_LIBRARY:
        return None, unknown
    keys = list(active)
    return ([dict(zip(keys, p)) for p in itertools.product(*(active[k] for k in keys))],
            unknown)


def _precheck(mol, combo: dict) -> tuple[bool, str]:
    for label, indices in _label_counts(mol).items():
        frag_smiles = combo.get(label)
        if frag_smiles is None:
            continue
        if len(indices) != 1:
            return False, f"label {label}: template side must have one dummy, found {len(indices)}"
        if mol.GetAtomWithIdx(indices[0]).GetDegree() != 1:
            return False, f"label {label}: template dummy must have one neighbour"
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return False, f"label {label}: fragment does not parse"
        f_indices = [a.GetIdx() for a in frag.GetAtoms()
                     if a.GetAtomicNum() == 0 and _dummy_label(a) == label]
        if f_indices and frag.GetAtomWithIdx(f_indices[0]).GetDegree() != 1:
            return False, f"label {label}: fragment dummy must have one neighbour"
    return True, ""


def _append_fragment(rw: Chem.RWMol, frag: Chem.Mol) -> int:
    """Append frag's atoms and bonds to rw; return frag's atom-0 index."""
    offset = rw.GetNumAtoms()
    for a in frag.GetAtoms():
        rw.AddAtom(Chem.Atom(a))
    for b in frag.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx() + offset,
                   b.GetEndAtomIdx() + offset, b.GetBondType())
    return offset


def _stitch(mol, combo: dict):
    """Merge template and its fragments. Labels present on both sides
    route through molzip (it forms the seam bonds); labels only on the
    template bond the dummy's neighbour to the fragment's first atom and
    drop the dummy. Returns None on any seam failure; the caller records
    the gap."""
    em = Chem.RWMol(mol)
    pending_plain: list[tuple[str, Chem.Mol]] = []
    for label, frag_smiles in combo.items():
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return None
        t_dummy = next((a.GetIdx() for a in em.GetAtoms()
                        if a.GetAtomicNum() == 0
                        and _dummy_label(a) == label), None)
        f_dummy = next((a.GetIdx() for a in frag.GetAtoms()
                        if a.GetAtomicNum() == 0
                        and _dummy_label(a) == label), None)
        if t_dummy is not None and f_dummy is not None:
            _append_fragment(em, frag)  # both dummies kept so molzip pairs
        elif t_dummy is not None:
            pending_plain.append((label, frag))

    # molzip only pairs labels that appear on both sides; running it with
    # just unpaired dummies would tear the template into fragments
    seam_counts: dict[str, int] = {}
    for a in em.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum():
            lbl = _dummy_label(a)
            seam_counts[lbl] = seam_counts.get(lbl, 0) + 1
    paired = any(n >= 2 for n in seam_counts.values())
    try:
        zipped = Chem.molzip(em.GetMol()) if paired else em.GetMol()
    except (ValueError, RuntimeError):
        return None

    merged = Chem.RWMol(zipped)
    for label, frag in pending_plain:
        t_idx = next((a.GetIdx() for a in merged.GetAtoms()
                      if a.GetAtomicNum() == 0 and _dummy_label(a) == label),
                     None)
        if t_idx is None:
            return None
        anchor_idx = next((n.GetIdx()
                           for n in merged.GetAtomWithIdx(t_idx).GetNeighbors()),
                          None)
        if anchor_idx is None:
            return None
        merged.RemoveAtom(t_idx)
        if anchor_idx > t_idx:
            anchor_idx -= 1
        frag_atom = _append_fragment(merged, frag)
        merged.AddBond(anchor_idx, frag_atom, Chem.BondType.SINGLE)
    return merged.GetMol()


def _gate(mol, declared_formula) -> tuple[dict | None, str]:
    """Graded sanitization + formula cross-validation."""
    try:
        Chem.SanitizeMol(mol)
    except Chem.AtomValenceException as exc:
        return None, f"valence rejected: {str(exc)[:90]}"
    except Chem.KekulizeException as exc:
        return None, f"kekulization rejected: {str(exc)[:90]}"
    except Exception as exc:   # last graded tier: reject one member, keep
        # the batch running; the reason is surfaced in the gap message
        return None, f"sanitization failed ({type(exc).__name__}): {str(exc)[:80]}"
    formula = rdMolDescriptors.CalcMolFormula(mol)
    note = ""
    if declared_formula and formula != declared_formula:
        note = f"formula differs from declared ({declared_formula})"
    member = {
        "smiles": Chem.MolToSmiles(mol),
        "formula": formula,
        "exact_mass": round(rdMolDescriptors.CalcExactMolWt(mol), 3),
        "inchikey": Chem.MolToInchiKey(mol),
    }
    if note:
        member["formula_note"] = note
    return member, ""


def _join_activity(members: list[dict], rows: list[dict]) -> list[dict]:
    """Evidence-chain join: a member claims an activity row through its
    assembly order only when ids exist; conflicts stay visible."""
    if not rows or not members:
        return []
    out = []
    for i, m in enumerate(members):
        for row in rows:
            cid = str(row.get("compound_id", ""))
            match = cid and (cid == f"member_{i + 1}" or cid == m["formula"])
            out.append({
                "member": i + 1, "smiles": m["smiles"],
                "compound_id": cid or None,
                "activity": {k: v for k, v in row.items() if k != "compound_id"},
                "matched": bool(match),
                "confidence": "high" if match else "unmatched",
                "evidence": [f"assembly index {i + 1}"],
            })
    return out

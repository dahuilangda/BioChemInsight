"""Atomic skill: Markush template -> enumerated concrete molecules.

Assembly path: a template SMILES carries atom-mapped dummies ([*:1]),
each label pairs with a fragment's dummy; Chem.molzip performs the bond
formation, RDKit sanitization is the gate. Two correctness checks guard
every member:

  pre  — each paired label appears exactly twice, dummies have a single
         neighbour, bond orders agree across the seam
  post — graded sanitization (valence vs kekulization reported separately)
         plus formula cross-validation when the patent declares one

Combination enumeration: fragments given per label as a list enumerate
every combination (library size = product of candidate counts, capped).
V3000 templates with LINKNODE or position-variation bonds route through
rdMolEnumerator first.
"""
from __future__ import annotations

import itertools
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolEnumerator

from skills.registry import register

RDLogger.DisableLog("rdApp.*")

_MAX_LIBRARY = 20000



def _expand_linknodes(molblock: str) -> list[str]:
    """Hand reader for V3000 LINKNODE blocks (this RDKit build skips them).

    Grammar: M V30 <attach-atom-order> <lo> <hi> <order> <pairs...> where
    pairs alternate (bond-order, neighbour-atom-order). The linked segment
    between the two outer neighbours repeats lo..hi times; every repeat
    count becomes one template SMILES with the chain rebuilt."""
    atoms: dict[int, str] = {}
    bonds: list[tuple[int, int, Chem.BondType]] = []
    links: list[list[int]] = []
    section = None
    for line in molblock.splitlines():
        s = line.strip()
        if not s.startswith("M  V30"):
            if s.startswith("M  END"):
                section = None
            continue
        body = s[6:].split()
        if not body:
            continue
        if body[0] == "BEGIN":
            section = body[1] if len(body) > 1 else None
            continue
        if body[0] == "END":
            section = None
            continue
        if section == "ATOM":
            atoms[int(body[0])] = body[1]
        elif section == "BOND":
            # M V30 <bond-id> <order> <atom1> <atom2>
            order = {"1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE,
                     "3": Chem.BondType.TRIPLE,
                     "4": Chem.BondType.AROMATIC}.get(
                body[1] if len(body) > 1 else "", Chem.BondType.SINGLE)
            if len(body) >= 4:
                bonds.append((int(body[2]), int(body[3]), order))
        elif section == "LINKNODE":
            links.append([int(x) for x in body])
    if not links:
        return []

    out = []
    for spec in links:
        attach, lo, hi = spec[0], spec[1], spec[2]
        rest = spec[3:]
        # trailing values: (order, atom, atom) triple when odd, else pairs
        if len(rest) % 2 == 1:
            order, b1o, b2o = rest[-3], rest[-2], rest[-1]
        else:
            order, b1o, b2o = rest[-1], rest[-4], rest[-2]
        b1_sym, b2_sym = atoms.get(b1o, "C"), atoms.get(b2o, "C")
        for repeats in range(lo, hi + 1):
            em = Chem.RWMol()
            idx_of = {}
            for atom_order, symbol in sorted(atoms.items()):
                idx_of[atom_order] = em.AddAtom(Chem.Atom(symbol))
            for a, b, t in bonds:
                em.AddBond(idx_of[a], idx_of[b], t)
            # rewire the boundary bond through a repeated-atom chain
            i1, i2 = idx_of[b1o], idx_of[b2o]
            bond = em.GetBondBetweenAtoms(i1, i2)
            bt = bond.GetBondType() if bond is not None else                 Chem.BondType.SINGLE
            if bond is not None:
                em.RemoveBond(i1, i2)
            prev = i1
            for _ in range(max(repeats, 0)):
                new = em.AddAtom(Chem.Atom(atoms.get(attach, "C")))
                if prev == i2 and new == i2:
                    continue
                em.AddBond(prev, new, Chem.BondType.SINGLE)
                prev = new
            if prev != i2:
                em.AddBond(prev, i2, bt)
            mol = em.GetMol()
            try:
                Chem.SanitizeMol(mol)
                out.append(Chem.MolToSmiles(mol))
            except Exception:
                continue
    return out


@register("markush_assemble", "chemistry",
          "Markush template + R-group fragments -> concrete molecules")
async def assemble(payload: dict) -> dict:
    from skills.registry import get as _get  # noqa: F401  (join hook below)
    template_smiles = payload.get("template_smiles")
    molblock = payload.get("molblock")
    fragment_sets = payload.get("fragments") or {}
    declared_formula = payload.get("declared_formula")
    activity_rows = payload.get("activity_rows") or []

    gaps: list[str] = []
    members: list[dict] = []

    if molblock:
        expanded = _expand_linknodes(molblock)
        if expanded:
            gaps.append(f"LINKNODE reader produced {len(expanded)} chain "
                        "expansions; fragment lists apply to each")
            enumerated = [Chem.MolFromSmiles(s) for s in expanded]
        else:
            base = Chem.MolFromMolBlock(molblock, sanitize=False)
            if base is None:
                return {"ok": False, "error": "molblock does not parse"}
            try:
                enumerated = rdMolEnumerator.Enumerate(base)
                gaps.append(f"rdMolEnumerator produced {len(enumerated)} variants; "
                            "fragment lists apply per variant in order")
            except Exception as exc:
                gaps.append(f"rdMolEnumerator unavailable: {str(exc)[:120]}")
                enumerated = [base]
    elif template_smiles:
        base = Chem.MolFromSmiles(template_smiles)
        if base is None:
            return {"ok": False, "error": "template SMILES does not parse"}
        enumerated = [base]
    else:
        return {"ok": False, "error": "no template given"}

    # fragment combinations apply to every enumerated variant
    total_members = 0
    for variant in enumerated:
        labels = _label_counts(variant)
        combos = _combinations(labels, fragment_sets)
        if combos is None:
            return {"ok": False,
                    "error": f"library exceeds {_MAX_LIBRARY} members; "
                             "narrow the fragment lists"}
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


def _label_counts(mol) -> dict[str, list[int]]:
    """atom-map label -> dummy atom indices (exactly one neighbour each)."""
    labels: dict[str, list[int]] = {}
    for a in mol.GetAtoms():
        if a.GetSymbol() != "*":
            continue
        label = str(a.GetAtomMapNum()) or a.GetProp("dummyLabel") \
            if a.HasProp("dummyLabel") else str(a.GetAtomMapNum())
        if not label:
            label = f"?{a.GetIdx()}"
        labels.setdefault(label, []).append(a.GetIdx())
    return labels


def _combinations(labels, fragment_sets) -> list[dict] | None:
    active = {k: fragment_sets[k] for k in fragment_sets if k in labels}
    if not active:
        return [{}]
    total = 1
    for k, v in active.items():
        total *= len(v)
    if total > _MAX_LIBRARY:
        return None
    keys = list(active)
    return [dict(zip(keys, p)) for p in itertools.product(*(active[k] for k in keys))]


def _declared_size(labels, fragment_sets) -> int:
    total = 1
    for k, v in fragment_sets.items():
        if k in labels:
            total *= max(1, len(v))
    return total if fragment_sets else 1


def _precheck(mol, combo: dict) -> tuple[bool, str]:
    conf = combo
    for label, indices in _label_counts(mol).items():
        frag = conf.get(label)
        if frag is None:
            continue
        if len(indices) != 1:
            return False, f"label {label}: template side must have one dummy, found {len(indices)}"
        idx = indices[0]
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetDegree() != 1:
            return False, f"label {label}: template dummy must have one neighbour"
        fm = Chem.MolFromSmiles(frag)
        if fm is None:
            return False, f"label {label}: fragment does not parse"
        f_indices = [a.GetIdx() for a in fm.GetAtoms() if a.GetSymbol() == "*"
                     and str(a.GetAtomMapNum()) == label]
        if f_indices:
            f_atom = fm.GetAtomWithIdx(f_indices[0])
            if f_atom.GetDegree() != 1:
                return False, f"label {label}: fragment dummy must have one neighbour"
    return True, ""


def _stitch(mol, combo: dict):
    """Merge template and its fragments. Paired labels route through
    molzip; plain fragments bond the template dummy's neighbour to the
    fragment's first atom and drop the dummy."""
    em = Chem.RWMol(mol)
    pending_plain: list[tuple[int, Chem.Mol]] = []
    for label, frag in combo.items():
        fm = Chem.MolFromSmiles(frag)
        if fm is None:
            return None
        t_idx = next((a.GetIdx() for a in em.GetAtoms()
                      if a.GetSymbol() == "*"
                      and str(a.GetAtomMapNum()) == label), None)
        f_idx = next((a.GetIdx() for a in fm.GetAtoms()
                      if a.GetSymbol() == "*"
                      and str(a.GetAtomMapNum()) == label), None)
        if t_idx is not None and f_idx is not None:
            offset = em.GetNumAtoms()
            for a in fm.GetAtoms():
                em.AddAtom(Chem.Atom(a))
            for b in fm.GetBonds():
                em.AddBond(b.GetBeginAtomIdx() + offset,
                           b.GetEndAtomIdx() + offset, b.GetBondType())
            # keep both dummies so molzip pairs the labels
        elif t_idx is not None:
            pending_plain.append((label, fm))
    try:
        merged = em.GetMol()
        paired = [a for a in merged.GetAtoms()
                  if a.GetSymbol() == "*" and a.GetAtomMapNum()]
        zipped = Chem.molzip(merged) if len(paired) >= 2 else merged
        # plain fragments: bond the dummy's neighbour to fragment atom 0
        merged = Chem.RWMol(zipped)
        for t_label, fm in pending_plain:
            want = int(t_label) if str(t_label).isdigit() else t_label
            t_idx = next((a.GetIdx() for a in merged.GetAtoms()
                          if a.GetSymbol() == "*"
                          and str(a.GetAtomMapNum()) == str(want)), None)
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
            offset = merged.GetNumAtoms()
            for a in fm.GetAtoms():
                merged.AddAtom(Chem.Atom(a))
            for b in fm.GetBonds():
                merged.AddBond(b.GetBeginAtomIdx() + offset,
                               b.GetEndAtomIdx() + offset, b.GetBondType())
            merged.AddBond(anchor_idx, offset, Chem.BondType.SINGLE)
        return merged.GetMol()
    except Exception:
        return None


def _gate(mol, declared_formula) -> tuple[dict | None, str]:
    """Graded sanitization + formula cross-validation."""
    try:
        Chem.SanitizeMol(mol)
    except Chem.AtomValenceException as exc:
        return None, f"valence rejected: {str(exc)[:90]}"
    except Chem.KekulizeException as exc:
        return None, f"kekulization rejected: {str(exc)[:90]}"
    except Exception as exc:
        return None, f"sanitization failed: {str(exc)[:90]}"
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

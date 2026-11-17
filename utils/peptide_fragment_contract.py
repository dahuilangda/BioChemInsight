"""Fragment contracts for the peptide harness.

A reading (lexicon, catalog, OPSIN, vision, model) may verify a row only
after it satisfies the residue contract (chainable backbone amine + free
carboxyl, alpha OH stripped) or the acyl contract (N-cap). Chemistry
classification only — no code vocabulary lives here.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from skills.residue_lexicon import NotBackboneForm, _canonical_fragment


def _is_carbonyl_carbon(atom) -> bool:
    """Carbon bearing a C=O double bond."""
    return any(b.GetBondTypeAsDouble() == 2 and
               atom.GetOwningMol().GetAtomWithIdx(
                   b.GetOtherAtomIdx(atom.GetIdx())).GetSymbol() == "O"
               for b in atom.GetBonds())


def _is_amide_nitrogen(atom) -> bool:
    """A nitrogen acylated by a carbonyl: no hydrogen chemistry left to
    chain the next residue, so it cannot serve as a backbone amine."""
    for bond in atom.GetBonds():
        other = atom.GetOwningMol().GetAtomWithIdx(
            bond.GetOtherAtomIdx(atom.GetIdx()))
        if other.GetSymbol() != "C":
            continue
        if any(b.GetBondTypeAsDouble() == 2
               and other.GetOwningMol().GetAtomWithIdx(
                   b.GetOtherAtomIdx(other.GetIdx())).GetSymbol() == "O"
               for b in other.GetBonds()):
            return True
    return False


def _chainable_amine_indices(mol) -> list[int]:
    """Nitrogens with room for the incoming amide bond: heavy-atom degree
    <= 2 and not acylated. The FIRST N in SMILES order is often a side-
    chain amide N (Asn/Gln/N6-acetyl-...) — all nitrogens are candidates."""
    out = []
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == "N" and not _is_amide_nitrogen(atom)
                and atom.GetDegree() <= 2):
            out.append(atom.GetIdx())
    return out


def _chainable_amine_indices(mol) -> list[int]:
    """Nitrogens with room for the incoming amide bond: heavy-atom degree
    <= 2 and not acylated. The FIRST N in SMILES order is often a side-
    chain amide N (Asn/Gln/N6-acetyl-...) — all nitrogens are candidates."""
    out = []
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == "N" and not _is_amide_nitrogen(atom)
                and atom.GetDegree() <= 2):
            out.append(atom.GetIdx())
    return out


def _chainable_amine_indices(mol) -> list[int]:
    """Nitrogens with room for the incoming amide bond: heavy-atom degree
    <= 2 and not acylated. The FIRST N in SMILES order is often a side-
    chain amide N (Asn/Gln/N6-acetyl-...) — all nitrogens are candidates."""
    out = []
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == "N" and not _is_amide_nitrogen(atom)
                and atom.GetDegree() <= 2):
            out.append(atom.GetIdx())
    return out


def _nearest_carbonyl_carbon(mol, alpha_idx: int, blocked: int,
                             max_depth: int = 3) -> int | None:
    """Carbonyl carbon within `max_depth` bonds of alpha, any path that
    avoids the amine (the amine is where the previous residue bonds)."""
    seen = {alpha_idx, blocked}
    frontier = [alpha_idx]
    for depth in range(1, max_depth + 1):
        nxt = []
        for idx in frontier:
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                nb_idx = b.GetOtherAtomIdx(idx)
                if nb_idx in seen:
                    continue
                nb = mol.GetAtomWithIdx(nb_idx)
                if nb.GetSymbol() == "C" and _is_carbonyl_carbon(nb):
                    return nb_idx
                nxt.append(nb_idx)
                seen.add(nb_idx)
        frontier = nxt
    return None


def _strip_free_oh(em: Chem.RWMol, carbonyl_idx: int) -> bool:
    """Remove the carboxyl OH single-bonded to a carbonyl carbon."""
    for b in em.GetAtomWithIdx(carbonyl_idx).GetBonds():
        oc = em.GetAtomWithIdx(b.GetOtherAtomIdx(carbonyl_idx))
        if (oc.GetSymbol() == "O" and oc.GetDegree() == 1
                and any(bb.GetBondTypeAsDouble() == 1
                        for bb in oc.GetBonds())):
            em.RemoveAtom(oc.GetIdx())
            return True
    return False


def backbone_fragment(smiles: str) -> str | None:
    """Free-acid or already-backbone form -> canonical backbone fragment
    rooted at a chainable backbone amine. None when no reading can chain
    as a residue: every amine is acylated or over-substituted, or no
    carboxyl lies within three bonds of a chainable amine. Attachment-
    dummy readings ([n*]) expand first — only the free-acid variant
    survives. Cap-shaped readings route through cap_fragment. The
    operation is idempotent: backbone-form inputs pass through."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if has_dummy_atoms(smiles):
        for variant in resolve_dummy_atoms(smiles):
            frag = backbone_fragment(variant)
            if frag:
                return frag
        return None
    for amine_idx in _chainable_amine_indices(mol):
        # renumber so this amine is the first atom: the proven canonical
        # walk (_canonical_fragment) then reads the backbone from here
        order = [amine_idx] + [i for i in range(mol.GetNumAtoms())
                               if i != amine_idx]
        reordered = Chem.RenumberAtoms(mol, order)
        try:
            return _canonical_fragment(reordered)
        except (NotBackboneForm, Exception):
            continue
    return None


def is_amino_acid_skeleton(smiles: str) -> bool:
    return backbone_fragment(smiles) is not None


def _carboxyl_oh_pairs(mol):
    """(carboxyl C idx, terminal OH idx) for every free carboxylic acid."""
    pairs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        dbl_o = oh = None
        for bond in atom.GetBonds():
            other = mol.GetAtomWithIdx(bond.GetOtherAtomIdx(atom.GetIdx()))
            if bond.GetBondTypeAsDouble() == 2 and other.GetSymbol() == "O":
                dbl_o = other.GetIdx()
            elif (other.GetSymbol() == "O" and other.GetDegree() == 1
                    and other.GetFormalCharge() == 0):
                oh = other.GetIdx()
        if dbl_o is not None and oh is not None:
            pairs.append((atom.GetIdx(), oh))
    return pairs


def has_dummy_atoms(smiles: str) -> bool:
    return bool(smiles) and "*" in smiles


def resolve_dummy_atoms(smiles: str) -> list[str]:
    """Dictionary entries store attachment-point dummies ([1*], [2*], ...)
    where the residue bonds its neighbors. Expand to concrete readings:
    dummies on a carbonyl carbon become the free carboxyl OH, every other
    dummy becomes H. RDKit sanitizes each variant — an impossible expansion
    simply dies here instead of reaching arbitration."""
    if not smiles or "*" not in smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    variants = []
    for to_oxygen in (True, False):
        em = Chem.RWMol(mol)
        oxygens = []
        # replace carbonyl-adjacent dummies with O when requested; removal
        # order descends so earlier removals do not shift later indices
        for idx in sorted(dummies, reverse=True):
            atom = em.GetAtomWithIdx(idx)
            on_carbonyl = any(
                em.GetAtomWithIdx(b.GetOtherAtomIdx(idx)).GetSymbol() == "C"
                and any(bb.GetBondTypeAsDouble() == 2 and
                        em.GetAtomWithIdx(bb.GetOtherAtomIdx(
                            other.GetIdx())).GetSymbol() == "O"
                        for bb in other.GetBonds())
                for b in atom.GetBonds()
                for other in [em.GetAtomWithIdx(
                    b.GetOtherAtomIdx(idx))])
            if to_oxygen and on_carbonyl:
                atom.SetAtomicNum(8)
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(1)
                atom.SetFormalCharge(0)
                oxygens.append(idx)
            else:
                em.RemoveAtom(idx)
                oxygens_adjusted = [i if i < idx else i - 1
                                    for i in oxygens]
                oxygens[:] = oxygens_adjusted
        try:
            out = em.GetMol()
            Chem.SanitizeMol(out)
        except Exception:
            continue
        variants.append(Chem.MolToSmiles(out))
    deduped = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def cap_fragment(smiles: str) -> str | None:
    """N-cap acyl form. Three acceptable readings: a free carboxyl (its OH
    is stripped), an attachment-dummy acyl (dummies resolved first), or an
    explicit acyl already ending at C(=O). A cap must carry an acyl carbon
    — that is what bonds the head amine; readings without one are rejected
    here, not guessed around. Multi-acid caps (diacid half-life-extension
    moieties) are attachment-ambiguous under exact mass alone: arbitration
    accepts them and a document SMILES reading settles the attachment when
    one exists."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    pairs = _carboxyl_oh_pairs(mol)
    if pairs:
        carboxyl_c, oh = pairs[-1]       # SMILES-order last free acid
        em = Chem.RWMol(mol)
        em.RemoveAtom(oh)
        try:
            out = em.GetMol()
            Chem.SanitizeMol(out)
        except Exception:
            return None
        return Chem.MolToSmiles(out)
    if has_dummy_atoms(smiles):
        for variant in resolve_dummy_atoms(smiles):
            mol2 = Chem.MolFromSmiles(variant)
            if mol2 is None:
                continue
            if not _carboxyl_oh_pairs(mol2) and _has_acyl_carbon(mol2) \
                    and not _has_backbone_amine(mol2):
                return variant
        return None
    # already-acyl readings (document memory / lexicon store the contracted
    # form) pass through unchanged: the contract must be idempotent or
    # verified learning poisons the next resolution. A molecule carrying
    # BOTH a free amine and an acyl (e.g. trimethyllysine) legitimately
    # satisfies both contracts — the caller's role order (residue first,
    # cap second) plus the mass gate arbitrate.
    if _has_acyl_carbon(mol):
        return smiles
    return None


def _has_acyl_carbon(mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and any(
                b.GetBondTypeAsDouble() == 2
                and mol.GetAtomWithIdx(
                    b.GetOtherAtomIdx(atom.GetIdx())).GetSymbol() == "O"
                for b in atom.GetBonds()):
            return True
    return False


def _has_backbone_amine(mol) -> bool:
    amine = next((a for a in mol.GetAtoms() if a.GetSymbol() == "N"), None)
    return amine is not None and not _is_amide_nitrogen(amine)

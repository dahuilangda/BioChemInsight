"""Conservative review flags for fully specified molecular annotations."""

from rdkit import Chem


def review_reasons(molecule):
    reasons = []
    if any(abs(atom.GetFormalCharge()) > 8 for atom in molecule.GetAtoms()):
        reasons.append("extreme_atomic_charge")
    if any(atom.GetNumRadicalElectrons() > 8 for atom in molecule.GetAtoms()):
        reasons.append("extreme_atomic_radicals")
    fragments = Chem.GetMolFrags(molecule)
    if len(fragments) > 8:
        sizes = [
            sum(molecule.GetAtomWithIdx(index).GetAtomicNum() > 1 for index in fragment)
            for fragment in fragments
        ]
        if max(sizes, default=0) < 0.5 * sum(sizes):
            reasons.append("fragmented_annotation_requires_review")
    return reasons

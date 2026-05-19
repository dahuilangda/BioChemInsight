from __future__ import annotations

from rdkit import Chem


def get_atom_render_label(atom: Chem.Atom) -> str:
    """Return a human-visible label for RDKit rendering, if one exists."""
    for prop_name in ("molFileAlias", "atomLabel"):
        try:
            if atom.HasProp(prop_name):
                label = atom.GetProp(prop_name).strip()
                if label:
                    return label
        except Exception:
            continue

    try:
        label = Chem.GetAtomAlias(atom).strip()
        if label:
            return label
    except Exception:
        pass
    return ""


def apply_atom_render_labels(drawer, mol: Chem.Mol) -> None:
    draw_options = drawer.drawOptions()
    for atom in mol.GetAtoms():
        label = get_atom_render_label(atom)
        if label:
            draw_options.atomLabels[atom.GetIdx()] = label

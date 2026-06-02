from __future__ import annotations

from typing import Any

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - RDKit is expected in the runtime image
    Chem = None

from utils.molecule_2d_layout import copy_conformer_with_added_attachment_dummy
from utils.molecule_2d_layout import layout_fragment_on_scaffold_attachment

ALLOWED_ASSEMBLY_ATOMIC_NUMBERS = {
    0,
    1,
    5,
    6,
    7,
    8,
    9,
    14,
    15,
    16,
    17,
    35,
    53,
}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _molblock_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).rstrip("\r\n")


def _load_mol(molblock: str = ""):
    if Chem is None:
        return None
    if molblock:
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            return mol
    return None


def _dummy_atoms(mol) -> list[int]:
    if mol is None:
        return []
    indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() == "*":
            indices.append(atom.GetIdx())
    return indices


def _single_attachment_neighbor(mol, dummy_idx: int) -> tuple[int | None, Chem.BondType | None]:
    atom = mol.GetAtomWithIdx(dummy_idx)
    bonds = list(atom.GetBonds())
    if len(bonds) != 1:
        return None, None
    bond = bonds[0]
    return bond.GetOtherAtomIdx(dummy_idx), bond.GetBondType()


def _remove_atom_desc(mol, atom_idx: int):
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom_idx)
    return rw.GetMol()


def _fragment_count(mol) -> int:
    if mol is None:
        return 0
    return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))


def _unsupported_atom_symbols(mol) -> list[str]:
    if mol is None:
        return []
    symbols = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ASSEMBLY_ATOMIC_NUMBERS:
            symbols.append(atom.GetSymbol())
    return sorted(set(symbols))


def _normalized_fragment_with_visual_attachment(mol, visual_review: dict):
    if mol is None:
        return None, "fragment_mol_parse_failed"
    if _dummy_atoms(mol):
        return mol, ""
    if isinstance(visual_review, dict) and visual_review.get("has_attachment_evidence"):
        return mol, "fragment_attachment_atom_missing_from_molblock"
    return mol, ""


def _combine_single_substituent(scaffold_mol, fragment_mol):
    scaffold_dummies = _dummy_atoms(scaffold_mol)
    fragment_dummies = _dummy_atoms(fragment_mol)
    if len(scaffold_dummies) != 1:
        return None, None, "scaffold must contain exactly one dummy attachment atom", ""
    if len(fragment_dummies) != 1:
        return None, None, "fragment must contain exactly one dummy attachment atom", ""

    scaffold_dummy = scaffold_dummies[0]
    fragment_dummy = fragment_dummies[0]
    scaffold_anchor, scaffold_bond = _single_attachment_neighbor(scaffold_mol, scaffold_dummy)
    fragment_anchor, fragment_bond = _single_attachment_neighbor(fragment_mol, fragment_dummy)
    if scaffold_anchor is None or fragment_anchor is None:
        return None, None, "dummy attachment atom must have exactly one neighbor", ""

    original_scaffold_anchor = scaffold_anchor
    original_fragment_anchor = fragment_anchor

    scaffold_without_dummy = _remove_atom_desc(scaffold_mol, scaffold_dummy)
    scaffold_anchor = scaffold_anchor - 1 if scaffold_anchor > scaffold_dummy else scaffold_anchor

    fragment_without_dummy = _remove_atom_desc(fragment_mol, fragment_dummy)
    fragment_anchor = fragment_anchor - 1 if fragment_anchor > fragment_dummy else fragment_anchor
    unsupported_scaffold_atoms = _unsupported_atom_symbols(scaffold_without_dummy)
    unsupported_fragment_atoms = _unsupported_atom_symbols(fragment_without_dummy)
    if unsupported_scaffold_atoms:
        return None, None, f"scaffold contains unsupported atoms:{','.join(unsupported_scaffold_atoms)}", ""
    if unsupported_fragment_atoms:
        return None, None, f"fragment contains unsupported atoms:{','.join(unsupported_fragment_atoms)}", ""
    if _fragment_count(scaffold_without_dummy) != 1:
        return None, None, "scaffold attachment removal produced disconnected fragments", ""
    if _fragment_count(fragment_without_dummy) != 1:
        return None, None, "fragment attachment removal produced disconnected fragments", ""
    layout_result = layout_fragment_on_scaffold_attachment(
        fragment_mol=fragment_mol,
        fragment_without_dummy=fragment_without_dummy,
        fragment_anchor=original_fragment_anchor,
        adjusted_fragment_anchor=fragment_anchor,
        fragment_dummy=fragment_dummy,
        scaffold_mol=scaffold_mol,
        scaffold_without_dummy=scaffold_without_dummy,
        scaffold_anchor=original_scaffold_anchor,
        adjusted_scaffold_anchor=scaffold_anchor,
        scaffold_dummy=scaffold_dummy,
    )
    fragment_without_dummy = layout_result.fragment_mol
    layout_note = layout_result.note

    scaffold_atoms = scaffold_without_dummy.GetNumAtoms()
    combined = Chem.CombineMols(scaffold_without_dummy, fragment_without_dummy)
    rw = Chem.RWMol(combined)
    bond_type = scaffold_bond or fragment_bond or Chem.BondType.SINGLE
    rw.AddBond(scaffold_anchor, scaffold_atoms + fragment_anchor, bond_type)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, None, f"assembled molecule failed sanitization: {exc}", layout_note
    if _dummy_atoms(mol):
        return None, None, "assembled molecule still contains dummy attachment atoms", layout_note
    if _fragment_count(mol) != 1:
        return None, None, "assembled molecule is disconnected", layout_note
    return (
        Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
        Chem.MolToMolBlock(mol),
        "",
        layout_note,
    )


def build_markush_assembly_candidates(plan: dict, structure_candidates: list[dict]) -> list[dict]:
    candidates_by_ref = {
        _text(candidate.get("ref")): candidate
        for candidate in structure_candidates or []
        if isinstance(candidate, dict) and _text(candidate.get("ref"))
    }
    assembly_candidates = []
    relationships = _as_list((plan or {}).get("relationships"))
    relation_keys: dict[tuple[str, str], int] = {}
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if relationship.get("assembly_status") != "ready" or relationship.get("pose_consistency") != "consistent":
            continue
        compound_id = _text(relationship.get("compound_id"))
        for variable_position in _as_list(relationship.get("variable_positions")):
            key = (compound_id, _text(variable_position))
            if key[0] and key[1]:
                relation_keys[key] = relation_keys.get(key, 0) + 1

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        blocked_reasons = []
        compound_id = _text(relationship.get("compound_id"))
        scaffold_ref = _text(relationship.get("scaffold_ref"))
        fragment_refs = [_text(ref) for ref in _as_list(relationship.get("fragment_refs")) if _text(ref)]
        variable_positions = [_text(pos) for pos in _as_list(relationship.get("variable_positions")) if _text(pos)]
        visual_review = relationship.get("visual_review") if isinstance(relationship.get("visual_review"), dict) else {}
        evidence_key = {
            "compound_id": compound_id or None,
            "variable_positions": variable_positions,
            "scaffold_ref": scaffold_ref or None,
            "fragment_refs": fragment_refs,
        }

        if not scaffold_ref:
            blocked_reasons.append("missing_scaffold_ref")
        if not fragment_refs:
            blocked_reasons.append("missing_fragment_ref")
        if not variable_positions:
            blocked_reasons.append("missing_variable_position")
        if len(variable_positions) > 1 and len(fragment_refs) != len(variable_positions):
            blocked_reasons.append("multi_substituent_mapping_incomplete")
        if len(variable_positions) != len(fragment_refs):
            blocked_reasons.append("variable_fragment_count_mismatch")
        if len(variable_positions) == 1 and relation_keys.get((compound_id, variable_positions[0]), 0) > 1:
            blocked_reasons.append("ambiguous_compound_variable_mapping")
        if len(set(fragment_refs)) != len(fragment_refs):
            blocked_reasons.append("duplicate_fragment_ref_for_distinct_positions")
        if relationship.get("assembly_status") != "ready":
            blocked_reasons.append(f"relationship_not_ready:{relationship.get('assembly_status')}")
        if relationship.get("pose_consistency") != "consistent":
            blocked_reasons.append(f"pose_not_consistent:{relationship.get('pose_consistency')}")
        if visual_review and visual_review.get("assembly_status") != "ready":
            blocked_reasons.append("visual_review_not_ready")

        scaffold = candidates_by_ref.get(scaffold_ref)
        fragments = [candidates_by_ref.get(ref) for ref in fragment_refs]
        if scaffold_ref and not scaffold:
            blocked_reasons.append("scaffold_candidate_not_found")
        if any(fragment is None for fragment in fragments):
            blocked_reasons.append("fragment_candidate_not_found")
        if scaffold and _text(scaffold.get("structure_type")) != "markush":
            blocked_reasons.append("scaffold_candidate_not_markush")
        if any(fragment and _text(fragment.get("structure_type")) == "markush" for fragment in fragments):
            blocked_reasons.append("fragment_ref_points_to_markush_scaffold")
        if any(fragment and _text(fragment.get("structure_type")) == "text_substituent" for fragment in fragments):
            blocked_reasons.append("text_substituent_requires_molnextr_structure_evidence")

        assembled_smiles = ""
        assembled_molblock = ""
        assembly_status = "blocked"
        normalization_notes = []
        if not blocked_reasons and len(fragments) == 1 and scaffold:
            fragment_candidate = fragments[0]
            scaffold_molblock = _molblock_text(scaffold.get("molblock_full")) or _molblock_text(scaffold.get("molblock"))
            fragment_molblock = _molblock_text(fragment_candidate.get("molblock_full")) or _molblock_text(fragment_candidate.get("molblock"))
            if not scaffold_molblock:
                blocked_reasons.append("missing_scaffold_molblock")
            if not fragment_molblock:
                blocked_reasons.append("missing_fragment_molblock")
            scaffold_mol = _load_mol(scaffold_molblock)
            fragment_mol = _load_mol(fragment_molblock)
            if scaffold_mol is None:
                blocked_reasons.append("scaffold_mol_parse_failed")
            if fragment_mol is None:
                blocked_reasons.append("fragment_mol_parse_failed")
            if not blocked_reasons:
                fragment_mol, normalization_note = _normalized_fragment_with_visual_attachment(
                    fragment_mol,
                    fragment_candidate.get("fragment_visual_review") or {},
                )
                if normalization_note == "fragment_attachment_atom_missing_from_molblock":
                    blocked_reasons.append(normalization_note)
                elif normalization_note.startswith("visual_attachment_anchor_not_unique"):
                    blocked_reasons.append(normalization_note)
                elif normalization_note:
                    normalization_notes.append(normalization_note)
            if not blocked_reasons:
                assembled_smiles, assembled_molblock, error, layout_note = _combine_single_substituent(scaffold_mol, fragment_mol)
                if error:
                    blocked_reasons.append(error)
                elif assembled_smiles:
                    assembly_status = "assembled"
                if layout_note:
                    normalization_notes.append(layout_note)
        elif not blocked_reasons and len(fragments) > 1:
            blocked_reasons.append("multi_substituent_rdkit_assembly_not_implemented")

        assembly_candidates.append({
            "record_id": relationship.get("record_id"),
            "compound_id": relationship.get("compound_id"),
            "evidence_key": evidence_key,
            "source_pages": relationship.get("source_pages") or [],
            "scaffold_ref": scaffold_ref or None,
            "fragment_refs": fragment_refs,
            "variable_positions": variable_positions,
            "substituent_assignments": [
                {"variable_position": pos, "fragment_ref": fragment_refs[index] if index < len(fragment_refs) else None}
                for index, pos in enumerate(variable_positions)
            ],
            "assembly_status": assembly_status,
            "assembled_smiles": assembled_smiles,
            "assembled_molblock": assembled_molblock,
            "blocked_reasons": list(dict.fromkeys(blocked_reasons)),
            "normalization_notes": normalization_notes,
            "required_evidence": [
                "red_box_scaffold_ref",
                "molnextr_scaffold_molblock",
                "molnextr_fragment_molblock",
                "dummy_attachment_atoms",
                "variable_position_mapping",
                "visual_pose_consistent",
            ],
            "method": "rdkit_dummy_attachment_single_substituent",
        })
    return assembly_candidates

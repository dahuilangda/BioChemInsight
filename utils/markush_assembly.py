from __future__ import annotations

import math
import re
from typing import Any

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - RDKit is expected in the runtime image
    Chem = None

try:
    import constants as project_constants
except ImportError:  # pragma: no cover
    project_constants = None

from utils.molecule_2d_layout import copy_conformer_with_added_attachment_dummy
from utils.molecule_2d_layout import average_bond_length
from utils.molecule_2d_layout import coord
from utils.molecule_2d_layout import has_conformer
from utils.molecule_2d_layout import layout_fragment_on_scaffold_attachment
from utils.molecule_2d_layout import length
from utils.molecule_2d_layout import refine_assembled_layout
from utils.molecule_2d_layout import sub
from utils.markush_text_substituent import normalize_variable_position

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

MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE = float(
    getattr(project_constants, "MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE", 0.40)
)
MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE = float(
    getattr(project_constants, "MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE", 0.0)
)


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


def drop_disconnected_dummy_atoms(mol):
    """Remove degree-0 (disconnected) dummy atoms; keep real attachment sites."""
    if mol is None:
        return mol
    drop = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetDegree() == 0
    ]
    if not drop:
        return mol
    editable = Chem.RWMol(mol)
    for index in sorted(drop, reverse=True):
        editable.RemoveAtom(index)
    return editable.GetMol()


def _dummy_variable_label(atom) -> str:
    """Read the variable identity encoded by MolNexTR/RDKit molblock fields.

    Isotope/atom-map numbers (MolNexTR emits ``[1*]``-style isotopes) take
    priority; bare ``R``/``R#`` props are RDKit artifacts meaning no identity.
    """
    if atom is None or atom.GetAtomicNum() != 0:
        return ""
    isotope = int(atom.GetIsotope())
    if isotope > 0:
        return normalize_variable_position(f"R{isotope}")
    atom_map = int(atom.GetAtomMapNum())
    if atom_map > 0:
        return normalize_variable_position(f"R{atom_map}")
    for prop_name in ("atomLabel", "dummyLabel", "molFileAlias"):
        if not atom.HasProp(prop_name):
            continue
        raw = _text(atom.GetProp(prop_name))
        isotope_match = re.fullmatch(r"\[?(\d+)\*[^]]*\]?", raw)
        if isotope_match:
            return normalize_variable_position(f"R{int(isotope_match.group(1))}")
        normalized = normalize_variable_position(raw)
        if normalized and normalized != "R":
            return normalized
    return ""


def _dummy_sites_by_variable(mol) -> tuple[dict[str, list[int]], list[int]]:
    sites: dict[str, list[int]] = {}
    unlabeled = []
    for atom_index in _dummy_atoms(mol):
        label = _dummy_variable_label(mol.GetAtomWithIdx(atom_index))
        if label:
            sites.setdefault(label, []).append(atom_index)
        else:
            unlabeled.append(atom_index)
    return sites, unlabeled


def _single_attachment_neighbor(mol, dummy_idx: int) -> tuple[int | None, Chem.BondType | None]:
    atom = mol.GetAtomWithIdx(dummy_idx)
    bonds = list(atom.GetBonds())
    if len(bonds) != 1:
        return None, None
    bond = bonds[0]
    return bond.GetOtherAtomIdx(dummy_idx), bond.GetBondType()


def _anchor_parity_inverted(mol, dummy_idx: int) -> bool:
    """True when removing the dummy bond and appending the graft bond at the
    end of the anchor's bond list is an odd permutation of the neighbor order.

    RDKit chiral tags are order-relative: an odd permutation silently flips
    the configuration the tag denotes, so the caller must invert the tag.
    """
    anchor, _ = _single_attachment_neighbor(mol, dummy_idx)
    if anchor is None:
        return False
    bonds = list(mol.GetAtomWithIdx(anchor).GetBonds())
    for pos, bond in enumerate(bonds):
        if bond.GetOtherAtomIdx(anchor) == dummy_idx:
            return pos != len(bonds) - 1
    return False


def _invert_anchor_chirality(mol, anchor_idx: int, inverted: bool) -> None:
    if not inverted:
        return
    atom = mol.GetAtomWithIdx(anchor_idx)
    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)


def _remove_atom_desc(mol, atom_idx: int):
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom_idx)
    return rw.GetMol()


def _fragment_count(mol) -> int:
    if mol is None:
        return 0
    return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))


def _conformer_snapshot(mol) -> list[tuple[float, float, float]]:
    if not has_conformer(mol):
        return []
    return [coord(mol, atom_idx) for atom_idx in range(mol.GetNumAtoms())]


def _pose_drift_reason(reference_points, mol, label: str, atom_offset: int = 0) -> str:
    if not reference_points or not has_conformer(mol):
        return ""
    bond_length = max(average_bond_length(mol), 1e-6)
    if atom_offset + len(reference_points) > mol.GetNumAtoms():
        return f"{label}_pose_atom_count_changed"
    max_drift = 0.0
    for local_idx, reference in enumerate(reference_points):
        max_drift = max(max_drift, length(sub(coord(mol, atom_offset + local_idx), reference)))
    normalized = max_drift / bond_length
    if normalized > 0.02:
        return f"{label}_pose_drift:{normalized:.4f}"
    return ""


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
    scaffold_parity = _anchor_parity_inverted(scaffold_mol, scaffold_dummy)
    fragment_parity = _anchor_parity_inverted(fragment_mol, fragment_dummy)
    scaffold_pose_before_layout = _conformer_snapshot(scaffold_without_dummy)
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
    scaffold_pose_error = _pose_drift_reason(scaffold_pose_before_layout, scaffold_without_dummy, "scaffold")
    if scaffold_pose_error:
        return None, None, scaffold_pose_error, layout_note

    scaffold_atoms = scaffold_without_dummy.GetNumAtoms()
    combined = Chem.CombineMols(scaffold_without_dummy, fragment_without_dummy)
    rw = Chem.RWMol(combined)
    bond_type = scaffold_bond or fragment_bond or Chem.BondType.SINGLE
    if scaffold_bond and fragment_bond and scaffold_bond != fragment_bond:
        layout_note = (
            f"{layout_note};attachment_bond_order_conflict:scaffold={scaffold_bond},fragment={fragment_bond}"
        ).strip(';')
    rw.AddBond(scaffold_anchor, scaffold_atoms + fragment_anchor, bond_type)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, None, f"assembled molecule failed sanitization: {exc}", layout_note
    _invert_anchor_chirality(mol, scaffold_anchor, scaffold_parity)
    _invert_anchor_chirality(mol, scaffold_atoms + fragment_anchor, fragment_parity)
    scaffold_pose_error = _pose_drift_reason(scaffold_pose_before_layout, mol, "assembled_scaffold")
    if scaffold_pose_error:
        return None, None, scaffold_pose_error, layout_note
    if _dummy_atoms(mol):
        return None, None, "assembled molecule still contains dummy attachment atoms", layout_note
    if _fragment_count(mol) != 1:
        return None, None, "assembled molecule is disconnected", layout_note
    layout_refine_note = refine_assembled_layout(mol)
    combined_note = f"{layout_note};{layout_refine_note}" if layout_refine_note else layout_note
    return (
        Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
        Chem.MolToMolBlock(mol),
        "",
        combined_note,
    )


def _combine_at_site(scaffold_mol, scaffold_dummy: int, fragment_mol):
    fragment_dummies = _dummy_atoms(fragment_mol)
    if len(fragment_dummies) != 1:
        return None, "fragment must contain exactly one dummy attachment atom", ""
    fragment_dummy = fragment_dummies[0]
    scaffold_anchor, scaffold_bond = _single_attachment_neighbor(
        scaffold_mol, scaffold_dummy
    )
    fragment_anchor, fragment_bond = _single_attachment_neighbor(
        fragment_mol, fragment_dummy
    )
    if scaffold_anchor is None or fragment_anchor is None:
        return None, "dummy attachment atom must have exactly one neighbor", ""

    original_scaffold_anchor = scaffold_anchor
    original_fragment_anchor = fragment_anchor
    scaffold_without_dummy = _remove_atom_desc(scaffold_mol, scaffold_dummy)
    scaffold_anchor -= int(scaffold_anchor > scaffold_dummy)
    fragment_without_dummy = _remove_atom_desc(fragment_mol, fragment_dummy)
    fragment_anchor -= int(fragment_anchor > fragment_dummy)
    unsupported_scaffold_atoms = _unsupported_atom_symbols(scaffold_without_dummy)
    unsupported_fragment_atoms = _unsupported_atom_symbols(fragment_without_dummy)
    if unsupported_scaffold_atoms:
        return None, f"scaffold contains unsupported atoms:{','.join(unsupported_scaffold_atoms)}", ""
    if unsupported_fragment_atoms:
        return None, f"fragment contains unsupported atoms:{','.join(unsupported_fragment_atoms)}", ""
    if _fragment_count(scaffold_without_dummy) != 1:
        return None, "scaffold attachment removal produced disconnected fragments", ""
    if _fragment_count(fragment_without_dummy) != 1:
        return None, "fragment attachment removal produced disconnected fragments", ""

    scaffold_parity = _anchor_parity_inverted(scaffold_mol, scaffold_dummy)
    fragment_parity = _anchor_parity_inverted(fragment_mol, fragment_dummy)
    scaffold_pose = _conformer_snapshot(scaffold_without_dummy)
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
    scaffold_atoms = scaffold_without_dummy.GetNumAtoms()
    combined = Chem.CombineMols(scaffold_without_dummy, fragment_without_dummy)
    rw = Chem.RWMol(combined)
    site_note = ''
    if scaffold_bond and fragment_bond and scaffold_bond != fragment_bond:
        site_note = f"attachment_bond_order_conflict:scaffold={scaffold_bond},fragment={fragment_bond}"
    rw.AddBond(
        scaffold_anchor,
        scaffold_atoms + fragment_anchor,
        scaffold_bond or fragment_bond or Chem.BondType.SINGLE,
    )
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, f"assembled molecule failed sanitization: {exc}", layout_result.note
    _invert_anchor_chirality(mol, scaffold_anchor, scaffold_parity)
    _invert_anchor_chirality(mol, scaffold_atoms + fragment_anchor, fragment_parity)
    pose_error = _pose_drift_reason(scaffold_pose, mol, "assembled_scaffold")
    if pose_error:
        return None, pose_error, layout_result.note
    if _fragment_count(mol) != 1:
        return None, "assembled molecule is disconnected", layout_result.note
    refine_note = refine_assembled_layout(mol)
    combined_note = ";".join(part for part in (site_note, layout_result.note, refine_note) if part)
    return mol, "", combined_note


def _combine_labeled_substituents(scaffold_mol, assignments):
    """Assemble all explicitly labeled variables, including repeated sites."""
    if not assignments:
        return "", "", "missing substituent assignments", []
    site_map, unlabeled_sites = _dummy_sites_by_variable(scaffold_mol)
    assignment_labels = [item[0] for item in assignments]
    if unlabeled_sites:
        return "", "", "unlabeled_scaffold_dummy_prevents_deterministic_mapping", []
    missing = sorted(set(assignment_labels) - set(site_map))
    extra = sorted(set(site_map) - set(assignment_labels))
    if missing:
        return "", "", f"scaffold_missing_variable_sites:{','.join(missing)}", []
    if extra:
        return "", "", f"unassigned_scaffold_variable_sites:{','.join(extra)}", []

    fragment_by_label = {label: fragment for label, fragment in assignments}
    for label, fragment in assignments:
        fragment_dummies = _dummy_atoms(fragment)
        if len(fragment_dummies) != 1:
            return "", "", f"fragment_{label}_must_contain_exactly_one_dummy", []
        fragment_label = _dummy_variable_label(
            fragment.GetAtomWithIdx(fragment_dummies[0])
        )
        if fragment_label and fragment_label != label:
            return (
                "",
                "",
                f"fragment_variable_label_mismatch:{label}!={fragment_label}",
                [],
            )

    mol = Chem.Mol(scaffold_mol)
    notes = []
    for label in assignment_labels:
        # Re-scan after every edit because atom indices change when a dummy is removed.
        while True:
            current_sites, unlabeled = _dummy_sites_by_variable(mol)
            if unlabeled:
                return "", "", "unlabeled_scaffold_dummy_prevents_deterministic_mapping", notes
            sites = current_sites.get(label) or []
            if not sites:
                break
            mol, error, note = _combine_at_site(
                mol,
                sites[0],
                Chem.Mol(fragment_by_label[label]),
            )
            if note:
                notes.append(f"{label}:{note}")
            if error:
                return "", "", f"{label}:{error}", notes
    if _dummy_atoms(mol):
        return "", "", "assembled molecule still contains dummy attachment atoms", notes
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return "", "", f"assembled molecule failed sanitization: {exc}", notes
    refine_note = refine_assembled_layout(mol)
    if refine_note:
        notes.append(refine_note)
    return (
        Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
        Chem.MolToMolBlock(mol),
        "",
        notes,
    )
def _candidate_confidence(candidate) -> float | None:
    """Calibrated MolNexTR confidence (E[Tanimoto]) of a structure candidate.

    Returns None when no usable confidence value (missing, non-numeric,
    negative, NaN, or infinite).
    """
    if not isinstance(candidate, dict):
        return None
    raw = candidate.get("molnextr_confidence")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0.0 or math.isnan(value) or math.isinf(value):
        return None
    return value


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
        if compound_id.lower() == 'none':
            compound_id = ''
        for variable_position in _as_list(relationship.get("variable_positions")):
            key = (compound_id, normalize_variable_position(variable_position))
            if key[0] and key[1]:
                relation_keys[key] = relation_keys.get(key, 0) + 1

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        blocked_reasons = []
        compound_id = _text(relationship.get("compound_id"))
        if compound_id.lower() == 'none':
            compound_id = ''
        scaffold_ref = _text(relationship.get("scaffold_ref"))
        fragment_refs = [_text(ref) for ref in _as_list(relationship.get("fragment_refs")) if _text(ref)]
        variable_positions = [
            normalized
            for pos in _as_list(relationship.get("variable_positions"))
            for normalized in [normalize_variable_position(pos)]
            if normalized
        ]
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
        variable_fragment_pairs = list(zip(variable_positions, fragment_refs))
        if len(set(variable_positions)) != len(variable_positions):
            blocked_reasons.append("duplicate_variable_assignment")
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

        # The confidence gate skips markush scaffolds: the confidence head is
        # fragment-trained and unreliable for R-group scaffolds with dummies.
        scaffold_confidence = _candidate_confidence(scaffold) if scaffold and _text(scaffold.get("structure_type")) != "markush" else None
        if (
            scaffold_confidence is not None
            and scaffold_confidence < MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE
        ):
            blocked_reasons.append(
                f"low_confidence_scaffold_backbone:{scaffold_confidence:.3f}"
                f"<{MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE:.2f}"
            )
        fragment_confidences = [_candidate_confidence(fragment) for fragment in fragments]
        for variable_position, fragment_confidence in zip(
            variable_positions, fragment_confidences
        ):
            if (
                fragment_confidence is not None
                and fragment_confidence < MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE
            ):
                blocked_reasons.append(
                    f"low_confidence_fragment_backbone:{variable_position}:"
                    f"{fragment_confidence:.3f}<{MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE:.2f}"
                )

        assembled_smiles = ""
        assembled_molblock = ""
        assembly_status = "blocked"
        normalization_notes = []
        if not blocked_reasons and fragments and scaffold:
            scaffold_molblock = _molblock_text(scaffold.get("molblock_full")) or _molblock_text(scaffold.get("molblock"))
            if not scaffold_molblock:
                blocked_reasons.append("missing_scaffold_molblock")
            scaffold_mol = _load_mol(scaffold_molblock)
            if scaffold_mol is None:
                blocked_reasons.append("scaffold_mol_parse_failed")
            else:
                scaffold_mol = drop_disconnected_dummy_atoms(scaffold_mol)
            parsed_fragments = []
            for variable_position, fragment_candidate in zip(
                variable_positions, fragments
            ):
                fragment_molblock = _molblock_text(
                    fragment_candidate.get("molblock_full")
                ) or _molblock_text(fragment_candidate.get("molblock"))
                if not fragment_molblock:
                    blocked_reasons.append(
                        f"missing_fragment_molblock:{variable_position}"
                    )
                    continue
                fragment_mol = _load_mol(fragment_molblock)
                if fragment_mol is None:
                    blocked_reasons.append(
                        f"fragment_mol_parse_failed:{variable_position}"
                    )
                    continue
                fragment_mol, normalization_note = _normalized_fragment_with_visual_attachment(
                    fragment_mol,
                    fragment_candidate.get("fragment_visual_review") or {},
                )
                if normalization_note == "fragment_attachment_atom_missing_from_molblock":
                    blocked_reasons.append(
                        f"{normalization_note}:{variable_position}"
                    )
                elif normalization_note.startswith("visual_attachment_anchor_not_unique"):
                    blocked_reasons.append(
                        f"{normalization_note}:{variable_position}"
                    )
                elif normalization_note:
                    normalization_notes.append(
                        f"{variable_position}:{normalization_note}"
                    )
                parsed_fragments.append((variable_position, fragment_mol))
            error = ""
            layout_notes = []
            if not blocked_reasons:
                single_substituent = (
                    len(parsed_fragments) == 1 and len(_dummy_atoms(scaffold_mol)) == 1
                )
                if single_substituent:
                    scaffold_dummies = _dummy_atoms(scaffold_mol)
                    scaffold_label = _dummy_variable_label(
                        scaffold_mol.GetAtomWithIdx(scaffold_dummies[0])
                    ) if scaffold_dummies else ""
                    fragment_dummies = _dummy_atoms(parsed_fragments[0][1])
                    fragment_label = _dummy_variable_label(
                        parsed_fragments[0][1].GetAtomWithIdx(fragment_dummies[0])
                    ) if fragment_dummies else ""
                    relationship_label = variable_positions[0]
                    if scaffold_label and scaffold_label != relationship_label:
                        blocked_reasons.append(
                            f"scaffold_variable_label_mismatch:{scaffold_label}!={relationship_label}"
                        )
                    if fragment_label and fragment_label != relationship_label:
                        blocked_reasons.append(
                            f"fragment_variable_label_mismatch:{fragment_label}!={relationship_label}"
                        )
                if single_substituent:
                    # Skip when already blocked: the multi path would add noisy
                    # missing-site reasons on top of the real mismatch.
                    if not blocked_reasons:
                        assembled_smiles, assembled_molblock, error, layout_note = _combine_single_substituent(
                            scaffold_mol, parsed_fragments[0][1]
                        )
                        layout_notes = [layout_note] if layout_note else []
                else:
                    (
                        assembled_smiles,
                        assembled_molblock,
                        error,
                        layout_notes,
                    ) = _combine_labeled_substituents(
                        scaffold_mol,
                        parsed_fragments,
                    )
                if error:
                    blocked_reasons.append(error)
                elif assembled_smiles:
                    assembly_status = "assembled"
                normalization_notes.extend(note for note in layout_notes if note)

        assembly_candidates.append({
            "record_id": relationship.get("record_id"),
            "compound_id": relationship.get("compound_id"),
            "evidence_key": evidence_key,
            "source_pages": relationship.get("source_pages") or [],
            "scaffold_ref": scaffold_ref or None,
            "fragment_refs": fragment_refs,
            "variable_positions": variable_positions,
            "scaffold_confidence": scaffold_confidence,
            "fragment_confidences": fragment_confidences,
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
                "scaffold_pose_preserved",
            ],
            "method": "rdkit_label_mapped_dummy_attachment",
        })
    return assembly_candidates

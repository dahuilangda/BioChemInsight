"""Typed structure labels for SMILES, E-SMILES, and CXSMILES."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache

from rdkit import Chem, rdBase

_RECORD = re.compile(r"<([arcd])>(<c>)?([0-9]+(?:,[0-9]+)*):(.*?)</\1>")
_REPEAT = re.compile(r"\|Sg:([A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)\|")


def chemical_cx_flags():
    flags = Chem.CXSmilesFields.CX_ALL
    for excluded in (
        Chem.CXSmilesFields.CX_COORDS,
        Chem.CXSmilesFields.CX_ATOM_LABELS,
        Chem.CXSmilesFields.CX_ATOM_PROPS,
        Chem.CXSmilesFields.CX_BOND_CFG,
    ):
        flags &= ~excluded
    return flags


def _canonical_structure(molecule):
    molecule = Chem.Mol(molecule)
    Chem.MolToSmiles(molecule)
    order = ast.literal_eval(molecule.GetProp("_smilesAtomOutputOrder"))
    molecule = Chem.RenumberAtoms(molecule, order)
    for index in range(len(Chem.GetMolSubstanceGroups(molecule))):
        group = Chem.GetMolSubstanceGroupWithIdx(molecule, index)
        group.SetAtoms(sorted(group.GetAtoms()))
        group.SetBonds(sorted(group.GetBonds()))
        group.SetParentAtoms(sorted(group.GetParentAtoms()))
    parameters = Chem.SmilesWriteParams()
    parameters.canonical = False
    return Chem.MolToCXSmiles(molecule, parameters, flags=chemical_cx_flags())


def _native_model_text(molecule, colored, variations=(), *, explicit_bonds=True):
    """Serialize annotation-sensitive stereo before restoring the original isotopes."""
    colored = Chem.Mol(colored)
    parameters = Chem.SmilesWriteParams()
    parameters.canonical = False
    parameters.allBondsExplicit = explicit_bonds
    parameters.allHsExplicit = True
    with rdBase.BlockLogs():
        traversal = Chem.MolToSmiles(colored, parameters)
        traversal_order = ast.literal_eval(colored.GetProp("_smilesAtomOutputOrder"))
        order = iter(traversal_order)
        encoded = Chem.MolToCXSmiles(
            colored,
            parameters,
            flags=Chem.CXSmilesFields.CX_ALL_BUT_COORDS
            & ~Chem.CXSmilesFields.CX_BOND_CFG,
        )
    base, separator, extension = encoded.partition(" |")
    if base != traversal:
        raise StructureError("cxsmiles_atom_traversal_changed")

    def restore(match):
        atom = colored.GetAtomWithIdx(next(order))
        original = molecule.GetAtomWithIdx(atom.GetIntProp("_molsight_origin"))
        body = re.sub(
            r"^\[\d+",
            "[" + (str(original.GetIsotope()) if original.GetIsotope() else ""),
            match[0],
        )
        if original.GetAtomMapNum():
            body = body[:-1] + f":{original.GetAtomMapNum()}]"
        if body == "[*]" and original.HasQuery() and original.GetSmarts() == "[!#1]":
            return "*"
        return body

    base = re.sub(r"\[[^\]]+\]", restore, base)
    if variations:
        reverse = {index: position for position, index in enumerate(traversal_order)}
        records = [
            f"m:{reverse[anchor]}:"
            + ".".join(str(reverse[index]) for index in candidates)
            for anchor, candidates in variations
        ]
        extension = extension.removesuffix("|")
        extension += ("," if extension else "") + ",".join(records) + "|"
        separator = " |"
    return base + separator + extension


def _parse_variations(text, molecule):
    extension = text.partition(" |")[2].removesuffix("|")
    records = []
    for match in re.finditer(r"(?:^|,)m:([0-9:.]+(?:,[0-9]+:[0-9.]+)*)", extension):
        for record in match[1].split(","):
            anchor, candidates = record.split(":")
            anchor = int(anchor)
            candidates = tuple(sorted(set(map(int, candidates.split(".")))))
            if anchor >= molecule.GetNumAtoms() or any(
                index >= molecule.GetNumAtoms() for index in candidates
            ):
                raise StructureError("variation_index_out_of_range")
            if molecule.GetAtomWithIdx(anchor).GetDegree() != 1 or not candidates:
                raise StructureError("invalid_variation_anchor")
            records.append((anchor, candidates))
    return tuple(sorted(set(records)))


def _retain_declared_group_crossings(molecule, text):
    """Keep implicit polymer crossings implicit during serialization."""
    extension = text.partition(" |")[2].removesuffix("|")
    fields = list(re.finditer(r"(?:^|,)([A-Za-z&][A-Za-z0-9_&]*):", extension))
    records = {}
    group_index = 0
    for index, match in enumerate(fields):
        end = fields[index + 1].start() if index + 1 < len(fields) else len(extension)
        record = extension[match.end() : end].split(":")
        if match[1] == "Sg":
            records[group_index] = any(
                record[position] for position in (4, 5) if position < len(record)
            )
        if match[1] in ("Sg", "SgD"):
            group_index += 1
    for index in range(len(Chem.GetMolSubstanceGroups(molecule))):
        group = Chem.GetMolSubstanceGroupWithIdx(molecule, index)
        if (
            group.HasProp("_cxsmilesindex")
            and records.get(group.GetIntProp("_cxsmilesindex")) is False
        ):
            for name in ("XBHEAD", "XBCORR"):
                if group.HasProp(name):
                    group.ClearProp(name)


class StructureError(ValueError):
    pass


@dataclass(frozen=True)
class Annotation:
    kind: str
    indices: tuple[int, ...]
    label: str
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class StructureLabel:
    base: str
    annotations: tuple[Annotation, ...]
    repeat: str | None
    _molecule: Chem.Mol = field(repr=False, compare=False)
    _colored: Chem.Mol = field(repr=False, compare=False)
    colors: tuple[str, ...]
    canonical: str
    ring_records: tuple[tuple[tuple[int, ...], str], ...]
    semantic_labels: tuple[tuple[str, str], ...]
    native_cxsmiles: str | None = None
    variation_records: tuple[tuple[int, tuple[int, ...]], ...] = ()
    serialization_error: str | None = field(default=None, repr=False)

    @property
    def molecule(self):
        return Chem.Mol(self._molecule)

    def to_esmiles(self, *, version=1):
        if self.native_cxsmiles is not None:
            raise StructureError("native_cxsmiles_requires_native_serialization")
        if version not in (1, 2):
            raise StructureError("unsupported_esmiles_version")
        records = []
        for annotation in self.annotations:
            kind = "a" if version == 1 and annotation.kind == "d" else annotation.kind
            indices = ",".join(map(str, annotation.indices))
            if kind == "rc":
                records.append(f"<r><c>{indices}:{annotation.label}</r>")
            else:
                if version == 2 and kind == "r" and annotation.label == "<dum>":
                    raise StructureError("legacy_ring_attachment_requires_esmiles1")
                records.append(f"<{kind}>{indices}:{annotation.label}</{kind}>")
        suffix = "".join(records)
        if self.repeat is not None:
            suffix += f"|Sg:{self.repeat}|"
        return self.base + "<sep>" + suffix

    @property
    def model_text(self):
        if self.serialization_error is not None:
            raise StructureError(self.serialization_error)
        if self.native_cxsmiles is not None:
            return self.native_cxsmiles
        return self.to_esmiles() if self.annotations or self.repeat else self.base

    def equivalent(self, other):
        if not isinstance(other, StructureLabel):
            return False
        if (self.colors, self.canonical, self.repeat) != (
            other.colors,
            other.canonical,
            other.repeat,
        ):
            return False
        if len(self.ring_records) != len(other.ring_records):
            return False
        if len(self.variation_records) != len(other.variation_records):
            return False
        if not self.ring_records and not self.variation_records:
            return True
        expected = Counter(other.ring_records)
        expected_variations = Counter(other.variation_records)

        def check(target, match):
            if any(
                self._colored.GetAtomWithIdx(index).GetIsotope()
                != target.GetAtomWithIdx(atom).GetIsotope()
                for index, atom in enumerate(match)
            ):
                return False
            return (
                Counter(
                    (tuple(sorted(match[atom] for atom in atoms)), label)
                    for atoms, label in self.ring_records
                )
                == expected
            ) and Counter(
                (match[anchor], tuple(sorted(match[index] for index in candidates)))
                for anchor, candidates in self.variation_records
            ) == expected_variations

        parameters = Chem.SubstructMatchParameters()
        parameters.useChirality = True
        parameters.useEnhancedStereo = True
        parameters.useQueryQueryMatches = True
        parameters.uniquify = False
        parameters.maxMatches = 1
        parameters.setExtraFinalCheck(check)
        return bool(other._colored.GetSubstructMatch(self._colored, parameters))

    def without_stereo(self, *, atoms=False, bonds=True):
        molecule = Chem.Mol(self._colored)
        if atoms:
            Chem.RemoveStereochemistry(molecule)
        elif bonds:
            for bond in molecule.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
        canonical = _canonical_structure(molecule)
        return replace(self, _colored=molecule, canonical=canonical)


def _label(text):
    if not text or any(character.isspace() for character in text):
        raise StructureError("invalid_annotation_label")
    if text == "<dum>":
        return text
    if text.startswith("<id>[") and text.endswith("]"):
        note = text[5:-1]
        if note and not any(character in note for character in "[]<>?"):
            return text
        raise StructureError("invalid_special_annotation")
    if "<" in text or ">" in text or text.count("[") != text.count("]"):
        raise StructureError("invalid_annotation_label")
    match = re.fullmatch(r"R([0-9]+)", text)
    return f"R[{match[1]}]" if match else text


@lru_cache(maxsize=8192)
def parse_structure(text: str) -> StructureLabel:
    document = _parse_structure(text)
    if document.native_cxsmiles is None:
        return document
    for explicit_bonds in (True, False):
        try:
            target = _native_model_text(
                document._molecule,
                document._colored,
                document.variation_records,
                explicit_bonds=explicit_bonds,
            )
            if document.equivalent(_parse_structure(target)):
                return replace(document, native_cxsmiles=target)
        except (StructureError, RuntimeError, ValueError):
            continue
    return replace(document, serialization_error="serialization_changed_structure")


@lru_cache(maxsize=8192)
def _parse_structure(text: str) -> StructureLabel:
    if not isinstance(text, str) or not text or text.count("<sep>") > 1:
        raise StructureError("invalid_structure_label")
    base, _, extension = text.partition("<sep>")
    native_cxsmiles = " |" in base
    if native_cxsmiles:
        if extension or "<sep>" in text:
            raise StructureError("mixed_structure_extension_formats")
        base = text.partition(" |")[0]
    if not base or any(character.isspace() for character in base):
        raise StructureError("invalid_base_smiles")
    parameters = Chem.SmilesParserParams()
    parameters.sanitize = False
    parameters.removeHs = False
    parameters.parseName = False
    parameters.allowCXSMILES = native_cxsmiles
    parameters.strictCXSMILES = True
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromSmiles(text if native_cxsmiles else base, parameters)
            if molecule is None or not molecule.GetNumAtoms():
                raise StructureError("invalid_base_smiles")
            Chem.SanitizeMol(
                molecule,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY,
            )
            directions = {Chem.BondDir.ENDUPRIGHT: 1, Chem.BondDir.ENDDOWNRIGHT: -1}
            for bond in molecule.GetBonds():
                if bond.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                for endpoint in (bond.GetBeginAtom(), bond.GetEndAtom()):
                    signs = [
                        directions[adjacent.GetBondDir()]
                        * (1 if adjacent.GetBeginAtomIdx() == endpoint.GetIdx() else -1)
                        for adjacent in endpoint.GetBonds()
                        if adjacent.GetBondDir() in directions
                    ]
                    if len(signs) > 1 and len(set(signs)) == 1:
                        raise StructureError("conflicting_bond_stereochemistry")
            raw_variations = (
                _parse_variations(text, molecule) if native_cxsmiles else ()
            )
            for atom in molecule.GetAtoms():
                atom.SetIntProp("_molsight_cx_origin", atom.GetIdx())
            molecule = Chem.RemoveHs(molecule, sanitize=False)
            origin_map = {
                atom.GetIntProp("_molsight_cx_origin"): atom.GetIdx()
                for atom in molecule.GetAtoms()
            }
            try:
                variations = tuple(
                    (
                        origin_map[anchor],
                        tuple(sorted(origin_map[index] for index in candidates)),
                    )
                    for anchor, candidates in raw_variations
                )
            except KeyError as error:
                raise StructureError(
                    "variation_hydrogen_requires_explicit_mapping"
                ) from error
            rings = tuple(tuple(ring) for ring in Chem.GetSymmSSSR(molecule))
        except StructureError:
            raise
        except (RuntimeError, ValueError) as error:
            raise StructureError("invalid_base_smiles") from error
    annotations, repeat = [], None
    if native_cxsmiles:
        _retain_declared_group_crossings(molecule, text)
        for atom in molecule.GetAtoms():
            properties = atom.GetPropsAsDict(
                includePrivate=False, includeComputed=False
            )
            if set(properties) - {"dummyLabel", "atomLabel", "molAtomMapNumber"}:
                raise StructureError("unsupported_cxsmiles_atom_property")
            labels = {
                str(properties[name])
                for name in ("dummyLabel", "atomLabel")
                if properties.get(name) and properties[name] != "*"
            }
            if len(labels) > 1:
                raise StructureError("conflicting_cxsmiles_labels")
            if labels:
                label = labels.pop()
                if any(ord(character) < 32 for character in label):
                    raise StructureError("invalid_cxsmiles_label")
                match = re.fullmatch(r"R([0-9]+)", label)
                label = f"R[{match[1]}]" if match else label
                annotations.append(
                    Annotation("a", (atom.GetIdx(),), label, (atom.GetIdx(),))
                )
    position = 0
    while position < len(extension):
        match = _RECORD.match(extension, position)
        if match is None:
            marker = _REPEAT.fullmatch(extension[position:])
            if marker and repeat is None:
                repeat = marker[1]
                position = len(extension)
                continue
            raise StructureError(f"unsupported_or_invalid_extension_at_{position}")
        kind, circle, raw_indices, raw_label = match.groups()
        if circle:
            if kind != "r":
                raise StructureError("invalid_abstract_ring_reference")
            kind = "rc"
        if not raw_label and kind == "a":
            raw_label = "<dum>"
        indices = tuple(sorted(set(map(int, raw_indices.split(",")))))
        label = _label(raw_label)
        if kind == "r":
            if any(index >= len(rings) for index in indices):
                raise StructureError("ring_index_out_of_range")
            if "<" in label and label != "<dum>":
                raise StructureError("invalid_ring_label")
            atoms = tuple(sorted({atom for index in indices for atom in rings[index]}))
        else:
            if len(indices) != 1 or indices[0] >= molecule.GetNumAtoms():
                raise StructureError("atom_index_out_of_range")
            atoms = indices
            if kind == "d" and label != "<dum>":
                raise StructureError("invalid_attachment_label")
            if label == "<dum>":
                if kind not in ("a", "d", "rc"):
                    raise StructureError("invalid_attachment_record")
                if kind != "rc":
                    kind = "d"
            if (
                kind in ("c", "d", "rc")
                and molecule.GetAtomWithIdx(atoms[0]).GetAtomicNum() != 0
            ):
                raise StructureError("placeholder_index_requires_dummy")
            if label.startswith("<id>") and kind != "a":
                raise StructureError("invalid_special_annotation")
        annotations.append(Annotation(kind, indices, label, atoms))
        position = match.end()
    abstract_atoms = {item.atoms[0] for item in annotations if item.kind == "c"}
    if any(
        item.kind == "rc" and item.atoms[0] not in abstract_atoms
        for item in annotations
    ):
        raise StructureError("undefined_abstract_ring")
    annotations.sort(key=lambda item: (item.kind, item.indices, item.label))
    atom_labels = [[] for _ in molecule.GetAtoms()]
    ring_labels = [[] for _ in molecule.GetAtoms()]
    group_labels = [[] for _ in molecule.GetAtoms()]
    variation_labels = [[] for _ in molecule.GetAtoms()]
    for group in Chem.GetMolSubstanceGroups(molecule):
        properties = group.GetPropsAsDict()
        description = tuple(
            sorted(
                (key, str(value))
                for key, value in properties.items()
                if key in {"TYPE", "LABEL", "CONNECT", "MULT"}
            )
        )
        for index in group.GetAtoms():
            group_labels[index].append(description)
    ring_records = []
    for annotation in annotations:
        if annotation.kind == "r":
            ring_records.append((annotation.atoms, annotation.label))
            for atom in annotation.atoms:
                ring_labels[atom].append(annotation.label)
        else:
            atom_labels[annotation.atoms[0]].append((annotation.kind, annotation.label))
    for anchor, candidates in variations:
        context = tuple(
            sorted(
                tuple(atom_labels[neighbor.GetIdx()])
                for neighbor in molecule.GetAtomWithIdx(anchor).GetNeighbors()
            )
        )
        variation_labels[anchor].append(("anchor", context))
        for index in candidates:
            variation_labels[index].append(("candidate", context))
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if atom.GetAtomicNum() == 0 and atom.GetIsotope():
            label = ("a", f"R[{atom.GetIsotope()}]")
            declared = [item for item in atom_labels[index] if item[0] == "a"]
            if declared and label not in declared and not native_cxsmiles:
                raise StructureError("conflicting_numeric_rgroup")
            if label not in atom_labels[index] and not (native_cxsmiles and declared):
                atom_labels[index].append(label)
        if (
            atom.GetAtomicNum() == 0
            and atom.GetAtomMapNum()
            and not any(kind == "d" for kind, _ in atom_labels[index])
        ):
            atom_labels[index].append(("d", "<dum>"))
        atom.SetIntProp("_molsight_origin", index)
    colored = Chem.Mol(molecule)
    removal = Chem.RemoveHsParameters()
    removal.removeMapped = False
    protected_hydrogens = {}
    for atom in colored.GetAtoms():
        if (
            atom.GetAtomicNum() == 1
            and atom_labels[atom.GetIdx()]
            and not atom.GetAtomMapNum()
        ):
            protected_hydrogens[atom.GetIdx()] = 0
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    with rdBase.BlockLogs():
        colored = Chem.RemoveHs(colored, removal, sanitize=False)
    original_to_new = {}
    descriptions = []
    for atom in colored.GetAtoms():
        origin = atom.GetIntProp("_molsight_origin")
        original_to_new[origin] = atom.GetIdx()
        if origin in protected_hydrogens:
            atom.SetAtomMapNum(0)
        isotope = 0 if atom.GetAtomicNum() == 0 else atom.GetIsotope()
        descriptions.append(
            repr(
                (
                    atom.GetAtomicNum(),
                    isotope,
                    atom.GetAtomMapNum(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    atom.GetTotalNumHs(),
                    atom.GetIsAromatic(),
                    tuple(sorted(atom_labels[origin])),
                    tuple(sorted(ring_labels[origin])),
                    tuple(sorted(group_labels[origin])),
                    tuple(sorted(variation_labels[origin])),
                    atom.DescribeQuery() if atom.HasQuery() else None,
                )
            )
        )
    colors = tuple(sorted(set(descriptions)))
    palette = {description: 1000 + index for index, description in enumerate(colors)}
    if len(palette) >= 64536:
        raise StructureError("structure_label_too_large")
    for atom, description in zip(colored.GetAtoms(), descriptions, strict=True):
        atom.SetIsotope(palette[description])
        atom.SetAtomMapNum(0)
    with rdBase.BlockLogs():
        Chem.AssignStereochemistry(colored, cleanIt=True, force=True)
    canonical = _canonical_structure(colored)
    semantic_rings = tuple(
        sorted(
            (tuple(sorted(original_to_new[atom] for atom in atoms)), label)
            for atoms, label in ring_records
        )
    )
    semantic_variations = tuple(
        sorted(
            (
                original_to_new[anchor],
                tuple(sorted(original_to_new[index] for index in candidates)),
            )
            for anchor, candidates in variations
        )
    )
    semantic_labels = [
        (
            kind,
            f"{label}:{molecule.GetAtomWithIdx(index).GetAtomMapNum()}"
            if kind == "d"
            else label,
        )
        for index, records in enumerate(atom_labels)
        for kind, label in records
    ] + [("r", label) for _, label in ring_records]
    if repeat is not None:
        semantic_labels.append(("repeat", repeat))
    native_text = None
    if native_cxsmiles:
        if not text.endswith("|"):
            raise StructureError("invalid_cxsmiles_extension")
        native_text = text
        semantic_labels.extend(
            ("variation", str(len(candidates))) for _, candidates in variations
        )
        for group in Chem.GetMolSubstanceGroups(molecule):
            properties = group.GetPropsAsDict()
            semantic_labels.append(
                (
                    "sgroup",
                    repr(
                        tuple(
                            sorted(
                                (key, str(value))
                                for key, value in properties.items()
                                if key in {"TYPE", "LABEL", "CONNECT", "MULT"}
                            )
                        )
                    ),
                )
            )
    return StructureLabel(
        base,
        tuple(annotations),
        repeat,
        molecule,
        colored,
        colors,
        canonical,
        semantic_rings,
        tuple(sorted(semantic_labels)),
        native_text,
        semantic_variations,
    )


def structure_equal(first: str, second: str) -> bool:
    return parse_structure(first).equivalent(parse_structure(second))

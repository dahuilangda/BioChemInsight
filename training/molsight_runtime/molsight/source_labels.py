"""Conversion of source annotations without structural repair."""

import re

from rdkit import Chem, rdBase
from SmilesPE.pretokenizer import atomwise_tokenizer

from .structure import StructureError, parse_structure


def markushgrapher_to_cxsmiles(text):
    if not isinstance(text, str) or not text:
        raise StructureError("missing_markushgrapher_label")
    body, separator, extension = text.partition("|")
    chunks, labels, position, length = [], [], 0, 0
    for match in re.finditer(r"<r>(.*?)</r>", body):
        prefix = body[position : match.start()]
        chunks.extend((prefix, "*"))
        length += len(prefix)
        label = match[1]
        if not label or "<" in label or ">" in label:
            raise StructureError("invalid_markushgrapher_variable")
        labels.append((length, label))
        length += 1
        position = match.end()
    chunks.append(body[position:])
    base = "".join(chunks)
    if "<" in base or ">" in base:
        raise StructureError("unsupported_markushgrapher_markup")
    tokens = atomwise_tokenizer(base)
    if "".join(tokens) != base:
        raise StructureError("unsupported_markushgrapher_smiles")
    offsets, offset, index = {}, 0, 0
    for token in tokens:
        if token.isalpha() or token.startswith("[") or token == "*":
            offsets[offset] = index
            index += 1
        offset += len(token)
    properties = []
    for start, label in labels:
        escaped = "".join(
            f"&#{ord(character)};" if character in ":,.;$|<>&'\"" else character
            for character in label
        )
        properties.append(f"{offsets[start]}.dummyLabel.{escaped}")
    extension = extension.removesuffix("|") if separator else ""
    if properties:
        extension = (
            "atomProp:" + ":".join(properties) + ("," + extension if extension else "")
        )
    return base + (" |" + extension + "|" if extension else "")


def parse_source_label(text, *, notation="structure"):
    if notation == "markushgrapher":
        text = markushgrapher_to_cxsmiles(text)
    elif notation != "structure":
        raise StructureError("unsupported_source_notation")
    return parse_structure(text)


def cxsmiles_to_esmiles(text):
    if not isinstance(text, str) or not text:
        raise StructureError("missing_cxsmiles")
    base, separator, extension = text.partition(" |")
    if not separator:
        return parse_structure(text).model_text
    if not extension.endswith("|"):
        raise StructureError("invalid_cxsmiles_extension")
    parameters = Chem.SmilesParserParams()
    parameters.sanitize = False
    parameters.removeHs = True
    parameters.parseName = False
    parameters.strictCXSMILES = True
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(text, parameters)
    if molecule is None:
        raise StructureError("invalid_cxsmiles")
    if Chem.GetMolSubstanceGroups(molecule):
        raise StructureError("cxsmiles_substance_groups_require_typed_contract")
    if molecule.GetStereoGroups():
        raise StructureError("cxsmiles_enhanced_stereo_requires_typed_contract")
    labels = []
    for atom in molecule.GetAtoms():
        properties = atom.GetPropsAsDict(includePrivate=False, includeComputed=False)
        if set(properties) - {"dummyLabel", "atomLabel"}:
            raise StructureError("unsupported_cxsmiles_atom_property")
        names = {str(value) for value in properties.values() if value}
        if len(names) > 1:
            raise StructureError("conflicting_cxsmiles_labels")
        if names:
            labels.append(f"<a>{atom.GetIdx()}:{names.pop()}</a>")
    flags = Chem.CXSmilesFields.CX_ALL
    for excluded in (
        Chem.CXSmilesFields.CX_COORDS,
        Chem.CXSmilesFields.CX_ATOM_LABELS,
        Chem.CXSmilesFields.CX_ATOM_PROPS,
    ):
        flags &= ~excluded
    with rdBase.BlockLogs():
        try:
            remaining = Chem.MolToCXSmiles(
                molecule, Chem.SmilesWriteParams(), flags=flags
            )
        except (RuntimeError, ValueError) as error:
            raise StructureError("invalid_cxsmiles_chemistry") from error
    if " |" in remaining:
        raise StructureError("unsupported_cxsmiles_chemical_extension")
    return parse_structure(base + "<sep>" + "".join(labels)).model_text

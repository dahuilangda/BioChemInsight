"""Resolve defined single-port abbreviations without changing literal labels."""

import ast
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType

from rdkit import Chem, rdBase

from .structure import Annotation, StructureError, StructureLabel, parse_structure


REGISTRY_VERSION = "single-port-1"
_GROUPS = (
    (("Me",), "*C"),
    (("Et", "C2H5"), "*CC"),
    (("nPr", "n-Pr", "Pr"), "*CCC"),
    (("iPr", "i-Pr"), "*C(C)C"),
    (("nBu", "n-Bu", "Bu"), "*CCCC"),
    (("iBu", "i-Bu"), "*CC(C)C"),
    (("sBu", "s-Bu"), "*C(C)CC"),
    (("tBu", "t-Bu"), "*C(C)(C)C"),
    (("Ph",), "*c1ccccc1"),
    (("Bn",), "*Cc1ccccc1"),
    (("Ac",), "*C(C)=O"),
    (("Bz",), "*C(=O)c1ccccc1"),
    (("Boc",), "*C(=O)OC(C)(C)C"),
    (("Cbz",), "*C(=O)OCc1ccccc1"),
    (("Fmoc",), "*C(=O)OCC1c2ccccc2-c2ccccc21"),
    (("NHBoc",), "*NC(=O)OC(C)(C)C"),
    (("NHAc",), "*NC(C)=O"),
    (("OMe", "MeO", "OCH3", "H3CO"), "*OC"),
    (("OEt", "EtO"), "*OCC"),
    (("OiBu",), "*OCC(C)C"),
    (("OtBu",), "*OC(C)(C)C"),
    (("SMe", "MeS"), "*SC"),
    (("OAc",), "*OC(C)=O"),
    (("OBz",), "*OC(=O)c1ccccc1"),
    (("OBn",), "*OCc1ccccc1"),
    (("CF3", "F3C"), "*C(F)(F)F"),
    (("OCF3", "F3CO"), "*OC(F)(F)F"),
    (("NCF3", "F3CN"), "*NC(F)(F)F"),
    (("CCl3",), "*C(Cl)(Cl)Cl"),
    (("CN",), "*C#N"),
    (("NO2", "O2N"), "*[N+](=O)[O-]"),
    (("CHO", "OHC"), "*C=O"),
    (("CO2H", "COOH", "HO2C"), "*C(=O)O"),
    (("CO2Et", "COOEt"), "*C(=O)OCC"),
    (("CO2Me", "COOMe"), "*C(=O)OC"),
    (("Ms",), "*S(=O)(=O)C"),
    (("OMs",), "*OS(=O)(=O)C"),
    (("Ts",), "*S(=O)(=O)c1ccc(C)cc1"),
    (("Tf",), "*S(=O)(=O)C(F)(F)F"),
    (("OTf",), "*OS(=O)(=O)C(F)(F)F"),
    (("SO3H",), "*S(=O)(=O)O"),
    (("TMS",), "*[Si](C)(C)C"),
    (("TBS", "TBDMS"), "*[Si](C)(C)C(C)(C)C"),
)
REGISTRY = MappingProxyType(
    {label: smiles for labels, smiles in _GROUPS for label in labels}
)
REGISTRY_SHA256 = hashlib.sha256(
    json.dumps(dict(REGISTRY), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


@dataclass(frozen=True)
class AbbreviationResolution:
    document: StructureLabel
    replacements: tuple[tuple[int, str], ...]
    unresolved: tuple[tuple[int, str, str], ...]


@lru_cache(maxsize=None)
def _fragment(label):
    molecule = Chem.MolFromSmiles(REGISTRY[label])
    ports = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(ports) != 1 or ports[0].GetDegree() != 1:
        raise RuntimeError(f"Invalid abbreviation port: {label}")
    return molecule, ports[0].GetIdx(), ports[0].GetNeighbors()[0].GetIdx()


def _replace_atom(molecule, index, label):
    fragment, port, root = _fragment(label)
    mapping = {root: index}
    molecule.ReplaceAtom(index, fragment.GetAtomWithIdx(root))
    for atom in fragment.GetAtoms():
        if atom.GetIdx() not in (port, root):
            mapping[atom.GetIdx()] = molecule.AddAtom(atom)
    for bond in fragment.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if port not in (first, second):
            molecule.AddBond(mapping[first], mapping[second], bond.GetBondType())
    for group_index in range(len(Chem.GetMolSubstanceGroups(molecule))):
        group = Chem.GetMolSubstanceGroupWithIdx(molecule, group_index)
        if index in group.GetAtoms():
            group.SetAtoms(
                [
                    *group.GetAtoms(),
                    *(value for value in mapping.values() if value != index),
                ]
            )


def _serialize(document, molecule, annotations):
    parameters = Chem.SmilesWriteParams()
    parameters.canonical = False
    with rdBase.BlockLogs():
        base = Chem.MolToSmiles(molecule, parameters)
    order = ast.literal_eval(molecule.GetProp("_smilesAtomOutputOrder"))
    reverse = {index: position for position, index in enumerate(order)}
    if document.native_cxsmiles is not None:
        text = Chem.MolToCXSmiles(
            molecule,
            parameters,
            flags=Chem.CXSmilesFields.CX_ALL_BUT_COORDS
            & ~Chem.CXSmilesFields.CX_BOND_CFG,
        )
        if document.variation_records:
            base, separator, extension = text.partition(" |")
            records = [
                f"m:{reverse[anchor]}:" + ".".join(str(reverse[i]) for i in candidates)
                for anchor, candidates in document.variation_records
            ]
            text = (
                base
                + " |"
                + ((extension.removesuffix("|") + ",") if separator else "")
                + ",".join(records)
                + "|"
            )
        return text
    parsed = parse_structure(base)
    rings = {
        frozenset(ring): index
        for index, ring in enumerate(Chem.GetSymmSSSR(parsed.molecule))
    }
    records = []
    original_rings = tuple(Chem.GetSymmSSSR(document.molecule))
    for annotation in annotations:
        if annotation.kind == "r":
            indices = tuple(
                sorted(
                    rings[frozenset(reverse[atom] for atom in original_rings[index])]
                    for index in annotation.indices
                )
            )
        else:
            indices = tuple(reverse[index] for index in annotation.indices)
        records.append(Annotation(annotation.kind, indices, annotation.label, ()))
    return (
        replace(parsed, annotations=tuple(records), repeat=document.repeat).to_esmiles()
        if records or document.repeat
        else base
    )


def resolve_abbreviations(document, *, reserved_labels=()):
    if isinstance(document, str):
        document = parse_structure(document)
    molecule = Chem.RWMol(document.molecule)
    multiplicity = Counter(
        index for record in document.annotations for index in record.atoms
    )
    variable_sites = {
        index
        for anchor, candidates in document.variation_records
        for index in (anchor, *candidates)
    }
    reserved = set(reserved_labels)
    replacements, unresolved = [], []
    for record in document.annotations:
        if (
            record.kind != "a"
            or record.label not in REGISTRY
            or record.label in reserved
        ):
            continue
        index = record.atoms[0]
        atom = molecule.GetAtomWithIdx(index)
        reason = None
        if multiplicity[index] != 1 or index in variable_sites:
            reason = "conflicting_annotation_context"
        elif (
            atom.GetAtomicNum() != 0
            or atom.GetDegree() != 1
            or atom.GetIsotope()
            or atom.GetAtomMapNum()
            or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons()
        ):
            reason = "requires_unmodified_single_port_dummy"
        elif atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
            reason = "requires_single_attachment_bond"
        if reason:
            unresolved.append((index, record.label, reason))
            continue
        _replace_atom(molecule, index, record.label)
        replacements.append((index, record.label))
    if not replacements:
        return AbbreviationResolution(document, (), tuple(unresolved))
    with rdBase.BlockLogs():
        try:
            Chem.SanitizeMol(molecule)
            Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
            removed = set(replacements)
            remaining = tuple(
                record
                for record in document.annotations
                if (record.atoms[0], record.label) not in removed or record.kind != "a"
            )
            resolved = parse_structure(_serialize(document, molecule, remaining))
            resolved.model_text
        except (ValueError, RuntimeError, KeyError) as error:
            raise StructureError(
                "abbreviation_expansion_not_losslessly_serializable"
            ) from error
    return AbbreviationResolution(resolved, tuple(replacements), tuple(unresolved))

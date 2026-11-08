"""Versioned monomer definitions and attachment validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

from rdkit import Chem

from .errors import PeptideError


def parse_smiles(text: str) -> Chem.Mol:
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(text, parameters)
    if molecule is None:
        raise PeptideError("invalid_monomer_structure", text)
    return molecule


def identity(molecule: Chem.Mol) -> str:
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    copy = Chem.RemoveHs(copy)
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    return Chem.MolToSmiles(copy, isomericSmiles=True)


@dataclass(frozen=True, slots=True)
class Attachment:
    label: str
    cap_smiles: str


@dataclass(frozen=True, slots=True)
class Monomer:
    symbol: str
    polymer_type: str
    name: str
    smiles: str
    attachments: tuple[Attachment, ...]
    monomer_type: str = "Backbone"
    natural_analog: str | None = None

    @classmethod
    def from_helm(cls, record: Mapping) -> Monomer:
        return cls(
            symbol=record["symbol"],
            polymer_type=record["polymerType"],
            name=record["name"],
            smiles=record["smiles"],
            attachments=tuple(
                Attachment(item["label"], item["capGroupSmiles"]) for item in record["rgroups"]
            ),
            monomer_type=record["monomerType"],
            natural_analog=record.get("naturalAnalog"),
        )

    def template(self) -> Chem.Mol:
        return Chem.Mol(_template(self))

    def attachment(self, label: str) -> Attachment:
        for attachment in self.attachments:
            if attachment.label == label:
                return attachment
        raise PeptideError("unknown_attachment", f"{self.symbol}:{label}")

    def as_helm_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "polymerType": self.polymer_type,
            "name": self.name,
            "smiles": self.smiles,
            "monomerType": self.monomer_type,
            "naturalAnalog": self.natural_analog,
            "rgroups": [
                {"label": a.label, "capGroupSmiles": a.cap_smiles} for a in self.attachments
            ],
        }


@lru_cache(maxsize=4096)
def _template(monomer: Monomer) -> Chem.Mol:
    molecule = parse_smiles(monomer.smiles)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise PeptideError("disconnected_monomer", monomer.symbol)
    labels = [attachment.label for attachment in monomer.attachments]
    if len(labels) != len(set(labels)):
        raise PeptideError("duplicate_attachment", monomer.symbol)
    if any(not re.fullmatch(r"R[1-9][0-9]*", label) for label in labels):
        raise PeptideError("invalid_attachment_label", monomer.symbol)
    numbers = {int(label[1:]) for label in labels}
    marked = [atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()]
    if len(marked) != len(set(marked)) or set(marked) != numbers:
        raise PeptideError("attachment_map_mismatch", monomer.symbol)
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise PeptideError("uncapped_monomer", monomer.symbol)
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise PeptideError("radical_monomer", monomer.symbol)
    edit = Chem.RWMol(molecule)
    caps = []
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if not number:
            continue
        attachment = monomer.attachment(f"R{number}")
        cap = parse_smiles(attachment.cap_smiles)
        dummies = [a for a in cap.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) != 1 or dummies[0].GetAtomMapNum() != number or dummies[0].GetDegree() != 1:
            raise PeptideError("invalid_cap", f"{monomer.symbol}:{attachment.label}")
        if len(Chem.GetMolFrags(cap)) != 1 or any(
            a.GetAtomMapNum() for a in cap.GetAtoms() if a.GetAtomicNum()
        ):
            raise PeptideError("invalid_cap", f"{monomer.symbol}:{attachment.label}")
        if atom.GetDegree() != 1 or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
            raise PeptideError("unsupported_attachment", f"{monomer.symbol}:{attachment.label}")
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(number)
        edit.ReplaceAtom(atom.GetIdx(), dummy)
        caps.append(cap)
    template = edit.GetMol()
    template.RemoveAllConformers()
    Chem.SanitizeMol(template)
    combined = template
    for cap in caps:
        combined = Chem.CombineMols(combined, cap)
    restored = Chem.molzip(combined)
    if identity(restored) != identity(molecule):
        raise PeptideError("cap_roundtrip_mismatch", monomer.symbol)
    return template


@dataclass(frozen=True, slots=True, init=False)
class MonomerLibrary:
    _monomers: Mapping[tuple[str, str], Monomer]
    name: str
    digest: str
    source_url: str
    source_revision: str

    def __init__(
        self,
        monomers: Iterable[Monomer],
        *,
        name: str,
        source_url: str = "",
        source_revision: str = "",
    ):
        records = {}
        for monomer in monomers:
            key = (monomer.polymer_type, monomer.symbol)
            if key in records:
                raise PeptideError("duplicate_monomer", str(key))
            monomer.template()
            records[key] = monomer
        object.__setattr__(self, "_monomers", MappingProxyType(records))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_url", source_url)
        object.__setattr__(self, "source_revision", source_revision)
        content = [
            {
                "polymer_type": m.polymer_type,
                "symbol": m.symbol,
                "smiles": m.smiles,
                "monomer_type": m.monomer_type,
                "natural_analog": m.natural_analog,
                "attachments": sorted((a.label, a.cap_smiles) for a in m.attachments),
            }
            for _, m in sorted(records.items())
        ]
        object.__setattr__(
            self, "digest", hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        )

    @property
    def monomers(self) -> Mapping[tuple[str, str], Monomer]:
        return self._monomers

    def get(self, symbol: str, polymer_type: str = "PEPTIDE") -> Monomer:
        try:
            return self._monomers[polymer_type, symbol]
        except KeyError as error:
            raise PeptideError("unknown_monomer", f"{polymer_type}:{symbol}") from error

    def extend(self, monomers: Iterable[Monomer], *, name: str) -> MonomerLibrary:
        return MonomerLibrary((*self._monomers.values(), *monomers), name=name)

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "name": self.name,
                    "digest": self.digest,
                    "source_url": self.source_url,
                    "source_revision": self.source_revision,
                    "monomers": [m.as_helm_dict() for _, m in sorted(self._monomers.items())],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> MonomerLibrary:
        try:
            document = json.loads(text)
            if document["schema_version"] != 1:
                raise PeptideError("unsupported_library_schema", str(document["schema_version"]))
            library = cls(
                (Monomer.from_helm(row) for row in document["monomers"]),
                name=document["name"],
                source_url=document["source_url"],
                source_revision=document["source_revision"],
            )
            if library.digest != document["digest"]:
                raise PeptideError("library_checksum_mismatch", library.name)
            return library
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise PeptideError("invalid_library_document", str(error)) from error


@lru_cache(maxsize=1)
def core_library() -> MonomerLibrary:
    directory = files("peptidecore").joinpath("data")
    manifest = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
    payload = directory.joinpath("helm_core_peptide.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["data_sha256"]:
        raise PeptideError("library_checksum_mismatch", manifest["name"])
    records = json.loads(payload)
    if len(records) != manifest["entries"]:
        raise PeptideError("library_entry_count_mismatch", manifest["name"])
    return MonomerLibrary(
        (Monomer.from_helm(record) for record in records),
        name=manifest["name"],
        source_url=manifest["source_url"],
        source_revision=manifest["source_git_blob"],
    )

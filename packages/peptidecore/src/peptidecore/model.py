"""Monomer connections and their assembled atomic structure."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from rdkit import Chem, rdBase

from .errors import PeptideError
from .library import MonomerLibrary, identity


@dataclass(frozen=True, slots=True, order=True)
class Endpoint:
    chain: str
    position: int
    attachment: str


@dataclass(frozen=True, slots=True)
class Connection:
    source: Endpoint
    target: Endpoint


@dataclass(frozen=True, slots=True)
class Chain:
    identifier: str
    monomers: tuple[str, ...]
    polymer_type: str = "PEPTIDE"

    def __post_init__(self):
        if self.polymer_type not in ("PEPTIDE", "CHEM"):
            raise PeptideError("unsupported_polymer_type", self.polymer_type)
        if not re.fullmatch(re.escape(self.polymer_type) + r"[1-9][0-9]*", self.identifier):
            raise PeptideError("invalid_chain_identifier", self.identifier)
        if (
            not isinstance(self.monomers, tuple)
            or not self.monomers
            or any(not m for m in self.monomers)
        ):
            raise PeptideError("invalid_chain_monomers", self.identifier)
        if self.polymer_type == "CHEM" and len(self.monomers) != 1:
            raise PeptideError("invalid_chemical_chain", self.identifier)


@dataclass(frozen=True, slots=True)
class PeptideGraph:
    chains: tuple[Chain, ...]
    connections: tuple[Connection, ...] = ()

    def __post_init__(self):
        if not isinstance(self.chains, tuple) or not all(isinstance(c, Chain) for c in self.chains):
            raise PeptideError("invalid_chains", "Chains must be an immutable tuple")
        if not isinstance(self.connections, tuple) or not all(
            isinstance(c, Connection) for c in self.connections
        ):
            raise PeptideError("invalid_connections", "Connections must be an immutable tuple")

    def resolved_connections(self, library: MonomerLibrary) -> tuple[Connection, ...]:
        if not self.chains or len({c.identifier for c in self.chains}) != len(self.chains):
            raise PeptideError("duplicate_or_missing_chain", "Chain identifiers must be unique")
        chains = {chain.identifier: chain for chain in self.chains}
        for chain in self.chains:
            for symbol in chain.monomers:
                library.get(symbol, chain.polymer_type)
        implicit = tuple(
            Connection(Endpoint(chain.identifier, i, "R2"), Endpoint(chain.identifier, i + 1, "R1"))
            for chain in self.chains
            if chain.polymer_type == "PEPTIDE"
            for i in range(1, len(chain.monomers))
        )
        resolved = implicit + self.connections
        occupied = set()
        for connection in resolved:
            for endpoint in (connection.source, connection.target):
                chain = chains.get(endpoint.chain)
                if chain is None:
                    raise PeptideError("unknown_chain", endpoint.chain)
                if type(endpoint.position) is not int or not 1 <= endpoint.position <= len(
                    chain.monomers
                ):
                    raise PeptideError("invalid_monomer_position", str(endpoint))
                library.get(chain.monomers[endpoint.position - 1], chain.polymer_type).attachment(
                    endpoint.attachment
                )
                if endpoint in occupied:
                    raise PeptideError("attachment_already_used", str(endpoint))
                occupied.add(endpoint)
        return resolved


@dataclass(frozen=True, slots=True)
class AtomicConnection:
    connection: Connection
    source_atom: int
    target_atom: int


@dataclass(frozen=True, slots=True)
class ChemicalRecord:
    graph: PeptideGraph
    library_name: str
    library_digest: str
    atom_instances: tuple[str, ...]
    atom_origins: tuple[int, ...]
    connections: tuple[AtomicConnection, ...]
    source: str
    source_format: str
    _molecule: Chem.Mol = field(repr=False, compare=False)
    structure_id: str = field(init=False)

    def __post_init__(self):
        molecule = Chem.Mol(self._molecule)
        if (
            len(self.atom_instances) != molecule.GetNumAtoms()
            or len(self.atom_origins) != molecule.GetNumAtoms()
        ):
            raise PeptideError(
                "atom_mapping_mismatch", "Atom mappings must cover the complete structure"
            )
        if any(a.GetAtomicNum() == 0 for a in molecule.GetAtoms()):
            raise PeptideError("unresolved_attachment", "The structure contains dummy atoms")
        for edge in self.connections:
            bond = molecule.GetBondBetweenAtoms(edge.source_atom, edge.target_atom)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                raise PeptideError("connection_mapping_mismatch", str(edge.connection))
        object.__setattr__(self, "_molecule", molecule)
        object.__setattr__(self, "structure_id", identity(molecule))

    @property
    def molecule(self) -> Chem.Mol:
        return Chem.Mol(self._molecule)

    @property
    def smiles(self) -> str:
        return self.structure_id

    @property
    def attachment_atoms(self) -> Mapping[Endpoint, int]:
        result = {}
        for atom in self._molecule.GetAtoms():
            if not atom.HasProp("_peptide_attachment_labels"):
                continue
            chain, position = self.atom_instances[atom.GetIdx()].rsplit(":", 1)
            for label in atom.GetProp("_peptide_attachment_labels").split(","):
                endpoint = Endpoint(chain, int(position), label)
                if endpoint in result:
                    raise PeptideError("duplicate_attachment_anchor", str(endpoint))
                result[endpoint] = atom.GetIdx()
        for connection in self.connections:
            if (
                result.get(connection.connection.source) != connection.source_atom
                or result.get(connection.connection.target) != connection.target_atom
            ):
                raise PeptideError("missing_attachment_anchors", str(connection.connection))
        return MappingProxyType(result)

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "rdkit_version": rdBase.rdkitVersion,
            "smiles": self.smiles,
            "library": {"name": self.library_name, "digest": self.library_digest},
            "source": {"text": self.source, "format": self.source_format},
            "chains": [
                {"id": c.identifier, "polymer_type": c.polymer_type, "monomers": list(c.monomers)}
                for c in self.graph.chains
            ],
            "atom_instances": list(self.atom_instances),
            "atom_origins": list(self.atom_origins),
            "connections": [
                {
                    "source": {
                        "chain": e.connection.source.chain,
                        "position": e.connection.source.position,
                        "attachment": e.connection.source.attachment,
                        "atom": e.source_atom,
                    },
                    "target": {
                        "chain": e.connection.target.chain,
                        "position": e.connection.target.position,
                        "attachment": e.connection.target.attachment,
                        "atom": e.target_atom,
                    },
                }
                for e in self.connections
            ],
        }

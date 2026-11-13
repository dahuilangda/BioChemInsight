"""Ordinary amino-acid sequence input with explicit export restrictions."""

from __future__ import annotations

from types import MappingProxyType

from .assembly import assemble
from .errors import PeptideError
from .library import MonomerLibrary
from .model import Chain, ChemicalRecord, PeptideGraph

SEQUENCE_MONOMERS = MappingProxyType(
    {**{symbol: symbol for symbol in "ACDEFGHIKLMNPQRSTVWY"}, "U": "seC", "O": "Pyl"}
)
MONOMER_CODES = MappingProxyType({value: key for key, value in SEQUENCE_MONOMERS.items()})
AMINO_ACIDS = frozenset(SEQUENCE_MONOMERS)


def from_sequence(text: str, *, library: MonomerLibrary | None = None) -> ChemicalRecord:
    symbols = "".join(text.split()).upper()
    if not symbols or not set(symbols) <= AMINO_ACIDS:
        raise PeptideError(
            "unsupported_sequence_symbol", "Expected unambiguous protein amino-acid codes"
        )
    graph = PeptideGraph((Chain("PEPTIDE1", tuple(SEQUENCE_MONOMERS[s] for s in symbols)),))
    return assemble(graph, library=library, source=text, source_format="sequence")


def from_fasta(text: str, *, library: MonomerLibrary | None = None) -> ChemicalRecord:
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith(">") or any(line.startswith(">") for line in lines[1:]):
        raise PeptideError("invalid_fasta", "Expected one FASTA record")
    sequence = "".join(lines[1:])
    record = from_sequence(sequence, library=library)
    return ChemicalRecord(
        graph=record.graph,
        library_name=record.library_name,
        library_digest=record.library_digest,
        atom_instances=record.atom_instances,
        atom_origins=record.atom_origins,
        connections=record.connections,
        source=text,
        source_format="FASTA",
        _molecule=record.molecule,
    )


def to_sequence(record: ChemicalRecord) -> str:
    graph = record.graph
    if len(graph.chains) != 1 or graph.connections:
        raise PeptideError(
            "sequence_information_loss", "Connections or multiple chains require HELM"
        )
    chain = graph.chains[0]
    if chain.polymer_type != "PEPTIDE" or any(
        symbol not in MONOMER_CODES for symbol in chain.monomers
    ):
        raise PeptideError(
            "sequence_information_loss", "Modified residues or terminal groups require HELM"
        )
    if assemble(graph).smiles != record.smiles:
        raise PeptideError(
            "sequence_information_loss", "The structure differs from the standard sequence"
        )
    return "".join(MONOMER_CODES[symbol] for symbol in chain.monomers)

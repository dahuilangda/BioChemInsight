"""HELM input and output for explicit peptide and chemical connections."""

from __future__ import annotations

import re

from ._syntax import split_top_level
from .assembly import assemble
from .errors import PeptideError
from .library import MonomerLibrary
from .model import Chain, ChemicalRecord, Connection, Endpoint, PeptideGraph


def _split(text: str, delimiter: str) -> list[str]:
    return split_top_level(text, delimiter, error_code="invalid_helm_syntax")


def _symbol(token: str) -> str:
    if re.fullmatch(r"[A-Z]", token) and token != "X":
        return token
    if token.startswith("[") and token.endswith("]"):
        value = token[1:-1]
        if value and not any(c in value for c in '[]{}"') and value not in ("*", "?", "X", "_"):
            return value
    raise PeptideError("unsupported_helm_monomer", token)


def parse_helm(text: str) -> PeptideGraph:
    sections = _split(text.strip(), "$")
    if len(sections) != 5:
        raise PeptideError("invalid_helm_sections", "Expected five dollar-separated sections")
    polymers, connections, groups, annotations, version = sections
    if version not in ("", "V2.0"):
        raise PeptideError("unsupported_helm_version", version)
    if groups or annotations:
        raise PeptideError("unsupported_helm_sections", "Groups and annotations are not supported")
    chains = []
    for polymer in _split(polymers, "|"):
        match = re.fullmatch(r"((PEPTIDE|CHEM)[1-9][0-9]*)\{(.*)\}", polymer, re.DOTALL)
        if match is None:
            raise PeptideError("unsupported_helm_polymer", polymer)
        identifier, polymer_type, body = match.groups()
        chains.append(
            Chain(identifier, tuple(_symbol(token) for token in _split(body, ".")), polymer_type)
        )
    edges = []
    if connections:
        for connection in _split(connections, "|"):
            match = re.fullmatch(
                r"([A-Z]+[1-9][0-9]*),([A-Z]+[1-9][0-9]*),([1-9][0-9]*):(R[1-9][0-9]*)-([1-9][0-9]*):(R[1-9][0-9]*)",
                connection,
            )
            if match is None:
                raise PeptideError("unsupported_helm_connection", connection)
            source, target, source_pos, source_site, target_pos, target_site = match.groups()
            edges.append(
                Connection(
                    Endpoint(source, int(source_pos), source_site),
                    Endpoint(target, int(target_pos), target_site),
                )
            )
    return PeptideGraph(tuple(chains), tuple(edges))


def to_helm(graph: PeptideGraph) -> str:
    def symbol(value: str) -> str:
        if re.fullmatch(r"[A-Z]", value) and value != "X":
            return value
        if _symbol(f"[{value}]") != value:
            raise PeptideError("unrepresentable_helm_symbol", value)
        return f"[{value}]"

    chains = "|".join(
        f"{c.identifier}{{{'.'.join(symbol(s) for s in c.monomers)}}}" for c in graph.chains
    )
    connections = "|".join(
        f"{edge.source.chain},{edge.target.chain},{edge.source.position}:{edge.source.attachment}-{edge.target.position}:{edge.target.attachment}"
        for edge in graph.connections
    )
    text = f"{chains}${connections}$$$V2.0"
    if parse_helm(text) != graph:
        raise PeptideError("helm_serialization_mismatch", "The graph cannot be serialized exactly")
    return text


def from_helm(text: str, *, library: MonomerLibrary | None = None) -> ChemicalRecord:
    return assemble(parse_helm(text), library=library, source=text, source_format="HELM")

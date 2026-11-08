"""BILN input and output for peptide chains and explicit connections."""

from __future__ import annotations

import re

from ._syntax import split_top_level
from .assembly import assemble
from .errors import PeptideError
from .library import MonomerLibrary
from .model import Chain, ChemicalRecord, Connection, Endpoint, PeptideGraph

_CONNECTION = re.compile(r"\(([1-9][0-9]*),([1-9][0-9]*)\)")
_SUFFIX = re.compile(_CONNECTION.pattern + r"$")


def _split(text: str, delimiter: str) -> list[str]:
    return split_top_level(text, delimiter, error_code="invalid_biln_syntax")


def _suffixes(text: str) -> tuple[str, list[tuple[int, str]]]:
    annotations = []
    while match := _SUFFIX.search(text):
        annotations.append((int(match[1]), f"R{match[2]}"))
        text = text[: match.start()]
    return text, annotations[::-1]


def _monomer(token: str) -> tuple[str, list[tuple[int, str]]]:
    if token.startswith("["):
        if token.count("[") != 1 or token.count("]") != 1:
            raise PeptideError("invalid_biln_monomer", token)
        closing = token.index("]")
        symbol, inside = _suffixes(token[1:closing])
        outside = token[closing + 1 :]
        if "".join(match[0] for match in _CONNECTION.finditer(outside)) != outside:
            raise PeptideError("invalid_biln_connection", outside)
        annotations = inside + [(int(m[1]), f"R{m[2]}") for m in _CONNECTION.finditer(outside)]
    else:
        symbol, annotations = _suffixes(token)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol):
            raise PeptideError("invalid_biln_monomer", token)
    if not symbol or any(c in symbol for c in "[]\n\r"):
        raise PeptideError("invalid_biln_monomer", token)
    return symbol, annotations


def parse_biln(text: str) -> PeptideGraph:
    chains = []
    labels: dict[int, list[Endpoint]] = {}
    chain_texts = _split(text.strip(), ".")
    for number, chain_text in enumerate(chain_texts, 1):
        identifier = f"PEPTIDE{number}"
        symbols = []
        for position, token in enumerate(_split(chain_text, "-"), 1):
            symbol, annotations = _monomer(token)
            symbols.append(symbol)
            for label, attachment in annotations:
                labels.setdefault(label, []).append(Endpoint(identifier, position, attachment))
        chains.append(Chain(identifier, tuple(symbols)))
    connections = []
    for label, endpoints in sorted(labels.items()):
        if len(endpoints) != 2:
            raise PeptideError(
                "invalid_biln_connection_count",
                f"Connection {label} has {len(endpoints)} endpoints",
            )
        connections.append(Connection(*endpoints))
    return PeptideGraph(tuple(chains), tuple(connections))


def to_biln(graph: PeptideGraph) -> str:
    if any(chain.polymer_type != "PEPTIDE" for chain in graph.chains):
        raise PeptideError("biln_information_loss", "Chemical polymer types require HELM")
    references = {chain.identifier: len(chain.monomers) for chain in graph.chains}
    annotations: dict[tuple[str, int], list[str]] = {}
    for number, connection in enumerate(graph.connections, 1):
        for endpoint in (connection.source, connection.target):
            if (
                endpoint.chain not in references
                or not 1 <= endpoint.position <= references[endpoint.chain]
            ):
                raise PeptideError("invalid_biln_endpoint", str(endpoint))
            if not re.fullmatch(r"R[1-9][0-9]*", endpoint.attachment):
                raise PeptideError("invalid_biln_endpoint", str(endpoint))
            annotations.setdefault((endpoint.chain, endpoint.position), []).append(
                f"({number},{endpoint.attachment[1:]})"
            )
    chains = []
    for chain in graph.chains:
        monomers = []
        for position, symbol in enumerate(chain.monomers, 1):
            if not symbol or _SUFFIX.search(symbol) or any(c in symbol for c in "[]\n\r"):
                raise PeptideError("unrepresentable_biln_symbol", symbol)
            token = symbol if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol) else f"[{symbol}]"
            monomers.append(token + "".join(annotations.get((chain.identifier, position), ())))
        chains.append("-".join(monomers))
    result = ".".join(chains)
    parse_biln(result)
    return result


def from_biln(text: str, *, library: MonomerLibrary | None = None) -> ChemicalRecord:
    return assemble(parse_biln(text), library=library, source=text, source_format="BILN")

"""Exact monomer matches with explicit attachment boundaries."""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem

from .errors import PeptideError
from .library import Monomer, MonomerLibrary, parse_smiles


@dataclass(frozen=True, slots=True)
class ResidueMatch:
    symbol: str
    polymer_type: str
    aliases: tuple[str, ...]
    atoms: tuple[int, ...]
    origins: tuple[tuple[int, int], ...]
    ports: tuple[tuple[str, int, int], ...]

    @property
    def mask(self) -> int:
        return sum(1 << atom for atom in self.atoms)


@dataclass(frozen=True, slots=True)
class MatchSearch:
    matches: tuple[ResidueMatch, ...]
    limits_reached: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Variant:
    monomer: Monomer
    aliases: tuple[str, ...]
    query: Chem.Mol
    partial: Chem.Mol
    real: tuple[int, ...]
    ports: tuple[tuple[str, int, int], ...]
    elements: tuple[tuple[int, int], ...]
    query_indices: tuple[int, ...]


def _port_partitions(ports, groups=()):
    if not ports:
        yield groups
        return
    port, remaining = ports[0], ports[1:]
    yield from _port_partitions(remaining, groups + ((port,),))
    for index, group in enumerate(groups):
        if all(other[1] != port[1] for other in group):
            yield from _port_partitions(
                remaining, groups[:index] + (group + (port,),) + groups[index + 1 :]
            )


def _attachment_queries(query, ports):
    """Allow distinct attachment bonds to meet the same external atom."""
    for groups in _port_partitions(ports):
        replacements = {other[2]: group[0][2] for group in groups for other in group[1:]}
        if not replacements:
            yield query, tuple(range(query.GetNumAtoms()))
            continue
        edit = Chem.RWMol(query)
        for removed, retained in replacements.items():
            anchor = query.GetAtomWithIdx(removed).GetNeighbors()[0].GetIdx()
            edit.AddBond(retained, anchor, Chem.BondType.SINGLE)
        for index in sorted(replacements, reverse=True):
            edit.RemoveAtom(index)
        remaining = [i for i in range(query.GetNumAtoms()) if i not in replacements]
        renumber = {old: new for new, old in enumerate(remaining)}
        indices = tuple(renumber[replacements.get(i, i)] for i in range(query.GetNumAtoms()))
        merged = edit.GetMol()
        for original in query.GetAtoms():
            tag = original.GetChiralTag()
            if not tag or original.GetIdx() in replacements:
                continue
            atom = merged.GetAtomWithIdx(indices[original.GetIdx()])
            expected = [indices[a.GetIdx()] for a in original.GetNeighbors()]
            actual = [a.GetIdx() for a in atom.GetNeighbors()]
            permutation = [expected.index(i) for i in actual]
            parity = sum(a > b for i, a in enumerate(permutation) for b in permutation[i + 1 :]) % 2
            atom.SetChiralTag(tag)
            if parity:
                atom.InvertChirality()
        yield merged, indices


def _partial(monomer: Monomer, used: frozenset[str]) -> Chem.Mol:
    molecule = monomer.template()
    caps = []
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_origin", atom.GetIdx())
        if atom.GetAtomicNum():
            continue
        number = atom.GetAtomMapNum()
        label = f"R{number}"
        atom.SetProp("_attachment", label)
        if label in used:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(number)
        else:
            cap = parse_smiles(monomer.attachment(label).cap_smiles)
            for other in cap.GetAtoms():
                other.SetIntProp("_origin", atom.GetIdx())
            caps.append(cap)
    for cap in caps:
        molecule = Chem.CombineMols(molecule, cap)
    result = Chem.RemoveHs(Chem.molzip(molecule))
    for atom in result.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(int(atom.GetProp("_attachment")[1:]))
            atom.SetIsotope(0)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result


@lru_cache(maxsize=4)
def _variants(monomers: tuple[Monomer, ...]) -> tuple[_Variant, ...]:
    groups = {}
    for monomer in monomers:
        if len(monomer.attachments) > 8:
            raise PeptideError("decomposition_attachment_limit", monomer.symbol)
        key = (
            monomer.polymer_type,
            Chem.MolToSmiles(monomer.template()),
            tuple(sorted((a.label, a.cap_smiles) for a in monomer.attachments)),
        )
        groups.setdefault(key, []).append(monomer)
    variants = []
    for group in groups.values():
        monomer = group[0]
        aliases = tuple(item.symbol for item in group[1:])
        labels = tuple(a.label for a in monomer.attachments)
        for size in range(len(labels) + 1):
            for used in itertools.combinations(labels, size):
                partial = _partial(monomer, frozenset(used))
                mapped = Chem.Mol(partial)
                for atom in mapped.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx() + 1)
                query = Chem.MolFromSmarts(Chem.MolToSmarts(mapped))
                order = sorted(
                    range(query.GetNumAtoms()),
                    key=lambda i: query.GetAtomWithIdx(i).GetAtomMapNum(),
                )
                query = Chem.RenumberAtoms(query, order)
                for atom in query.GetAtoms():
                    atom.SetAtomMapNum(0)
                real, ports = [], []
                for atom in partial.GetAtoms():
                    index = atom.GetIdx()
                    if atom.GetAtomicNum() == 0:
                        ports.append(
                            (atom.GetProp("_attachment"), atom.GetNeighbors()[0].GetIdx(), index)
                        )
                    else:
                        real.append(index)
                        constraint = f"[D{atom.GetDegree()};H{atom.GetTotalNumHs(includeNeighbors=True)};{atom.GetFormalCharge():+d}]"
                        query.GetAtomWithIdx(index).ExpandQuery(Chem.AtomFromSmarts(constraint))
                edit = Chem.RWMol(query)
                for _, _, index in ports:
                    edit.ReplaceAtom(index, Chem.AtomFromSmarts("*"))
                elements = Counter(partial.GetAtomWithIdx(i).GetAtomicNum() for i in real)
                for attachment_query, indices in _attachment_queries(edit.GetMol(), tuple(ports)):
                    variants.append(
                        _Variant(
                            monomer,
                            aliases,
                            attachment_query,
                            partial,
                            tuple(real),
                            tuple(ports),
                            tuple(sorted(elements.items())),
                            indices,
                        )
                    )
    return tuple(variants)


def find_matches(
    molecule: Chem.Mol,
    library: MonomerLibrary,
    *,
    deadline: float,
    max_matches: int,
) -> MatchSearch:
    variants = _variants(tuple(m for _, m in sorted(library.monomers.items())))
    elements = Counter(a.GetAtomicNum() for a in molecule.GetAtoms())
    adjacency = [{a.GetIdx() for a in atom.GetNeighbors()} for atom in molecule.GetAtoms()]
    candidates = {}
    limits = set()
    for variant in variants:
        if time.monotonic() >= deadline:
            return MatchSearch(tuple(candidates.values()), tuple(sorted(limits | {"timeout"})))
        if variant.query.GetNumAtoms() > molecule.GetNumAtoms() or any(
            count > elements[number] for number, count in variant.elements
        ):
            continue
        matches = molecule.GetSubstructMatches(
            variant.query, useChirality=True, uniquify=False, maxMatches=max_matches + 1
        )
        if len(matches) > max_matches:
            limits.add("query_matches")
        for native_match in matches[:max_matches]:
            match = tuple(native_match[index] for index in variant.query_indices)
            owned = tuple(sorted(match[i] for i in variant.real))
            owned_set = set(owned)
            ports = tuple(
                sorted(
                    (label, match[anchor], match[external])
                    for label, anchor, external in variant.ports
                )
            )
            if any(other in owned_set for _, _, other in ports):
                continue
            boundary = {(a, b) for a in owned for b in adjacency[a] - owned_set}
            if boundary != {(a, b) for _, a, b in ports}:
                continue
            if any(
                variant.partial.GetAtomWithIdx(i).GetIsotope()
                != molecule.GetAtomWithIdx(match[i]).GetIsotope()
                or variant.partial.GetAtomWithIdx(i).GetIsAromatic()
                != molecule.GetAtomWithIdx(match[i]).GetIsAromatic()
                or bool(variant.partial.GetAtomWithIdx(i).GetChiralTag())
                != bool(molecule.GetAtomWithIdx(match[i]).GetChiralTag())
                for i in variant.real
            ):
                continue
            if any(
                bool(bond.GetStereo())
                != bool(
                    molecule.GetBondBetweenAtoms(
                        match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
                    ).GetStereo()
                )
                for bond in variant.partial.GetBonds()
                if bond.GetBeginAtomIdx() in variant.real and bond.GetEndAtomIdx() in variant.real
            ):
                continue
            origins = tuple(
                sorted(
                    (variant.partial.GetAtomWithIdx(i).GetIntProp("_origin"), match[i])
                    for i in variant.real
                )
            )
            candidate = ResidueMatch(
                variant.monomer.symbol,
                variant.monomer.polymer_type,
                variant.aliases,
                owned,
                origins,
                ports,
            )
            key = (candidate.polymer_type, candidate.symbol, candidate.atoms, candidate.ports)
            previous = candidates.get(key)
            if previous is None or candidate.origins < previous.origins:
                candidates[key] = candidate
            if len(candidates) > max_matches:
                return MatchSearch(
                    tuple(candidates.values())[:max_matches], tuple(sorted(limits | {"candidates"}))
                )
    return MatchSearch(tuple(candidates.values()), tuple(sorted(limits)))

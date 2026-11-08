"""Recover monomer graphs from an atomic structure and verify their assembly."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem

from .assembly import assemble
from .errors import PeptideError
from .helm import to_helm
from .library import MonomerLibrary, core_library, identity
from .matching import ResidueMatch, find_matches
from .model import AtomicConnection, Chain, ChemicalRecord, Connection, Endpoint, PeptideGraph


@dataclass(frozen=True, slots=True)
class UnidentifiedFragment:
    atoms: tuple[int, ...]
    smiles: str
    boundaries: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    status: Literal["resolved", "ambiguous", "partial", "budget_exceeded"]
    records: tuple[ChemicalRecord, ...]
    matches: tuple[ResidueMatch, ...]
    unidentified: tuple[UnidentifiedFragment, ...]
    source: str
    library_digest: str
    search_exhausted: bool
    states: int
    candidate_count: int
    seconds: float
    atom_count: int
    limits_reached: tuple[str, ...]
    verification_failures: int

    @property
    def unique(self) -> bool:
        return self.status == "resolved"

    @property
    def coverage(self) -> float:
        return len({a for match in self.matches for a in match.atoms}) / self.atom_count

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "status": self.status,
            "source": self.source,
            "library_digest": self.library_digest,
            "search_exhausted": self.search_exhausted,
            "coverage": self.coverage,
            "atom_count": self.atom_count,
            "states": self.states,
            "candidate_count": self.candidate_count,
            "seconds": self.seconds,
            "limits_reached": list(self.limits_reached),
            "verification_failures": self.verification_failures,
            "records": [record.as_dict() for record in self.records],
            "matches": [
                {
                    "symbol": match.symbol,
                    "polymer_type": match.polymer_type,
                    "aliases": list(match.aliases),
                    "atoms": list(match.atoms),
                    "origins": [list(pair) for pair in match.origins],
                    "ports": [
                        {"attachment": label, "atom": a, "neighbor": b}
                        for label, a, b in match.ports
                    ],
                }
                for match in self.matches
            ],
            "unidentified": [
                {
                    "atoms": list(fragment.atoms),
                    "smiles": fragment.smiles,
                    "boundaries": [list(pair) for pair in fragment.boundaries],
                }
                for fragment in self.unidentified
            ],
        }


def _graph(
    matches: tuple[ResidueMatch, ...],
    ranks: tuple[int, ...],
) -> tuple[PeptideGraph, dict[int, str]]:
    owners = {atom: i for i, match in enumerate(matches) for atom in match.atoms}
    ports = {(a, b): (i, label) for i, match in enumerate(matches) for label, a, b in match.ports}
    edges = []
    for (a, b), (owner, label) in sorted(ports.items()):
        if a > b:
            continue
        other = ports.get((b, a))
        if other is None or owners.get(b) != other[0]:
            raise PeptideError("unpaired_decomposition_attachment", f"{a}-{b}")
        edges.append((owner, label, other[0], other[1]))
    successor, predecessor = {}, {}
    for left, left_site, right, right_site in edges:
        if matches[left].polymer_type != "PEPTIDE" or matches[right].polymer_type != "PEPTIDE":
            continue
        if (left_site, right_site) == ("R1", "R2"):
            left, right = right, left
        elif (left_site, right_site) != ("R2", "R1"):
            continue
        successor[left], predecessor[right] = right, left

    def order(index):
        match = matches[index]
        return match.polymer_type, match.symbol, min(ranks[a] for a in match.atoms)

    remaining = set(range(len(matches)))
    paths = []
    while remaining:
        starts = [i for i in remaining if i not in predecessor]
        current = min(starts or remaining, key=order)
        path = []
        while current in remaining:
            remaining.remove(current)
            path.append(current)
            current = successor.get(current, -1)
        paths.append(path)
    paths.sort(key=lambda path: order(path[0]))
    chains, endpoints, instances, implicit = [], {}, {}, set()
    counts = {"PEPTIDE": 0, "CHEM": 0}
    for path in paths:
        polymer_type = matches[path[0]].polymer_type
        counts[polymer_type] += 1
        chain = f"{polymer_type}{counts[polymer_type]}"
        chains.append(Chain(chain, tuple(matches[i].symbol for i in path), polymer_type))
        for position, owner in enumerate(path, 1):
            endpoints[owner] = chain, position
            instances[owner] = f"{chain}:{position}"
        implicit.update((left, right) for left, right in zip(path, path[1:], strict=False))
    connections = []
    for left, left_site, right, right_site in edges:
        if ((left_site, right_site) == ("R2", "R1") and (left, right) in implicit) or (
            (left_site, right_site) == ("R1", "R2") and (right, left) in implicit
        ):
            continue
        source, target = sorted(
            (Endpoint(*endpoints[left], left_site), Endpoint(*endpoints[right], right_site))
        )
        connections.append(Connection(source, target))
    connections.sort(key=lambda edge: (edge.source, edge.target))
    return PeptideGraph(tuple(chains), tuple(connections)), instances


def _record(
    molecule: Chem.Mol,
    matches: tuple[ResidueMatch, ...],
    ranks: tuple[int, ...],
    library: MonomerLibrary,
    source: str,
) -> ChemicalRecord:
    graph, instances = _graph(matches, ranks)
    rebuilt = assemble(graph, library=library)
    if rebuilt.smiles != identity(molecule):
        raise PeptideError(
            "decomposition_structure_mismatch", "Reassembly changed the input structure"
        )
    original_by_origin = {
        (instances[index], origin): atom
        for index, match in enumerate(matches)
        for origin, atom in match.origins
    }
    mapping = [
        original_by_origin[instance, origin]
        for instance, origin in zip(rebuilt.atom_instances, rebuilt.atom_origins, strict=True)
    ]
    if set(mapping) != set(range(molecule.GetNumAtoms())) or len(mapping) != molecule.GetNumAtoms():
        raise PeptideError(
            "decomposition_mapping_mismatch", "Reassembly did not preserve the complete atom map"
        )
    target, assembled = Chem.Mol(molecule), rebuilt.molecule
    for atom in target.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    for atom in assembled.GetAtoms():
        atom.SetAtomMapNum(mapping[atom.GetIdx()] + 1)
    if Chem.MolToSmiles(target) != Chem.MolToSmiles(assembled):
        raise PeptideError(
            "decomposition_mapping_mismatch", "The atom map changes bonding or stereochemistry"
        )
    target = Chem.Mol(molecule)
    atom_instances = [""] * target.GetNumAtoms()
    atom_origins = [-1] * target.GetNumAtoms()
    for index, original in enumerate(mapping):
        atom_instances[original] = rebuilt.atom_instances[index]
        atom_origins[original] = rebuilt.atom_origins[index]
        atom = target.GetAtomWithIdx(original)
        atom.SetAtomMapNum(0)
        atom.SetProp("_peptide_instance", atom_instances[original])
        atom.SetIntProp("_peptide_origin", atom_origins[original])
    connections = tuple(
        AtomicConnection(edge.connection, mapping[edge.source_atom], mapping[edge.target_atom])
        for edge in rebuilt.connections
    )
    return ChemicalRecord(
        graph,
        library.name,
        library.digest,
        tuple(atom_instances),
        tuple(atom_origins),
        connections,
        source,
        "SMILES",
        target,
    )


def _unidentified(
    molecule: Chem.Mol, matches: tuple[ResidueMatch, ...]
) -> tuple[UnidentifiedFragment, ...]:
    known = {a for match in matches for a in match.atoms}
    unknown = set(range(molecule.GetNumAtoms())) - known
    cuts, boundaries = [], []
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in known) != (b in known):
            cuts.append(bond.GetIdx())
            boundaries.append((a, b) if a in unknown else (b, a))
    fragmented = Chem.FragmentOnBonds(molecule, cuts, addDummies=True) if cuts else molecule
    mappings = []
    fragments = Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mappings)
    result = []
    for fragment, mapping in zip(fragments, mappings, strict=True):
        atoms = tuple(sorted(set(mapping) & unknown))
        if atoms:
            result.append(
                UnidentifiedFragment(
                    atoms,
                    Chem.MolToSmiles(fragment),
                    tuple(sorted((a, b) for a, b in boundaries if a in atoms)),
                )
            )
    return tuple(sorted(result, key=lambda item: item.atoms))


def decompose_smiles(
    text: str,
    *,
    library: MonomerLibrary | None = None,
    timeout: float = 10.0,
    max_solutions: int = 8,
    max_states: int = 20000,
    max_matches: int = 20000,
) -> DecompositionResult:
    """Find exact monomer covers and retain unresolved structures and alternatives."""
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or any(type(value) is not int for value in (max_solutions, max_states, max_matches))
        or max_solutions < 2
        or max_states < 1
        or max_matches < 1
    ):
        raise ValueError("Positive search budgets and at least two solutions are required")
    started = time.monotonic()
    deadline = started + timeout
    library = core_library() if library is None else library
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise PeptideError("invalid_input_smiles", text)
    if not molecule.GetNumAtoms() or any(
        a.GetAtomicNum() == 0 or a.GetNumRadicalElectrons() for a in molecule.GetAtoms()
    ):
        raise PeptideError("unresolved_input_structure", "A resolved molecular graph is required")
    if molecule.GetStereoGroups():
        raise PeptideError(
            "unsupported_enhanced_stereochemistry",
            "Enhanced stereo groups require a separate representation",
        )
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    match_search = find_matches(molecule, library, deadline=deadline, max_matches=max_matches)
    candidates = tuple(
        sorted(
            match_search.matches,
            key=lambda m: (-len(m.atoms), m.polymer_type, m.symbol, m.atoms, m.ports),
        )
    )
    masks = tuple(match.mask for match in candidates)
    by_atom = [[] for _ in range(molecule.GetNumAtoms())]
    for index, match in enumerate(candidates):
        for atom in match.atoms:
            by_atom[atom].append(index)
    ranks = tuple(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    full = (1 << molecule.GetNumAtoms()) - 1
    best: tuple[ResidueMatch, ...] = ()
    best_coverage = 0
    records: dict[str, ChemicalRecord] = {}
    states = 0
    limits = set(match_search.limits_reached)
    verification_failures = 0

    pending = [(0, 0, ())]
    while pending:
        covered, skipped, selected = pending.pop()
        if records and skipped:
            continue
        if states >= max_states:
            limits.add("states")
        if time.monotonic() >= deadline:
            limits.add("timeout")
        if len(records) >= max_solutions:
            limits.add("solutions")
        if limits & {"states", "timeout", "solutions"}:
            break
        states += 1
        coverage = covered.bit_count()
        if coverage > best_coverage:
            best_coverage = coverage
            best = tuple(candidates[i] for i in selected)
        if covered == full:
            try:
                record = _record(
                    molecule, tuple(candidates[i] for i in selected), ranks, library, text
                )
            except PeptideError:
                verification_failures += 1
                continue
            records.setdefault(to_helm(record.graph), record)
            continue
        available = full & ~(covered | skipped)
        if not available:
            continue
        if not records and molecule.GetNumAtoms() - skipped.bit_count() < best_coverage:
            continue
        choices = []
        while available:
            bit = available & -available
            atom = bit.bit_length() - 1
            options = [index for index in by_atom[atom] if not masks[index] & (covered | skipped)]
            if not options:
                skipped |= bit
            else:
                choices.append((len(options), atom, options))
            available ^= bit
        if not choices or (records and skipped):
            continue
        _, atom, options = min(choices, key=lambda item: (item[0], ranks[item[1]]))
        if not records:
            pending.append((covered, skipped | (1 << atom), selected))
        pending.extend(
            (covered | masks[index], skipped, selected + (index,)) for index in reversed(options)
        )

    if records:
        status = "ambiguous" if len(records) > 1 else ("budget_exceeded" if limits else "resolved")
        unknown = ()
    else:
        status = "budget_exceeded" if limits else "partial"
        unknown = _unidentified(molecule, best)
    return DecompositionResult(
        status=status,
        records=tuple(records.values()),
        matches=best,
        unidentified=unknown,
        source=text,
        library_digest=library.digest,
        search_exhausted=not limits,
        states=states,
        candidate_count=len(candidates),
        seconds=time.monotonic() - started,
        atom_count=molecule.GetNumAtoms(),
        limits_reached=tuple(sorted(limits)),
        verification_failures=verification_failures,
    )

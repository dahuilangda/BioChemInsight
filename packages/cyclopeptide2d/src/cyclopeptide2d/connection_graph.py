"""Lift registered monomer connections to exact atomic layout boundaries."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peptidecore import ChemicalRecord


@dataclass
class ConnectionGraph:
    units: list[dict]
    owners: dict[int, int]
    links: list[tuple[int, int]]
    crosslinks: list[tuple[int, int]]
    paths: list[tuple[list[int], bool]]
    cycle_rank: int

    @property
    def order(self):
        return [atom for unit in self.units for atom in unit["atoms"]]

    @property
    def ring(self):
        return len(self.paths) == 1 and self.paths[0][1]


def _path(molecule, start, end, support):
    parents, queue = {start: None}, deque((start,))
    while queue:
        atom = queue.popleft()
        if atom == end:
            path = []
            while atom is not None:
                path.append(atom)
                atom = parents[atom]
            return path[::-1]
        for neighbour in molecule.GetAtomWithIdx(atom).GetNeighbors():
            index = neighbour.GetIdx()
            if index in support and index not in parents:
                parents[index] = atom
                queue.append(index)
    raise ValueError("disconnected_registered_backbone")


def connection_graph(record: ChemicalRecord) -> ConnectionGraph:
    molecule = record.molecule
    instances, polymer_types = {}, {}
    for chain in record.graph.chains:
        for position, symbol in enumerate(chain.monomers, 1):
            identifier = f"{chain.identifier}:{position}"
            instances[identifier] = symbol
            polymer_types[identifier] = chain.polymer_type
    atoms = defaultdict(set)
    for index, instance in enumerate(record.atom_instances):
        if instance not in instances:
            raise ValueError("unregistered_atom_owner")
        atoms[instance].add(index)
    ports = defaultdict(dict)
    for endpoint, index in record.attachment_atoms.items():
        ports[f"{endpoint.chain}:{endpoint.position}"][endpoint.attachment] = index
    units, unit_index = [], {}
    for identifier in instances:
        sites = ports[identifier]
        if "R1" in sites and "R2" in sites:
            first, last = sites["R1"], sites["R2"]
        elif polymer_types[identifier] == "CHEM" and len(set(sites.values())) >= 2:
            first, last = sorted(set(sites.values()))[:: len(set(sites.values())) - 1]
        else:
            continue
        if first == last:
            raise ValueError("coincident_registered_backbone_ports")
        unit_index[identifier] = len(units)
        units.append(
            {"atoms": _path(molecule, first, last, atoms[identifier]), "instance": identifier}
        )
    if not units:
        raise ValueError("no_registered_backbone_units")
    adjacency = defaultdict(set)
    for edge in record.connections:
        a = f"{edge.connection.source.chain}:{edge.connection.source.position}"
        b = f"{edge.connection.target.chain}:{edge.connection.target.position}"
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(instances) - set(unit_index)
    while remaining:
        assigned = []
        for identifier in sorted(remaining):
            neighbours = adjacency[identifier]
            if len(neighbours) == 1 and next(iter(neighbours)) in unit_index:
                unit_index[identifier] = unit_index[next(iter(neighbours))]
                assigned.append(identifier)
        if not assigned:
            raise ValueError("unresolved_registered_layout_component")
        remaining.difference_update(assigned)
    owners = {
        atom: unit_index[identifier] for identifier, support in atoms.items() for atom in support
    }
    links, crosslinks = [], []
    registered = set()
    closed_chains = set()
    chain_lengths = {chain.identifier: len(chain.monomers) for chain in record.graph.chains}
    for edge in record.connections:
        a, b = edge.source_atom, edge.target_atom
        registered.add(tuple(sorted((a, b))))
        endpoints = sorted(
            (edge.connection.source, edge.connection.target),
            key=lambda endpoint: endpoint.attachment,
        )
        first, second = endpoints
        sequential = (
            first.chain == second.chain
            and first.attachment == "R1"
            and second.attachment == "R2"
            and first.position == second.position + 1
        )
        closed = (
            first.chain == second.chain
            and first.attachment == "R1"
            and second.attachment == "R2"
            and first.position == 1
            and second.position == chain_lengths[first.chain]
        )
        if closed:
            closed_chains.add(first.chain)
        if owners[a] != owners[b]:
            links.append((a, b))
            if not sequential and not closed:
                crosslinks.append((a, b))
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if owners[a] != owners[b] and tuple(sorted((a, b))) not in registered:
            raise ValueError("unregistered_inter_monomer_bond")
    paths = []
    for chain in record.graph.chains:
        indices = []
        for position in range(1, len(chain.monomers) + 1):
            index = unit_index[f"{chain.identifier}:{position}"]
            if index not in indices:
                indices.append(index)
        path = [atom for index in indices for atom in units[index]["atoms"]]
        paths.append((path, chain.identifier in closed_chains))
    seen, components = set(), 0
    for start in instances:
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(adjacency[node] - seen)
    return ConnectionGraph(
        units,
        owners,
        links,
        crosslinks,
        paths,
        len(record.connections) - len(instances) + components,
    )

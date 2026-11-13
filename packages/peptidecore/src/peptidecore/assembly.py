"""Assemble monomers through explicit attachment pairs."""

from __future__ import annotations

from rdkit import Chem

from .errors import PeptideError
from .library import MonomerLibrary, core_library, parse_smiles
from .model import AtomicConnection, ChemicalRecord, Endpoint, PeptideGraph


def assemble(
    graph: PeptideGraph,
    *,
    library: MonomerLibrary | None = None,
    source: str = "",
    source_format: str = "graph",
) -> ChemicalRecord:
    library = core_library() if library is None else library
    connections = graph.resolved_connections(library)
    labels = {}
    for number, connection in enumerate(connections, 1):
        labels[connection.source] = number
        labels[connection.target] = number
    next_label = len(connections) + 1
    molecules = []
    caps = []
    anchors = {}
    serial = 0
    for chain in graph.chains:
        for position, symbol in enumerate(chain.monomers, 1):
            monomer = library.get(symbol, chain.polymer_type)
            molecule = monomer.template()
            instance = f"{chain.identifier}:{position}"
            for atom in molecule.GetAtoms():
                atom.SetProp("_peptide_instance", instance)
                atom.SetIntProp("_peptide_origin", atom.GetIdx())
                if atom.GetAtomicNum():
                    atom.SetIntProp("_peptide_serial", serial)
                    serial += 1
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum():
                    continue
                attachment = f"R{atom.GetAtomMapNum()}"
                endpoint = Endpoint(chain.identifier, position, attachment)
                anchor = atom.GetNeighbors()[0]
                anchors[endpoint] = anchor.GetIntProp("_peptide_serial")
                label = labels.get(endpoint)
                if label is None:
                    label = next_label
                    next_label += 1
                    cap = parse_smiles(monomer.attachment(attachment).cap_smiles)
                    for cap_atom in cap.GetAtoms():
                        if cap_atom.GetAtomicNum() == 0:
                            cap_atom.SetAtomMapNum(label)
                        else:
                            cap_atom.SetProp("_peptide_instance", instance)
                            cap_atom.SetIntProp("_peptide_origin", atom.GetIdx())
                            cap_atom.SetIntProp("_peptide_serial", serial)
                            serial += 1
                    caps.append(cap)
                atom.SetAtomMapNum(label)
            molecules.append(molecule)
    combined = Chem.Mol()
    for molecule in (*molecules, *caps):
        combined = Chem.CombineMols(combined, molecule)
    try:
        molecule = Chem.RemoveHs(Chem.molzip(combined))
        Chem.SanitizeMol(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    except (ValueError, RuntimeError) as error:
        raise PeptideError("invalid_assembled_structure", str(error)) from error
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise PeptideError("radical_assembled_structure", "Connections leave an incomplete valence")
    atom_by_serial = {a.GetIntProp("_peptide_serial"): a.GetIdx() for a in molecule.GetAtoms()}
    attachment_labels = {}
    for endpoint, serial in anchors.items():
        if serial not in atom_by_serial:
            raise PeptideError("attachment_anchor_removed", str(endpoint))
        attachment_labels.setdefault(atom_by_serial[serial], []).append(endpoint.attachment)
    for index, labels_at_atom in attachment_labels.items():
        molecule.GetAtomWithIdx(index).SetProp(
            "_peptide_attachment_labels", ",".join(sorted(labels_at_atom))
        )
    atomic_connections = tuple(
        AtomicConnection(
            edge, atom_by_serial[anchors[edge.source]], atom_by_serial[anchors[edge.target]]
        )
        for edge in connections
    )
    return ChemicalRecord(
        graph=graph,
        library_name=library.name,
        library_digest=library.digest,
        atom_instances=tuple(a.GetProp("_peptide_instance") for a in molecule.GetAtoms()),
        atom_origins=tuple(a.GetIntProp("_peptide_origin") for a in molecule.GetAtoms()),
        connections=atomic_connections,
        source=source,
        source_format=source_format,
        _molecule=molecule,
    )

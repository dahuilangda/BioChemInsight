"""Actual RDKit drawing cues -> strict predicted-graph assembly round trips."""
from dataclasses import replace
import copy

import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor

from training.ocsr_reboot.graph import (
    AtomState, ChemicalBond, WedgeBond, DoubleBondStereo, assemble_raw_graph,
)
from training.ocsr_reboot.protocol import canonical_smiles


def drawing(smiles):
    options = Chem.SmilesParserParams()
    options.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, options)
    assert molecule is not None
    rdDepictor.Compute2DCoords(molecule)
    Chem.WedgeMolBonds(molecule, molecule.GetConformer())
    block = Chem.MolToMolBlock(molecule, includeStereo=True, kekulize=False)
    atoms = [AtomState(atom.GetSymbol(), charge=atom.GetFormalCharge(), isotope=atom.GetIsotope(),
                       explicit_h=atom.GetNumExplicitHs(), no_implicit=atom.GetNoImplicit(),
                       radical_electrons=atom.GetNumRadicalElectrons(), aromatic=atom.GetIsAromatic())
             for atom in molecule.GetAtoms()]
    conformer = molecule.GetConformer()
    coords = [(conformer.GetAtomPosition(i).x, -conformer.GetAtomPosition(i).y) for i in range(len(atoms))]
    kinds = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
             Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    edges = [ChemicalBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), kinds[bond.GetBondType()])
             for bond in molecule.GetBonds()]
    # Read emitted Molfile directions. The parser may consume/clear these
    # markers when recovering atom chirality, so do not use a parsed bond dir.
    wedges = []
    for line in block.splitlines()[4 + len(atoms):4 + len(atoms) + len(edges)]:
        first, second, _, stereo = int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9]), int(line[9:12])
        if stereo in (1, 6): wedges.append(WedgeBond(first, second, 'solid' if stereo == 1 else 'hashed'))
    double = []
    for bond in molecule.GetBonds():
        if bond.GetStereo() not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            continue
        left, right = bond.GetStereoAtoms()
        relation = 'same' if bond.GetStereo() in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS) else 'opposite'
        double.append(DoubleBondStereo(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), left, right, relation))
    return molecule, atoms, coords, edges, wedges, double


@pytest.mark.parametrize('smiles', [
    'CCO', 'CC(=O)N', 'C#N',
    'N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O',
    'F[C@](Cl)(Br)I', 'F[C@@](Cl)(Br)I',
    'O=C(O)[C@@H](O)[C@@H](O)C(=O)O',
    'O=C(O)[C@@H](O)[C@H](O)C(=O)O',
    'N[C@@H](CS)C(=O)N[C@H](C)C(=O)O',
    'C[C@H]1CC[C@@H](O)CC1',
    'O=C1N[C@@H](C)C(=O)N[C@H](CO)C(=O)NCC(=O)NC1',
    'F/C=C/F', 'F/C=C\\F', 'C/C=C/C=C\\C', 'C/C=N/O',
    '[NH3+][C@@H]([13CH3])C(=O)[O-].[Na+]',
    '[2H]O[2H]', '[H]O[H]',
    'c1ccc2[nH]ccc2c1', 'c1nc[nH]c1',
    'C[S@](=O)CC', 'C[S@@](=O)CC', 'C[P@](=O)(O)CC',
    'N[C@@H](C)/C=C/F',
    'C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O',
])
def test_actual_drawing_round_trip_preserves_entire_molecule(smiles):
    original, atoms, coords, edges, wedges, double = drawing(smiles)
    result = assemble_raw_graph(atoms, coords, edges, wedges=wedges, double_stereo=double)
    assert result.ok, (smiles, result.error_code, result.error)
    expected = Chem.MolToSmiles(original, canonical=True, isomericSmiles=True)
    assert canonical_smiles(result.smiles) == canonical_smiles(expected)
    assert result.num_atoms == original.GetNumAtoms()
    assert result.num_bonds == original.GetNumBonds()
    # Molfile exports must agree too; a correct SMILES cannot hide incorrect
    # y-axis conventions, lost wedges, or contradictory double-bond geometry.
    recovered = Chem.MolFromMolBlock(result.molblock, sanitize=True, removeHs=False)
    assert recovered is not None
    assert recovered.GetNumAtoms() == original.GetNumAtoms()
    assert canonical_smiles(Chem.MolToSmiles(recovered, isomericSmiles=True)) == canonical_smiles(expected)


def test_reordered_atoms_edges_and_noncanonical_wedge_bond_orientation():
    original, _, _, _, _, _ = drawing('N[C@@H](CS)C(=O)N[C@H](C)C(=O)O')
    reordered = Chem.RenumberAtoms(original, list(range(original.GetNumAtoms()))[::-1])
    _, atoms, coords, edges, wedges, double = drawing(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    # Chemical edge endpoint order must not reinterpret the explicit apex.
    edges = [ChemicalBond(edge.second, edge.first, edge.kind) for edge in edges[::-1]]
    result = assemble_raw_graph(atoms, coords, edges, wedges=wedges, double_stereo=double)
    assert result.ok, result.error
    assert canonical_smiles(result.smiles) == canonical_smiles(Chem.MolToSmiles(original, isomericSmiles=True))


def test_missing_tetrahedral_cues_remain_unspecified_instead_of_guessing():
    _, atoms, coords, edges, wedges, _ = drawing('N[C@@H](C)C(=O)O')
    assert wedges
    result = assemble_raw_graph(atoms, coords, edges)
    assert result.ok
    assert '@' not in result.smiles
    assert result.unassigned_stereo
    assert canonical_smiles(result.smiles) != canonical_smiles('N[C@@H](C)C(=O)O')


def test_mirroring_coordinates_changes_chirality_if_wedge_cues_are_unchanged():
    _, atoms, coords, edges, wedges, _ = drawing('N[C@@H](C)C(=O)O')
    ordinary = assemble_raw_graph(atoms, coords, edges, wedges=wedges)
    mirrored = assemble_raw_graph(atoms, [(x, -y) for x, y in coords], edges, wedges=wedges)
    assert ordinary.ok and mirrored.ok
    assert canonical_smiles(ordinary.smiles) != canonical_smiles(mirrored.smiles)


def test_double_stereo_is_explicit_and_missing_information_is_not_invented():
    _, atoms, coords, edges, _, double = drawing('F/C=C/F')
    missing = assemble_raw_graph(atoms, coords, edges)
    geometry = assemble_raw_graph(atoms, coords, edges, double_stereo=[replace(double[0], relation='geometry')])
    unknown = assemble_raw_graph(atoms, coords, edges, double_stereo=[replace(double[0], relation='unknown')])
    assert missing.ok and geometry.ok and unknown.ok
    assert '/' not in missing.smiles and '\\' not in missing.smiles
    assert '/' not in unknown.smiles and '\\' not in unknown.smiles
    assert missing.unassigned_stereo and unknown.unassigned_stereo
    for result in (missing, unknown):
        recovered = Chem.MolFromMolBlock(result.molblock, sanitize=True, removeHs=False)
        assert recovered is not None
        assert canonical_smiles(Chem.MolToSmiles(recovered, isomericSmiles=True)) == canonical_smiles('FC=CF')
    assert canonical_smiles(geometry.smiles) == canonical_smiles('F/C=C/F')
    contradictory = assemble_raw_graph(atoms, coords, edges, double_stereo=[replace(double[0], relation='same')])
    assert not contradictory.ok and contradictory.error_code == 'double_geometry_conflict'


def test_double_substituent_reference_order_is_not_cip_or_bond_insertion_order():
    _, atoms, coords, edges, _, double = drawing('F/C(Cl)=C(Br)/I')
    spec = double[0]
    reversed_spec = DoubleBondStereo(spec.second, spec.first, spec.second_substituent, spec.first_substituent, spec.relation)
    result = assemble_raw_graph(atoms, coords, edges, double_stereo=[reversed_spec])
    assert result.ok, result.error
    assert canonical_smiles(result.smiles) == canonical_smiles('F/C(Cl)=C(Br)/I')


def test_invalid_graph_is_not_repaired_and_caller_inputs_are_unchanged():
    atoms = [AtomState('C')] + [AtomState('F')] * 5
    coords = [(float(index), float(index % 2)) for index in range(6)]
    edges = [ChemicalBond(0, index, 1) for index in range(1, 6)]
    before = copy.deepcopy((atoms, coords, edges))
    result = assemble_raw_graph(atoms, coords, edges)
    assert not result.ok
    assert result.smiles is None and result.molblock is None
    assert (atoms, coords, edges) == before


def test_missing_aromatic_h_does_not_trigger_enumeration_or_repair():
    _, atoms, coords, edges, _, _ = drawing('c1ccc2[nH]ccc2c1')
    atoms = [replace(atom, explicit_h=0) if atom.symbol == 'N' else atom for atom in atoms]
    result = assemble_raw_graph(atoms, coords, edges)
    assert not result.ok and result.smiles is None


def test_invalid_stereo_and_coordinate_contracts_fail_explicitly():
    _, atoms, coords, edges, wedges, _ = drawing('N[C@@H](C)C(=O)O')
    for kwargs in [
        {'wedges': [replace(wedges[0], other=99)]},
        {'wedges': [wedges[0], wedges[0]]},
        {'coordinate_system': 'xy_up'},
    ]:
        result = assemble_raw_graph(atoms, coords, edges, **kwargs)
        assert not result.ok and result.error_code
    result = assemble_raw_graph(atoms, [(float('nan'), 0)] + coords[1:], edges)
    assert not result.ok and result.error_code == 'invalid_coordinates'
    result = assemble_raw_graph(atoms, coords, edges + [edges[0]])
    assert not result.ok and result.error_code == 'duplicate_bond'


def test_element_states_cannot_be_silently_corrected_to_make_a_valid_molecule():
    # Neutral N with two double-bond oxygens is not an invitation to rewrite
    # its formal charges into a nitro group during sanitization.
    result = assemble_raw_graph([AtomState('N'), AtomState('O'), AtomState('O'), AtomState('C')],
                                [(0, 0), (-1, 1), (1, 1), (0, -1)],
                                [ChemicalBond(0, 1, 2), ChemicalBond(0, 2, 2), ChemicalBond(0, 3, 1)])
    assert not result.ok
    # A three-bond carbon declared to have neither H nor radicals cannot
    # silently acquire a radical as a "valid" prediction.
    result = assemble_raw_graph([AtomState('C', explicit_h=0, no_implicit=True)] + [AtomState('F')] * 3,
                                [(0, 0), (-1, 1), (1, 1), (0, -1)],
                                [ChemicalBond(0, index, 1) for index in range(1, 4)])
    assert not result.ok and result.error_code == 'atom_state_changed'


def test_nonstereogenic_apex_and_degenerate_cues_cannot_create_chirality():
    _, atoms, coords, edges, _, _ = drawing('CC(C)O')
    result = assemble_raw_graph(atoms, coords, edges, wedges=[WedgeBond(1, 3, 'solid')])
    assert not result.ok and result.error_code in ('invalid_stereo_apex', 'unresolved_wedge')
    _, atoms, coords, edges, wedges, _ = drawing('N[C@@H](C)C(=O)O')
    result = assemble_raw_graph(atoms, [(float(index), 0.) for index in range(len(atoms))], edges, wedges=wedges)
    assert not result.ok and result.error_code in ('unresolved_wedge', 'conflicting_wedges')

"""Conservative chemical-graph supervision aligned to unchanged SMILES tokens.

Initial support is literal SMILES only. Drawing coordinates and wedge direction
are not inferred from chemical SMILES. Extended semantics require a later,
explicit mapping implementation rather than dropping their annotations.
"""
from rdkit import Chem, rdBase


class AlignmentError(ValueError):
    pass


def align_smiles(smiles, tokenizer):
    if not isinstance(smiles, str) or not smiles or any(c.isspace() for c in smiles) or '<sep>' in smiles or '|' in smiles:
        raise AlignmentError('Only literal, nonempty SMILES are supported')
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, parameters)
    if molecule is None:
        raise AlignmentError('Invalid chemical graph')
    try:
        tokens, indices = tokenizer.smiles_to_sequence(smiles)
        decoded = tokenizer.sequence_to_smiles(tokens[1:])
    except (ValueError, KeyError) as error:
        raise AlignmentError(f'Tokenization failed: {error}') from error
    if decoded['smiles'] != smiles or decoded['indices'] != [index - 1 for index in indices]:
        raise AlignmentError('Token round trip or SOS-relative query positions disagree')
    if len(indices) != molecule.GetNumAtoms() or len(decoded['atoms']) != len(indices):
        raise AlignmentError('Lexical atoms and parsed graph atoms differ')
    atoms = []
    for index, (atom, text, query) in enumerate(zip(molecule.GetAtoms(), decoded['atoms'], indices, strict=True)):
        with rdBase.BlockLogs():
            isolated = Chem.MolFromSmiles(text, sanitize=False)
        if isolated is None or isolated.GetNumAtoms() != 1:
            raise AlignmentError(f'Cannot validate atom token {index}: {text}')
        lexical = isolated.GetAtomWithIdx(0)
        for getter in ('GetAtomicNum', 'GetIsotope', 'GetFormalCharge', 'GetAtomMapNum'):
            if getattr(atom, getter)() != getattr(lexical, getter)():
                raise AlignmentError(f'Atom property/order mismatch at {index}: {getter}')
        start = query - len(text) + 1
        if ''.join(tokenizer.itos[token] for token in tokens[start:query + 1]) != text:
            raise AlignmentError(f'Atom span mismatch at {index}')
        atoms.append({'atom_id': index, 'token_start_with_sos': start,
                      'token_end_with_sos_inclusive': query, 'atom_token': text,
                      'atomic_number': atom.GetAtomicNum(), 'isotope': atom.GetIsotope(),
                      'formal_charge': atom.GetFormalCharge(), 'aromatic': atom.GetIsAromatic(),
                      'atom_map': atom.GetAtomMapNum(), 'chiral_tag': str(atom.GetChiralTag()),
                      'explicit_hydrogens': atom.GetNumExplicitHs(), 'no_implicit': atom.GetNoImplicit(),
                      'radical_electrons': atom.GetNumRadicalElectrons()})
    bonds = [{'begin': bond.GetBeginAtomIdx(), 'end': bond.GetEndAtomIdx(),
              'chemical_type': str(bond.GetBondType()), 'aromatic': bond.GetIsAromatic(),
              'stereo': str(bond.GetStereo()), 'smiles_bond_direction': str(bond.GetBondDir()), 'stereo_atoms': list(bond.GetStereoAtoms())}
             for bond in molecule.GetBonds()]
    # Keep original atom/bond stereo references.
    # The independent lexical checks above establish the token-to-atom mapping;
    # graph identity remains tied to the original parsed target, not a repair.
    result = {'schema_version': 1, 'supervision_level': 'chemical_graph_only',
            'smiles': smiles, 'token_ids': tokens, 'atom_indices': indices,
            'atoms': atoms, 'bonds': bonds, 'coordinate_targets': None,
            'drawing_bond_direction_targets': None,
            'canonical_isomeric_smiles': Chem.MolToSmiles(molecule, isomericSmiles=True),
            'scope': 'Literal SMILES lexical atom order; chiral tags are traversal-relative. No drawing alignment or rendered wedge labels.'}

    rebuilt = rebuild_molecule(result)
    if Chem.MolToSmiles(rebuilt, isomericSmiles=True) != result['canonical_isomeric_smiles']:
        raise AlignmentError('Exported graph does not preserve complete isomeric identity')
    return result


def rebuild_molecule(annotation):
    """Reconstruct solely from exported chemical atoms, bonds and stereo."""
    graph = Chem.RWMol()
    for record in annotation['atoms']:
        atom = Chem.Atom(record['atomic_number'])
        atom.SetIsotope(record['isotope'])
        atom.SetFormalCharge(record['formal_charge'])
        atom.SetIsAromatic(record['aromatic'])
        atom.SetAtomMapNum(record['atom_map'])
        atom.SetNumExplicitHs(record['explicit_hydrogens'])
        atom.SetNoImplicit(record['no_implicit'])
        atom.SetNumRadicalElectrons(record['radical_electrons'])
        atom.SetChiralTag(Chem.ChiralType.names[record['chiral_tag']])
        if graph.AddAtom(atom) != record['atom_id']:
            raise AlignmentError('Noncontiguous exported atom identifiers')
    for record in annotation['bonds']:
        graph.AddBond(record['begin'], record['end'], Chem.BondType.names[record['chemical_type']])
        graph.GetBondBetweenAtoms(record['begin'], record['end']).SetIsAromatic(record['aromatic'])
    molecule = graph.GetMol()
    for record in annotation['bonds']:
        bond = molecule.GetBondBetweenAtoms(record['begin'], record['end'])
        if record['stereo_atoms']:
            bond.SetStereoAtoms(*record['stereo_atoms'])
        bond.SetStereo(Chem.BondStereo.names[record['stereo']])
        bond.SetBondDir(Chem.BondDir.names[record['smiles_bond_direction']])
    Chem.SanitizeMol(molecule)
    return molecule

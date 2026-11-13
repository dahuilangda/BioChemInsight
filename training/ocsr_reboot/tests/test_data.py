"""Independent data-contract checks using actual drawings and tokenizer.

No checkpoints, GPUs or source corpus are required. Coordinates are checked
against the visible ink bounds, not by replaying the implementation's transform.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from training.ocsr_reboot.data import digest, draw_small, exact_key, peptide_record, prepare_sample


@pytest.fixture
def tokenizer():
    from utils.MolNexTR.tokenization import CharTokenizer
    vocab = Path(__file__).resolve().parents[3] / 'utils' / 'MolNexTR' / 'vocab' / 'vocab_chars.json'
    return CharTokenizer(input_size=64, path=str(vocab), sep_xy=True)


def drawing_source(tmp_path, smiles='CCNO'):
    """A self-contained record in the actual peptide artifact directory layout."""
    target = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    molecule = Chem.MolFromSmiles(target)
    rdDepictor.Compute2DCoords(molecule)
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, kekulize=False, addChiralHs=False)
    root = tmp_path / 'render'
    directory = root / 'images' / 'example' / 'serif_clean'
    directory.mkdir(parents=True)
    image = directory / 'input_768.png'
    drawer = rdMolDraw2D.MolDraw2DCairo(768, 768)
    drawer.drawOptions().prepareMolsBeforeDrawing = False
    drawer.DrawMolecule(molecule)
    coords = [[drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y] for i in range(molecule.GetNumAtoms())]
    drawer.FinishDrawing()
    image.write_bytes(drawer.GetDrawingText())
    Chem.MolToMolFile(molecule, str(directory.parent / 'drawing.mol'))
    (root / 'example.json').write_text(json.dumps({
        'images': [{'inputs': [{'file': str(image), 'atom_positions': coords}]}],
    }))
    row = {'record_key': 'source-example', 'file_path': str(image), 'SMILES': target,
           'component': 'chemical-family-example', 'peptide_stratum': 'linear_short',
           'perm': list(range(molecule.GetNumAtoms())), 'image_sha256': digest(image)}
    return row, molecule, coords


def test_correct_permutation_preserves_target_pixel_order(tmp_path):
    row, molecule, coords = drawing_source(tmp_path)
    record = peptide_record(row, 'train')
    assert record['coords_pixels'] == coords
    assert record['source_order'] == list(range(molecule.GetNumAtoms()))
    assert record['SMILES'] == row['SMILES']
    assert record['drawing_mol_sha256'] == digest(Path(row['file_path']).parents[1] / 'drawing.mol')


def test_peptide_import_rejects_changed_admitted_image(tmp_path):
    row, _, _ = drawing_source(tmp_path)
    # A changed but still readable image must not be silently re-admitted.
    path = Path(row['file_path'])
    changed = Image.open(path).convert('RGB')
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(path)
    assert digest(path) != row['image_sha256']
    with pytest.raises(ValueError, match='hash|changed|image'):
        peptide_record(row, 'train')


def test_wrong_element_permutation_is_rejected_even_though_canonical_identity_is_unchanged(tmp_path):
    row, molecule, _ = drawing_source(tmp_path)
    broken = copy.deepcopy(row)
    broken['perm'][0], broken['perm'][-1] = broken['perm'][-1], broken['perm'][0]
    # This is why canonical identity cannot certify coordinate correspondence.
    assert exact_key(Chem.MolToSmiles(Chem.RenumberAtoms(molecule, broken['perm']))) == exact_key(row['SMILES'])
    with pytest.raises(ValueError, match='attributes|order|mapping'):
        peptide_record(broken, 'train')


def test_wrong_same_element_permutation_is_rejected_by_bond_correspondence(tmp_path):
    row, molecule, _ = drawing_source(tmp_path, 'CCCCO')
    broken = copy.deepcopy(row)
    broken['perm'][1], broken['perm'][2] = broken['perm'][2], broken['perm'][1]
    assert exact_key(Chem.MolToSmiles(Chem.RenumberAtoms(molecule, broken['perm']))) == exact_key(row['SMILES'])
    with pytest.raises(ValueError, match='bonds|order|mapping'):
        peptide_record(broken, 'train')


def test_stereo_inverting_graph_automorphism_is_not_a_valid_coordinate_map(tmp_path):
    row, molecule, _ = drawing_source(tmp_path, 'C[C@H](O)[C@H](O)C')
    # Canonical target has two equivalent graph halves, but opposite CIP centers.
    target = Chem.MolFromSmiles(row['SMILES'])
    assert [atom.GetAtomicNum() for atom in target.GetAtoms()] == [6, 6, 8, 6, 6, 8]
    broken = copy.deepcopy(row)
    broken['perm'] = [4, 3, 5, 1, 0, 2]
    assert exact_key(Chem.MolToSmiles(Chem.RenumberAtoms(molecule, broken['perm']))) == exact_key(row['SMILES'])
    # Atomic attributes and undirected topology alone cannot reject this map.
    with pytest.raises(ValueError, match='stereo|chirality|attributes|mapping'):
        peptide_record(broken, 'train')


@pytest.mark.parametrize('smiles', [
    'CCO', 'C[C@H](O)c1ccccc1', '[13CH3][C@@H](N)C(=O)O',
    'C/C=C/C(=O)[O-].[Na+]',
])
def test_actual_tokenizer_and_image_preserve_complete_target(tmp_path, tokenizer, smiles):
    record = draw_small(smiles, tmp_path, 'train', 0)
    sample = prepare_sample(record, tokenizer, input_size=384)
    labels = sample['refs']['chartok_coords'][0][0].tolist()
    assert labels[0] == 1 and labels[-1] == 2 and 3 not in labels
    assert exact_key(tokenizer.sequence_to_smiles(labels[1:])['smiles']) == exact_key(smiles)
    assert sample['image'].shape == (3, 384, 384)
    mol = Chem.MolFromSmiles(record['SMILES'])
    edges = sample['refs']['edges'][0].numpy()
    assert edges.shape == (mol.GetNumAtoms(), mol.GetNumAtoms())
    assert (np.diag(edges) == -100).all()
    assert int((np.triu(edges, k=1) > 0).sum()) == mol.GetNumBonds()


def test_image_crop_and_coordinates_use_the_same_transform(tmp_path, tokenizer):
    record = draw_small('CCCO', tmp_path, 'train', 0)
    original = np.asarray(Image.open(record['file_path']).convert('RGB'))
    ink_y, ink_x = np.nonzero(np.any(original != 255, axis=-1))
    left, right = ink_x.min(), ink_x.max() + 1
    top, bottom = ink_y.min(), ink_y.max() + 1
    # Native clean preprocessing crops to ink and adds 50 source pixels per side.
    expected = (np.asarray(record['coords_pixels']) - [left, top] + 50) / [right - left + 100, bottom - top + 100]
    sample = prepare_sample(record, tokenizer, input_size=384)
    np.testing.assert_allclose(sample['coords_normalized'], expected, atol=1e-6)
    assert not np.allclose(expected, np.asarray(record['coords_pixels']) / original.shape[1::-1])
    decoded = tokenizer.sequence_to_smiles(sample['refs']['chartok_coords'][0][0].tolist()[1:])
    np.testing.assert_allclose(decoded['coords'], expected, atol=.5 / 63 + 1e-6)
    # Independently inspect transformed image ink rather than trusting keypoints.
    channel = sample['image'][0].numpy() * .229 + .485
    out_y, out_x = np.nonzero(channel < .95)
    expected_left = 384 * 50 / (right - left + 100)
    expected_top = 384 * 50 / (bottom - top + 100)
    assert abs(out_x.min() - expected_left) <= 2
    assert abs(out_y.min() - expected_top) <= 2


def test_missing_coordinate_is_rejected_before_tokenizer_can_randomly_fill_it(tmp_path):
    record = draw_small('CCO', tmp_path, 'train', 0)
    record['coords_pixels'].pop()
    class MustNotRun:
        def smiles_to_sequence(self, *args, **kwargs):
            raise AssertionError('Tokenizer reached despite missing coordinate')
    with pytest.raises(ValueError, match='atom-coordinate'):
        prepare_sample(record, MustNotRun(), input_size=384)


def test_unknown_target_token_is_rejected(tmp_path, tokenizer):
    record = draw_small('CCO', tmp_path, 'train', 0)
    class UnknownToken:
        def smiles_to_sequence(self, *args, **kwargs):
            labels, indices = tokenizer.smiles_to_sequence(*args, **kwargs)
            labels[1] = 3
            return labels, indices
    with pytest.raises(ValueError, match='tokenizer coverage'):
        prepare_sample(record, UnknownToken(), input_size=384)


def test_overlong_target_is_rejected_instead_of_silently_removing_eos(tmp_path, tokenizer):
    record = draw_small('CCCCCO', tmp_path, 'train', 0)
    with pytest.raises(ValueError, match='decode budget'):
        prepare_sample(record, tokenizer, input_size=384, max_tokens=5)


def test_changed_image_is_rejected(tmp_path, tokenizer):
    record = draw_small('CCO', tmp_path, 'train', 0)
    Path(record['file_path']).write_bytes(b'changed source image')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_sample(record, tokenizer, input_size=384)


@pytest.mark.parametrize('coordinate', [[-1, 2], [800, 2], [float('nan'), 2]])
def test_invalid_coordinates_are_rejected(tmp_path, tokenizer, coordinate):
    record = draw_small('CCO', tmp_path, 'train', 0)
    record['coords_pixels'][0] = coordinate
    with pytest.raises(ValueError, match='coordinate|outside'):
        prepare_sample(record, tokenizer, input_size=384)

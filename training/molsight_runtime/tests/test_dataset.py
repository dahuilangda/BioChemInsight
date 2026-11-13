from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from molsight.dataset import MolParser7MDataset, PubchemDataset, USPTODataset, collate_fn
from molsight.tokenizer import CharTokenizer
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


@pytest.fixture
def fixture(tmp_path):
    molecule = Chem.MolFromSmiles('NCCO')
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(200, 160)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    path = tmp_path / 'structure.png'
    path.write_bytes(drawer.GetDrawingText())
    args = SimpleNamespace(input_size=128, encoder='efficientvit', augment=False,
                           debug=False, data_path=str(tmp_path), max_len=128, mask_ratio=0,
                           mol_augment=False, default_option=False, shuffle_nodes=False,
                           include_condensed=False)
    tokenizer = CharTokenizer('training/molsight_runtime/vocab/vocab_chars.json')
    return args, tokenizer, path


def test_unaugmented_images_are_deterministic_and_retain_targets(fixture):
    args, tokenizer, path = fixture
    dataset = USPTODataset(args, pd.DataFrame({'SMILES': ['NCCO'], 'file_path': [path.name]}), tokenizer)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first['image'], second['image'])
    assert first['smiles'] == 'NCCO'
    assert first['coords'].shape == (4, 2)
    assert first['coords'].eq(-100).all()


def test_missing_image_fails_without_a_blank_training_sample(fixture):
    args, tokenizer, path = fixture
    dataset = USPTODataset(args, pd.DataFrame({'SMILES': ['NCCO'], 'file_path': ['missing.png']}), tokenizer)
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_missing_parquet_image_is_rejected(fixture):
    args, tokenizer, _ = fixture
    dataset = MolParser7MDataset(args, [{'image': None, 'SMILES': 'NCCO'}], tokenizer)
    with pytest.raises(ValueError, match='Missing image'):
        dataset[0]


def test_parquet_samples_include_policy_targets(fixture):
    args, tokenizer, path = fixture
    with Image.open(path) as image:
        dataset = MolParser7MDataset(args, [{'image': image.copy(), 'SMILES': 'NCCO'}], tokenizer)
    assert dataset[0]['smiles'] == 'NCCO'
    assert torch.equal(dataset[0]['image'], dataset[0]['image'])


def test_failed_dynamic_rendering_does_not_create_a_training_sample(fixture, monkeypatch):
    args, tokenizer, _ = fixture
    monkeypatch.setattr('molsight.dataset.generate_indigo_image', lambda *a, **kw: (None, None, {}, False))
    dataset = PubchemDataset(args, pd.DataFrame({'SMILES': ['NCCO']}), tokenizer)
    with pytest.raises(ValueError, match='rendering failed'):
        dataset[0]


def test_collation_rejects_annotation_misalignment_instead_of_truncating(fixture):
    args, tokenizer, path = fixture
    dataset = USPTODataset(args, pd.DataFrame({'SMILES': ['NCCO'], 'file_path': [str(path)]}), tokenizer)
    sample = dataset[0]
    collated = collate_fn([sample, sample])
    assert collated['image'].shape == (2, 3, 128, 128)
    assert collated['smiles'] == ['NCCO', 'NCCO']
    sample['coords'] = sample['coords'][:2]
    with pytest.raises(ValueError, match='not aligned'):
        collate_fn([sample])

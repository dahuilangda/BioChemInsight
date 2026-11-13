import hashlib

import pytest

from training.ocsr_reboot.protocol import (
    EvaluationTarget, ImageInput, Prediction, ProtocolError,
    assert_disjoint_splits, assert_image_only_payload, canonical_smiles,
    image_only_input, score_predictions,
)


def record(key, split, smiles='CCO', component=None, image_hash=None, **extra):
    return {'record_key': key, 'split': split, 'SMILES': smiles,
            'component': component or key, 'image_sha256': image_hash or key,
            'file_path': '/images/' + key + '.png', **extra}


def test_split_rejects_same_molecule_in_another_traversal():
    with pytest.raises(ProtocolError, match='smiles'):
        assert_disjoint_splits([record('a', 'train', 'CCO'), record('b', 'validation', 'OCC')])


@pytest.mark.parametrize('field', ['component', 'image_sha256', 'family_id', 'composition_id', 'publication_family'])
def test_split_family_guards(field):
    first, second = record('a', 'train'), record('b', 'test', 'CCN')
    first[field] = second[field] = 'shared'
    with pytest.raises(ProtocolError, match=field):
        assert_disjoint_splits([first, second])


def test_split_accepts_multiple_views_in_the_same_split_and_counts_entities():
    audit = assert_disjoint_splits([
        record('a', 'train'), record('b', 'train', 'OCC', component='a'),
        record('c', 'validation', 'CCN'),
    ])
    assert audit['records'] == 3
    assert audit['unique_smiles'] == {'train': 1, 'validation': 1}
    assert audit['cross_split_overlaps'] == 0


@pytest.mark.parametrize('left,right', [
    ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O'),
    ('N[C@@H](C)C(=O)O', 'NC(C)C(=O)O'),
    ('[13CH3]CO', 'CCO'), ('[NH4+]', 'N'),
    ('CCO.Cl', 'CCO'), ('F/C=C/F', 'F/C=C\\F'),
])
def test_identity_preserves_complete_chemistry(left, right):
    assert canonical_smiles(left) != canonical_smiles(right)


@pytest.mark.parametrize('text', ['', 'CCO compound-name', '*CCO', 'C1CC'])
def test_invalid_or_unresolved_targets_are_rejected(text):
    with pytest.raises(ProtocolError): canonical_smiles(text)


def test_recognition_request_does_not_expose_gold(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'image bytes')
    source = record('a', 'test', 'CCO')
    source.update(file_path=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  atom_positions=[[.2, .4]], n_atoms=3, targets=[[1]])
    request = image_only_input(source)
    request.verify_file()
    assert set(request.as_request()) == {'record_key', 'image_path', 'image_sha256', 'input_size', 'model_input_preprocessed'}
    for forbidden in ('SMILES', 'smiles', 'n_atoms', 'atom_positions', 'points', 'targets', 'gold'):
        with pytest.raises(ProtocolError, match='Oracle'):
            assert_image_only_payload({**request.as_request(), forbidden: source.get(forbidden)})
    image.write_bytes(b'changed')
    with pytest.raises(ProtocolError, match='changed'): request.verify_file()


def test_missing_failed_unterminated_and_repaired_results_keep_the_denominator():
    targets = [EvaluationTarget(str(index), 'CCO') for index in range(5)]
    result = score_predictions(targets, [
        Prediction('0', 'OCC', terminated=True),
        Prediction('1', 'CCO', terminated=False),
        Prediction('2', 'CCO', error='inference_failed'),
        Prediction('3', 'C1CC', terminated=True, repaired_smiles='CCO'),
    ])
    assert result['n'] == result['raw']['denominator'] == result['repaired']['denominator'] == 5
    assert result['raw']['resolved_exact'] == 1
    assert result['repaired']['resolved_exact'] == 2
    assert result['failed'] == 2
    assert result['repair_applied'] == 1
    assert result['rows'][4]['error'] == 'missing_prediction'


def test_parseable_text_is_not_assumed_to_have_reached_eos():
    result = score_predictions([EvaluationTarget('a', 'CCO')], [Prediction('a', 'CCO')])
    assert result['terminated'] == 0
    assert result['raw']['parse_success'] == 1
    assert result['raw']['resolved_exact'] == 0
    assert result['repaired']['resolved_exact'] == 0


def test_duplicate_unknown_predictions_and_empty_targets_fail_closed():
    targets = [EvaluationTarget('a', 'CCO')]
    for predictions in ([Prediction('b', 'CCO')], [Prediction('a', 'CCO'), Prediction('a', 'CCO')]):
        with pytest.raises(ProtocolError): score_predictions(targets, predictions)
    with pytest.raises(ProtocolError): score_predictions([], [])
    with pytest.raises(ProtocolError): score_predictions(targets * 2, [])

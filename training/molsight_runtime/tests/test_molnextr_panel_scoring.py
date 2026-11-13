import json
import pytest
from score_molnextr_panel import score
from molsight.prepared import file_hash


def artifacts(tmp_path, *, complete=True):
    pairs = [{'record_key': 'a', 'category': 'complete', 'SMILES': 'CCO', 'peptide_stratum': None},
             {'record_key': 'b', 'category': 'peptide', 'SMILES': 'C[C@H](O)F', 'peptide_stratum': 'stereo'},
             {'record_key': 'c', 'category': 'peptide', 'SMILES': 'CCN', 'peptide_stratum': 'stereo'}]
    pair_path = tmp_path / 'pairs.json'
    pair_path.write_text(json.dumps(pairs))
    rows = [dict(index=i, record_key=p['record_key'], category=p['category'], peptide_stratum=p['peptide_stratum'],
                 gold=p['SMILES'], prediction=prediction, error=None, quality_issues=[])
            for i, (p, prediction) in enumerate(zip(pairs, ['OCC', 'C[C@@H](O)F', 'C1']))]
    output = tmp_path / 'benchmark'
    output.mkdir()
    predictions = output / 'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (output / 'execution.json').write_text(json.dumps({'state': 'complete' if complete else 'running',
        'pairs_sha256': file_hash(pair_path), 'predictions_sha256': file_hash(predictions), 'records': 3}))
    return output, pair_path


def test_same_graph_matches_different_smiles_but_wrong_stereo_and_invalid_output_fail(tmp_path):
    output, pairs = artifacts(tmp_path)
    report = score(output, pairs)
    assert report['metrics']['complete']['literal_exact'] == 1
    assert report['metrics']['peptide']['count'] == 2
    assert report['metrics']['peptide']['literal_exact'] == 0
    assert report['metrics']['peptide']['parse_success'] == 0.5


def test_incomplete_inference_cannot_publish_accuracy(tmp_path):
    output, pairs = artifacts(tmp_path, complete=False)
    with pytest.raises(ValueError, match='incomplete'):
        score(output, pairs)
    assert not (output / 'report.json').exists()

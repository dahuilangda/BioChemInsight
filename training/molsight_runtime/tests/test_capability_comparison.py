import json
from pathlib import Path

import pytest
from compare_capability_panel import compare, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    panel, before, after = [tmp_path / name for name in ('panel', 'before', 'after')]
    files = {}
    for size in (512, 768):
        gold = [{'record_key': str(size), 'component': str(size), 'SMILES': 'CCO',
                 'category': 'peptide', 'panel_group': 'peptide:large', 'source': 'synthetic', 'input_size': size}]
        pair_path = panel / f'pairs_{size}.json'
        write(pair_path, gold)
        files[str(size)] = {'path': str(pair_path), 'sha256': digest(pair_path), 'records': 1}
        for root in (before, after):
            write(root / str(size) / 'report.json', {'pairs_sha256': digest(pair_path), 'examples': 1,
                'precision': 'fp32', 'optimizer_updates': 0, 'configuration': {'input_size': size, 'max_len': 2048},
                'checkpoint_sha256': root.name, 'vocab_sha256': 'vocab', 'abbreviation_registry_sha256': 'registry'})
            row = {'index': 0, 'record_key': str(size), 'category': 'peptide', 'gold': 'CCO',
                   'literal_exact': root == after, 'resolved_exact': root == after,
                   'parse_success': True, 'terminated': True}
            (root / str(size) / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
    write(panel / 'report.json', {'files': files, 'records': 2, 'unique_components': 2, 'limitations': ['test fixture']})
    return panel, before, after


def test_compares_pooled_and_stratum_results_without_merging_resolutions(artifacts):
    report = compare(*artifacts)
    result = report['results']['panel_group:peptide:large']['literal_exact']
    assert (result['records'], result['gained'], result['lost']) == (2, 2, 0)
    assert report['results']['resolution:768']['literal_exact']['records'] == 1


@pytest.mark.parametrize('mutation', ['precision', 'limit', 'identity', 'truncated'])
def test_rejects_incomparable_or_incomplete_predictions(artifacts, mutation):
    panel, before, after = artifacts
    directory = after / '768'
    if mutation in ('precision', 'limit'):
        report = json.loads((directory / 'report.json').read_text())
        if mutation == 'precision':
            report['precision'] = 'bf16'
        else:
            report['configuration']['max_len'] = 1024
        write(directory / 'report.json', report)
    elif mutation == 'identity':
        row = json.loads((directory / 'predictions.jsonl').read_text())
        row['record_key'] = 'other'
        (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
    else:
        (directory / 'predictions.jsonl').write_text('')
    with pytest.raises(ValueError):
        compare(panel, before, after)


def test_reports_exposed_and_unexposed_without_changing_frozen_panel(artifacts, tmp_path):
    panel, before, after = artifacts
    registry = tmp_path / 'exposures.json'
    original = (panel / 'report.json').read_bytes()
    write(registry, {'panel_report_sha256': digest(panel / 'report.json'), 'components': [{'component': '512'}]})
    report = compare(panel, before, after, registry)
    assert report['results']['all']['literal_exact']['records'] == 2
    assert report['results']['unexposed']['literal_exact']['records'] == 1
    assert report['results']['development_exposed']['literal_exact']['records'] == 1
    assert (panel / 'report.json').read_bytes() == original


def test_rejects_exposure_registry_for_unknown_component(artifacts, tmp_path):
    panel, before, after = artifacts
    registry = tmp_path / 'exposures.json'
    write(registry, {'panel_report_sha256': digest(panel / 'report.json'), 'components': [{'component': 'other'}]})
    with pytest.raises(ValueError, match='absent'):
        compare(panel, before, after, registry)

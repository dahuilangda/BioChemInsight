import json
from pathlib import Path
import pysqlite3 as sqlite3
import zlib

import pyarrow as pa
from pyarrow import ipc
import pytest
from molsight.tokenizer import CharTokenizer
from molsight_multitask.export_graphs import digest, export


def make_manifest(tmp_path, *, wrong_tokens=False):
    vocab = Path(__file__).resolve().parents[3] / 'output/molsight_data_v4/vocab.json'
    tokenizer = CharTokenizer(vocab)
    targets = ['CCO', 'CCO', '*<sep><a>0:R1</a>']
    rows = []
    for index, smiles in enumerate(targets):
        rows.append({'record_key': f'key{index}', 'component': f'component{index}',
                     'image_sha256': str(index), 'SMILES': smiles,
                     'token_ids': tokenizer.smiles_to_sequence(smiles)[0]})
    if wrong_tokens:
        rows[0]['token_ids'] = tokenizer.smiles_to_sequence('CC')[0]
        rows[1]['token_ids'] = rows[0]['token_ids']
    table = pa.Table.from_pylist(rows)
    arrow = tmp_path / 'train.arrow'
    with pa.OSFile(str(arrow), 'wb') as stream, ipc.new_file(stream, table.schema) as writer:
        writer.write_table(table)
    admission = tmp_path / 'admission.json'
    admission.write_text(json.dumps({'passed': True, 'records': {'train': 3}}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'training_release_approved': True,
        'records': {'train': 3}, 'files': {'train': {'path': str(arrow), 'sha256': digest(arrow)}},
        'admission_report': {'path': str(admission), 'sha256': digest(admission)},
        'vocabulary': str(vocab), 'vocabulary_sha256': digest(vocab)}))
    return manifest


def test_full_record_coverage_deduplication_and_explicit_unsupported_labels(tmp_path):
    manifest = make_manifest(tmp_path)
    out = tmp_path / 'sidecar'
    report = export(manifest, 'train', out, 1)
    assert report['records'] == 3 and report['unique_targets'] == 2
    assert report['edge_supervised_records'] == 2
    with sqlite3.connect(out / 'graphs.sqlite') as db:
        assert db.execute('SELECT count(*) FROM records').fetchone()[0] == 3
        payload = db.execute('SELECT graph_zlib FROM targets WHERE edge_supported=1').fetchone()[0]
        assert json.loads(zlib.decompress(payload))['smiles'] == 'CCO'
        assert db.execute('SELECT error FROM targets WHERE graph_zlib IS NULL').fetchone()[0]
    assert digest(out / 'graphs.sqlite') == report['database_sha256']


def test_token_mismatch_cannot_publish_completed_sidecar(tmp_path):
    manifest = make_manifest(tmp_path, wrong_tokens=True)
    out = tmp_path / 'sidecar'
    with pytest.raises(ValueError, match='differs from admitted'):
        export(manifest, 'train', out, 1)
    assert not (out / 'report.json').exists()


def test_persisted_audit_checks_source_identity_and_graph_payload(tmp_path):
    from molsight_multitask.audit_graphs import audit
    manifest = make_manifest(tmp_path)
    out = tmp_path / 'sidecar'
    export(manifest, 'train', out, 1)
    report = audit(manifest, out, tmp_path / 'audit.json', 1)
    assert report['passed'] and report['records'] == 3


@pytest.mark.parametrize('mutation', ['graph_charge', 'record_component'])
def test_auditor_detects_semantic_tampering_even_with_updated_database_digest(tmp_path, mutation):
    from molsight_multitask.audit_graphs import audit
    manifest = make_manifest(tmp_path)
    out = tmp_path / 'sidecar'
    export(manifest, 'train', out, 1)
    with sqlite3.connect(out / 'graphs.sqlite') as db:
        if mutation == 'graph_charge':
            target, payload = db.execute('SELECT target_sha256,graph_zlib FROM targets WHERE graph_zlib IS NOT NULL').fetchone()
            graph = json.loads(zlib.decompress(payload))
            graph['atoms'][0]['formal_charge'] = 1
            db.execute('UPDATE targets SET graph_zlib=? WHERE target_sha256=?',
                       (zlib.compress(json.dumps(graph).encode()), target))
        else:
            db.execute('UPDATE records SET component="wrong" WHERE row_index=0')
    report_path = out / 'report.json'
    report = json.loads(report_path.read_text())
    report['database_sha256'] = digest(out / 'graphs.sqlite')
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(manifest, out, tmp_path / 'audit.json', 1)
    assert not (tmp_path / 'audit.json').exists()


def test_graph_reader_requires_admission_and_binds_lookup_to_source_record(tmp_path):
    from molsight_multitask.audit_graphs import audit
    from molsight_multitask.data import GraphStore
    manifest = make_manifest(tmp_path)
    out = tmp_path / 'sidecar'
    export(manifest, 'train', out, 1)
    with pytest.raises(ValueError, match='no admission'):
        GraphStore(out, manifest_sha256=digest(manifest), split='train')
    audit(manifest, out, out / 'admission.json', 1)
    store = GraphStore(out, manifest_sha256=digest(manifest), split='train')
    tokenizer = CharTokenizer(json.loads(manifest.read_text())['vocabulary'])
    kwargs = dict(record_key='key0', component='component0', image_sha256='0', smiles='CCO',
                  token_ids=tokenizer.smiles_to_sequence('CCO')[0])
    assert store.get(0, **kwargs)['smiles'] == 'CCO'
    kwargs['record_key'] = 'wrong'
    with pytest.raises(ValueError, match='lookup'):
        store.get(0, **kwargs)

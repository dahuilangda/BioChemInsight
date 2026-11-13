"""HTTP save/export contracts, isolated from Redis and model inference."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip('redis')
pytest.importorskip('celery')
from fastapi.testclient import TestClient

# pipeline's historical import redirects stderr; retain pytest's capture stream.
_previous_stderr = sys.stderr
try:
    from frontend.backend import main
finally:
    sys.stderr = _previous_stderr
from frontend.backend.task_manager import Task


@pytest.fixture(params=['structure_extraction', 'full_pipeline'])
def task_client(request, monkeypatch, tmp_path):
    original = {'COMPOUND_ID': '53', 'SMILES': 'CCO'}
    csv_path = tmp_path / 'structures.csv'
    csv_path.write_text('COMPOUND_ID,SMILES\n53,CCO\n')
    task = Task(id='save-contract', type=request.param, status='completed', result_path=str(csv_path),
                data=[original], params={'structure_records': [original], 'assay_records': []})

    class Manager:
        def get(self, task_id):
            return task if task_id == task.id else None

        def update(self, task_id, **fields):
            assert task_id == task.id
            for key, value in fields.items():
                setattr(task, key, value)
            return task

    monkeypatch.setattr(main, 'task_manager', Manager())
    monkeypatch.setattr(main, 'TASK_OUTPUT_ROOT', tmp_path)
    with TestClient(main.app) as client:
        yield client, task, tmp_path


def test_save_then_reload_and_export_use_current_rows(task_client):
    client, task, folder = task_client
    # A previous download's derived merge must not survive a structural edit.
    merged = folder / 'merged.csv'
    merged.write_text('COMPOUND_ID,SMILES\n53,CCO\n')
    response = client.put(f'/api/tasks/{task.id}/structures', json={
        'records': [{'COMPOUND_ID': '53-reviewed', 'SMILES': 'CCN'}],
    })
    assert response.status_code == 200, response.text
    loaded = client.get(f'/api/tasks/{task.id}/structures').json()['records']
    assert loaded == [{'COMPOUND_ID': '53-reviewed', 'SMILES': 'CCN'}]
    if task.type == 'full_pipeline':
        assert not merged.exists()
        assert task.params['structure_records_count'] == 1
    downloaded = client.get(f'/api/tasks/{task.id}/download')
    assert downloaded.status_code == 200
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert archive.testzip() is None
        rows = list(csv.DictReader(io.StringIO(archive.read('structures.csv').decode('utf-8-sig'))))
        assert rows == loaded
        assert json.loads(archive.read('raw_task_records.json')) == loaded
        merged_rows = list(csv.DictReader(io.StringIO(archive.read('merged_results.csv').decode('utf-8-sig'))))
        assert merged_rows == loaded


def test_deleting_last_row_persists_without_resurrecting_legacy_rows(task_client):
    client, task, folder = task_client
    response = client.put(f'/api/tasks/{task.id}/structures', json={'records': []})
    assert response.status_code == 200
    assert client.get(f'/api/tasks/{task.id}/structures').json()['records'] == []
    assert task.data == []
    with (folder / 'structures.csv').open(encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['COMPOUND_ID', 'SMILES']
        assert list(reader) == []
    exported = client.get(f'/api/tasks/{task.id}/download')
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert 'raw_task_records.json' not in archive.namelist()
        assert list(csv.DictReader(io.StringIO(archive.read('structures.csv').decode('utf-8-sig')))) == []


def test_completed_task_without_artifact_path_can_save(task_client):
    client, task, folder = task_client
    task.result_path = None
    result = client.put(f'/api/tasks/{task.id}/structures', json={
        'records': [{'COMPOUND_ID': '53', 'SMILES': 'CCN'}],
    })
    assert result.status_code == 200, result.text
    assert Path(task.result_path) == folder / task.id / 'structures.csv'
    assert client.get(f'/api/tasks/{task.id}/structures').json()['records'][0]['SMILES'] == 'CCN'

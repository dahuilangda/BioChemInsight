"""Read immutable drawing runs in explicit replacement order."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_renderings(design_path, manifests):
    design = json.loads(Path(design_path).read_text())
    expected = {row['id']: row['chemical_record']['smiles'] for row in design['records']}
    if len(expected) != len(design['records']):
        raise ValueError('Duplicate design identities')
    records, inputs = {}, {}
    for path in map(Path, manifests):
        manifest = json.loads(path.read_text())
        if manifest.get('sources_unchanged') is not True:
            raise ValueError(f'Unverified rendering sources: {path}')
        inputs[str(path.resolve())] = digest(path)
        seen = set()
        if isinstance(manifest.get('records'), list):
            rows = manifest['records']
        else:
            records_path = path.parent / 'records.jsonl'
            if digest(records_path) != manifest.get('records_sha256'):
                raise ValueError('Rendering records differ from their manifest')
            rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        for row in rows:
            if row['id'] not in expected or row['id'] in seen:
                raise ValueError(f'Unexpected or duplicate rendering identity: {row["id"]}')
            seen.add(row['id'])
            records[row['id']] = row
    if records.keys() != expected.keys():
        raise ValueError('Rendering manifests do not cover the complete design')
    return expected, records, inputs

"""Admitted chemical graph sidecars alongside unchanged image/character data."""
import json
from pathlib import Path
import zlib

import pysqlite3 as sqlite3
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from molsight.prepared import PreparedDataset, file_hash
from molsight_multitask.edges import chemical_targets


class GraphStore:
    def __init__(self, directory, *, manifest_sha256, split):
        self.directory = Path(directory).resolve()
        self.path = self.directory / 'graphs.sqlite'
        report_path = self.directory / 'report.json'
        admission_path = self.directory / 'admission.json'
        if not admission_path.is_file():
            raise ValueError('Chemical graph sidecar has no admission report')
        admission = json.loads(admission_path.read_text())
        report = json.loads(report_path.read_text())
        if (admission.get('passed') is not True or admission.get('chemical_graph_sidecar_approved') is not True
                or admission['manifest_sha256'] != manifest_sha256 or admission['split'] != split
                or report['state'] != 'complete' or report['schema_version'] != 1
                or report['manifest_sha256'] != manifest_sha256 or report['split'] != split
                or admission['export_report_sha256'] != file_hash(report_path)
                or admission['database_sha256'] != report['database_sha256']
                or file_hash(self.path) != report['database_sha256']):
            raise ValueError('Chemical graph sidecar identity or approval differs')
        wal = Path(str(self.path) + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('Chemical graph sidecar still has uncheckpointed writes')
        self.records = admission['records']
        self.database_sha256 = admission['database_sha256']
        self.admission_sha256 = file_hash(admission_path)
        self.file_stat = self._stat()
        self.connection = None

    def _stat(self):
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connection'] = None
        return state

    def get(self, index, *, record_key, component, image_sha256, smiles, token_ids):
        if self._stat() != self.file_stat:
            raise ValueError('Admitted chemical graph database changed')
        if self.connection is None:
            self.connection = sqlite3.connect('file:' + str(self.path) + '?mode=ro&immutable=1', uri=True)
        row = self.connection.execute('''SELECT record_key,component,image_sha256,smiles,
          graph_zlib,error,edge_supported FROM records JOIN targets USING(target_sha256) WHERE row_index=?''', (index,)).fetchone()
        if row is None or row[:4] != (record_key, component, image_sha256, smiles):
            raise ValueError('Graph lookup does not match the admitted image/structure record')
        payload, error, supported = row[4:]
        if not supported:
            # Preserve the example for its character objective; do not fabricate edges.
            return None
        if payload is None or error is not None:
            raise ValueError('Graph-supervised record lacks a valid annotation')
        graph = json.loads(zlib.decompress(payload))
        if graph['smiles'] != smiles or graph['token_ids'] != token_ids:
            raise ValueError('Graph lookup changed the character target')
        return graph


class GraphDataset(PreparedDataset):
    def __init__(self, manifest, split, *, sidecar, max_len=2048):
        super().__init__(manifest, split, max_len=max_len)
        self.graphs = GraphStore(sidecar, manifest_sha256=self.manifest_sha256, split=split)
        if self.graphs.records != len(self):
            raise ValueError('Graph sidecar record coverage differs from source dataset')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.table.slice(index, 1).to_pylist()[0]
        graph = self.graphs.get(index, record_key=row['record_key'], component=row['component'],
                                image_sha256=row['image_sha256'], smiles=sample['smiles'],
                                token_ids=sample['label'].tolist())
        if graph is None:
            sample['atom_indices'] = torch.empty(0, dtype=torch.long)
            sample['chemical_edges'] = torch.empty(0, 0, dtype=torch.long)
        else:
            sample['atom_indices'] = torch.tensor(graph['atom_indices'], dtype=torch.long)
            sample['chemical_edges'] = chemical_targets(graph)
        return sample


def collate_graphs(rows):
    required = {'idx', 'image', 'label', 'smiles', 'atom_indices', 'chemical_edges'}
    if not rows or any(set(row) != required for row in rows):
        raise ValueError('Inconsistent graph/character sample fields')
    if len({tuple(row['image'].shape) for row in rows}) != 1:
        raise ValueError('Graph batches require a common image resolution')
    for row in rows:
        count = len(row['atom_indices'])
        if row['chemical_edges'].shape != (count, count):
            raise ValueError('Chemical edge labels do not align with atom queries')
        if count and ((row['atom_indices'] < 1).any() or (row['atom_indices'] >= len(row['label']) - 1).any()):
            raise ValueError('Atom query lies outside the non-EOS target span')
    count = max(len(row['atom_indices']) for row in rows)
    return {'idx': torch.tensor([row['idx'] for row in rows], dtype=torch.long),
            'image': torch.stack([row['image'] for row in rows]),
            'label': pad_sequence([row['label'] for row in rows], batch_first=True, padding_value=0),
            'smiles': [row['smiles'] for row in rows],
            'atom_indices': pad_sequence([row['atom_indices'] for row in rows], batch_first=True, padding_value=0),
            'chemical_edges': torch.stack([F.pad(row['chemical_edges'],
                (0, count - len(row['atom_indices']), 0, count - len(row['atom_indices'])), value=-100) for row in rows])}

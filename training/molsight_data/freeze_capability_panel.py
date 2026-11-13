"""Freeze a component-disjoint test panel before candidate inference.

Produces one pair file per input resolution for the fixed-resolution benchmark.
This proves separation from this adaptation release, not historical pretraining.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

import pyarrow as pa
from pyarrow import ipc
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def table_for(specification, split):
    entry = specification['files'][split]
    if digest(entry['path']) != entry['sha256']:
        raise ValueError(f'{split} Arrow hash differs from the admitted release')
    with pa.memory_map(entry['path'], 'r') as source:
        return ipc.open_file(source).read_all()


def select(rows, *, peptide_per_stratum=32):
    """Round-robin sources within each predeclared group; one row/component."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        category = row['category']
        if category == 'peptide':
            group = 'peptide:' + (row.get('peptide_stratum') or 'unspecified')
        elif row.get('peptide_candidate'):
            group = 'discovered_cyclic' if row.get('cyclic_peptide_candidate') else 'discovered_linear'
        else:
            group = category
        rank = hashlib.sha256(('molsight-capability-panel-v1:' + row['record_key']).encode()).hexdigest()
        buckets[group][row['source']].append((rank, row))
    # Peptide examples cannot be consumed by a broad complete-molecule quota.
    groups = sorted(buckets, key=lambda key: (not (key.startswith('peptide:') or key.startswith('discovered_')), key))
    selected, seen = [], set()
    for group in groups:
        limit = peptide_per_stratum if group.startswith('peptide:') else {
            'complete': 384, 'markush': 96, 'attachment': 64,
            'discovered_cyclic': 64, 'discovered_linear': 32,
        }.get(group)
        if limit is None:
            raise ValueError(f'No predeclared budget for {group}')
        sources = {name: deque(row for _, row in sorted(values, key=lambda item: item[0]))
                   for name, values in sorted(buckets[group].items())}
        count = 0
        while sources and count < limit:
            for name in list(sources):
                queue = sources[name]
                while queue and queue[0]['component'] in seen:
                    queue.popleft()
                if queue:
                    row = dict(queue.popleft())
                    seen.add(row['component'])
                    row['panel_group'] = group
                    selected.append(row)
                    count += 1
                if not queue:
                    del sources[name]
                if count == limit:
                    break
    return sorted(selected, key=lambda row: row['record_key'])


def freeze(manifest, tags_path, output):
    specification = json.loads(Path(manifest).read_text())
    if specification.get('training_release_approved') is not True:
        raise ValueError('Release is not admitted')
    admission = specification['admission_report']
    if digest(admission['path']) != admission['sha256']:
        raise ValueError('Admission report changed')
    if json.loads(Path(admission['path']).read_text()).get('passed') is not True:
        raise ValueError('Admission report did not pass')
    test = table_for(specification, 'test').to_pylist()
    test_components = {row['component'] for row in test}
    overlap = {}
    for split in ('train', 'validation'):
        components = set(table_for(specification, split).column('component').to_pylist())
        overlap[split] = len(test_components & components)
        if overlap[split]:
            raise ValueError(f'Test component overlaps {split}')
    tags = json.loads(Path(tags_path).read_text())
    if (tags['schema_version'] != 1 or tags['manifest_sha256'] != digest(manifest)
            or tags['splits']['test']['file_sha256'] != specification['files']['test']['sha256']
            or tags['splits']['test']['records'] != len(test)):
        raise ValueError('Sampling tags belong to a different release')
    seen_indices = set()
    for item in tags['splits']['test']['peptides']:
        index = item['index']
        if (type(index) is not int or not 0 <= index < len(test) or index in seen_indices
                or item['peptide_candidate'] is not True
                or type(item['cyclic_peptide_candidate']) is not bool):
            raise ValueError('Invalid peptide tag')
        seen_indices.add(index)
        test[index].update({key: item[key] for key in ('peptide_candidate', 'cyclic_peptide_candidate')})
    selected = select(test)
    pairs = defaultdict(list)
    for row in selected:
        path = Path(row['file_path'])
        if digest(path) != row['image_sha256']:
            raise ValueError(f'Selected image changed: {path}')
        with Image.open(path) as image:
            image.verify()
        pair = {key: row[key] for key in ('record_key', 'component', 'category', 'source',
                'source_split', 'file_path', 'image_sha256', 'model_input_preprocessed',
                'input_size', 'SMILES', 'peptide_stratum', 'panel_group')}
        # Existing paired benchmark groups atom_bin; keep unknown explicit.
        pair['atom_bin'] = 'not_computed'
        pair['evaluation_scope'] = 'adaptation_test_component_disjoint; historical_pretraining_independence_unknown'
        pairs[row['input_size']].append(pair)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for resolution, values in sorted(pairs.items()):
        path = output / f'pairs_{resolution}.json'
        path.write_text(json.dumps(values, indent=2) + '\n')
        files[str(resolution)] = {'path': str(path.resolve()), 'sha256': digest(path), 'records': len(values)}
    report = {'manifest': str(Path(manifest).resolve()), 'manifest_sha256': digest(manifest),
        'tags_sha256': digest(tags_path), 'code_sha256': digest(__file__),
        'files': files, 'records': len(selected), 'unique_components': len({r['component'] for r in selected}),
        'all_test_component_overlap': overlap,
        'panel_groups': dict(Counter(r['panel_group'] for r in selected)),
        'sources': dict(Counter(r['source'] for r in selected)),
        'limitations': ['Adaptation-release separation only; historical pretraining overlap unknown.',
                       'Source labels have not all been independently reviewed.',
                       'No candidate predictions used for selection; reserve this panel for final comparisons.',
                       'Run each file at its specified resolution; do not downsample verified inputs.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tags', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.manifest, args.tags, args.out), indent=2))

"""Paired exact-identity comparison across frozen resolution-specific panels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from paired_benchmark import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(panel, before, after, exposures=None):
    panel = Path(panel)
    manifest = json.loads((panel / 'report.json').read_text())
    exposed = set()
    exposure_hash = None
    if exposures is not None:
        registry = json.loads(Path(exposures).read_text())
        if registry['panel_report_sha256'] != digest(panel / 'report.json'):
            raise ValueError('Exposure registry belongs to a different panel')
        listed = [row['component'] for row in registry['components']]
        if len(listed) != len(set(listed)):
            raise ValueError('Repeated development exposure component')
        exposed = set(listed)
        exposure_hash = digest(exposures)
    paired, provenance, components = [], [], set()
    for resolution, entry in sorted(manifest['files'].items()):
        pair_path = panel / Path(entry['path']).name
        if digest(pair_path) != entry['sha256']:
            raise ValueError('Frozen panel changed')
        gold = json.loads(pair_path.read_text())
        if not gold or len(gold) != entry['records']:
            raise ValueError('Panel coverage differs')
        predictions, reports = [], []
        for root in (before, after):
            directory = Path(root) / str(resolution)
            report = json.loads((directory / 'report.json').read_text())
            configuration = report['configuration']
            if (report['pairs_sha256'] != entry['sha256'] or report['examples'] != len(gold)
                    or report['precision'] != 'fp32' or configuration['input_size'] != int(resolution)
                    or report['optimizer_updates'] != 0):
                raise ValueError('Benchmark provenance, precision, resolution or coverage differs')
            rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
            if len(rows) != len(gold):
                raise ValueError('Incomplete benchmark predictions')
            for index, (row, target) in enumerate(zip(rows, gold, strict=True)):
                if (row['index'] != index or row['record_key'] != target['record_key']
                        or row['gold'] != target['SMILES'] or row['category'] != target['category']
                        or any(type(row[key]) is not bool for key in
                               ('literal_exact', 'resolved_exact', 'parse_success', 'terminated'))):
                    raise ValueError('Prediction identity or metric types differ')
            predictions.append(rows)
            reports.append(report)
            provenance.append({'directory': str(directory.resolve()),
                               'report_sha256': digest(directory / 'report.json'),
                               'predictions_sha256': digest(directory / 'predictions.jsonl'),
                               'checkpoint_sha256': report['checkpoint_sha256']})
        if (reports[0]['configuration']['max_len'] != reports[1]['configuration']['max_len']
                or reports[0]['vocab_sha256'] != reports[1]['vocab_sha256']
                or reports[0]['abbreviation_registry_sha256'] != reports[1]['abbreviation_registry_sha256']):
            raise ValueError('Decode limit, vocabulary or abbreviation contract differs')
        for target, old, new in zip(gold, *predictions, strict=True):
            if target['component'] in components:
                raise ValueError('Repeated component across panel files')
            components.add(target['component'])
            paired.append((target, old, new))
    if len(paired) != manifest['records'] or len(components) != manifest['unique_components']:
        raise ValueError('Frozen panel totals differ')
    if exposed - components:
        raise ValueError('Exposure registry names components absent from the panel')
    groups = defaultdict(list)
    for target, old, new in paired:
        for group in ('all', 'panel_group:' + target['panel_group'],
                      'resolution:' + str(target['input_size']), 'source:' + target['source']):
            groups[group].append((old, new))
        if exposures is not None:
            label = 'development_exposed' if target['component'] in exposed else 'unexposed'
            groups[label].append((old, new))
            groups[label + ':panel_group:' + target['panel_group']].append((old, new))
    results = {}
    for group, rows in sorted(groups.items()):
        results[group] = {metric: summarize([(old[metric], new[metric]) for old, new in rows])
                          for metric in ('literal_exact', 'resolved_exact')}
        results[group]['before_parse_success'] = sum(old['parse_success'] for old, _ in rows)
        results[group]['after_parse_success'] = sum(new['parse_success'] for _, new in rows)
        results[group]['before_terminated'] = sum(old['terminated'] for old, _ in rows)
        results[group]['after_terminated'] = sum(new['terminated'] for _, new in rows)
    return {'panel_report_sha256': digest(panel / 'report.json'), 'provenance': provenance,
            'development_exposures_sha256': exposure_hash, 'development_exposed_components': len(exposed),
            'results': results, 'limitations': manifest['limitations'],
            'interpretation': 'Exploratory group comparisons; intervals are not adjusted for multiple comparisons. No automatic deployment decision.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'before', 'after', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exposures', type=Path, help='Development exposure registry; report full and unexposed results')
    args = parser.parse_args()
    report = compare(args.panel, args.before, args.after, args.exposures)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')

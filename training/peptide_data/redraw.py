"""Regenerate audited renderings without repeating chemical layout searches."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from molsight.cpu import available_cpus, initialize_worker
from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D

from .artifacts import digest, load_renderings
from .review import STYLES, _render
from .quality import chemical_identity


def redraw(job):
    original, expected, output, sizes = job
    result = dict(original)
    result['training_release_approved'] = False
    if original['status'] != 'rendered_for_review':
        return result
    source = Path(original['images'][0]['inputs'][0]['file']).parent.parent / 'drawing.mol'
    molecule = Chem.MolFromMolFile(str(source), sanitize=True, removeHs=False)
    if molecule is None or Chem.MolToSmiles(molecule) != expected:
        raise ValueError(f'Drawing structure mismatch: {original["id"]}')
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)
    if chemical_identity(molecule) != expected:
        raise ValueError(f'Drawing preparation changed structure: {original["id"]}')
    directory = Path(output) / 'images' / original['id']
    directory.mkdir(parents=True, exist_ok=False)
    Chem.MolToMolFile(molecule, str(directory / 'drawing.mol'), includeStereo=True)
    result['molecule_source'] = {'file': str(source), 'sha256': digest(source)}
    result['images'] = [_render(molecule, directory / name, name, sizes) for name in STYLES]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--manifests', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 768, 1024])
    parser.add_argument('--workers', type=int, default=available_cpus())
    args = parser.parse_args()
    if min(args.workers, *args.sizes) < 1:
        parser.error('Counts and image dimensions must be positive')
    started = time.monotonic()
    expected, records, manifests = load_renderings(args.design, args.manifests)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {str(path.resolve()): digest(path) for path in Path(__file__).parent.glob('*.py')}
    jobs = [(row, expected[row['id']], str(args.out.resolve()), args.sizes) for row in records.values()]
    results, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(0,)) as executor:
        for row in executor.map(redraw, jobs, chunksize=2):
            results.append(row)
            (args.out / (row['id'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
            counts[row['status']] += 1
            if len(results) % 128 == 0:
                print(json.dumps({'records': len(results), 'seconds': time.monotonic() - started}), flush=True)
    report = {'design_sha256': digest(args.design), 'manifests': manifests,
              'rdkit_version': rdBase.rdkitVersion, 'sources': sources,
              'sources_unchanged': all(digest(path) == value for path, value in sources.items()),
              'counts': dict(counts), 'seconds': time.monotonic() - started,
              'workers': args.workers, 'records': results, 'training_release_approved': False}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('counts', 'seconds', 'sources_unchanged')}))


if __name__ == '__main__':
    main()

"""Audit frozen peptide drawings at their final model input resolution."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from molsight.cpu import available_cpus, initialize_worker
from PIL import Image
from rdkit import Chem

from .svg_quality import drawing_report, primitives
from .artifacts import digest, load_renderings


def audit_style(job):
    identity, expected, style = job
    directory = Path(style['inputs'][0]['file']).parent
    svg_path, mol_path = directory / 'drawing.svg', directory.parent / 'drawing.mol'
    molecule = Chem.MolFromMolFile(str(mol_path), sanitize=True, removeHs=False)
    if molecule is None or Chem.MolToSmiles(molecule) != expected:
        raise ValueError(f'Drawing structure mismatch: {identity}')
    svg = svg_path.read_text()
    native = np.asarray(style['atom_positions'])
    _, shapes = primitives(svg)
    checks = []
    for sample in style['inputs']:
        final = np.asarray(sample['atom_positions'])
        affine = np.linalg.lstsq(np.c_[native, np.ones(len(native))], final, rcond=None)[0]
        mapped = np.c_[native, np.ones(len(native))] @ affine
        residual = float(np.max(np.abs(mapped - final)))
        singular = np.linalg.svd(affine[:2], compute_uv=False)
        scale = float(singular.mean())
        image_path = Path(sample['file'])
        with Image.open(image_path) as image:
            width, height = image.size
        report = drawing_report(svg, molecule, native, scale=scale)
        reasons = list(report['reasons'])
        if residual > 1e-4 or singular[0] - singular[-1] > 1e-5 or np.linalg.det(affine[:2]) <= 0:
            reasons.append('invalid_input_coordinate_transform')
        for shape in shapes:
            x0, y0, x1, y1 = shape.bounds
            corners = np.array([[x0, y0, 1], [x0, y1, 1], [x1, y0, 1], [x1, y1, 1]]) @ affine
            if np.any(corners < 0) or np.any(corners > [width, height]):
                reasons.append('clipped_input_drawing')
                break
        if digest(image_path) != sample['sha256']:
            reasons.append('input_image_hash_mismatch')
        checks.append({
            'size': sample['size'], 'file': str(image_path),
            'accepted': not reasons, 'reasons': sorted(set(reasons)),
            'affine_residual': residual, 'vector_quality': report,
            'resolution_screen_pass': (
                sample['nominal_font_pixels'] >= 8
                and sample['median_bond_pixels'] >= 10
                and sample['nominal_stroke_pixels'] >= 0.7
            ),
            'training_release_approved': False,
        })
    return {'id': identity, 'style': style['style'], 'svg_sha256': digest(svg_path),
            'molecule_sha256': digest(mol_path), 'inputs': checks,
            'training_release_approved': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--manifests', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=available_cpus())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Worker count must be positive')
    started = time.monotonic()
    expected, records, inputs = load_renderings(args.design, args.manifests)
    jobs = [(row['id'], expected[row['id']], style)
            for row in records.values() if row['status'] == 'rendered_for_review'
            for style in row['images']]
    args.out.mkdir(parents=True, exist_ok=False)
    counts, reasons = Counter(), Counter()
    sources = {str(path.resolve()): digest(path) for path in Path(__file__).parent.glob('*.py')}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(0,)) as executor, (args.out / 'records.jsonl').open('x') as stream:
        for result in executor.map(audit_style, jobs, chunksize=4):
            stream.write(json.dumps(result, separators=(',', ':')) + '\n')
            for check in result['inputs']:
                counts['input_images'] += 1
                counts['vector_pass' if check['accepted'] else 'vector_rejected'] += 1
                counts['resolution_pass' if check['resolution_screen_pass'] else 'resolution_rejected'] += 1
                reasons.update(check['reasons'])
            if counts['input_images'] % 256 == 0:
                stream.flush()
                print(json.dumps({'counts': dict(counts), 'seconds': time.monotonic() - started}), flush=True)
    report = {
        'design_sha256': digest(args.design), 'manifests': inputs,
        'layout_status': dict(Counter(row['status'] for row in records.values())),
        'counts': dict(counts), 'reasons': dict(reasons), 'workers': args.workers,
        'seconds': time.monotonic() - started, 'code_sha256': sources,
        'sources_unchanged': all(digest(path) == value for path, value in sources.items()),
        'training_release_approved': False,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'code_sha256'}))


if __name__ == '__main__':
    main()

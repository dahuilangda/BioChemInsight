"""Export audited peptide images to the shared recognition-data schema."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from molsight.cpu import available_cpus, initialize_worker
from molsight.structure import parse_structure
from molsight.tokenizer import CharTokenizer
from PIL import Image
from rdkit import Chem

from .artifacts import digest, load_renderings

_tokenizer = None


def initialize(vocabulary):
    global _tokenizer
    initialize_worker(0)
    _tokenizer = CharTokenizer(vocabulary)


def inspect_image(job):
    index, source, design_path, style, sample, max_len = job
    image_path = Path(sample['file'])
    encoded = image_path.read_bytes()
    image_hash = hashlib.sha256(encoded).hexdigest()
    if image_hash != sample['sha256']:
        raise ValueError(f'Image changed after drawing audit: {image_path}')
    with Image.open(image_path) as original:
        image = original.convert('RGB')
        width, height = image.size
        pixels = hashlib.sha256(f'{width},{height}:'.encode() + image.tobytes()).hexdigest()
        bounds = image.convert('L').point(lambda value: 255 if value < 220 else 0).getbbox()
    label = source['chemical_record']['smiles']
    document = parse_structure(label)
    tokens, atoms = _tokenizer.smiles_to_sequence(label)
    if len(tokens) > max_len or _tokenizer.sequence_to_smiles(tokens[1:])['smiles'] != label:
        raise ValueError(f'Target tokenization failed: {source["id"]}')
    identity = json.dumps([document.colors, document.canonical, document.repeat,
                           document.variation_records], separators=(',', ':'))
    family = document.molecule
    Chem.RemoveStereochemistry(family)
    for atom in family.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    return {
        'source': source.get('source_name', 'peptide_chemistry_pilot'), 'source_path': design_path, 'source_row': index,
        'source_split': 'review', 'split': 'review', 'image_domain': 'synthetic',
        'training_release_approved': False, 'original_label': label, 'SMILES': label,
        'source_notation': 'structure', 'structure_hash': hashlib.sha256(identity.encode()).hexdigest(),
        'structure_family': hashlib.sha256(Chem.MolToSmiles(family).encode()).hexdigest(),
        'token_length': len(tokens), 'atom_count': len(atoms), 'category': 'peptide',
        'peptide_id': source['id'], 'peptide_stratum': source['stratum'],
        'composition_group': source['composition_group'], 'helm': source['helm'], 'biln': source['biln'],
        'library_digest': source['chemical_record']['library']['digest'],
        'drawing_style': style, 'input_size': sample['size'], 'model_input_preprocessed': True,
        'file_path': str(image_path), 'image_sha256': image_hash, 'pixel_sha256': pixels,
        'width': width, 'height': height, 'ink_bbox': bounds, 'rejection_reasons': [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=available_cpus())
    parser.add_argument('--max-len', type=int, default=2048)
    args = parser.parse_args()
    if min(args.workers, args.max_len) < 1:
        parser.error('Worker count and target length must be positive')
    _, renderings, _ = load_renderings(args.design, [args.manifest])
    audit = json.loads((args.audit / 'report.json').read_text())
    if (audit.get('sources_unchanged') is not True
            or audit['design_sha256'] != digest(args.design)
            or audit['manifests'].get(str(args.manifest.resolve())) != digest(args.manifest)):
        raise ValueError('Drawing audit does not match the requested design and rendering')
    decisions = {}
    with (args.audit / 'records.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['id'], row['style'])
            if key in decisions:
                raise ValueError('Duplicate drawing-audit decision')
            decisions[key] = row['inputs']
    design = json.loads(args.design.read_text())
    jobs, excluded, excluded_structures = [], [], []
    for index, row in enumerate(design['records']):
        count = 0
        for style in renderings[row['id']].get('images', []):
            checks = decisions[(row['id'], style['style'])]
            samples = {sample['file']: sample for sample in style['inputs']}
            passing = [check for check in checks if check['accepted'] and check['resolution_screen_pass']
                       and samples[check['file']]['input_quality_pass']]
            if not passing:
                excluded.append({'id': row['id'], 'style': style['style'], 'checks': checks})
                continue
            chosen = min(passing, key=lambda check: check['size'])
            sample = next(item for item in style['inputs'] if item['file'] == chosen['file'])
            jobs.append((index, row, str(args.design.resolve()), style['style'], sample, args.max_len))
            count += 1
        if not count:
            # Structures whose every render variant failed the audit stay
            # out of the deterministic-label release, with the reason
            # recorded; they do not abort the batch.
            excluded_structures.append({'id': row['id'], 'stratum': row['stratum']})
            continue
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {str(path.resolve()): digest(path) for path in Path(__file__).parent.glob('*.py')}
    for name in ('molsight.structure', 'molsight.tokenizer', 'molsight.cpu'):
        path = Path(__import__(name, fromlist=['']).__file__)
        sources[str(path.resolve())] = digest(path)
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.vocab),)) as executor, (args.out / 'records.jsonl').open('x') as stream:
        for row in executor.map(inspect_image, jobs, chunksize=8):
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            counts['images'] += 1
            counts['resolution:' + str(row['input_size'])] += 1
            counts['stratum:' + row['peptide_stratum']] += 1
    report = {
        'records_file': str((args.out / 'records.jsonl').resolve()),
        'records_sha256': digest(args.out / 'records.jsonl'),
        'design_sha256': digest(args.design), 'render_manifest_sha256': digest(args.manifest),
        'audit_report_sha256': digest(args.audit / 'report.json'),
        'vocab_sha256': digest(args.vocab), 'chemical_records': len(design['records']),
        'counts': dict(counts), 'excluded_styles': excluded,
        'excluded_structures': excluded_structures,
        'code_sha256': sources,
        'sources_unchanged': all(digest(path) == value for path, value in sources.items()),
        'training_release_approved': False,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'excluded_styles'}))


if __name__ == '__main__':
    main()

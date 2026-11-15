"""Generate traceable synthetic sequences from explicitly defined monomers.

These random combinations are synthetic chemistry labels, never patent gold.
No topology, stereochemistry or monomer is selected to fit a target mass.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from patent_reference import assemble_codes

MONOMERS = [
    'bhcLeu', 'Nle', '4Pal', 'Pipt5OH', 'W', 'dProt4OH', 'N',
    'Trp7Cl', 'aMeS', 'Phe4CONH2', 'Phe4COOH', 'Sar', 'SbMeW7OMe',
    'Pip2c', 'S', 'Q', 'T', 'NMeA', 'NxG21',
]


def generate(sizes, per_size, seed):
    rng = random.Random(seed)
    rows = []
    for size in sizes:
        if size < 3:
            raise ValueError('At least three residues are required')
        for _ in range(per_size):
            codes = rng.choices(MONOMERS, k=size)
            _, record = assemble_codes(codes)
            rows.append({'SMILES': record['smiles'], 'codes': codes,
                         'size': size, 'index': len(rows)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[14, 18, 22, 26])
    parser.add_argument('--per-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args()
    if args.per_size < 1:
        parser.error('--per-size must be positive')
    rows = generate(args.sizes, args.per_size, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'input.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['SMILES'])
        writer.writeheader()
        writer.writerows({'SMILES': row['SMILES']} for row in rows)
    manifest = {'seed': args.seed, 'sizes': args.sizes, 'per_size': args.per_size,
        'source': 'typed named monomer assembly; synthetic combinations, not patent gold',
        'compiler_sha256': hashlib.sha256(Path(__file__).with_name('patent_reference.py').read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'monomers': MONOMERS, 'rows': rows}
    (args.out/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Generated {len(rows)} named-monomer synthetic inputs')


if __name__ == '__main__':
    main()

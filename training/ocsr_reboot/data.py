"""Frozen diagnostic inputs with exact atom-order and pixel-coordinate lineage.

The small synthetic drawings exercise the input contract; they are not a real
world benchmark. Existing peptide inputs retain their admitted source splits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid target')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def draw_small(smiles, directory, split, index):
    molecule = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(molecule)
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, kekulize=False, addChiralHs=False)
    target = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    order = json.loads(molecule.GetProp('_smilesAtomOutputOrder'))
    drawer = rdMolDraw2D.MolDraw2DCairo(768, 768)
    drawer.drawOptions().prepareMolsBeforeDrawing = False
    drawer.DrawMolecule(molecule)
    coords = [[drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y] for i in order]
    drawer.FinishDrawing()
    path = directory / f'{split}-small-{index}.png'
    path.write_bytes(drawer.GetDrawingText())
    return {'record_key': hashlib.sha256((split + ':' + target).encode()).hexdigest(),
            'component': exact_key(target), 'split': split, 'category': 'complete',
            'source': 'diagnostic_rdkit_render', 'file_path': str(path),
            'image_sha256': digest(path), 'SMILES': target, 'coords_pixels': coords,
            'peptide_stratum': None, 'coordinate_scope': 'visible_atomic_drawing',
            'source_order': order}


def peptide_record(row, split):
    image = Path(row['file_path'])
    if not row.get('image_sha256') or digest(image) != row['image_sha256']:
        raise ValueError('Admitted image hash mismatch or missing authority')
    render_root, record_id = image.parents[3], image.parents[1].name
    entry_path = render_root / (record_id + '.json')
    if entry_path.is_file():
        metadata = json.loads(entry_path.read_text())
    else:
        entry_path = render_root / 'records.jsonl'
        metadata = None
        with entry_path.open() as stream:
            for line in stream:
                if record_id not in line: continue
                value = json.loads(line)
                if value.get('id') == record_id:
                    metadata = value; break
    entries = [] if metadata is None else [e for style in metadata['images'] for e in style['inputs']
        if Path(e['file']).resolve() == image.resolve()]
    if len(entries) != 1:
        raise ValueError(f'No unique coordinate metadata for {image}')
    entry = entries[0]
    mol_path = image.parents[1] / 'drawing.mol'
    drawing = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    target = row.get('SMILES')
    if not target:
        raise ValueError('Peptide source requires authoritative SMILES')
    gold = Chem.MolFromSmiles(target)
    if drawing is None or gold is None or exact_key(Chem.MolToSmiles(drawing)) != exact_key(target):
        raise ValueError('Drawing and target chemical identities differ')
    if 'perm' in row:
        order = list(row['perm'])
    else:
        # Find an atom-attribute/bond/stereo preserving target-to-drawing map.
        order = list(drawing.GetSubstructMatch(gold, useChirality=True))
    if len(order) != gold.GetNumAtoms() or len(set(order)) != len(order):
        raise ValueError('Incomplete atom mapping')
    # Canonical identity alone is permutation invariant and cannot verify xy.
    def atom_state(atom):
        return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                atom.GetNumExplicitHs(), atom.GetIsAromatic())
    for target_index, source_index in enumerate(order):
        if atom_state(gold.GetAtomWithIdx(target_index)) != atom_state(drawing.GetAtomWithIdx(source_index)):
            raise ValueError('Atom attributes do not align with drawing coordinates')
    if gold.GetNumBonds() != drawing.GetNumBonds():
        raise ValueError('Drawing bond count differs')
    for bond in gold.GetBonds():
        source_bond = drawing.GetBondBetweenAtoms(order[bond.GetBeginAtomIdx()], order[bond.GetEndAtomIdx()])
        if source_bond is None or source_bond.GetBondType() != bond.GetBondType():
            raise ValueError('Drawing bonds do not align with target atom order')
    restored = Chem.RenumberAtoms(drawing, order)
    def indexed_identity(molecule):
        labeled = Chem.Mol(molecule)
        for index, atom in enumerate(labeled.GetAtoms()):
            atom.SetAtomMapNum(index + 1)
        return Chem.MolToSmiles(labeled, canonical=True, isomericSmiles=True)
    if indexed_identity(restored) != indexed_identity(gold):
        raise ValueError('Drawing atom mapping changes indexed stereochemical identity')
    if exact_key(Chem.MolToSmiles(restored)) != exact_key(target):
        raise ValueError('Atom order changed chemical identity')
    positions = entry['atom_positions']
    coords = [positions[i] for i in order]
    if len(coords) != gold.GetNumAtoms():
        raise ValueError('Coordinate count differs')
    return {'record_key': row['record_key'], 'component': row.get('component') or exact_key(target),
            'split': split, 'category': 'peptide', 'peptide_stratum': row['peptide_stratum'],
            'source': 'admitted_peptide_render', 'file_path': str(image),
            'image_sha256': digest(image), 'SMILES': target, 'coords_pixels': coords,
            'coordinate_scope': 'visible_atomic_drawing', 'coordinate_file': str(entry_path),
            'coordinate_sha256': digest(entry_path), 'drawing_mol_sha256': digest(mol_path),
            'source_order': order}


def prepare_sample(record, tokenizer, input_size, max_tokens=4096):
    """Use one transform for image + coordinates; never independently recrop."""
    import torch
    from utils.MolNexTR.dataset import get_transforms
    if digest(record['file_path']) != record['image_sha256']:
        raise ValueError('Input image hash mismatch')
    image = np.asarray(Image.open(record['file_path']).convert('RGB'))
    coords = np.asarray(record['coords_pixels'], dtype=np.float64)
    mol = Chem.MolFromSmiles(record['SMILES'])
    if coords.shape != (mol.GetNumAtoms(), 2) or not np.isfinite(coords).all():
        raise ValueError('Invalid atom-coordinate correspondence')
    h, w = image.shape[:2]
    if (coords < 0).any() or (coords[:, 0] >= w).any() or (coords[:, 1] >= h).any():
        raise ValueError('Coordinates outside source pixels')
    result = get_transforms(input_size, input_size, augment=False)(image=image, keypoints=coords)
    xy = np.asarray(result['keypoints'], dtype=np.float64) / input_size
    if xy.shape != coords.shape or (xy < 0).any() or (xy > 1).any():
        raise ValueError('Preprocessing lost/out-of-frame atoms')
    labels, indices = tokenizer.smiles_to_sequence(record['SMILES'], coords=xy.tolist())
    if len(indices) != mol.GetNumAtoms() or len(labels) > max_tokens or 3 in labels:
        raise ValueError('Incomplete tokenizer coverage or exceeded decode budget')
    # Verify tokenizer roundtrip; no unknowns/random coordinate fallback allowed.
    text = tokenizer.sequence_to_smiles(labels[1:])
    if exact_key(text['smiles']) != exact_key(record['SMILES']) or len(text['coords']) != len(indices):
        raise ValueError('Tokenizer roundtrip changed complete structure')
    edges = torch.zeros(len(indices), len(indices), dtype=torch.long)
    types = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
             Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[i, j] = edges[j, i] = types[bond.GetBondType()]
    edges.fill_diagonal_(-100)
    return {'image': result['image'], 'refs': {
        'chartok_coords': [torch.tensor([labels]), torch.tensor([[len(labels)]])],
        'atom_indices': [torch.tensor([indices]), torch.tensor([[len(indices)]])],
        'edges': edges.unsqueeze(0)}, 'record_key': record['record_key'],
        'smiles': record['SMILES'], 'coords_normalized': xy.tolist()}


def build_gate(out):
    """Freeze a difficult training memorization gate and separate diagnostics."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.compute as pc
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    small = ['CC(=O)Oc1ccccc1C(=O)O', 'C[C@H](O)c1ccccc1',
             'C/C=C/C(=O)[O-].[Na+]', '[13CH3][C@@H](N)C(=O)O']
    records = [draw_small(s, out, 'train', i) for i, s in enumerate(small)]
    perm = Path('/data/BioChemInsight/output/molsight_training_audit/peptide_atom_perm_train.jsonl')
    pool = [json.loads(line) for line in perm.read_text().splitlines()]
    selected = []
    seen = set()
    for strata in [('linear_short', 'linear_modified', 'linear_medium', 'linear_long'),
                   ('head_to_tail_small', 'head_to_tail_medium', 'head_to_tail_modified', 'multiple_closures'),
                   ('head_to_tail_patent_large',)]:
        candidates = [row for row in pool if row['peptide_stratum'] in strata]
        if len(strata) == 1:
            candidates.sort(key=lambda r: (r['n_atoms'], r['record_key']))
            # Cover the range, including large (>300-atom) cases.
            candidates = [candidates[int((len(candidates)-1)*q/3)] for q in range(4)] + candidates
        for stratum in strata:
            count = 0
            for row in candidates:
                family = str(Path(row['file_path']).parents[1])
                if row['peptide_stratum'] != stratum or family in seen: continue
                selected.append(row); seen.add(family); count += 1
                if count >= (4 if len(strata) == 1 else 1): break
    keys = {r['record_key'] for r in selected}
    manifest = Path('/home/dahuilangda/molsight_work/molsight_patent_merge_v5/admitted/manifest.json')
    info = json.loads(manifest.read_text()); lookup = {}
    with pa.memory_map(info['files']['train']['path'], 'r') as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            mask = pc.is_in(batch.column('record_key'), value_set=pa.array(sorted(keys)))
            rows = pa.Table.from_batches([batch]).select(['record_key','SMILES','component']).filter(mask).to_pylist()
            lookup.update({r['record_key']: r for r in rows})
    for row in selected:
        records.append(peptide_record({**row, **lookup[row['record_key']]}, 'train'))
    # Distinct chemistry and original validation split; never random heldout views.
    for i, s in enumerate(['CCN(CC)CC', 'N[C@@H](Cc1ccccc1)C(=O)O']):
        records.append(draw_small(s, out, 'validation', i))
    validation = []
    with pa.memory_map(info['files']['validation']['path'], 'r') as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            mask = pc.equal(batch.column('category'), 'peptide')
            validation.extend(pa.Table.from_batches([batch]).select([
                'record_key','SMILES','component','file_path','image_sha256','peptide_stratum','input_size']).filter(mask).to_pylist())
    used = {exact_key(row['SMILES']) for row in records}
    for stratum in ['linear_medium', 'head_to_tail_small', 'head_to_tail_patent_large']:
        matched = False
        for row in validation:
            if row.get('peptide_stratum') == stratum and exact_key(row['SMILES']) not in used:
                records.append(peptide_record(row, 'validation'))
                used.add(exact_key(row['SMILES'])); matched = True; break
        if not matched: raise ValueError(f'Missing validation stratum {stratum}')
    train_ids={exact_key(r['SMILES']) for r in records if r['split']=='train'}
    val_ids={exact_key(r['SMILES']) for r in records if r['split']=='validation'}
    if train_ids & val_ids: raise ValueError('Identity crosses diagnostic splits')
    payload={'purpose':'memorization gate plus development diagnostics; not blind real-world evidence',
             'source_manifest_sha256':digest(manifest),'records':records}
    path=out/'manifest.json';path.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'manifest':str(path),'train':sum(r['split']=='train' for r in records),
                      'validation':sum(r['split']=='validation' for r in records)}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    build_gate(parser.parse_args().out)

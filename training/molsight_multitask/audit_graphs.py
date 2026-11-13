"""Audit persisted sidecar joins and source-relative chemical graph identity."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import zlib

import pysqlite3 as sqlite3
import pyarrow as pa
from pyarrow import ipc
from rdkit import Chem, rdBase
from molsight_data.graph_supervision import rebuild_molecule
from molsight_multitask.export_graphs import digest, token_digest


def verify_targets(jobs):
    outcomes = Counter()
    for target_hash, smiles, tokens_hash, payload, error, supported in jobs:
        if hashlib.sha256(smiles.encode()).hexdigest() != target_hash:
            raise ValueError('Stored target digest differs from text')
        if payload is None:
            if not error or supported != 0:
                raise ValueError('Missing graph lacks explicit rejection')
            outcomes[error] += 1
            continue
        if error is not None:
            raise ValueError('Graph simultaneously marked rejected')
        annotation = json.loads(zlib.decompress(payload))
        if annotation['smiles'] != smiles or token_digest(annotation['token_ids']) != tokens_hash:
            raise ValueError('Graph target/token identity differs')
        if annotation['coordinate_targets'] is not None or annotation['drawing_bond_direction_targets'] is not None:
            raise ValueError('Chemical-only sidecar invents drawing supervision')
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        with rdBase.BlockLogs():
            source = Chem.MolFromSmiles(smiles, parser)
        if source is None or source.GetNumAtoms() != len(annotation['atoms']) or source.GetNumBonds() != len(annotation['bonds']):
            raise ValueError('Source atom/bond coverage differs')
        for index, (atom, record) in enumerate(zip(source.GetAtoms(), annotation['atoms'], strict=True)):
            if record['atom_id'] != index:
                raise ValueError('Graph atom order differs from source parse')
            expected = {'atomic_number': atom.GetAtomicNum(), 'isotope': atom.GetIsotope(),
                        'formal_charge': atom.GetFormalCharge(), 'aromatic': atom.GetIsAromatic(),
                        'atom_map': atom.GetAtomMapNum(), 'chiral_tag': str(atom.GetChiralTag()),
                        'explicit_hydrogens': atom.GetNumExplicitHs(), 'no_implicit': atom.GetNoImplicit(),
                        'radical_electrons': atom.GetNumRadicalElectrons()}
            if any(record[key] != value for key, value in expected.items()):
                raise ValueError('Persisted atom properties differ from source')
            start, end = record['token_start_with_sos'], record['token_end_with_sos_inclusive']
            if not 1 <= start <= end < len(annotation['token_ids']) - 1:
                raise ValueError('Atom span lies outside source token sequence')
            # Literal SMILES use character tokens in the unchanged vocabulary.
            if smiles[start - 1:end] != record['atom_token'] or annotation['atom_indices'][index] != end:
                raise ValueError('Atom query/token span changed')
        for bond, record in zip(source.GetBonds(), annotation['bonds'], strict=True):
            expected = {'begin': bond.GetBeginAtomIdx(), 'end': bond.GetEndAtomIdx(),
                        'chemical_type': str(bond.GetBondType()), 'aromatic': bond.GetIsAromatic(),
                        'stereo': str(bond.GetStereo()), 'stereo_atoms': list(bond.GetStereoAtoms()),
                        'smiles_bond_direction': str(bond.GetBondDir())}
            if any(record[key] != value for key, value in expected.items()):
                raise ValueError('Persisted bond properties differ from source')
        canonical = Chem.MolToSmiles(source, isomericSmiles=True)
        if Chem.MolToSmiles(rebuild_molecule(annotation), isomericSmiles=True) != canonical:
            raise ValueError('Persisted graph changes complete isomeric identity')
        expected_supported = int(all(bond['chemical_type'] in ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')
                                     for bond in annotation['bonds']))
        if supported != expected_supported:
            raise ValueError('Chemical edge eligibility differs from graph')
        outcomes['aligned'] += 1
    return outcomes


def audit(manifest, sidecar, output, workers=2):
    started = time.monotonic()
    audit_code_hash = digest(__file__)
    if workers < 1:
        raise ValueError('Positive auditor worker count is required')
    if Path(output).exists():
        raise FileExistsError(output)
    specification = json.loads(Path(manifest).read_text())
    admission = specification['admission_report']
    if specification.get('training_release_approved') is not True or digest(admission['path']) != admission['sha256']:
        raise ValueError('Source release admission changed')
    source_admission = json.loads(Path(admission['path']).read_text())
    if source_admission.get('passed') is not True or source_admission.get('records') != specification['records']:
        raise ValueError('Source release admission lacks coverage')
    sidecar = Path(sidecar)
    report_path = sidecar / 'report.json'
    report_hash = digest(report_path)
    report = json.loads(report_path.read_text())
    split = report['split']
    entry = specification['files'][split]
    database_path = sidecar / 'graphs.sqlite'
    if (report['state'] != 'complete' or report['manifest_sha256'] != digest(manifest)
            or report['arrow_sha256'] != entry['sha256'] or digest(entry['path']) != entry['sha256']
            or report['vocabulary_sha256'] != specification['vocabulary_sha256']
            or digest(specification['vocabulary']) != specification['vocabulary_sha256']
            or digest(database_path) != report['database_sha256']):
        raise ValueError('Sidecar/release identity or completion differs')
    if any(digest(path) != expected for path, expected in report['code_sha256'].items()):
        raise ValueError('Exporter/alignment implementation changed since export')
    db = sqlite3.connect('file:' + str(database_path.resolve()) + '?mode=ro', uri=True)
    if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)] or db.execute('PRAGMA foreign_key_check').fetchall():
        raise ValueError('Sidecar database integrity failed')
    with pa.memory_map(entry['path'], 'r') as source:
        table = ipc.open_file(source).read_all().select(
            ['record_key', 'component', 'image_sha256', 'SMILES', 'token_ids'])
        expected_rows = (row for batch in table.to_batches(max_chunksize=1024) for row in batch.to_pylist())
        actual_rows = db.execute('''SELECT row_index,record_key,component,image_sha256,
          targets.target_sha256,smiles,token_sha256 FROM records JOIN targets USING(target_sha256) ORDER BY row_index''')
        records = 0
        for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=True)):
            row_index, key, component, image_hash, target_hash, smiles, tokens_hash = actual
            if (row_index != index or key != expected['record_key'] or component != expected['component']
                    or image_hash != expected['image_sha256'] or smiles != expected['SMILES']
                    or target_hash != hashlib.sha256(smiles.encode()).hexdigest()
                    or tokens_hash != token_digest(expected['token_ids'])):
                raise ValueError(f'Sidecar row identity mismatch at {index}')
            records += 1
    if records != specification['records'][split] or records != report['records']:
        raise ValueError('Sidecar record coverage differs')
    print(json.dumps({'phase': 'record_joins_verified', 'records': records}), flush=True)
    pending, jobs, counts = set(), [], Counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
        for row in db.execute('SELECT target_sha256,smiles,token_sha256,graph_zlib,error,edge_supported FROM targets'):
            jobs.append(row)
            if len(jobs) == 32:
                pending.add(executor.submit(verify_targets, jobs))
                jobs = []
            if len(pending) >= workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    counts.update(future.result())
        if jobs:
            pending.add(executor.submit(verify_targets, jobs))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                counts.update(future.result())
    supported = db.execute('SELECT count(*) FROM records JOIN targets USING(target_sha256) WHERE edge_supported=1').fetchone()[0]
    db.close()
    if dict(counts) != report['target_outcomes'] or sum(counts.values()) != report['unique_targets'] or supported != report['edge_supervised_records']:
        raise ValueError('Persisted graph coverage differs from export report')
    if (digest(database_path) != report['database_sha256'] or digest(report_path) != report_hash
            or digest(__file__) != audit_code_hash or digest(manifest) != report['manifest_sha256']
            or digest(entry['path']) != entry['sha256']
            or any(digest(path) != expected for path, expected in report['code_sha256'].items())):
        raise ValueError('Sidecar changed during audit')
    result = {'passed': True, 'chemical_graph_sidecar_approved': True, 'split': split,
              'records': records, 'unique_targets': sum(counts.values()), 'target_outcomes': dict(counts),
              'edge_supervised_records': supported, 'export_report_sha256': report_hash,
              'database_sha256': report['database_sha256'], 'manifest_sha256': report['manifest_sha256'],
              'auditor_sha256': audit_code_hash, 'rdkit_version': rdBase.rdkitVersion,
              'seconds': time.monotonic() - started,
              'scope': 'Persisted joins and graph semantics relative to admitted source labels only. No drawing-coordinate, source-image label correctness, multitask trainer, or recognition capability approval.'}
    with Path(output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    audit(args.manifest, args.sidecar, args.out, args.workers)

"""Build an immutable graph sidecar for an admitted adaptation split.

Every source record remains represented. Unsupported annotations retain an
explicit reason and remain eligible for their original character objective.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pysqlite3 as sqlite3
import time
import zlib

import pyarrow as pa
from pyarrow import ipc
from molsight.tokenizer import CharTokenizer
from molsight_data.graph_supervision import AlignmentError, align_smiles

TOKENIZER = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def initialize(vocab):
    global TOKENIZER
    TOKENIZER = CharTokenizer(vocab)


def annotate_batch(jobs):
    results = []
    for target_hash, smiles, expected_tokens in jobs:
        try:
            graph = align_smiles(smiles, TOKENIZER)
        except AlignmentError as error:
            results.append((target_hash, None, str(error), 0))
            continue
        if token_digest(graph['token_ids']) != expected_tokens:
            raise ValueError('Graph encoding differs from admitted character target')
        supported = int(all(bond['chemical_type'] in ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')
                            for bond in graph['bonds']))
        payload = zlib.compress(json.dumps(graph, separators=(',', ':')).encode(), level=3)
        results.append((target_hash, payload, None, supported))
    return results


def export(manifest, split, output, workers):
    manifest_hash = digest(manifest)
    specification = json.loads(Path(manifest).read_text())
    if workers < 1 or not specification.get('training_release_approved'):
        raise ValueError('Positive worker count and admitted release are required')
    admission = specification['admission_report']
    if digest(admission['path']) != admission['sha256'] or not json.loads(Path(admission['path']).read_text())['passed']:
        raise ValueError('Admission evidence changed or failed')
    if json.loads(Path(admission['path']).read_text()).get('records') != specification['records']:
        raise ValueError('Admission does not cover the declared split counts')
    entry = specification['files'][split]
    if digest(entry['path']) != entry['sha256'] or digest(specification['vocabulary']) != specification['vocabulary_sha256']:
        raise ValueError('Admitted Arrow or vocabulary changed')
    import molsight.tokenizer as tokenizer_module
    import molsight_data.graph_supervision as alignment_module
    sources = {str(Path(path).resolve()): digest(path) for path in
               (__file__, tokenizer_module.__file__, alignment_module.__file__)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    database_path = output / 'graphs.sqlite'
    database = sqlite3.connect(database_path)
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('PRAGMA foreign_keys=ON')
    database.executescript('''
        CREATE TABLE targets(target_sha256 TEXT PRIMARY KEY, smiles TEXT NOT NULL,
          token_sha256 TEXT NOT NULL, graph_zlib BLOB, error TEXT, edge_supported INTEGER);
        CREATE TABLE records(record_key TEXT PRIMARY KEY, row_index INTEGER UNIQUE NOT NULL,
          component TEXT NOT NULL, image_sha256 TEXT NOT NULL,
          target_sha256 TEXT NOT NULL REFERENCES targets(target_sha256));
    ''')
    known, pending, jobs = {}, set(), []
    started = time.monotonic()
    count = 0

    def collect(futures):
        for future in futures:
            for target_hash, payload, error, supported in future.result():
                database.execute('UPDATE targets SET graph_zlib=?,error=?,edge_supported=? WHERE target_sha256=?',
                                 (payload, error, supported, target_hash))

    def progress():
        document = {'state': 'running', 'records': count, 'unique_targets': len(known),
                    'seconds': time.monotonic() - started}
        temporary = output / 'progress.tmp'
        temporary.write_text(json.dumps(document) + '\n')
        temporary.replace(output / 'progress.json')
        print(json.dumps(document), flush=True)

    with pa.memory_map(entry['path'], 'r') as source, ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context('spawn'),
            initializer=initialize, initargs=(specification['vocabulary'],)) as executor:
        table = ipc.open_file(source).read_all()
        columns = table.select(['record_key', 'component', 'image_sha256', 'SMILES', 'token_ids'])
        for batch in columns.to_batches(max_chunksize=1024):
            for row in batch.to_pylist():
                smiles = row['SMILES']
                key = hashlib.sha256(smiles.encode()).hexdigest()
                tokens_hash = token_digest(row['token_ids'])
                if key in known:
                    if known[key] != tokens_hash:
                        raise ValueError('Identical structure text has different admitted tokens')
                else:
                    known[key] = tokens_hash
                    database.execute('INSERT INTO targets(target_sha256,smiles,token_sha256) VALUES(?,?,?)',
                                     (key, smiles, tokens_hash))
                    jobs.append((key, smiles, tokens_hash))
                    if len(jobs) == 32:
                        pending.add(executor.submit(annotate_batch, jobs))
                        jobs = []
                    if len(pending) >= workers * 2:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        collect(done)
                database.execute('INSERT INTO records VALUES(?,?,?,?,?)',
                                 (row['record_key'], count, row['component'], row['image_sha256'], key))
                count += 1
            database.commit()
            if count % 10240 == 0:
                progress()
        if jobs:
            pending.add(executor.submit(annotate_batch, jobs))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
        database.commit()
    if count != specification['records'][split]:
        raise ValueError('Sidecar does not cover the full admitted split')
    if database.execute('SELECT count(*) FROM targets WHERE graph_zlib IS NULL AND error IS NULL').fetchone()[0]:
        raise ValueError('Unfinished graph annotations')
    if (any(digest(path) != value for path, value in sources.items())
            or digest(entry['path']) != entry['sha256'] or digest(manifest) != manifest_hash
            or digest(specification['vocabulary']) != specification['vocabulary_sha256']
            or digest(admission['path']) != admission['sha256']):
        raise ValueError('Code or Arrow changed during export')
    counts = dict(database.execute('SELECT COALESCE(error,"aligned"),count(*) FROM targets GROUP BY error'))
    supported_records = database.execute('SELECT count(*) FROM records JOIN targets USING(target_sha256) WHERE edge_supported=1').fetchone()[0]
    database.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    database.close()
    report = {'schema_version': 1, 'state': 'complete', 'split': split, 'records': count,
              'unique_targets': len(known), 'target_outcomes': counts, 'edge_supervised_records': supported_records,
              'manifest_sha256': digest(manifest), 'arrow_sha256': entry['sha256'],
              'vocabulary_sha256': specification['vocabulary_sha256'], 'code_sha256': sources,
              'database_sha256': digest(database_path), 'seconds': time.monotonic() - started,
              'scope': 'Chemical-graph sidecar only; no drawing coordinates/wedges. Unsupported targets keep character supervision. Not yet admitted for multitask training.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'progress.json').write_text(json.dumps({'state': 'complete', 'records': count}) + '\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    export(args.manifest, args.split, args.out, args.workers)

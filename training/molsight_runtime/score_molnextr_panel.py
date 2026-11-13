"""Score MolNexTR outputs with the common MolSight typed-graph evaluator."""
import argparse
import json
from pathlib import Path
from molsight.evaluation import aggregate, compare
from molsight import evaluation, structure, abbreviations
from rdkit import rdBase
from molsight.prepared import file_hash


def score(directory, pairs_path):
    directory = Path(directory)
    metadata = json.loads((directory / 'execution.json').read_text())
    if metadata['state'] != 'complete' or metadata['pairs_sha256'] != file_hash(pairs_path):
        raise ValueError('Inference incomplete or panel identity differs')
    prediction_path = directory / 'predictions.jsonl'
    if metadata['predictions_sha256'] != file_hash(prediction_path):
        raise ValueError('Predictions changed')
    pairs = json.loads(Path(pairs_path).read_text())
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if len(predictions) != len(pairs) or metadata['records'] != len(pairs):
        raise ValueError('Prediction coverage differs')
    rows = []
    for index, (prediction, pair) in enumerate(zip(predictions, pairs, strict=True)):
        if (prediction['index'] != index or prediction['record_key'] != pair['record_key']
                or prediction['gold'] != pair['SMILES'] or prediction['category'] != pair['category']):
            raise ValueError('Prediction record identity differs')
        issues = prediction['quality_issues']
        terminated = prediction['error'] is None and not any('missing_eos' in str(issue) for issue in issues)
        rows.append({**prediction, 'terminated': terminated,
                     **compare(pair['SMILES'], prediction['prediction'])})
    report = {'records': len(rows), 'pairs_sha256': metadata['pairs_sha256'],
              'prediction_sha256': metadata['predictions_sha256'], 'inference': metadata,
              'metrics': aggregate(rows), 'peptide_strata': aggregate(rows, group_by='peptide_stratum'),
              'evaluator_source_sha256': file_hash(__file__),
              'evaluator_rdkit': rdBase.rdkitVersion,
              'evaluator_contract_sha256': {module.__name__: file_hash(module.__file__)
                                          for module in (evaluation, structure, abbreviations)},
              'scope': 'Native local MolNexTR pipeline, including graph assembly/postprocessing, on the same development images. Common typed graph scorer. CPU inference; no matched-hardware latency comparison. Termination is inferred from local missing-EOS guards, not raw-token telemetry.'}
    with (directory / 'scored_predictions.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    with (directory / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'metrics': report['metrics'], 'peptide_strata': report['peptide_strata']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    args = parser.parse_args()
    score(args.directory, args.pairs)

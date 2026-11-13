"""Persistent JSON-line serving for the validated conservative MolSight hybrid."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'molsight_runtime'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from training.molsight_dense.artifact import load_artifact
    from training.molsight_hybrid import predict_file
    artifact = Path(os.environ['MOLSIGHT_ARTIFACT'])
    branch_policy = os.environ.get('MOLSIGHT_POLICY', 'peptide_visible')
    if branch_policy not in ('preserve_ar', 'cyclic_visible', 'peptide_visible'):
        raise ValueError(f'Unknown MOLSIGHT_POLICY: {branch_policy}')
    try:
        graph_sizes = [int(x) for x in os.environ.get('MOLSIGHT_GRAPH_SIZES', '768 1024 512').split()]
    except ValueError as error:
        raise ValueError(f'MOLSIGHT_GRAPH_SIZES must be integers: {error}') from error
    if not graph_sizes or any(n < 128 or n % 32 for n in graph_sizes) \
            or len(set(graph_sizes)) != len(graph_sizes):
        raise ValueError(f'Invalid MOLSIGHT_GRAPH_SIZES: {graph_sizes}')
    policy_path = os.environ.get('MOLSIGHT_INFERENCE_POLICY')
    threshold_override = None
    if policy_path:
        policy = json.loads(Path(policy_path).read_text())
        if policy['artifact_sha256'] != hashlib.sha256((artifact / 'artifact.json').read_bytes()).hexdigest():
            raise ValueError('Policy/artifact mismatch')
        threshold_override = policy['threshold']
    model, metadata = load_artifact(artifact, args.device)
    if threshold_override is not None:
        model.head.config = replace(model.head.config, peak_threshold=threshold_override)
    print(json.dumps({'event': 'ready', 'model_sha256': metadata['model_sha256'],
                      'policy': branch_policy, 'threshold': model.head.config.peak_threshold,
                      'graph_sizes': graph_sizes}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = predict_file(model, metadata, request['image_path'], policy=branch_policy,
                                  graph_sizes=graph_sizes if len(graph_sizes) > 1 else None)
            result.update(model_sha256=metadata.get('model_sha256'), artifact=str(artifact),
                          ok=result['parse_success'], raw=result['ar_prediction'],
                          chemical_repair_applied=False)
        except Exception as error:
            result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

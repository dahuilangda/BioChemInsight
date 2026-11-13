"""Opt-in strict worker for workspace-trained MolSight checkpoints.

Use MOLSIGHT_WORKER to select this script; the legacy service is unchanged.
No chemical repair is applied to the primary recognition result.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
RUNTIME = Path(__file__).resolve().parents[1] / 'training' / 'molsight_runtime'


def configuration_for(checkpoint, state, vocabulary, *, input_size, max_len):
    from molsight.checkpoint import model_configuration
    if not isinstance(checkpoint.get('plan'), dict):
        raise ValueError('Workspace worker requires a recorded training plan')
    if checkpoint.get('vocabulary') != vocabulary:
        raise ValueError('Checkpoint and runtime token identifiers differ')
    plan = checkpoint['plan']
    # The trainer records the vocabulary file hash, which depends on JSON
    # whitespace; exact identifier equality above is the semantic check.
    configuration = dict(plan['configuration'])
    for key, value in model_configuration(state).items():
        recorded = configuration.get(key)
        if isinstance(value, tuple):
            recorded = tuple(recorded) if recorded is not None else None
        if recorded != value:
            raise ValueError(f'Architecture contradicts checkpoint tensors: {key}')
    if type(configuration.get('dec_n_head')) is not int or configuration['dec_n_head'] <= 0:
        raise ValueError('Missing recorded attention head count')
    if configuration.get('lora'):
        raise ValueError('Adapter checkpoints require a separate validated merge')
    if input_size not in (512, 768, 1024) or not 1 <= max_len <= configuration['max_len']:
        raise ValueError('Unsupported resolution or decode limit')
    configuration.update(input_size=input_size, max_len=max_len, resume=True,
                         lora=False, use_checkpoint=False, compile_layers=False)
    return SimpleNamespace(**configuration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=os.environ.get('MOLSIGHT_CHECKPOINT'))
    parser.add_argument('--vocab', default=os.environ.get('MOLSIGHT_VOCAB'))
    parser.add_argument('--input-size', type=int, default=int(os.environ.get('MOLSIGHT_INPUT_SIZE', '512')))
    parser.add_argument('--max-len', type=int, default=int(os.environ.get('MOLSIGHT_MAX_LEN', '2048')))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not args.checkpoint or not args.vocab:
        parser.error('An explicit workspace checkpoint and matching vocabulary are required')
    if os.environ.get('MOLSIGHT_ADAPTER'):
        parser.error('This worker does not apply legacy adapters')
    sys.path.insert(0, str(RUNTIME))
    import torch
    from molsight.checkpoint import load_parameters, model_state
    from molsight.inference import predict_images
    from molsight.model import MolsightModel
    from molsight.prepared import ImageReader, file_hash
    from molsight.tokenizer import CharTokenizer
    from molsight.structure import parse_structure, StructureError

    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = model_state(checkpoint)
    vocabulary = json.loads(Path(args.vocab).read_text())
    configuration = configuration_for(checkpoint, state, vocabulary,
                                      input_size=args.input_size, max_len=args.max_len)
    if checkpoint['plan']['vocabulary_sha256'] != file_hash(args.vocab):
        raise ValueError('Vocabulary file hash differs from the recorded training plan')
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('Checkpoint contains nonfinite model tensors')
    tokenizer = CharTokenizer(args.vocab)
    model = MolsightModel(configuration, tokenizer)
    load_parameters(model, state, source_vocabulary=vocabulary, target_vocabulary=vocabulary)
    model.requires_grad_(False).eval().to(args.device)
    metadata = {'runtime': 'workspace', 'checkpoint_sha256': file_hash(args.checkpoint),
                'vocabulary_sha256': file_hash(args.vocab), 'input_size': args.input_size,
                'max_len': args.max_len, 'precision': 'fp32', 'update': checkpoint['update']}
    del checkpoint, state
    reader = ImageReader()
    print(json.dumps({'event': 'ready', 'missing': 0, 'unexpected': 0, **metadata}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = request.get('image_path')
            if not isinstance(image_path, str) or not Path(image_path).is_file():
                raise ValueError('Image not found')
            preprocessed = request.get('model_input_preprocessed', False)
            if type(preprocessed) is not bool:
                raise ValueError('model_input_preprocessed must be boolean')
            row = {'file_path': image_path, 'image_sha256': file_hash(image_path),
                   'model_input_preprocessed': preprocessed}
            tensor = reader(row, args.input_size).unsqueeze(0).to(args.device)
            prediction = predict_images(model, tensor)
            raw = prediction['smiles'][0]
            terminated = bool(prediction['terminated'][0])
            try:
                parse_structure(raw)
                parsed = True
            except StructureError:
                parsed = False
            response = {'ok': parsed and terminated, 'smiles': raw, 'raw': raw,
                        'terminated': terminated, 'parse_success': parsed,
                        'postprocess_ok': False, 'chemical_repair_applied': False,
                        'precision': 'fp32', 'input_size': args.input_size}
            if not response['ok']:
                response['error'] = 'Recognition did not terminate' if not terminated else 'Prediction is not parseable'
        except Exception as error:
            response = {'ok': False, 'error': f'{type(error).__name__}: {str(error)[:300]}'}
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()

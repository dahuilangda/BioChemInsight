"""CPU graph-loss backward through an explicitly extended real checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.compute as pc
import torch
from molsight.checkpoint import load_parameters, model_state
from molsight.model import MolsightModel
from molsight.prepared import PreparedDataset, file_hash
from molsight.tokenizer import CharTokenizer
from molsight_multitask.edges import chemical_edge_sums, chemical_targets


def run(checkpoint_path, manifest, graphs, output):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    original = model_state(checkpoint)
    specification = json.loads(Path(manifest).read_text())
    tokenizer = CharTokenizer(specification['vocabulary'])
    if checkpoint['vocabulary'] != tokenizer.stoi:
        raise ValueError('Vocabulary differs')
    candidates = [json.loads(line) for line in Path(graphs).read_text().splitlines()]
    eligible = [row for row in candidates if row['group'] == 'complete']
    if not eligible:
        raise ValueError('Pilot has no complete-molecule case')
    graph = min(eligible, key=lambda row: len(row['token_ids']))
    dataset = PreparedDataset(manifest, 'train')
    dataset._open()
    indices = pc.indices_nonzero(pc.equal(dataset.table['record_key'], graph['record_key'])).to_pylist()
    if len(indices) != 1:
        raise ValueError('Pilot identity is not unique in training export')
    sample = dataset[indices[0]]
    if sample['label'].tolist() != graph['token_ids'] or sample['smiles'] != graph['smiles']:
        raise ValueError('Pilot target changed')
    configuration = dict(checkpoint['plan']['configuration'])
    configuration.update(formats=['char', 'edges'], use_checkpoint=True, compile_layers=False)
    torch.manual_seed(42)
    model = MolsightModel(SimpleNamespace(**configuration), tokenizer)
    destinations = model.state_dict()
    new_keys = set(destinations) - set(original)
    if not new_keys or any(not key.startswith('decoder.edge_predictor.') for key in new_keys):
        raise ValueError('Unexpected new parameter scope')
    if set(original) - set(destinations):
        raise ValueError('Existing checkpoint parameters would be dropped')
    migrated = {**original, **{key: destinations[key] for key in new_keys}}
    load_parameters(model, migrated, source_vocabulary=tokenizer.stoi, target_vocabulary=tokenizer.stoi)
    model.train()
    model.image_encoder.eval()
    prediction = model(image=sample['image'][None], label=sample['label'][None],
                       atom_indices=torch.tensor(graph['atom_indices'])[None])
    terms = chemical_edge_sums(prediction['edge_pred'], chemical_targets(graph)[None])
    loss = terms['loss_sum'] / terms['images'].clamp_min(1)
    loss.backward()
    gradients = {}
    for group, prefix in [('encoder', 'image_encoder.'), ('decoder', 'decoder.blocks.'),
                          ('new_edge_head', 'decoder.edge_predictor.')]:
        values = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not values or not all(torch.isfinite(value).all() for value in values):
            raise ValueError(f'Missing or nonfinite gradients: {group}')
        norm = torch.stack([value.float().norm() for value in values]).norm().item()
        if norm <= 0:
            raise ValueError(f'Zero graph-loss gradient: {group}')
        gradients[group] = {'tensor_count': len(values), 'l2_norm': norm}
    unchanged = all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    if not unchanged:
        raise ValueError('Original checkpoint state changed without optimizer update')
    report = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_hash(checkpoint_path),
              'manifest_sha256': file_hash(manifest), 'graph_pilot_sha256': file_hash(graphs),
              'record_key': graph['record_key'], 'split': 'train', 'device': 'cpu',
              'input_size': sample['image'].shape[-1], 'tokens': len(graph['token_ids']),
              'atoms': len(graph['atoms']), 'chemical_edge_loss': loss.item(),
              'new_parameter_keys': sorted(new_keys), 'gradients': gradients,
              'original_state_unchanged': unchanged, 'optimizer_updates': 0,
              'scope': 'One real image graph-loss backward only. No DDP/BF16/capability proof; chemical-only head cannot be decoded as drawing stereochemistry.'}
    with Path(output).open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'graphs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.manifest, args.graphs, args.out)

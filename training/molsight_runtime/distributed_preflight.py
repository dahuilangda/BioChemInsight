"""Compare production DDP accumulation with the same global supervised batch."""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from molsight.checkpoint import load_parameters, model_configuration, model_state
from molsight.dataset import get_transforms
from molsight.loss import Criterion
from molsight.model import MolsightModel
from molsight.optimization import supervised_update
from molsight.structure import parse_structure
from molsight.tokenizer import CharTokenizer
from PIL import Image
from torch.nn.parallel import DistributedDataParallel


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GradientCapture(torch.optim.SGD):
    def step(self, closure=None):
        self.gradients = [parameter.grad.detach().clone() if parameter.grad is not None else None
                          for group in self.param_groups for parameter in group['params']]
        return super().step(closure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-vocab', type=Path, required=True)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--max-len', type=int, default=2048)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        tokenizer = CharTokenizer(args.vocab)
        state = model_state(torch.load(args.checkpoint, map_location='cpu', mmap=True,
                                       weights_only=False))
        configuration = SimpleNamespace(
            **model_configuration(state), dec_n_head=8, max_len=args.max_len,
            input_size=args.input_size, encoder='efficientvit', resume=True,
            lora=False, use_checkpoint=True, formats=['char'], label_smoothing=.1,
            regression=False,
        )
        torch.manual_seed(914)
        model = MolsightModel(configuration, tokenizer)
        load_parameters(model, state,
                        source_vocabulary=json.loads(args.source_vocab.read_text()),
                        target_vocabulary=tokenizer.stoi)
        model.image_encoder.requires_grad_(False)
        model.eval().to(device)
        transform = get_transforms(configuration, augment=False, rotate=False)
        pairs = json.loads(args.pairs.read_text())
        if len(pairs) < world or len(pairs) % world:
            raise ValueError('The production check requires an equal number of microbatches per rank')
        batches = []
        for pair in pairs:
            target = parse_structure(pair['SMILES']).model_text
            tokens, _ = tokenizer.smiles_to_sequence(target)
            if len(tokens) > args.max_len:
                raise ValueError('Preflight target exceeds the context')
            with Image.open(pair['file_path']) as image:
                pixels = np.asarray(image.convert('RGB'))
            pixels = transform(image=pixels, keypoints=[])['image']
            batches.append({
                'image': pixels[None], 'label': torch.tensor([tokens]),
                'atom_indices': torch.empty((1, 0), dtype=torch.long),
                'edges': torch.empty((1, 0, 0), dtype=torch.long),
                'coords': torch.empty((1, 0, 2)),
            })
        criterion = Criterion(configuration, tokenizer).to(device)
        parameters = list(model.parameters())
        before = parameter_digest(model)
        wrapper = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = GradientCapture(parameters, lr=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        torch.cuda.reset_peak_memory_stats(device)
        losses, norm, stepped = supervised_update(
            wrapper, criterion, batches[rank::world], optimizer, scheduler,
            torch.amp.GradScaler('cuda', enabled=False), device, float('inf'), amp=args.amp,
        )
        unchanged = parameter_digest(model) == before
        unchanged_flag = torch.tensor(int(unchanged), device=device)
        dist.all_reduce(unchanged_flag, op=dist.ReduceOp.MIN)
        report = None
        if rank == 0:
            normalizers = {}
            for batch in batches:
                for name, value in criterion.normalizers(batch).items():
                    normalizers[name] = normalizers.get(name, 0) + value
            normalizers = {name: value.to(device) for name, value in normalizers.items()}
            reference_loss = torch.zeros((), device=device)
            for host in batches:
                batch = {name: value.to(device) for name, value in host.items()}
                context = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else nullcontext()
                with context:
                    loss = sum(criterion(**model(**batch), **batch,
                                         normalizers=normalizers, world_size=1).values())
                reference_loss += loss.detach()
                loss.backward()
            difference, magnitude = 0.0, 0.0
            maximum = 0.0
            missing = []
            for index, (parameter, actual) in enumerate(zip(parameters, optimizer.gradients)):
                expected = parameter.grad
                if (actual is None) != (expected is None):
                    missing.append(index)
                    continue
                if expected is None:
                    continue
                delta = actual.float() - expected.float()
                difference += delta.square().sum().item()
                magnitude += expected.float().square().sum().item()
                maximum = max(maximum, delta.abs().max().item())
            relative = (difference / max(magnitude, 1e-30)) ** .5
            loss_error = abs(sum(losses.values()) - reference_loss.item())
            tolerance = .005 if args.amp else 1e-5
            passed = (not missing and relative < tolerance and loss_error < tolerance
                      and bool(unchanged_flag.item()) and stepped)
            report = {
                'passed': passed, 'checkpoint': str(args.checkpoint),
                'configuration': vars(configuration), 'world_size': world,
                'microbatches_per_rank': len(batches) // world, 'bf16': args.amp,
                'distributed_losses': losses, 'reference_loss': reference_loss.item(),
                'loss_absolute_error': loss_error, 'gradient_relative_l2_error': relative,
                'gradient_max_absolute_error': maximum, 'missing_gradients': missing,
                'gradient_norm': float(norm), 'parameters_unchanged': bool(unchanged_flag.item()),
                'parameter_sha256': before, 'parameter_updates': 0,
                'optimizer_steps': 1, 'learning_rate': 0,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        verdict = [report['passed'] if rank == 0 else None]
        dist.broadcast_object_list(verdict, src=0, device=device)
        if not verdict[0]:
            raise RuntimeError('Production distributed accumulation differs from the global batch')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

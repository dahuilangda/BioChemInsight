from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from molsight_multitask.objective import GraphCriterion
from molsight_multitask.update import multitask_update


class Tokenizer:
    stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'C': 3, 'O': 4}
    def __len__(self):
        return 5


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.decoder = nn.Linear(8, 5)
        self.edge_head = nn.Linear(8, 7)
        self.unused = nn.Parameter(torch.ones(3))

    def forward(self, image, label, atom_indices=None, **kwargs):
        feature = self.encoder(image).tanh()
        logits = self.decoder(feature)[:, None].expand(-1, label.shape[1], -1)
        edge = None
        if atom_indices is not None:
            count = atom_indices.shape[1]
            edge = self.edge_head(feature)[:, None, None].expand(-1, count, count, -1)
        return {'logits': logits, 'edge_pred': edge, 'loc_pred': None}


class Replay:
    def losses(self, model, batch, criterion, normalizers, world_size, *, amp):
        result = model(**batch)
        losses = criterion(**result, **batch, normalizers=normalizers, world_size=world_size)
        return {'retention_sequence': 0.25 * sum(losses.values())}


class CaptureSGD(torch.optim.SGD):
    def step(self, closure=None):
        self.captured = [None if p.grad is None else p.grad.detach().clone()
                         for group in self.param_groups for p in group['params']]
        return super().step(closure)


def batch(offset, *, graph, short=False):
    label = torch.tensor([[1, 3, 2, 0] if short else [1, 3, 4, 2]])
    return {'image': torch.arange(offset, offset + 4, dtype=torch.float32)[None] / 10,
            'label': label, 'atom_indices': torch.tensor([[1, 2]]) if graph else torch.empty(1, 0, dtype=torch.long),
            'chemical_edges': torch.tensor([[[-100, 1], [1, -100]]]) if graph else torch.empty(1, 0, 0, dtype=torch.long)}


def inputs(rank, *, graph=True):
    batches = [batch(rank * 8, graph=graph and rank == 0, short=rank == 1),
               batch(rank * 8 + 4, graph=False, short=True)]
    if rank == 1:
        batches[0] = {key: value.repeat_interleave(2, dim=0) for key, value in batches[0].items()}
    anchor = {'image': torch.full((1, 4), float(rank + 1)),
              'label': torch.tensor([[1, 4, 2, 0] if rank == 0 else [1, 4, 3, 2]])}
    return batches, anchor


def setup_model():
    torch.manual_seed(53)
    return TinyModel()


def criterion():
    return GraphCriterion(SimpleNamespace(label_smoothing=0.1), Tokenizer(), edge_weight=0.3)


def anchor_inputs(rank, multiple):
    anchor = inputs(rank)[1]
    if not multiple:
        return [{'name': '', 'weight': 1., 'batch': anchor}]
    extra = {'image': anchor['image'] + 1,
             'label': torch.tensor([[1, 3, 3, 2] if rank == 0 else [1, 3, 2, 0]])}
    return [{'name': 'complete', 'weight': .6, 'batch': anchor},
            {'name': 'peptide', 'weight': .4, 'batch': extra}]


def distributed_worker(rank, initialization, output, graph, multiple):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=initialization, rank=rank, world_size=2)
    try:
        model = setup_model()
        distributed = DistributedDataParallel(model)
        optimizer = CaptureSGD(model.parameters(), lr=0.03, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        batches, _ = inputs(rank, graph=graph)
        values = anchor_inputs(rank, multiple)
        anchor = values if multiple else values[0]['batch']
        metrics, norm, stepped = multitask_update(distributed, criterion(), batches, optimizer, scheduler,
            torch.amp.GradScaler('cpu', enabled=False), 'cpu', float('inf'), retention=Replay(), anchor_batch=anchor)
        torch.save({'state': model.state_dict(), 'gradients': optimizer.captured, 'metrics': metrics,
                    'norm': norm, 'stepped': stepped, 'scheduler': scheduler.last_epoch}, Path(output) / f'{rank}.pth')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('graph', [True, False])
@pytest.mark.parametrize('multiple', [False, True])
def test_ddp_accumulation_and_character_only_replay_match_global_batch(tmp_path, graph, multiple):
    model = setup_model()
    loss_function = criterion()
    all_batches, anchors = [], []
    for rank in range(2):
        batches, anchor = inputs(rank, graph=graph)
        all_batches.extend(batches)
        anchors.append(anchor_inputs(rank, multiple))
    normalizers = {'sequence': sum(b['label'][:, 1:].ne(0).sum().float() for b in all_batches),
                   'chemical_edges': torch.tensor(sum(b['chemical_edges'].ge(0).any().item() for b in all_batches))}
    total = 0
    for value in all_batches:
        arguments = dict(value)
        if not value['chemical_edges'].ge(0).any():
            arguments['atom_indices'] = None
        total = total + sum(loss_function(**model(**arguments), **value, normalizers=normalizers).values())
    for pool in range(len(anchors[0])):
        pool_normalizers = {'sequence': sum(rank[pool]['batch']['label'][:, 1:].ne(0).sum().float() for rank in anchors),
                            'chemical_edges': torch.tensor(0.)}
        for rank in anchors:
            value = rank[pool]['batch']
            total = total + .25 * rank[pool]['weight'] * sum(
                loss_function(**model(**value), **value, normalizers=pool_normalizers).values())
    total.backward()
    optimizer = CaptureSGD(model.parameters(), lr=0.03, weight_decay=0.1)
    optimizer.step()
    mp.spawn(distributed_worker, args=(f'file://{tmp_path / "init"}', str(tmp_path), graph, multiple), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f'{rank}.pth', weights_only=False)
        assert actual['stepped'] and actual['scheduler'] == 1
        for key, expected in model.state_dict().items():
            torch.testing.assert_close(actual['state'][key], expected, atol=2e-7, rtol=2e-6)
        for expected, gradient in zip(optimizer.captured, actual['gradients'], strict=True):
            if expected is None:
                assert gradient is None
            else:
                torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=2e-6)
        assert actual['state']['unused'].eq(1).all()


def test_nonfinite_multitask_objective_does_not_step_parameters_or_schedule():
    model = setup_model()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = CaptureSGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    batches, _ = inputs(0)
    batches[0]['image'][0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        multitask_update(model, criterion(), batches, optimizer, scheduler,
                         torch.amp.GradScaler('cpu', enabled=False), 'cpu', 1.)
    assert scheduler.last_epoch == 0
    for key, expected in original.items():
        torch.testing.assert_close(model.state_dict()[key], expected)

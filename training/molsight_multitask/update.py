"""One verified effective-batch update including optional character replay."""
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist
from molsight.optimization import optimizer_update, update_normalizers
from molsight.policy import policy_mode


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward_graph_batch(model, batch):
    arguments = dict(batch)
    targets = batch.get('chemical_edges')
    if targets is None or not targets.ge(0).any():
        arguments['atom_indices'] = None
    return model(**arguments)


def finite_loss(loss, world_size):
    finite = torch.isfinite(loss).to(torch.int32)
    if world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite:
        raise FloatingPointError('Nonfinite multitask objective')


def synchronize_gradients(model, device, world_size):
    if world_size == 1:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    used = torch.tensor([parameter.grad is not None for parameter in parameters],
                        device=device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    for parameter, active in zip(parameters, used.tolist()):
        if not active:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def multitask_update(model, criterion, batches, optimizer, scheduler, scaler, device,
                     max_grad_norm, *, amp=False, retention=None, anchor_batch=None, observer=None):
    if not batches or (retention is None) != (anchor_batch is None):
        raise ValueError('Nonempty batches and paired replay objective/batch are required')
    normalizers, world_size = update_normalizers(criterion, batches, device)
    optimizer.zero_grad(set_to_none=True)
    totals = {name: torch.zeros((), device=device) for name in normalizers}
    for host_batch in batches:
        batch = move_batch(host_batch, device)
        context = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
        with context:
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                result = forward_graph_batch(model, batch)
                losses = criterion(**result, **batch, normalizers=normalizers, world_size=world_size)
                loss = sum(losses.values())
            if observer is not None:
                observer(result, batch)
            finite_loss(loss, world_size)
            scaler.scale(loss).backward()
        for name, value in losses.items():
            totals[name] += value.detach()
    if retention is not None:
        anchors = anchor_batch if isinstance(anchor_batch, list) else [
            {'name': '', 'weight': 1., 'batch': anchor_batch}]
        if (not anchors or len({item['name'] for item in anchors}) != len(anchors)
                or any(not math.isfinite(item['weight']) or item['weight'] < 0 for item in anchors)
                or not math.isclose(sum(item['weight'] for item in anchors), 1., abs_tol=1e-9)):
            raise ValueError('Replay pools require unique names and normalized nonnegative weights')
        for item in anchors:
            anchor = move_batch(item['batch'], device)
            anchor_normalizers, _ = update_normalizers(criterion, [anchor], device)
            context = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
            with context, policy_mode(model, adapters=None):
                losses = retention.losses(model, anchor, criterion, anchor_normalizers, world_size, amp=amp)
                losses = {name: value * item['weight'] for name, value in losses.items()}
                loss = sum(losses.values())
                finite_loss(loss, world_size)
                scaler.scale(loss).backward()
            prefix = item['name'] + ':' if item['name'] else ''
            totals.update({prefix + name: value.detach() for name, value in losses.items()})
    # All forwards/backwards above suppress DDP reduction. In particular the
    # final replay need not use the edge head; used masks include every rank.
    synchronize_gradients(model, device, world_size)
    norm, stepped = optimizer_update(model, optimizer, scheduler, scaler, max_grad_norm)
    names = sorted(totals)
    packed = torch.stack([totals[name] for name in names])
    if world_size > 1:
        dist.all_reduce(packed)
        packed /= world_size
    return dict(zip(names, packed.tolist())), norm, stepped

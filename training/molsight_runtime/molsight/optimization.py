"""Effective-batch normalization and optimizer updates."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from .policy import policy_mode


def update_normalizers(criterion, batches, device):
    totals = {}
    for batch in batches:
        for name, value in criterion.normalizers(batch).items():
            totals[name] = totals.get(name, 0) + value
    names = sorted(totals)
    packed = torch.stack([totals[name] for name in names]).to(device)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(packed)
    return dict(zip(names, packed.unbind())), world_size


def optimizer_update(model, optimizer, scheduler, scaler, max_grad_norm):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise FloatingPointError("Nonfinite gradient norm")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= previous_scale
    if stepped:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return norm, stepped


def policy_update(
    model,
    criterion,
    batches,
    reward_function,
    optimizer,
    scheduler,
    scaler,
    device,
    max_grad_norm,
    amp=False,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    count = torch.tensor(
        sum(batch["image"].shape[0] for batch in batches),
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(count)
    if count <= 0:
        raise ValueError("Policy updates require at least one image")
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    metrics = {}
    weights = {}
    for batch in batches:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["smiles"]
        if len(targets) != len(images):
            raise ValueError("Each policy image requires one target")

        def reward(predictions):
            if len(predictions) % len(targets):
                raise ValueError("Incomplete completion groups")
            group_size = len(predictions) // len(targets)
            expanded = [target for target in targets for _ in range(group_size)]
            return reward_function(predictions, expanded)

        with model.no_sync() if hasattr(model, "no_sync") else nullcontext():
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp
            ):
                loss, statistics = criterion(model, images, reward)
                objective = loss * (len(images) * world_size / count)
            finite = torch.isfinite(objective).to(torch.int32)
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Nonfinite policy objective")
            scaler.scale(objective).backward()
        total_loss += loss.detach() * len(images)
        for name, value in statistics.items():
            weight = len(images)
            if name == "clip_fraction":
                weight *= statistics.get("tokens_per_image", 1.0)
            metrics[name] = metrics.get(name, 0.0) + float(value) * weight
            weights[name] = weights.get(name, 0.0) + weight
    # Synchronize once, including parameters used on only some ranks.
    if world_size > 1:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        used = torch.tensor(
            [parameter.grad is not None for parameter in parameters],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        for parameter, active in zip(parameters, used.tolist()):
            if not active:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)
    norm, stepped = optimizer_update(model, optimizer, scheduler, scaler, max_grad_norm)
    names = sorted(metrics)
    packed = torch.cat(
        (
            total_loss.reshape(1),
            torch.tensor(
                [metrics[name] for name in names] + [weights[name] for name in names],
                device=device,
            ),
        )
    )
    if world_size > 1:
        dist.all_reduce(packed)
    values = packed[1 : 1 + len(names)] / packed[1 + len(names) :].clamp_min(1)
    return (
        {"loss": (packed[0] / count).item(), **dict(zip(names, values.tolist()))},
        norm,
        stepped,
    )


def supervised_update(
    model,
    criterion,
    batches,
    optimizer,
    scheduler,
    scaler,
    device,
    max_grad_norm,
    amp=False,
    retention=None,
    anchor_batch=None,
    observer=None,
):
    if (retention is None) != (anchor_batch is None):
        raise ValueError("Retention requires an objective and a replay batch")
    normalizers, world_size = update_normalizers(criterion, batches, device)
    optimizer.zero_grad(set_to_none=True)
    totals = {name: torch.zeros((), device=device) for name in normalizers}
    for index, host_batch in enumerate(batches):
        batch = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in host_batch.items()
        }
        synchronize = (index == len(batches) - 1 and retention is None) or not hasattr(
            model, "no_sync"
        )
        context = nullcontext() if synchronize else model.no_sync()
        with context:
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp
            ):
                result = model(**batch)
                losses = criterion(
                    **result, **batch, normalizers=normalizers, world_size=world_size
                )
                loss = sum(losses.values())
            if observer is not None:
                observer(result, batch)
            finite = torch.isfinite(loss).to(torch.int32)
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Nonfinite supervised objective")
            scaler.scale(loss).backward()
        for name, value in losses.items():
            totals[name] += value.detach()
    if retention is not None:
        anchor = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in anchor_batch.items()
        }
        anchor_normalizers, _ = update_normalizers(criterion, [anchor], device)
        with policy_mode(model, adapters=None):
            losses = retention.losses(
                model, anchor, criterion, anchor_normalizers, world_size, amp=amp
            )
            loss = sum(losses.values())
            finite = torch.isfinite(loss).to(torch.int32)
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Nonfinite retention objective")
            scaler.scale(loss).backward()
        totals.update({name: value.detach() for name, value in losses.items()})
    norm, stepped = optimizer_update(model, optimizer, scheduler, scaler, max_grad_norm)
    packed = torch.stack(list(totals.values()))
    if world_size > 1:
        dist.all_reduce(packed)
        packed /= world_size
    return dict(zip(totals, packed.tolist())), norm, stepped

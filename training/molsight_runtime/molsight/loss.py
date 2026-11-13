import math

import torch
import torch.nn.functional as F
from torch import nn

from .decoding import GreedyDecoder
from .policy import (
    completion_mask,
    frozen_forward,
    grouped_advantages,
    grpo_objective,
    policy_mode,
    sample_completions,
    token_logps,
    validate_adapter_policy,
)
from .tokenizer import EOS_ID, MASK, PAD_ID


class SmoothedKLDLoss(nn.Module):
    def __init__(
        self,
        label_smoothing,
        tgt_vocab_size,
        ignore_index=None,
        ignore_indices=(),
        reduction="mean",
    ):
        super().__init__()
        if not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be mean or sum")
        self.ignore_index = ignore_index
        self.ignore_indices = tuple(
            set(ignore_indices)
            | ({ignore_index} if ignore_index is not None else set())
        )
        self.reduction = reduction
        self.confidence = 1.0 - label_smoothing
        allowed = torch.ones(tgt_vocab_size, dtype=torch.bool)
        for index in self.ignore_indices:
            if 0 <= index < tgt_vocab_size:
                allowed[index] = False
        if allowed.sum() < 2 and label_smoothing:
            raise ValueError("label smoothing requires at least two target classes")
        self.register_buffer("allowed", allowed)
        self.label_smoothing = label_smoothing
        self.smoothing_denominator = max(int(allowed.sum()) - 1, 1)

    @property
    def uniform(self):
        return (
            self.allowed.float() * (self.label_smoothing / self.smoothing_denominator)
        ).unsqueeze(0)

    def forward(self, output, target):
        valid = torch.ones_like(target, dtype=torch.bool)
        for index in self.ignore_indices:
            valid &= target.ne(index)
        output, target = output[valid].float(), target[valid]
        if target.numel() == 0:
            return output.sum()
        target_prob = self.uniform.expand(target.size(0), -1).clone()
        target_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = F.kl_div(F.log_softmax(output, dim=-1), target_prob, reduction="sum")
        return loss / target.numel() if self.reduction == "mean" else loss


class SequenceLoss(nn.Module):
    def __init__(
        self,
        label_smoothing,
        vocab_size,
        ignore_index=-100,
        ignore_indices=(),
        reduction="mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ignore_indices = tuple(ignore_indices)
        self.criterion = SmoothedKLDLoss(
            label_smoothing, vocab_size, ignore_index, ignore_indices, reduction
        )

    def forward(self, output, target):
        output = output[:, :-1].reshape(-1, output.size(-1))
        target = target[:, 1:].reshape(-1)
        return self.criterion(output, target)


class EdgeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        weight = torch.ones(7)
        weight[0] = 0.1
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        preds = preds.reshape(-1, preds.size(-1)).float()
        targets = targets.reshape(-1)
        valid = targets.ge(0)
        loss = F.cross_entropy(
            preds[valid], targets[valid], weight=self.weight.float(), reduction="sum"
        )
        if self.reduction == "sum":
            return loss
        return loss / self.weight[targets[valid]].sum().clamp_min(1e-12)


def coordinate_mask(targets):
    valid = targets.ne(-100)
    if (
        not torch.isfinite(targets).all()
        or ((targets[valid] < 0) | (targets[valid] > 1)).any()
    ):
        raise ValueError("Coordinate targets must be in [0, 1] or equal -100")
    return valid


class CoordMLELoss(nn.Module):
    """Independent mixtures of truncated Laplace densities on [0, 1]."""

    def __init__(self, args, reduction="mean"):
        super().__init__()
        self.register_buffer(
            "bins", torch.linspace(0, 1, args.n_coord_bins), persistent=False
        )
        self.reduction = reduction

    def forward(self, x_hms, y_hms, sigmas, targets):
        targets = targets.float()
        valid = coordinate_mask(targets)
        losses = []
        for axis, probabilities in enumerate((x_hms, y_hms)):
            mask = valid[..., axis]
            probabilities = probabilities[mask].float()
            sigma = sigmas[..., 0][mask].float().unsqueeze(-1)
            target = targets[..., axis][mask].unsqueeze(-1)
            if not torch.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError("Coordinate scales must be finite and positive")
            if (
                not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not torch.allclose(
                    probabilities.sum(-1),
                    torch.ones_like(target[:, 0]),
                    atol=2e-3,
                    rtol=2e-3,
                )
            ):
                raise ValueError(
                    "Coordinate probabilities must be normalized and finite"
                )
            mass = -0.5 * (
                torch.expm1(-self.bins / sigma) + torch.expm1(-(1 - self.bins) / sigma)
            )
            log_density = (
                -(target - self.bins).abs() / sigma - (2 * sigma).log() - mass.log()
            )
            log_weights = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            losses.append(-torch.logsumexp(log_weights + log_density, dim=-1).sum())
        loss = sum(losses)
        return loss if self.reduction == "sum" else loss / valid.sum().clamp_min(1)


class CoordLoss(nn.Module):
    def __init__(self, args, aux_loss=False, reduction="mean"):
        super().__init__()
        self.loss_mle = CoordMLELoss(args, reduction="sum")
        self.aux_loss = aux_loss
        self.n_bins = args.n_coord_bins
        self.reduction = reduction
        if aux_loss:
            self.loss_kld = SmoothedKLDLoss(
                args.label_smoothing, args.n_coord_bins, reduction="sum"
            )
        self.regression = args.regression

    def forward(self, pred, targets):
        loc_pred, x_hms, y_hms, sigmas = pred
        valid = coordinate_mask(targets)
        if self.regression:
            loss = F.l1_loss(
                loc_pred.float()[valid], targets.float()[valid], reduction="sum"
            )
        else:
            loss = self.loss_mle(x_hms, y_hms, sigmas, targets)
            if self.aux_loss:
                for axis, probabilities in enumerate((x_hms, y_hms)):
                    mask = valid[..., axis]
                    classes = (
                        (targets[..., axis][mask] * (self.n_bins - 1)).round().long()
                    )
                    log_probs = (
                        probabilities[mask]
                        .float()
                        .clamp_min(torch.finfo(torch.float32).tiny)
                        .log()
                    )
                    loss = loss + self.loss_kld(log_probs, classes)
        return loss if self.reduction == "sum" else loss / valid.sum().clamp_min(1)


class Criterion(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        criterion = {}
        if "char" in args.formats:
            if MASK in tokenizer.stoi:
                ignore_indices = [PAD_ID, tokenizer.stoi[MASK]]
            else:
                ignore_indices = []
            criterion["sequence"] = SequenceLoss(
                args.label_smoothing,
                len(tokenizer),
                ignore_index=PAD_ID,
                ignore_indices=ignore_indices,
                reduction="sum",
            )
        if "edges" in args.formats:
            criterion["edges"] = EdgeLoss(reduction="sum")
        if "coords" in args.formats:
            criterion["coords"] = CoordLoss(args, reduction="sum")
        self.criterion = nn.ModuleDict(criterion)

    def normalizers(self, batch):
        values = {}
        if "sequence" in self.criterion:
            targets = batch["label"][:, 1:]
            valid = torch.ones_like(targets, dtype=torch.bool)
            for index in self.criterion["sequence"].criterion.ignore_indices:
                valid &= targets.ne(index)
            values["sequence"] = valid.sum().float()
        if "edges" in self.criterion:
            if batch.get("edges") is None:
                raise ValueError(
                    "Edge supervision is required by the configured objective"
                )
            targets = batch["edges"]
            values["edges"] = torch.where(targets.eq(0), 0.1, 1.0)[targets.ge(0)].sum()
        if "coords" in self.criterion:
            if batch.get("coords") is None:
                raise ValueError(
                    "Coordinate supervision is required by the configured objective"
                )
            values["coords"] = coordinate_mask(batch["coords"]).sum().float()
        return values

    def forward(
        self,
        logits,
        edge_pred,
        loc_pred,
        label,
        edges=None,
        coords=None,
        normalizers=None,
        world_size=1,
        **kwargs,
    ):
        losses = {}
        if normalizers is None:
            normalizers = self.normalizers(
                {"label": label, "edges": edges, "coords": coords}
            )
        with torch.autocast(device_type=logits.device.type, enabled=False):
            for format in self.criterion:
                if format == "sequence":
                    losses[format] = self.criterion[format](logits, label)
                elif format == "edges":
                    losses[format] = self.criterion[format](edge_pred, edges)
                elif format == "coords":
                    losses[format] = self.criterion[format](loc_pred, coords)
                else:
                    raise NotImplementedError
                losses[format] = (
                    losses[format] * world_size / normalizers[format].clamp_min(1e-12)
                )

        return losses


class SCSTLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.n_samples < 1:
            raise ValueError("SCST requires at least one sample per image")
        self.args = args
        self.baseline_decode_strategy = GreedyDecoder(temperature=0, eot=EOS_ID)
        self.stochastic_decode_strategy = GreedyDecoder(temperature=1, eot=EOS_ID)

    def forward(self, model, image, reward_fn):
        policy = getattr(model, "module", model)
        validate_adapter_policy(policy)
        with policy_mode(policy):
            with frozen_forward(image.device):
                features = policy.get_image_features(image)
                baseline, _ = sample_completions(
                    policy, image, features, self.baseline_decode_strategy, 1
                )
                predictions, intermediate = sample_completions(
                    policy,
                    image,
                    features,
                    self.stochastic_decode_strategy,
                    self.args.n_samples,
                )
                baseline_rewards = (
                    reward_fn(baseline["smiles"]).detach().to(image.device)
                )
                rewards = reward_fn(predictions["smiles"]).detach().to(image.device)
            ids = intermediate["raw_tokens"]
            if baseline_rewards.shape != (image.shape[0],) or rewards.shape != (
                ids.shape[0],
            ):
                raise ValueError(
                    "Reward counts must match baseline and sampled completions"
                )
            if (
                not torch.isfinite(rewards).all()
                or not torch.isfinite(baseline_rewards).all()
            ):
                raise ValueError("Rewards must be finite")
            output = model(
                image=None,
                label=ids,
                image_features=features.repeat_interleave(self.args.n_samples, dim=0),
            )
            logps = token_logps(output["logits"][:, :-1], ids[:, 1:])
            mask = completion_mask(ids[:, 1:], sampled=True)
            advantage = rewards - baseline_rewards.repeat_interleave(
                self.args.n_samples
            )
            loss = -(logps.masked_fill(~mask, 0).sum(dim=1) * advantage).mean()
        return loss, {
            "avg_reward": rewards.mean().item(),
            "avg_baseline": baseline_rewards.mean().item(),
        }


class GRPOLoss(nn.Module):
    def __init__(self, args, reference_model=None):
        super().__init__()
        if args.n_samples < 2:
            raise ValueError("GRPO requires at least two samples per image")
        self.beta = getattr(args, "kl_beta", 0.04)
        self.clip_epsilon = getattr(args, "clip_epsilon", 0.2)
        self.args = args
        self.reference_model = reference_model
        self.temperature = getattr(args, "policy_temperature", 1.0)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Policy temperature must be finite and positive")
        if reference_model is not None and any(
            parameter.requires_grad for parameter in reference_model.parameters()
        ):
            raise ValueError("Policy reference parameters must be frozen")
        self.stochastic_decode_strategy = GreedyDecoder(
            temperature=self.temperature, eot=EOS_ID
        )

    def forward(self, model, image, reward_fn):
        policy = getattr(model, "module", model)
        if self.reference_model is None:
            validate_adapter_policy(policy)
        elif self.reference_model is policy:
            raise ValueError("Policy and frozen reference must be independent models")
        with policy_mode(policy):
            with frozen_forward(image.device):
                features = policy.get_image_features(image)
                predictions, intermediate = sample_completions(
                    policy,
                    image,
                    features,
                    self.stochastic_decode_strategy,
                    self.args.n_samples,
                )
            ids = intermediate["raw_tokens"]
            mask = completion_mask(ids[:, 1:], sampled=True)
            features = features.repeat_interleave(self.args.n_samples, dim=0)
            output = (
                model(image=image, label=ids, feature_repeats=self.args.n_samples)
                if self.reference_model is not None
                else model(image=None, label=ids, image_features=features)
            )
            logps = token_logps(
                output["logits"][:, :-1], ids[:, 1:], temperature=self.temperature
            )
            old_logps = intermediate["token_logps"]
            if self.reference_model is None:
                with policy_mode(policy, adapters=False), frozen_forward(image.device):
                    reference = policy(image=None, label=ids, image_features=features)
            else:
                with (
                    policy_mode(self.reference_model, adapters=None),
                    frozen_forward(image.device),
                ):
                    reference_features = self.reference_model.get_image_features(image)
                    reference = self.reference_model(
                        image=None,
                        label=ids,
                        image_features=reference_features.repeat_interleave(
                            self.args.n_samples, dim=0
                        ),
                    )
            if reference["logits"].shape != output["logits"].shape:
                raise ValueError("Policy and reference logits have incompatible shapes")
            reference_logps = token_logps(
                reference["logits"][:, :-1], ids[:, 1:], temperature=self.temperature
            )
            rewards = reward_fn(predictions["smiles"]).detach().to(logps.device)
            if rewards.shape != (ids.shape[0],):
                raise ValueError("Each sampled completion requires one reward")
            advantages, reward_std = grouped_advantages(rewards, self.args.n_samples)
            loss, kl, clip_fraction = grpo_objective(
                logps,
                old_logps,
                reference_logps,
                advantages,
                mask,
                self.beta,
                self.clip_epsilon,
            )
        return loss, {
            "avg_rewards": rewards.mean().item(),
            "rewards_std": reward_std.item(),
            "nonconstant_reward_group_fraction": rewards.reshape(
                -1, self.args.n_samples
            )
            .std(dim=1, correction=0)
            .gt(0)
            .float()
            .mean()
            .item(),
            "zero_reward_fraction": rewards.eq(0).float().mean().item(),
            "kl": kl.item(),
            "clip_fraction": clip_fraction.item(),
            "tokens_per_image": mask.sum().item() / image.shape[0],
        }

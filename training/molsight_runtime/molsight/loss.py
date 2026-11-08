import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .tokenizer import EOS_ID, PAD_ID, MASK, MASK_ID
from .decoding import GreedyDecoder
from .policy import (
    completion_mask, frozen_forward, grouped_advantages, grpo_objective, policy_mode,
    sample_completions, token_logps, validate_adapter_policy,
)


class SmoothedKLDLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=None,
                 ignore_indices=(), reduction="mean"):
        super().__init__()
        if not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be mean or sum")
        self.ignore_index = ignore_index
        self.ignore_indices = tuple(set(ignore_indices) | ({ignore_index} if ignore_index is not None else set()))
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

    @property
    def uniform(self):
        denominator = max(int(self.allowed.sum()) - 1, 1)
        return (self.allowed.float() * (self.label_smoothing / denominator)).unsqueeze(0)

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
    def __init__(self, label_smoothing, vocab_size, ignore_index=-100,
                 ignore_indices=(), reduction="mean"):
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
    def __init__(self):
        super().__init__()
        weight = torch.ones(7)
        weight[0] = 0.1
        self.register_buffer("weight", weight)

    def forward(self, preds, targets):
        preds = preds.reshape(-1, preds.size(-1)).float()
        targets = targets.reshape(-1)
        valid = targets.ge(0)
        if not valid.any():
            return preds[valid].sum()
        return F.cross_entropy(preds[valid], targets[valid], weight=self.weight.float())


class CoordMLELoss(nn.Module):

    def __init__(self, args):
        super(CoordMLELoss, self).__init__()
        self.register_buffer('x_bins', torch.linspace(0, 1, args.n_coord_bins), persistent=False)
        self.register_buffer('y_bins', torch.linspace(0, 1, args.n_coord_bins), persistent=False)

    def forward(self, x_hms, y_hms, sigmas, targets):
        '''
        x_hms: [batch, num_atoms, n_bins]
        y_hms: [batch, num_atoms, n_bins]
        sigmas: [batch, num_atoms, 1]
        targets: [batch, num_atoms, 2]
        '''
        dist_x = torch.abs(targets.narrow(2, 0, 1) - self.x_bins)  # [batch, num_atoms, n_bins]
        dist_y = torch.abs(targets.narrow(2, 1, 1) - self.y_bins)

        hm_x = torch.exp(-dist_x / sigmas) / (2 * sigmas)   # Laplace distribution
        hm_y = torch.exp(-dist_y / sigmas) / (2 * sigmas)

        denom_x = hm_x.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        denom_y = hm_y.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        hm_x = hm_x / denom_x
        hm_y = hm_y / denom_y

        prob_x = (x_hms * hm_x).sum(dim=-1)  # [batch, num_atoms]
        prob_y = (y_hms * hm_y).sum(dim=-1)

        loss = -torch.log((prob_x * prob_y) + 1e-4) / 2
        loss[torch.isnan(loss)] = 0.0
        mask = targets[:, :, 0].ge(0) # [batch, num_atoms]
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)

        return loss
    

class CoordLoss(nn.Module):

    def __init__(self, args, aux_loss=False):
        super(CoordLoss, self).__init__()
        self.weight = {
                'mle': 1.0,
                'kld': 1.0
            }
        self.loss_mle = CoordMLELoss(args)
        self.aux_loss = aux_loss
        if aux_loss:
            self.loss_kld = SmoothedKLDLoss(args.label_smoothing, args.n_coord_bins)
        self.regression = args.regression

    def forward(self, pred, targets):
        '''
        pred_coords: Tuple of (loc_pred, x_hms, y_hms, sigmas)
        targets: [batch, num_atoms, 2]
        '''        
        loc_pred, x_hms, y_hms, sigmas = pred

        if self.regression:
            # loc_pred: [batch, num_atoms, 2]
            # targets: [batch, num_atoms, 2]
            loss = F.l1_loss(loc_pred, targets, reduction='none')   # [batch, num_atoms, 2]
            loss[torch.isnan(loss)] = 0.0
            mask = targets.ge(0)
            return (loss * mask).sum() / (mask.sum() + 1e-5)

        loss_mle = self.loss_mle(x_hms, y_hms, sigmas, targets)
        
        if self.aux_loss:
            x_hms = x_hms.view(-1, x_hms.size(-1))  # [batch * num_atoms, n_bins]
            y_hms = y_hms.view(-1, y_hms.size(-1))  # [batch * num_atoms, n_bins]
            targets = targets.view(-1, 2)  # [batch * num_atoms, 2]
            loss_kld = self.loss_kld(x_hms, targets[:, 0]) + self.loss_kld(y_hms, targets[:, 1])           
            return self.weight['mle'] * loss_mle + self.weight['kld'] * loss_kld

        return self.weight['mle'] * loss_mle


class Criterion(nn.Module):

    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        if 'char' in args.formats:
            if MASK in tokenizer.stoi:
                ignore_indices = [PAD_ID, MASK_ID]
            else:
                ignore_indices = []
            criterion['sequence'] = SequenceLoss(args.label_smoothing, len(tokenizer),
                                                ignore_index=PAD_ID, ignore_indices=ignore_indices)
        if 'edges' in args.formats:
            criterion['edges'] = EdgeLoss()
        if 'coords' in args.formats:
            criterion['coords'] = CoordLoss(args)
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, logits, edge_pred, loc_pred, label, edges, coords, **kwargs):
        losses = {}
        with torch.cuda.amp.autocast(enabled=False):
            for format in self.criterion.keys():
                if format == 'sequence':
                    losses[format] = self.criterion[format](logits, label)
                elif format == 'edges':
                    losses[format] = self.criterion[format](edge_pred, edges)
                elif format == 'coords':
                    losses[format] = self.criterion[format](loc_pred, coords)
                else:
                    raise NotImplementedError
        
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
                    policy, image, features, self.stochastic_decode_strategy, self.args.n_samples
                )
                baseline_rewards = reward_fn(baseline["smiles"]).detach().to(image.device)
                rewards = reward_fn(predictions["smiles"]).detach().to(image.device)
            ids = intermediate["raw_tokens"]
            if baseline_rewards.shape != (image.shape[0],) or rewards.shape != (ids.shape[0],):
                raise ValueError("Reward counts must match baseline and sampled completions")
            if not torch.isfinite(rewards).all() or not torch.isfinite(baseline_rewards).all():
                raise ValueError("Rewards must be finite")
            output = model(image=None, label=ids,
                           image_features=features.repeat_interleave(self.args.n_samples, dim=0))
            logps = token_logps(output["logits"][:, :-1], ids[:, 1:])
            mask = completion_mask(ids[:, 1:])
            advantage = rewards - baseline_rewards.repeat_interleave(self.args.n_samples)
            loss = -(logps.masked_fill(~mask, 0).sum(dim=1) * advantage).mean()
        return loss, {
            "avg_reward": rewards.mean().item(),
            "avg_baseline": baseline_rewards.mean().item(),
        }


class GRPOLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.n_samples < 2:
            raise ValueError("GRPO requires at least two samples per image")
        self.beta = getattr(args, "kl_beta", 0.04)
        self.clip_epsilon = getattr(args, "clip_epsilon", 0.2)
        self.args = args
        self.stochastic_decode_strategy = GreedyDecoder(temperature=1.0, eot=EOS_ID)

    def forward(self, model, image, reward_fn):
        policy = getattr(model, "module", model)
        validate_adapter_policy(policy)
        with policy_mode(policy):
            with frozen_forward(image.device):
                features = policy.get_image_features(image)
                predictions, intermediate = sample_completions(
                    policy, image, features, self.stochastic_decode_strategy, self.args.n_samples
                )
            ids = intermediate["raw_tokens"]
            mask = completion_mask(ids[:, 1:])
            features = features.repeat_interleave(self.args.n_samples, dim=0)
            output = model(image=None, label=ids, image_features=features)
            logps = token_logps(output["logits"][:, :-1], ids[:, 1:])
            old_logps = intermediate["token_logps"]
            with policy_mode(policy, adapters=False), frozen_forward(image.device):
                reference = policy(image=None, label=ids, image_features=features)
                reference_logps = token_logps(reference["logits"][:, :-1], ids[:, 1:])
            rewards = reward_fn(predictions["smiles"]).detach().to(logps.device)
            if rewards.shape != (ids.shape[0],):
                raise ValueError("Each sampled completion requires one reward")
            advantages, reward_std = grouped_advantages(rewards, self.args.n_samples)
            loss, kl, clip_fraction = grpo_objective(
                logps, old_logps, reference_logps, advantages, mask, self.beta, self.clip_epsilon
            )
        return loss, {
            "avg_rewards": rewards.mean().item(),
            "rewards_std": reward_std.item(),
            "kl": kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

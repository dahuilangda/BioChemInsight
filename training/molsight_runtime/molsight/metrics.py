"""Token diagnostics independent of the training objective."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


class TokenMetrics:
    def __init__(self, ignore_indices, device):
        self.ignore_indices = tuple(ignore_indices)
        self.totals = torch.zeros(3, dtype=torch.float64, device=device)

    @torch.no_grad()
    def __call__(self, outputs, batch):
        targets = batch["label"][:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        for index in self.ignore_indices:
            valid &= targets.ne(index)
        targets = targets[valid]
        logits = outputs["logits"][:, :-1][valid].float()
        if targets.numel() == 0:
            return
        logps = F.log_softmax(logits, dim=-1)
        nll = -logps.gather(-1, targets[:, None]).sum()
        correct = logits.argmax(dim=-1).eq(targets).sum()
        self.totals += torch.stack((valid.sum(), nll, correct)).double()

    def compute(self):
        totals = self.totals.clone()
        if dist.is_initialized():
            dist.all_reduce(totals)
        count, nll, correct = totals.tolist()
        return {
            "teacher_forced_tokens": int(count),
            "teacher_forced_nll": nll / count if count else None,
            "teacher_forced_token_accuracy": correct / count if count else None,
        }

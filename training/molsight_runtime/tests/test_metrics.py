import math

import pytest
import torch

from molsight.loss import SequenceLoss
from molsight.metrics import TokenMetrics


def test_near_zero_smoothed_kl_does_not_mean_zero_nll():
    targets = torch.tensor([[1, 4, 2, 0]])
    probabilities = torch.tensor([[[0, 0.05, 0.05, 0, 0.9], [0, 0.05, 0.9, 0, 0.05]]])
    logits = torch.cat(
        (probabilities.clamp_min(1e-12).log(), torch.full((1, 2, 5), float("nan"))),
        dim=1,
    ).requires_grad_()
    loss = SequenceLoss(0.1, 5, ignore_index=0, ignore_indices=(3,))(logits, targets)
    assert abs(loss.item()) < 1e-6
    metrics = TokenMetrics((0, 3), "cpu")
    metrics({"logits": logits}, {"label": targets})
    before = metrics.totals.clone()
    values = metrics.compute()
    assert values["teacher_forced_tokens"] == 2
    assert values["teacher_forced_nll"] == pytest.approx(-math.log(0.9))
    assert values["teacher_forced_token_accuracy"] == 1.0
    assert logits.grad is None
    torch.testing.assert_close(metrics.totals, before)

    other = (
        torch.tensor([[[0, 0.1, 0.7, 0, 0.2], [0, 0.1, 0.7, 0, 0.2]]])
        .clamp_min(1e-12)
        .log()
    )
    metrics({"logits": other}, {"label": torch.tensor([[1, 4]])})
    values = metrics.compute()
    assert values["teacher_forced_nll"] == pytest.approx(
        (-2 * math.log(0.9) - math.log(0.2)) / 3
    )
    assert values["teacher_forced_token_accuracy"] == pytest.approx(2 / 3)


def test_no_valid_tokens_has_no_fabricated_accuracy():
    metrics = TokenMetrics((0, 3), "cpu")
    metrics(
        {"logits": torch.full((1, 3, 5), float("nan"))},
        {"label": torch.tensor([[1, 0, 3]])},
    )
    assert metrics.compute() == {
        "teacher_forced_tokens": 0,
        "teacher_forced_nll": None,
        "teacher_forced_token_accuracy": None,
    }


def metrics_worker(rank, initialization, output):
    import json
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=initialization, rank=rank, world_size=2)
    try:
        count = 1 if rank == 0 else 3
        probabilities = (
            torch.tensor([0.1, 0.7, 0.2])
            if rank == 0
            else torch.tensor([0.05, 0.05, 0.9])
        )
        logits = probabilities.log().expand(1, count + 1, 3)
        labels = torch.tensor([[1] + [2] * count])
        metrics = TokenMetrics((), "cpu")
        metrics({"logits": logits}, {"label": labels})
        first, second = metrics.compute(), metrics.compute()
        assert first == second
        assert metrics.totals[0].item() == count
        if rank == 0:
            from pathlib import Path

            Path(output).write_text(json.dumps(first))
    finally:
        dist.destroy_process_group()


def test_distributed_metrics_weight_tokens_and_preserve_local_state(tmp_path):
    import json
    import torch.multiprocessing as mp

    output = tmp_path / "metrics.json"
    mp.spawn(
        metrics_worker,
        args=("file://" + str(tmp_path / "store"), str(output)),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text())
    assert result["teacher_forced_tokens"] == 4
    assert result["teacher_forced_token_accuracy"] == 0.75
    assert result["teacher_forced_nll"] == pytest.approx(
        (-math.log(0.2) - 3 * math.log(0.9)) / 4
    )

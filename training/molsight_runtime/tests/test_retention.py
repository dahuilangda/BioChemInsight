import copy
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from molsight.loss import Criterion
from molsight.optimization import supervised_update
from molsight.retention import (
    CompleteAnchorSampler,
    RetentionObjective,
    distillation_sum,
)


def test_distillation_keeps_new_token_probability_and_masks_padding():
    student = torch.tensor([[[2.0, -1.0, 3.0], [float("nan")] * 3]], requires_grad=True)
    teacher = torch.tensor([[[2.0, -1.0], [float("nan")] * 2]], requires_grad=True)
    targets = torch.tensor([[1, 0]])
    actual = distillation_sum(student, teacher, targets, temperature=2)
    q = F.softmax(teacher[0, 0].detach() / 2, dim=-1)
    expected = (q * (q.log() - F.log_softmax(student[0, 0] / 2, dim=-1)[:2])).sum() * 4
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher.grad is None and student.grad[0, 0, 2] > 0
    assert torch.equal(student.grad[0, 1], torch.zeros(3))


class Vocabulary:
    stoi = {"<mask>": 3}

    def __len__(self):
        return 7


class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 4).requires_grad_(False)
        self.embedding = nn.Embedding(7, 4)
        self.projection = nn.Linear(4, 7)

    def get_image_features(self, image):
        return self.image_encoder(image)[:, :, None, None]

    def forward(self, label, image=None, image_features=None, **kwargs):
        features = (
            self.get_image_features(image) if image_features is None else image_features
        )
        hidden = self.embedding(label) + features.flatten(1)[:, None]
        return {"logits": self.projection(hidden), "edge_pred": None, "loc_pred": None}


def fixture(full_encoder=False):
    torch.manual_seed(916)
    model = CaptionModel()
    model.image_encoder.eval()
    reference = copy.deepcopy(model)
    if full_encoder:
        model.image_encoder.requires_grad_(True)
    else:
        reference.image_encoder = model.image_encoder
    with torch.no_grad():
        model.projection.weight[4].add_(0.3)
    criterion = Criterion(
        SimpleNamespace(formats=["char"], label_smoothing=0.1), Vocabulary()
    )
    primary = {
        "image": torch.randn(4, 2),
        "label": torch.tensor(
            [[1, 4, 2, 0, 0], [1, 4, 5, 6, 2], [1, 2, 0, 0, 0], [1, 5, 4, 2, 0]]
        ),
    }
    replay = {
        "image": torch.randn(2, 2),
        "label": torch.tensor([[1, 4, 2, 0], [1, 5, 4, 2]]),
    }
    return model, RetentionObjective(reference), criterion, primary, replay


def select(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def reference_step(model, retention, criterion, primary, replay):
    loss = sum(criterion(**model(**primary), **primary).values())
    loss += sum(
        retention.losses(
            model, replay, criterion, criterion.normalizers(replay), 1, amp=False
        ).values()
    )
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.03).step()


def distributed_worker(rank, initialization, destination, full_encoder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=initialization, rank=rank, world_size=2)
    try:
        model, retention, criterion, primary, replay = fixture(full_encoder)
        initial_reference = {
            key: value.clone()
            for key, value in retention.reference.state_dict().items()
        }
        model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        supervised_update(
            model,
            criterion,
            [select(primary, [rank * 2 + offset]) for offset in range(2)],
            optimizer,
            scheduler,
            torch.amp.GradScaler("cpu", enabled=False),
            "cpu",
            float("inf"),
            retention=retention,
            anchor_batch=select(replay, [rank]),
        )
        assert all(
            parameter.grad is None for parameter in retention.reference.parameters()
        )
        assert model.training and model.module.embedding.training
        assert not model.module.image_encoder.training
        for key, value in retention.reference.state_dict().items():
            torch.testing.assert_close(value, initial_reference[key], rtol=0, atol=0)
        if rank == 0:
            torch.save(model.module.state_dict(), destination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("full_encoder", [False, True])
def test_distributed_retention_matches_the_global_objective(tmp_path, full_encoder):
    path = tmp_path / "model.pth"
    mp.spawn(
        distributed_worker,
        args=("file://" + str(tmp_path / "store"), str(path), full_encoder),
        nprocs=2,
        join=True,
    )
    model, retention, criterion, primary, replay = fixture(full_encoder)
    reference_step(model, retention, criterion, primary, replay)
    actual = torch.load(path, weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(actual[key], value)


def test_anchor_sampling_resumes_and_keeps_ranks_disjoint():
    class Records:
        max_len, split = 16, "train"
        specification = {"files": {"train": {"sha256": "frozen"}}}
        table = pa.Table.from_pylist(
            [
                {"category": "complete", "input_size": 512, "token_ids": [1, 4, 2]}
                for _ in range(8)
            ]
            + [
                {"category": "markush", "input_size": 512, "token_ids": [1, 4, 2]},
                {"category": "complete", "input_size": 768, "token_ids": [1, 4, 2]},
                {"category": "complete", "input_size": 512, "token_ids": [1, 8, 2]},
            ]
        )

        def _open(self):
            pass

    common = dict(vocabulary_size=7, updates=10, batch_size=2, world_size=2)
    first, second = [
        CompleteAnchorSampler(Records(), rank=rank, **common) for rank in (0, 1)
    ]
    assert np.array_equal(first.indices, np.arange(8))
    left, right = list(first), list(second)
    assert all(not set(a) & set(b) for a, b in zip(left, right, strict=True))
    assert list(CompleteAnchorSampler(Records(), start_update=4, **common)) == left[4:]


def test_full_encoder_replay_uses_independent_reference_features():
    model, _, criterion, _, replay = fixture()
    reference = copy.deepcopy(model)
    reference.image_encoder = copy.deepcopy(model.image_encoder)
    reference.requires_grad_(False)
    model.image_encoder.requires_grad_(True)
    with torch.no_grad():
        model.image_encoder.weight.add_(0.4)
    retention = RetentionObjective(reference)
    expected_state = {
        key: value.clone() for key, value in reference.state_dict().items()
    }
    loss = sum(
        retention.losses(
            model, replay, criterion, criterion.normalizers(replay), 1, amp=False
        ).values()
    )
    loss.backward()
    assert model.image_encoder.weight.grad is not None
    assert model.image_encoder.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reference.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)

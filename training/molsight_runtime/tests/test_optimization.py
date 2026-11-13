import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from molsight.loss import Criterion
from molsight.optimization import optimizer_update, supervised_update
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class Vocabulary:
    def __init__(self):
        self.stoi = {"<mask>": 3}

    def __len__(self):
        return 7


def test_character_objective_requires_no_fabricated_graph_targets():
    criterion = Criterion(
        SimpleNamespace(formats=["char"], label_smoothing=0.1), Vocabulary()
    )
    logits = torch.randn(1, 3, 7, requires_grad=True)
    labels = torch.tensor([[1, 4, 2]])
    loss = criterion(logits=logits, edge_pred=None, loc_pred=None, label=labels)[
        "sequence"
    ]
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad.abs().sum() > 0
    criterion = Criterion(
        SimpleNamespace(formats=["char", "edges"], label_smoothing=0.1), Vocabulary()
    )
    with pytest.raises(ValueError, match="Edge supervision"):
        criterion.normalizers({"label": labels})


class PredictionHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(5, 6, 7))
        self.edges = nn.Parameter(torch.randn(5, 2, 2, 7))
        self.coords = nn.Parameter(torch.rand(5, 2, 2))

    def forward(self, idx, **kwargs):
        return {
            "logits": self.logits[idx],
            "edge_pred": self.edges[idx],
            "loc_pred": (self.coords[idx], None, None, None),
        }


def fixture():
    torch.manual_seed(414)
    batch = {
        "idx": torch.arange(5),
        "label": torch.tensor(
            [
                [1, 4, 2, 0, 0, 0],
                [1, 5, 6, 4, 2, 0],
                [1, 4, 5, 6, 4, 2],
                [1, 3, 2, 0, 0, 0],
                [1, 4, 2, 0, 0, 0],
            ]
        ),
        "edges": torch.tensor(
            [
                [[0, 1], [1, 0]],
                [[-100, -100], [-100, -100]],
                [[0, 2], [2, 0]],
                [[0, 0], [0, 0]],
                [[-100, -100], [-100, -100]],
            ]
        ),
        "coords": torch.tensor(
            [
                [[0.1, 0.9], [0.2, -100]],
                [[-100, -100], [-100, -100]],
                [[0.3, 0.4], [0.5, 0.6]],
                [[0.8, 0.9], [-100, -100]],
                [[-100, -100], [-100, -100]],
            ]
        ),
    }
    args = SimpleNamespace(
        formats=["char", "edges", "coords"],
        n_coord_bins=3,
        label_smoothing=0.1,
        regression=True,
    )
    return PredictionHeads(), Criterion(args, Vocabulary()), batch


def select(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def reference_step(model, criterion, batch):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    losses = criterion(**model(**batch), **batch)
    sum(losses.values()).backward()
    optimizer.step()
    return {key: value.item() for key, value in losses.items()}


@pytest.mark.parametrize(
    "groups", [[[0, 1, 2, 3, 4]], [[0, 1], [2], [3, 4]], [[0], [1], [2], [3], [4]]]
)
def test_accumulation_matches_complete_batch(groups):
    model, criterion, batch = fixture()
    reference = copy.deepcopy(model)
    expected = reference_step(reference, criterion, batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    actual, _, stepped = supervised_update(
        model,
        criterion,
        [select(batch, indices) for indices in groups],
        optimizer,
        scheduler,
        scaler,
        "cpu",
        float("inf"),
    )
    assert stepped
    assert actual == pytest.approx(expected, rel=1e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.027)
    for first, second in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(first, second)


def distributed_worker(rank, initialization, destination):
    dist.init_process_group("gloo", init_method=initialization, rank=rank, world_size=2)
    try:
        model, criterion, batch = fixture()
        model = DistributedDataParallel(model)
        groups = [[0], [1]] if rank == 0 else [[2], [3, 4]]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        supervised_update(
            model,
            criterion,
            [select(batch, indices) for indices in groups],
            optimizer,
            scheduler,
            torch.amp.GradScaler("cpu", enabled=False),
            "cpu",
            float("inf"),
        )
        if rank == 0:
            torch.save(model.module.state_dict(), destination)
    finally:
        dist.destroy_process_group()


def test_distributed_unequal_annotations_match_complete_batch(tmp_path):
    initialization = "file://" + str(tmp_path / "store")
    destination = tmp_path / "model.pth"
    mp.spawn(
        distributed_worker, args=(initialization, str(destination)), nprocs=2, join=True
    )
    model, criterion, batch = fixture()
    reference_step(model, criterion, batch)
    actual = torch.load(destination, weights_only=True)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(actual[name], expected)


def test_nonfinite_gradients_do_not_advance_scheduler():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    (model(torch.ones(1, 1)).sum() * float("nan")).backward()
    with pytest.raises(FloatingPointError):
        optimizer_update(
            model, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False), 1
        )
    assert scheduler.last_epoch == 0


class MeanPolicyObjective(nn.Module):
    def forward(self, model, images, reward):
        prediction = model(images).squeeze(-1)
        return prediction.square().mean(), {
            "avg_rewards": images[:, 0].mean().item(),
            "tokens_per_image": images[:, 1].mean().item(),
            "clip_fraction": (images[:, 0] * images[:, 1]).sum().item()
            / images[:, 1].sum().item(),
        }


def policy_fixture():
    torch.manual_seed(716)
    model = nn.Linear(2, 1)
    images = torch.tensor([[0.1, 2], [0.2, 5], [0.3, 7], [0.4, 3], [0.5, 1]])
    return model, images


def policy_batches(images, groups):
    return [
        {"image": images[indices], "smiles": ["C"] * len(indices)} for indices in groups
    ]


@pytest.mark.parametrize("groups", [[[0, 1, 2, 3, 4]], [[0, 1], [2], [3, 4]]])
def test_policy_accumulation_weights_images_and_token_metrics(groups):
    from molsight.optimization import policy_update

    model, images = policy_fixture()
    reference = copy.deepcopy(model)
    expected_loss = reference(images).square().mean()
    expected_loss.backward()
    torch.optim.SGD(reference.parameters(), lr=0.03).step()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    metrics, _, stepped = policy_update(
        model,
        MeanPolicyObjective(),
        policy_batches(images, groups),
        lambda p, t: None,
        optimizer,
        scheduler,
        torch.amp.GradScaler("cpu", enabled=False),
        "cpu",
        float("inf"),
    )
    assert stepped and scheduler.last_epoch == 1
    assert metrics["loss"] == pytest.approx(expected_loss.item())
    assert metrics["avg_rewards"] == pytest.approx(images[:, 0].mean().item())
    assert metrics["clip_fraction"] == pytest.approx(
        (images[:, 0] * images[:, 1]).sum().item() / images[:, 1].sum().item()
    )
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


class UnwrappedPolicyObjective(MeanPolicyObjective):
    def __call__(self, model, images, reward):
        return super().__call__(getattr(model, "module", model), images, reward)


class ConditionalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Parameter(torch.tensor(0.7))
        self.right = nn.Parameter(torch.tensor(-0.3))
        self.unused = nn.Parameter(torch.tensor(1.1))

    def forward(self, images):
        return torch.stack(
            [
                (self.left if image[0] < 0.35 else self.right) * image[0]
                for image in images
            ]
        )[:, None]


def distributed_policy_worker(
    rank, initialization, destination, unwrap=False, conditional=False
):
    from molsight.optimization import policy_update

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=initialization, rank=rank, world_size=2)
    try:
        model, images = policy_fixture()
        if conditional:
            model = ConditionalPolicy()
        model = DistributedDataParallel(model)
        groups = [[0, 1], [2]] if rank == 0 else [[3], [4]]
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.03, weight_decay=0.1 if conditional else 0.0
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        metrics, _, _ = policy_update(
            model,
            UnwrappedPolicyObjective() if unwrap else MeanPolicyObjective(),
            policy_batches(images, groups),
            lambda p, t: None,
            optimizer,
            scheduler,
            torch.amp.GradScaler("cpu", enabled=False),
            "cpu",
            float("inf"),
        )
        if rank == 0:
            torch.save(
                {"model": model.module.state_dict(), "metrics": metrics}, destination
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("unwrap", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_distributed_policy_accumulation_matches_the_global_batch(
    tmp_path, unwrap, conditional
):
    destination = tmp_path / "policy.pth"
    mp.spawn(
        distributed_policy_worker,
        args=(
            "file://" + str(tmp_path / "policy_store"),
            str(destination),
            unwrap,
            conditional,
        ),
        nprocs=2,
        join=True,
    )
    model, images = policy_fixture()
    if conditional:
        model = ConditionalPolicy()
    model(images).square().mean().backward()
    torch.optim.SGD(
        model.parameters(), lr=0.03, weight_decay=0.1 if conditional else 0.0
    ).step()
    actual = torch.load(destination, weights_only=True)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(actual["model"][name], expected)
    assert actual["metrics"]["avg_rewards"] == pytest.approx(images[:, 0].mean().item())

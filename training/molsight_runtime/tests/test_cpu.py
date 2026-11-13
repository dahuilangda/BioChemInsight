import pytest
import torch
from molsight import cpu
from torch.utils.data import DataLoader, TensorDataset


def test_local_ranks_share_cpu_budget(monkeypatch):
    monkeypatch.setattr(cpu, "available_cpus", lambda: 12)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert cpu.worker_count() == 5
    assert cpu.worker_count(local_ranks=1) == 11
    assert cpu.worker_count(local_ranks=16) == 0
    assert cpu.worker_count(3) == 3
    with pytest.raises(ValueError):
        cpu.worker_count(-1)


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_supports_serial_and_parallel_workers(workers):
    dataset = TensorDataset(torch.arange(12))
    loader = DataLoader(dataset, batch_size=3, **cpu.loader_options(workers))
    for _ in range(2):
        assert torch.equal(torch.cat([batch[0] for batch in loader]), torch.arange(12))


def test_worker_limits_native_threads():
    import cv2

    cpu.initialize_worker(0)
    assert torch.get_num_threads() == 1
    assert cv2.getNumThreads() == 1

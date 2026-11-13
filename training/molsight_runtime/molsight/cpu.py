"""CPU allocation for local training ranks and data workers."""

import math
import os
from pathlib import Path

_thread_limits = None


def available_cpus():
    count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return count


def worker_count(requested=None, *, local_ranks=None):
    if requested is not None:
        if requested < 0:
            raise ValueError("Worker count must be nonnegative")
        return requested
    ranks = (
        int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if local_ranks is None
        else local_ranks
    )
    if ranks < 1:
        raise ValueError("Local rank count must be positive")
    return max(0, available_cpus() // ranks - 1)


def initialize_worker(worker_id):
    global _thread_limits
    import cv2
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _thread_limits = threadpool_limits(limits=1)


def loader_options(requested=None):
    count = worker_count(requested)
    options = {"num_workers": count, "worker_init_fn": initialize_worker}
    if count:
        options.update(
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    return options

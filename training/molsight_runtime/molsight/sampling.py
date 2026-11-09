"""Deterministic domain sampling with resolution-consistent microbatches."""

import hashlib
import json
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class DomainBatchSampler(Sampler):
    def __init__(
        self,
        metadata,
        *,
        updates,
        accumulation,
        batch_size,
        weights,
        rank=0,
        world_size=1,
        seed=42,
        start_update=0,
    ):
        if min(updates, accumulation, batch_size, world_size) < 1:
            raise ValueError("Training dimensions must be positive")
        if not 0 <= rank < world_size or not 0 <= start_update <= updates:
            raise ValueError("Invalid rank or update position")
        if not weights or any(
            not np.isfinite(value) or value <= 0 for value in weights.values()
        ):
            raise ValueError("Domain weights must be finite and positive")
        self.updates, self.accumulation, self.batch_size = (
            updates,
            accumulation,
            batch_size,
        )
        self.rank, self.world_size, self.seed, self.start_update = (
            rank,
            world_size,
            seed,
            start_update,
        )
        self.domains = sorted(weights)
        probabilities = np.asarray(
            [weights[domain] for domain in self.domains], dtype=float
        )
        self.probabilities = probabilities / probabilities.sum()
        present = {row["category"] for row in metadata}
        if present - set(weights):
            raise ValueError(
                f"Corpus categories missing from domain weights: "
                f"{sorted(present - set(weights))}"
            )
        self.groups = defaultdict(lambda: defaultdict(dict))
        grouped = defaultdict(list)
        for index, row in enumerate(metadata):
            if row["category"] in weights:
                stratum = row.get("peptide_stratum") or ""
                grouped[(row["category"], stratum, row["input_size"])].append(index)
        for (domain, stratum, resolution), indices in sorted(grouped.items()):
            self.groups[domain][stratum][resolution] = np.asarray(
                indices, dtype=np.int64
            )
        if set(self.groups) != set(self.domains):
            raise ValueError("Requested training domain has no admitted records")
        self.definition_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "metadata": metadata,
                    "weights": weights,
                    "accumulation": accumulation,
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "seed": seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def __len__(self):
        return (self.updates - self.start_update) * self.accumulation

    def __iter__(self):
        for update in range(self.start_update, self.updates):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, update]))
            domain = self.domains[rng.choice(len(self.domains), p=self.probabilities)]
            strata = sorted(self.groups[domain])
            stratum = strata[rng.integers(len(strata))]
            groups = self.groups[domain][stratum]
            resolutions = sorted(groups)
            counts = np.asarray(
                [len(groups[value]) for value in resolutions], dtype=float
            )
            for _ in range(self.accumulation):
                resolution = resolutions[
                    rng.choice(len(resolutions), p=counts / counts.sum())
                ]
                indices = groups[resolution]
                count = self.batch_size * self.world_size
                selected = rng.choice(indices, size=count, replace=len(indices) < count)
                yield selected[
                    self.rank * self.batch_size : (self.rank + 1) * self.batch_size
                ].tolist()

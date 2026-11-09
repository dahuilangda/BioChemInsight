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
        excluded_domains=(),
        peptide_group_weights=None,
        stratum_weights=None,
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
        excluded = set(excluded_domains)
        if excluded & set(weights):
            raise ValueError("Excluded domains also have sampling weights")
        present = {row["category"] for row in metadata}
        if present - set(weights) - excluded:
            raise ValueError(
                f"Corpus categories missing from domain weights: "
                f"{sorted(present - set(weights) - excluded)}"
            )
        self.groups = defaultdict(lambda: defaultdict(dict))
        grouped = defaultdict(list)
        for index, row in enumerate(metadata):
            if row["category"] in excluded:
                continue
            domains = {row["category"]}
            if row.get("peptide_candidate"):
                domains.add("peptide")
            for domain in sorted(domains & weights.keys()):
                stratum = ""
                if domain == "peptide":
                    stratum = row.get("peptide_stratum") or (
                        "discovered_cyclic"
                        if row.get("cyclic_peptide_candidate")
                        else "discovered_linear"
                    )
                grouped[(domain, stratum, row["input_size"])].append(index)
        for (domain, stratum, resolution), indices in sorted(grouped.items()):
            self.groups[domain][stratum][resolution] = np.asarray(
                indices, dtype=np.int64
            )
        if set(self.groups) != set(self.domains):
            raise ValueError("Requested training domain has no admitted records")
        self.peptide_groups = defaultdict(list)
        for stratum in self.groups.get("peptide", {}):
            group = (
                stratum
                if stratum in ("discovered_cyclic", "discovered_linear")
                else "synthetic"
            )
            self.peptide_groups[group].append(stratum)
        self.peptide_group_weights = peptide_group_weights
        if peptide_group_weights is not None:
            if (
                set(peptide_group_weights) != set(self.peptide_groups)
                or not peptide_group_weights
            ):
                raise ValueError(
                    "Peptide group weights must cover exactly the available groups"
                )
            if any(
                not np.isfinite(value) or value <= 0
                for value in peptide_group_weights.values()
            ):
                raise ValueError("Peptide group weights must be finite and positive")
        self.stratum_weights = stratum_weights
        if stratum_weights is not None:
            available = {
                stratum
                for domain_strata in self.groups.values()
                for stratum in domain_strata
                if stratum
            }
            unknown = set(stratum_weights) - available
            if unknown:
                raise ValueError(
                    f"Stratum weights name strata absent from the corpus: {sorted(unknown)}"
                )
            if any(
                not np.isfinite(value) or value <= 0
                for value in stratum_weights.values()
            ):
                raise ValueError("Stratum weights must be finite and positive")
        self.definition_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "metadata": metadata,
                    "weights": weights,
                    "accumulation": accumulation,
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "seed": seed,
                    "excluded_domains": sorted(excluded),
                    **(
                        {"peptide_group_weights": peptide_group_weights}
                        if peptide_group_weights is not None
                        else {}
                    ),
                    **(
                        {"stratum_weights": stratum_weights}
                        if stratum_weights is not None
                        else {}
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def __len__(self):
        return (self.updates - self.start_update) * self.accumulation

    def domain_for_update(self, update):
        if not 0 <= update < self.updates:
            raise ValueError("Update lies outside the sampling plan")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, update]))
        return self.domains[rng.choice(len(self.domains), p=self.probabilities)]

    def __iter__(self):
        for update in range(self.start_update, self.updates):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, update]))
            domain = self.domains[rng.choice(len(self.domains), p=self.probabilities)]
            strata = sorted(self.groups[domain])
            if domain == "peptide" and self.peptide_group_weights is not None:
                names = sorted(self.peptide_groups)
                probabilities = np.asarray(
                    [self.peptide_group_weights[name] for name in names], dtype=float
                )
                group = names[
                    rng.choice(len(names), p=probabilities / probabilities.sum())
                ]
                strata = sorted(self.peptide_groups[group])
            if self.stratum_weights is not None:
                # Weighted stratum choice: unlisted strata keep weight 1, so
                # concentration never drops the rest entirely.
                probabilities = np.asarray(
                    [
                        self.stratum_weights.get(stratum, 1.0)
                        for stratum in strata
                    ],
                    dtype=float,
                )
                stratum = strata[
                    rng.choice(len(strata), p=probabilities / probabilities.sum())
                ]
            else:
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

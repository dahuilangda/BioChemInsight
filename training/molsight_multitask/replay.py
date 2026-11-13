"""Independent native-resolution replay pools for retained recognition domains."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler


class DomainReplaySampler(Sampler):
    def __init__(self, dataset, category, *, vocabulary_size, updates, batch_size,
                 rank=0, world_size=1, seed=42, start_update=0):
        if min(vocabulary_size, updates, batch_size, world_size) < 1:
            raise ValueError('Replay dimensions must be positive')
        if not 0 <= rank < world_size or not 0 <= start_update <= updates:
            raise ValueError('Invalid replay rank or update position')
        dataset._open()
        grouped = defaultdict(list)
        columns = dataset.table.select(['category', 'peptide_stratum', 'input_size', 'token_ids'])
        offset = 0
        for batch in columns.to_batches(max_chunksize=1024):
            for index, row in enumerate(batch.to_pylist(), start=offset):
                tokens = row['token_ids']
                if (row['category'] == category and row['input_size'] == 512
                        and 2 <= len(tokens) <= dataset.max_len
                        and all(0 <= token < vocabulary_size for token in tokens)):
                    # Keep native peptide subclasses represented in replay,
                    # independently of the focused main-task stratum weights.
                    group = row['peptide_stratum'] or '' if category == 'peptide' else ''
                    grouped[group].append(index)
            offset += len(batch)
        if not grouped:
            raise ValueError(f'No reference-compatible replay for {category}')
        self.groups = {key: np.asarray(value, dtype=np.int64) for key, value in sorted(grouped.items())}
        self.category, self.updates, self.batch_size = category, updates, batch_size
        self.rank, self.world_size, self.seed, self.start_update = rank, world_size, seed, start_update
        self.namespace = int.from_bytes(hashlib.sha256(category.encode()).digest()[:4], 'little')
        self.definition_sha256 = hashlib.sha256(json.dumps({
            'category': category, 'pools': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in self.groups.items()},
            'file_sha256': dataset.specification['files'][dataset.split]['sha256'],
            'batch_size': batch_size, 'world_size': world_size, 'seed': seed,
            'vocabulary_size': vocabulary_size, 'strata_uniform': category == 'peptide'}, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return self.updates - self.start_update

    def __iter__(self):
        names = sorted(self.groups)
        for update in range(self.start_update, self.updates):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, update, self.namespace]))
            pool = self.groups[names[rng.integers(len(names))]]
            count = self.batch_size * self.world_size
            selected = rng.choice(pool, size=count, replace=len(pool) < count)
            yield selected[self.rank * self.batch_size:(self.rank + 1) * self.batch_size].tolist()

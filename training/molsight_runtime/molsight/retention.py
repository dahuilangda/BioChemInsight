"""Complete-molecule replay against an immutable reference model."""

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from .policy import policy_mode
from .tokenizer import PAD_ID


def distillation_sum(student, teacher, targets, *, temperature=2.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Distillation temperature must be finite and positive")
    if student.shape[:-1] != targets.shape or teacher.shape[:-1] != targets.shape:
        raise ValueError("Distillation predictions do not align with targets")
    if teacher.shape[-1] > student.shape[-1]:
        raise ValueError("Reference vocabulary exceeds the student vocabulary")
    valid = targets != PAD_ID
    prediction = F.log_softmax(student[valid].float() / temperature, dim=-1)
    reference = F.softmax(teacher[valid].detach().float() / temperature, dim=-1)
    return (
        F.kl_div(prediction[:, : teacher.shape[-1]], reference, reduction="sum")
        * temperature**2
    )


class CompleteAnchorSampler(Sampler):
    def __init__(
        self,
        dataset,
        *,
        vocabulary_size,
        updates,
        batch_size,
        rank=0,
        world_size=1,
        seed=42,
        start_update=0,
    ):
        if min(vocabulary_size, updates, batch_size, world_size) < 1:
            raise ValueError("Anchor dimensions must be positive")
        if not 0 <= rank < world_size or not 0 <= start_update <= updates:
            raise ValueError("Invalid anchor rank or update position")
        dataset._open()
        indices, offset = [], 0
        columns = dataset.table.select(["category", "input_size", "token_ids"])
        for batch in columns.to_batches(max_chunksize=1024):
            for index, row in enumerate(batch.to_pylist(), start=offset):
                tokens = row["token_ids"]
                if (
                    row["category"] == "complete"
                    and row["input_size"] == 512
                    and 2 <= len(tokens) <= dataset.max_len
                    and all(0 <= token < vocabulary_size for token in tokens)
                ):
                    indices.append(index)
            offset += len(batch)
        if len(indices) < batch_size * world_size:
            raise ValueError(
                "Insufficient complete-molecule reference-compatible replay"
            )
        self.indices = np.asarray(indices, dtype=np.int64)
        self.updates, self.batch_size = updates, batch_size
        self.rank, self.world_size, self.seed, self.start_update = (
            rank,
            world_size,
            seed,
            start_update,
        )
        self.definition_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "pool_sha256": hashlib.sha256(self.indices.tobytes()).hexdigest(),
                    "file_sha256": dataset.specification["files"][dataset.split][
                        "sha256"
                    ],
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "seed": seed,
                    "vocabulary_size": vocabulary_size,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def __len__(self):
        return self.updates - self.start_update

    def __iter__(self):
        for update in range(self.start_update, self.updates):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, update, 0xA11CE])
            )
            selected = rng.choice(
                self.indices, size=self.batch_size * self.world_size, replace=False
            )
            yield selected[
                self.rank * self.batch_size : (self.rank + 1) * self.batch_size
            ].tolist()


class RetentionObjective:
    def __init__(
        self, reference, *, weight=1.0, supervised_weight=0.25, temperature=2.0
    ):
        if any(
            not math.isfinite(value) or value < 0
            for value in (weight, supervised_weight)
        ):
            raise ValueError("Retention weights must be finite and nonnegative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Retention temperature must be finite and positive")
        self.reference = reference.requires_grad_(False).eval()
        self.weight, self.supervised_weight, self.temperature = (
            weight,
            supervised_weight,
            temperature,
        )

    def losses(self, model, batch, criterion, normalizers, world_size, *, amp):
        raw = getattr(model, "module", model)
        with (
            policy_mode(model, adapters=None),
            torch.autocast(
                device_type=batch["image"].device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ),
        ):
            with torch.no_grad():
                features = self.reference.get_image_features(batch["image"])
                expected = self.reference(
                    image=None, label=batch["label"], image_features=features
                )
            if raw.image_encoder is self.reference.image_encoder:
                if any(
                    parameter.requires_grad
                    for parameter in raw.image_encoder.parameters()
                ):
                    raise ValueError(
                        "A trainable encoder cannot share the frozen reference"
                    )
                actual = model(
                    image=None, label=batch["label"], image_features=features
                )
            else:
                actual = model(image=batch["image"], label=batch["label"])
            target = batch["label"][:, 1:]
            divergence = distillation_sum(
                actual["logits"][:, :-1],
                expected["logits"][:, :-1],
                target,
                temperature=self.temperature,
            )
            replay = criterion(
                **actual, **batch, normalizers=normalizers, world_size=world_size
            )
            return {
                "retention_kl": self.weight
                * divergence
                * world_size
                / normalizers["sequence"].clamp_min(1),
                "retention_sequence": self.supervised_weight * sum(replay.values()),
            }

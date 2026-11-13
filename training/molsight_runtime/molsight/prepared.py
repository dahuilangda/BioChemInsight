"""Verified image loading and memory-mapped character supervision."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import numpy as np
import pyarrow as pa
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pyarrow import ipc
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .dataset import get_transforms
from .tokenizer import EOS_ID, PAD_ID, SOS_ID


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class ImageReader:
    def __init__(self):
        self.legacy = {}
        self.prepared = A.Compose(
            [
                A.ToGray(p=1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __call__(self, row, input_size):
        encoded = Path(row["file_path"]).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed after validation: {row['file_path']}")
        with Image.open(io.BytesIO(encoded)) as source:
            image = source.convert("RGB")
            if row.get("model_input_preprocessed"):
                if image.width != image.height or image.width > input_size:
                    raise ValueError(
                        "Verified model input cannot be cropped or downsampled"
                    )
                if image.width != input_size:
                    image = image.resize(
                        (input_size, input_size), Image.Resampling.BICUBIC
                    )
                return self.prepared(image=np.asarray(image))["image"]
            if input_size not in self.legacy:
                args = SimpleNamespace(input_size=input_size, encoder="efficientvit")
                self.legacy[input_size] = get_transforms(
                    args, augment=False, rotate=False
                )
            return self.legacy[input_size](image=np.asarray(image), keypoints=[])[
                "image"
            ]


class PreparedDataset(Dataset):
    def __init__(self, manifest, split, *, max_len=2048, require_release=True):
        self.manifest_sha256 = file_hash(manifest)
        self.manifest_path = Path(manifest)
        specification = json.loads(self.manifest_path.read_text())
        if (
            require_release
            and specification.get("training_release_approved") is not True
        ):
            raise ValueError("Recognition data has not passed release checks")
        if require_release:
            admission = specification.get("admission_report", {})
            if not admission.get("path") or file_hash(
                admission["path"]
            ) != admission.get("sha256"):
                raise ValueError("Recognition admission report is missing or changed")
            report = json.loads(Path(admission["path"]).read_text())
            if (
                report.get("passed") is not True
                or report.get("records") != specification["records"]
            ):
                raise ValueError(
                    "Recognition admission report does not cover this export"
                )
            if (
                report.get("files") != specification["files"]
                or report.get("vocabulary_sha256") != specification["vocabulary_sha256"]
            ):
                raise ValueError("Recognition files differ from their admission report")
        if specification["formats"] != ["char"]:
            raise ValueError(
                "Prepared recognition data requires character-only training"
            )
        if (
            not specification["input_files_unchanged"]
            or not specification["sources_unchanged"]
        ):
            raise ValueError("Recognition data provenance changed during preparation")
        self.path = Path(specification["files"][split]["path"])
        if file_hash(self.path) != specification["files"][split]["sha256"]:
            raise ValueError("Recognition targets changed after export")
        stat = self.path.stat()
        self.file_stat = (stat.st_size, stat.st_mtime_ns)
        self.specification = specification
        self.split = split
        self.max_len = max_len
        self.table = None
        self.reader = None
        self.mapping = None

    def _open(self):
        if self.table is None:
            stat = self.path.stat()
            if (stat.st_size, stat.st_mtime_ns) != self.file_stat:
                raise ValueError("Recognition targets changed after export")
            self.mapping = pa.memory_map(str(self.path), "r")
            self.table = ipc.open_file(self.mapping).read_all()
            if len(self.table) != self.specification["records"][self.split]:
                raise ValueError("Recognition split has an unexpected record count")
            self.reader = ImageReader()

    def __getstate__(self):
        return {**self.__dict__, "table": None, "reader": None, "mapping": None}

    def __len__(self):
        return self.specification["records"][self.split]

    def __getitem__(self, index):
        self._open()
        row = self.table.slice(index, 1).to_pylist()[0]
        tokens = row["token_ids"]
        if (
            not 2 <= len(tokens) <= self.max_len
            or tokens[0] != SOS_ID
            or tokens[-1] != EOS_ID
        ):
            raise ValueError("Invalid recognition context or termination tokens")
        return {
            "idx": index,
            "image": self.reader(row, row["input_size"]),
            "label": torch.tensor(tokens, dtype=torch.long),
            "smiles": row["SMILES"],
        }

    def metadata(self, peptide_tags=None):
        self._open()
        rows = self.table.select(
            ["category", "peptide_stratum", "input_size"]
        ).to_pylist()
        if peptide_tags is not None:
            document = json.loads(Path(peptide_tags).read_text())
            tagged = document["splits"][self.split]
            if (
                document["schema_version"] != 1
                or document["manifest_sha256"] != self.manifest_sha256
                or tagged["file_sha256"]
                != self.specification["files"][self.split]["sha256"]
                or tagged["records"] != len(rows)
            ):
                raise ValueError("Peptide sampling tags belong to a different release")
            seen = set()
            for item in tagged["peptides"]:
                index = item["index"]
                if (
                    type(index) is not int
                    or not 0 <= index < len(rows)
                    or index in seen
                    or item["peptide_candidate"] is not True
                    or type(item["cyclic_peptide_candidate"]) is not bool
                ):
                    raise ValueError("Invalid or repeated peptide sampling tag")
                seen.add(index)
                rows[index].update(
                    {
                        "peptide_candidate": True,
                        "cyclic_peptide_candidate": item["cyclic_peptide_candidate"],
                    }
                )
        return rows


def collate_characters(rows):
    if not rows or any(set(row) != {"idx", "image", "label", "smiles"} for row in rows):
        raise ValueError("Inconsistent character-supervision batch")
    if len({tuple(row["image"].shape) for row in rows}) != 1:
        raise ValueError("Recognition batches require a common input resolution")
    return {
        "idx": torch.tensor([row["idx"] for row in rows], dtype=torch.long),
        "image": torch.stack([row["image"] for row in rows]),
        "label": pad_sequence(
            [row["label"] for row in rows], batch_first=True, padding_value=PAD_ID
        ),
        "smiles": [row["smiles"] for row in rows],
    }

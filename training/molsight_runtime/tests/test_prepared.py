import json
from pathlib import Path

import pyarrow as pa
import pytest
import torch
from PIL import Image
from pyarrow import ipc

from molsight.prepared import PreparedDataset, collate_characters
from molsight.tokenizer import CharTokenizer
from training.molsight_data.export import SCHEMA, export_batch, initialize
from training.molsight_data.index import file_hash


@pytest.fixture
def prepared(tmp_path):
    vocabulary = tmp_path / "vocab.json"
    words = ["<pad>", "<sos>", "<eos>", "<mask>", "<sep>", *map(chr, range(32, 127))]
    vocabulary.write_text(json.dumps({word: index for index, word in enumerate(words)}))
    initialize(vocabulary)
    tokenizer = CharTokenizer(vocabulary)
    path = tmp_path / "input.png"
    Image.new("RGB", (512, 512), "white").save(path)
    label = "NC(=O)C"
    tokens, _ = tokenizer.smiles_to_sequence(label)
    payload = {
        "category": "complete",
        "source": "fixture",
        "source_split": "train",
        "file_path": str(path),
        "image_sha256": file_hash(path),
        "pixel_sha256": "pixels",
        "model_input_preprocessed": True,
        "input_size": 512,
        "rejection_reasons": [],
    }
    rows = export_batch(
        [
            (
                1,
                "key",
                "group",
                "train",
                "complete",
                label,
                len(tokens),
                json.dumps(payload),
            )
        ]
    )
    arrow = tmp_path / "train.arrow"
    with pa.OSFile(str(arrow), "wb") as stream, ipc.new_file(stream, SCHEMA) as writer:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "formats": ["char"],
                "training_release_approved": False,
                "input_files_unchanged": True,
                "sources_unchanged": True,
                "records": {"train": 1},
                "files": {"train": {"path": str(arrow), "sha256": file_hash(arrow)}},
            }
        )
    )
    return manifest, tokens


def test_exported_targets_are_loaded_without_invented_graph_annotations(prepared):
    manifest, tokens = prepared
    with pytest.raises(ValueError, match="release checks"):
        PreparedDataset(manifest, "train")
    dataset = PreparedDataset(manifest, "train", require_release=False)
    batch = collate_characters([dataset[0]])
    assert batch["label"].tolist() == [tokens]
    assert batch["image"].shape == (1, 3, 512, 512)
    assert set(batch) == {"idx", "image", "label", "smiles"}
    assert dataset.metadata() == [
        {"category": "complete", "peptide_stratum": None, "input_size": 512}
    ]
    assert dataset.__getstate__()["table"] is None


def test_sampling_tags_preserve_targets_and_require_the_same_release(
    prepared, tmp_path
):
    manifest, tokens = prepared
    dataset = PreparedDataset(manifest, "train", require_release=False)
    tags = {
        "schema_version": 1,
        "manifest_sha256": file_hash(manifest),
        "splits": {
            "train": {
                "file_sha256": dataset.specification["files"]["train"]["sha256"],
                "records": 1,
                "peptides": [
                    {
                        "index": 0,
                        "peptide_candidate": True,
                        "cyclic_peptide_candidate": False,
                    }
                ],
            }
        },
    }
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps(tags))
    row = dataset.metadata(path)[0]
    assert row["category"] == "complete" and row["peptide_candidate"]
    assert dataset[0]["label"].tolist() == tokens
    tags["splits"]["train"]["peptides"] *= 2
    path.write_text(json.dumps(tags))
    with pytest.raises(ValueError, match="repeated"):
        dataset.metadata(path)
    tags["manifest_sha256"] = "other release"
    path.write_text(json.dumps(tags))
    with pytest.raises(ValueError, match="different release"):
        dataset.metadata(path)


def test_incompatible_contexts_and_image_resolutions_fail(prepared):
    manifest, _ = prepared
    dataset = PreparedDataset(manifest, "train", require_release=False, max_len=3)
    with pytest.raises(ValueError, match="context"):
        dataset[0]
    first = {
        "idx": 1,
        "image": torch.zeros(3, 512, 512),
        "label": torch.tensor([1, 5, 2]),
        "smiles": "C",
    }
    second = {**first, "image": torch.zeros(3, 768, 768)}
    with pytest.raises(ValueError, match="common input resolution"):
        collate_characters([first, second])


def test_changed_target_files_fail_before_worker_start(prepared):
    manifest, _ = prepared
    path = Path(json.loads(manifest.read_text())["files"]["train"]["path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="targets changed"):
        PreparedDataset(manifest, "train", require_release=False)

"""Bind verified backbone tags to an admitted release without changing targets."""

import argparse
import inspect
import json
from pathlib import Path

import pysqlite3 as sqlite3
from molsight.prepared import PreparedDataset
from molsight.structure import parse_structure

from .index import file_hash
from .peptide_tags import backbone_tags


def prepare(manifest, labels, peptides, output):
    source_paths = [
        __file__,
        inspect.getsourcefile(inspect.unwrap(backbone_tags)),
        inspect.getsourcefile(inspect.unwrap(parse_structure)),
        inspect.getsourcefile(PreparedDataset),
    ]
    sources = {str(Path(path).resolve()): file_hash(path) for path in source_paths}
    inputs = {
        str(Path(path).resolve()): file_hash(path)
        for path in (manifest, labels, peptides)
    }
    with sqlite3.connect(
        Path(labels).resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS peptides",
            (Path(peptides).resolve().as_uri() + "?mode=ro",),
        )
        lookup = {}
        for text, features in connection.execute(
            "SELECT l.resolved_target,p.features FROM labels l "
            "JOIN peptides.candidates p ON p.target=l.target WHERE l.error IS NULL"
        ):
            expected = json.loads(features)
            if text not in lookup:
                actual = backbone_tags(parse_structure(text).molecule)
                if actual != expected:
                    raise ValueError(
                        "Peptide discovery features differ from the target structure"
                    )
                lookup[text] = actual
            elif lookup[text] != expected:
                raise ValueError(
                    "Equivalent target texts carry conflicting backbone features"
                )
    splits = {}
    for split in ("train", "validation", "test"):
        dataset = PreparedDataset(manifest, split, max_len=65536)
        dataset._open()
        selected, offset = [], 0
        for batch in dataset.table.select(["SMILES"]).to_batches(max_chunksize=1024):
            for index, text in enumerate(batch.column(0).to_pylist(), start=offset):
                if text in lookup:
                    selected.append({"index": index, **lookup[text]})
            offset += len(batch)
        splits[split] = {
            "file_sha256": dataset.specification["files"][split]["sha256"],
            "records": len(dataset),
            "peptides": selected,
        }
    if not all(file_hash(path) == value for path, value in inputs.items()):
        raise ValueError("Sampling inputs changed during preparation")
    if not all(file_hash(path) == value for path, value in sources.items()):
        raise ValueError("Sampling sources changed during preparation")
    result = {
        "schema_version": 1,
        "manifest_sha256": file_hash(manifest),
        "inputs": inputs,
        "code_sha256": sources,
        "verified_target_tags": len(lookup),
        "splits": splits,
        "scope": "Sampling and diagnostic tags; source labels and chemical identities are unchanged.",
    }
    output = Path(output)
    with output.open("x") as stream:
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    return {split: len(value["peptides"]) for split, value in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "peptides", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.labels, args.peptides, args.out)))


if __name__ == "__main__":
    main()

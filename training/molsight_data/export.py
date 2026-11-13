"""Export grouped candidates as immutable, tokenized recognition records."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pysqlite3 as sqlite3
from molsight.cpu import available_cpus
from molsight.tokenizer import CharTokenizer
from pyarrow import ipc

from .index import file_hash

_tokenizer = None

SCHEMA = pa.schema(
    [
        ("record_key", pa.string()),
        ("component", pa.string()),
        ("split", pa.string()),
        ("category", pa.string()),
        ("source", pa.string()),
        ("source_split", pa.string()),
        ("file_path", pa.string()),
        ("image_sha256", pa.string()),
        ("pixel_sha256", pa.string()),
        ("model_input_preprocessed", pa.bool_()),
        ("input_size", pa.int16()),
        ("SMILES", pa.string()),
        ("token_ids", pa.list_(pa.uint16())),
        ("peptide_stratum", pa.string()),
        ("source_record_id", pa.int64()),
    ]
)


def initialize(vocabulary):
    global _tokenizer
    _tokenizer = CharTokenizer(vocabulary)


def export_batch(rows):
    results = []
    for record_id, key, group, split, category, label, token_length, payload in rows:
        row = json.loads(payload)
        tokens, _ = _tokenizer.smiles_to_sequence(label)
        if (
            len(tokens) != token_length
            or _tokenizer.sequence_to_smiles(tokens[1:])["smiles"] != label
        ):
            raise ValueError("Target differs from the validated label interpretation")
        prepared = bool(row.get("model_input_preprocessed", False))
        size = row.get("input_size", 512) if prepared else 512
        if size not in (512, 768, 1024):
            raise ValueError("Unknown validated input resolution")
        if row.get("rejection_reasons"):
            raise ValueError("Rejected candidate reached recognition export")
        results.append(
            {
                "record_key": key,
                "component": group,
                "split": split,
                "category": "peptide" if row["category"] == "peptide" else category,
                "source": row["source"],
                "source_split": row["source_split"],
                "file_path": row["file_path"],
                "image_sha256": row["image_sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "model_input_preprocessed": prepared,
                "input_size": size,
                "SMILES": label,
                "token_ids": tokens,
                "peptide_stratum": row.get("peptide_stratum"),
                "source_record_id": record_id,
            }
        )
    return results


def export_records(index, labels, assignments, vocabulary, output, *, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        str(Path(path).resolve()): file_hash(path)
        for path in (index, labels, assignments, vocabulary)
    }
    code_paths = [
        Path(__file__),
        Path(__import__("training.molsight_data.index", fromlist=[""]).__file__),
    ]
    code_paths += [
        Path(__import__(name, fromlist=[""]).__file__)
        for name in ("molsight.tokenizer", "molsight.cpu")
    ]
    code = {str(path.resolve()): file_hash(path) for path in code_paths}
    counts, categories, resolutions = Counter(), Counter(), Counter()
    source = sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)
    source.execute(
        "ATTACH DATABASE ? AS resolved", (Path(labels).resolve().as_uri() + "?mode=ro",)
    )
    source.execute(
        "ATTACH DATABASE ? AS assigned",
        (Path(assignments).resolve().as_uri() + "?mode=ro",),
    )
    query = source.execute("""SELECT r.id,r.record_key,a.component,a.split,l.category,
        l.resolved_target,l.token_length,r.payload FROM records r
        JOIN resolved.labels l ON r.target=l.target JOIN assigned.assignments a ON r.id=a.record_id
        WHERE a.selected=1 AND r.valid=1 AND l.error IS NULL ORDER BY r.id""")
    files = {
        split: output / (split + ".arrow") for split in ("train", "validation", "test")
    }
    streams = {split: pa.OSFile(str(path), "wb") for split, path in files.items()}
    writers = {split: ipc.new_file(stream, SCHEMA) for split, stream in streams.items()}

    def collect(future):
        rows = future.result()
        for split in files:
            selected = [row for row in rows if row["split"] == split]
            if selected:
                writers[split].write_table(
                    pa.Table.from_pylist(selected, schema=SCHEMA)
                )
                counts[split] += len(selected)
        for row in rows:
            categories[f"{row['split']}:{row['category']}"] += 1
            resolutions[f"{row['split']}:{row['input_size']}"] += 1

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize,
            initargs=(str(vocabulary),),
        ) as executor:
            pending = []
            while rows := query.fetchmany(1024):
                pending.append(executor.submit(export_batch, rows))
                if len(pending) >= 2 * workers:
                    collect(pending.pop(0))
            for future in pending:
                collect(future)
    finally:
        source.close()
        for writer in writers.values():
            writer.close()
        for stream in streams.values():
            stream.close()
    report = {
        "schema_version": 1,
        "inputs": inputs,
        "code_sha256": code,
        "records": dict(counts),
        "categories": dict(categories),
        "resolutions": dict(resolutions),
        "vocabulary": str(Path(vocabulary).resolve()),
        "vocabulary_sha256": file_hash(vocabulary),
        "files": {
            split: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for split, path in files.items()
        },
        "workers": workers,
        "formats": ["char"],
        "input_files_unchanged": all(
            file_hash(path) == digest for path, digest in inputs.items()
        ),
        "sources_unchanged": all(
            file_hash(path) == digest for path, digest in code.items()
        ),
        "training_release_approved": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("index", "labels", "assignments", "vocab", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    print(
        json.dumps(
            export_records(
                args.index,
                args.labels,
                args.assignments,
                args.vocab,
                args.out,
                workers=args.workers,
            )
        )
    )


if __name__ == "__main__":
    main()

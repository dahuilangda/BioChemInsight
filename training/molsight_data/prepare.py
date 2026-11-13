from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from molsight.cpu import available_cpus, initialize_worker
from molsight.source_labels import parse_source_label
from molsight.structure import StructureError, parse_structure
from molsight.tokenizer import CharTokenizer
from PIL import Image
from pyarrow import ipc
from rdkit import Chem

_tokenizer = None
_settings = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def initialize(vocabulary, settings):
    global _tokenizer, _settings
    initialize_worker(0)
    _tokenizer = CharTokenizer(vocabulary)
    _settings = settings


def source_batches(specification, batch_size=128):
    path = Path(specification["path"])
    if (
        specification.get("source_mtime_ns", path.stat().st_mtime_ns)
        != path.stat().st_mtime_ns
    ):
        raise ValueError(f"Source changed after inventory: {path}")
    columns = specification["columns"]
    if path.suffix == ".parquet":
        batches = pq.ParquetFile(path).iter_batches(
            columns=columns, batch_size=batch_size
        )
    elif path.suffix == ".arrow":
        reader = ipc.open_stream(pa.memory_map(str(path), "r"))
        batches = (
            batch.select(columns).slice(start, batch_size)
            for batch in reader
            for start in range(0, batch.num_rows, batch_size)
        )
    else:
        raise ValueError(f"Unsupported source format: {path}")
    offset = 0
    for batch in batches:
        rows = batch.to_pylist()
        yield specification, offset, rows
        offset += len(rows)


def inspect_batch(task):
    specification, offset, rows = task
    results = []
    for row_number, row in enumerate(rows, offset):
        record = {
            "source": specification["name"],
            "source_path": specification["path"],
            "source_row": row_number,
            "source_split": specification["split"],
            "split": specification["split"],
            "image_domain": specification["image_domain"],
            "training_release_approved": False,
            "source_notation": specification.get("notation", "structure"),
        }
        reasons = []
        try:
            original = row[specification["label_column"]]
            record["original_label"] = original
            document = parse_source_label(
                original, notation=specification.get("notation", "structure")
            )
            label = document.model_text
            if not parse_structure(label).equivalent(document):
                raise StructureError("serialization_changed_structure")
            tokens, indices = _tokenizer.smiles_to_sequence(label)
            if _tokenizer.sequence_to_smiles(tokens[1:])["smiles"] != label:
                raise StructureError("tokenizer_changed_label")
            identity = json.dumps(
                [
                    document.colors,
                    document.canonical,
                    document.repeat,
                    document.variation_records,
                ],
                separators=(",", ":"),
            )
            record.update(
                SMILES=label,
                original_label=original,
                structure_hash=digest(identity.encode()),
                token_length=len(tokens),
                atom_count=len(indices),
            )
            roles = {kind for kind, _ in document.semantic_labels}
            molecule = document.molecule
            if roles & {"a", "r", "c", "rc", "sgroup", "repeat", "variation"}:
                category = "markush"
            elif "d" in roles or any(
                atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()
            ):
                category = "attachment"
            else:
                category = specification.get("category", "complete")
            record["category"] = category
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            Chem.RemoveStereochemistry(molecule)
            record["structure_family"] = digest(Chem.MolToSmiles(molecule).encode())
            if len(tokens) > _settings["max_len"]:
                reasons.append("target_exceeds_max_len")
            image_value = row[specification["image_column"]]
            if isinstance(image_value, str):
                image_path = Path(image_value)
                encoded = image_path.read_bytes()
            else:
                encoded = image_value["bytes"]
                if not encoded:
                    raise ValueError("Source image bytes are missing")
                image_path = None
            image_hash = digest(encoded)
            with Image.open(io.BytesIO(encoded)) as source_image:
                source_image.load()
                image = source_image.convert("RGB")
            width, height = image.size
            pixel_hash = digest(f"{width},{height}:".encode() + image.tobytes())
            gray = image.convert("L")
            bounds = gray.point(lambda value: 255 if value < 220 else 0).getbbox()
            if bounds is None:
                reasons.append("blank_image")
            if min(width, height) < 16:
                reasons.append("image_too_small")
            if image_path is None and not reasons:
                image_path = (
                    Path(_settings["image_root"])
                    / image_hash[:2]
                    / (image_hash + ".img")
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists():
                    if digest(image_path.read_bytes()) != image_hash:
                        raise ValueError("Stored image content does not match its hash")
                else:
                    with tempfile.NamedTemporaryFile(
                        dir=image_path.parent, delete=False
                    ) as stream:
                        stream.write(encoded)
                        temporary = Path(stream.name)
                    temporary.replace(image_path)
            record.update(
                file_path=str(image_path) if image_path else None,
                image_sha256=image_hash,
                pixel_sha256=pixel_hash,
                width=width,
                height=height,
                ink_bbox=bounds,
            )
        except (StructureError, ValueError, OSError, RuntimeError, KeyError) as error:
            reasons.append(f"{type(error).__name__}:{error}")
        record["rejection_reasons"] = reasons
        results.append(record)
    return results


def bounded_results(executor, tasks, window):
    tasks = iter(tasks)
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < window and not exhausted:
            try:
                pending.add(executor.submit(inspect_batch, next(tasks)))
            except StopIteration:
                exhausted = True
        if pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()


def prepare(manifest, output, vocabulary, *, workers, max_len):
    output.mkdir(parents=True, exist_ok=False)
    settings = {"max_len": max_len, "image_root": str((output / "images").resolve())}
    started = time.monotonic()
    counts = Counter()
    categories = Counter()
    tasks = (
        batch
        for specification in manifest["sources"]
        for batch in source_batches(specification)
    )
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize,
            initargs=(str(vocabulary), settings),
        ) as executor,
        (output / "records.jsonl").open("x") as stream,
    ):
        for records in bounded_results(executor, tasks, workers * 2):
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts["rows"] += 1
                counts["rejected" if record["rejection_reasons"] else "validated"] += 1
                counts.update(
                    "rejection:" + reason for reason in record["rejection_reasons"]
                )
                if not record["rejection_reasons"]:
                    categories[record["category"]] += 1
            if counts["rows"] % 4096 < len(records):
                stream.flush()
                print(
                    json.dumps(
                        {
                            "rows": counts["rows"],
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    report = {
        "schema_version": 1,
        "sources": manifest,
        "vocabulary_sha256": digest(vocabulary.read_bytes()),
        "max_len": max_len,
        "workers": workers,
        "seconds": time.monotonic() - started,
        "counts": dict(counts),
        "categories": dict(categories),
        "code_sha256": {
            str(path): digest(path.read_bytes())
            for path in (
                Path(__file__),
                Path(__import__("molsight.structure", fromlist=["__file__"]).__file__),
                Path(__import__("molsight.tokenizer", fromlist=["__file__"]).__file__),
            )
        },
        "training_release_approved": False,
        "remaining_checks": [
            "source_annotation_review",
            "cross_source_duplicates",
            "split_leakage",
            "final_input_legibility",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    if min(args.workers, args.max_len) < 1:
        parser.error("Worker count and sequence length must be positive")
    prepare(
        json.loads(args.manifest.read_text()),
        args.out,
        args.vocab,
        workers=args.workers,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()

"""Recheck exported supervision and cached image content before admission."""

import argparse
import hashlib
import io
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
from molsight.cpu import available_cpus
from molsight.tokenizer import CharTokenizer
from PIL import Image
from pyarrow import ipc

from .index import file_hash

_tokenizer = None


def initialize(vocabulary):
    global _tokenizer
    _tokenizer = CharTokenizer(vocabulary)


def audit_batch(rows):
    failures = []
    for row in rows:
        reasons = []
        try:
            tokens, _ = _tokenizer.smiles_to_sequence(row["SMILES"])
            if (
                tokens != row["token_ids"]
                or _tokenizer.sequence_to_smiles(tokens[1:])["smiles"] != row["SMILES"]
            ):
                reasons.append("cached_token_mismatch")
            encoded = Path(row["file_path"]).read_bytes()
            if hashlib.sha256(encoded).hexdigest() != row["image_sha256"]:
                reasons.append("image_content_changed")
            with Image.open(io.BytesIO(encoded)) as source:
                image = source.convert("RGB")
                width, height = image.size
                pixel_hash = hashlib.sha256(
                    f"{width},{height}:".encode() + image.tobytes()
                ).hexdigest()
                if pixel_hash != row["pixel_sha256"]:
                    reasons.append("decoded_image_changed")
                if min(width, height) < 16 or image.convert("L").getextrema()[0] >= 220:
                    reasons.append("empty_or_small_image")
                if row["model_input_preprocessed"] and (width, height) != (
                    row["input_size"],
                    row["input_size"],
                ):
                    reasons.append("verified_input_dimensions_changed")
        except (ValueError, OSError, RuntimeError) as error:
            reasons.append(f"{type(error).__name__}:{error}")
        if reasons:
            failures.append({"record_key": row["record_key"], "reasons": reasons})
    return len(rows), failures


def audit(manifest, output, *, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specification = json.loads(Path(manifest).read_text())
    inputs = {str(Path(manifest).resolve()): file_hash(manifest)}
    inputs.update(
        {item["path"]: item["sha256"] for item in specification["files"].values()}
    )
    inputs[specification["vocabulary"]] = specification["vocabulary_sha256"]
    if not all(file_hash(path) == digest for path, digest in inputs.items()):
        raise ValueError("Export files differ from their recorded hashes")
    code_paths = [
        Path(__file__),
        Path(__import__("training.molsight_data.index", fromlist=[""]).__file__),
    ]
    code_paths += [
        Path(__import__(name, fromlist=[""]).__file__)
        for name in ("molsight.tokenizer", "molsight.cpu")
    ]
    code = {str(path.resolve()): file_hash(path) for path in code_paths}
    counts, reasons, failed = Counter(), Counter(), 0
    started = last_progress = time.monotonic()
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize,
            initargs=(specification["vocabulary"],),
        ) as executor,
        (output / "failures.jsonl").open("x") as stream,
    ):
        pending = []

        def collect(future, split):
            nonlocal failed, last_progress
            count, failures = future.result()
            counts[split] += count
            failed += len(failures)
            for failure in failures:
                stream.write(json.dumps(failure) + "\n")
                reasons.update(failure["reasons"])
            if time.monotonic() - last_progress >= 10:
                print(
                    json.dumps(
                        {
                            "images_checked": sum(counts.values()),
                            "failed": failed,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_progress = time.monotonic()

        for split, item in specification["files"].items():
            with pa.memory_map(item["path"], "r") as mapping:
                reader = ipc.open_file(mapping)
                for index in range(reader.num_record_batches):
                    batch = reader.get_batch(index)
                    for start in range(0, len(batch), 128):
                        pending.append(
                            (
                                executor.submit(
                                    audit_batch, batch.slice(start, 128).to_pylist()
                                ),
                                split,
                            )
                        )
                        if len(pending) >= 2 * workers:
                            collect(*pending.pop(0))
        for future, split in pending:
            collect(future, split)
    report = {
        "records": dict(counts),
        "failed_records": failed,
        "reasons": dict(reasons),
        "inputs": inputs,
        "code_sha256": code,
        "workers": workers,
        "seconds": time.monotonic() - started,
        "input_files_unchanged": all(
            file_hash(path) == digest for path, digest in inputs.items()
        ),
        "sources_unchanged": all(
            file_hash(path) == digest for path, digest in code.items()
        ),
    }
    report["passed"] = (
        failed == 0
        and dict(counts) == specification["records"]
        and report["input_files_unchanged"]
        and report["sources_unchanged"]
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    report = audit(args.manifest, args.out, workers=args.workers)
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

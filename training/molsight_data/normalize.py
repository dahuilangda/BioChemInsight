"""Revalidate prepared records against a frozen structure-label contract."""

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from molsight.cpu import available_cpus
from molsight.source_labels import parse_source_label
from molsight.structure import StructureError, parse_structure
from pyarrow import ipc
from rdkit import Chem

from . import prepare

_specifications = None


def initialize(vocabulary, settings, specifications):
    global _specifications
    prepare.initialize(vocabulary, settings)
    _specifications = specifications


@lru_cache(maxsize=1)
def source_table(path, columns, row_group):
    if Path(path).suffix == ".arrow":
        return ipc.open_stream(pa.memory_map(path, "r")).read_all().select(columns)
    return pq.ParquetFile(path).read_row_group(row_group, columns=list(columns))


def source_row(specification, index, columns=None):
    path = specification["path"]
    offset, row_group = 0, 0
    if Path(path).suffix == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        while offset + metadata.row_group(row_group).num_rows <= index:
            offset += metadata.row_group(row_group).num_rows
            row_group += 1
    table = source_table(path, tuple(columns or specification["columns"]), row_group)
    return table.slice(index - offset, 1).to_pylist()[0]


def normalize_batch(records):
    results = []
    for original in records:
        if not original.get("file_path"):
            specification = _specifications[original["source_path"]]
            row = source_row(specification, original["source_row"])
            results.extend(
                prepare.inspect_batch((specification, original["source_row"], [row]))
            )
            continue
        record = dict(original)
        reasons = [
            reason
            for reason in original["rejection_reasons"]
            if reason in {"blank_image", "image_too_small"}
        ]
        try:
            specification = _specifications[original["source_path"]]
            notation = specification.get("notation", "structure")
            label_source = original["original_label"]
            if (
                notation == "markushgrapher"
                and original.get("source_notation") != notation
            ):
                column = specification["label_column"]
                label_source = source_row(
                    specification, original["source_row"], columns=(column,)
                )[column]
            record["original_label"] = label_source
            record["source_notation"] = notation
            document = parse_source_label(label_source, notation=notation)
            label = document.model_text
            if not document.equivalent(parse_structure(label)):
                raise StructureError("serialization_changed_structure")
            tokens, indices = prepare._tokenizer.smiles_to_sequence(label)
            if prepare._tokenizer.sequence_to_smiles(tokens[1:])["smiles"] != label:
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
                structure_hash=prepare.digest(identity.encode()),
                token_length=len(tokens),
                atom_count=len(indices),
            )
            if len(tokens) > prepare._settings["max_len"]:
                reasons.append("target_exceeds_max_len")
            molecule = document.molecule
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            Chem.RemoveStereochemistry(molecule)
            record["structure_family"] = prepare.digest(
                Chem.MolToSmiles(molecule).encode()
            )
        except (StructureError, ValueError, RuntimeError) as error:
            reasons.append(f"{type(error).__name__}:{error}")
        record["rejection_reasons"] = reasons
        results.append(record)
    return results


def batches(path, size=128):
    records = []
    with path.open() as stream:
        for line in stream:
            records.append(json.loads(line))
            if len(records) == size:
                yield records
                records = []
    if records:
        yield records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    specifications = {
        row["path"]: row for row in json.loads(args.manifest.read_text())["sources"]
    }
    settings = {
        "max_len": args.max_len,
        "image_root": str((args.out / "images").resolve()),
    }
    counts = Counter()
    categories = Counter()
    started = time.monotonic()
    source_directory = Path(__import__("molsight").__file__).parent
    sources = [*source_directory.glob("*.py"), *Path(__file__).parent.glob("*.py")]
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    with (
        ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize,
            initargs=(str(args.vocab), settings, specifications),
        ) as executor,
        (args.out / "records.jsonl").open("x") as stream,
    ):
        pending = []
        for batch in batches(args.records):
            pending.append(executor.submit(normalize_batch, batch))
            if len(pending) < args.workers * 2:
                continue
            for record in pending.pop(0).result():
                stream.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                )
                counts["rows"] += 1
                counts["rejected" if record["rejection_reasons"] else "validated"] += 1
                counts.update(
                    "rejection:" + value for value in record["rejection_reasons"]
                )
                if not record["rejection_reasons"]:
                    categories[record["category"]] += 1
            if counts["rows"] % 16384 == 0:
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
        for future in pending:
            for record in future.result():
                stream.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                )
                counts["rows"] += 1
                counts["rejected" if record["rejection_reasons"] else "validated"] += 1
                counts.update(
                    "rejection:" + value for value in record["rejection_reasons"]
                )
                if not record["rejection_reasons"]:
                    categories[record["category"]] += 1
    report = {
        "schema_version": 1,
        "counts": dict(counts),
        "categories": dict(categories),
        "seconds": time.monotonic() - started,
        "workers": args.workers,
        "code_sha256": source_hashes,
        "sources_unchanged": all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
            for path, value in source_hashes.items()
        ),
        "input_records": str(args.records.resolve()),
        "training_release_approved": False,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "code_sha256"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

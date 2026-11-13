"""Build a versioned chemical interpretation of validated candidate labels."""

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pysqlite3 as sqlite3
from molsight.abbreviations import (
    REGISTRY_SHA256,
    REGISTRY_VERSION,
    resolve_abbreviations,
)
from molsight.cpu import available_cpus
from molsight.structure import StructureError, parse_structure
from molsight.tokenizer import CharTokenizer
from rdkit import Chem, rdBase

from .index import file_hash
from .chemical_quality import review_reasons

_tokenizer = None
_max_len = None


def initialize(vocabulary, max_len):
    global _tokenizer, _max_len
    _tokenizer = CharTokenizer(vocabulary)
    _max_len = max_len


def interpret(target):
    try:
        result = resolve_abbreviations(parse_structure(target))
        document = result.document
        label = document.model_text
        tokens, indices = _tokenizer.smiles_to_sequence(label)
        if _tokenizer.sequence_to_smiles(tokens[1:])["smiles"] != label:
            raise StructureError("tokenizer_changed_label")
        if len(tokens) > _max_len:
            raise StructureError("resolved_target_exceeds_max_len")
        roles = {kind for kind, _ in document.semantic_labels}
        molecule = document.molecule
        if roles & {"a", "r", "c", "rc", "sgroup", "repeat", "variation"}:
            category = "markush"
        elif "d" in roles or any(
            atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()
        ):
            category = "attachment"
        else:
            category = "complete"
            reasons = review_reasons(molecule)
            if reasons:
                raise StructureError("annotation_requires_review:" + ",".join(reasons))
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        Chem.RemoveStereochemistry(molecule)
        family = hashlib.sha256(Chem.MolToSmiles(molecule).encode()).hexdigest()
        return (
            target,
            label,
            family,
            category,
            len(tokens),
            len(indices),
            json.dumps(result.replacements),
            json.dumps(result.unresolved),
            None,
        )
    except (StructureError, ValueError, RuntimeError) as error:
        return (
            target,
            None,
            None,
            None,
            None,
            None,
            "[]",
            "[]",
            f"{type(error).__name__}:{error}",
        )


def interpret_batch(targets):
    return [interpret(target) for target in targets]


def batches(connection, size=128):
    cursor = connection.execute(
        "SELECT DISTINCT target FROM records WHERE valid=1 ORDER BY target"
    )
    while rows := cursor.fetchmany(size):
        yield [row[0] for row in rows]


def build_resolutions(index, vocabulary, output, *, workers, max_len=2048):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        str(Path(path).resolve()): file_hash(path) for path in (index, vocabulary)
    }
    modules = (
        "molsight.structure",
        "molsight.abbreviations",
        "molsight.tokenizer",
        "molsight.cpu",
        "training.molsight_data.chemical_quality",
    )
    sources = [
        Path(__file__),
        Path(__import__("training.molsight_data.index", fromlist=[""]).__file__),
    ]
    sources += [Path(__import__(name, fromlist=[""]).__file__) for name in modules]
    code = {str(path.resolve()): file_hash(path) for path in sources}
    counts, categories, failures = Counter(), Counter(), Counter()
    started = time.monotonic()
    with (
        sqlite3.connect(
            Path(index).resolve().as_uri() + "?mode=ro", uri=True
        ) as source,
        sqlite3.connect(output / "labels.sqlite") as destination,
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize,
            initargs=(str(vocabulary), max_len),
        ) as executor,
    ):
        destination.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE labels(
                target TEXT PRIMARY KEY, resolved_target TEXT, structure_family TEXT,
                category TEXT, token_length INTEGER, atom_count INTEGER,
                replacements TEXT NOT NULL, unresolved TEXT NOT NULL, error TEXT
            ) WITHOUT ROWID;
        """)
        pending = []

        def collect(future):
            rows = future.result()
            destination.executemany(
                "INSERT INTO labels VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            counts["targets"] += len(rows)
            for row in rows:
                if row[-1] is not None:
                    failures[row[-1]] += 1
                    counts["rejected"] += 1
                    continue
                counts["validated"] += 1
                counts["expanded"] += int(row[6] != "[]")
                counts["restricted_abbreviation_context"] += int(row[7] != "[]")
                categories[row[3]] += 1
            if counts["targets"] % 8192 == 0:
                destination.commit()
                print(
                    json.dumps(
                        {
                            "targets": counts["targets"],
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

        for batch in batches(source):
            pending.append(executor.submit(interpret_batch, batch))
            if len(pending) >= 2 * workers:
                collect(pending.pop(0))
        for future in pending:
            collect(future)
        destination.commit()
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    report = {
        "counts": dict(counts),
        "categories": dict(categories),
        "rejections": dict(failures),
        "workers": workers,
        "seconds": time.monotonic() - started,
        "max_len": max_len,
        "inputs": inputs,
        "code_sha256": code,
        "rdkit_version": rdBase.rdkitVersion,
        "registry": {"version": REGISTRY_VERSION, "sha256": REGISTRY_SHA256},
        "input_files_unchanged": all(
            file_hash(path) == digest for path, digest in inputs.items()
        ),
        "sources_unchanged": all(
            file_hash(path) == digest for path, digest in code.items()
        ),
        "training_release_approved": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    if min(args.workers, args.max_len) < 1:
        parser.error("Worker count and context length must be positive")
    print(
        json.dumps(
            build_resolutions(
                args.index,
                args.vocab,
                args.out,
                workers=args.workers,
                max_len=args.max_len,
            )
        )
    )


if __name__ == "__main__":
    main()

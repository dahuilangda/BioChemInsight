"""Index recognition candidates and audit conflicting labels for identical images."""

import argparse
import hashlib
import json
import re
import pysqlite3 as sqlite3
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from multiprocessing import get_context

from molsight.cpu import available_cpus
from molsight.abbreviations import (
    REGISTRY_SHA256,
    REGISTRY_VERSION,
    resolve_abbreviations,
)
from molsight.structure import parse_structure
from rdkit import rdBase


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_id(value):
    match = re.search(r"(US|WO|EP|JP|CN)([0-9]{7,})", str(value))
    return match[1] + str(int(match[2])) if match else None


def source_key(record):
    fields = [
        record["source_path"],
        record["source_row"],
        record.get("drawing_style", ""),
    ]
    return hashlib.sha256(
        json.dumps(fields, separators=(",", ":")).encode()
    ).hexdigest()


def ingest(connection, path, *, override=False):
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            reasons = row["rejection_reasons"]
            if not isinstance(reasons, list):
                raise ValueError("Rejection reasons must be a list")
            valid = not reasons
            if valid and any(
                not row.get(key)
                for key in ("SMILES", "file_path", "pixel_sha256", "structure_family")
            ):
                raise ValueError(
                    "Validated candidate has incomplete label or image metadata"
                )
            key = source_key(row)
            if override:
                previous = connection.execute(
                    "SELECT valid FROM records WHERE record_key=?", (key,)
                ).fetchone()
                if previous != (0,):
                    raise ValueError(
                        "Overrides must refer to an existing rejected source record"
                    )
            values = (
                key,
                row["source_path"],
                row["source_row"],
                row.get("drawing_style", ""),
                row.get("source"),
                row.get("source_split"),
                row.get("category"),
                int(valid),
                row.get("pixel_sha256"),
                row.get("structure_family"),
                row.get("composition_group"),
                publication_id(row.get("file_path")),
                row.get("SMILES"),
                line.strip(),
            )
            if override:
                connection.execute(
                    """UPDATE records SET source_path=?, source_row=?, variant=?,
                    source=?, source_split=?, category=?, valid=?, pixel_hash=?, structure_family=?,
                    composition_group=?, publication_id=?, target=?, payload=? WHERE record_key=?""",
                    values[1:] + values[:1],
                )
            else:
                connection.execute(
                    """INSERT INTO records(record_key,source_path,source_row,variant,
                    source,source_split,category,valid,pixel_hash,structure_family,composition_group,
                    publication_id,target,payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
            count += 1
    connection.commit()
    return count


def enrich_publications(connection, specifications=None):
    import pyarrow.parquet as pq

    updates = 0
    paths = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT source_path FROM records WHERE valid=1"
        )
    ]
    for path in paths:
        if Path(path).suffix != ".parquet" or not Path(path).is_file():
            continue
        if specifications is not None and path in specifications:
            specification = specifications[path]
            stat = Path(path).stat()
            if stat.st_size != specification.get(
                "source_size", stat.st_size
            ) or stat.st_mtime_ns != specification.get(
                "source_mtime_ns", stat.st_mtime_ns
            ):
                raise ValueError(f"Source metadata changed after inventory: {path}")
        parquet = pq.ParquetFile(path)
        if "image_name" not in parquet.schema.names:
            continue
        names = parquet.read(columns=["image_name"]).column("image_name").to_pylist()
        values = [
            (identity, path, index)
            for index, name in enumerate(names)
            if (identity := publication_id(name)) is not None
        ]
        connection.executemany(
            "UPDATE records SET publication_id=? WHERE source_path=? AND source_row=?",
            values,
        )
        updates += len(values)
    connection.commit()
    return updates


def compare_pixel_group(job):
    pixel_hash, _, conflict, size, _ = compare_pixel_details(job)
    return pixel_hash, conflict, size


def compare_pixel_details(job):
    pixel_hash, rows = job
    labels = list(dict.fromkeys(label for _, label in rows))
    if len(labels) == 1:
        return pixel_hash, False, False, len(rows), []
    documents = [parse_structure(label) for label in labels]
    literal_conflict = any(
        not documents[0].equivalent(other) for other in documents[1:]
    )
    if not literal_conflict:
        return pixel_hash, False, False, len(rows), []
    resolutions = [resolve_abbreviations(document) for document in documents]
    first = resolutions[0].document
    conflict = any(not first.equivalent(other.document) for other in resolutions[1:])
    details = [
        {
            "target": text,
            "resolved_target": result.document.model_text,
            "replacements": result.replacements,
            "unresolved": result.unresolved,
        }
        for text, result in zip(labels, resolutions, strict=True)
    ]
    return pixel_hash, literal_conflict, conflict, len(rows), details


def audit_conflicts(connection, output, workers):
    groups = connection.execute("""SELECT pixel_hash FROM records WHERE valid=1
        GROUP BY pixel_hash HAVING COUNT(*)>1""")
    counts = Counter()
    pending = set()

    def record(result, stream):
        pixel_hash, literal_conflict, conflict, size, resolutions = result
        counts["duplicate_pixel_groups"] += 1
        counts["duplicate_pixel_records"] += size
        if literal_conflict:
            counts["literal_conflicting_pixel_groups"] += 1
            counts["literal_conflicting_pixel_records"] += size
        if literal_conflict and not conflict:
            counts["abbreviation_equivalent_pixel_groups"] += 1
            counts["abbreviation_equivalent_pixel_records"] += size
        if conflict:
            counts["conflicting_pixel_groups"] += 1
            counts["conflicting_pixel_records"] += size
            connection.execute(
                "UPDATE records SET image_conflict=1 WHERE pixel_hash=?", (pixel_hash,)
            )
            rows = connection.execute(
                "SELECT record_key,source_path,source_row,target FROM records WHERE valid=1 AND pixel_hash=?",
                (pixel_hash,),
            ).fetchall()
            stream.write(
                json.dumps(
                    {
                        "pixel_sha256": pixel_hash,
                        "records": rows,
                        "resolution": resolutions,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    with (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as executor,
        Path(output).open("x") as stream,
    ):
        for (pixel_hash,) in groups:
            rows = connection.execute(
                "SELECT record_key,target FROM records WHERE valid=1 AND pixel_hash=?",
                (pixel_hash,),
            ).fetchall()
            pending.add(executor.submit(compare_pixel_details, (pixel_hash, rows)))
            if len(pending) >= 2 * workers:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(future.result(), stream)
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record(future.result(), stream)
    connection.commit()
    return dict(counts)


def build_index(records, overrides, output, *, workers, sources=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_paths = [*records, *overrides, *([sources] if sources is not None else [])]
    inputs = {str(Path(path).resolve()): file_hash(path) for path in input_paths}
    code_paths = list(Path(__file__).parent.glob("*.py"))
    code_paths += [
        Path(__import__(name, fromlist=[""]).__file__)
        for name in ("molsight.structure", "molsight.abbreviations", "molsight.cpu")
    ]
    code = {str(path.resolve()): file_hash(path) for path in code_paths}
    specifications = None
    if sources is not None:
        specifications = {
            row["path"]: row for row in json.loads(Path(sources).read_text())["sources"]
        }
    with sqlite3.connect(output / "candidates.sqlite") as connection:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE records(
                id INTEGER PRIMARY KEY, record_key TEXT NOT NULL UNIQUE,
                source_path TEXT NOT NULL, source_row INTEGER NOT NULL, variant TEXT NOT NULL,
                source TEXT, source_split TEXT, category TEXT, valid INTEGER NOT NULL,
                pixel_hash TEXT, structure_family TEXT, composition_group TEXT, publication_id TEXT,
                target TEXT, payload TEXT NOT NULL, image_conflict INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX source_rows ON records(source_path,source_row);
        """)
        row_counts = {}
        for path in records:
            row_counts[str(path)] = ingest(connection, path)
        for path in overrides:
            row_counts[str(path)] = ingest(connection, path, override=True)
        connection.executescript("""
            CREATE INDEX pixels ON records(pixel_hash);
            CREATE INDEX families ON records(structure_family);
            CREATE INDEX publications ON records(publication_id);
            CREATE INDEX compositions ON records(composition_group);
        """)
        publication_rows = enrich_publications(connection, specifications)
        conflicts = audit_conflicts(
            connection, output / "conflicting_images.jsonl", workers
        )
        counts = dict(
            connection.execute("SELECT valid,COUNT(*) FROM records GROUP BY valid")
        )
        categories = dict(
            connection.execute(
                "SELECT category,COUNT(*) FROM records WHERE valid=1 GROUP BY category"
            )
        )
        unique_images = connection.execute(
            "SELECT COUNT(DISTINCT pixel_hash) FROM records WHERE valid=1"
        ).fetchone()[0]
        report = {
            "inputs": inputs,
            "input_rows": row_counts,
            "records": sum(counts.values()),
            "parser_validated": counts.get(1, 0),
            "parser_rejected": counts.get(0, 0),
            "categories": categories,
            "unique_validated_pixel_images": unique_images,
            "publication_metadata_rows": publication_rows,
            "conflicts": conflicts,
            "training_release_approved": False,
            "rdkit_version": rdBase.rdkitVersion,
            "code_sha256": code,
            "sqlite_version": sqlite3.sqlite_version,
            "abbreviation_registry": {
                "version": REGISTRY_VERSION,
                "sha256": REGISTRY_SHA256,
            },
            "sources_unchanged": all(
                file_hash(path) == value for path, value in code.items()
            ),
            "input_files_unchanged": all(
                file_hash(path) == value for path, value in inputs.items()
            ),
        }
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--overrides", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    report = build_index(
        args.records,
        args.overrides,
        args.out,
        workers=args.workers,
        sources=args.sources,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "code_sha256"}
        )
    )


if __name__ == "__main__":
    main()

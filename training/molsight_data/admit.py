"""Admit an immutable recognition export after its recorded checks pass."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pysqlite3 as sqlite3
from pyarrow import ipc

from .index import file_hash, source_key


def audit_catalog(catalog, index):
    path = Path(catalog["records_file"])
    digest = catalog["records_sha256"]
    if file_hash(path) != digest:
        raise ValueError("Peptide catalog records changed after review")
    keys = set()
    with sqlite3.connect(
        Path(index).resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                key = source_key(record)
                if key in keys:
                    raise ValueError(
                        "Peptide catalog contains duplicate source records"
                    )
                keys.add(key)
                row = connection.execute(
                    "SELECT valid,category,payload FROM records WHERE record_key=?",
                    (key,),
                ).fetchone()
                if (
                    row is None
                    or row[:2] != (1, "peptide")
                    or json.loads(row[2]) != record
                ):
                    raise ValueError(
                        "Peptide catalog differs from the indexed supervision"
                    )
    if len(keys) != catalog["counts"]["images"] or file_hash(path) != digest:
        raise ValueError("Peptide catalog coverage or provenance changed")
    return {"records": len(keys), "records_sha256": digest}


def audit_lineage(export, index, labels, assignments):
    with sqlite3.connect(":memory:", uri=True) as connection:
        connection.execute(
            "CREATE TABLE exported(id INTEGER PRIMARY KEY, key TEXT, component TEXT, split TEXT, target TEXT, "
            "image_path TEXT, image_hash TEXT, pixel_hash TEXT, input_size INTEGER, category TEXT)"
        )
        for item in export["files"].values():
            with pa.memory_map(item["path"], "r") as mapping:
                reader = ipc.open_file(mapping)
                for batch_index in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_index).select(
                        [
                            "source_record_id",
                            "record_key",
                            "component",
                            "split",
                            "SMILES",
                            "file_path",
                            "image_sha256",
                            "pixel_sha256",
                            "input_size",
                            "category",
                        ]
                    )
                    rows = [
                        (
                            row["source_record_id"],
                            row["record_key"],
                            row["component"],
                            row["split"],
                            row["SMILES"],
                            row["file_path"],
                            row["image_sha256"],
                            row["pixel_sha256"],
                            row["input_size"],
                            row["category"],
                        )
                        for row in batch.to_pylist()
                    ]
                    connection.executemany(
                        "INSERT INTO exported VALUES (?,?,?,?,?,?,?,?,?,?)", rows
                    )
        for name, path in (
            ("staging", index),
            ("resolved", labels),
            ("assigned", assignments),
        ):
            connection.execute(
                f"ATTACH DATABASE ? AS {name}",
                (Path(path).resolve().as_uri() + "?mode=ro",),
            )
        mismatches = connection.execute("""SELECT COUNT(*) FROM exported e
            LEFT JOIN staging.records r ON e.id=r.id
            LEFT JOIN assigned.assignments a ON e.id=a.record_id
            LEFT JOIN resolved.labels l ON r.target=l.target
            WHERE r.id IS NULL OR a.record_id IS NULL OR l.target IS NULL
            OR r.valid<>1 OR a.selected<>1 OR l.error IS NOT NULL
            OR e.key<>r.record_key OR e.component<>a.component OR e.split<>a.split
            OR e.target IS NOT l.resolved_target
            OR e.image_path IS NOT json_extract(r.payload,'$.file_path')
            OR e.image_hash IS NOT json_extract(r.payload,'$.image_sha256')
            OR e.pixel_hash IS NOT r.pixel_hash
            OR e.input_size IS NOT CASE WHEN json_extract(r.payload,'$.model_input_preprocessed')
                THEN json_extract(r.payload,'$.input_size') ELSE 512 END
            OR e.category IS NOT CASE WHEN r.category='peptide' THEN r.category ELSE l.category END
            """).fetchone()[0]
        records = connection.execute("SELECT COUNT(*) FROM exported").fetchone()[0]
        selected = connection.execute(
            "SELECT COUNT(*) FROM assigned.assignments WHERE selected=1"
        ).fetchone()[0]
        if (
            mismatches
            or records != selected
            or records != sum(export["records"].values())
        ):
            raise ValueError(
                "Exported records do not match their selected source supervision"
            )
        overlaps = {}
        for name, field in (
            ("pixel_sha256", "r.pixel_hash"),
            ("resolved_structure_family", "l.structure_family"),
            ("publication_id", "r.publication_id"),
            ("peptide_composition", "r.composition_group"),
        ):
            overlaps[name] = connection.execute(f"""SELECT COUNT(*) FROM (
                SELECT {field} FROM exported e JOIN staging.records r ON e.id=r.id
                JOIN resolved.labels l ON r.target=l.target
                WHERE {field} IS NOT NULL GROUP BY {field} HAVING COUNT(DISTINCT e.split)>1)""").fetchone()[0]
        if any(overlaps.values()):
            raise ValueError("Exported supervision crosses protected split relations")
        return {
            "records": records,
            "target_mismatches": mismatches,
            "image_identity_checked": True,
            "category_checked": True,
            "cross_split_groups": overlaps,
        }


def admit(
    manifest,
    image_audit,
    split_report,
    peptide_catalog,
    output,
    *,
    index,
    labels,
    assignments,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        manifest,
        image_audit,
        split_report,
        peptide_catalog,
        index,
        labels,
        assignments,
    ]
    inputs = {str(Path(path).resolve()): file_hash(path) for path in paths}
    export, images, splits, peptides = [
        json.loads(Path(path).read_text()) for path in paths[:4]
    ]
    for path in (index, labels, assignments):
        if (
            export["inputs"].get(str(Path(path).resolve()))
            != inputs[str(Path(path).resolve())]
        ):
            raise ValueError("Source supervision differs from the exported provenance")
    for report in (export, images, splits):
        if (
            report.get("sources_unchanged") is not True
            or report.get("input_files_unchanged") is not True
        ):
            raise ValueError("Preparation provenance did not remain unchanged")
    if images.get("passed") is not True or images.get("failed_records") != 0:
        raise ValueError("Recognition image/token audit did not pass")
    if images["records"] != export["records"] or images["inputs"].get(
        str(Path(manifest).resolve())
    ) != file_hash(manifest):
        raise ValueError("Image audit does not cover this recognition export")
    if set(splits["cross_split_groups"]) != {
        "pixel_sha256",
        "resolved_structure_family",
        "publication_id",
        "peptide_composition",
    }:
        raise ValueError("Required split relations were not audited")
    if any(splits["cross_split_groups"].values()):
        raise ValueError("Recognition export contains overlapping holdouts")
    if {name: splits["selected"][name] for name in export["records"]} != export[
        "records"
    ]:
        raise ValueError("Split selections differ from the recognition export")
    if (
        sum(
            export["categories"].get(f"{split}:peptide", 0)
            for split in export["records"]
        )
        != peptides["counts"]["images"]
    ):
        raise ValueError("Peptide catalog is not covered by the recognition export")
    if peptides.get("sources_unchanged") is not True:
        raise ValueError("Peptide catalog provenance changed during export")
    catalog_lineage = audit_catalog(peptides, index)
    for item in export["files"].values():
        if (
            file_hash(item["path"]) != item["sha256"]
            or images["inputs"].get(item["path"]) != item["sha256"]
        ):
            raise ValueError("Recognition file changed after its image/token audit")
    lineage = audit_lineage(export, index, labels, assignments)
    report = {
        "passed": True,
        "inputs": inputs,
        "records": export["records"],
        "formats": ["char"],
        "source_code_sha256": file_hash(__file__),
        "lineage": lineage,
        "peptide_catalog_lineage": catalog_lineage,
        "files": export["files"],
        "vocabulary_sha256": export["vocabulary_sha256"],
        "scope": "Supported-label pilot and source replay; source annotations are not exhaustively reviewed by a human.",
    }
    report_path = output / "admission.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    admitted = {
        **export,
        "training_release_approved": True,
        "admission_report": {
            "path": str(report_path.resolve()),
            "sha256": file_hash(report_path),
        },
    }
    (output / "manifest.json").write_text(json.dumps(admitted, indent=2) + "\n")
    return admitted


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "image-audit",
        "split-report",
        "peptide-catalog",
        "index",
        "labels",
        "assignments",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            admit(
                args.manifest,
                args.image_audit,
                args.split_report,
                args.peptide_catalog,
                args.out,
                index=args.index,
                labels=args.labels,
                assignments=args.assignments,
            )
        )
    )


if __name__ == "__main__":
    main()

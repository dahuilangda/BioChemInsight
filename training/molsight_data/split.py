"""Assign connected structure, image, publication, and composition groups."""

import argparse
import hashlib
import json
import time
from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import pysqlite3 as sqlite3
from molsight.cpu import available_cpus

from .index import compare_pixel_group, file_hash


class Components:
    def __init__(self, size):
        self.parents = array("I", range(size))
        self.sizes = array("I", [1]) * size

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first, second):
        first, second = self.root(first), self.root(second)
        if first == second:
            return
        if self.sizes[first] < self.sizes[second]:
            first, second = second, first
        self.parents[second] = first
        self.sizes[first] += self.sizes[second]


def pixel_conflicts(source, output, workers):
    pending, conflicts = set(), set()
    counts = Counter()

    def collect(future, stream):
        pixel, conflict, size = future.result()
        counts["duplicate_groups"] += 1
        counts["duplicate_records"] += size
        if conflict:
            conflicts.add(pixel)
            counts["conflicting_groups"] += 1
            counts["conflicting_records"] += size
            stream.write(json.dumps({"pixel_sha256": pixel, "records": size}) + "\n")

    with (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as executor,
        Path(output).open("x") as stream,
    ):
        for (pixel,) in source.execute("""SELECT pixel_hash FROM records WHERE valid=1
                GROUP BY pixel_hash HAVING COUNT(*)>1"""):
            rows = source.execute(
                """SELECT r.record_key,l.resolved_target FROM records r
                JOIN resolved.labels l ON r.target=l.target
                WHERE r.valid=1 AND r.pixel_hash=? AND l.error IS NULL""",
                (pixel,),
            ).fetchall()
            if len(rows) < 2:
                continue
            pending.add(executor.submit(compare_pixel_group, (pixel, rows)))
            if len(pending) >= 2 * workers:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    collect(future, stream)
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                collect(future, stream)
    return conflicts, dict(counts)


def assign_components(
    rows,
    *,
    reserved_compositions=(),
    validation_fraction=0.02,
    peptide_fraction=0.1,
    protected_splits=None,
):
    if not 0 <= validation_fraction < 1 or not 0 <= peptide_fraction < 0.5:
        raise ValueError("Invalid holdout fraction")
    rows = list(rows)
    components = Components(max((row[0] for row in rows), default=0) + 1)
    relations = [dict() for _ in range(4)]
    for index, _, _, pixel, family, composition, publication, _ in rows:
        for relation, value in zip(
            relations, (pixel, family, composition, publication), strict=True
        ):
            if value is not None:
                previous = relation.setdefault(value, index)
                components.union(index, previous)
    reserved = set(reserved_compositions)
    protected_splits = {} if protected_splits is None else protected_splits
    if set(protected_splits.values()) - {"train", "validation", "test"}:
        raise ValueError("Invalid protected split")
    priorities, keys, peptides = {}, {}, set()
    for index, key, split, _, _, composition, _, category in rows:
        root = components.root(index)
        priority = {"test": 3, "validation": 2, "train": 1, "review": 1}.get(split)
        if priority is None:
            raise ValueError(f"Unsupported source split: {split}")
        if key in protected_splits:
            priority = max(
                priority,
                {"train": 1, "validation": 2, "test": 3}[protected_splits[key]],
            )
        if composition in reserved:
            priority = max(priority, 2)
        priorities[root] = max(priorities.get(root, 0), priority)
        keys[root] = min(keys.get(root, key), key)
        if category == "peptide":
            peptides.add(root)
    assignments = {}
    for root, key in keys.items():
        priority = priorities[root]
        fraction = (
            int(
                hashlib.sha256(("molsight-split-1:" + key).encode()).hexdigest()[:16],
                16,
            )
            / 2**64
        )
        if priority == 3:
            split = "test"
        elif priority == 2:
            split = "validation"
        elif root in peptides and fraction < peptide_fraction:
            split = "test"
        elif fraction < (
            2 * peptide_fraction if root in peptides else validation_fraction
        ):
            split = "validation"
        else:
            split = "train"
        assignments[root] = (key, split)
    return {row[0]: assignments[components.root(row[0])] for row in rows}


def build_split(
    index,
    labels,
    baseline,
    output,
    *,
    workers,
    previous_index=None,
    previous_assignments=None,
):
    if (previous_index is None) != (previous_assignments is None):
        raise ValueError("Previous index and assignments must be supplied together")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = {
        str(Path(path).resolve()): file_hash(path) for path in (index, labels, baseline)
    }
    for path in (previous_index, previous_assignments):
        if path is not None:
            inputs[str(Path(path).resolve())] = file_hash(path)
    code_paths = [
        Path(__file__),
        Path(__import__("training.molsight_data.index", fromlist=[""]).__file__),
    ]
    code_paths += [
        Path(__import__(name, fromlist=[""]).__file__)
        for name in ("molsight.structure", "molsight.abbreviations", "molsight.cpu")
    ]
    code = {str(path.resolve()): file_hash(path) for path in code_paths}
    reserved = {
        row["composition_group"]
        for row in json.loads(Path(baseline).read_text())
        if row.get("composition_group")
    }
    with (
        sqlite3.connect(
            Path(index).resolve().as_uri() + "?mode=ro", uri=True
        ) as source,
        sqlite3.connect(output / "assignments.sqlite") as destination,
    ):
        source.execute(
            "ATTACH DATABASE ? AS resolved",
            (Path(labels).resolve().as_uri() + "?mode=ro",),
        )
        conflicts, conflict_counts = pixel_conflicts(
            source, output / "conflicting_images.jsonl", workers
        )
        print(json.dumps({"pixel_audit": conflict_counts}), flush=True)
        rows = source.execute("""SELECT r.id,r.record_key,r.source_split,r.pixel_hash,
            l.structure_family,r.composition_group,r.publication_id,
            CASE WHEN r.category='peptide' THEN r.category ELSE l.category END
            FROM records r JOIN resolved.labels l ON r.target=l.target
            WHERE r.valid=1 AND l.error IS NULL ORDER BY r.id""").fetchall()
        protected = {}
        if previous_index is not None:
            source.execute(
                "ATTACH DATABASE ? AS previous_records",
                (Path(previous_index).resolve().as_uri() + "?mode=ro",),
            )
            source.execute(
                "ATTACH DATABASE ? AS previous_split",
                (Path(previous_assignments).resolve().as_uri() + "?mode=ro",),
            )
            protected = dict(
                source.execute(
                    "SELECT r.record_key,a.split FROM previous_records.records r "
                    "JOIN previous_split.assignments a ON a.record_id=r.id"
                )
            )
        assignments = assign_components(
            rows, reserved_compositions=reserved, protected_splits=protected
        )
        violations = sum(
            (protected.get(row[1]) == "test" and assignments[row[0]][1] != "test")
            or (
                protected.get(row[1]) == "validation"
                and assignments[row[0]][1] == "train"
            )
            for row in rows
        )
        if violations:
            raise ValueError("A protected evaluation record reached a training split")
        destination.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE assignments(
                record_id INTEGER PRIMARY KEY, component TEXT NOT NULL, split TEXT NOT NULL,
                selected INTEGER NOT NULL, reason TEXT
            );
            CREATE INDEX groups ON assignments(component);
        """)
        priority = {"test": 3, "validation": 2, "review": 1, "train": 0}
        winners = {}
        for index_id, key, original, pixel, *_ in rows:
            if pixel in conflicts:
                continue
            rank = (-priority[original], key)
            if pixel not in winners or rank < winners[pixel][0]:
                winners[pixel] = (rank, index_id)
        counts, categories, original_moves = Counter(), Counter(), Counter()
        sizes = Counter(group for group, _ in assignments.values())
        values = []
        for index_id, _, original, pixel, _, _, _, category in rows:
            group, split = assignments[index_id]
            reason = (
                "conflicting_image_labels"
                if pixel in conflicts
                else (
                    None
                    if winners[pixel][1] == index_id
                    else "equivalent_duplicate_image"
                )
            )
            selected = reason is None
            values.append((index_id, group, split, int(selected), reason))
            if selected:
                counts[split] += 1
                categories[f"{split}:{category}"] += 1
                if original == "train" and split != "train":
                    original_moves[split] += 1
            else:
                counts["excluded:" + reason] += 1
        destination.executemany("INSERT INTO assignments VALUES (?,?,?,?,?)", values)
        destination.commit()
        destination.execute(
            "ATTACH DATABASE ? AS staging",
            (Path(index).resolve().as_uri() + "?mode=ro",),
        )
        destination.execute(
            "ATTACH DATABASE ? AS resolved",
            (Path(labels).resolve().as_uri() + "?mode=ro",),
        )
        overlaps = {}
        for name, field in (
            ("pixel_sha256", "r.pixel_hash"),
            ("resolved_structure_family", "l.structure_family"),
            ("publication_id", "r.publication_id"),
            ("peptide_composition", "r.composition_group"),
        ):
            overlaps[name] = destination.execute(f"""SELECT COUNT(*) FROM (
                SELECT {field} FROM assignments a JOIN staging.records r ON a.record_id=r.id
                JOIN resolved.labels l ON r.target=l.target
                WHERE a.selected=1 AND {field} IS NOT NULL GROUP BY {field}
                HAVING COUNT(DISTINCT a.split)>1)""").fetchone()[0]
        if any(overlaps.values()):
            raise ValueError("Cross-split relationships remain after grouping")
        destination.execute("PRAGMA main.wal_checkpoint(TRUNCATE)")
        report = {
            "inputs": inputs,
            "code_sha256": code,
            "workers": workers,
            "records": len(rows),
            "components": len(sizes),
            "largest_component_records": max(sizes.values(), default=0),
            "selected": dict(counts),
            "categories": dict(categories),
            "original_train_records_moved_to_holdouts": dict(original_moves),
            "pixel_audit": conflict_counts,
            "reserved_peptide_compositions": len(reserved),
            "relations": [
                "pixel_sha256",
                "resolved_structure_family",
                "publication_id",
                "peptide_composition",
            ],
            "cross_split_groups": overlaps,
            "previous_split_records": len(protected),
            "previous_holdout_violations": violations,
            "seconds": time.monotonic() - started,
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
    for name in ("index", "labels", "baseline", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--previous-index", type=Path)
    parser.add_argument("--previous-assignments", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    print(
        json.dumps(
            build_split(
                args.index,
                args.labels,
                args.baseline,
                args.out,
                workers=args.workers,
                previous_index=args.previous_index,
                previous_assignments=args.previous_assignments,
            )
        )
    )


if __name__ == "__main__":
    main()

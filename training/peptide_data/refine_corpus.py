"""Rebuild rejected layouts while retaining verified corpus artifacts and lineage."""

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from .engine_corpus import _worker, source_hashes
from peptidecore import from_helm


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_file(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f"Corpus artifact differs from its manifest: {relative}")
    return path


def refine(corpus, output, *, workers, deadline_seconds):
    if workers < 1 or deadline_seconds <= 0:
        raise ValueError("Workers and layout budget must be positive")
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    manifest_path = corpus / "manifest.json"
    parent_hash = digest(manifest_path)
    parent = json.loads(manifest_path.read_text())
    if (
        not parent.get("sources_unchanged")
        or len(parent["record_files_sha256"]) != parent["designed"]
    ):
        raise ValueError("Corpus lacks complete generation provenance")
    sources = source_hashes()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("records", "mol", "svg"):
        (output / name).mkdir()
    started, records, jobs = time.monotonic(), {}, []
    lineage = {}
    for name, expected in sorted(parent["record_files_sha256"].items()):
        path = verified_file(corpus, "records/" + name, expected)
        row = json.loads(path.read_text())
        index = row["index"]
        if type(index) is not int or index in lineage:
            raise ValueError("Corpus has inconsistent record identifiers")
        lineage[index] = {"manifest_sha256": parent_hash, "record_sha256": expected}
        if row["status"] == "accepted":
            for kind in ("mol", "svg"):
                artifact = verified_file(
                    corpus, kind + "/" + row[kind], row[kind + "_sha256"]
                )
                target = output / kind / row[kind]
                if target.exists():
                    raise ValueError("Corpus artifact identifiers are not unique")
                shutil.copyfile(artifact, target)
            records[index] = row
        else:
            chemical = from_helm(row["helm"])
            jobs.append(
                (
                    index,
                    chemical.smiles,
                    row["helm"],
                    row["stratum"],
                    row["seed"],
                    deadline_seconds,
                    str(output),
                )
            )
    if jobs:
        with Pool(min(workers, len(jobs))) as pool:
            for row in pool.imap_unordered(_worker, jobs):
                records[row["index"]] = row
                print(
                    json.dumps(
                        {
                            "index": row["index"],
                            "status": row["status"],
                            "seconds": row["seconds"],
                        }
                    ),
                    flush=True,
                )
    hashes = {}
    for index, row in sorted(records.items()):
        row["source_record"] = lineage[index]
        row["training_admitted"] = False
        path = output / "records" / f"{index:06d}.json"
        path.write_text(json.dumps(row, indent=2) + "\n")
        hashes[path.name] = digest(path)
    if len(hashes) != parent["designed"] or digest(manifest_path) != parent_hash:
        raise ValueError("Corpus coverage or source manifest changed during refinement")
    report = {
        "designed": len(records),
        "accepted": sum(row["status"] == "accepted" for row in records.values()),
        "failed": sum(row["status"] != "accepted" for row in records.values()),
        "refined": len(jobs),
        "by_stratum": dict(
            Counter(
                row["stratum"]
                for row in records.values()
                if row["status"] == "accepted"
            )
        ),
        "source_manifest_sha256": parent_hash,
        "source_sha256": sources,
        "sources_unchanged": all(
            digest(path) == value for path, value in sources.items()
        ),
        "record_files_sha256": hashes,
        "workers": min(workers, len(jobs)),
        "deadline_seconds": deadline_seconds,
        "wall_seconds": time.monotonic() - started,
        "training_admitted": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        key: value
        for key, value in report.items()
        if key not in ("source_sha256", "record_files_sha256")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            refine(
                args.corpus,
                args.out,
                workers=args.workers,
                deadline_seconds=args.deadline_seconds,
            )
        )
    )


if __name__ == "__main__":
    main()

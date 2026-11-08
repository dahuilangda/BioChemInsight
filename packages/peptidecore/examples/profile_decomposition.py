"""Measure decomposition of SMILES supplied in a JSON list or mapping."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from importlib.resources import files
from pathlib import Path

from rdkit import rdBase

from peptidecore import PeptideError, core_library, decompose_smiles, to_helm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-solutions", type=int, default=8)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.max_solutions < 2
    ):
        parser.error("Positive repeat/timeout values and at least two solutions are required")
    payload = args.input.read_bytes()
    source = json.loads(payload)
    if isinstance(source, dict):
        source = [dict(record, id=key) for key, record in source.items()]
    if (
        not isinstance(source, list)
        or not source
        or any(
            not isinstance(row, dict) or not isinstance(row.get("smiles"), str) for row in source
        )
    ):
        parser.error("Input must contain nonempty records with SMILES strings")
    started = time.perf_counter()
    library = core_library()
    library_seconds = time.perf_counter() - started
    try:
        warmup = decompose_smiles(
            source[0]["smiles"], timeout=args.timeout, max_solutions=args.max_solutions
        )
        warmup_seconds = warmup.seconds
    except PeptideError:
        warmup_seconds = None
    rows, durations, statuses = [], [], Counter()
    verified = 0
    for index, case in enumerate(source):
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            try:
                result = decompose_smiles(
                    case["smiles"], timeout=args.timeout, max_solutions=args.max_solutions
                )
                sample = {
                    "status": result.status,
                    "seconds": result.seconds,
                    "coverage": result.coverage,
                    "candidate_count": result.candidate_count,
                    "states": result.states,
                    "search_exhausted": result.search_exhausted,
                    "limits_reached": list(result.limits_reached),
                    "verification_failures": result.verification_failures,
                    "helm": [to_helm(record.graph) for record in result.records],
                }
                verified += bool(result.records)
            except PeptideError as error:
                sample = {
                    "status": "error",
                    "seconds": time.perf_counter() - started,
                    "error": error.code,
                }
            samples.append(sample)
            durations.append(sample["seconds"])
            statuses[sample["status"]] += 1
        rows.append(
            {
                "id": str(case.get("symbol", case.get("id", index))),
                "smiles": case["smiles"],
                "runs": samples,
            }
        )
    summary = {
        "cases": len(rows),
        "measurements": len(durations),
        "statuses": dict(statuses),
        "verified_measurements": verified,
        "total_seconds": sum(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
    }
    report = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "rdkit_version": rdBase.rdkitVersion,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "library_digest": library.digest,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files("peptidecore").iterdir()
            if p.name.endswith(".py")
        },
        "configuration": {
            "repeats": args.repeats,
            "timeout": args.timeout,
            "max_solutions": args.max_solutions,
        },
        "setup": {
            "library_seconds": library_seconds,
            "first_decomposition_seconds": warmup_seconds,
        },
        "summary": summary,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

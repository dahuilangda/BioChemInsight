"""Measure layout acceptance and runtime on fixed molecular inputs."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from rdkit import rdBase

import cyclopeptide2d
from cyclopeptide2d import validate_input
from cyclopeptide2d.solver import residue_pose_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=350)
    parser.add_argument("--budget", type=float, default=100)
    args = parser.parse_args()
    if args.repeats < 1 or args.maxiter < 1 or args.budget <= 0:
        parser.error("Repeats, iterations and budget must be positive")
    cases = json.loads(args.cases.read_text())
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases require unique identifiers")
    directory = Path(cyclopeptide2d.__file__).parent
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.py"))
    }
    result = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "source_directory": str(directory),
        "sources": sources,
        "maxiter": args.maxiter,
        "budget_seconds": args.budget,
        "repeats": args.repeats,
        "records": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        molecule = validate_input(case["smiles"])
        runs = []
        for repeat in range(args.repeats):
            start = time.monotonic()
            _, report = residue_pose_layout(
                molecule,
                seed=case.get("seed", 0),
                maxiter=args.maxiter,
                deadline=start + args.budget,
            )
            runs.append(
                {
                    "repeat": repeat,
                    "seconds": time.monotonic() - start,
                    "accepted": bool(report.get("qa_pass")),
                    "report": report,
                }
            )
        result["records"].append({"case": case, "runs": runs})
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "accepted": [run["accepted"] for run in runs],
                    "seconds": [run["seconds"] for run in runs],
                }
            ),
            flush=True,
        )
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    result["sources_unchanged"] = all(
        hashlib.sha256((directory / name).read_bytes()).hexdigest() == value
        for name, value in sources.items()
    )
    timings = [run["seconds"] for record in result["records"] for run in record["runs"]]
    result["summary"] = {
        "cases": len(cases),
        "runs": len(timings),
        "accepted_runs": sum(
            run["accepted"] for record in result["records"] for run in record["runs"]
        ),
        "median_seconds": float(np.median(timings)),
        "p95_seconds": float(np.percentile(timings, 95)),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

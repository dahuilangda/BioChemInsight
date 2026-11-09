"""Build a cyclic-peptide corpus from engine-accepted layouts.

Pipeline per structure: design (HELM graph) -> assemble -> engine layout
(cyclopeptide2d) -> accepted record with SVG. Every accepted record carries
its canonical SMILES, HELM, engine report (defect rank, phase timings), and
an SVG rendering; failures keep their reason. A final manifest records the
SHA-256 of every record file and the per-stratum outcome counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "cyclopeptide2d" / "src"))
sys.path.insert(0, str(REPO / "packages" / "peptidecore" / "src"))
sys.path.insert(0, str(REPO / "training"))

from cyclopeptide2d import depict, render_svg  # noqa: E402
from peptide_data.design import STRATA, build_design  # noqa: E402


def _worker(job):
    index, smiles, helm, stratum, seed, deadline_seconds, out_dir = job
    started = time.monotonic()
    deadline = started + deadline_seconds
    record = {
        "index": index,
        "stratum": stratum,
        "helm": helm,
        "seed": seed,
        "status": "failed",
    }
    try:
        result = depict(smiles, seed=seed, maxiter=350, time_budget=deadline_seconds)
        report = result.report
        record["engine_report"] = {
            "residues": report.get("residues"),
            "topology": report.get("topology"),
            "defect_rank": report.get("defect_rank"),
            "phase_seconds": report.get("phase_seconds"),
            "constraints": report.get("constraints"),
        }
        if not result.accepted:
            record["reason"] = report.get("error") or "qa_failed"
            return record
        svg = render_svg(result, width=1024, height=1024)
        payload = json.dumps(
            {**record, "smiles": result.report.get("input_smiles", smiles)},
            sort_keys=True,
        ).encode()
        record_key = hashlib.sha256(payload).hexdigest()[:24]
        svg_path = Path(out_dir) / "svg" / f"{record_key}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        record.update(
            status="accepted",
            record_key=record_key,
            smiles=result.report.get("input_smiles", smiles),
            svg=str(svg_path.name),
            seconds=round(time.monotonic() - started, 2),
        )
        return record
    except Exception as exc:  # noqa: BLE001 - keep the batch alive
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.monotonic() - started, 2)
        return record


def build_corpus(
    *, out: Path, seed: int, per_stratum: int, strata: tuple[str, ...],
    workers: int, deadline_seconds: int,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(exist_ok=True)
    designs = build_design(seed=seed, per_stratum=per_stratum, strata=strata)
    records = designs if isinstance(designs, list) else designs.get("records", [])
    jobs = []
    for index, r in enumerate(records):
        smiles = r["chemical_record"]["smiles"]
        jobs.append((index, smiles, r["helm"], r["stratum"], seed + index,
                     deadline_seconds, str(out)))
    print(f"designed {len(jobs)} structures; laying out on {workers} workers",
          flush=True)
    started = time.monotonic()
    outcomes = []
    with Pool(workers) as pool:
        for done, record in enumerate(pool.imap_unordered(_worker, jobs), 1):
            outcomes.append(record)
            if done % 50 == 0 or done == len(jobs):
                accepted = sum(1 for r in outcomes if r["status"] == "accepted")
                print(f"  {done}/{len(jobs)} accepted={accepted} "
                      f"({time.monotonic()-started:.0f}s)", flush=True)
    for record in outcomes:
        path = out / "records" / f"{record['index']:06d}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    accepted = [r for r in outcomes if r["status"] == "accepted"]
    manifest = {
        "seed": seed,
        "strata": list(strata),
        "designed": len(jobs),
        "accepted": len(accepted),
        "failed": len(outcomes) - len(accepted),
        "by_stratum": dict(Counter(r["stratum"] for r in accepted)),
        "failures": Counter(r.get("reason", "?") for r in outcomes
                            if r["status"] != "accepted").most_common(8),
        "workers": workers,
        "deadline_seconds": deadline_seconds,
        "wall_seconds": round(time.monotonic() - started, 1),
        "record_files_sha256": {
            f"{r['index']:06d}.json": hashlib.sha256(
                (out / "records" / f"{r['index']:06d}.json").read_bytes()
            ).hexdigest()
            for r in outcomes
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--per-stratum", type=int, default=500)
    parser.add_argument("--strata", default="head_to_tail_patent,head_to_tail_patent_large")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--deadline-seconds", type=int, default=100)
    args = parser.parse_args()
    strata = tuple(s for s in args.strata.split(",") if s)
    unknown = set(strata) - set(STRATA)
    if unknown:
        raise SystemExit(f"unknown strata: {sorted(unknown)}")
    manifest = build_corpus(
        out=args.out, seed=args.seed, per_stratum=args.per_stratum,
        strata=strata, workers=args.workers, deadline_seconds=args.deadline_seconds,
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "record_files_sha256"}, indent=1))


if __name__ == "__main__":
    main()

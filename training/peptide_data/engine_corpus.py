"""Build a cyclic-peptide corpus from engine-accepted layouts.

Pipeline per structure: design (HELM graph) -> assemble -> engine layout
(cyclopeptide2d) -> accepted record with SVG. Every accepted record carries
its canonical SMILES, HELM, full engine report, coordinates, and SVG.
Geometry acceptance requires a separate rendered-input review before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from cyclopeptide2d import depict_record, render_drawing  # noqa: E402
from peptidecore import from_helm  # noqa: E402
from rdkit import Chem  # noqa: E402
from peptide_data.design import STRATA, build_design  # noqa: E402
from training.molsight_runtime.molsight.cpu import available_cpus  # noqa: E402


def source_hashes():
    paths = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.json")]
    for package in ("peptidecore", "cyclopeptide2d"):
        root = REPO / "packages" / package / "src" / package
        paths.extend(root.rglob("*.py"))
        paths.extend(root.rglob("*.json"))
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _worker(job):
    index, smiles, helm, stratum, seed, deadline_seconds, out_dir = job
    started = time.monotonic()
    record = {
        "index": index,
        "stratum": stratum,
        "helm": helm,
        "seed": seed,
        "status": "failed",
        "training_admitted": False,
    }
    try:
        chemical = from_helm(helm)
        expected = Chem.MolFromSmiles(smiles)
        if expected is None or Chem.MolToSmiles(expected) != chemical.smiles:
            raise ValueError("HELM and design structure differ")
        result = depict_record(
            chemical, seed=seed, maxiter=350, time_budget=deadline_seconds
        )
        report = result.report
        record["engine_report"] = report
        if not result.accepted:
            record["reason"] = report.get("error") or "qa_failed"
            record["seconds"] = round(time.monotonic() - started, 2)
            return record
        svg, atom_positions = render_drawing(result, width=1024, height=1024)
        payload = json.dumps(
            {"helm": helm, "smiles": chemical.smiles, "seed": seed, "stratum": stratum},
            sort_keys=True,
        ).encode()
        record_key = hashlib.sha256(payload).hexdigest()[:24]
        svg_path = Path(out_dir) / "svg" / f"{record_key}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        mol_path = Path(out_dir) / "mol" / f"{record_key}.mol"
        mol_path.write_text(Chem.MolToMolBlock(result.molecule), encoding="utf-8")
        record.update(
            status="accepted",
            record_key=record_key,
            smiles=result.report.get("input_smiles", smiles),
            svg=str(svg_path.name),
            atom_positions=atom_positions,
            canvas_size=[1024, 1024],
            svg_sha256=hashlib.sha256(svg_path.read_bytes()).hexdigest(),
            mol=str(mol_path.name),
            mol_sha256=hashlib.sha256(mol_path.read_bytes()).hexdigest(),
            seconds=round(time.monotonic() - started, 2),
        )
        return record
    except Exception as exc:  # noqa: BLE001 - keep the batch alive
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.monotonic() - started, 2)
        return record


def build_corpus(
    *,
    out: Path,
    seed: int,
    per_stratum: int,
    strata: tuple[str, ...],
    workers: int,
    deadline_seconds: int,
    design: Path | None = None,
) -> dict:
    if workers < 1 or deadline_seconds <= 0 or per_stratum < 1:
        raise ValueError("Workers, design count, and time budget must be positive")
    out.mkdir(parents=True, exist_ok=False)
    (out / "svg").mkdir(exist_ok=True)
    (out / "mol").mkdir()
    (out / "records").mkdir()
    sources = source_hashes()
    design_hash = hashlib.sha256(design.read_bytes()).hexdigest() if design else None
    designs = (
        json.loads(design.read_text())
        if design
        else build_design(seed=seed, per_stratum=per_stratum, strata=strata)
    )
    records = designs if isinstance(designs, list) else designs.get("records", [])
    if not records:
        raise ValueError("Corpus requires nonempty chemical designs")
    if design:
        strata = tuple(sorted({record["stratum"] for record in records}))
    jobs = []
    for index, r in enumerate(records):
        smiles = r["chemical_record"]["smiles"]
        jobs.append(
            (
                index,
                smiles,
                r["helm"],
                r["stratum"],
                seed + index,
                deadline_seconds,
                str(out),
            )
        )
    print(
        f"designed {len(jobs)} structures; laying out on {workers} workers", flush=True
    )
    started = time.monotonic()
    outcomes = []
    with Pool(workers) as pool:
        for done, record in enumerate(pool.imap_unordered(_worker, jobs), 1):
            outcomes.append(record)
            path = out / "records" / f"{record['index']:06d}.json"
            path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            if done % 50 == 0 or done == len(jobs):
                accepted = sum(1 for r in outcomes if r["status"] == "accepted")
                print(
                    f"  {done}/{len(jobs)} accepted={accepted} "
                    f"({time.monotonic() - started:.0f}s)",
                    flush=True,
                )
    accepted = [r for r in outcomes if r["status"] == "accepted"]
    if design and hashlib.sha256(design.read_bytes()).hexdigest() != design_hash:
        raise ValueError("Chemical design changed during corpus generation")
    manifest = {
        "design_file": str(design.resolve()) if design else None,
        "design_sha256": design_hash,
        "design_file_unchanged": hashlib.sha256(design.read_bytes()).hexdigest()
        == design_hash
        if design
        else True,
        "source_sha256": sources,
        "sources_unchanged": all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in sources.items()
        ),
        "seed": seed,
        "training_admitted": False,
        "quality_scope": "Engine geometry only; final rendered-input review is required.",
        "strata": list(strata),
        "designed": len(jobs),
        "accepted": len(accepted),
        "failed": len(outcomes) - len(accepted),
        "by_stratum": dict(Counter(r["stratum"] for r in accepted)),
        "failures": Counter(
            r.get("reason", "?") for r in outcomes if r["status"] != "accepted"
        ).most_common(8),
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
    parser.add_argument(
        "--strata", default="head_to_tail_patent,head_to_tail_patent_large"
    )
    parser.add_argument("--workers", type=int, default=max(1, available_cpus() - 1))
    parser.add_argument("--deadline-seconds", type=int, default=100)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    strata = tuple(s for s in args.strata.split(",") if s)
    unknown = set(strata) - set(STRATA)
    if unknown:
        raise SystemExit(f"unknown strata: {sorted(unknown)}")
    manifest = build_corpus(
        out=args.out,
        seed=args.seed,
        per_stratum=args.per_stratum,
        strata=strata,
        workers=args.workers,
        deadline_seconds=args.deadline_seconds,
        design=args.design,
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "record_files_sha256"}, indent=1
        )
    )


if __name__ == "__main__":
    main()

"""Render coordinate-bearing engine corpora at verified model input sizes."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from molsight.cpu import available_cpus, initialize_worker
from peptidecore import from_helm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .artifacts import digest
from .quality import chemical_identity
from .review import STYLES, _render


def source_file(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact leaves the declared corpus directory")
    if digest(path) != expected:
        raise ValueError(f"Corpus artifact changed: {relative}")
    return path


def render_record(job):
    source, relative, expected, output, sizes = job
    root = Path(source)
    path = source_file(root, "records/" + relative, expected)
    record = json.loads(path.read_text())
    result = {
        "index": record["index"],
        "stratum": record["stratum"],
        "source_record_sha256": expected,
        "training_release_approved": False,
    }
    if record["status"] != "accepted":
        return {**result, "status": "layout_failed", "reason": record.get("reason")}
    mol_path = source_file(root, "mol/" + record["mol"], record["mol_sha256"])
    source_file(root, "svg/" + record["svg"], record["svg_sha256"])
    molecule = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    if molecule is None or chemical_identity(molecule) != record["smiles"]:
        raise ValueError(f"Coordinate structure mismatch: {relative}")
    if from_helm(record["helm"]).smiles != record["smiles"]:
        raise ValueError(f"HELM structure mismatch: {relative}")
    molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)
    if chemical_identity(molecule) != record["smiles"]:
        raise ValueError(f"Drawing preparation changed structure: {relative}")
    directory = Path(output) / "images" / record["record_key"]
    directory.mkdir(parents=True, exist_ok=False)
    Chem.MolToMolFile(molecule, str(directory / "drawing.mol"), includeStereo=True)
    images = [_render(molecule, directory / name, name, sizes) for name in STYLES]
    return {
        **result,
        "id": record["record_key"],
        "helm": record["helm"],
        "smiles": record["smiles"],
        "status": "rendered_for_review",
        "images": images,
    }


def render(corpus, output, *, sizes, workers):
    if workers < 1 or not sizes or min(sizes) < 1:
        raise ValueError("Workers and input dimensions must be positive")
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    manifest_path = corpus / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("sources_unchanged"):
        raise ValueError("Corpus generation lacks unchanged-source evidence")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        str(path.resolve()): digest(path) for path in Path(__file__).parent.glob("*.py")
    }
    jobs = [
        (str(corpus), name, expected, str(output), sizes)
        for name, expected in sorted(manifest["record_files_sha256"].items())
    ]
    if len(jobs) != manifest["designed"]:
        raise ValueError("Corpus manifest does not cover its design count")
    started, counts = time.monotonic(), Counter()
    with (
        ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_worker, initargs=(0,)
        ) as executor,
        (output / "records.jsonl").open("x") as stream,
    ):
        for row in executor.map(render_record, jobs, chunksize=1):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            counts[row["status"]] += 1
            for style in row.get("images", []):
                for image in style["inputs"]:
                    counts["input_images"] += 1
                    counts[
                        "quality_pass"
                        if image["input_quality_pass"]
                        else "quality_rejected"
                    ] += 1
            if (
                sum(counts[key] for key in ("layout_failed", "rendered_for_review"))
                % 25
                == 0
            ):
                print(
                    json.dumps(
                        {"counts": dict(counts), "seconds": time.monotonic() - started}
                    ),
                    flush=True,
                )
    report = {
        "corpus_manifest_sha256": manifest_hash,
        "counts": dict(counts),
        "sizes": sizes,
        "workers": workers,
        "seconds": time.monotonic() - started,
        "source_sha256": sources,
        "sources_unchanged": all(
            digest(path) == value for path, value in sources.items()
        ),
        "input_manifest_unchanged": digest(manifest_path) == manifest_hash,
        "records_sha256": digest(output / "records.jsonl"),
        "training_release_approved": False,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    print(
        json.dumps(
            render(args.corpus, args.out, sizes=args.sizes, workers=args.workers)
        )
    )


if __name__ == "__main__":
    main()

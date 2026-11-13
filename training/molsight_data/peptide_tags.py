"""Find connected peptide-backbone candidates without rewriting source labels."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import pysqlite3 as sqlite3
from molsight.cpu import available_cpus
from molsight.structure import parse_structure
from rdkit import Chem

from .index import file_hash

_ROUTES = tuple(
    Chem.MolFromSmarts("[N]-" + "[C;!$(C=O)]-" * length + "[C](=[O])")
    for length in (1, 2, 3)
)


def backbone_tags(molecule, *, minimum_nodes=3):
    routes = set()
    for query in _ROUTES:
        for match in molecule.GetSubstructMatches(query, maxMatches=4096):
            routes.add((match[0], match[-2]))
    nitrogen_nodes = {nitrogen for nitrogen, _ in routes}
    edges = set()
    for nitrogen, carbonyl in routes:
        for neighbour in molecule.GetAtomWithIdx(carbonyl).GetNeighbors():
            target = neighbour.GetIdx()
            if target in nitrogen_nodes and target != nitrogen:
                edges.add((nitrogen, target, carbonyl))
    neighbours = {node: set() for node in nitrogen_nodes}
    for first, second, _ in edges:
        neighbours[first].add(second)
        neighbours[second].add(first)
    remaining, components = set(nitrogen_nodes), []
    while remaining:
        component, stack = set(), [min(remaining)]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbours[node] - component)
        remaining.difference_update(component)
        components.append(component)
    selected = max(components, key=len, default=set())
    if len(selected) < minimum_nodes:
        return None
    bonds = [
        molecule.GetBondBetweenAtoms(carbonyl, target)
        for nitrogen, target, carbonyl in edges
        if nitrogen in selected and target in selected
    ]
    return {
        "peptide_candidate": True,
        "cyclic_peptide_candidate": any(bond.IsInRing() for bond in bonds),
        "backbone_nitrogen_nodes": len(selected),
        "inter_residue_amide_bonds": len(bonds),
        "cyclic_amide_bonds": sum(bond.IsInRing() for bond in bonds),
        "specified_stereocenters": len(
            Chem.FindMolChiralCenters(molecule, includeUnassigned=False)
        ),
    }


@lru_cache(maxsize=8192)
def analyze(label):
    if label.count("N") < 3 or label.count("O") < 3:
        return None
    return backbone_tags(parse_structure(label).molecule)


def analyze_batch(rows):
    candidates, errors = [], []
    for target, resolved in rows:
        try:
            features = analyze(resolved)
            if features is not None:
                candidates.append((target, json.dumps(features, separators=(",", ":"))))
        except (ValueError, RuntimeError) as error:
            errors.append(
                {"target": target, "error": f"{type(error).__name__}:{error}"}
            )
    return len(rows), candidates, errors


def scan(labels, output, *, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(Path(labels).resolve()): file_hash(labels)}
    modules = ("molsight.structure", "molsight.cpu", "training.molsight_data.index")
    code_paths = [
        Path(__file__),
        *(Path(__import__(name, fromlist=[""]).__file__) for name in modules),
    ]
    code = {str(path.resolve()): file_hash(path) for path in code_paths}
    counts = Counter()
    started = time.monotonic()
    with (
        sqlite3.connect(
            Path(labels).resolve().as_uri() + "?mode=ro", uri=True
        ) as source,
        sqlite3.connect(output / "peptides.sqlite") as destination,
        ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as executor,
        (output / "errors.jsonl").open("x") as error_stream,
    ):
        destination.execute(
            "CREATE TABLE candidates(target TEXT PRIMARY KEY, features TEXT NOT NULL) WITHOUT ROWID"
        )
        cursor = source.execute(
            "SELECT target,resolved_target FROM labels WHERE error IS NULL ORDER BY target"
        )
        pending = []

        def collect(future):
            total, candidates, errors = future.result()
            destination.executemany("INSERT INTO candidates VALUES (?,?)", candidates)
            counts["labels"] += total
            counts["peptide_candidates"] += len(candidates)
            counts["cyclic_peptide_candidates"] += sum(
                json.loads(features)["cyclic_peptide_candidate"]
                for _, features in candidates
            )
            counts["errors"] += len(errors)
            for error in errors:
                error_stream.write(json.dumps(error) + "\n")
            if counts["labels"] % 16384 == 0:
                print(
                    json.dumps(
                        {
                            "counts": dict(counts),
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

        while rows := cursor.fetchmany(128):
            pending.append(executor.submit(analyze_batch, rows))
            if len(pending) >= 2 * workers:
                collect(pending.pop(0))
        for future in pending:
            collect(future)
        destination.commit()
    report = {
        "counts": dict(counts),
        "inputs": inputs,
        "code_sha256": code,
        "workers": workers,
        "seconds": time.monotonic() - started,
        "scope": "Backbone-based discovery candidates, not unique monomer assignments or a source-annotation review.",
        "input_files_unchanged": all(
            file_hash(path) == digest for path, digest in inputs.items()
        ),
        "sources_unchanged": all(
            file_hash(path) == digest for path, digest in code.items()
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    print(json.dumps(scan(args.labels, args.out, workers=args.workers)))


if __name__ == "__main__":
    main()

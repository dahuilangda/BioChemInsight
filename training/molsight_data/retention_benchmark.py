"""Select a source- and size-balanced complete-molecule retention cohort."""

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from molsight.prepared import PreparedDataset
from molsight.structure import parse_structure

from .index import file_hash
from .chemical_quality import review_reasons


def prepare(manifest, tags, output, *, limit=2000):
    if limit < 1:
        raise ValueError("Retention cohort size must be positive")
    dataset = PreparedDataset(manifest, "validation")
    flags = dataset.metadata(tags)
    columns = [
        "record_key",
        "component",
        "source",
        "category",
        "SMILES",
        "file_path",
        "image_sha256",
        "model_input_preprocessed",
        "input_size",
    ]
    rows = dataset.table.select(columns).to_pylist()
    groups = defaultdict(deque)
    for index in sorted(range(len(rows)), key=lambda value: rows[value]["record_key"]):
        row = rows[index]
        if row["category"] != "complete" or flags[index].get("peptide_candidate"):
            continue
        molecule = parse_structure(row["SMILES"]).molecule
        if review_reasons(molecule):
            continue
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            continue
        count = molecule.GetNumAtoms()
        size = (
            "1_20"
            if count <= 20
            else "21_50"
            if count <= 50
            else "51_100"
            if count <= 100
            else "101_plus"
        )
        row["atom_count"], row["atom_bin"] = count, size
        groups[(size, row["source"])].append(row)
    selected, components = [], set()
    while len(selected) < limit:
        progress = False
        for key in sorted(groups):
            queue = groups[key]
            while queue and queue[0]["component"] in components:
                queue.popleft()
            if not queue:
                continue
            row = queue.popleft()
            components.add(row["component"])
            selected.append(row)
            progress = True
            if len(selected) == limit:
                break
        if not progress:
            break
    if len(selected) != limit:
        raise ValueError(
            "Insufficient independent complete-molecule validation components"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pairs = output / "pairs.json"
    pairs.write_text(json.dumps(selected, indent=2) + "\n")
    report = {
        "records": len(selected),
        "components": len(components),
        "manifest_sha256": file_hash(manifest),
        "tags_sha256": file_hash(tags),
        "pairs_sha256": file_hash(pairs),
        "code_sha256": file_hash(__file__),
        "sources": dict(Counter(row["source"] for row in selected)),
        "atom_bins": dict(Counter(row["atom_bin"] for row in selected)),
        "scope": "Balanced validation diagnostics for paired checkpoint retention; historical pretraining overlap is unknown.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.tags, args.out, limit=args.limit)))


if __name__ == "__main__":
    main()

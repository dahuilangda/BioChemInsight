"""Merge verified peptide catalogs with topology-independent composition groups."""

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import peptidecore
from peptidecore import MonomerLibrary, core_library, from_biln, from_helm
from training.molsight_data.index import source_key

from .artifacts import digest
from .design import composition_group


def merge(reports, output, *, libraries=()):
    source_paths = [
        Path(__file__),
        Path(composition_group.__code__.co_filename),
        Path(source_key.__code__.co_filename),
    ]
    package = Path(peptidecore.__file__).parent
    source_paths += list(package.rglob("*.py")) + list(package.rglob("*.json"))
    sources = {str(path.resolve()): digest(path) for path in source_paths}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registry = {core_library().digest: core_library()}
    inputs = {}
    for path in map(Path, libraries):
        library = MonomerLibrary.from_json(path.read_text())
        registry[library.digest] = library
        inputs[str(path.resolve())] = digest(path)

    @lru_cache(maxsize=8192)
    def chemical_record(helm, biln, library_digest):
        if library_digest not in registry:
            raise ValueError("Peptide catalog requires an undeclared monomer library")
        library = registry[library_digest]
        record = from_helm(helm, library=library)
        if from_biln(biln, library=library).smiles != record.smiles:
            raise ValueError("Peptide notations describe different structures")
        return record.smiles, composition_group(record.graph, library)

    counts, seen, structures = Counter(), set(), set()
    parents = []
    with (output / "records.jsonl").open("x") as destination:
        for path in map(Path, reports):
            report = json.loads(path.read_text())
            if report.get("sources_unchanged") is not True:
                raise ValueError("Peptide catalog has unverified source provenance")
            records_path = Path(report["records_file"])
            if digest(records_path) != report["records_sha256"]:
                raise ValueError("Peptide catalog records changed after export")
            for item in (path, records_path):
                inputs[str(item.resolve())] = digest(item)
            count = 0
            for line in records_path.read_text().splitlines():
                row = json.loads(line)
                if row["category"] != "peptide" or row["rejection_reasons"]:
                    raise ValueError(
                        "Peptide catalog contains a rejected or unrelated record"
                    )
                key = source_key(row)
                if key in seen:
                    raise ValueError("Peptide catalog source records overlap")
                seen.add(key)
                label, group = chemical_record(
                    row["helm"], row["biln"], row["library_digest"]
                )
                if label != row["SMILES"]:
                    raise ValueError(
                        "Peptide notation differs from its recognition target"
                    )
                structures.add(label)
                row["composition_group"] = group
                row["composition_policy"] = "canonical_monomer_template_multiset_v2"
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts["images"] += 1
                counts["resolution:" + str(row["input_size"])] += 1
                counts["stratum:" + row["peptide_stratum"]] += 1
                count += 1
            if count != report["counts"]["images"]:
                raise ValueError("Peptide catalog count differs from its report")
            parents.append({"report": str(path.resolve()), "images": count})
    if not all(digest(path) == value for path, value in inputs.items()):
        raise ValueError("Peptide catalogs changed during consolidation")
    if not all(digest(path) == value for path, value in sources.items()):
        raise ValueError("Peptide consolidation sources changed during execution")
    report = {
        "inputs": inputs,
        "parents": parents,
        "counts": dict(counts),
        "chemical_records": len(structures),
        "records_file": str((output / "records.jsonl").resolve()),
        "records_sha256": digest(output / "records.jsonl"),
        "composition_policy": "canonical_monomer_template_multiset_v2",
        "library_digests": sorted(registry),
        "code_sha256": sources,
        "sources_unchanged": True,
        "input_files_unchanged": True,
        "training_release_approved": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"images": counts["images"], "chemical_records": len(structures)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--libraries", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.reports, args.out, libraries=args.libraries)))


if __name__ == "__main__":
    main()

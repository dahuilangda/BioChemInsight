from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from peptidecore import (
    Chain,
    Connection,
    Endpoint,
    PeptideGraph,
    assemble,
    core_library,
    from_biln,
    from_helm,
    to_biln,
    to_helm,
)
from rdkit import Chem, rdBase
from peptidecore.library import identity

STRATA = (
    "linear_short",
    "linear_medium",
    "linear_long",
    "linear_modified",
    "head_to_tail_small",
    "head_to_tail_medium",
    "head_to_tail_large",
    "head_to_tail_modified",
    "disulfide",
    "lactam_lariat",
    "branched",
    "multiple_closures",
    "head_to_tail_patent",
    "head_to_tail_patent_large",
    "head_to_tail_aromatic",
)
STANDARD = tuple("ACDEFGHIKLMNPQRSTVWY")
MODIFIED = ("dA", "dF", "dL", "dP", "meA", "meL", "meF")
PATENT_MODIFIED = (
    "meA",
    "meG",
    "meL",
    "meF",
    "dA",
    "dF",
    "dL",
    "dP",
    "dW",
    "Nle",
    "Hyp",
    "Pip",
    "Cha",
    "Chg",
    "Oic_3aR-7aS",
    "Ac3c",
    "Aib",
    "Thz",
    "Tic",
    "D-Phe_4ureido",
    "Phe_4Cl",
    "Phe_4Br",
    "Trp_Me",
    "Trp_Ome",
    "Trp_5OH",
    "D-Trp_2Me",
    "3Pal",
    "D-2Pal",
    "D-3Pal",
    "1Nal",
    "2Nal",
    "D-2Nal",
    "D-1Nal",
    "Abu",
    "Apm",
)
AROMATIC = (
    "F",
    "W",
    "Y",
    "H",
    "dF",
    "dW",
    "3Pal",
    "D-2Pal",
    "D-3Pal",
    "Trp_Me",
    "Trp_Ome",
    "Trp_5OH",
    "D-Trp_2Me",
    "1Nal",
    "2Nal",
)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=4096)
def _monomer_identity(monomer):
    return identity(monomer.template())


def composition_group(graph, library):
    composition = Counter(
        _monomer_identity(library.get(symbol, chain.polymer_type))
        for chain in graph.chains
        for symbol in chain.monomers
    )
    return digest(sorted(composition.items()))


def _connection(start, end, start_port, end_port, end_chain="PEPTIDE1"):
    return Connection(
        Endpoint("PEPTIDE1", start, start_port), Endpoint(end_chain, end, end_port)
    )


def _graph(stratum, ordinal, rng):
    ranges = {
        "linear_short": (2, 5),
        "linear_medium": (6, 12),
        "linear_long": (13, 30),
        "linear_modified": (4, 12),
        "head_to_tail_small": (3, 5),
        "head_to_tail_medium": (6, 10),
        "head_to_tail_large": (11, 20),
        "head_to_tail_modified": (4, 12),
        "disulfide": (6, 14),
        "lactam_lariat": (6, 14),
        "branched": (5, 12),
        "multiple_closures": (8, 16),
        "head_to_tail_patent": (16, 26),
        "head_to_tail_patent_large": (26, 40),
        "head_to_tail_aromatic": (8, 18),
    }
    low, high = ranges[stratum]
    length = (
        low + ordinal % (high - low + 1) if high - low < 8 else rng.randint(low, high)
    )
    symbols = rng.choices(STANDARD, k=length)
    edges = []
    extra_chains = []
    if stratum.startswith("head_to_tail_patent"):
        positions = rng.sample(range(length), min(length, 3 + ordinal % 6))
        for position in positions:
            symbols[position] = rng.choice(PATENT_MODIFIED)
    if stratum == "head_to_tail_aromatic":
        aromatic_positions = rng.sample(range(length), min(length, 4 + ordinal % 5))
        for position in aromatic_positions:
            symbols[position] = rng.choice(AROMATIC)
        methylated_positions = rng.sample(range(length), 2 + ordinal % 3)
        for position in methylated_positions:
            symbols[position] = (
                "meF"
                if position in aromatic_positions
                else rng.choice(("meA", "meG", "meL"))
            )
        if ordinal % 2 == 0:
            available = [
                position
                for position in aromatic_positions
                if position not in methylated_positions
            ]
            if not available:
                available = [
                    position
                    for position in range(length)
                    if position not in methylated_positions
                ]
            symbols[rng.choice(available)] = rng.choice(("Phe_4Cl", "Phe_4Br"))
    if stratum.endswith("modified"):
        positions = rng.sample(range(length), min(2, length))
        for position in positions:
            symbols[position] = rng.choice(MODIFIED)
    if stratum.startswith("head_to_tail"):
        edges.append(_connection(length, 1, "R2", "R1"))
    if stratum == "disulfide":
        symbols[0] = symbols[-1] = "C"
        edges.append(_connection(1, length, "R3", "R3"))
    if stratum == "lactam_lariat":
        if ordinal % 2:
            symbols[-1] = "D"
            edges.append(_connection(length, 1, "R3", "R1"))
        else:
            symbols[1], symbols[-2] = "K", "E"
            edges.append(_connection(2, length - 1, "R3", "R3"))
    if stratum == "branched":
        branch_position = length // 2
        symbols[branch_position - 1] = "K"
        branch = tuple(rng.choices(STANDARD, k=2 + ordinal % 3))
        extra_chains.append(Chain("PEPTIDE2", branch))
        edges.append(_connection(branch_position, len(branch), "R3", "R2", "PEPTIDE2"))
    if stratum == "multiple_closures":
        symbols[0] = symbols[-1] = "C"
        edges.append(_connection(1, length, "R3", "R3"))
        if ordinal % 2:
            symbols[2] = symbols[-3] = "C"
            edges.append(_connection(3, length - 2, "R3", "R3"))
        else:
            edges.append(_connection(length, 1, "R2", "R1"))
    if stratum.startswith("linear") and ordinal % 2:
        symbols = ["ac", *symbols, "am"]
    return PeptideGraph(
        (Chain("PEPTIDE1", tuple(symbols)), *extra_chains), tuple(edges)
    )


def build_design(*, seed=20261108, per_stratum=8, strata=STRATA):
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if not strata or len(set(strata)) != len(strata) or set(strata) - set(STRATA):
        raise ValueError("Invalid or duplicate design strata")
    library = core_library()
    for symbol in (*STANDARD, *MODIFIED, *PATENT_MODIFIED, *AROMATIC, "ac", "am"):
        library.get(symbol)
    rng = random.Random(seed)
    records, seen = [], set()
    for stratum in strata:
        ordinal, attempts = 0, 0
        while ordinal < per_stratum:
            attempts += 1
            if attempts > per_stratum * 100:
                raise RuntimeError(f"Unable to fill unique structures for {stratum}")
            graph = _graph(stratum, ordinal, rng)
            helm = to_helm(graph)
            record = assemble(graph, library=library, source=helm, source_format="HELM")
            if record.smiles in seen:
                continue
            biln = to_biln(graph)
            if from_helm(helm, library=library).smiles != record.smiles:
                raise RuntimeError("HELM round-trip changed chemical identity")
            if from_biln(biln, library=library).smiles != record.smiles:
                raise RuntimeError("BILN round-trip changed chemical identity")
            molecule = record.molecule
            if len(Chem.GetMolFrags(molecule)) != 1:
                raise RuntimeError("Designed peptide is disconnected")
            records.append(
                {
                    "id": digest(record.smiles),
                    "stratum": stratum,
                    "ordinal": ordinal,
                    "split": "review",
                    "composition_group": composition_group(graph, library),
                    "provenance": {
                        "kind": "synthetic",
                        "seed": seed,
                        "attempt": attempts,
                    },
                    "helm": helm,
                    "biln": biln,
                    "chemical_record": record.as_dict(),
                    "atom_count": molecule.GetNumAtoms(),
                    "specified_tetrahedral_centers": sum(
                        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                        for atom in molecule.GetAtoms()
                    ),
                    "explicit_connection_count": len(graph.connections),
                    "training_release_approved": False,
                }
            )
            seen.add(record.smiles)
            ordinal += 1
    return {
        "schema_version": 1,
        "composition_policy": "canonical_monomer_template_multiset_v2",
        "generator": "training.peptide_data.design",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rdkit_version": rdBase.rdkitVersion,
        "library": {
            "name": library.name,
            "digest": library.digest,
            "source_url": library.source_url,
            "source_revision": library.source_revision,
        },
        "seed": seed,
        "per_stratum": per_stratum,
        "counts": dict(Counter(record["stratum"] for record in records)),
        "training_release_approved": False,
        "records": records,
    }


def _design_stratum(task):
    seed, per_stratum, stratum = task
    stream_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{stratum}".encode()).digest()[:8], "big"
    )
    return build_design(seed=stream_seed, per_stratum=per_stratum, strata=(stratum,))


def build_design_parallel(*, seed=20261108, per_stratum=8, workers=1):
    if workers < 1:
        raise ValueError("Worker count must be positive")
    tasks = [(seed, per_stratum, stratum) for stratum in STRATA]
    if workers == 1:
        parts = list(map(_design_stratum, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(STRATA))) as executor:
            parts = list(executor.map(_design_stratum, tasks))
    result = parts[0]
    result["seed"] = seed
    result["seed_policy"] = "sha256(seed:stratum), first eight bytes, big endian"
    result["records"] = [record for part in parts for record in part["records"]]
    if len({record["id"] for record in result["records"]}) != len(result["records"]):
        raise RuntimeError("Design strata contain overlapping chemical identities")
    result["counts"] = dict(Counter(record["stratum"] for record in result["records"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--per-stratum", type=int, default=8)
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    parser.add_argument("--workers", type=int, default=min(available, len(STRATA)))
    args = parser.parse_args()
    result = build_design_parallel(
        seed=args.seed, per_stratum=args.per_stratum, workers=args.workers
    )
    result["library_snapshot"] = json.loads(core_library().to_json())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"records": len(result["records"]), "counts": result["counts"]}))


if __name__ == "__main__":
    main()

"""Design distinct peptide contexts around chemically compatible core monomers."""

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
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

from .design import AROMATIC, STANDARD, composition_group, digest
from .engine_corpus import source_hashes


def port_roles(monomer):
    molecule = monomer.template()
    roles = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbour = atom.GetNeighbors()[0]
        label = f"R{atom.GetAtomMapNum()}"
        if neighbour.GetAtomicNum() == 7:
            roles[label] = "amine"
        elif neighbour.GetAtomicNum() == 6 and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(neighbour).GetAtomicNum() == 8
            for bond in neighbour.GetBonds()
        ):
            roles[label] = "carbonyl"
    return roles


@lru_cache(maxsize=4096)
def classification(monomer):
    roles = port_roles(monomer)
    if roles.get("R1") == "amine" and roles.get("R2") == "carbonyl":
        molecule = monomer.template()
        ports = {
            atom.GetAtomMapNum(): atom.GetNeighbors()[0].GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() == 0
        }
        distance = len(Chem.GetShortestPath(molecule, ports[1], ports[2])) - 1
        return "internal", {2: "alpha", 3: "beta", 4: "gamma"}.get(distance, "extended")
    if roles.get("R1") == "amine":
        return "c_terminal", "terminal"
    if roles.get("R2") == "carbonyl":
        return "n_terminal", "terminal"
    return None


@lru_cache(maxsize=1)
def internal_symbols():
    return tuple(
        m.symbol
        for m in core_library().monomers.values()
        if m.monomer_type == "Backbone"
        and classification(m) is not None
        and classification(m)[0] == "internal"
    )


def make_record(symbol, ordinal, seed):
    library = core_library()
    role, backbone = classification(library.get(symbol))
    stream = int.from_bytes(
        hashlib.sha256(f"{seed}:{symbol}:{ordinal}".encode()).digest()[:8], "big"
    )
    rng = random.Random(stream)
    mode = ordinal % 4 if role == "internal" else 0
    cyclic = mode in (1, 2)
    topology = {
        0: "linear",
        1: "cyclic",
        2: "bicyclic",
        3: "branched" if ordinal % 8 == 3 else "lariat",
    }[mode]
    length = rng.randint(8, 18) if mode else rng.randint(4, 20)
    symbols = rng.choices(STANDARD, k=length)
    internal = internal_symbols()
    for position in rng.sample(range(length), min(length, 2 + ordinal % 3)):
        symbols[position] = rng.choice(
            AROMATIC if cyclic and ordinal % 4 == 1 else internal
        )
    position = (
        rng.randrange(length)
        if role == "internal"
        else 0
        if role == "n_terminal"
        else length - 1
    )
    symbols[position] = symbol
    edges = (
        [Connection(Endpoint("PEPTIDE1", length, "R2"), Endpoint("PEPTIDE1", 1, "R1"))]
        if cyclic
        else []
    )
    extra_chains = []
    available = [index for index in range(length) if index != position]
    if mode == 2:
        first = rng.choice(available)
        second = rng.choice(
            [
                index
                for index in available
                if min(abs(index - first), length - abs(index - first)) >= 3
            ]
        )
        symbols[first] = symbols[second] = "C"
        edges.append(
            Connection(
                Endpoint("PEPTIDE1", first + 1, "R3"),
                Endpoint("PEPTIDE1", second + 1, "R3"),
            )
        )
    elif mode == 3:
        first = rng.choice(available)
        symbols[first] = "K"
        if topology == "branched":
            branch = tuple(rng.choices(STANDARD, k=rng.randint(2, 6)))
            extra_chains.append(Chain("PEPTIDE2", branch))
            edges.append(
                Connection(
                    Endpoint("PEPTIDE1", first + 1, "R3"),
                    Endpoint("PEPTIDE2", len(branch), "R2"),
                )
            )
        else:
            second = rng.choice(
                [index for index in available if abs(index - first) >= 3]
            )
            symbols[second] = "D"
            edges.append(
                Connection(
                    Endpoint("PEPTIDE1", first + 1, "R3"),
                    Endpoint("PEPTIDE1", second + 1, "R3"),
                )
            )
    graph = PeptideGraph(
        (Chain("PEPTIDE1", tuple(symbols)), *extra_chains), tuple(edges)
    )
    helm = to_helm(graph)
    record = assemble(graph, library=library, source=helm, source_format="HELM")
    biln = to_biln(graph)
    if (
        from_helm(helm, library=library).smiles != record.smiles
        or from_biln(biln, library=library).smiles != record.smiles
    ):
        raise ValueError("Coverage design failed notation round-trip")
    if len(Chem.GetMolFrags(record.molecule)) != 1:
        raise ValueError("Coverage design is disconnected")
    if len(record.smiles) + 2 > 2048:
        return None
    return {
        "id": digest(record.smiles),
        "stratum": f"coverage_{topology}_{backbone}",
        "ordinal": ordinal,
        "focus_monomer": symbol,
        "split": "review",
        "helm": helm,
        "biln": biln,
        "composition_group": composition_group(graph, library),
        "chemical_record": record.as_dict(),
        "provenance": {"kind": "synthetic", "seed": seed, "context_seed": stream},
        "training_release_approved": False,
    }


def design_focus(task):
    symbol, count, seed = task
    rows, seen, ordinal = [], set(), 0
    while len(rows) < count:
        row = make_record(symbol, ordinal, seed)
        ordinal += 1
        if ordinal > count * 20:
            raise ValueError(f"Unable to fill chemical contexts for {symbol}")
        if row is None or row["id"] in seen:
            continue
        rows.append(row)
        seen.add(row["id"])
    return rows


def build(output, *, contexts=40, seed=20261112, workers=8):
    if min(contexts, workers) < 1:
        raise ValueError("Context count and workers must be positive")
    library = core_library()
    sources = source_hashes()
    monomers = sorted(
        [
            m
            for m in library.monomers.values()
            if m.polymer_type == "PEPTIDE" and m.monomer_type == "Backbone"
        ],
        key=lambda m: m.symbol,
    )
    eligible = [m.symbol for m in monomers if classification(m) is not None]
    quarantine = [
        {
            "symbol": m.symbol,
            "reason": "no_compatible_amine_or_carbonyl_attachment",
            "definition": m.as_helm_dict(),
        }
        for m in monomers
        if classification(m) is None
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_sha256": sources,
        "contexts_per_focus": contexts,
        "eligible_monomers": eligible,
        "quarantined_monomers": quarantine,
        "seed": seed,
        "library": {
            "name": library.name,
            "digest": library.digest,
            "source_url": library.source_url,
            "source_revision": library.source_revision,
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rdkit_version": rdBase.rdkitVersion,
        "training_release_approved": False,
    }
    (output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = [(symbol, contexts, seed) for symbol in eligible]
    records, seen, counts = [], set(), Counter()
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        (output / "records.jsonl").open("x") as stream,
    ):
        for part in executor.map(design_focus, tasks):
            for row in part:
                symbol = row["focus_monomer"]
                if row["id"] in seen:
                    continue
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                records.append(row)
                seen.add(row["id"])
                counts[symbol] += 1
            stream.flush()
            print(
                json.dumps(
                    {"focus_monomers": len(counts), "chemical_records": len(records)}
                ),
                flush=True,
            )
        for symbol in eligible:
            ordinal = contexts
            while counts[symbol] < contexts:
                row = make_record(symbol, ordinal, seed)
                ordinal += 1
                if row is None or row["id"] in seen:
                    continue
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                records.append(row)
                seen.add(row["id"])
                counts[symbol] += 1
    if any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != value
        for path, value in sources.items()
    ):
        raise ValueError("Coverage generator changed during design")
    result = {
        **manifest,
        "sources_unchanged": True,
        "schema_version": 1,
        "composition_policy": "canonical_monomer_template_multiset_v2",
        "records": records,
        "counts": dict(Counter(row["stratum"] for row in records)),
        "focus_counts": dict(counts),
    }
    (output / "design.json").write_text(
        json.dumps(result, separators=(",", ":")) + "\n"
    )
    return {
        "chemical_records": len(records),
        "covered_monomers": len(counts),
        "quarantined_monomers": len(quarantine),
        "counts": result["counts"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.out, contexts=args.contexts, workers=args.workers, seed=args.seed
            )
        )
    )


if __name__ == "__main__":
    main()

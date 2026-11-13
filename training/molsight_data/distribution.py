"""Audit selected recognition data, peptide coverage, and source provenance."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pysqlite3 as sqlite3
from peptidecore import core_library
from peptidecore.helm import parse_helm
from rdkit import Chem

from .index import file_hash


def quantiles(values):
    return (
        dict(
            zip(
                ("minimum", "p10", "median", "p90", "maximum"),
                np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            )
        )
        if values
        else {}
    )


def audit(index, labels, assignments, peptides, output):
    inputs = {
        str(Path(path).resolve()): file_hash(path)
        for path in (index, labels, assignments, peptides)
    }
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        for name, path in (
            ("idx", index),
            ("lab", labels),
            ("spl", assignments),
            ("pep", peptides),
        ):
            connection.execute(
                f"ATTACH DATABASE ? AS {name}",
                (Path(path).resolve().as_uri() + "?mode=ro",),
            )
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("""CREATE TEMP TABLE selected AS
            SELECT a.split,r.source,CASE WHEN r.category='peptide' THEN 'peptide' ELSE l.category END category,
            COALESCE(json_extract(r.payload,'$.image_domain'),'unknown') image_domain,
            a.component,l.structure_family,r.publication_id,l.atom_count,l.token_length,
            json_extract(r.payload,'$.peptide_id') peptide_id,json_extract(r.payload,'$.peptide_stratum') stratum,
            json_extract(r.payload,'$.helm') helm,r.composition_group,p.features,
            CASE WHEN json_extract(r.payload,'$.model_input_preprocessed')=1 THEN json_extract(r.payload,'$.input_size') ELSE 512 END input_size
            FROM idx.records r JOIN lab.labels l ON l.target=r.target JOIN spl.assignments a ON a.record_id=r.id
            LEFT JOIN pep.candidates p ON p.target=r.target WHERE a.selected=1 AND l.error IS NULL""")

        def query(text):
            return [dict(row) for row in connection.execute(text)]

        counts = query(
            "SELECT split,category,COUNT(*) images,COUNT(DISTINCT structure_family) families_without_stereo,COUNT(DISTINCT component) components FROM selected GROUP BY split,category"
        )
        sources = query(
            "SELECT source,image_domain,category,COUNT(*) images,COUNT(DISTINCT structure_family) families_without_stereo FROM selected WHERE split='train' AND category IN ('complete','peptide') GROUP BY source,image_domain,category"
        )
        candidates = query("""SELECT split,source,image_domain,
            json_extract(features,'$.cyclic_peptide_candidate') cyclic,COUNT(*) images,
            COUNT(DISTINCT structure_family) families_without_stereo,
            COUNT(DISTINCT publication_id) known_publications,
            SUM(publication_id IS NULL) images_without_publication_id,
            SUM(json_extract(features,'$.backbone_nitrogen_nodes')>=10) backbone_at_least_10,
            SUM(json_extract(features,'$.backbone_nitrogen_nodes')>=10 AND atom_count>=100) large_backbone_at_least_10
            FROM selected WHERE category='complete' AND features IS NOT NULL GROUP BY split,source,image_domain,cyclic""")
        synthetic = query(
            "SELECT split,stratum,COUNT(*) images,COUNT(DISTINCT peptide_id) chemical_records,COUNT(DISTINCT composition_group) compositions FROM selected WHERE category='peptide' GROUP BY split,stratum"
        )
        resolutions = query(
            "SELECT split,category,input_size,COUNT(*) images FROM selected WHERE category IN ('complete','peptide') GROUP BY split,category,input_size"
        )
        pool = query(
            "SELECT split,peptide_id,stratum,helm,MIN(atom_count) atom_count,MIN(token_length) token_length FROM selected WHERE category='peptide' GROUP BY split,peptide_id,stratum,helm"
        )
        real = query(
            "SELECT atom_count,token_length,features FROM selected WHERE split='train' AND category='complete' AND image_domain='real' AND features IS NOT NULL"
        )
    library = core_library()
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    occurrences, observed, lengths, modifications = (
        Counter(),
        set(),
        defaultdict(list),
        Counter(),
    )
    backbone_classes = Counter()
    for row in pool:
        graph = parse_helm(row["helm"])
        monomers = [
            library.get(symbol, chain.polymer_type)
            for chain in graph.chains
            for symbol in chain.monomers
        ]
        residues = [
            monomer for monomer in monomers if monomer.monomer_type == "Backbone"
        ]
        lengths[row["split"]].append(len(residues))
        if row["split"] != "train":
            continue
        symbols = [monomer.symbol for monomer in residues]
        occurrences.update(symbols)
        observed.update(symbols)
        modifications["chemical_records"] += 1
        modifications["with_noncanonical_residues"] += any(
            symbol not in standard for symbol in symbols
        )
        has_n_methyl = False
        for monomer in residues:
            molecule = monomer.template()
            ports = {
                atom.GetAtomMapNum(): atom.GetNeighbors()[0].GetIdx()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() == 0
            }
            if 1 not in ports or 2 not in ports:
                backbone_classes["other"] += 1
                continue
            nitrogen = molecule.GetAtomWithIdx(ports[1])
            distance = len(Chem.GetShortestPath(molecule, ports[1], ports[2])) - 1
            backbone_classes[
                f"{nitrogen.GetSymbol()}_to_carbonyl_{distance}_bonds"
            ] += 1
            has_n_methyl |= nitrogen.GetAtomicNum() == 7 and any(
                atom.GetAtomicNum() == 6
                and atom.GetDegree() == 1
                and atom.GetTotalNumHs() == 3
                for atom in nitrogen.GetNeighbors()
            )
        modifications["with_backbone_n_methyl"] += has_n_methyl
    all_backbone = {
        monomer.symbol
        for monomer in library.monomers.values()
        if monomer.polymer_type == "PEPTIDE" and monomer.monomer_type == "Backbone"
    }
    report = {
        "inputs": inputs,
        "code_sha256": file_hash(__file__),
        "selected": counts,
        "eligible_training_sources": sources,
        "discovered_candidates": candidates,
        "synthetic_strata": synthetic,
        "resolutions": resolutions,
        "synthetic_residue_lengths_by_split": {
            split: quantiles(values) for split, values in lengths.items()
        },
        "synthetic_training_coverage": {
            "chemical_records": modifications["chemical_records"],
            "observed_backbone_symbols": len(observed),
            "observed_canonical20_symbols": sorted(observed & standard),
            "missing_canonical20_symbols": sorted(standard - observed),
            "available_core_backbone_symbols": len(all_backbone),
            "available_core_monomers": len(library.monomers),
            "missing_backbone_symbols": sorted(all_backbone - observed),
            "residue_occurrences": dict(occurrences),
            "modifications": dict(modifications),
            "backbone_port_distances": dict(backbone_classes),
        },
        "real_candidate_training_profiles": {
            name: quantiles([json.loads(row["features"])[name] for row in real])
            for name in ("backbone_nitrogen_nodes", "specified_stereocenters")
        },
        "scope": "Selected data counts, not recognition accuracy. Candidate tags are not reviewed peptide ground truth. Structure-family counts ignore stereochemistry. Monomer coverage describes the explicit synthetic training corpus, not everything seen during historical pretraining.",
        "input_files_unchanged": all(
            file_hash(path) == value for path, value in inputs.items()
        ),
    }
    if not report["input_files_unchanged"]:
        raise ValueError("Distribution inputs changed during analysis")
    with Path(output).open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("index", "labels", "assignments", "peptides", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.index, args.labels, args.assignments, args.peptides, args.out)
    print(
        json.dumps(
            {
                "synthetic_training_coverage": {
                    key: value
                    for key, value in result["synthetic_training_coverage"].items()
                    if key not in ("missing_backbone_symbols", "residue_occurrences")
                },
                "real_candidate_training_profiles": result[
                    "real_candidate_training_profiles"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()

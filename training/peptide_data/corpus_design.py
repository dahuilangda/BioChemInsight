"""Recover notation and composition metadata from a verified engine corpus."""

import argparse
import json
from pathlib import Path

from peptidecore import core_library, from_helm, to_biln

from .artifacts import digest
from .design import composition_group
from .refine_corpus import verified_file


def prepare(corpus, output, *, source_name="peptide_patent_profiles"):
    corpus = Path(corpus).resolve()
    manifest_path = corpus / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("sources_unchanged") or manifest["failed"]:
        raise ValueError("Corpus has rejected layouts or unverified generation sources")
    library, rows, identities = core_library(), [], set()
    for name, expected in sorted(manifest["record_files_sha256"].items()):
        path = verified_file(corpus, "records/" + name, expected)
        source = json.loads(path.read_text())
        record = from_helm(source["helm"], library=library)
        if source["status"] != "accepted" or record.smiles != source["smiles"]:
            raise ValueError("Corpus structure differs from its HELM record")
        if record.smiles in identities:
            raise ValueError("Corpus contains duplicate chemical identities")
        identities.add(record.smiles)
        rows.append(
            {
                "id": source["record_key"],
                "source_name": source_name,
                "stratum": source["stratum"],
                "helm": source["helm"],
                "biln": to_biln(record.graph),
                "composition_group": composition_group(record.graph, library),
                "chemical_record": record.as_dict(),
                "source_record_sha256": expected,
                "split": "review",
                "training_release_approved": False,
            }
        )
    if len(rows) != manifest["designed"] or digest(manifest_path) != manifest_hash:
        raise ValueError("Corpus coverage or manifest changed during conversion")
    result = {
        "schema_version": 1,
        "source_manifest_sha256": manifest_hash,
        "composition_policy": "canonical_monomer_template_multiset_v2",
        "library": rows[0]["chemical_record"]["library"],
        "records": rows,
        "training_release_approved": False,
    }
    with Path(output).open("x") as stream:
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    return {"records": len(rows), "source_manifest_sha256": manifest_hash}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-name", default="peptide_patent_profiles")
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.out, source_name=args.source_name)))


if __name__ == "__main__":
    main()

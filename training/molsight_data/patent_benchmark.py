"""Bind reviewed patent figures to recognition diagnostics and source evidence."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from molsight.structure import parse_structure
from skills.patent_gold_review import validate_reference

from .index import file_hash
from .peptide_tags import backbone_tags


def prepare(references, crops, pdfs, training_roots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    documents = json.loads(Path(references).read_text())["rows"]
    historical = {}
    inputs = {str(Path(references).resolve()): file_hash(references)}
    for root in map(Path, training_roots):
        path = root / "uspto_mol/train_680k.csv"
        with path.open() as stream:
            historical[str(root.resolve())] = [
                row["file_path"] for row in csv.DictReader(stream)
            ]
        inputs[str(path.resolve())] = file_hash(path)
    pairs = []
    for key, record in sorted(documents.items()):
        review = record["review"]
        path = (Path(crops) / review["crop"]).resolve()
        if not path.is_relative_to(Path(crops).resolve()):
            raise ValueError("Patent crop leaves the declared source directory")
        valid, reason = validate_reference(record, str(path), key)
        if not valid:
            raise ValueError(f"Unverified source figure {key}: {reason}")
        publication = Path(review["crop"]).parts[0]
        pdf = Path(pdfs) / (publication + ".pdf")
        if file_hash(pdf) != review["source_pdf_sha256"]:
            raise ValueError("Patent PDF differs from its reviewed source")
        inputs[str(pdf.resolve())] = file_hash(pdf)
        inputs[str(path)] = file_hash(path)
        document = parse_structure(record["smiles"])
        overlap = {
            root: {
                "same_figure_path_rows": sum(
                    value.endswith(review["crop"]) for value in values
                ),
                "same_publication_rows": sum(publication in value for value in values),
            }
            for root, values in historical.items()
        }
        pairs.append(
            {
                "name": f"{publication}_SEQ{key}",
                "sequence_id": int(key),
                "file_path": str(path),
                "SMILES": document.model_text,
                "image_sha256": review["crop_sha256"],
                "category": "reviewed_patent_cyclic_peptide",
                "model_input_preprocessed": False,
                "publication": publication,
                "source_pdf": str(pdf.resolve()),
                "source_pdf_sha256": review["source_pdf_sha256"],
                "pdf_page": review["pdf_page"],
                "closure": record["closure"],
                "source_residue_count": len(record["codes"]),
                "backbone_features": backbone_tags(document.molecule),
                "historical_overlap": overlap,
                "evaluation_role": "source_reviewed_diagnostic",
                "historical_family_independence": "not_established",
            }
        )
    pair_path = output / "pairs.json"
    pair_path.write_text(json.dumps(pairs, indent=2) + "\n")
    report = {
        "records": len(pairs),
        "closures": dict(Counter(row["closure"] for row in pairs)),
        "inputs": inputs,
        "pairs_sha256": file_hash(pair_path),
        "evaluation_role": "Source-reviewed diagnostics; historical family overlap excludes a generalization claim.",
        "input_files_unchanged": all(
            file_hash(path) == digest for path, digest in inputs.items()
        ),
        "code_sha256": {str(Path(__file__).resolve()): file_hash(__file__)},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("references", "crops", "pdfs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-roots", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.references, args.crops, args.pdfs, args.training_roots, args.out
            )
        )
    )


if __name__ == "__main__":
    main()

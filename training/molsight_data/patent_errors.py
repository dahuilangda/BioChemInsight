"""Separate connectivity, stereochemistry, and parsing failures in recognition output."""

import argparse
import json
from collections import Counter
from pathlib import Path

from molsight.abbreviations import resolve_abbreviations
from molsight.structure import StructureError, parse_structure

from .index import file_hash
from .peptide_tags import backbone_tags


def diagnose(gold_text, prediction):
    gold = resolve_abbreviations(parse_structure(gold_text)).document
    result = {"gold_backbone": backbone_tags(gold.molecule)}
    try:
        predicted = resolve_abbreviations(parse_structure(prediction)).document
    except StructureError as error:
        return {**result, "outcome": "unparsed_prediction", "error": str(error)}
    result["predicted_backbone"] = backbone_tags(predicted.molecule)
    if gold.equivalent(predicted):
        outcome = "exact"
    elif gold.without_stereo(atoms=True).equivalent(
        predicted.without_stereo(atoms=True)
    ):
        outcome = "stereochemistry"
    else:
        outcome = "connectivity_or_annotation"
    return {**result, "outcome": outcome}


def audit(predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_hash = file_hash(predictions)
    counts = Counter()
    with (
        Path(predictions).open() as source,
        (output / "errors.jsonl").open("x") as dest,
    ):
        for line in source:
            row = json.loads(line)
            result = diagnose(row["gold"], row["prediction"])
            counts[result["outcome"]] += 1
            dest.write(json.dumps({**row, "diagnosis": result}) + "\n")
    if file_hash(predictions) != source_hash:
        raise ValueError("Predictions changed during analysis")
    report = {
        "predictions_sha256": source_hash,
        "counts": dict(counts),
        "records": sum(counts.values()),
        "scope": "Diagnostic error categories; backbone counts do not establish sequence identity.",
        "code_sha256": file_hash(__file__),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.predictions, args.out)))


if __name__ == "__main__":
    main()

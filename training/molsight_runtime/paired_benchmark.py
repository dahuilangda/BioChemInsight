"""Compare recognition changes on identical, independent evaluation components."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def summarize(rows):
    count = len(rows)
    gained = sum(not old and new for old, new in rows)
    lost = sum(old and not new for old, new in rows)
    discordant = gained + lost
    probability = min(
        1.0,
        2
        * sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
        / 2**discordant,
    )
    draws = np.random.default_rng(42).multinomial(
        count, [lost / count, (count - discordant) / count, gained / count], size=10000
    )
    interval = np.quantile((draws[:, 2] - draws[:, 0]) / count, [0.025, 0.975]).tolist()
    return {
        "records": count,
        "before_correct": sum(old for old, _ in rows),
        "after_correct": sum(new for _, new in rows),
        "gained": gained,
        "lost": lost,
        "accuracy_change": (gained - lost) / count,
        "paired_bootstrap_95_interval": interval,
        "mcnemar_exact_two_sided_p": probability,
    }


def compare(before, after, pairs):
    pairs = Path(pairs)
    digest = hashlib.sha256(pairs.read_bytes()).hexdigest()
    gold = json.loads(pairs.read_text())
    if not gold or len({row["component"] for row in gold}) != len(gold):
        raise ValueError(
            "Paired evaluation requires one image per independent component"
        )
    predictions, provenance = [], []
    for directory in (Path(before), Path(after)):
        report = json.loads((directory / "report.json").read_text())
        if report["pairs_sha256"] != digest or report["examples"] != len(gold):
            raise ValueError("Benchmark provenance or coverage differs")
        if report["precision"] != "fp32":
            raise ValueError("Paired capability evaluation requires FP32 predictions")
        rows = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text().splitlines()
        ]
        if len(rows) != len(gold):
            raise ValueError("Incomplete benchmark predictions")
        for index, (row, target) in enumerate(zip(rows, gold, strict=True)):
            if (
                row["index"] != index
                or row["record_key"] != target["record_key"]
                or row["gold"] != target["SMILES"]
                or type(row["resolved_exact"]) is not bool
            ):
                raise ValueError("Paired prediction identity differs")
        predictions.append(rows)
        provenance.append(
            {
                "directory": str(directory.resolve()),
                "report_sha256": hashlib.sha256(
                    (directory / "report.json").read_bytes()
                ).hexdigest(),
                "predictions_sha256": hashlib.sha256(
                    (directory / "predictions.jsonl").read_bytes()
                ).hexdigest(),
            }
        )
    results = {}
    for field in (None, "source", "atom_bin"):
        groups = (
            {"all": list(range(len(gold)))}
            if field is None
            else {
                value: [index for index, row in enumerate(gold) if row[field] == value]
                for value in sorted({row[field] for row in gold})
            }
        )
        for value, indices in groups.items():
            name = "all" if field is None else f"{field}:{value}"
            results[name] = summarize(
                [
                    (
                        predictions[0][index]["resolved_exact"],
                        predictions[1][index]["resolved_exact"],
                    )
                    for index in indices
                ]
            )
    return {
        "pairs_sha256": digest,
        "benchmarks": provenance,
        "results": results,
        "scope": "Paired validation diagnostics; subgroup intervals and tests are exploratory. Historical pretraining overlap is unknown.",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("before", "after", "pairs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after, args.pairs)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

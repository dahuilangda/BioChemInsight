"""Generate a cyclic peptide layout from a SMILES string."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rdkit import Chem

from .api import depict, render_svg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=350)
    parser.add_argument("--time-budget", type=float)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        result = depict(
            args.smiles, seed=args.seed, maxiter=args.maxiter, time_budget=args.time_budget
        )
    except ValueError as error:
        (args.out / "report.json").write_text(
            json.dumps({"qa_pass": False, "error": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 2
    (args.out / "report.json").write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    if not result.accepted:
        return 2
    Chem.MolToMolFile(result.molecule, str(args.out / "structure.mol"))
    (args.out / "structure.svg").write_text(render_svg(result), encoding="utf-8")
    return 0

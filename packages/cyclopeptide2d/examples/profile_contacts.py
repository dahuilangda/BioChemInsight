"""Compare spatial and exhaustive contact calculations and layout times."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from rdkit import Chem, rdBase

from cyclopeptide2d.solver import PoseProblem, peptide_frames, residue_pose_layout


def exhaustive(self, xy, aa_margin=0.0, ab_margin=0.0):
    return np.arange(len(self.ai)), np.arange(len(self.ab_a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references", type=Path, default=Path(__file__).with_name("reviewed_references.json")
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.budget <= 0:
        parser.error("repeats and budget must be positive")
    original = PoseProblem.contact_candidates
    result = {
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "seed": args.seed,
        "budget_seconds": args.budget,
        "cases": [],
    }
    for key, reference in json.loads(args.references.read_text()).items():
        mol = Chem.MolFromSmiles(reference["smiles"])
        frames, order = peptide_frames(mol)
        problem = PoseProblem(mol, frames, order, seed=args.seed)
        x = problem.initial.copy()
        row = {"id": key, "atoms": mol.GetNumAtoms(), "timings": {}}
        outputs = {}
        for name, fn in [("spatial", original), ("exhaustive", exhaustive)]:
            try:
                PoseProblem.contact_candidates = fn
                outputs[name] = problem.objective(x, 40.0, 40.0)
                samples = []
                for _ in range(5):
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        problem.objective(x, 40.0, 40.0)
                    samples.append((time.perf_counter() - start) / args.repeats)
                row["timings"][name] = {
                    "median_seconds": float(np.median(samples)),
                    "samples": samples,
                }
                if args.layout:
                    start = time.monotonic()
                    _, report = residue_pose_layout(
                        mol, seed=args.seed, deadline=start + args.budget
                    )
                    row[name + "_layout"] = {
                        "seconds": time.monotonic() - start,
                        "qa_pass": report["qa_pass"],
                        "defect_rank": report["defect_rank"],
                        "constraints": report["constraints"],
                    }
            finally:
                PoseProblem.contact_candidates = original
        e, g = outputs["spatial"]
        full_e, full_g = outputs["exhaustive"]
        row["energy_error"] = abs(e - full_e)
        row["max_gradient_error"] = float(np.max(np.abs(g - full_g)))
        if not np.isclose(e, full_e, rtol=1e-13, atol=1e-10) or not np.allclose(
            g, full_g, rtol=1e-12, atol=1e-9
        ):
            raise AssertionError(f"spatial/exhaustive mismatch: {key}")
        result["cases"].append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

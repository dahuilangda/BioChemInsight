"""Runaway-bond / chord-in-ring audit across the gold set + hard cases.

The hires detail audit (commit 88d3da1) found a closure bond slicing
through a drawn piperidine: atom-pair distance gates and bond-crossing
counts both pass while the segment crosses the ring's interior.  This
scan runs beam_layout_coords over every seed and reports, per case and
seed, the new ring-intrusion detector (chords through drawn small rings
+ atoms drawn strictly inside them), so construction-layer violations
are found before they reach a render.

Usage:
  python scan_ring_intrusions.py [--seeds 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

import beam_sidechain_layout as bsl

RDLogger.DisableLog("rdApp.*")

PAIRS_6FIG = "/data/molsight_realpatent_sft/eval/pairs_6fig.json"


def hard_case_smiles(name: str) -> str:
    """Gold SMILES of a six-figure pair (round-4 chronic laggards)."""
    rows = json.load(open(PAIRS_6FIG, encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("rows", [])
    for row in rows:
        if row.get("name") == name:
            return row["gold"]
    raise KeyError(name)


def scan_cases(seeds: int) -> dict:
    rows: dict[str, dict] = {}
    cases: list[tuple[str, str]] = [(case, smi) for case, smi, _crop
                                    in bsl.beam_validation_cases()]
    for name in ("fig_s52_0", "fig_s77_0"):
        cases.append((name, hard_case_smiles(name)))
    for case, smi in cases:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows[case] = {"error": "SMILES parse failed"}
            continue
        per_seed = []
        for seed in range(seeds):
            laid, rep = bsl.beam_layout_coords(mol, seed=seed)
            m = rep.get("metrics", {})
            per_seed.append({
                "seed": seed,
                "ok": laid is not None,
                "error": rep.get("error"),
                "qa": rep.get("qa_pass"),
                "cross": m.get("bond_crossings"),
                "close": m.get("n_close_pairs"),
                "intr": m.get("n_ring_intrusions"),
                "intr_detail": m.get("ring_intrusions"),
            })
        rows[case] = {"n_atoms": mol.GetNumAtoms(),
                      "qa_seeds": sum(1 for r in per_seed if r["qa"]),
                      "violation_seeds": sum(1 for r in per_seed
                                             if r["ok"] and (r["intr"] or 0) > 0),
                      "per_seed": per_seed}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    rows = scan_cases(args.seeds)
    for case, r in rows.items():
        if "error" in r and "per_seed" not in r:
            print(f"{case:22s} ERROR {r['error']}")
            continue
        marks = []
        for s in r["per_seed"]:
            if not s["ok"]:
                marks.append(f"s{s['seed']}:FAIL({s['error']})")
            elif (s["intr"] or 0) > 0:
                kinds = sorted({d['kind'] for d in s['intr_detail']})
                marks.append(f"s{s['seed']}:INTR{','.join(kinds)}")
            elif s["qa"]:
                marks.append(f"s{s['seed']}:qa")
            else:
                marks.append(f"s{s['seed']}:pass"
                             f"(close={s['close']},cross={s['cross']})")
        print(f"{case:22s} atoms={r['n_atoms']:3d} "
              f"qa={r['qa_seeds']}/{len(r['per_seed'])} "
              f"intrusions={r['violation_seeds']}  " + "  ".join(marks))


if __name__ == "__main__":
    main()

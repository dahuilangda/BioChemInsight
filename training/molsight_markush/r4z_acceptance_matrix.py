"""Acceptance matrix for the round-4z (zig-zag) SFT run.

Reads the eval JSON/log artifacts and prints the full matrix against the
round-3 baselines:
  6fig bare decode      r3 0.9116  — target: > 0.9116 (seq52 > 0.90)
  val26                 r3 0.9020  — target: no regression
  reg20                 r3 0.9405  — target: no regression
  WildMol identity 384  r3 baseline measured in this round — <= 0.005 drop
  markush200            r3 0.9501  — no regression
  linear150             r3 0.6974  — no regression (observed)

Usage: python r4z_acceptance_matrix.py [--ev /data/molsight_realpatent_sft_r4z/eval]
"""
from __future__ import annotations

import argparse
import json
import os
import re

BASE = {
    "6fig": 0.9116,
    "val26": 0.9020,
    "reg20": 0.9405,
    "markush200": 0.9501,
    "linear150": 0.6974,
    "wildmol": None,     # measured in this round from r3_wildmol.json/log
}


def mean_from_json(path):
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh).get("mean_tanimoto")
    return None


def identity_from_log(path):
    if os.path.isfile(path):
        with open(path) as fh:
            m = re.search(r"identity=(\d+)/384=([0-9.]+)", fh.read())
            if m:
                return float(m.group(2))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev", default="/data/molsight_realpatent_sft_r4z/eval")
    args = ap.parse_args()
    ev = args.ev

    wildmol_r3 = identity_from_log(os.path.join(ev, "r3_wildmol.log"))
    wildmol_r4z = identity_from_log(os.path.join(ev, "r4z_wildmol.log"))
    base = dict(BASE)
    if wildmol_r3 is not None:
        base["wildmol"] = wildmol_r3

    rows = []
    for name in ("6fig", "val26", "reg20", "markush200", "linear150"):
        got = mean_from_json(os.path.join(ev, f"r4z_{name}.json"))
        b = base[name]
        delta = None if (got is None or b is None) else got - b
        rows.append((name, b, got, delta))
    rows.append(("wildmol", base["wildmol"], wildmol_r4z,
                 None if (wildmol_r4z is None or base["wildmol"] is None)
                 else wildmol_r4z - base["wildmol"]))

    print(f"{'set':<12}{'r3 base':>10}{'r4z':>10}{'delta':>10}  gate")
    for name, b, got, delta in rows:
        if name == "6fig":
            ok = delta is not None and delta > 0
            gate = "PASS (>r3)" if ok else ("PENDING" if got is None else "FAIL")
        elif name == "wildmol":
            ok = delta is not None and delta >= -0.005
            gate = "PASS (drop<=0.005)" if ok else (
                "PENDING" if got is None else "FAIL")
        else:
            ok = delta is not None and delta >= -1e-9
            gate = "PASS (no-regress)" if ok else (
                "PENDING" if got is None else "FAIL")
        bs = f"{b:.4f}" if b is not None else "-"
        gs = f"{got:.4f}" if got is not None else "-"
        ds = f"{delta:+.4f}" if delta is not None else "-"
        print(f"{name:<12}{bs:>10}{gs:>10}{ds:>10}  {gate}")

    # seq52 per-figure detail from the 6fig json
    p = os.path.join(ev, "r4z_6fig.json")
    if os.path.isfile(p):
        with open(p) as fh:
            res = json.load(fh)["results"]
        print("\n6fig per-figure:")
        for r in sorted(res, key=lambda r: r["tanimoto"]):
            flag = " (seq52 target >0.90)" if "s52" in r["name"] else ""
            print(f"  {r['name']:<14}{r['tanimoto']:.4f}{flag}")


if __name__ == "__main__":
    main()

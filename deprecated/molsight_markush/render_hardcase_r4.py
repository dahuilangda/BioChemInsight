"""Hard-case multiple augmentation renders for the round-4 SFT mix.

Rounds the diagnosis-driven flywheel at the two chronic six-figure laggards:
seq52 (fig_s52_0, stuck at 0.82 since round 1) and seq77 (fig_s77_0 /
fig_s77_1, 0.85-0.92).  Each gold SMILES from the six-figure pair list is
rendered 12 times with the ALIGNED style profile (commit 59ba6dd), each
variant perturbing one axis of the measured real-crop envelope:

  * scale      — fixedBondLength 22/24/26 px around the aligned 24
  * rotation   — uniform random orientation (labels kept upright)
  * ink noise  — ink_level 0.82/0.90 x sensor sigma 1.0/2.5 around the
                 aligned 0.86 / 1.5
  * tight-crop — margin uniform 4-18 px around the aligned 6-14
  * depiction  — atom-shuffle seed per variant, so CoordGen produces a
                 different drawing of the same molecule

Every render then goes through the render_stage11 finish() patent-scan
post-chain (clean/scan/photocopy regimes) and the ink-density reject, the
same path as the round-2/3 aligned pools.  Rows that survive are written as
a train_680k.csv (file_path,SMILES) plus a manifest.

Usage:
  python render_hardcase_r4.py --pairs /data/molsight_realpatent_sft/eval/pairs_6fig.json \
      --out /data/molsight_realpatent_sft_r4/hardcase_aug --variants 12 --workers 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

# render_stage11.py parses sys.argv[1] at import time — hide our CLI from it
_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish  # ALIGNED draw + patent-scan post-chain
finally:
    sys.argv = _argv
from style_calibration import draw_molecule, ALIGNED

RDLogger.DisableLog("rdApp.*")

# named hard cases: the pair names whose gold SMILES the rounds 1-3 model
# chronically mis-decodes (fig_s77_0 / fig_s77_1 carry the same gold — the
# molecule is drawn twice in the patent, so both crops track it)
CASES = {
    "fig_s52_0": "seq52",
    "fig_s77_1": "seq77",
    "fig_s77_0": "seq77_0",
}

BOND_LENGTHS = [22.0, 24.0, 26.0]
INK_LEVELS = [0.82, 0.90]
NOISE_SIGMAS = [1.0, 2.5]
MARGIN_RANGE = (4, 18)


def coordgen_seeded(mol: Chem.Mol, seed: int):
    """coordgen_coords with an explicit depiction seed: atom order shuffle
    drives the CoordGen layout, so distinct seeds give distinct drawings of
    one molecule (style_calibration.coordgen_coords pins a molecule-derived
    seed — fine for pool diversity, too static for 12 renders of one
    molecule)."""
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    shuffled = Chem.RenumberAtoms(mol, order)
    try:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(shuffled, clearConfs=True)
        rdDepictor.SetPreferCoordGen(False)
    except Exception:
        rdDepictor.SetPreferCoordGen(False)
        return None
    return shuffled


def variant_profile(i: int) -> dict:
    """ALIGNED profile perturbed along the augmentation grid (variant i of
    the 3x2x2 product; rotation/margin stay random through the rng)."""
    bond_len, ink, sigma = list(itertools.product(
        BOND_LENGTHS, INK_LEVELS, NOISE_SIGMAS))[i % 12]
    prof = dict(ALIGNED)
    prof["fixed_bond_length"] = bond_len
    prof["ink_level"] = ink
    prof["noise_sigma"] = sigma
    prof["tight_crop_pad"] = MARGIN_RANGE
    return prof


def _render_one(job):
    idx, smiles, case, v, out_dir, seed = job
    try:
        np.random.seed((seed * 1000003 + idx * 31 + time.time_ns()) % (2**32))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return idx, None
        shuffled = coordgen_seeded(mol, seed * 977 + v)
        if shuffled is None:
            return idx, None
        rng = random.Random(seed * 1000003 + idx)
        img = draw_molecule(shuffled, variant_profile(v), rng)
        path = os.path.join(out_dir, "images", f"{case}_v{v:02d}_{idx:04d}.png")
        if not finish(img, rng, path):
            return idx, None  # ink density over the measured envelope
        return idx, path
    except Exception:
        return idx, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="/data/molsight_realpatent_sft/eval/pairs_6fig.json")
    ap.add_argument("--out", default="/data/molsight_realpatent_sft_r4/hardcase_aug")
    ap.add_argument("--variants", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = json.load(open(args.pairs))
    golds = {}
    for item in pairs:
        case = CASES.get(item["name"])
        if case and case not in golds:
            mol = Chem.MolFromSmiles(item["gold"])
            assert mol is not None and mol.GetNumAtoms() > 20, item["name"]
            golds[case] = item["gold"]
    assert set(golds) == set(CASES.values()), golds.keys()
    print(f"hard cases: { {k: len(v) for k, v in golds.items()} }", flush=True)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    jobs, meta = [], []
    idx = 0
    for case, smi in sorted(golds.items()):
        for v in range(args.variants):
            jobs.append((idx, smi, case, v, args.out, args.seed))
            meta.append({"idx": idx, "case": case, "variant": v})
            idx += 1

    rows = []
    t0 = time.time()
    with Pool(processes=args.workers, maxtasksperchild=8) as mp:
        asyncs = [(j[0], mp.apply_async(_render_one, (j,))) for j in jobs]
        for m, (jidx, a) in zip(meta, asyncs):
            try:
                _, path = a.get(timeout=300)
            except Exception:
                path = None
            if path:
                rows.append({"file_path": path, "case": m["case"],
                             "variant": m["variant"], "SMILES": golds[m["case"]]})
            else:
                print(f"  rejected: {m['case']} v{m['variant']}", flush=True)

    csv_path = os.path.join(args.out, "train_680k.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")

    manifest = {
        "purpose": "round-4 hard-case augmentation (seq52/seq77/seq77_0)",
        "profile": "ALIGNED perturbed: bond 22/24/26 px, ink 0.82/0.90, "
                   "sigma 1.0/2.5, margin 4-18 px, per-variant depiction seed",
        "pipeline": "render_stage11.py finish() via render_hardcase_r4.py",
        "pairs_source": os.path.abspath(args.pairs),
        "seed": args.seed,
        "cases": sorted(golds),
        "variants_per_case": args.variants,
        "rows_rendered": len(rows),
        "rows_rejected": len(jobs) - len(rows),
        "csv": csv_path,
        "images_dir": os.path.join(args.out, "images"),
    }
    with open(os.path.join(args.out, "render_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

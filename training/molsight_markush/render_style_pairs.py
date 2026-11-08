"""Render the verified real-crop gold SMILES with the calibrated synthetic
profiles, producing molecule-paired (real crop, synthetic render) datasets
for the style-alignment study.

For every verified row (verified is True, gold SMILES present) of each
input manifest, the same molecule is laid out with CoordGen (atom-shuffled,
as in render_stage11.py) and drawn with one style profile. The output
directory receives

  images/          the synthetic renders
  pairs.json       [{real, synth, smiles, patent, crop}] pairing rows

Usage:
  python render_style_pairs.py --profile baseline|aligned \
      --out /tmp/style_pairs_baseline \
      [--manifests m1.json m2.json ...] [--limit N] [--workers 8]

Manifest crop paths are resolved relative to the manifest's own directory
(the repo real_crops manifest and the /tmp/crops_* manifests all follow
this layout).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from style_calibration import ALIGNED, BASELINE, coordgen_coords, draw_molecule

RDLogger.DisableLog("rdApp.*")

PROFILES = {"baseline": BASELINE, "aligned": ALIGNED}
RENDER_TIMEOUT_S = 60


def verified_rows(manifest_path: str, limit: int | None) -> list[dict]:
    data = json.load(open(manifest_path))
    base = os.path.dirname(os.path.abspath(manifest_path))
    rows = [r for r in data["rows"] if r.get("verified") is True
            and r.get("smiles")]
    out = []
    for r in rows:
        path = os.path.join(base, r["crop"])
        if os.path.exists(path):
            out.append({"real": path, "smiles": r["smiles"],
                        "patent": r.get("patent", ""), "crop": r["crop"]})
    if limit:
        out = out[:limit]
    return out


def _render_one(job):
    idx, smiles, profile_name, out_dir, seed = job
    profile = PROFILES[profile_name]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, None
        shuffled = coordgen_coords(mol)
        if shuffled is None:
            return idx, None
        rng = random.Random(seed * 1000003 + idx)
        img = draw_molecule(shuffled, profile, rng)
        path = os.path.join(out_dir, "images", f"{idx:06d}.png")
        img.save(path)
        return idx, path
    except Exception:
        return idx, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifests", nargs="+", default=[
        "/data/BioChemInsight/data/real_crops/real_crops_manifest.json",
        "/tmp/crops_kras/real_crops_manifest.json",
        "/tmp/crops_gpc3/real_crops_manifest.json"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per manifest (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rows = []
    for m in args.manifests:
        rows.extend(verified_rows(m, args.limit))
    print(f"{len(rows)} verified real crops paired by gold SMILES", flush=True)

    jobs = [(i, r["smiles"], args.profile, args.out, 7)
            for i, r in enumerate(rows)]
    results = {}
    with Pool(processes=args.workers, maxtasksperchild=32) as pool:
        asyncs = [(j[0], pool.apply_async(_render_one, (j,))) for j in jobs]
        for idx, a in asyncs:
            try:
                _, path = a.get(timeout=RENDER_TIMEOUT_S)
            except Exception:
                path = None
            results[idx] = path

    pairs = []
    for i, r in enumerate(rows):
        synth = results.get(i)
        if synth:
            pairs.append({"real": r["real"], "synth": synth,
                          "smiles": r["smiles"], "patent": r["patent"],
                          "crop": r["crop"]})
    with open(os.path.join(args.out, "pairs.json"), "w") as fh:
        json.dump(pairs, fh, indent=1)
    print(f"wrote {len(pairs)}/{len(rows)} pairs -> "
          f"{os.path.join(args.out, 'pairs.json')}")


if __name__ == "__main__":
    main()

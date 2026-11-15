"""Render a large aligned-style synthetic SFT set from an existing
synthetic SMILES pool (round-2 real-domain SFT).

Re-renders SMILES sampled from the round-1 synthetic replay pool with the
style-aligned stage-11 pipeline (commit 59ba6dd):
  coordgen_coords()  — atom-shuffled CoordGen layout (render_stage11.py)
  draw()             — MolDraw2D with style_calibration.ALIGNED profile
                       (serif labels, fixed bond length / font, print-gray
                       ink + faint noise, tight crop to ink bbox)
  finish()           — patent-scan post-chain (illumination, blur, sensor
                       noise, JPEG), regimes matched to the measured real
                       crop envelope

The ALIGNED style passed the MolSight blind decode test (KS p > 0.29:
decodes of aligned renders are statistically indistinguishable from real
verified crops), so training pairs rendered here are style-matched to the
real domain while carrying exact gold SMILES.

Usage:
  python render_aligned_pool.py --pool /data/molsight_markush_v13/uspto_mol/train_680k.csv \
      --out /data/molsight_realpatent_sft_r2/aligned_synth \
      --smiles 2500 --variants 2 --target-rows 5000 --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

# render_stage11.py parses sys.argv[1] at import time — hide our CLI from it
_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish  # ALIGNED draw + patent-scan post-chain
finally:
    sys.argv = _argv
from style_calibration import coordgen_coords, draw_molecule, ALIGNED

RDLogger.DisableLog("rdApp.*")

RENDER_TIMEOUT_S = 180


def _init_worker():
    # fork() copies the parent numpy RNG state — reseed per worker so
    # finish() noise fields differ across children
    np.random.seed((os.getpid() * 2654435761 + time.time_ns()) % (2**32))


def _render_one(job):
    idx, smiles, out_dir, seed = job
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return idx, None
        shuffled = coordgen_coords(mol)
        if shuffled is None:
            return idx, None
        rng = random.Random(seed * 1000003 + idx)
        img = draw_molecule(shuffled, ALIGNED, rng)
        path = os.path.join(out_dir, "images", f"{idx:06d}.png")
        if not finish(img, rng, path):
            return idx, None  # ink density over the measured envelope
        return idx, path
    except Exception:
        return idx, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="/data/molsight_markush_v13/uspto_mol/train_680k.csv",
                    help="synthetic SMILES pool CSV (file_path,SMILES)")
    ap.add_argument("--out", required=True,
                    help="output root; images/ + train_680k.csv + manifest")
    ap.add_argument("--smiles", type=int, default=2500,
                    help="max distinct pool SMILES to consume")
    ap.add_argument("--variants", type=int, default=2,
                    help="renders per SMILES (independent layout/scan draw)")
    ap.add_argument("--target-rows", type=int, default=5000,
                    help="stop once this many rendered rows exist")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    pool = pd.read_csv(args.pool)
    pool = pool[pool["file_path"].apply(os.path.exists)]
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    print(f"pool: {len(pool)} usable rows, sampling up to {args.smiles} "
          f"SMILES x {args.variants} variants -> target {args.target_rows} rows",
          flush=True)

    jobs, meta = [], []
    idx = 0
    consumed = 0
    for pos in order[:args.smiles]:
        if len(meta) >= args.target_rows:
            break
        smi = pool["SMILES"].iloc[pos]
        for v in range(args.variants):
            jobs.append((idx, smi, args.out, args.seed + v))
            meta.append({"idx": idx, "pool_row": int(pos), "variant": v})
            idx += 1
        consumed += 1

    rows = []
    t0 = time.time()
    with Pool(processes=args.workers, maxtasksperchild=16,
              initializer=_init_worker) as mp:
        asyncs = [(j[0], mp.apply_async(_render_one, (j,))) for j in jobs]
        done = 0
        for m, a in zip(meta, [a for _, a in asyncs]):
            try:
                _, path = a.get(timeout=RENDER_TIMEOUT_S)
            except Exception:
                path = None
            done += 1
            if path:
                rows.append({"file_path": path,
                             "SMILES": pool["SMILES"].iloc[m["pool_row"]],
                             "pool_row": m["pool_row"], "variant": m["variant"]})
            if done % 500 == 0:
                print(f"  {done}/{len(meta)} rendered, {len(rows)} rows kept "
                      f"({time.time() - t0:.0f}s)", flush=True)

    csv_path = os.path.join(args.out, "train_680k.csv")
    with open(csv_path, "w", newline="") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")

    manifest = {
        "profile": "ALIGNED (style_calibration.py, commit 59ba6dd)",
        "pipeline": "render_stage11.py draw()+finish() via render_aligned_pool.py",
        "pool": os.path.abspath(args.pool),
        "seed": args.seed,
        "smiles_consumed": consumed,
        "variants_per_smiles": args.variants,
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

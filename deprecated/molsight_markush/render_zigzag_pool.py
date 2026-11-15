"""Render the round-2/3 synthetic pool with the zig-zag template layout.

Round-4z real-domain SFT: the synthetic corpus switches its spatial
preparation from CoordGen to the patent-style zig-zag macrocycle template
(zigzag_layout.py — hex-lattice ragged-box contours, all-120-degree ring
angles, side chains outward), reusing the SAME SMILES pools the round-2/3
aligned corpus was drawn from:

  round-1 synth replay (1000 SMILES, the absolute-path rows of
  /data/molsight_realpatent_sft/uspto_mol/train_680k.csv) + round-2
  aligned pool (2490 unique SMILES, /data/molsight_realpatent_sft_r2/
  aligned_synth) — their union is the "round-1/2 synthetic pool" — plus
  the round-3 vocabulary pool (467 unique SMILES,
  /data/molsight_realpatent_sft_r3/aligned_synth_vocab, fluoro/quaternary/
  proline motifs).

Style pipeline identical to the aligned corpus (commit 59ba6dd):
  draw_molecule()   — style_calibration.ALIGNED profile (serif labels,
                      fixed bond length / font, print-gray ink, tight crop)
  finish()          — patent-scan post-chain (illumination, blur, sensor
                      noise, JPEG), regimes matched to the measured real
                      envelope

Layout QA: each render retries zig-zag seeds; molecules that never pass
geometry QA fall back to CoordGen so the pool stays at full size (the
fallback share is counted in the manifest).

Usage:
  python render_zigzag_pool.py --out /data/molsight_realpatent_sft_r4z/zigzag_synth \
      --variants 1 --vocab-variants 2 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

# render_stage11.py parses sys.argv[1] at import time — hide our CLI from it
_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from render_stage11 import finish          # ALIGNED draw + post-chain
finally:
    sys.argv = _argv
from style_calibration import ALIGNED, coordgen_coords, draw_molecule
from zigzag_layout import zigzag_coords

RDLogger.DisableLog("rdApp.*")

ROUND1_TRAIN = "/data/molsight_realpatent_sft/uspto_mol/train_680k.csv"
R2_ALIGNED = "/data/molsight_realpatent_sft_r2/aligned_synth/train_680k.csv"
R3_VOCAB = "/data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv"

ZIGZAG_SEED_RETRIES = 3
RENDER_TIMEOUT_S = 300


def _read_smiles(path, absolute=None):
    out = []
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for r in csv.reader(fh):
            if len(r) >= 2:
                if absolute is None or os.path.isabs(r[0]) == absolute:
                    out.append(r[1])
    return out


def _init_worker():
    # fork() copies the parent numpy RNG state — reseed per worker so
    # finish() noise fields differ across children
    np.random.seed((os.getpid() * 2654435761 + time.time_ns()) % (2 ** 32))


def _render_one(job):
    idx, smiles, out_dir, seed = job
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return idx, None, "invalid"
        laid = None
        for k in range(ZIGZAG_SEED_RETRIES):
            laid = zigzag_coords(mol, random.Random(seed + k), seed=seed + k)
            if laid is not None:
                used = "zigzag"
                break
        if laid is None:
            laid = coordgen_coords(mol)      # keep the pool at full size
            used = "coordgen_fallback"
        if laid is None:
            return idx, None, "layout_failed"
        rng = random.Random(seed * 1000003 + idx)
        img = draw_molecule(laid, ALIGNED, rng)
        path = os.path.join(out_dir, "images", f"{idx:06d}.png")
        if not finish(img, rng, path):
            return idx, None, "ink_rejected"   # over measured envelope
        return idx, path, used
    except Exception:
        return idx, None, "exception"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="output root; images/ + train_680k.csv + manifest")
    ap.add_argument("--variants", type=int, default=1,
                    help="renders per round-1/2 pool SMILES")
    ap.add_argument("--vocab-variants", type=int, default=2,
                    help="renders per round-3 vocabulary SMILES")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    # ---- pools: union of round-1 synth replay + round-2 aligned, and the
    #      round-3 vocabulary pool ----
    pool = []
    seen = set()
    for smi in _read_smiles(ROUND1_TRAIN, absolute=True):
        if smi not in seen:
            seen.add(smi)
            pool.append(smi)
    n_r1 = len(pool)
    for smi in _read_smiles(R2_ALIGNED):
        if smi not in seen:
            seen.add(smi)
            pool.append(smi)
    n_r12 = len(pool)
    vocab = []
    for smi in _read_smiles(R3_VOCAB):
        if smi not in seen:
            seen.add(smi)
            vocab.append(smi)
    print(f"pool: round-1 replay {n_r1} + round-2 aligned {n_r12 - n_r1} "
          f"= {n_r12} SMILES; vocab {len(vocab)} SMILES", flush=True)

    jobs, smi_of_job = [], {}
    idx = 0
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    for pos in order:
        for v in range(args.variants):
            smi_of_job[idx] = pool[pos]
            jobs.append((idx, pool[pos], args.out, args.seed + v))
            idx += 1
    for pos in range(len(vocab)):
        for v in range(args.vocab_variants):
            smi_of_job[idx] = vocab[pos]
            jobs.append((idx, vocab[pos], args.out, args.seed + v))
            idx += 1
    print(f"{len(jobs)} render jobs ({args.variants}x pool + "
          f"{args.vocab_variants}x vocab), {args.workers} workers", flush=True)

    rows, layout_used = [], {"zigzag": 0, "coordgen_fallback": 0,
                             "dropped": 0}
    t0 = time.time()
    with Pool(processes=args.workers, maxtasksperchild=8,
              initializer=_init_worker) as mp:
        asyncs = [(j[0], mp.apply_async(_render_one, (j,))) for j in jobs]
        done = 0
        for idx, a in asyncs:
            try:
                _, path, used = a.get(timeout=RENDER_TIMEOUT_S)
            except Exception:
                path, used = None, "exception"
            if path:
                rows.append({"file_path": path, "SMILES": smi_of_job[idx],
                             "idx": idx})
                layout_used[used] += 1
            else:
                layout_used["dropped"] += 0 if used in layout_used else 0
                layout_used[used] = layout_used.get(used, 0) + 1
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)} rendered, {len(rows)} kept "
                      f"({time.time() - t0:.0f}s)", flush=True)

    csv_path = os.path.join(args.out, "train_680k.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("file_path,SMILES\n")
        for r in rows:
            smi = r["SMILES"]
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{r['file_path']},{smi}\n")

    manifest = {
        "profile": "ALIGNED (style_calibration.py, commit 59ba6dd)",
        "pipeline": "render_stage11.py finish() + zigzag_layout."
                    "zigzag_coords() via render_zigzag_pool.py",
        "layout_counts": layout_used,
        "zigzag_seed_retries": ZIGZAG_SEED_RETRIES,
        "pools": {"round1_replay": n_r1, "round2_aligned": n_r12 - n_r1,
                  "vocab": len(vocab)},
        "seed": args.seed,
        "variants_per_smiles": args.variants,
        "vocab_variants": args.vocab_variants,
        "rows_rendered": len(rows),
        "csv": csv_path,
        "images_dir": os.path.join(args.out, "images"),
    }
    with open(os.path.join(args.out, "render_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

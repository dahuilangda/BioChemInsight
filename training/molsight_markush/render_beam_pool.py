"""Render the synthetic pool with the beam side-chain layout engine.

Multi-seed per molecule with automatic conflict-based selection: the
first seed whose render passes the collision gate wins.  Molecules that
fail all seeds are named skips (not silent).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SEEDS = (13, 42, 77)
MAX_RENDER = 5000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data/molsight_markush_v13/uspto_mol/train_680k.csv")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--out-dir", default="/data/molsight_realpatent_sft_r4/images_beam")
    ap.add_argument("--out-csv", default="/data/molsight_realpatent_sft_r4/uspto_mol/train_680k.csv")
    ap.add_argument("--limit", type=int, default=MAX_RENDER)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import beam_sidechain_layout as bsl
    from style_calibration import ALIGNED, draw_molecule

    smiles_pool = []
    with open(args.pool) as fh:
        for row in csv.reader(fh):
            if row and row[0] != "file_path" and len(row) > 1 and row[1]:
                smiles_pool.append(row[1])
    if args.vocab and os.path.exists(args.vocab):
        with open(args.vocab) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[-1]:
                    smiles_pool.append(parts[-1].strip())
    random.shuffle(smiles_pool)
    smiles_pool = smiles_pool[: args.limit]
    print(f"pool: {len(smiles_pool)} SMILES")

    out_rows = []
    skipped = 0
    for i, smi in enumerate(smiles_pool):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        AllChem.Compute2DCoords(mol)
        rendered = False
        for seed in SEEDS:
            try:
                out, rep = bsl.beam_layout_coords(mol, seed=seed)
            except Exception:
                out = None
            if out is None:
                continue
            img_name = f"beam_{i:05d}.png"
            img = draw_molecule(out, ALIGNED, random.Random(seed))
            img.save(os.path.join(args.out_dir, img_name))
            out_rows.append({"file_path": os.path.join(args.out_dir, img_name),
                             "SMILES": smi})
            rendered = True
            break
        if not rendered:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(smiles_pool)}] rendered={len(out_rows)} skipped={skipped}",
                  flush=True)
            with open(args.out_csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["file_path", "SMILES"])
                w.writerows(out_rows)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_path", "SMILES"])
        w.writerows(out_rows)
    print(f"DONE rendered={len(out_rows)} skipped={skipped}")


if __name__ == "__main__":
    main()

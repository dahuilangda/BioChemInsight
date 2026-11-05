"""Assemble the round-2 real-domain SFT mix (aligned-synthesis re-mix).

Round-2 strategy on top of the round-1 winner (markush_v13_realpatent_sft_mix,
6-figure bare-decode mean 0.8988): replace the style-drifted portion of the
synthetic replay with a large aligned-style synthetic set rendered by the
calibrated stage-11 ALIGNED pipeline (style_calibration.py, commit 59ba6dd,
blind decode KS p > 0.29 vs real crops), while keeping the exact round-1
real/replay rows so the only distribution change is the aligned synthesis.

Train mix (per task spec):
  - aligned synthetic renders (>= 3000 rows, from render_aligned_pool.py)
  - real verified crops x3 oversampled (round-1 train split, 229 rows -> 687)
  - round-1 synthetic replay (same 1000 rows, verbatim)

The 26-crop observation val and the 20-crop regression subset are carried
over byte-identical from the round-1 dataset root, so every round-2 metric
is directly comparable to round-1. Val crops are never trained on.

Output root layout (MolSight train.py conventions,
data_path_map['uspto'] = 'uspto_mol/train_680k.csv'):
  <out>/crops/<patent>/<crop>.png
  <out>/uspto_mol/train_680k.csv
  <out>/real/val.csv
  <out>/real/regression20.csv
  <out>/assembly_summary_r2.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

ROUND1_ROOT = "/data/molsight_realpatent_sft"
ALIGNED_CSV_DEFAULT = "/data/molsight_realpatent_sft_r2/aligned_synth/train_680k.csv"
ROUND1_TRAIN_N_REAL = 229
ROUND1_REPLAY_N = 1000


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        assert header.strip() == "file_path,SMILES", header
        import csv
        for r in csv.reader(fh):
            if len(r) >= 2:
                rows.append((r[0], ",".join(r[1:])))
    return rows


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("file_path,SMILES\n")
        for p, smi in rows:
            if "," in smi or '"' in smi:
                smi = '"' + smi.replace('"', '""') + '"'
            fh.write(f"{p},{smi}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round1_root", default=ROUND1_ROOT)
    ap.add_argument("--out_root", default="/data/molsight_realpatent_sft_r2")
    ap.add_argument("--aligned_csv", default=ALIGNED_CSV_DEFAULT)
    ap.add_argument("--real_oversample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(os.path.join(out_root, "uspto_mol"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "real"), exist_ok=True)

    # ---- round-1 train rows: split back into real crops vs synth replay ----
    r1_rows = read_rows(os.path.join(args.round1_root, "uspto_mol", "train_680k.csv"))
    r1_real = [r for r in r1_rows if not os.path.isabs(r[0])]
    r1_replay = [r for r in r1_rows if os.path.isabs(r[0])]
    assert len(r1_real) == ROUND1_TRAIN_N_REAL, f"round-1 real rows: {len(r1_real)}"
    assert len(r1_replay) == ROUND1_REPLAY_N, f"round-1 replay rows: {len(r1_replay)}"
    for p, _ in r1_real:
        assert os.path.isfile(os.path.join(args.round1_root, p)), p

    # ---- observation val + regression20: byte-identical carry-over ----
    for name in ("val.csv", "regression20.csv"):
        shutil.copy2(os.path.join(args.round1_root, "real", name),
                     os.path.join(out_root, "real", name))

    # ---- consolidated crops (round-1 already verified every crop) ----
    if not os.path.isdir(os.path.join(out_root, "crops")):
        shutil.copytree(os.path.join(args.round1_root, "crops"),
                        os.path.join(out_root, "crops"))
    n_crops = sum(len(f) for _, _, f in os.walk(os.path.join(out_root, "crops")))

    # ---- aligned synthetic rows ----
    aligned = read_rows(args.aligned_csv)
    assert len(aligned) >= 3000, f"aligned synth too small: {len(aligned)}"
    for p, _ in aligned:
        assert os.path.isfile(p), p

    train = (r1_real * args.real_oversample) + r1_replay + \
            [(p, s) for p, s in aligned]
    random.Random(args.seed).shuffle(train)
    train_csv = os.path.join(out_root, "uspto_mol", "train_680k.csv")
    write_rows(train_csv, train)

    summary = {
        "round2_of": "markush_v13_realpatent_sft_mix (round-1 winner, 6fig 0.8988)",
        "round1_root": args.round1_root,
        "aligned_csv": os.path.abspath(args.aligned_csv),
        "train_rows": len(train),
        "aligned_synth_rows": len(aligned),
        "real_train_rows": len(r1_real),
        "real_oversample": args.real_oversample,
        "real_rows_in_train": len(r1_real) * args.real_oversample,
        "synth_replay_rows": len(r1_replay),
        "val_rows": len(read_rows(os.path.join(out_root, "real", "val.csv"))),
        "regression_rows": len(read_rows(os.path.join(out_root, "real", "regression20.csv"))),
        "crops_present": n_crops,
        "val_source": "byte-identical copy of round-1 real/val.csv (held out from training)",
        "train_csv": train_csv,
    }
    with open(os.path.join(out_root, "assembly_summary_r2.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Assemble the round-3 real-domain SFT mix (vocabulary-expanded remix).

Round-3 strategy on top of the round-2 winner (markush_v14_realpatent_sft_r2
epoch_1, 6-figure bare-decode mean 0.9045): add aligned-style renders of the
round-3 vocabulary expansion pool (build_vocab_r3.py: fluorinated side
chains, quaternary chiral centres, 2-substituted proline analogs — the
three content classes round-2 chronically missed, including the two 0.0
cases page_260_box_1 / page_219_box_3) while keeping every round-2 training
row verbatim. The 229 unique real crops are oversampled two additional
times so the real domain stays strongly represented against the enlarged
synthetic share (anti-drift / anti-overfit for this single-epoch
continuation).

Train mix:
  - round-2 train rows verbatim: 6687 (real 229 x3 = 687, synth replay
    1000, aligned synth 5000)
  - new vocabulary aligned renders (target ~2800)
  - real crops oversampled x2 additional (458 rows)

Observation val (26) and regression subset (20) are carried over
byte-identical from the round-1 root — every round-3 metric stays directly
comparable. Val crops are never trained on.

Output root layout (MolSight train.py conventions,
data_path_map['uspto'] = 'uspto_mol/train_680k.csv'):
  <out>/uspto_mol/train_680k.csv
  <out>/real/val.csv
  <out>/real/regression20.csv
  <out>/assembly_summary_r3.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil

ROUND1_ROOT = "/data/molsight_realpatent_sft"
ROUND2_ROOT = "/data/molsight_realpatent_sft_r2"
VOCAB_ALIGNED_CSV = "/data/molsight_realpatent_sft_r3/aligned_synth_vocab/train_680k.csv"
ROUND1_TRAIN_N_REAL = 229
ROUND2_TRAIN_ROWS = 6687
REAL_EXTRA_OVERSAMPLE = 2


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        assert header.strip() == "file_path,SMILES", header
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
    ap.add_argument("--round2_root", default=ROUND2_ROOT)
    ap.add_argument("--round1_root", default=ROUND1_ROOT)
    ap.add_argument("--vocab_csv", default=VOCAB_ALIGNED_CSV)
    ap.add_argument("--out_root", default="/data/molsight_realpatent_sft_r3")
    ap.add_argument("--real_extra_oversample", type=int, default=REAL_EXTRA_OVERSAMPLE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(os.path.join(out_root, "uspto_mol"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "real"), exist_ok=True)

    # ---- round-2 train rows, verbatim ----
    r2_rows = read_rows(os.path.join(args.round2_root, "uspto_mol", "train_680k.csv"))
    assert len(r2_rows) == ROUND2_TRAIN_ROWS, f"round-2 rows: {len(r2_rows)}"
    r2_real = [r for r in r2_rows if not os.path.isabs(r[0])]
    assert len(r2_real) == ROUND1_TRAIN_N_REAL * 3, \
        f"round-2 real rows (x3): {len(r2_real)}"

    # real crop paths are relative to the round-1/2 root; they resolve
    # identically under the round-3 root once crops/ is linked
    for p, _ in r2_real:
        assert os.path.isfile(os.path.join(args.round1_root, p)), p

    # ---- observation val + regression20: byte-identical carry-over ----
    for name in ("val.csv", "regression20.csv"):
        shutil.copy2(os.path.join(args.round1_root, "real", name),
                     os.path.join(out_root, "real", name))

    # ---- crops: reuse the consolidated round-1 tree via symlink (the same
    #      tree round-2 copied); falls back to a plain copy ----
    crops_dst = os.path.join(out_root, "crops")
    if not os.path.isdir(crops_dst):
        try:
            os.symlink(os.path.join(args.round1_root, "crops"), crops_dst)
        except OSError:
            shutil.copytree(os.path.join(args.round1_root, "crops"), crops_dst)
    n_crops = sum(len(f) for _, _, f in os.walk(crops_dst))

    # ---- new vocabulary aligned renders ----
    vocab = read_rows(args.vocab_csv)
    assert len(vocab) >= 2000, f"vocab aligned set too small: {len(vocab)}"
    for p, _ in vocab:
        assert os.path.isfile(p), p

    # ---- real crops oversampled x2 additional (unique real = 229) ----
    real_unique = []
    seen = set()
    for p, s in r2_real:
        if p not in seen:
            seen.add(p)
            real_unique.append((p, s))
    assert len(real_unique) == ROUND1_TRAIN_N_REAL
    train = r2_rows + vocab + real_unique * args.real_extra_oversample
    random.Random(args.seed).shuffle(train)
    train_csv = os.path.join(out_root, "uspto_mol", "train_680k.csv")
    write_rows(train_csv, train)

    summary = {
        "round3_of": "markush_v14_realpatent_sft_r2 epoch_1 (6fig 0.9045)",
        "round2_root": args.round2_root,
        "vocab_pool_csv": "/data/molsight_realpatent_sft_r3/vocab_r3_pool.csv",
        "vocab_aligned_csv": os.path.abspath(args.vocab_csv),
        "train_rows": len(train),
        "round2_rows_verbatim": len(r2_rows),
        "vocab_aligned_rows": len(vocab),
        "real_unique": len(real_unique),
        "real_extra_oversample": args.real_extra_oversample,
        "real_rows_in_train": len(r2_real) + len(real_unique) * args.real_extra_oversample,
        "val_rows": len(read_rows(os.path.join(out_root, "real", "val.csv"))),
        "regression_rows": len(read_rows(os.path.join(out_root, "real", "regression20.csv"))),
        "crops_present": n_crops,
        "val_source": "byte-identical copy of round-1 real/val.csv (held out from training)",
        "train_csv": train_csv,
    }
    with open(os.path.join(out_root, "assembly_summary_r3.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

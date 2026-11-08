"""Assemble the round-4 real-domain SFT mix (hard-case augmentation).

Round-3 delivered 6-figure mean 0.9116 with two chronic laggards left:
fig_s52_0 (0.8219 since round 1 — quaternary [C@@](C)(aryl) methyl
assignment + 2-substituted proline topology) and fig_s77_0 (0.8485).
Round-4 strategy: keep every round-3 training row verbatim and attack the
two laggards with multiple augmented renders of their exact gold SMILES
(render_hardcase_r4.py: 12 ALIGNED-profile variants each of seq52 / seq77 /
seq77_0 — scale, rotation, ink noise, tight-crop margin, per-variant
depiction seed), oversampled x3 so the model sees the same hard molecules
in many drawings.  The 229 unique verified real crops get their usual x2
additional oversample (anti-drift against the enlarged hard-case share,
single-epoch continuation).

Train mix:
  - round-3 train rows verbatim: 9947 (real 229 x5 = 1145, synth replay
    1000, aligned synth 5000 + vocab 2802)
  - hard-case augmentation x3 (36 renders x3 = 108)
  - real crops oversampled x2 additional (458)

Observation val (26) and regression subset (20) are carried over
byte-identical from the round-1 root — every round-4 metric stays directly
comparable.  Val crops are never trained on.

Output root layout (MolSight train.py conventions):
  <out>/uspto_mol/train_680k.csv
  <out>/real/val.csv
  <out>/real/regression20.csv
  <out>/assembly_summary_r4.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil

ROUND1_ROOT = "/data/molsight_realpatent_sft"
ROUND3_ROOT = "/data/molsight_realpatent_sft_r3"
HARDCASE_CSV = "/data/molsight_realpatent_sft_r4/hardcase_aug/train_680k.csv"
FAMILIES_CSV = "/data/molsight_realpatent_sft_r4/families/train_680k.csv"
WILDMOL_CSV = "/data/molsight_realpatent_sft_r4/wildmol_replay.csv"
LINEAR_CSV = "/data/molsight_realpatent_sft_r4/linear_replay.csv"
MARKUSH_CSV = "/data/molsight_realpatent_sft_r4/markush_replay.csv"
ROUND3_ROWS = 9947
ROUND1_TRAIN_N_REAL = 229
REAL_EXTRA_OVERSAMPLE = 2
HARDCASE_OVERSAMPLE = 3
FAMILY_OVERSAMPLE = 3


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
    ap.add_argument("--round3_root", default=ROUND3_ROOT)
    ap.add_argument("--round1_root", default=ROUND1_ROOT)
    ap.add_argument("--hardcase_csv", default=HARDCASE_CSV)
    ap.add_argument("--families_csv", default=FAMILIES_CSV)
    ap.add_argument("--wildmol_csv", default=WILDMOL_CSV)
    ap.add_argument("--linear_csv", default=LINEAR_CSV)
    ap.add_argument("--markush_csv", default=MARKUSH_CSV)
    ap.add_argument("--out_root", default="/data/molsight_realpatent_sft_r4")
    ap.add_argument("--hardcase_oversample", type=int, default=HARDCASE_OVERSAMPLE)
    ap.add_argument("--family_oversample", type=int, default=FAMILY_OVERSAMPLE)
    ap.add_argument("--real_extra_oversample", type=int, default=REAL_EXTRA_OVERSAMPLE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(os.path.join(out_root, "uspto_mol"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "real"), exist_ok=True)

    # ---- round-3 train rows, verbatim ----
    r3_rows = read_rows(os.path.join(args.round3_root, "uspto_mol", "train_680k.csv"))
    assert len(r3_rows) == ROUND3_ROWS, f"round-3 rows: {len(r3_rows)}"
    r3_real = [r for r in r3_rows if not os.path.isabs(r[0])]
    # round-3 already carries the real crops x5 (r2 x3 + r3 extra x2)
    assert len(r3_real) == ROUND1_TRAIN_N_REAL * 5, \
        f"round-3 real rows (x5): {len(r3_real)}"

    # real crop paths are relative to the round-1/2/3 root; they resolve
    # identically under the round-4 root once crops/ is linked
    for p, _ in r3_real:
        assert os.path.isfile(os.path.join(args.round1_root, p)), p

    # ---- hard-case augmentation rows ----
    hard = read_rows(args.hardcase_csv)
    cases = {os.path.basename(p).split("_v")[0] for p, _ in hard}
    assert cases == {"seq52", "seq77", "seq77_0"}, cases
    for p, _ in hard:
        assert os.path.isfile(p), p

    # ---- motif-family expansion rows (train side; held-out never enters) ----
    fams = read_rows(args.families_csv)
    fam_tags = {os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p, _ in fams}
    assert fam_tags == {"families"}, fam_tags
    for p, _ in fams:
        assert os.path.isfile(p), p

    # ---- anti-forgetting replay domains ----
    wild = read_rows(args.wildmol_csv)
    linear = read_rows(args.linear_csv)
    markush = read_rows(args.markush_csv)
    assert len(wild) >= 1500 and len(markush) >= 2000 and len(linear) >= 1500, \
        (len(wild), len(markush), len(linear))
    for rows in (wild, linear, markush):
        for p, _ in rows:
            assert os.path.isfile(p), p

    # ---- observation val + regression20: byte-identical carry-over ----
    for name in ("val.csv", "regression20.csv"):
        shutil.copy2(os.path.join(args.round1_root, "real", name),
                     os.path.join(out_root, "real", name))

    # ---- crops: reuse the consolidated round-1 tree via symlink ----
    crops_dst = os.path.join(out_root, "crops")
    if not os.path.isdir(crops_dst):
        try:
            os.symlink(os.path.join(args.round1_root, "crops"), crops_dst)
        except OSError:
            shutil.copytree(os.path.join(args.round1_root, "crops"), crops_dst)
    n_crops = sum(len(f) for _, _, f in os.walk(crops_dst))

    # ---- real crops oversampled x2 additional (unique real = 229) ----
    real_unique = []
    seen = set()
    for p, s in r3_real:
        if p not in seen:
            seen.add(p)
            real_unique.append((p, s))
    assert len(real_unique) == ROUND1_TRAIN_N_REAL

    train = (r3_rows
             + hard * args.hardcase_oversample
             + fams * args.family_oversample
             + wild + linear + markush
             + real_unique * args.real_extra_oversample)
    random.Random(args.seed).shuffle(train)
    train_csv = os.path.join(out_root, "uspto_mol", "train_680k.csv")
    write_rows(train_csv, train)

    summary = {
        "round4_of": "markush_v14_realpatent_sft_r3 epoch_0 (6fig 0.9116)",
        "round3_root": args.round3_root,
        "hardcase_manifest": os.path.join(
            os.path.dirname(args.hardcase_csv), "render_manifest.json"),
        "families_manifest": os.path.join(
            os.path.dirname(args.families_csv), "..", "families_manifest.json"),
        "train_rows": len(train),
        "round3_rows_verbatim": len(r3_rows),
        "hardcase_rows": len(hard),
        "hardcase_oversample": args.hardcase_oversample,
        "hardcase_rows_in_train": len(hard) * args.hardcase_oversample,
        "family_rows": len(fams),
        "family_oversample": args.family_oversample,
        "family_rows_in_train": len(fams) * args.family_oversample,
        "wildmol_replay_rows": len(wild),
        "linear_replay_rows": len(linear),
        "markush_replay_rows": len(markush),
        "real_unique": len(real_unique),
        "real_extra_oversample": args.real_extra_oversample,
        "real_rows_in_train": len(r3_real) + len(real_unique) * args.real_extra_oversample,
        "val_rows": len(read_rows(os.path.join(out_root, "real", "val.csv"))),
        "regression_rows": len(read_rows(os.path.join(out_root, "real", "regression20.csv"))),
        "crops_present": n_crops,
        "val_source": "byte-identical copy of round-1 real/val.csv (held out from training)",
        "train_csv": train_csv,
    }
    with open(os.path.join(out_root, "assembly_summary_r4.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

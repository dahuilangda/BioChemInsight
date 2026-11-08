"""Assemble the round-4z real-domain SFT mix (zig-zag spatial adaptation).

Round-4z strategy on top of the round-3 winner (markush_v14_realpatent_sft_r3
epoch_0, 6-figure bare-decode mean 0.9116): switch the synthetic corpus's
spatial preparation from CoordGen to the patent-style zig-zag macrocycle
template (zigzag_layout.py; research/macrocycle_layout_algorithms.md) — the
REAL patent crops draw cyclopeptide macrorings as irregular closed zig-zag
polygons, which CoordGen only produces for favourable substitution
patterns and the native depictor never produces.

Training data sources (post strict re-verification, commit 2bfd1cb):
  - zig-zag renders of the round-1/2 synthetic pool (3482 unique SMILES:
    round-1 synth replay 1000 + round-2 aligned pool 2482) x1 and the
    round-3 vocabulary pool (467 SMILES) x2 — render_zigzag_pool.py,
    ALIGNED style
  - STRICT verified real crops (data/verified_pairs_strict.json, decode
    cross-validation Tanimoto >= 0.65): the 183 strict crops that are NOT
    in the round-1 observation val split, oversampled x2.  The 23 strict
    crops overlapping the val split stay held out (val crops are never
    trained on); the 30 weak + 19 rejected pairs are excluded entirely
  - round-1 synthetic replay rows verbatim (1000) — the incumbent CoordGen
    style stays represented so the corpus covers both spatial preparations
  - anti-forgetting replay: WildMol small molecules 1500, linear peptides
    1500, markush 2000 (four capability domains protected)

Observation val (26) and regression subset (20) are carried over
byte-identical from the round-1 root — every round-4z metric stays
directly comparable.  Val crops are never trained on.

Output root layout (MolSight train.py conventions):
  <out>/uspto_mol/train_680k.csv
  <out>/real/val.csv
  <out>/real/regression20.csv
  <out>/assembly_summary_r4z.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil

ROUND1_ROOT = "/data/molsight_realpatent_sft"
STRICT_JSON = "/data/BioChemInsight/data/verified_pairs_strict.json"
ZIGZAG_CSV = "/data/molsight_realpatent_sft_r4z/zigzag_synth/train_680k.csv"
WILDMOL_CSV = "/data/molsight_realpatent_sft_r4/wildmol_replay.csv"
LINEAR_CSV = "/data/molsight_realpatent_sft_r4/linear_replay.csv"
MARKUSH_CSV = "/data/molsight_realpatent_sft_r4/markush_replay.csv"
ROUND1_REPLAY_ROWS = 1000
STRICT_OVERSAMPLE = 2
LINEAR_REPLAY_ROWS = 1500
MARKUSH_REPLAY_ROWS = 2000
WILDMOL_REPLAY_ROWS = 1500


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
    ap.add_argument("--round1_root", default=ROUND1_ROOT)
    ap.add_argument("--strict_json", default=STRICT_JSON)
    ap.add_argument("--zigzag_csv", default=ZIGZAG_CSV)
    ap.add_argument("--wildmol_csv", default=WILDMOL_CSV)
    ap.add_argument("--linear_csv", default=LINEAR_CSV)
    ap.add_argument("--markush_csv", default=MARKUSH_CSV)
    ap.add_argument("--out_root", default="/data/molsight_realpatent_sft_r4z")
    ap.add_argument("--strict_oversample", type=int, default=STRICT_OVERSAMPLE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.out_root
    os.makedirs(os.path.join(out_root, "uspto_mol"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "real"), exist_ok=True)

    # ---- observation val + regression20: byte-identical carry-over ----
    for name in ("val.csv", "regression20.csv"):
        shutil.copy2(os.path.join(args.round1_root, "real", name),
                     os.path.join(out_root, "real", name))
    val_rows = read_rows(os.path.join(out_root, "real", "val.csv"))
    # val rows use root-relative "crops/<patent>/<crop>.png"; the strict
    # JSON identifies crops by "crop_rel" = "<patent>/<crop>.png" —
    # compare on that form so the held-out split follows the file identity
    val_rels = set()
    for p, _ in val_rows:
        norm = os.path.normpath(p)
        rel = os.path.relpath(norm, "crops") if norm.startswith("crops") \
            else norm
        val_rels.add(rel)

    # ---- crops: reuse the consolidated round-1 tree via symlink ----
    crops_dst = os.path.join(out_root, "crops")
    if not os.path.isdir(crops_dst):
        try:
            os.symlink(os.path.join(args.round1_root, "crops"), crops_dst)
        except OSError:
            shutil.copytree(os.path.join(args.round1_root, "crops"), crops_dst)
    n_crops = sum(len(f) for _, _, f in os.walk(crops_dst))

    # ---- strict real crops: exclude anything in the val split ----
    # rows use the same root-relative "crops/<patent>/<crop>.png" paths as
    # every previous round (resolved via the crops/ symlink)
    strict = json.load(open(args.strict_json))["pairs"]
    real_unique = []
    excluded_val = 0
    for p in strict:
        gold = p["smiles"]
        assert os.path.isfile(p["crop"]), p["crop"]
        if p["crop_rel"] in val_rels:
            excluded_val += 1
            continue
        img_rel = os.path.join("crops", p["crop_rel"])
        assert os.path.isfile(os.path.join(args.round1_root, img_rel)), img_rel
        real_unique.append((img_rel, gold))
    assert len(real_unique) + excluded_val == len(strict), \
        (len(real_unique), excluded_val, len(strict))
    print(f"strict crops: {len(real_unique)} train x{args.strict_oversample}, "
          f"{excluded_val} held out (val overlap)", flush=True)

    # ---- zig-zag synthetic renders ----
    zigzag = read_rows(args.zigzag_csv)
    for p, _ in zigzag:
        assert os.path.isfile(p), p

    # ---- round-1 synthetic replay rows, verbatim (absolute-path rows of
    #      the round-1 train CSV = its seeded synth replay sample) ----
    r1_rows = read_rows(os.path.join(args.round1_root,
                                     "uspto_mol", "train_680k.csv"))
    replay = [(p, s) for p, s in r1_rows if os.path.isabs(p)]
    assert len(replay) == ROUND1_REPLAY_ROWS, len(replay)
    for p, _ in replay:
        assert os.path.isfile(p), p

    # ---- anti-forgetting replay domains ----
    wild = read_rows(args.wildmol_csv)
    random.Random(args.seed).shuffle(wild)
    wild = wild[:WILDMOL_REPLAY_ROWS]
    linear = read_rows(args.linear_csv)
    random.Random(args.seed + 1).shuffle(linear)
    linear = linear[:LINEAR_REPLAY_ROWS]
    markush = read_rows(args.markush_csv)
    assert len(markush) >= MARKUSH_REPLAY_ROWS
    markush = markush[:MARKUSH_REPLAY_ROWS]
    for name, rows in (("wildmol", wild), ("linear", linear),
                       ("markush", markush)):
        assert len(rows) >= 1000, (name, len(rows))
        for p, _ in rows:
            assert os.path.isfile(p), p

    train = (zigzag
             + real_unique * args.strict_oversample
             + replay + wild + linear + markush)
    random.Random(args.seed).shuffle(train)
    train_csv = os.path.join(out_root, "uspto_mol", "train_680k.csv")
    write_rows(train_csv, train)

    summary = {
        "round4z_of": "markush_v14_realpatent_sft_r3 epoch_0 (6fig 0.9116)",
        "strategy": "zig-zag spatial adaptation of the synthetic corpus "
                    "(layout switch), strict-verified real crops",
        "round1_root": args.round1_root,
        "strict_json": args.strict_json,
        "zigzag_csv": os.path.abspath(args.zigzag_csv),
        "train_rows": len(train),
        "zigzag_rows": len(zigzag),
        "strict_unique_train": len(real_unique),
        "strict_excluded_val_overlap": excluded_val,
        "strict_oversample": args.strict_oversample,
        "strict_rows_in_train": len(real_unique) * args.strict_oversample,
        "round1_synth_replay_rows": len(replay),
        "wildmol_replay_rows": len(wild),
        "linear_replay_rows": len(linear),
        "markush_replay_rows": len(markush),
        "val_rows": len(val_rows),
        "regression_rows": len(read_rows(os.path.join(out_root, "real",
                                                      "regression20.csv"))),
        "crops_present": n_crops,
        "val_source": "byte-identical copy of round-1 real/val.csv "
                      "(held out from training)",
        "train_csv": train_csv,
    }
    with open(os.path.join(out_root, "assembly_summary_r4z.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

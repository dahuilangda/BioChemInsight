"""Blind MolSight decode acceptance test for render style alignment.

Samples 10 real verified crops and 10 aligned synthetic renders (blind:
no gold SMILES reaches the decoder; the model just decodes images), then
decodes each through the MolSight worker and compares Tanimoto similarity
of the decode against each image's paired gold SMILES.

Acceptance criterion (task spec): the two Tanimoto distributions must show
no significant difference — two-sided Kolmogorov-Smirnov p > 0.05, or
|mean_real - mean_synth| < 0.05.

Usage:
  python blind_decode_tanimoto.py \
      --real-pairs /data/BioChemInsight/data/real_crops/real_crops_manifest.json \
      --synth-pairs /tmp/style_pairs_aligned/pairs.json \
      --n 10 [--seed 13] [--out /tmp/blind_decode_result.json]

The real side reads the verified rows of a crop manifest directly; the
synth side reads a pairs.json from render_style_pairs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/BioChemInsight")

os.environ.setdefault("MOLSIGHT_CHECKPOINT",
                      "/data/molsight_repo/runs/markush_v13_rft/ckpt_model/epoch_0.pth")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MOLSIGHT_DEVICE", "cuda:0")


def sample_real(manifest_path: str, n: int, seed: int) -> list[dict]:
    data = json.load(open(manifest_path))
    base = os.path.dirname(os.path.abspath(manifest_path))
    rows = [r for r in data["rows"] if r.get("verified") is True
            and r.get("smiles")]
    rng = random.Random(seed)
    picks = rng.sample(rows, n)
    return [{"image": os.path.join(base, r["crop"]),
             "gold": r["smiles"],
             "src": os.path.basename(manifest_path) + ":" + r["crop"]}
            for r in picks]


def sample_synth(pairs_file: str, n: int, seed: int) -> list[dict]:
    pairs = json.load(open(pairs_file))
    rng = random.Random(seed + 1)
    picks = rng.sample(pairs, n)
    return [{"image": p["synth"], "gold": p["smiles"],
             "src": os.path.basename(pairs_file) + ":" +
                    os.path.basename(p["synth"])} for p in picks]


def sample_paired(pairs_file: str, n: int, seed: int) -> list[dict]:
    """Sample n molecule pairs and return BOTH renderings of each: the real
    crop and the synthetic render of the same gold SMILES. Holding the
    chemistry constant isolates the rendering-style effect on decode."""
    pairs = json.load(open(pairs_file))
    rng = random.Random(seed)
    picks = rng.sample(pairs, n)
    items = []
    for i, p in enumerate(picks):
        items.append({"image": p["real"], "gold": p["smiles"],
                      "group": "real",
                      "src": "pair%02d:real:%s" % (i, os.path.basename(p["real"]))})
        items.append({"image": p["synth"], "gold": p["smiles"],
                      "group": "synth",
                      "src": "pair%02d:synth:%s" % (i, os.path.basename(p["synth"]))})
    return items


def tanimoto(a: str, b: str) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fa, fb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-pairs", default=
                    "/data/BioChemInsight/data/real_crops/real_crops_manifest.json")
    ap.add_argument("--synth-pairs", default=
                    "/tmp/style_pairs_aligned/pairs.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--paired", action="store_true",
                    help="sample n molecule pairs and decode BOTH renderings "
                         "of each (real crop + synthetic render of the same "
                         "gold SMILES) so both sides see identical chemistry")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="/tmp/blind_decode_result.json")
    args = ap.parse_args()

    if args.paired:
        items = sample_paired(args.synth_pairs, args.n, args.seed)
    else:
        items = sample_real(args.real_pairs, args.n, args.seed) + \
            sample_synth(args.synth_pairs, args.n, args.seed)
        for it in items:
            it["group"] = "real" if it["src"].startswith("real_crops_manifest") \
                else "synth"
    rng = random.Random(args.seed + 2)
    rng.shuffle(items)  # blind the decoder to the group order

    from utils.molsight_service import MolsightService
    service = MolsightService()
    try:
        results = service.predict_many([it["image"] for it in items])
    finally:
        service.shutdown()

    rows = []
    for it, res in zip(items, results):
        dec = res.get("smiles") if res.get("ok") else None
        sim = tanimoto(dec, it["gold"]) if dec else None
        rows.append({"src": it["src"], "group": it["group"],
                     "decoded_ok": bool(dec),
                     "tanimoto_vs_gold": sim,
                     "decoded_len": len(dec) if dec else 0})

    tr = [r["tanimoto_vs_gold"] for r in rows
          if r["group"] == "real" and r["tanimoto_vs_gold"] is not None]
    ts = [r["tanimoto_vs_gold"] for r in rows
          if r["group"] == "synth" and r["tanimoto_vs_gold"] is not None]
    from scipy.stats import ks_2samp
    ks = ks_2samp(tr, ts) if len(tr) > 2 and len(ts) > 2 else None
    out = {
        "n": args.n, "seed": args.seed,
        "rows": rows,
        "tanimoto_real": tr, "tanimoto_synth": ts,
        "mean_real": sum(tr) / len(tr) if tr else None,
        "mean_synth": sum(ts) / len(ts) if ts else None,
        "mean_diff": (abs(sum(tr) / len(tr) - sum(ts) / len(ts))
                      if tr and ts else None),
        "ks_stat": ks.statistic if ks else None,
        "ks_p": ks.pvalue if ks else None,
        "pass": bool(ks and ks.pvalue > 0.05) or
                bool(tr and ts and abs(sum(tr) / len(tr) - sum(ts) / len(ts)) < 0.05),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"  {r['group']:5s} tanimoto={r['tanimoto_vs_gold']} "
              f"ok={r['decoded_ok']} {r['src']}")


if __name__ == "__main__":
    main()

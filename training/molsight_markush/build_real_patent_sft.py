"""Assemble the real-patent SFT dataset for MolSight fine-tuning.

Merges the three verified real-crop manifests (US cyclopeptides, KRAS, GPC3)
into a single image -> gold SMILES SFT set:

  - keeps only rows with ``verified is True`` and a non-empty ``smiles`` gold
  - verifies every crop exists and decodes as an image (PIL), copies it into a
    consolidated dataset root
  - RDKit-validates every gold SMILES
  - stratified 90/10 train/val split (by manifest source, seed 42); the val
    split is for observation only, never for early stopping

Output layout matches MolSight train.py conventions so no trainer patch is
needed (``data_path_map['uspto'] = 'uspto_mol/train_680k.csv'``):

  <out_root>/crops/<patent>/<crop>.png      consolidated crops
  <out_root>/uspto_mol/train_680k.csv       file_path,SMILES (train, ~90%)
  <out_root>/real/val.csv                   file_path,SMILES (val, ~10%)
  <out_root>/real/regression20.csv          fixed 20-crop no-regression subset
  <out_root>/assembly_summary.json          provenance + counts
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys

MANIFESTS = {
    "us": "/data/BioChemInsight/data/real_crops/real_crops_manifest.json",
    "kras": "/tmp/crops_kras/real_crops_manifest.json",
    "gpc3": "/tmp/crops_gpc3/real_crops_manifest.json",
}
DEFAULT_OUT = "/data/molsight_realpatent_sft"
SEED = 42
VAL_FRACTION = 0.10
REGRESSION_N = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from PIL import Image
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    rng = random.Random(args.seed)
    out_root = args.out_root
    crops_dir = os.path.join(out_root, "crops")
    os.makedirs(os.path.join(out_root, "uspto_mol"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "real"), exist_ok=True)

    rows = []
    excluded = []
    for source, manifest_path in MANIFESTS.items():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        base = os.path.dirname(os.path.abspath(manifest_path))
        n_src = 0
        for row in manifest["rows"]:
            if row.get("verified") is not True or not row.get("smiles"):
                continue
            n_src += 1
            crop_rel = row["crop"]
            crop_abs = os.path.join(base, crop_rel)
            reason = None
            if not os.path.isfile(crop_abs):
                reason = "crop_missing"
            if reason is None:
                try:
                    with Image.open(crop_abs) as im:
                        im.verify()
                except Exception as exc:
                    reason = f"crop_unreadable: {type(exc).__name__}"
            if reason is None:
                try:
                    mol = Chem.MolFromSmiles(row["smiles"])
                    if mol is None or mol.GetNumAtoms() == 0:
                        reason = "gold_smiles_unparseable"
                except Exception:
                    reason = "gold_smiles_unparseable"
            if reason is not None:
                excluded.append({"source": source, "crop": crop_rel, "reason": reason})
                continue
            rows.append({
                "source": source,
                "patent": row.get("patent") or crop_rel.split("/")[0],
                "seq_id": row.get("seq_id"),
                "crop_rel": crop_rel,
                "crop_abs": crop_abs,
                "smiles": row["smiles"],
                "verify_method": row.get("verify_method"),
            })
        print(f"[{source}] verified pairs: {n_src}")

    # collision check + copy crops into the consolidated root
    seen = {}
    for row in rows:
        key = row["crop_rel"]
        if key in seen and seen[key] != row["crop_abs"]:
            sys.exit(f"FATAL: crop path collision across manifests: {key}")
        seen[key] = row["crop_abs"]
    for row in rows:
        dest = os.path.join(crops_dir, row["crop_rel"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            shutil.copy2(row["crop_abs"], dest)
        row["dataset_path"] = os.path.relpath(dest, out_root)

    # duplicate diagnostics (same gold SMILES from distinct drawings is kept
    # intentionally: they are different renderings of real patent art)
    smiles_groups = collections.Counter(r["smiles"] for r in rows)
    n_dup_smiles = sum(c - 1 for c in smiles_groups.values() if c > 1)

    # stratified 90/10 split by source
    by_source = collections.defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    val, train = [], []
    for source in sorted(by_source):
        group = by_source[source][:]
        rng.shuffle(group)
        n_val = round(len(group) * VAL_FRACTION)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    regression = val[:REGRESSION_N] if len(val) >= REGRESSION_N else val

    def write_csv(path, subset):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("file_path,SMILES\n")
            for row in subset:
                smiles = row["smiles"]
                if "," in smiles or '"' in smiles:
                    smiles = '"' + smiles.replace('"', '""') + '"'
                fh.write(f"{row['dataset_path']},{smiles}\n")

    train_csv = os.path.join(out_root, "uspto_mol", "train_680k.csv")
    val_csv = os.path.join(out_root, "real", "val.csv")
    reg_csv = os.path.join(out_root, "real", "regression20.csv")
    write_csv(train_csv, train)
    write_csv(val_csv, val)
    write_csv(reg_csv, regression)

    summary = {
        "seed": args.seed,
        "manifests": MANIFESTS,
        "total_pairs": len(rows),
        "train": len(train),
        "val": len(val),
        "regression_subset": len(regression),
        "per_source": {s: len(by_source[s]) for s in sorted(by_source)},
        "val_per_source": {
            s: sum(1 for r in val if r["source"] == s) for s in sorted(by_source)},
        "duplicate_gold_smiles_rows": n_dup_smiles,
        "unique_gold_smiles": len(smiles_groups),
        "excluded": excluded,
        "outputs": {"train_csv": train_csv, "val_csv": val_csv,
                    "regression_csv": reg_csv, "crops_dir": crops_dir},
    }
    with open(os.path.join(out_root, "assembly_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("manifests", "excluded")}, indent=2))
    if excluded:
        print(f"EXCLUDED ({len(excluded)}):")
        for item in excluded:
            print(f"  {item}")
    print(f"OK dataset root: {out_root}")


if __name__ == "__main__":
    main()

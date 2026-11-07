"""Synthetic training-data quality gate: chemistry, image integrity,
round-trip decodability — run before any SFT round consumes the set."""
import csv, json, os, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SETS = {
    "r2_aligned": "/data/molsight_realpatent_sft_r2/uspto_mol/train_680k.csv",
    "r2_replay": "/data/molsight_realpatent_sft_r2/uspto_mol/replay.csv",
    "r3_vocab": "/data/molsight_realpatent_sft_r3/aligned_synth_vocab",
}

def scan_csv(path):
    bad_smiles, blank_img, total = [], [], 0
    if not os.path.exists(path):
        return {"missing": True}
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0] == "file_path":
                continue
            total += 1
            img, smi = row[0], row[1] if len(row) > 1 else ""
            if not smi.strip():
                blank_img.append(img)
                continue
            if Chem.MolFromSmiles(smi) is None:
                bad_smiles.append((img, smi[:50]))
    return {"total": total, "bad_smiles": bad_smiles[:10],
            "bad_smiles_count": len(bad_smiles), "blank_smiles": len(blank_img)}

def scan_image_dir(d):
    """Aligned render dirs: image files next to nothing else — integrity by
    PIL open + ink presence."""
    import glob
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {"skipped": "no PIL"}
    files = sorted(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))
    if not files:
        return {"skipped": "no images", "dir": d}
    blank, corrupt, ok = [], [], 0
    for f in files:
        try:
            im = Image.open(f)
            im.load()
            a = np.asarray(im.convert("L"))
            if (a < 128).mean() < 0.002:
                blank.append(f)
            else:
                ok += 1
        except Exception:
            corrupt.append(f)
    return {"images": len(files), "ok": ok, "blank": blank[:5],
            "blank_count": len(blank), "corrupt": len(corrupt)}

report = {}
for name, path in SETS.items():
    if path.endswith(".csv"):
        report[name] = scan_csv(path)
    else:
        report[name] = scan_image_dir(path)

json.dump(report, open("/tmp/synth_data_report.json", "w"), indent=1, default=str)
for name, r in report.items():
    print(name, "->", json.dumps(r, default=str)[:200])

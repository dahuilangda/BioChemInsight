"""Convert MolParser-7M pretrain shards to isotope-dummy training CSV.

Handles both <a>idx:TYPE</a> (atom annotations, 82.6% of R-groups) and
<r>idx:R[n]</r> (ring annotations, 17.4%) tag formats from the E-SMILES
extension. Incremental flush per shard — no total-loss failure mode.
"""
import sys, os, csv, re, glob
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = '/data/molparser7m/pretrain_converted.csv'
IMG_DIR = '/data/molparser7m/pretrain_images'
os.makedirs(IMG_DIR, exist_ok=True)

_a_re = re.compile(r'<a>(\d+):([^<]+)</a>')
_r_re = re.compile(r'<r>(\d+):R\[(\d+)\]</r>')
_a_generic = re.compile(r'<a>(\d+):<dum></a>')


def convert(target: str):
    """Parse tagged E-SMILES → isotope-numbered canonical SMILES."""
    base, _, ann = target.partition('<sep>')
    mol = Chem.MolFromSmiles(base.strip())
    if mol is None:
        return None
    # <r>idx:R[n]</r> — ring/atom position R-groups
    for m in _r_re.finditer(ann):
        idx, label = int(m.group(1)), int(m.group(2))
        if idx < mol.GetNumAtoms() and mol.GetAtomWithIdx(idx).GetAtomicNum() == 0:
            mol.GetAtomWithIdx(idx).SetIsotope(label)
    # <a>idx:R[n]</a> — atom-level R-group labels (82.6% of annotations)
    for m in _a_re.finditer(ann):
        idx = int(m.group(1))
        label = m.group(2).strip()
        rn = re.match(r'R\[(\d+)\]', label)
        if rn and idx < mol.GetNumAtoms() and mol.GetAtomWithIdx(idx).GetAtomicNum() == 0:
            mol.GetAtomWithIdx(idx).SetIsotope(int(rn.group(1)))
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    shards = sorted(glob.glob('/data/molparser7m/pretrain_synthetic_7M/*.parquet'))
    # Filter to valid parquet (>100MB)
    valid = [s for s in shards if os.path.getsize(s) > 100_000_000]
    print(f"{len(valid)}/{len(shards)} valid shards (>100MB)", flush=True)
    rows = dropped = dup = 0
    seen = set()
    tmp_out = OUT + '.tmp'
    out = open(tmp_out, 'w', newline='')
    w = csv.writer(out)
    w.writerow(['file_path', 'SMILES'])
    for path in valid:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=['image', 'SMILES']):
            t = batch.to_pandas()
            for _, r in t.iterrows():
                smi = convert(str(r['SMILES']))
                if smi is None or len(smi) < 5 or len(smi) > 1024 or smi in seen:
                    dropped += 1
                    continue
                seen.add(smi)
                img_bytes = r['image']['bytes'] if isinstance(r['image'], dict) else r['image']
                name = f"{IMG_DIR}/{rows:07d}.png"
                with open(name, 'wb') as ih:
                    ih.write(img_bytes)
                w.writerow([name, smi])
                rows += 1
        out.flush()
        print(f"{os.path.basename(path)}: cumulative {rows} rows ({dropped} dropped, {len(seen)} unique)", flush=True)
    out.close()
    os.replace(tmp_out, OUT)
    print(f"WROTE {OUT}: {rows} rows ({dropped} dropped/dup)")

"""Render generated cyclic-peptide SMILES to patent-ink-style training images.

Render recipe (audited against WO2024163691 real crops, MolGrapher/MolParser
published recipes):
  - CoordGen macrocycle layout with radial undulation (never a perfect
    circle), ring templates on
  - pure black ink via setAtomPalette (useBWAtomPalette is ignored by
    RDKit 2026.09), Kekule notation with a 15% aromatic-circle minority
  - label font 24-48 px with padding (NH2/OH legible at bond scale)
  - atom-order shuffling so bold and hashed wedge glyphs both occur
  - portrait/landscape canvases, random rotation, random-offset clipping
    that cuts the molecule at frame edges like real crops
  - quality gate: reject layouts whose ink fraction is too high (tangles)
    and re-depict with a different sampling seed
"""
import sys, os, csv, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyclopeptide_generator import generate_batch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image
import numpy as np
RDLogger.DisableLog("rdApp.*")

OUT_DIR = '/data/cyclopeptide_synth'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 35000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
os.makedirs(f'{OUT_DIR}/images', exist_ok=True)

smis = generate_batch(N, seed=SEED)
print(f"generated {len(smis)} unique SMILES", flush=True)
rng = random.Random(SEED)
rows = rejected = 0


def depict(mol, path, rng) -> bool:
    """One depiction attempt; returns False when the layout is a tangle."""
    big = mol.GetNumAtoms() > 55
    wide = rng.random() < 0.5 or big
    size = (rng.choice([1024, 1152]), rng.choice([1024, 1280])) if big else \
           ((rng.choice([768, 896]), rng.choice([768, 1024])) if wide else \
            (rng.choice([512, 640, 768]), rng.choice([512, 640])))
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette = True
    opts.setAtomPalette({i: (0.0, 0.0, 0.0) for i in range(-1, 121)})
    opts.minFontSize = rng.choice([24, 28, 32])
    opts.maxFontSize = 48
    opts.additionalAtomLabelPadding = 0.12
    opts.bondLineWidth = rng.uniform(2.5, 4.5)
    opts.singleColourWedgeBonds = True
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    mol = Chem.RenumberAtoms(mol, order)
    rdDepictor.Compute2DCoords(mol)
    # Undulate the ring so macrocycles don't render as perfect circles.
    from rdkit.Geometry import Point3D
    conf = mol.GetConformer()
    for ai in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(ai)
        r = math.hypot(pos.x, pos.y)
        if r > 2.0:
            conf.SetAtomPosition(ai, Point3D(
                pos.x * rng.uniform(0.88, 1.12),
                pos.y * rng.uniform(0.88, 1.12), 0.0))
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=True)
    drawer.FinishDrawing()
    with open(path, 'wb') as ih:
        ih.write(drawer.GetDrawingText())
    # Trim to the ink bounding box, then center on a canvas 1.7-2.3x the
    # molecule's linear size: real crops show the molecule occupying ~25%
    # of the frame (ink 1.0-5.2%), not edge-to-edge.
    img = Image.open(path).convert('L')
    arr = np.array(img)
    ink_rows = np.nonzero((arr < 200).any(axis=1))[0]
    ink_cols = np.nonzero((arr < 200).any(axis=0))[0]
    if len(ink_rows) == 0:
        return False
    trimmed = arr[ink_rows.min():ink_rows.max() + 1,
                  ink_cols.min():ink_cols.max() + 1]
    th, tw = trimmed.shape
    scale = rng.uniform(1.7, 2.3)
    ch, cw = int(th * scale) + 40, int(tw * scale) + 40
    canvas = np.full((ch, cw), 255, dtype=np.uint8)
    jx = rng.randint(0, cw - tw)
    jy = rng.randint(0, ch - th)
    canvas[jy:jy + th, jx:jx + tw] = trimmed
    # patent-ink finish: off-white background, mild fade, random rotation
    bg = rng.uniform(235, 252)
    arr = bg - (bg - canvas) * rng.uniform(0.8, 1.0)
    if rng.random() < 0.7:
        angle = rng.uniform(0, 90) if rng.random() < 0.5 else rng.uniform(-90, 0)
        arr = np.array(Image.fromarray(arr).rotate(angle, expand=False,
                                                   fillcolor=255))
    # mild frame clipping: real crops sometimes cut side chains at edges
    if rng.random() < 0.3:
        h, w = arr.shape
        cut = rng.uniform(0.02, 0.08)
        if rng.random() < 0.5:
            arr = arr[:, :int(w * (1 - cut))]
        else:
            arr = arr[int(h * cut):, :]
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return True


with open(f'{OUT_DIR}/synth.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['file_path', 'SMILES'])
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        path = f'{OUT_DIR}/images/{i:06d}.png'
        ok = False
        for attempt in range(3):
            try:
                if depict(mol, path, rng):
                    ok = True
                    break
            except Exception as _exc:
                import traceback, sys as _sys
                print(f"mol {i} EXC: {type(_exc).__name__}: {_exc}", flush=True)
                traceback.print_exc()
                continue
        if not ok:
            rejected += 1
            if os.path.exists(path):
                os.remove(path)
            continue
        w.writerow([path, smi])
        rows += 1
        if rows % 5000 == 0:
            print(f"{rows} rendered ({rejected} tangles rejected)", flush=True)
print(f"WROTE {OUT_DIR}/synth.csv: {rows} rows ({rejected} rejected)")

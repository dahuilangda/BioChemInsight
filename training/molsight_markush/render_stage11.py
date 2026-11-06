"""Stage-11 render batch: giant macrocycles with the real-patent backbone
motifs (lanthionine, pseudoproline), rendered across THREE spatial
preparations —
  coordgen (55%): Schrodinger coordgenlibs layout (RDKit
                 SetPreferCoordGen) — the open ChemDraw-equivalent
                 macrocycle optimizer; irregular organic meander with
                 uniform bond lengths, matching real patent figures
  circle  (25%): classic ring layout, tight-framed         — retains the
                 canonical style earlier stages trained on
  canvas  (25%): ring layout on a large sparse canvas      — page-figure
                 preparation (hollow centre, wide margins)
plus scan artifacts (rotation, margin crop) on every image.
"""
import sys, os, csv, io, random, math
from collections import deque
sys.path.insert(0, '/data/BioChemInsight/training/molsight_markush')
from cyclopeptide_generator import generate_cyclopeptide
from style_calibration import ALIGNED, coordgen_coords, draw_molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image
import numpy as np
import cv2
RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.environ.get('S11_OUT', '/data/cyclopeptide_stage11')
SHARD = os.environ.get('S11_SHARD', '')
PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
# --layout zigzag: patent-style zig-zag macroring template
# (zigzag_layout.py) instead of CoordGen; default stays coordgen
_argv_layout = 'zigzag' in sys.argv and len(sys.argv) > sys.argv.index('--layout') + 1 \
    and sys.argv[sys.argv.index('--layout') + 1] == 'zigzag'
LAYOUT = 'zigzag' if _argv_layout else 'coordgen'
img_dir = f'{OUT_DIR}/images' + (f'_{SHARD}' if SHARD else '')
os.makedirs(img_dir, exist_ok=True)
rng = random.Random(SEED)


def layout_mol(mol, seed):
    """Coordinate generation for one depiction: CoordGen (default) or the
    zig-zag template with seed-retried layout QA (falls back to CoordGen
    when the molecule has no usable macroring / fails geometry gates)."""
    if LAYOUT == 'zigzag':
        from zigzag_layout import zigzag_coords
        laid = None
        for k in range(3):
            laid = zigzag_coords(mol, random.Random(seed + k), seed=seed + k)
            if laid is not None:
                return laid
    return coordgen_coords(mol)


def norm2(v):
    mm = math.hypot(*v) or 1.0
    return (v[0] / mm, v[1] / mm)


def rotv(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


MIN_DIST = 1.8


def organic_layout(mol, rng):
    """Closed zigzag walk with STRICT turn alternation (signs never repeat;
    magnitudes drift slowly; net turning exactly 360 deg) — residues stay
    evenly spaced with no local crowding, while the silhouette meanders.
    Side chains rotate rigidly onto the outward bisector with greedy clash
    avoidance — the ChemDraw patent-figure look."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pos = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n)]
    ring = max(mol.GetRingInfo().AtomRings(), key=len)
    m = len(ring)
    net = 720.0 / m
    base = rng.uniform(44, 60) + net / 2
    drift = rng.uniform(-8, 8)
    drift_step = rng.uniform(-1.5, 1.5)
    phi = rng.uniform(0, 360.0)
    dirs = []
    for k in range(m):
        mag = base + drift
        drift = max(-8.0, min(8.0, drift + drift_step))
        turn = (mag if k % 2 == 0 else -(mag - net)) + rng.uniform(-2.5, 2.5)
        phi += turn
        dirs.append(math.radians(phi))
    walk = [(0.0, 0.0)]
    x = y = 0.0
    for k in range(m):
        x += 1.5 * math.cos(dirs[k])
        y += 1.5 * math.sin(dirs[k])
        walk.append((x, y))
    walk = walk[:-1]
    cx = sum(p[0] for p in walk) / m
    cy = sum(p[1] for p in walk) / m
    walk = [(p[0] - cx, p[1] - cy) for p in walk]
    sx = rng.uniform(0.85, 1.2)
    sy = rng.uniform(0.85, 1.2)
    neighbors = {i: [] for i in range(n)}
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        neighbors[a].append(c)
        neighbors[c].append(a)
    anchor = {i: i for i in ring}
    order = deque(ring)
    seen = set(ring)
    while order:
        i = order.popleft()
        for j in neighbors[i]:
            if j not in seen:
                seen.add(j)
                anchor[j] = anchor[i]
                order.append(j)
    by_anchor = {}
    for j, a in anchor.items():
        if a != j:
            by_anchor.setdefault(a, []).append(j)
    placed = {}
    for k, i in enumerate(ring):
        wx, wy = walk[k]
        placed[i] = (wx * sx, wy * sy)
    rcx = sum(p[0] for p in placed.values()) / m
    rcy = sum(p[1] for p in placed.values()) / m
    deltas = [0, 18, -18, 36, -36, 60, -60, 85, -85, 110, -110, 140, -140, 180]
    for k, i in enumerate(ring):
        pi = placed[i]
        outward = norm2((pi[0] - rcx, pi[1] - rcy))
        members = by_anchor.get(i, [])
        if not members:
            continue
        j0 = members[0]
        u0 = norm2((pos[j0][0] - pos[i][0], pos[j0][1] - pos[i][1]))
        base_du = math.atan2(u0[1], u0[0]) - math.atan2(outward[1], outward[0])
        best = None
        for delta in deltas:
            du = base_du + math.radians(delta)
            cand = {}
            for j in members:
                rr = rotv((pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]), du)
                cand[j] = (pi[0] + rr[0], pi[1] + rr[1])
            clashes = 0
            for q in cand.values():
                for p in placed.values():
                    if math.dist(q, p) < MIN_DIST:
                        clashes += 1
                        break
            if best is None or clashes < best[0]:
                best = (clashes, du, cand)
            if clashes == 0:
                break
        _, du, cand = best
        placed.update(cand)
    for i in range(n):
        if i in placed:
            conf.SetAtomPosition(i, Point3D(placed[i][0], placed[i][1], 0.0))


def draw(mol, rng):
    """Style-aligned drawing (MolDraw2D options from the measured
    real-crop calibration in style_calibration.ALIGNED): serif label font,
    fixed bond length / font size, stroke and label-clearance ratios
    matched to the 255 verified real patent crops, plus print-gray ink.
    The previous hard-coded parameters (canvas 1400, line 3.6-4.6,
    font 32-46, sans font) are retained in style_calibration.BASELINE
    for the style study."""
    return draw_molecule(mol, ALIGNED, rng)


def frame_tight(img, rng):
    arr = np.array(img)
    ys, xs = np.where(arr < 200)
    pad = rng.randint(10, 45)
    return Image.fromarray(arr[max(0, ys.min() - pad):min(arr.shape[0], ys.max() + pad),
                               max(0, xs.min() - pad):min(arr.shape[1], xs.max() + pad)])


def _illumination_field(shape, rng, depth):
    """Low-frequency multiplicative illumination gradient (5-15%)."""
    h, w = shape
    small = (np.random.uniform(0.85, 1.0, (max(2, h // 200), max(2, w // 200)))).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=min(h, w) / 6)
    field = field / field.max() * (1.0 - depth) + depth
    return field


def finish(img, rng, path):
    """Patent-scan post-chain matched to literature + measured profiles
    (MolGrapher ablation: image-level augmentation is the highest-impact
    component; USPTO figures are 300-dpi bilevel CCITT-G4: crisp strokes,
    near-black ink, light sensor noise, occasional JPEG).

    Three regimes per figure: clean 30% / typical scan 50% / degraded
    photocopy 20%. Blur hard-capped at sigma 1.5 (thicker would merge
    double bonds); noise sigma capped at 9; no dithering (reads as
    spurious atoms); no rotation, no margin crop."""
    # regime parameters clipped to the MEASURED real envelope (this patent
    # family: bg 255, ink level 0-44, background noise sigma 1.1-4.8)
    regime = rng.choices(['clean', 'scan', 'photocopy'],
                         weights=[0.35, 0.5, 0.15], k=1)[0]
    blur_sigma = {'clean': rng.uniform(0.0, 0.4),
                  'scan': rng.uniform(0.3, 0.8),
                  'photocopy': rng.uniform(0.5, 1.2)}[regime]
    noise_sigma = {'clean': rng.uniform(1.0, 2.0),
                   'scan': rng.uniform(2.0, 3.6),
                   'photocopy': rng.uniform(3.2, 4.8)}[regime]
    ink_level = {'clean': rng.uniform(0.94, 1.0),
                 'scan': rng.uniform(0.9, 1.0),
                 'photocopy': rng.uniform(0.8, 0.93)}[regime]
    arr = np.array(img).astype(np.float32)
    if (arr < 128).mean() > 0.075:
        return False
    # ink gray-level: near-black core, gray for photocopy lineage
    arr = 255.0 - (255.0 - arr) * ink_level
    if regime != 'clean':
        depth = rng.uniform(0.05, 0.15)
        arr = arr * _illumination_field(arr.shape, rng, depth)
    if blur_sigma > 0.05:
        arr = cv2.GaussianBlur(arr, (0, 0), blur_sigma)
    arr = arr + np.random.normal(0.0, noise_sigma, arr.shape)
    if regime == 'photocopy' and rng.random() < 0.5:
        speck = np.random.random(arr.shape) < 0.0008
        arr[speck] = rng.uniform(40, 120)
    arr = np.clip(arr, 0, 255)
    jpeg_q = {'clean': None,
              'scan': int(rng.uniform(70, 92)),
              'photocopy': int(rng.uniform(55, 80))}[regime]
    if jpeg_q:
        ok_j, buf = cv2.imencode('.jpg', arr.astype(np.uint8),
                                 [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok_j:
            arr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return True


if __name__ == '__main__':
    rows = rejected = made = 0
    i = 0
    with open(f'{OUT_DIR}/synth{SHARD}.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file_path', 'SMILES'])
        while made < PAIRS and i < PAIRS * 4:
            i += 1
            size = rng.randint(16, 24)
            smi = generate_cyclopeptide(rng, size=size)
            if not smi or len(smi) > 1400:
                continue
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            mol = layout_mol(mol, SEED * 1000003 + rows)
            if mol is None:
                continue
            for variant in range(2):
                img = draw(mol, rng)
                path = f'{img_dir}/{rows:06d}.png'
                if finish(img, rng, path):
                    w.writerow([path, smi])
                    rows += 1
                else:
                    rejected += 1
            made += 1
            if made % 500 == 0:
                fh.flush()
                print(f"{made} pairs, {rows} rows, {rejected} rejected", flush=True)
    print(f"DONE: {made} pairs, {rows} rows -> {img_dir}")
    
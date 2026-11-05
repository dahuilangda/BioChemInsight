"""Raster style-metric extraction for patent structure drawings.

Measures the drawing style of a structure image (real patent crop or
synthetic render) directly from pixels, so real crops and RDKit renders
are scored by exactly the same code path:

  - bond angles           included angles between Hough line segments that
                          share an endpoint (junction vertices)
  - ring style proxies    per-image bond-length CV, junction max/min
                          incident-length ratio, concentration of angles
                          around 120 deg (regular-hexagon sharpness)
  - label placement       glyph height / bond length, clearance between
                          glyph ink and bond ink / bond length
  - stroke and ink        ink thickness at line-support pixels (distance
                          transform) / bond length, mean ink gray level,
                          ink fraction inside the ink bounding box
  - double bond style     gap between the parallel double-bond lines / bond
                          length, inner/outer line length ratio
  - abbreviation density  glyph clusters per bond segment (expanded
                          structures carry fewer text clusters than
                          Me/OMe-abbreviated ones)

All length-valued outputs are normalized by the image's median bond
length, so images of different dpi/crop scale are comparable.

Extraction pipeline: binarize -> connected components (glyph ink vs
structure ink) -> probabilistic-free Hough accumulator over (theta, rho)
-> line-support runs -> deduplicated segments -> junction graph.
Skeleton thinning is deliberately avoided: patent strokes are 1-2 px
anti-aliased lines, too thin for stable morphological thinning.

Dependencies: numpy + scipy only (no OpenCV / scikit-image).

Reference: training/molsight_markush/compare_render_style.py drives this
module for the real-vs-synthetic style alignment report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np
from PIL import Image
from scipy import ndimage

# --------------------------------------------------------------------------
# binarization
# --------------------------------------------------------------------------

INK_SMOOTH_SIGMA = 0.7       # pre-binarization grayscale smoothing
INK_THRESHOLD = 0.50         # on the smoothed [0,1] grayscale
GLYPH_DIAG_BOOT = 14.0       # px: component bbox diag floor for the
                             # bootstrap structure/glyph split
GLYPH_DIAG_FRAC = 0.75       # final: diag < frac * L  => glyph
DASH_W_H_MAX = 8.0           # glyph filter: drop very wide flat components
GLYPH_ELONGATION_MAX = 6.0   # glyph filter: PCA elongation ceiling
BOND_LEN_FLOOR_FRAC = 0.40   # segment shorter than frac * L => not a bond
BOND_LEN_CEIL_FRAC = 2.6     # segment longer than frac * L => not a bond
HOUGH_THETA_STEP = 0.5       # degrees
HOUGH_PEAK_EXCLUDE = (8, 3)  # NMS exclusion radius in (theta_idx, rho_px);
                             # rho radius must stay <= the double-bond gap so
                             # parallel inner lines are not suppressed
HOUGH_MAX_PEAKS = 400
LINE_DIST_TOL = 1.4          # px: pixel-to-line support distance
CORRIDOR_INK_THRESHOLD = 0.8  # smoothed-gray level counting as corridor ink
GHOST_PARALLEL_MAX_OFF = 6.0  # px: consider parallels up to this offset
RUN_GAP_MAX = 3.0            # px: max gap when clustering support pixels
RUN_DENSITY_MIN = 1.1        # support px per px of run span
COLLINEAR_GAP_MAX = 5.0      # px: restitch gap cap (rasterization breaks)
MIN_SEG_BOOT = 10.0          # px: bootstrap minimum run length
ANGLE_MAX_DEG = 165.0        # drop near-collinear "angles"
VERTEX_DIR_GROUP_DEG = 15.0  # incident ends closer than this are one stroke
JUNCTION_TOL_FRAC = 0.25     # endpoint match radius, frac of L (min 3 px)
ANGLE_NEAR_120 = 8.0         # +/- window counted as a regular hexagon angle
ANGLE_BAND = (100.0, 140.0)  # band over which angle dispersion is measured
DBL_MAX_SKEW_DEG = 6.0       # parallelism tolerance for double-bond pairs
DBL_OVERLAP_FRAC = 0.50      # projection overlap for double-bond pairs
DBL_GAP_RANGE = (0.12, 0.55)  # accepted gap/bond-length window


def load_ink(path: str, threshold: float | None = None):
    """Return (raw gray [0,1], smoothed gray [0,1], boolean ink mask).

    The mask comes from a lightly smoothed grayscale so anti-aliased
    1-2 px strokes binarize continuously."""
    threshold = INK_THRESHOLD if threshold is None else threshold
    gray = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    smooth = ndimage.gaussian_filter(gray, INK_SMOOTH_SIGMA)
    return gray, smooth, smooth < threshold


# --------------------------------------------------------------------------
# connected components: glyph vs structure ink
# --------------------------------------------------------------------------

def _components(ink: np.ndarray) -> list[dict]:
    lab, _ = ndimage.label(ink, structure=np.ones((3, 3)))
    out = []
    for li, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        out.append({"id": li, "slice": sl, "h": bh, "w": bw,
                    "diag": math.hypot(bh, bw),
                    "mask": lab[sl] == li})
    return out


def _elongation(comp: dict) -> float:
    """PCA elongation of a component's pixels (sqrt eigenvalue ratio).
    Line-like fragments (bond stubs, dashes) score high; text glyphs are
    compact blobs."""
    sl = comp["slice"]
    ys, xs = np.nonzero(comp["mask"])
    if len(ys) < 5:
        return 99.0
    pts = np.stack([xs + sl[1].start, ys + sl[0].start], 1).astype(float)
    pts -= pts.mean(0)
    cov = np.cov(pts.T)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-6, None)
    return float(math.sqrt(ev[1] / ev[0]))


def _glyph_filter(comps: list[dict], L: float) -> list[dict]:
    """Keep text-glyph components; drop line-like components (bond stubs,
    dashed table borders) and height outliers."""
    if not comps:
        return []
    cand = [c for c in comps
            if not (c["h"] <= 3 and c["w"] >= 0.5 * L)
            and c["w"] / max(c["h"], 1) < DASH_W_H_MAX
            and _elongation(c) < GLYPH_ELONGATION_MAX]
    if not cand:
        return []
    h_med = float(np.median([c["h"] for c in cand]))
    return [c for c in cand if c["h"] >= 0.35 * h_med]


def _cluster_glyphs(comps: list[dict]) -> list[dict]:
    """Merge glyph components with overlapping/touching boxes into label
    clusters (the N, H, 2 of one 'NH2')."""
    boxes = [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in comps]
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            h = min(ay1 - ay0, by1 - by0)
            if (ax0 <= bx1 and bx0 <= ax1 and
                    ay0 - 0.5 * h <= by1 and by0 - 0.5 * h <= ay1):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        out.append({"x0": min(boxes[i][0] for i in idx),
                    "y0": min(boxes[i][1] for i in idx),
                    "x1": max(boxes[i][2] for i in idx),
                    "y1": max(boxes[i][3] for i in idx),
                    "h": max(boxes[i][3] - boxes[i][1] for i in idx),
                    "n_glyphs": len(idx)})
    return out


# --------------------------------------------------------------------------
# Hough line-segment extraction
# --------------------------------------------------------------------------

@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    length: float
    votes: int

    @property
    def mid(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2])

    @property
    def direction(self) -> np.ndarray:
        d = np.array([self.x1 - self.x0, self.y1 - self.y0])
        return d / (np.linalg.norm(d) + 1e-9)

    @property
    def theta(self) -> float:
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0


def _hough_peaks(ys: np.ndarray, xs: np.ndarray, min_votes: float):
    """Accumulate a (theta, rho) Hough tabulation of ink pixels and return
    peak (theta, rho) after greedy non-maximum suppression."""
    thetas = np.arange(0.0, 180.0, HOUGH_THETA_STEP)
    diag = math.hypot(ys.max() - ys.min(), xs.max() - xs.min()) + 4
    nrho = int(2 * diag) + 4
    rho_off = nrho // 2
    acc = np.zeros((len(thetas), nrho), dtype=np.float32)
    for t0 in range(0, len(thetas), 64):
        chunk = thetas[t0:t0 + 64]
        tr = np.deg2rad(chunk)
        # rho = x cos + y sin  (image x=col, y=row)
        rho = (np.outer(xs, np.cos(tr)) + np.outer(ys, np.sin(tr)))
        ridx = np.rint(rho).astype(np.int32) + rho_off
        ok = (ridx >= 0) & (ridx < nrho)
        for k in range(len(chunk)):
            r = ridx[:, k][ok[:, k]]
            if r.size:
                acc[t0 + k] += np.bincount(r, minlength=nrho)
    # greedy NMS peak picking
    flat_order = np.argsort(acc.ravel())[::-1]
    taken = []
    accv = acc.ravel()
    for f in flat_order[: 40000]:
        v = accv[f]
        if v < min_votes or len(taken) >= HOUGH_MAX_PEAKS:
            break
        ti, ri = divmod(int(f), nrho)
        if any(abs(ti - t2) < HOUGH_PEAK_EXCLUDE[0] and
               ((abs(ri - r2) < HOUGH_PEAK_EXCLUDE[1]) or
                (abs(ri + nrho - r2) < HOUGH_PEAK_EXCLUDE[1]) or
                (abs(ri - nrho - r2) < HOUGH_PEAK_EXCLUDE[1]))
               for t2, r2 in taken):
            continue
        taken.append((ti, ri))
    return [(float(thetas[ti]), ri - rho_off) for ti, ri in taken]


def _runs_from_support(coords_sorted: np.ndarray, proj: np.ndarray,
                       min_len: float):
    """Cluster line-support pixels (ordered by projection) into runs split
    at gaps > RUN_GAP_MAX. A run must also be dense (support pixels per
    pixel of span >= RUN_DENSITY_MIN): grazing Hough lines that pick up
    scattered pixels from unrelated strokes fail this test."""
    if len(proj) == 0:
        return []
    splits = np.nonzero(np.diff(proj) > RUN_GAP_MAX)[0]
    bounds = np.concatenate([[0], splits + 1, [len(proj)]])
    runs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        span = proj[b - 1] - proj[a]
        if span >= min_len and (b - a) / span >= RUN_DENSITY_MIN:
            runs.append((coords_sorted[a], coords_sorted[b - 1], span))
    return runs


def _merge_collinear_adjacent(segs: list[Segment]) -> list[Segment]:
    """Restitch segments that continue one drawn line: near-equal direction,
    endpoints on the same support band, small end-to-end gap along the line.
    Zigzag chain vertices always turn, so true bonds are never collinear
    neighbours; only rasterization breaks one line into pieces."""
    out = list(segs)
    changed = True
    while changed:
        changed = False
        out.sort(key=lambda s: -s.length)
        for i in range(len(out)):
            for j in range(len(out)):
                if i == j:
                    continue
                a, b = out[i], out[j]
                if abs(((a.theta - b.theta + 90) % 180) - 90) > 4.0:
                    continue
                d, n = a.direction, np.array([-a.direction[1], a.direction[0]])
                rel0 = np.array([b.x0, b.y0]) - a.mid
                rel1 = np.array([b.x1, b.y1]) - a.mid
                if abs(float(rel0 @ n)) > 2.2 or abs(float(rel1 @ n)) > 2.2:
                    continue
                p0, p1 = float(rel0 @ d), float(rel1 @ d)
                gap = abs(p1 - p0)
                if gap > COLLINEAR_GAP_MAX:
                    continue
                # endpoints of b nearest/farthest along a's axis
                pts = {(a.x0, a.y0), (a.x1, a.y1)}
                far = (b.x0, b.y0) if abs(p0) > abs(p1) else (b.x1, b.y1)
                near_candidates = [(a.x0, a.y0), (a.x1, a.y1)]
                near = min(near_candidates, key=lambda p: math.hypot(
                    p[0] - (b.x0 if abs(p0) > abs(p1) else b.x1),
                    p[1] - (b.y0 if abs(p0) > abs(p1) else b.y1)))
                if near in pts and far not in pts:
                    new = Segment(x0=near[0], y0=near[1],
                                  x1=far[0], y1=far[1],
                                  length=math.hypot(far[0] - near[0],
                                                    far[1] - near[1]),
                                    votes=a.votes + b.votes)
                    out = [s for k, s in enumerate(out) if s is not a and s is not b]
                    out.append(new)
                    changed = True
                    break
            if changed:
                break
    return out


def _parallel_offset(a: Segment, b: Segment) -> float:
    """Perpendicular offset between near-parallel segments."""
    n = np.array([-a.direction[1], a.direction[0]])
    return abs(float(np.dot(b.mid - a.mid, n)))


def _is_inset(cand: Segment, ref: Segment) -> bool:
    """True when cand's extent is trimmed inside ref's extent on both ends
    — the classic double-bond inner line; anti-aliasing ridge ghosts of
    ref run ref's full length instead."""
    d = ref.direction
    half = ref.length / 2
    for ex, ey in ((cand.x0, cand.y0), (cand.x1, cand.y1)):
        p = float(np.dot(np.array([ex, ey]) - ref.mid, d))
        if p < -half + 1.0 or p > half - 1.0:
            return False
    return True


def _corridor_is_inked(a: Segment, b: Segment, smooth: np.ndarray) -> bool:
    """True when the space between two near-parallel segments is itself
    inked — i.e. b is an anti-aliasing ridge of the same fat stroke as a,
    not a genuine parallel line (a double-bond inner line has white space
    between the two strokes)."""
    d = a.direction
    n = np.array([-d[1], d[0]])
    tmid, bmid = a.mid, b.mid
    rel = bmid - tmid
    off = float(rel @ n)
    pa0 = float(np.dot(np.array([b.x0, b.y0]) - tmid, d))
    pa1 = float(np.dot(np.array([b.x1, b.y1]) - tmid, d))
    lo, hi = sorted((pa0, pa1))
    lo = max(lo, -a.length / 2 + 1)
    hi = min(hi, a.length / 2 - 1)
    if hi <= lo:
        return True
    h, w = smooth.shape
    samples = np.linspace(lo + 1, hi - 1, 12)
    inked = 0
    total = 0
    for tproj in samples:
        base = tmid + d * tproj
        for f in (0.35, 0.6, 0.8):
            px = int(round((base + n * off * f)[0]))
            py = int(round((base + n * off * f)[1]))
            if 0 <= px < w and 0 <= py < h:
                total += 1
                inked += smooth[py, px] < CORRIDOR_INK_THRESHOLD
    return total > 0 and inked / total > 0.35


def _hough_segments(ink: np.ndarray, min_len: float,
                    smooth: np.ndarray | None = None) -> list[Segment]:
    """Extract deduplicated straight segments from an ink mask.

    Peaks are processed in decreasing vote order; a peak whose support is
    mostly already claimed by an accepted segment is skipped, and a
    candidate running alongside an accepted segment with no white space
    between them is treated as a stroke-edge ghost and skipped."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return []
    peaks = _hough_peaks(ys.astype(np.float64), xs.astype(np.float64),
                         min_votes=min_len)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)  # (x, y)
    segments: list[Segment] = []
    claimed = np.zeros(len(pts), dtype=bool)
    for theta, rho in peaks:
        tr = math.radians(theta)
        n = np.array([math.cos(tr), math.sin(tr)])       # line normal
        d = np.array([-math.sin(tr), math.cos(tr)])      # line direction
        dist = np.abs(pts @ n - rho)
        sel = dist <= LINE_DIST_TOL
        if sel.sum() < min_len:
            continue
        if claimed.any() and (sel & ~claimed).sum() < 0.35 * sel.sum():
            continue
        proj = pts @ d
        order = np.argsort(proj)
        for pa, pb, span in _runs_from_support(pts[order][sel[order]],
                                               proj[order][sel[order]],
                                               min_len):
            cand = Segment(x0=pa[0], y0=pa[1], x1=pb[0], y1=pb[1],
                           length=float(span), votes=int(span))
            if smooth is not None:
                ghost = False
                for t in segments:
                    if abs(((cand.theta - t.theta + 90) % 180) - 90) > 3.0:
                        continue
                    off = _parallel_offset(t, cand)
                    if off > GHOST_PARALLEL_MAX_OFF:
                        continue
                    if off > 4.2:
                        # clear white space separates a real parallel line
                        if _corridor_is_inked(t, cand, smooth) or \
                                _corridor_is_inked(cand, t, smooth):
                            ghost = True
                            break
                    elif not _is_inset(cand, t):
                        # hugging the accepted stroke and running its full
                        # length: an anti-aliasing ridge, not an inner line
                        ghost = True
                        break
                if ghost:
                    continue
            segments.append(cand)
            # claim pixels near the accepted segment
            rel = pts - cand.mid
            perp = np.abs(rel @ np.array([-cand.direction[1],
                                          cand.direction[0]]))
            along = rel @ cand.direction
            claimed |= (perp <= LINE_DIST_TOL + 0.4) & \
                (np.abs(along) <= cand.length / 2 + 1.5)
    segments = _dedupe_segments(segments)
    return _merge_collinear_adjacent(segments)


def _dedupe_segments(segs: list[Segment]) -> list[Segment]:
    """Drop duplicate detections of the same physical line (near-equal
    direction, both endpoints lying on the kept segment's support band,
    overlapping projections), keeping the longest representative."""
    out: list[Segment] = []
    for s in sorted(segs, key=lambda s: -s.length):
        dup = False
        for t in out:
            if abs(((s.theta - t.theta + 90) % 180) - 90) > 7.0:
                continue
            d = t.direction
            n = np.array([-d[1], d[0]])
            ends_close = 0
            for ex, ey in ((s.x0, s.y0), (s.x1, s.y1)):
                for fx, fy in ((t.x0, t.y0), (t.x1, t.y1)):
                    if math.hypot(ex - fx, ey - fy) <= 3.5:
                        ends_close += 1
                        break
            if ends_close == 2:
                dup = True
                break
            on_line = all(
                abs(float(np.dot(np.array([ex, ey]) - t.mid, n))) <= 2.5
                for ex, ey in ((s.x0, s.y0), (s.x1, s.y1)))
            if on_line:
                a1 = float(np.dot(np.array([s.x0, s.y0]) - t.mid, d))
                a2 = float(np.dot(np.array([s.x1, s.y1]) - t.mid, d))
                lo, hi = sorted((a1, a2))
                half = t.length / 2
                overlap = min(hi, half) - max(lo, -half)
                if overlap >= 0.4 * min(s.length, t.length):
                    dup = True
                    break
        if not dup:
            out.append(s)
    return out


# --------------------------------------------------------------------------
# per-image measurement
# --------------------------------------------------------------------------

@dataclass
class StyleMeasurements:
    """Per-image style measurements (all lengths normalized by median
    bond length L unless suffixed _px)."""
    bond_len_median_px: float = 0.0
    bond_len_cv: float = 0.0              # std/mean of bond lengths
    junction_len_ratio: float = 0.0       # median max/min incident length
    angles_deg: list = field(default_factory=list)
    angle_120_frac: float = 0.0
    angle_band_std: float = 0.0
    n_bonds: int = 0
    n_angle_vertices: int = 0
    stroke_w_px: float = 0.0
    stroke_ratio: float = 0.0             # stroke_w_px / L
    n_glyph_clusters: int = 0
    glyph_height_ratio: float = 0.0       # median glyph height / L
    glyph_clearance_ratio: float = 0.0    # glyph-to-bond-ink distance / L
    labels_per_bond: float = 0.0
    dbl_gap_ratio: float = 0.0            # double-bond line gap / L
    dbl_inner_outer: float = 0.0          # inner/outer line length ratio
    dbl_pair_count: int = 0
    ink_frac: float = 0.0                 # ink fraction inside ink bbox
    ink_gray_mean: float = 0.0            # mean gray of ink pixels (0-1)
    scale_density: float = 0.0            # ink-bbox diagonal / L
    aspect: float = 0.0                   # bbox w / h

    def to_dict(self) -> dict:
        return asdict(self)


def measure_style(path: str) -> StyleMeasurements:
    """Measure drawing style of one image (real crop or synthetic render)."""
    gray, smooth, ink = load_ink(path)
    m = StyleMeasurements()
    if not ink.any():
        return m

    ys, xs = np.nonzero(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m.ink_frac = float(ink[y0:y1, x0:x1].mean())
    m.ink_gray_mean = float(gray[ink].mean())
    m.aspect = float((x1 - x0) / max(y1 - y0, 1))

    # ---- bootstrap pass on big components only ---------------------------
    comps = _components(ink)
    struct = [c for c in comps if c["diag"] >= GLYPH_DIAG_BOOT]
    if not struct:
        return m
    ink_struct = np.zeros_like(ink)
    for c in struct:
        ink_struct[c["slice"]] |= c["mask"]
    segs0 = _hough_segments(ink_struct, MIN_SEG_BOOT, smooth)
    if not segs0:
        return m
    L = float(np.median([s.length for s in segs0]))
    if L < 4:
        return m

    # ---- final classification with calibrated L --------------------------
    glyphs = [c for c in comps if c["diag"] < GLYPH_DIAG_FRAC * L]
    glyphs = _glyph_filter([dict(g, x0=g["slice"][1].start,
                                 y0=g["slice"][0].start,
                                 x1=g["slice"][1].stop,
                                 y1=g["slice"][0].stop)
                            for g in glyphs], L)
    bond_mask = ink.copy()
    for g in glyphs:
        sl = g["slice"]
        bond_mask[sl] &= ~g["mask"]
    clusters = _cluster_glyphs(glyphs)
    m.n_glyph_clusters = len(clusters)

    min_len = max(5.0, BOND_LEN_FLOOR_FRAC * L)
    segs = _hough_segments(bond_mask, min_len, smooth)
    if not segs:
        return m

    # ---- double-bond parallel pairs (before length filtering: inner lines
    # are legitimately shorter than L) -------------------------------------
    def seg_extent(s: Segment, d: np.ndarray):
        a = float(np.dot(np.array([s.x0, s.y0]) - s.mid, d))
        b = float(np.dot(np.array([s.x1, s.y1]) - s.mid, d))
        return min(a, b), max(a, b)

    def proj_overlap(s: Segment, t: Segment) -> float:
        d = s.direction
        slo, shi = seg_extent(s, d)
        tlo, thi = seg_extent(t, d)
        return min(shi, thi) - max(slo, tlo)

    segs_pair = [s for s in segs if s.length >= 0.35 * L]  # skip hash marks
    pairs = []
    for i in range(len(segs_pair)):
        for j in range(i + 1, len(segs_pair)):
            si, sj = segs_pair[i], segs_pair[j]
            dang = abs(((si.theta - sj.theta + 90) % 180) - 90)
            if dang > DBL_MAX_SKEW_DEG:
                continue
            p = _parallel_offset(si, sj)
            if not (DBL_GAP_RANGE[0] * L <= p <= DBL_GAP_RANGE[1] * L):
                continue
            if proj_overlap(si, sj) < DBL_OVERLAP_FRAC * min(si.length,
                                                             sj.length):
                continue
            pairs.append((si, sj, p))
    inner_ids = set()
    for si, sj, p in pairs:
        # the shorter of the pair is the trimmed inner line
        inner_ids.add(id(si) if si.length < sj.length else id(sj))

    # ---- bond set ---------------------------------------------------------
    bonds = [s for s in segs
             if BOND_LEN_FLOOR_FRAC * L <= s.length <= BOND_LEN_CEIL_FRAC * L
             and id(s) not in inner_ids]
    if len(bonds) < 5:
        return m
    lengths = np.asarray([s.length for s in bonds])
    L = float(np.median(lengths))
    m.bond_len_median_px = L
    m.n_bonds = len(bonds)
    m.bond_len_cv = float(lengths.std() / max(lengths.mean(), 1e-6))
    m.scale_density = float(math.hypot(x1 - x0, y1 - y0) / L)
    m.labels_per_bond = m.n_glyph_clusters / len(bonds)

    # ---- angles at junction vertices ---------------------------------------
    ends = []
    for bi, s in enumerate(bonds):
        ends.append((bi, 0, s.x0, s.y0))
        ends.append((bi, 1, s.x1, s.y1))
    tol = max(3.0, JUNCTION_TOL_FRAC * L)
    parent = list(range(len(ends)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            if ends[i][0] == ends[j][0]:
                continue
            if math.hypot(ends[i][2] - ends[j][2],
                          ends[i][3] - ends[j][3]) <= tol:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(ends)):
        groups.setdefault(find(i), []).append(i)
    angles, junc_ratios = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        cx = float(np.mean([ends[i][2] for i in idx]))
        cy = float(np.mean([ends[i][3] for i in idx]))
        dirs = []
        for i in idx:
            bi, _, ex, ey = ends[i]
            s = bonds[bi]
            other = (s.x1, s.y1) if (ex, ey) == (s.x0, s.y0) \
                else (s.x0, s.y0)
            d = np.array([other[0] - ex, other[1] - ey])
            dirs.append((d / (np.linalg.norm(d) + 1e-9), s.length))
        # group near-parallel incident ends (multi-traced strokes of the
        # same bond) and keep only the longest trace per direction
        order = sorted(range(len(dirs)), key=lambda k: -dirs[k][1])
        kept = []
        for k in order:
            if any(math.degrees(math.acos(abs(float(np.dot(dirs[k][0],
                   dirs[m][0]))))) < VERTEX_DIR_GROUP_DEG for m in kept):
                continue
            kept.append(k)
        dirs = [dirs[k] for k in kept]
        if len(dirs) < 2:
            continue
        if len(dirs) == 2:
            junc_ratios.append(max(dirs[0][1], dirs[1][1]) /
                               max(min(dirs[0][1], dirs[1][1]), 1e-6))
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                ang = math.degrees(math.acos(
                    float(np.clip(np.dot(dirs[i][0], dirs[j][0]), -1.0, 1.0))))
                if ang <= ANGLE_MAX_DEG:
                    angles.append(ang)
    m.angles_deg = [round(a, 1) for a in angles]
    m.n_angle_vertices = len(groups)
    if angles:
        arr = np.asarray(angles)
        m.angle_120_frac = float(((arr >= 120 - ANGLE_NEAR_120) &
                                  (arr <= 120 + ANGLE_NEAR_120)).mean())
        band = arr[(arr >= ANGLE_BAND[0]) & (arr <= ANGLE_BAND[1])]
        if len(band) >= 5:
            m.angle_band_std = float(band.std())
    if junc_ratios:
        m.junction_len_ratio = float(np.median(junc_ratios))

    # ---- stroke width at line-support pixels -------------------------------
    edt = ndimage.distance_transform_edt(bond_mask)
    thick = []
    gys, gxs = np.nonzero(bond_mask)
    gpts = np.stack([gxs, gys], axis=1).astype(np.float64)
    for s in bonds:
        d = s.direction
        n = np.array([-d[1], d[0]])
        rel = gpts - s.mid
        dist = np.abs(rel @ n)
        along = rel @ d
        sel = (dist <= LINE_DIST_TOL) & (np.abs(along) <= s.length / 2)
        if sel.any():
            thick.append(float(np.median(2.0 * edt[gys[sel], gxs[sel]])))
    if thick:
        m.stroke_w_px = float(np.median(thick))
        m.stroke_ratio = m.stroke_w_px / L

    # ---- glyph placement -----------------------------------------------------
    if clusters:
        edt_bond = ndimage.distance_transform_edt(~bond_mask)
        hgt, clear = [], []
        for g in clusters:
            hgt.append(g["h"])
            sl = (slice(max(0, g["y0"]), g["y1"]),
                  slice(max(0, g["x0"]), g["x1"]))
            sub = edt_bond[sl]
            if sub.size:
                clear.append(float(sub.min()))
        m.glyph_height_ratio = float(np.median(hgt)) / L
        if clear:
            m.glyph_clearance_ratio = float(np.median(clear)) / L

    # ---- double-bond style ----------------------------------------------------
    if pairs:
        m.dbl_gap_ratio = float(np.median([p for _, _, p in pairs]) / L)
        m.dbl_inner_outer = float(np.median(
            [min(si.length, sj.length) / max(si.length, sj.length)
             for si, sj, _ in pairs]))
        m.dbl_pair_count = len(pairs)
    return m


# --------------------------------------------------------------------------
# distribution comparison helpers
# --------------------------------------------------------------------------

def distribution_stats(real: np.ndarray, synth: np.ndarray, bins: int = 12):
    """Mean/std per side plus Wasserstein-1 distance and smoothed KL
    divergence (real || synth) over a shared histogram grid."""
    real = np.asarray(real, float)
    synth = np.asarray(synth, float)
    out = {"mean_real": float(real.mean()) if len(real) else None,
           "std_real": float(real.std()) if len(real) else None,
           "mean_synth": float(synth.mean()) if len(synth) else None,
           "std_synth": float(synth.std()) if len(synth) else None,
           "n_real": int(len(real)), "n_synth": int(len(synth))}
    if len(real) < 2 or len(synth) < 2:
        return out
    from scipy.stats import wasserstein_distance
    out["wasserstein"] = float(wasserstein_distance(real, synth))
    lo = min(real.min(), synth.min())
    hi = max(real.max(), synth.max())
    if hi - lo < 1e-12:
        out["kl_real_to_synth"] = 0.0
        return out
    edges = np.linspace(lo, hi, bins + 1)
    pr = np.histogram(real, bins=edges)[0].astype(float)
    ps = np.histogram(synth, bins=edges)[0].astype(float)
    eps = 1e-6
    pr = (pr / pr.sum()) + eps
    ps = (ps / ps.sum()) + eps
    pr /= pr.sum()
    ps /= ps.sum()
    out["kl_real_to_synth"] = float(np.sum(pr * np.log(pr / ps)))
    return out


def pooled_angle_stats(real_angles: list[list[float]],
                       synth_angles: list[list[float]],
                       bins: int = 18):
    """Histogram-level comparison over pooled junction angles."""
    ra = np.concatenate([np.asarray(a) for a in real_angles if len(a)]) \
        if any(len(a) for a in real_angles) else np.asarray([])
    sa = np.concatenate([np.asarray(a) for a in synth_angles if len(a)]) \
        if any(len(a) for a in synth_angles) else np.asarray([])
    out = distribution_stats(ra, sa, bins=bins)
    edges = np.linspace(0, 180, bins + 1)
    out["hist_real"] = (np.histogram(ra, bins=edges)[0]
                        / max(len(ra), 1)).round(4).tolist()
    out["hist_synth"] = (np.histogram(sa, bins=edges)[0]
                         / max(len(sa), 1)).round(4).tolist()
    return out

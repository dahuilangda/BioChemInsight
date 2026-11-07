"""Freehand-style irregular zig-zag macrocycle layout, v2.

Second-generation macroring layout that reproduces how patent draftsmen
actually draw macrocyclic peptides (measured on the verified real crops,
e.g. data/real_crops/US20240287138A1/page_024_box_0.png):

  * the drawn macroring is a CLOSED FREEHAND ZIG-ZAG chain: straight
    sawtooth runs of uneven length separated by polygon corners (two
    consecutive same-sign 60 deg turns, ~120 deg accumulated heading
    change) at irregular intervals — an irregular polygon, neither a
    circle nor a regular n-gon.  The base skeleton is the hex-lattice
    polyomino contour of zigzag_layout (v1; exact +/-60 turns, closed by
    construction, crossing-free) and the freehand irregularity comes
    from per-turn drift inside the 120 +/- ANGLE_TOL tolerance band plus
    the Newton position polish, all clipped to the same band;
  * wherever a small ring shares a long path with the macroring (the
    piperidine of seq53/68/77), the drawn cycle routes through the
    ring's SHORT path and the pendant machinery completes the small ring
    as a regular |R|-gon bulging to the better-clearance side — the ±60
    alphabet cannot reproduce a full polygon curl and still close the
    loop, and the crop's own embedded rings are visibly non-regular;
  * side chains are RIGID bodies (standard zig-zag geometry built once
    from the SMILES graph at exact bond length / 120 deg angles) placed
    OUTSIDE first, flipped INSIDE when the exterior slot is crowded, and
    rotated to intermediate angles — then micro-rotated — when both
    fail: the outside/inside/middle candidate order of the real crops,
    where most substituents point away from the ring but small ones
    (Ser-OH, ring hydroxyls) end up inside;
  * chirality is never edited: the wedge/dash marks at every C alpha are
    recomputed by the renderer from the chiral tags and the placed
    coordinates, so a side chain flipped from outside to inside carries
    its stereo-mark flip with it automatically; a molblock round-trip
    gate re-derives every stereocenter from the drawn wedges and compares
    isomeric canonical SMILES against the input before the layout is
    accepted;
  * rigidity: every backbone bond is exactly BOND long (odd macrorings
    carry a sub-pixel 0.02-bond closing-bond deviation — an exact
    closure is impossible on the ±60 alphabet for odd sizes — reported
    through the metrics), every backbone interior angle stays within
    120 +/- ANGLE_TOL degrees, side-chain internals are
    SMILES-translated standard geometry, and once a layout is accepted
    its coordinates are only ever transformed globally (rendering draws
    a copy).

Deviations from the sketch this module was specified with, both forced
by its own hard bars: the sketch's "+/-10 deg per-bond jitter" is
realized as +/-ANGLE_TOL (3 deg) drift (a wider jitter breaks the
"backbone angles 120 +/- 3 deg" acceptance bar), and residual crowding
after the repair passes is REPORTED (qa_pass flag + close-pair list in
the metrics) instead of discarding the depiction — the spec's "still
colliding -> report" behaviour; `layout_with_retry` still prefers a
seed whose layout is collision-free when one exists.

Spatial-utilisation tuning (visual review vs the real crops, round 2):

  * page scale: the drawn ring is RING_RADIUS_SCALE = 1.4x larger — the
    target radius is R0 = n_ring * L / (2*pi) * 1.4 (L = the 30 px base
    bond), so a 42-atom macroring opens a clearly visible interior
    cavity (chemists need to see the ring void);
  * the backbone generator now plans the turn sequence around a ROUND
    outline (corner signs biased to the 360 deg curl budget, corners
    spread evenly, candidates scored on rx/ry aspect ~ ASPECT_TARGET
    near 1:1 plus radial mean near the ideal circle) instead of the
    vertically stretched polyomino contours, and the accepted chain is
    rotated wide-side-up;
  * every side-chain body is scored with a RADIAL OUTWARD SPRING
    (R_out = R0 * OUT_SPRING = R0 * 1.3): candidate angles that extend
    the subtree toward R_out win and angles reaching into the ring
    interior are penalised, so side chains no longer squeeze inward.

Fused-ring chemistry fix found by the same review: Chem.FastFindRings
misses the small rings of fused systems (an indole's 5- and 6-rings
collapse into the 9-cycle perimeter), which made `_place_full_ring`
draw the whole fused bicycle as one regular 9-gon with the fusion bond
stretched to a 5-step chord (~2.9x bond).  Ring info from parsing
(SSSR) is now kept, and `_unplaced_ring` accepts rings that already
carry OWN-BODY anchors, completing them through the regular-polygon
arc machinery — fused aryls draw as pentagon+hexagon sharing the
fusion bond, every edge at exact bond length.

Entry points: `zigzag_v2_coords()` / `layout_with_retry()` (layout,
returns molecule + report), `render_v2()` (ALIGNED-style PNG, backbone
bonds at exactly BOND_PX * RING_RADIUS_SCALE = 42 px after the render
scaling), `validation_suite()` (seq / KRAS / GPC3 renders + real-crop
comparison sheets).
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

from style_calibration import ALIGNED, apply_profile, draw_molecule

# ---------------------------------------------------------------------------
# layout constants
# ---------------------------------------------------------------------------

BOND = 30.0                   # backbone bond length, layout units
BOND_PX = 30.0                # base bond spec; page bond = BOND_PX * RING_RADIUS_SCALE
RING_RADIUS_SCALE = 1.4       # page scale-up: R0 = n_ring * L / (2*pi) * 1.4
OUT_SPRING = 1.3              # side-chain outward spring: R_out = R0 * 1.3
SPRING_W = 1.0                # outward-spring weight in the placement score
INSIDE_W = 0.6                # inward-reach penalty weight (ring-cavity guard)
R_IN_FLOOR_FRAC = 0.80        # side chains stay beyond 0.80 * R0_ideal
ASPECT_TARGET = 1.15          # backbone rx/ry target (wide-flat, ~1:1)
ANGLE_TOL = 3.0               # hard bar: backbone interior angle 120 +/- tol
ZIG_TURN = 60.0               # heading turn at every zig-zag vertex
CORNER_GAP_CHOICES = (3, 4, 5)  # free atoms between polygon corners
FULL_TURN_BUDGET = 360.0      # signed curvature of a simple closed loop

FLIP_THRESHOLD = 0.80 * BOND  # clearance below which a body leaves its slot
ACCEPT_CLEARANCE = 0.55 * BOND
QA_FLOOR = 0.30 * BOND        # hard gate (collapse check)
MIN_ATTACH_ANGLE = 55.0       # side-chain bond vs the ring bonds at its root
MAX_REPAIR_ROUNDS = 3
CLOSURE_TOL = 3e-8            # position closure residual, bond units
                              # (keeps the closing bond at exact length)
MAX_CROSSINGS = 0             # v2 bar: the accepted layout is crossing-free

# candidate first-bond directions at an attachment vertex, most-preferred
# first (degrees relative to the exterior bisector): 0 = outside (standard
# 120/120 geometry), 180 = flipped inside, then the intermediate angles
CANDIDATE_DELTAS = (0.0, 180.0, 30.0, -30.0, 60.0, -60.0)


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------

def _rot(vec: tuple[float, float], deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c)


def _norm(vec: tuple[float, float]) -> tuple[float, float]:
    l = math.hypot(*vec)
    return (vec[0] / l, vec[1] / l) if l > 1e-9 else (1.0, 0.0)


def _seg_cross(p, q, r, s) -> bool:
    def orient(u, v, w):
        return ((v[0] - u[0]) * (w[1] - u[1])
                - (v[1] - u[1]) * (w[0] - u[0]))
    d1, d2 = orient(p, q, r), orient(p, q, s)
    d3, d4 = orient(r, s, p), orient(r, s, q)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _angle_between(u: tuple[float, float], v: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(dot))


def _adjacency(mol: Chem.Mol) -> list[set[int]]:
    n = mol.GetNumAtoms()
    adj = [set() for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _is_carbonyl_carbon(mol: Chem.Mol, idx: int) -> bool:
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() != 6:
        return False
    for b in a.GetBonds():
        other = b.GetOtherAtomIdx(idx)
        if (b.GetBondType() == Chem.BondType.DOUBLE
                and mol.GetAtomWithIdx(other).GetAtomicNum() == 8):
            return True
    return False


def _carbonyl_oxygen(mol: Chem.Mol, idx: int) -> int | None:
    a = mol.GetAtomWithIdx(idx)
    for b in a.GetBonds():
        other = b.GetOtherAtomIdx(idx)
        if (b.GetBondType() == Chem.BondType.DOUBLE
                and mol.GetAtomWithIdx(other).GetAtomicNum() == 8):
            return other
    return None


def find_macroring(mol: Chem.Mol) -> list[int]:
    """Atom cycle of the macroring: the largest ring; ties broken toward
    the ring carrying the most amide carbonyls."""
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        raise ValueError("molecule has no ring")
    best = sorted(rings,
                  key=lambda r: (len(r),
                                 sum(1 for i in r
                                     if _is_carbonyl_carbon(mol, i))),
                  reverse=True)[0]
    return list(best)


def ordered_ring(mol: Chem.Mol, atom_ids: list[int]) -> list[int]:
    """Walk the cycle in bond order starting from atom_ids[0]."""
    n = len(atom_ids)
    aset = set(atom_ids)
    cur, prev = atom_ids[0], None
    order = [cur]
    while len(order) < n:
        nxts = [b.GetOtherAtomIdx(cur)
                for b in mol.GetAtomWithIdx(cur).GetBonds()
                if b.GetOtherAtomIdx(cur) in aset
                and b.GetOtherAtomIdx(cur) != prev]
        if not nxts:
            raise ValueError("ring walk failed")
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


# ---------------------------------------------------------------------------
# backbone: embedded ring chords, turn-sequence generation, closure
# ---------------------------------------------------------------------------

def _turns_from_coords(pts: list[tuple[float, float]]) -> list[float]:
    """Signed heading turn at every vertex of a closed polygon (deg)."""
    n = len(pts)
    heads = []
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        heads.append(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
    turns = []
    for k in range(n):
        d = heads[k] - heads[k - 1]
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        turns.append(d)
    return turns


def _detect_path_rings(mol: Chem.Mol, order: list[int]
                       ) -> list[dict]:
    """Embedded rings whose atoms are ALL macroring atoms and contiguous
    along the macroring cycle (the piperidine of seq53/68/77).

    The macroring passes through |R|-1 of R's bonds; drawing that shared
    path as zig-zag makes R's closing bond cross its own path (a zig-zag
    oscillates across any straight chord), and drawing it as a regular
    ±60 polygon arc consumes 300 degrees of curl no 39-bond ±60 tail can
    balance.  The depiction therefore routes the drawn macrocycle through
    R's SHORT path — the shared path collapses to the single chord edge,
    which the proline-class pendant machinery then completes into a
    regular hexagon bulging to the better-clearance side (the way the
    real crops draw small rings on the boundary).  Returns per-ring dicts
    with the atom lists to drop from the drawn cycle."""
    pos = {a: i for i, a in enumerate(order)}
    aset = set(order)
    n = len(order)
    out = []
    for ring in mol.GetRingInfo().AtomRings():
        size = len(ring)
        if size < 5 or size >= n:
            continue
        if not all(a in aset for a in ring):
            continue
        idxs = sorted(pos[a] for a in ring)
        gaps = [(idxs[(k + 1) % size] - idxs[k]) % n for k in range(size)]
        big = max(range(size), key=lambda k: gaps[k])
        if gaps[big] != n - size + 1:
            continue                     # not one contiguous run
        # the run in cyclic order: idxs rotated past the big gap
        members = idxs[big + 1:] + idxs[:big + 1]
        out.append({"atoms": [order[m] for m in members[1:-1]],
                    "chord_atoms": (order[members[0]],
                                    order[members[-1]]),
                    "size": size,
                    "ring": tuple(sorted(ring))})
    return out


def _positions_from_turns(n: int, turns: list[float], start_heading: float,
                          bond: float) -> list[tuple[float, float]]:
    """Open walk: atom i at the head of bond i; bond i heads from
    start_heading accumulating the turns at vertices 1..i."""
    pts = [(0.0, 0.0)]
    h = start_heading
    for i in range(n):
        pts.append((pts[-1][0] + bond * math.cos(math.radians(h)),
                    pts[-1][1] + bond * math.sin(math.radians(h))))
        h += turns[(i + 1) % n]
    return pts


def _close_positions(n: int, turns: list[float], bond: float,
                     rng: random.Random,
                     gap_tol: float | None = None,
                     bound_scale: float = 1.0
                     ) -> list[tuple[float, float]] | None:
    """Newton correction of the turns so the chain closes.

    Position closure is a damped two-equation least squares over the
    turns (Jacobian per-degree to match), each step projected to keep
    the heading budget (sum of turns == 360) and clipped back into the
    sign band (|turn| within [ZIG-ANGLE_TOL, ZIG+ANGLE_TOL] times
    `bound_scale`, sign preserved); a backtracking line search re-scores
    the clipped step because bounds can absorb part of the move.  Odd
    macrorings run a homotopy: a wide-band phase first pulls the open
    gap down, then the strict +/-ANGLE_TOL band polishes from the near
    solution — only in-band results can ever be accepted, so the rigidity
    contract is unaffected."""
    work = list(turns)
    heading = rng.uniform(0.0, 360.0)

    def heading_pass(w, scale):
        """Re-centre the heading budget inside the band (the clip can
        absorb asymmetric parts of a projected step and drift the sum of
        turns; this redistributes the residual through each turn's
        directional capacity, so it stays in band)."""
        residual = FULL_TURN_BUDGET - sum(w)
        if abs(residual) < 1e-9:
            return w
        cap0 = ANGLE_TOL * scale
        caps = [(cap0 - (t - math.copysign(ZIG_TURN, t)))
                if residual > 0 else
                (cap0 + (t - math.copysign(ZIG_TURN, t)))
                for t in w]
        tot = sum(caps)
        if tot < 1e-9 or abs(residual) > tot:
            return None
        return [t + residual * (c / tot) for t, c in zip(w, caps)]

    def solve(work, scale, max_iter):
        bounds = []
        cap0 = ZIG_TURN + ANGLE_TOL * scale
        cap1 = ZIG_TURN - ANGLE_TOL * scale
        for t in work:
            lo, hi = sorted((math.copysign(cap1, t),
                             math.copysign(cap0, t)))
            bounds.append((lo, hi))
        gap = float("inf")
        for _ in range(max_iter):
            pts = _positions_from_turns(n, work, heading, bond)
            rx, ry = pts[-1]
            gap = math.hypot(rx, ry)
            if gap < CLOSURE_TOL * bond:
                break
            # d(end)/d(turn_k) = bond * sum_{bonds i >= k} perp(u_i)
            # (per-radian geometry, scaled to per-degree to match turns)
            cols = np.zeros((2, n))
            hs, h = [], heading
            for _i in range(n):
                hs.append(h)
                h += work[(_i + 1) % n]
            us = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                  for a in hs]
            accx = accy = 0.0
            for k in range(n - 1, -1, -1):
                accx += -us[k][1]
                accy += us[k][0]
                cols[0, k], cols[1, k] = accx * bond, accy * bond
            cols[:, 0] = 0.0         # turn 0 affects no open-chain bond
            jtj = cols @ cols.T + np.eye(2) * 1e-6
            delta = cols.T @ np.linalg.solve(jtj, -np.array([rx, ry]))
            delta *= 180.0 / math.pi     # radians -> degrees
            delta -= delta.sum() / n     # keep the heading budget intact
            best_work, best_gap = None, gap
            for frac in (1.0, 0.5, 0.25, 0.12):
                cand = list(work)
                for k in range(n):
                    lo, hi = bounds[k]
                    cand[k] = max(lo, min(hi,
                                          work[k] + frac * float(delta[k])))
                pts2 = _positions_from_turns(n, cand, heading, bond)
                g2 = math.hypot(pts2[-1][0], pts2[-1][1])
                if g2 < best_gap - 1e-12:
                    best_work, best_gap = cand, g2
                    if frac == 1.0:
                        break
            if best_work is None:
                break                    # no improving clipped step
            work = best_work
            w2 = heading_pass(work, scale)
            if w2 is None:
                break
            work = w2
        return work, gap

    tol = gap_tol if gap_tol is not None else CLOSURE_TOL * bond
    if bound_scale == 1.0:
        work, _gap = solve(work, 1.0, 200)
    else:
        # direct strict pass first; only when it fails, the homotopy
        # (wide band pulls the open gap down, strict band polishes —
        # only in-band results can ever be accepted)
        work, gap = solve(work, 1.0, 200)
        if gap >= tol:
            w_wide, _ = solve(work, bound_scale, 120)
            w_polish, gap_p = solve(w_wide, 1.0, 200)
            if gap_p < gap:
                work = w_polish
    pts = _positions_from_turns(n, work, heading, bond)
    if math.hypot(pts[-1][0], pts[-1][1]) < tol \
            and abs(FULL_TURN_BUDGET - sum(work)) < 0.05 \
            and all(abs(abs(t) - ZIG_TURN) <= ANGLE_TOL + 1e-9
                    for t in work):
        return pts[:-1]
    return None


def _backbone_crossings(pts: list[tuple[float, float]]) -> int:
    n = len(pts)
    cnt = 0
    for a in range(n):
        p, q = pts[a], pts[(a + 1) % n]
        for b in range(a + 1, n):
            if b == a or (a + 1) % n == b or (b + 1) % n == a:
                continue             # cycle-adjacent bonds share a vertex
            r, s = pts[b], pts[(b + 1) % n]
            if _seg_cross(p, q, r, s):
                cnt += 1
    return cnt


def _half_turns(m: int, rng: random.Random,
                target: float = 180.0, n_corners: int = 2,
                s0: float = 1.0) -> list[float] | None:
    """One half of a 2-fold-symmetric turn sequence: +60/-60 alternation
    with base sign `s0`, CONVEX corner pairs at opposite sites, then the
    half sum pushed to `target` through the per-turn directional
    capacity of the +/-ANGLE_TOL band.

    Corner curl is +120 deg per pair (the corner pair runs OPPOSITE to
    the alternation base when s0 = -1, so two corners still land near
    180: base -60 + 240 = 180).  Convex-only corners are what keeps the
    drawn ring ROUND and its cavity open: same-sign (inward) corners
    dent the outline and were the reason the earlier layouts crumpled to
    ~0.6x of the ideal circle radius with no visible interior void (now
    ~0.76x radius, ~0.5x cavity)."""
    turns = [(s0 * ZIG_TURN if i % 2 == 0 else -s0 * ZIG_TURN)
             for i in range(m)]
    starts: list[int] = []
    occupied: set[int] = set()
    for k in range(n_corners):
        site = int(round((k + rng.uniform(0.15, 0.85)) * m
                         / n_corners)) % m
        # corner vertex pairs {site, site+1} must not overlap
        while any((site + d) % m in occupied for d in (-1, 0, 1, 2)):
            site = (site + 1) % m
        occupied.update({site, (site + 1) % m})
        starts.append(site)
    for site in starts:
        turns[site] = ZIG_TURN
        turns[(site + 1) % m] = ZIG_TURN
    # freehand drift inside the tolerance band, then the corner-curl
    # residual absorbed through the per-turn directional capacity —
    # every turn stays provably inside +/-ANGLE_TOL of its sign
    turns = [t + rng.uniform(-ANGLE_TOL, ANGLE_TOL) for t in turns]
    residual = target - sum(turns)
    caps = [(ANGLE_TOL - (t - math.copysign(ZIG_TURN, t)))
            if residual > 0 else
            (ANGLE_TOL + (t - math.copysign(ZIG_TURN, t)))
            for t in turns]
    tot_cap = sum(caps)
    if tot_cap < 1e-9 or abs(residual) > tot_cap:
        return None
    return [t + residual * (c / tot_cap) for t, c in zip(turns, caps)]


def _round_turns(n: int, rng: random.Random) -> list[float] | None:
    """Turn sequence of a ROUND wavy ring, closed near-exactly.

    Even n: the sequence is 2-fold symmetric, t_{k+m} = t_k with
    m = n//2 and half curl exactly +/-180 — any window of m consecutive
    turns sums to the half curl, so bond i+m travels at
    heading(bond i) +/- 180 and every displacement pair cancels: the
    chain closes EXACTLY, with the symmetric oval look of a draftsman's
    macroring.

    Odd n: the free last turn (+/-60) completes the budget, so the half
    targets sit at 150/165/195/210 and the pairing carries a matching
    mismatch; the ladder below keeps BOTH the band absorption light
    (the band must keep Newton freedom) and the mismatch small.  Uneven
    sawtooth runs, jittered corner sites and the +/-3 deg drift keep
    the half irregular — round without being a regular polygon (the
    round-2 fix for the vertically stretched, cavity-less polyomino
    layouts)."""
    m = n // 2
    if n % 2 == 0:
        # round plan first (two corners, 30 deg absorption), then the
        # exact-180 plan (absorption = drift only) for tight small halves
        plans = ([(210.0, 2, -1.0), (150.0, 2, -1.0), (180.0, 2, -1.0)]
                 if m % 2 == 1 else
                 [(210.0, 2, 1.0), (150.0, 2, 1.0), (180.0, 2, 1.0)])
        rng.shuffle(plans)
        half = None
        for target, k, s0 in plans:
            half = _half_turns(m, rng, target=target, n_corners=k, s0=s0)
            if half is not None:
                break
        if half is None:
            return None
        turns = half + half
    else:
        m_odd = (m % 2 == 1)
        if m_odd:
            # odd half: base curl 60 + one aligned corner = 180 exact;
            # 165/195 add a 15 deg absorption and a 15 deg pair mismatch
            plans = [(195.0, 1, 1.0), (165.0, 1, 1.0),
                     (180.0, 1, 1.0), (210.0, 1, 1.0), (150.0, 1, 1.0)]
        else:
            # even half: two corners = 240; 150/210 keep the absorption
            # at 30 deg; 180 costs 60 (only for roomy bands)
            plans = [(210.0, 2, 1.0), (150.0, 2, 1.0), (180.0, 2, 1.0)]
        rng.shuffle(plans)
        half = None
        for target, k, s0 in plans:
            half = _half_turns(m, rng, target=target, n_corners=k, s0=s0)
            if half is not None:
                break
        if half is None:
            return None
        s = 1.0 if target < 180.0 else -1.0
        turns = half + half + [s * ZIG_TURN]
    if rng.random() < 0.5:                 # depiction curl mirror (2D only)
        turns = [-t for t in turns]
    return turns


def _ring_shape_score(pts: list[tuple[float, float]], n: int) -> float:
    """Roundness score of a closed chain: rx/ry aspect near
    ASPECT_TARGET (wide-flat ~1:1), mean radius near the ideal circle
    radius n*BOND/(2*pi) (the zigzag alphabet realises ~0.85x of it),
    low radial spread."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if h > w:                               # wide-side-up (rx >= ry)
        pts = [(p[1], -p[0]) for p in pts]
        w, h = h, w
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    ds = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r_mean = sum(ds) / n
    r_ideal = n * BOND / (2.0 * math.pi)
    circ = math.sqrt(sum((d - r_mean) ** 2 for d in ds) / n) / max(r_mean, 1e-9)
    aspect_pen = abs((w / max(h, 1e-9)) - ASPECT_TARGET)
    radius_pen = max(0.0, 0.80 - r_mean / r_ideal) * 6.0
    cavity_pen = max(0.0, 0.40 - min(ds) / r_ideal) * 3.0
    return 1.0 - aspect_pen * 1.5 - radius_pen - cavity_pen - circ * 1.0


def build_backbone(n: int, rng: random.Random,
                   chord_idx_pairs: list[tuple] | None = None
                   ) -> list[tuple[float, float]] | None:
    """Closed irregular zig-zag polygon: exact bond length BOND, every
    interior angle within 120 +/- ANGLE_TOL, crossing-free.

    Base skeleton: `_round_turns` — the ±60 sawtooth with same-sign
    corner pairs budgeted to the 360 deg curl and spread evenly, closed
    exactly by the 2-fold turn pairing (odd sizes leave a ~1-bond gap to
    the Newton pass).  Candidates are scored by `_ring_shape_score`
    (wide-flat aspect ~1:1, mean radius near n*BOND/(2*pi), visible
    interior cavity) and the best crossing-free chain wins — the round-2
    spatial-utilisation bar that replaced the folded polyomino
    contours."""
    best: tuple[float, list[tuple[float, float]]] | None = None
    fallback: list[tuple[float, float]] | None = None
    # tiny drawn cycles: the C2 half capacity cannot absorb the corner
    # curl and the small ring packs tighter than the round-oval gain is
    # worth — the round-1 polyomino net handles them directly
    if n <= 16:
        poly = _backbone_polyomino(n, rng, chord_idx_pairs)
        if poly is not None:
            return poly
    n_attempts = 160 if n % 2 else 40
    for _ in range(n_attempts):
        turns = _round_turns(n, rng)
        if turns is None:
            continue
        if n % 2:
            # odd macroring: the unpaired turn completes the 360 deg
            # budget exactly (half targets 150/210), leaving only a small
            # position gap.  Cheap pre-filter: the Newton band freedom is
            # ~0.7-1.6 deg per turn on these chains, so open gaps past
            # ~2 bonds never close — skip them without the expensive pass
            g0 = math.hypot(*_positions_from_turns(n, turns, 0.0, BOND)[-1])
            if g0 > 2.0 * BOND:
                continue
            work = turns
        else:
            work = turns               # closed exactly by the C2 pairing
        pts2 = _close_positions(n, work, BOND, rng,
                                gap_tol=0.02 * BOND if n % 2 else None,
                                bound_scale=2.5 if n % 2 else 1.0)
        if pts2 is None:
            continue
        if _backbone_crossings(pts2) > 0:
            continue
        if any(abs(abs(t) - ZIG_TURN) > ANGLE_TOL + 1e-9 for t in work):
            continue
        # embedded-ring anchors: after the short-path reroute the anchors
        # are ADJACENT drawn-cycle vertices and the small ring completes
        # as a regular |R|-gon arc on one side of their bond — PATH-ring
        # chords (hard) reject contours whose local cycle geometry wraps
        # around the chord so tightly that BOTH sides collapse onto cycle
        # atoms; pendant-completion chords (soft) only penalise the shape
        # score, so one tight proline chord never discards an otherwise
        # round contour
        arc_pen = 0.0
        arc_ok = True
        for ch in (chord_idx_pairs or []):
            i, j, ring_size = ch[0], ch[1], ch[2]
            hard = ch[3] if len(ch) > 3 else True
            best_local = _arc_min_clearance(pts2, i, j, ring_size, n)
            if best_local is None:
                continue
            if best_local < 0.45:
                if hard:
                    arc_ok = False
                    break
                arc_pen += 0.45 - best_local
        if not arc_ok:
            # keep the first crossing-free contour as a fallback: a
            # crowded embedded-ring anchor is reported through the layout
            # metrics instead of discarding the whole depiction
            if fallback is None and chord_idx_pairs:
                fallback = pts2
            continue
        # wide-side-up before scoring (rx >= ry, the patent wide-flat look)
        ys = [p[1] for p in pts2]
        xs = [p[0] for p in pts2]
        if max(ys) - min(ys) > max(xs) - min(xs):
            pts2 = [(p[1], -p[0]) for p in pts2]
        score = _ring_shape_score(pts2, n) - 0.5 * arc_pen
        if best is None or score > best[0]:
            best = (score, pts2)
    if best is not None:
        return best[1]
    return _backbone_polyomino(n, rng, chord_idx_pairs)


def _backbone_polyomino(n: int, rng: random.Random,
                        chord_idx_pairs: list[tuple] | None = None
                        ) -> list[tuple[float, float]] | None:
    """Safety net for sizes the C2 pairing cannot absorb (very small
    halves whose directional capacity is below the 30 deg corner-curl
    residual): the round-1 hex-lattice polyomino contour of zigzag_layout
    (v1) with per-turn drift inside the tolerance band and the Newton
    position polish — closed by construction, crossing-free, but without
    the round-2 oval shaping."""
    from zigzag_layout import select_template

    size = n if n % 2 == 0 else n + 1
    fallback: list[tuple[float, float]] | None = None
    for _ in range(24):
        poly = select_template(size, rng)
        if poly is None:
            return None
        path = poly.contour()
        if rng.random() < 0.5:
            path = path[::-1]
        start = rng.randrange(len(path))
        path = path[start:] + path[:start]
        pts = [poly.vertex_xy(v) for v in path]
        if rng.random() < 0.5:               # depiction mirror (2D only)
            pts = [(-x, y) for x, y in pts]
        turns = _turns_from_coords(pts)
        if sum(turns) < 0:                   # normalize the curl direction
            turns = [-t for t in turns]
        if any(abs(abs(t) - ZIG_TURN) > 0.5 for t in turns):
            continue                         # pentagon jump: not pure zigzag
        if size != n:
            # odd macroring: drop ONE vertex from the even contour and
            # search pop index/composition for the smallest open gap
            base = turns
            best: tuple[float, list[float]] | None = None
            for _try in range(14):
                k = rng.randrange(len(base))
                cand_turns = base[:k] + base[k + 1:]
                cand_turns = [t + rng.uniform(-ANGLE_TOL, ANGLE_TOL)
                              for t in cand_turns]
                residual = FULL_TURN_BUDGET - sum(cand_turns)
                caps = [(ANGLE_TOL - (t - math.copysign(ZIG_TURN, t)))
                        if residual > 0 else
                        (ANGLE_TOL + (t - math.copysign(ZIG_TURN, t)))
                        for t in cand_turns]
                tot_cap = sum(caps)
                if tot_cap < 1e-9 or abs(residual) > tot_cap:
                    continue
                cand_work = [t + residual * (c / tot_cap)
                             for t, c in zip(cand_turns, caps)]
                heading = rng.uniform(0.0, 360.0)
                pts_walk = _positions_from_turns(n, [0.0] + cand_work,
                                                 heading, BOND)
                gap = math.hypot(pts_walk[-1][0], pts_walk[-1][1])
                if best is None or gap < best[0]:
                    best = (gap, cand_work)
                if gap < 0.4 * BOND:
                    break
            if best is None or best[0] > 1.1 * BOND:
                continue
            work = best[1]
            odd = True
        else:
            odd = False
            turns = [t + rng.uniform(-ANGLE_TOL, ANGLE_TOL) for t in turns]
            residual = FULL_TURN_BUDGET - sum(turns)
            caps = [(ANGLE_TOL - (t - math.copysign(ZIG_TURN, t)))
                    if residual > 0 else
                    (ANGLE_TOL + (t - math.copysign(ZIG_TURN, t)))
                    for t in turns]
            tot_cap = sum(caps)
            if tot_cap < 1e-9 or abs(residual) > tot_cap:
                continue
            work = [t + residual * (c / tot_cap)
                    for t, c in zip(turns, caps)]
        pts2 = _close_positions(n, work, BOND, rng,
                                gap_tol=0.02 * BOND if odd else None)
        if pts2 is None:
            continue
        if _backbone_crossings(pts2) > 0:
            continue
        if any(abs(abs(t) - ZIG_TURN) > ANGLE_TOL + 1e-9 for t in work):
            continue
        if not _chords_fit(pts2, n, chord_idx_pairs):
            if fallback is None and chord_idx_pairs:
                fallback = pts2
            continue
        ys = [p[1] for p in pts2]
        xs = [p[0] for p in pts2]
        if max(ys) - min(ys) > max(xs) - min(xs):
            pts2 = [(p[1], -p[0]) for p in pts2]
        return pts2
    return fallback


def _arc_min_clearance(pts2: list[tuple[float, float]], i: int, j: int,
                       ring_size: int, n: int) -> float | None:
    """Min clearance (bond units) of the regular |R|-gon completion arc
    over chord (i, j) against the drawn cycle, best side wins; None when
    the chord is shorter than the arc's own radius gate."""
    a, b = pts2[i], pts2[j]
    edges = ring_size - 1
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    radius = (chord / 2.0) / math.sin(math.pi * edges / ring_size)
    if chord / 2.0 > radius + 1e-9:
        return None
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    perp = _norm((-(b[1] - a[1]), b[0] - a[0]))
    hh = math.sqrt(max(radius * radius - chord * chord / 4.0, 0.0))
    best_local = -1.0
    for side in (1.0, -1.0):
        center = (mid[0] + perp[0] * hh * side,
                  mid[1] + perp[1] * hh * side)
        ang0 = math.atan2(a[1] - center[1], a[0] - center[0])
        ang1 = math.atan2(b[1] - center[1], b[0] - center[0])
        delta = (ang1 - ang0) % (2.0 * math.pi)
        if delta > math.pi:
            delta -= 2.0 * math.pi
        sweep_mag = edges * (2.0 * math.pi / ring_size)
        if sweep_mag <= math.pi:
            sweep = math.copysign(sweep_mag, delta)
        else:
            sweep = -math.copysign(sweep_mag, delta)
        mn = float("inf")
        for kk in range(1, edges):
            t = ang0 + sweep * kk / edges
            c = (center[0] + radius * math.cos(t),
                 center[1] + radius * math.sin(t))
            for mm in range(n):
                if mm in (i, j):
                    continue
                d = math.hypot(pts2[mm][0] - c[0],
                               pts2[mm][1] - c[1])
                mn = min(mn, d)
        best_local = max(best_local, mn / BOND)
    return best_local


def _chords_fit(pts2: list[tuple[float, float]], n: int,
                chord_idx_pairs: list[tuple] | None) -> bool:
    """Hard arc-fit gate for the PATH-ring chords: after the short-path
    reroute the anchors are ADJACENT drawn-cycle vertices and the small
    ring completes as a regular |R|-gon arc on one side of their bond —
    reject contours whose local cycle geometry wraps around the chord so
    tightly that BOTH sides collapse onto cycle atoms."""
    for ch in (chord_idx_pairs or []):
        hard = ch[3] if len(ch) > 3 else True
        if not hard:
            continue
        best_local = _arc_min_clearance(pts2, ch[0], ch[1], ch[2], n)
        if best_local is not None and best_local < 0.45:
            return False
    return True


# ---------------------------------------------------------------------------
# rigid side-chain bodies (local-frame subtree builder, v1 lattice rules)
# ---------------------------------------------------------------------------

class RigidBody:
    """A side chain built ONCE in a local frame and placed only through
    rigid transforms (rotation about the attachment atom + translation).

    Local geometry is the standard draftsman zig-zag: +/-60 deg travel
    turns (every interior angle 120 deg), carbonyl oxygens on the
    exterior bisector of their carbon's two heavy bonds, N-methyls in the
    free trigonal slot of their nitrogen, small rings as regular-polygon
    arcs anchored on already-placed atoms.  Rigidity: none of this is
    ever re-drawn after the transform."""

    def __init__(self, mol: Chem.Mol, root: int, entry: int, bond: float,
                 forbidden: set[int]):
        self.mol = mol
        self.root = root
        self.entry = entry
        self.bond = bond
        self.forbidden = forbidden       # atoms owned by other bodies/placed
        self.bonded = _adjacency(mol)
        self.local: dict[int, tuple[float, float]] = {root: (0.0, 0.0)}
        self.internal_bonds: list[tuple[int, int]] = []
        self.queue: list[tuple[int, int]] = []
        self._build()

    # -- local geometry rules -------------------------------------------------

    def _third_slot(self, parent: int) -> tuple[float, float]:
        """Direction of the free trigonal slot at `parent` (parent already
        carries two placed bonds at 120 deg): the bisector of the open
        240-degree sector — the exterior-bisector rule."""
        pp = self.local[parent]
        dirs = [_norm((self.local[nb][0] - pp[0], self.local[nb][1] - pp[1]))
                for nb in self.bonded[parent] if nb in self.local]
        if len(dirs) < 2:
            return (1.0, 0.0)
        base = _norm((dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]))
        return (-base[0], -base[1])

    def _incoming(self, parent: int) -> tuple[float, float]:
        """Travel direction of the bond that entered `parent`."""
        for nb in self.bonded[parent]:
            if nb in self.local:
                v = (self.local[parent][0] - self.local[nb][0],
                     self.local[parent][1] - self.local[nb][1])
                if math.hypot(*v) > 1e-9:
                    return _norm(v)
        return (1.0, 0.0)

    def _place_chain_atom(self, atom: int, parent: int,
                          lean: float | None = None) -> float:
        """Standard zig-zag step off `parent`; returns the new lean so the
        chain keeps alternating (+/-60 deg turns, 120 deg angles)."""
        pp = self.local[parent]
        heavy = [nb for nb in self.bonded[parent]
                 if nb in self.local and nb != atom]
        if len(heavy) >= 2:
            d = self._third_slot(parent)     # branch slot (120 to both)
            lean = 0.0
        else:
            hint = self._incoming(parent) if heavy else (1.0, 0.0)
            if lean is None or lean == 0.0:
                lean = 1.0
            # clearance-scored zig-zag: the ±60 candidates are scored
            # against the atoms the body has already drawn (the draftsman's
            # small kink) so branches never fold back over themselves
            best = None
            for s in (lean, -lean):
                d = _rot(hint, s * ZIG_TURN)
                c = (pp[0] + self.bond * d[0], pp[1] + self.bond * d[1])
                cl = self._local_clearance({atom: c}, exclude={parent})
                key = (min(cl, 2.0 * self.bond), 1.0 if s == lean else 0.0)
                if best is None or key > best[0]:
                    best = (key, d, s)
            d, lean = best[1], best[2]
        self.local[atom] = (pp[0] + self.bond * d[0],
                            pp[1] + self.bond * d[1])
        return -lean

    def _unplaced_ring(self, idx: int, done: set[tuple[int, ...]]
                       ) -> tuple[int, ...] | None:
        """Smallest ring through `idx` still to draw.  A ring qualifies
        when it has atoms left to place and touches no atom owned by
        ANOTHER body (`forbidden`); atoms this body already drew are
        welcome as arc anchors — that is how fused systems (the indole
        5-ring + 6-ring sharing a bond) complete as pentagon+hexagon
        instead of one stretched perimeter polygon."""
        cands = [r for r in self.mol.GetRingInfo().AtomRings()
                 if idx in r and r not in done
                 and not any(a in self.forbidden for a in r)
                 and not all(a in self.local for a in r)]
        return min(cands, key=len) if cands else None

    def _local_clearance(self, pos: dict[int, tuple[float, float]],
                         exclude: set[int]) -> float:
        best = float("inf")
        items = list(self.local.items())
        for i, c in pos.items():
            for j, p in items:
                if j in exclude or j == i:
                    continue
                if j in self.bonded[i]:
                    continue
                best = min(best, math.hypot(p[0] - c[0], p[1] - c[1]))
        ids = list(pos)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if j in self.bonded[i]:
                    continue
                best = min(best, math.hypot(pos[i][0] - pos[j][0],
                                            pos[i][1] - pos[j][1]))
        return best

    # -- small rings (ported v1 lattice rules, local frame) -------------------

    def _place_ring(self, ring: tuple[int, ...], entry: int, parent: int,
                    ring_done: set[tuple[int, ...]]) -> None:
        n = len(ring)
        pre = [i for i in ring if i in self.local]
        if entry not in self.local and len(pre) < 2:
            lean = self._place_chain_atom(entry, parent)
            self.queue.append((entry, parent, lean))
            self._register_bond(parent, entry)
        anchors = [i for i in ring if i in self.local]
        start = ring.index(anchors[0])
        order = list(ring[start:]) + list(ring[:start])
        if len(anchors) == 1:
            self._place_full_ring(order[0], order[1:] + order[:1], n)
        else:
            for k in range(len(order)):
                i = order[k]
                if i in self.local:
                    seq = [i]
                    j = (k + 1) % len(order)
                    while order[j] not in self.local:
                        seq.append(order[j])
                        j = (j + 1) % len(order)
                    seq.append(order[j])
                    if len(seq) > 2:
                        self._place_arc(seq, n)
        ring_done.add(ring)
        for i in order:
            if i in self.local:
                self.queue.append((i, None, None))

    def _place_full_ring(self, entry: int, seq: list[int], n: int) -> None:
        """Regular n-gon walked from the entry bond, opening on the side
        with more local clearance."""
        a = self.local[entry]
        entry_dir = self._incoming(entry)
        best = None
        for sign in (1.0, -1.0):
            heading = math.degrees(math.atan2(entry_dir[1], entry_dir[0]))
            heading += sign * ZIG_TURN
            cur = a
            trial = []
            for i in seq:
                cur = (cur[0] + self.bond * math.cos(math.radians(heading)),
                       cur[1] + self.bond * math.sin(math.radians(heading)))
                trial.append((i, cur))
                heading += 360.0 / n
            cl = self._local_clearance({i: c for i, c in trial},
                                       exclude={entry})
            if best is None or cl > best[0]:
                best = (cl, trial)
        for i, c in best[1]:
            self.local[i] = c
            self.queue.append((i, None, None))
        ring_pts = [a] + [c for _, c in best[1]]
        for k in range(len(ring_pts)):
            i1 = ([entry] + [i for i, _ in best[1]])[k]
            i2 = ([entry] + [i for i, _ in best[1]])[(k + 1) % n]
            self._register_bond(i1, i2)

    def _place_arc(self, seq: list[int], n: int) -> None:
        """Lay seq[1:-1] (new vertices) between placed endpoints as the
        arc of a regular n-gon (major arc for fused rings)."""
        a, b = self.local[seq[0]], self.local[seq[-1]]
        edges = len(seq) - 1
        chord = math.hypot(b[0] - a[0], b[1] - a[1])
        radius = (chord / 2.0) / math.sin(math.pi * edges / n)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        perp = _norm((-(b[1] - a[1]), b[0] - a[0]))
        h = math.sqrt(max(radius * radius - chord * chord / 4.0, 0.0))
        best = None
        for side in (1.0, -1.0):
            center = (mid[0] + perp[0] * h * side,
                      mid[1] + perp[1] * h * side)
            ang0 = math.atan2(a[1] - center[1], a[0] - center[0])
            ang1 = math.atan2(b[1] - center[1], b[0] - center[0])
            delta = (ang1 - ang0) % (2.0 * math.pi)
            if delta > math.pi:
                delta -= 2.0 * math.pi
            sweep_mag = edges * (2.0 * math.pi / n)
            if sweep_mag <= math.pi:
                sweep = math.copysign(sweep_mag, delta)
            else:
                sweep = -math.copysign(sweep_mag, delta)
            trial = []
            for k in range(1, edges):
                t = ang0 + sweep * k / edges
                trial.append((center[0] + radius * math.cos(t),
                              center[1] + radius * math.sin(t)))
            cand = {i: c for i, c in zip(seq[1:-1], trial)}
            cl = self._local_clearance(cand, exclude={seq[0], seq[-1]})
            if best is None or cl > best[0]:
                best = (cl, cand)
        for i, c in best[1].items():
            self.local[i] = c
            self.queue.append((i, None, None))
        chain = [seq[0]] + list(best[1].keys()) + [seq[-1]]
        for k in range(len(chain) - 1):
            self._register_bond(chain[k], chain[k + 1])

    def _register_bond(self, i: int, j: int) -> None:
        if i != self.root and j != self.root and i in self.local \
                and j in self.local and i < j:
            self.internal_bonds.append((i, j))

    # -- construction driver ----------------------------------------------------

    def _build(self) -> None:
        """Worklist construction off the attachment bond root->entry; the
        carbonyl oxygen of each carbon is placed after its heavy bonds
        (the bisector rule needs both)."""
        lean = self._place_chain_atom(self.entry, self.root, lean=None)
        self.queue.append((self.entry, self.root, lean))
        ring_done: set[tuple[int, ...]] = set()
        self._register_bond(self.root, self.entry)
        while self.queue:
            atom, parent, atom_lean = self.queue.pop(0)
            o = _carbonyl_oxygen(self.mol, atom)
            kids = [k for k in sorted(self.bonded[atom])
                    if k not in self.local and k not in self.forbidden]
            if o is not None and o in kids:
                kids.remove(o)
            chains, ring_kids = [], []
            for kid in kids:
                if self._unplaced_ring(kid, ring_done) is not None:
                    ring_kids.append(kid)
                else:
                    chains.append(kid)
            for k, kid in enumerate(chains):
                kid_lean = atom_lean if len(chains) == 1 else \
                    (atom_lean or 1.0) * (1.0 if k == 0 else -1.0)
                nl = self._place_chain_atom(kid, atom, lean=kid_lean)
                self.queue.append((kid, atom, nl))
                self._register_bond(atom, kid)
            for kid in ring_kids:
                self._place_ring(self._unplaced_ring(kid, ring_done),
                                 kid, atom, ring_done)
            if o is not None and o not in self.local \
                    and o not in self.forbidden:
                d = self._third_slot(atom) if len(
                    [nb for nb in self.bonded[atom] if nb in self.local]
                ) >= 2 else _rot(self._incoming(atom), 180.0)
                pa = self.local[atom]
                self.local[o] = (pa[0] + self.bond * d[0],
                                 pa[1] + self.bond * d[1])
                self._register_bond(atom, o)
        # normalize the frame: first bond (root -> entry) points along +X
        d = self.local[self.entry]
        ang = math.degrees(math.atan2(d[1], d[0]))
        if abs(ang) > 1e-9:
            self.local = {i: _rot(p, -ang) for i, p in self.local.items()}
        self.atoms = [i for i in self.local if i != self.root]
        self.size = len(self.atoms)

    # -- rigid placement --------------------------------------------------------

    def transform(self, origin: tuple[float, float],
                  angle_deg: float) -> dict[int, tuple[float, float]]:
        """Rigid rotate+translate; the root stays at `origin` and the
        first bond points along `angle_deg` degrees."""
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        out = {}
        for i, (x, y) in self.local.items():
            if i == self.root:
                out[i] = origin
            else:
                out[i] = (origin[0] + x * c - y * s,
                          origin[1] + x * s + y * c)
        return out

    def bonds(self) -> list[tuple[int, int]]:
        return list(self.internal_bonds)


# ---------------------------------------------------------------------------
# placement engine
# ---------------------------------------------------------------------------

class ZigzagLayout:
    """Assembles the depiction: backbone, embedded-ring chords, pendant
    arcs, rigid side-chain bodies (outside -> inside -> intermediate),
    then the collision repair pass."""

    def __init__(self, mol: Chem.Mol, seed: int = 0):
        self.mol = mol
        self.rng = random.Random(seed)
        self.bonded = _adjacency(mol)
        self.pos: dict[int, tuple[float, float]] = {}
        self.ring_atoms: list[int] = []
        self.ring_center = (0.0, 0.0)
        self.chords: list[tuple[int, int]] = []
        self.bodies: dict[int, RigidBody] = {}
        self.orientation: dict[int, str] = {}
        self.unresolved: list[tuple[int, int]] = []
        self.all_bonds: list[tuple[int, int]] = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    # -- scoring -------------------------------------------------------------

    def clearance(self, positions: dict[int, tuple[float, float]],
                  exclude_atoms: set[int]) -> float:
        """Min NON-BONDED distance from candidate positions to placed
        atoms and among the candidates themselves (bonded pairs are the
        molecule's own geometry, never crowding)."""
        own = set(positions)
        best = float("inf")
        for i, c in positions.items():
            for j, p in self.pos.items():
                if j in exclude_atoms or j in own or j in self.bonded[i]:
                    continue
                best = min(best, math.hypot(p[0] - c[0], p[1] - c[1]))
            for j, p in positions.items():
                if j == i or j in self.bonded[i]:
                    continue
                best = min(best, math.hypot(p[0] - c[0], p[1] - c[1]))
        return best

    def _crossings(self, positions: dict[int, tuple[float, float]]) -> int:
        """Bond crossings of the candidate bonds against placed bonds."""
        own = set(positions)
        own_bonds = []
        for (a, b) in self.all_bonds:
            a_o, b_o = a in own, b in own
            if a_o and b_o:
                own_bonds.append((a, b))
            elif a_o and b in self.pos:
                own_bonds.append((a, b))
            elif b_o and a in self.pos:
                own_bonds.append((a, b))
        cnt = 0
        for (a, b) in own_bonds:
            pa = positions.get(a, self.pos.get(a))
            pb = positions.get(b, self.pos.get(b))
            if pa is None or pb is None:
                continue
            for (i1, j1) in self.all_bonds:
                if len({a, b, i1, j1}) < 4:
                    continue
                if i1 in own or j1 in own:
                    continue
                if i1 not in self.pos or j1 not in self.pos:
                    continue
                if _seg_cross(pa, pb, self.pos[i1], self.pos[j1]):
                    cnt += 1
        return cnt

    # -- backbone ------------------------------------------------------------

    def layout_ring(self) -> bool:
        ring = find_macroring(self.mol)
        n = len(ring)
        order = ordered_ring(self.mol, ring)
        start = self.rng.randrange(n)
        if self.rng.random() < 0.5:
            order = order[::-1]
        order = order[start:] + order[:start]
        # embedded path-rings (piperidine class): route the drawn cycle
        # through the SHORT path; their internal atoms come back as the
        # pendant-arc hexagon in layout_pendant_arcs
        dropped: set[int] = set()
        ring_sizes: dict[int, int] = {}
        for pr in _detect_path_rings(self.mol, order):
            if dropped & set(pr["atoms"]):
                continue
            dropped |= set(pr["atoms"])
            ring_sizes[pr["chord_atoms"][0]] = pr["size"]
            ring_sizes[pr["chord_atoms"][1]] = pr["size"]
        drawn = [a for a in order if a not in dropped] \
            if dropped else order
        self.ring_atoms = drawn
        m = len(drawn)
        # anchor index pairs in the drawn cycle (adjacent) + ring size,
        # for the backbone's arc-fit check
        pos_d = {a: i for i, a in enumerate(drawn)}
        chord_pairs = []
        seen_pairs = set()
        for pr in _detect_path_rings(self.mol, order):
            ea, eb = pr["chord_atoms"]
            ia, ib = pos_d.get(ea), pos_d.get(eb)
            if ia is None or ib is None:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            chord_pairs.append((ia, ib, pr["size"], True))
        # pendant-completion chords (rings over TWO adjacent drawn-cycle
        # atoms — the proline pentagons and the fused-on hexagons): feed
        # them to the backbone's arc-fit scoring as a SOFT term, so the
        # accepted contour prefers leaving one side of the chord with
        # room for the regular completion arc
        for ring in self.mol.GetRingInfo().AtomRings():
            if len(ring) < 4 or len(ring) >= m:
                continue
            in_ids = [a for a in ring if a in pos_d]
            if len(in_ids) != 2:
                continue
            ia, ib = pos_d[in_ids[0]], pos_d[in_ids[1]]
            if (ib - ia) % m not in (1, m - 1):
                continue                # not adjacent on the drawn cycle
            key = (min(ia, ib), max(ia, ib))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            chord_pairs.append((key[0], key[1], len(ring), False))
        built = build_backbone(m, self.rng, chord_pairs)
        if built is None:
            return False
        pts = built
        if self.rng.random() < 0.5:          # depiction mirror (2D only)
            pts = [(-x, y) for x, y in pts]
        cx = sum(p[0] for p in pts) / m
        cy = sum(p[1] for p in pts) / m
        for i, v in enumerate(drawn):
            self.pos[v] = (pts[i][0] - cx, pts[i][1] - cy)
        self.ring_center = (cx, cy)
        self.chords = []
        # spatial-utilisation targets (round 2): R0 = n*L/(2pi) in layout
        # units — the page multiplies by RING_RADIUS_SCALE, so the drawn
        # ring renders at R0_page = n*L/(2pi)*1.4; side chains stretch
        # toward R_out = R0 * 1.3 and stay out of the cavity floor
        self.r0_ideal = m * BOND / (2.0 * math.pi)
        self.r_out = self.r0_ideal * OUT_SPRING
        self.r_in_floor = self.r0_ideal * R_IN_FLOOR_FRAC
        return True

    # -- pendant arcs ----------------------------------------------------------

    def layout_pendant_arcs(self) -> bool:
        """Small rings sharing ONE contiguous path with the macroring but
        carrying new atoms (proline class): the new atoms complete a
        regular |R|-gon anchored on the shared path endpoints — both
        bulge sides scored, better clearance (outward on ties) wins."""
        pos_of = {a: i for i, a in enumerate(self.ring_atoms)}
        n_ring = len(self.ring_atoms)
        placed_rings: list[tuple[int, ...]] = []
        for ring in sorted(self.mol.GetRingInfo().AtomRings(), key=len):
            if len(ring) < 4 or len(ring) >= n_ring:
                continue
            if tuple(sorted(ring)) in placed_rings:
                continue
            in_ring = [a for a in ring if a in pos_of]
            if len(in_ring) < 2 or len(in_ring) == len(ring):
                continue                 # tree / full-path (chord) case
            idxs = sorted(pos_of[a] for a in in_ring)
            k = len(in_ring)
            gaps = [(idxs[(j + 1) % k] - idxs[j]) % n_ring
                    for j in range(k)]
            big = max(range(k), key=lambda j: gaps[j])
            if gaps[big] != n_ring - k + 1:
                continue                     # interleaved: tree fallback
            # the shared run in cyclic order (idxs rotated past the gap)
            members = idxs[big + 1:] + idxs[:big + 1]
            new_atoms = [a for a in ring if a not in pos_of]
            a_id = self.ring_atoms[members[0]]
            b_id = self.ring_atoms[members[-1]]
            if not self._place_completion_arc(a_id, b_id, new_atoms,
                                              len(ring)):
                return False
            placed_rings.append(tuple(sorted(ring)))
        return True

    def _place_completion_arc(self, a_id: int, b_id: int,
                              new_atoms: list[int], ring_size: int) -> bool:
        pa, pb = self.pos[a_id], self.pos[b_id]
        edges = len(new_atoms) + 1
        chord = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        radius = (chord / 2.0) / math.sin(math.pi * edges / ring_size)
        if chord / 2.0 > radius + 1e-9:
            return False
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        perp = _norm((-(pb[1] - pa[1]), pb[0] - pa[0]))
        h = math.sqrt(max(radius * radius - chord * chord / 4.0, 0.0))
        sides = []
        for side in (1.0, -1.0):
            center = (mid[0] + perp[0] * h * side,
                      mid[1] + perp[1] * h * side)
            ang0 = math.atan2(pa[1] - center[1], pa[0] - center[0])
            ang1 = math.atan2(pb[1] - center[1], pb[0] - center[0])
            delta = (ang1 - ang0) % (2.0 * math.pi)
            if delta > math.pi:
                delta -= 2.0 * math.pi
            sweep_mag = edges * (2.0 * math.pi / ring_size)
            if sweep_mag <= math.pi:
                sweep = math.copysign(sweep_mag, delta)
            else:
                sweep = -math.copysign(sweep_mag, delta)
            trial = []
            for kk in range(1, edges):
                t = ang0 + sweep * kk / edges
                trial.append((center[0] + radius * math.cos(t),
                              center[1] + radius * math.sin(t)))
            cand = {i: c for i, c in zip(new_atoms, trial)}
            cl = self.clearance(cand, exclude_atoms={a_id, b_id})
            cross = self._crossings(cand)
            cent = (sum(p[0] for p in cand.values()) / len(cand),
                    sum(p[1] for p in cand.values()) / len(cand))
            outward = (cent[0] - self.ring_center[0]) * perp[0] * side + \
                (cent[1] - self.ring_center[1]) * perp[1] * side
            sides.append((cross == 0 and cl >= ACCEPT_CLEARANCE,
                          1.0 if outward >= 0 else 0.0, cand,
                          min(cl, 2.0 * BOND)
                          - 3.0 * max(0.0, ACCEPT_CLEARANCE - cl)
                          - 12.0 * BOND * cross
                          + 0.8 * BOND * (1.0 if outward >= 0 else 0.0)))
        # exterior bulge preferred (the real crops draw boundary small
        # rings outside the outline) but COLLAPSE-free placement first:
        # the score adds an outward bonus to the clearance so the
        # exterior side wins whenever it is viable, and the interior
        # side takes over when the exterior crosses or collapses
        sides.sort(key=lambda s: s[3], reverse=True)
        for i, c in sides[0][2].items():
            self.pos[i] = c
        return True

    # -- rigid bodies -----------------------------------------------------------

    def _placed_order(self) -> list[int]:
        ring_set = set(self.ring_atoms)
        order = [a for a in self.ring_atoms if a in self.pos]
        others = [i for i in self.pos if i not in ring_set]
        return order + others

    def exterior_bisector(self, idx: int) -> tuple[float, float]:
        """Outward direction at `idx` against its already-placed bonds:
        the bisector of the LARGEST angular gap between the placed bond
        directions — the standard exterior bisector for two ring bonds
        (120 deg apart, 240-deg reflex sector), and the surviving free
        trigonal slot for fuller vertices (the quaternary C alpha of the
        real crops)."""
        dirs = [_norm((self.pos[nb][0] - self.pos[idx][0],
                       self.pos[nb][1] - self.pos[idx][1]))
                for nb in self.bonded[idx] if nb in self.pos]
        away = _norm((self.pos[idx][0] - self.ring_center[0],
                      self.pos[idx][1] - self.ring_center[1]))
        if not dirs:
            return away
        if len(dirs) == 1:
            return _rot(dirs[0], 180.0)
        angs = sorted(math.degrees(math.atan2(d[1], d[0])) % 360.0
                      for d in dirs)
        best_gap, best_mid = -1.0, 0.0
        for j, a in enumerate(angs):
            b = angs[(j + 1) % len(angs)]
            gap = (b - a) % 360.0
            if gap > best_gap:
                best_gap, best_mid = gap, a + gap / 2.0
        rad = math.radians(best_mid)
        return (math.cos(rad), math.sin(rad))

    def place_bodies(self) -> None:
        """Build one rigid body per unplaced neighbour subtree and place
        it: outside first, flipped inside when crowded, intermediate
        angles last (the CANDIDATE_DELTAS order).  Largest subtrees claim
        their slot first."""
        pos_of = {a: i for i, a in enumerate(self.ring_atoms)}
        owned: set[int] = set(self.pos)
        sized = []
        for root in self._placed_order():
            for kid in sorted(self.bonded[root]):
                if kid in owned:
                    continue
                body = RigidBody(self.mol, root, kid, BOND, owned)
                for i in body.atoms:
                    owned.add(i)
                self.bodies[kid] = body
                sized.append((body.size, pos_of.get(root, 10_000), root, kid))
        sized.sort(key=lambda t: (-t[0], t[1]))
        for _size, _ri, root, kid in sized:
            self._place_one(root, kid)

    def _attach_angle_ok(self, root: int, direction: tuple[float, float]
                         ) -> bool:
        for nb in self.bonded[root]:
            if nb not in self.pos:
                continue
            v = _norm((self.pos[nb][0] - self.pos[root][0],
                       self.pos[nb][1] - self.pos[root][1]))
            if _angle_between(direction, v) < MIN_ATTACH_ANGLE:
                return False
        return True

    def _spring_score(self, root: int,
                      cand: dict[int, tuple[float, float]]) -> float:
        """Radial outward spring of a candidate body placement (round 2
        spatial tuning): reward for the subtree's OUTWARD RADIAL GAIN
        beyond its root, saturating at R_out = R0 * 1.3, and a penalty
        for atoms reaching into the ring cavity (inside the R_IN_FLOOR
        radius) — so side chains extend outward and never squeeze into
        the macroring interior.  The gain form (rho_max - rho_root)
        cannot push a body into a neighbouring arc across a locally
        concave ring segment: tangential placements simply earn no
        spring."""
        cx, cy = self.ring_center
        root_p = self.pos[root]
        rho_root = math.hypot(root_p[0] - cx, root_p[1] - cy)
        rs = [math.hypot(p[0] - cx, p[1] - cy)
              for i, p in cand.items() if i != root]
        if not rs:
            return 0.0
        gain = max(0.0, max(rs) - rho_root)
        spring = SPRING_W * BOND * min(gain,
                                       max(0.0, self.r_out - rho_root)) \
            / self.r_out
        inside = INSIDE_W * max(0.0, self.r_in_floor - min(rs))
        return spring - inside

    def _place_one(self, root: int, kid: int, taken: list[float] = ()
                   ) -> bool:
        body = self.bodies[kid]
        origin = self.pos[root]
        bis = self.exterior_bisector(root)
        base_deg = math.degrees(math.atan2(bis[1], bis[0]))
        best = None
        # candidate directions: the coarse out/in/middle deltas of the
        # real-crop vocabulary, refined at 15 deg steps (attach angles
        # still >= MIN_ATTACH_ANGLE against every placed ring bond)
        deltas: list[float] = []
        for delta in CANDIDATE_DELTAS:
            if any(abs((delta - t + 180.0) % 360.0 - 180.0) < 25.0
                   for t in taken):
                continue
            deltas.append(delta)
        for delta in CANDIDATE_DELTAS:
            if delta in (0.0, 180.0):
                continue
            for fine in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
                fd = delta + fine
                if not any(abs((fd - t + 180.0) % 360.0 - 180.0) < 12.0
                           for t in taken) \
                        and all(abs(fd - d) > 1e-9 for d in deltas):
                    deltas.append(fd)
        for delta in deltas:
            angle = math.radians(base_deg + delta)
            d = (math.cos(angle), math.sin(angle))
            if not self._attach_angle_ok(root, d):
                continue
            cand = body.transform(origin, math.degrees(angle))
            cross = self._crossings(cand)
            cl = self.clearance(cand, exclude_atoms={root})
            tag = "out" if abs(delta) <= 45.0 else \
                ("in" if abs(delta - 180.0) <= 45.0 else "mid")
            # clearance-first packing with the mild outside-first
            # preference of the real crops (out -> in -> intermediate as
            # the fallback order when slots tie), plus the round-2 radial
            # outward spring (R_out = R0 * 1.3): stretch outward, never
            # reach into the ring cavity
            pref = {0.0: 0.35, 180.0: 0.0}.get(delta, 0.0) * BOND
            score = (min(cl, 1.6 * BOND)
                     - 3.0 * max(0.0, ACCEPT_CLEARANCE - cl)
                     - 100.0 * BOND * cross) + pref \
                + self._spring_score(root, cand)
            if best is None or score > best[0]:
                best = (score, cand, cl, cross, tag, float(delta))
        if best is None:
            self.unresolved.append((root, kid))
            return False
        _score, cand, cl, cross, tag, delta = best
        for i, p in cand.items():
            if i != root:
                self.pos[i] = p
        self.orientation[kid] = tag
        self.taken_angles = getattr(self, "taken_angles", {})
        self.taken_angles.setdefault(root, []).append(delta)
        if cl < ACCEPT_CLEARANCE or cross > MAX_CROSSINGS:
            self.unresolved.append((root, kid))
        return True

    # -- repair ------------------------------------------------------------------

    def _fine_tune(self, root: int, kid: int) -> bool:
        """Sub-candidate micro-rotation of a placed body: +-5..30 deg
        around its current first-bond direction, best clearance wins
        (the spec's 'rotate the side chain to the next-best angle'
        adjustment, at fine granularity)."""
        body = self.bodies[kid]
        if kid not in self.pos or root not in self.pos:
            return False
        origin = self.pos[root]
        cur = math.degrees(math.atan2(self.pos[kid][1] - origin[1],
                                      self.pos[kid][0] - origin[0]))
        best = None
        for off in (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0,
                    20.0, -20.0, 25.0, -25.0, 30.0, -30.0,
                    45.0, -45.0, 60.0, -60.0, 90.0, -90.0,
                    120.0, -120.0, 150.0, -150.0, 180.0):
            cand = body.transform(origin, cur + off)
            if not self._attach_angle_ok(root, (math.cos(math.radians(
                    cur + off)), math.sin(math.radians(cur + off)))):
                continue
            cross = self._crossings(cand)
            cl = self.clearance(cand, exclude_atoms={root})
            score = (min(cl, 1.6 * BOND)
                     - 3.0 * max(0.0, ACCEPT_CLEARANCE - cl)
                     - 100.0 * BOND * (cross > 0)) \
                + self._spring_score(root, cand)
            if best is None or score > best[0]:
                best = (score, cand)
        if best is None:
            return False
        for i, p in best[1].items():
            if i != root:
                self.pos[i] = p
        return self._body_ok(kid)

    def repair(self) -> None:
        """Collision pass: lift EVERY offending body off the canvas, then
        re-place them one by one (largest first) against the settled
        neighbourhood — the draftsman's redraw of the crowded branch.
        Bodies still crowded are micro-rotated; anything left after
        MAX_REPAIR_ROUNDS is reported."""
        for _ in range(MAX_REPAIR_ROUNDS):
            if not self.unresolved:
                break
            batch = sorted({kid: (root, kid)
                            for (root, kid) in self.unresolved}.values(),
                           key=lambda rk: -self.bodies[rk[1]].size
                           if rk[1] in self.bodies else 0)
            for (root, kid) in batch:
                body = self.bodies.get(kid)
                if body is None:
                    continue
                for i in body.atoms:
                    self.pos.pop(i, None)
                self.orientation.pop(kid, None)
                if getattr(self, "taken_angles", {}).get(root):
                    self.taken_angles[root] = []
            still: list[tuple[int, int]] = []
            for (root, kid) in batch:
                self._place_one(root, kid)
                if kid in self.orientation and self._body_ok(kid):
                    continue
                still.append((root, kid))
            # fine-grained rotation for whatever is still crowded
            for (root, kid) in list(still):
                if self.bodies.get(kid) is not None and kid in self.pos \
                        and self._fine_tune(root, kid):
                    still.remove((root, kid))
            if still == self.unresolved:
                break
            self.unresolved = still

    def _body_ok(self, kid: int) -> bool:
        body = self.bodies[kid]
        cand = {i: self.pos[i] for i in body.atoms if i in self.pos}
        if len(cand) != len(body.atoms):
            return False
        cl = self.clearance(cand, exclude_atoms={body.root})
        return cl >= ACCEPT_CLEARANCE and self._crossings(cand) == 0

    # -- QA -----------------------------------------------------------------------

    def qa(self) -> dict:
        """Rigidity + collision report of the FINAL coordinates."""
        n = len(self.ring_atoms)
        lens = []
        for k, a in enumerate(self.ring_atoms):
            b = self.ring_atoms[(k + 1) % n]
            p, q = self.pos[a], self.pos[b]
            lens.append(math.hypot(q[0] - p[0], q[1] - p[1]))
        angles = []
        for k in range(n):
            a = self.ring_atoms[k - 1]
            b = self.ring_atoms[k]
            c = self.ring_atoms[(k + 1) % n]
            v1 = _norm((self.pos[a][0] - self.pos[b][0],
                        self.pos[a][1] - self.pos[b][1]))
            v2 = _norm((self.pos[c][0] - self.pos[b][0],
                        self.pos[c][1] - self.pos[b][1]))
            angles.append(_angle_between(v1, v2))
        mind = float("inf")
        pairs = []
        ids = list(self.pos)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if b in self.bonded[a]:
                    continue
                d = math.hypot(self.pos[a][0] - self.pos[b][0],
                               self.pos[a][1] - self.pos[b][1])
                if d < mind:
                    mind = d
                if d < ACCEPT_CLEARANCE:
                    pairs.append((a, b, round(d / BOND, 2)))
        crossings = 0
        pts = self.pos
        for k in range(len(self.all_bonds)):
            i1, j1 = self.all_bonds[k]
            for l in range(k + 1, len(self.all_bonds)):
                i2, j2 = self.all_bonds[l]
                if len({i1, j1, i2, j2}) < 4:
                    continue
                if _seg_cross(pts[i1], pts[j1], pts[i2], pts[j2]):
                    crossings += 1
        return {
            "ring_bond_len_min": round(min(lens), 4),
            "ring_bond_len_max": round(max(lens), 4),
            "angle_min": round(min(angles), 2),
            "angle_max": round(max(angles), 2),
            "angle_mean": round(sum(angles) / n, 2),
            "min_nonbonded_ratio": round(mind / BOND, 3),
            "close_pairs": pairs,
            "n_close_pairs": len(pairs),
            "bond_crossings": crossings,
        }

    def qa_pass(self) -> bool:
        m = self.qa()
        # odd macrorings cannot close on the exact +/-60 alphabet: their
        # closing bond carries a sub-pixel deviation (0.02 bond = 0.6 px,
        # invisible at render scale) reported through the metrics
        len_tol = 0.02 * BOND if len(self.ring_atoms) % 2 else 1e-6
        return (abs(m["ring_bond_len_min"] - BOND) < len_tol
                and abs(m["ring_bond_len_max"] - BOND) < len_tol
                and m["angle_min"] >= 120.0 - ANGLE_TOL
                and m["angle_max"] <= 120.0 + ANGLE_TOL
                and m["min_nonbonded_ratio"] >= QA_FLOOR / BOND
                and m["bond_crossings"] <= MAX_CROSSINGS)


# ---------------------------------------------------------------------------
# stereo (chirality) gate + report
# ---------------------------------------------------------------------------

def stereo_roundtrip_ok(mol: Chem.Mol) -> bool:
    """True when the drawn 2D coordinates reproduce the input stereo: the
    molblock writer derives every wedge from the chiral tags + placed
    coordinates; re-parsing re-derives the tags from the wedges.  Any
    coordinate arrangement that would print the wrong wedge/dash at a
    C alpha (e.g. after an inside flip) fails here."""
    try:
        mb = Chem.MolToMolBlock(mol, includeStereo=True)
        back = Chem.MolFromMolBlock(mb, sanitize=True)
        if back is None:
            return False
        return Chem.MolToSmiles(back) == Chem.MolToSmiles(mol)
    except Exception:
        return False


def stereo_marks(mol: Chem.Mol) -> dict[int, str]:
    """wedge/dash mark at every chiral center per the renderer's own
    wedging (Chem.WedgeMolBonds over the placed coordinates)."""
    work = Chem.Mol(mol)
    conf = work.GetConformer()
    try:
        Chem.rdmolops.WedgeMolBonds(work, conf)
    except Exception:
        return {}
    names = {Chem.BondDir.BEGINWEDGE: "wedge",
             Chem.BondDir.BEGINDASH: "dash"}
    out: dict[int, str] = {}
    for b in work.GetBonds():
        d = b.GetBondDir()
        if d not in names:
            continue
        for idx in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()):
            a = work.GetAtomWithIdx(idx)
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED \
                    and idx not in out:
                out[idx] = names[d]
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def zigzag_v2_coords(mol: Chem.Mol, seed: int = 0
                     ) -> tuple[Chem.Mol | None, dict]:
    """Lay `mol` out as a freehand irregular zig-zag macroring.

    Returns (molecule with coordinates, report); the molecule is None
    only for hard failures (no usable backbone, incomplete placement,
    stereo round-trip mismatch).  Remaining non-bonded crowding after
    the repair passes does NOT discard the layout: the report carries
    `qa_pass` plus the offending pairs — the spec's report-instead-of-
    fail behaviour — and `layout_with_retry` still prefers a seed whose
    layout is collision-free when one exists."""
    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 0:
        return None, {"error": "empty"}
    # keep the parsed SSSR ring info: FastFindRings misses the small
    # rings of fused systems (an indole's 5- and 6-rings collapse into
    # the 9-cycle perimeter) and fused aryls would draw as one stretched
    # perimeter polygon; the fast pass is only a fallback for molecules
    # that arrive without ring perception
    if work.GetRingInfo().NumRings() == 0:
        Chem.FastFindRings(work)
    lay = ZigzagLayout(work, seed=seed)
    if not lay.layout_ring():
        return None, {"error": "backbone"}
    if not lay.layout_pendant_arcs():
        return None, {"error": "pendant_arc"}
    lay.place_bodies()
    lay.repair()
    if any(i not in lay.pos for i in range(work.GetNumAtoms())):
        return None, {"error": "incomplete"}
    qa = lay.qa()
    qa_pass = lay.qa_pass()
    conf = Chem.Conformer(work.GetNumAtoms())
    for i in range(work.GetNumAtoms()):
        x, y = lay.pos[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    if not stereo_roundtrip_ok(work):
        return None, {"error": "stereo", "qa": qa}
    qa["qa_pass"] = qa_pass
    qa["stereo_roundtrip"] = True
    qa["orientation_out"] = sum(1 for v in lay.orientation.values()
                                if v == "out")
    qa["orientation_in"] = sum(1 for v in lay.orientation.values()
                               if v == "in")
    qa["orientation_mid"] = sum(1 for v in lay.orientation.values()
                                if v == "mid")
    qa["unresolved"] = list(lay.unresolved)
    qa["stereo_marks"] = stereo_marks(work)
    return work, qa


def layout_with_retry(mol: Chem.Mol, seed: int = 0, seeds: int = 6
                      ) -> tuple[Chem.Mol | None, dict]:
    """Seed loop around `zigzag_v2_coords`: the first collision-free
    layout wins; if no seed reaches the full QA bar, the best-effort
    layout (fewest crossings, then largest minimum non-bonded distance)
    is returned with its crowding report."""
    best: tuple[Chem.Mol | None, dict] | None = None

    def _rank(rep) -> tuple[int, float]:
        # zigzag_v2_coords reports the QA metrics at the TOP level of the
        # report dict (bond_crossings, min_nonbonded_ratio, ...); the old
        # nested rep["qa"] lookup defaulted both sides to (99, -9) and the
        # seed loop silently kept the FIRST seed instead of the cleanest
        qa = rep.get("qa") or {}
        return (qa.get("bond_crossings",
                       rep.get("bond_crossings", 99)),
                -qa.get("min_nonbonded_ratio",
                        rep.get("min_nonbonded_ratio", -9)))

    def better(rep_a, rep_b) -> bool:
        return _rank(rep_a) < _rank(rep_b)

    last: dict = {"error": "no_seed_passed"}
    for k in range(seeds):
        laid, rep = zigzag_v2_coords(mol, seed=seed + k)
        if laid is None:
            last = dict(rep)
            last["seed_used"] = seed + k
            continue
        rep["seed_used"] = seed + k
        if rep.get("qa_pass"):
            return laid, rep
        if best is None or better(rep, best[1]):
            best = (laid, rep)
    if best is not None:
        return best
    return None, last


# ---------------------------------------------------------------------------
# rendering (ALIGNED profile, pure black/white, backbone = 30 px * 1.4)
# ---------------------------------------------------------------------------

def render_profile() -> dict:
    """ALIGNED style with the backbone bond pinned to
    BOND_PX * RING_RADIUS_SCALE = 42 px (round-2 spatial utilisation:
    R0 = n_ring * L / (2*pi) * 1.4 on the page — a 42-atom macroring
    opens a clearly visible interior cavity).  Every backbone bond is
    exactly BOND layout units, so the render scaling (fixedBondLength)
    makes all of them measure the scaled length on the page."""
    return dict(ALIGNED, fixed_bond_length=BOND_PX * RING_RADIUS_SCALE)


def _assert_black_white(profile: dict) -> None:
    """Palette assertion: the render path must be pure black on white."""
    import rdkit.Chem.Draw.rdMolDraw2D as rd
    opts = rd.MolDrawOptions()
    apply_profile(opts, profile, random.Random(0))
    assert opts.useBWAtomPalette, "BW atom palette must be on"
    for colour in opts.getAtomPalette().values():
        assert tuple(colour)[:3] == (0.0, 0.0, 0.0),             f"non-black palette entry {colour}"
    assert opts.singleColourWedgeBonds, "wedge bonds must be single colour"


def _assert_grayscale(img: Image.Image) -> None:
    """Ink assertion: no colour-channel divergence anywhere."""
    arr = np.asarray(img.convert("RGB")).astype(int)
    span = arr.max(axis=2) - arr.min(axis=2)
    assert int(span.max()) == 0, "coloured pixels in a BW render"


def render_v2(smiles: str, out_png: str, seed: int = 0, upright: bool = True,
              retries: int = 24) -> dict:
    """Layout + ALIGNED-style render of one SMILES.

    Pure black/white asserted (palette + single-channel ink), no atom
    indices (RDKit default), arbitrary global rotation like the real
    crops unless `upright` pre-cancels it for shape review.  Rigidity:
    the accepted coordinates are only ever transformed globally, so the
    renderer draws a COPY of the laid-out molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES failed to parse: {smiles[:80]}")
    laid, rep = layout_with_retry(mol, seed=seed, seeds=retries)
    if laid is None:
        raise ValueError(f"zigzag v2 layout failed: {rep}")
    profile = render_profile()
    _assert_black_white(profile)
    rng_draw = random.Random(rep["seed_used"] + 101)
    if upright:
        angle = rng_draw.uniform(0, 360)      # first draw of draw_molecule
        rad = math.radians(-angle)
        render_mol = Chem.Mol(laid)
        conf = render_mol.GetConformer()
        for i in range(render_mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(
                p.x * math.cos(rad) - p.y * math.sin(rad),
                p.x * math.sin(rad) + p.y * math.cos(rad), 0.0))
    else:
        render_mol = Chem.Mol(laid)
    np.random.seed(rep["seed_used"] + 101)    # ALIGNED ink noise
    img = draw_molecule(render_mol, profile, rng_draw)
    _assert_grayscale(img)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)
    rep["png"] = out_png
    rep["shape"] = shape_metrics(laid)
    return rep


# ---------------------------------------------------------------------------
# geometry metrics (shape review)
# ---------------------------------------------------------------------------

def shape_metrics(mol: Chem.Mol) -> dict:
    """Shape of the placed macroring: circularity (radial std/mean, -> 0
    for a circle), bounding-box aspect, irregularity of the radial
    profile (a regular n-gon has near-zero radial spread like a circle,
    so the zigzag corner census is what separates them).  Round-2 adds
    the spatial-utilisation gauges: `radius_ratio` (mean radius over the
    ideal circle n*BOND/(2*pi) — the page renders it times
    RING_RADIUS_SCALE) and `interior_void_ratio` (min centroid distance
    over the same ideal — the visible cavity radius)."""
    ring = ordered_ring(mol, find_macroring(mol))
    conf = mol.GetConformer()
    n = len(ring)
    cx = sum(conf.GetAtomPosition(a).x for a in ring) / n
    cy = sum(conf.GetAtomPosition(a).y for a in ring) / n
    ds = [math.hypot(conf.GetAtomPosition(a).x - cx,
                     conf.GetAtomPosition(a).y - cy) for a in ring]
    md = sum(ds) / n
    circ = math.sqrt(sum((d - md) ** 2 for d in ds) / n) / md
    xs = [conf.GetAtomPosition(a).x for a in ring]
    ys = [conf.GetAtomPosition(a).y for a in ring]
    r_ideal = n * BOND / (2.0 * math.pi)
    return {"ring_atoms": n,
            "circularity": round(circ, 3),
            "aspect": round((max(xs) - min(xs)) / max(1e-9, max(ys) - min(ys)),
                            2),
            "radius_ratio": round(md / r_ideal, 3),
            "interior_void_ratio": round(min(ds) / r_ideal, 3)}


# ---------------------------------------------------------------------------
# validation suite (seq gold/crops + KRAS + GPC3, comparison sheets)
# ---------------------------------------------------------------------------

GOLD_JSON = "/data/BioChemInsight/data/gold_smiles_289.json"
SEQ53_GOLD_KEY = "3"
US_MANIFEST = ("/data/BioChemInsight/data/real_crops/"
               "real_crops_manifest.json")
KRAS_MANIFEST = "/tmp/crops_kras/real_crops_manifest.json"
GPC3_MANIFEST = "/tmp/crops_gpc3/real_crops_manifest.json"
REAL_CROP_ROOT = "/data/BioChemInsight/data/real_crops"


def _gold_smiles(key: str) -> str:
    gold = json.load(open(GOLD_JSON, encoding="utf-8"))
    return gold[key]["smiles"]


def _verified_smiles_for_seq(seq_id: int) -> tuple[str, str]:
    """(smiles, real crop path) of the first verified SMILES under a SEQ
    ID NO: label."""
    manifest = json.load(open(US_MANIFEST, encoding="utf-8"))
    for row in manifest["rows"]:
        if (row.get("seq_id") == seq_id and row.get("verified") is True
                and row.get("smiles")):
            crop = os.path.join(REAL_CROP_ROOT, row["crop"])
            return row["smiles"], (crop if os.path.exists(crop) else "")
    raise ValueError(f"no verified SMILES for seq_id {seq_id}")


def _verified_pairs(manifest_path: str, count: int, root: str
                    ) -> list[tuple[str, str, str]]:
    """(id, smiles, real crop path) of distinct verified SMILES."""
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for row in manifest["rows"]:
        smi = row.get("smiles")
        if row.get("verified") is not True or not smi or smi in seen:
            continue
        seen.add(smi)
        crop = os.path.join(root, row.get("crop", ""))
        out.append((f"{len(out) + 1}", smi,
                    crop if os.path.exists(crop) else ""))
        if len(out) == count:
            break
    if len(out) < count:
        raise ValueError(f"{manifest_path}: not enough verified SMILES")
    return out


def validation_cases() -> list[tuple[str, str, str]]:
    """(case id, smiles, real crop path) — the seq acceptance set plus
    two KRAS and two GPC3 verified crops for generality."""
    cases: list[tuple[str, str, str]] = []
    cases.append(("seq53_gold", _gold_smiles(SEQ53_GOLD_KEY),
                  os.path.join(REAL_CROP_ROOT,
                               "US20240287138A1/page_024_box_0.png")))
    for sid in (53, 60, 68, 77):
        smi, crop = _verified_smiles_for_seq(sid)
        cases.append((f"seq{sid}_crop", smi, crop))
    cases.append(("seq68_gold", _gold_smiles("68"),
                  os.path.join(REAL_CROP_ROOT,
                               "US20240287138A1/page_024_box_2.png")))
    for mid, smi, crop in _verified_pairs(KRAS_MANIFEST, 2, "/tmp/crops_kras"):
        cases.append((f"kras_{mid}", smi, crop))
    for mid, smi, crop in _verified_pairs(GPC3_MANIFEST, 2, "/tmp/crops_gpc3"):
        cases.append((f"gpc3_{mid}", smi, crop))
    return cases


def comparison_sheet(real_png: str, rendered_png: str, title: str,
                     out_png: str, height: int = 420) -> None:
    """Side-by-side panel: real patent crop (left) vs the v2 render
    (right), with a title bar — the visual review artifact."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype(ALIGNED["font_file"], 20)
    panels = []
    for path in (real_png, rendered_png):
        if path and os.path.exists(path):
            img = Image.open(path).convert("L")
            w = max(1, int(img.width * height / img.height))
            panels.append(img.resize((w, height)))
        else:
            panels.append(Image.new("L", (height, height), 255))
    pad = 12
    width = sum(p.width for p in panels) + pad * (len(panels) + 1)
    sheet = Image.new("L", (width, height + 46), 255)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), title, fill=0, font=font)
    x = pad
    for p in panels:
        sheet.paste(p, (x, 40))
        x += p.width + pad
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    sheet.save(out_png)


def validation_suite(out_dir: str, seed: int = 0) -> dict:
    """Render every validation case + per-case real-vs-render comparison
    sheet; returns the JSON-able summary (printed by __main__)."""
    rows: dict[str, dict] = {}
    for case, smi, crop in validation_cases():
        png = os.path.join(out_dir, f"{case}.png")
        try:
            rep = render_v2(smi, png, seed=seed)
            sheet = os.path.join(out_dir, f"compare_{case}.png")
            comparison_sheet(crop, png,
                             f"{case} — real patent crop (left) vs v2 "
                             f"layout (right)", sheet)
            rows[case] = {
                "ok": True,
                "png": png,
                "compare": sheet,
                "seed": rep["seed_used"],
                "shape": rep["shape"],
                "orientation": {k: rep[k] for k in
                                ("orientation_out", "orientation_in",
                                 "orientation_mid")},
                "angle_min": rep["angle_min"],
                "angle_max": rep["angle_max"],
                "min_nonbonded_ratio": rep["min_nonbonded_ratio"],
                "bond_crossings": rep["bond_crossings"],
                "n_close_pairs": rep["n_close_pairs"],
                "unresolved": rep["unresolved"],
                "stereo_marks": len(rep["stereo_marks"]),
            }
        except Exception as exc:             # per-case isolation
            rows[case] = {"ok": False, "error": str(exc)[:200]}
    return rows


if __name__ == "__main__":  # pragma: no cover - validation CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="freehand zig-zag v2 validation suite")
    ap.add_argument("--out", default="/data/BioChemInsight/images/zigzag_v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(validation_suite(args.out, seed=args.seed), indent=1))

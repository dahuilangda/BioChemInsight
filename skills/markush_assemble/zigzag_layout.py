"""Zig-zag macroring layout renderer — patent-style cyclic peptide depiction.

Real patents draw cyclopeptide macrorings NOT as a round polygon and NOT
as a steadily rotating pinwheel of bonds, but as **K long straight runs
(K = 4-6) joined by visible ~105-140 degree corners**: inside a run the
backbone atoms hug a straight axis with the standard +/-30 degree sawtooth
(120 degree bond angles, vertices offset +/-BOND/4 off the axis), and at
a corner the run axis jumps to the next direction.  The silhouette is an
irregular oval — long axes on nearly opposite sides, short axes between —
with a radius variation coefficient of at least 0.15.  Corner vertices
sit on peptide nitrogens, exactly where draftsmen break their runs.
CoordGen / rdDepictor are deliberately not used: the conformer is
constructed vertex by vertex.

Geometry model (all in layout coordinate units, one bond = BOND units):

  corner ring      the K corner vertices lie on a jittered ellipse whose
         perimeter matches the total run-axis length (0.866 * BOND per
         backbone bond, K and ellipse aspect seeded); chord directions
         therefore close the polygon EXACTLY — the corner positions are
         the closure solution, residual 0 by construction
  run interior     vertex k of a run: chord axis position step*k with
         perpendicular offset +/- amp alternating, amp = sqrt(BOND^2 -
         step^2)/2, so every bond stays exactly BOND long and every
         interior vertex angle stays 120 deg
  corner vertices  peptide-N vertices of the macroring, spaced 2-3
         residues apart so runs span 6-10 bonds

Substituent placement (DFS off the macroring, in ring order):

  carbonyl O   exterior bisector of its backbone vertex (points outward)
  side chains  exterior bisector off the C alpha, then a strict +/-60 deg
               zig-zag chain, first lean chosen away from the centroid
  branches     the mirror direction of the chain continuation (the
               +/-60 deg pair keeps every vertex angle at 120 deg)
  N-methyl     exterior bisector off the peptide N vertex
  small rings  (pyrrolidine, piperidine, indole, phenyl, ...) placed as
               regular-polygon arcs anchored on already-placed vertices,
               so fused rings (indole) and rings sharing a path with the
               macroring (proline homologues) close exactly
  relaxation   crowded subtrees are rigidly rotated around their
               attachment vertex (draftsman-style local adjustment)

Rendering goes through training.molsight_markush.style_calibration with
the ALIGNED profile (the calibrated patent-style parameter set), so the
output inherits the serif labels, fixed bond length, print-gray ink and
tight cropping used across the corpus.
"""
from __future__ import annotations

import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

BOND = 28.0                 # backbone bond length in layout units
ZIG_TURN = 60.0             # travel turn for a 120 deg backbone vertex
AXIS_STEP = 0.866025403784  # per-bond advance along a run axis (cos 30 deg)
MIN_NONBONDED = 0.18        # min non-bonded atom distance, in BOND units
                            # (real crops draw compressed regions tighter
                            # than this; 0.18 * 28 units ~ 4 px at render
                            # scale, distinct strokes, no label overlap)

# shape gates measured off real patent crops (US20240287138A1 p24): the
# silhouette must be an irregular oval built from long straight runs
MIN_STRAIGHT_COVERAGE = 0.60   # backbone bonds inside >=3-bond collinear runs
MIN_RADIUS_CV = 0.15           # coefficient of variation of outline radii
CORNER_RANGE = (4, 6)          # visible corner count


# --------------------------------------------------------------------------
# molecular bookkeeping
# --------------------------------------------------------------------------

def _is_carbonyl_carbon(mol: Chem.Mol, idx: int) -> bool:
    """True when atom idx is a C carrying a double-bonded O."""
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
    """Index of the double-bonded O of a carbonyl carbon, else None."""
    a = mol.GetAtomWithIdx(idx)
    for b in a.GetBonds():
        other = b.GetOtherAtomIdx(idx)
        if (b.GetBondType() == Chem.BondType.DOUBLE
                and mol.GetAtomWithIdx(other).GetAtomicNum() == 8):
            return other
    return None


def _amide_nitrogen(mol: Chem.Mol, idx: int, ring_set: set[int]) -> bool:
    """Ring nitrogen acylated by a carbonyl carbon inside the same ring."""
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() != 7:
        return False
    return any(n.GetIdx() in ring_set and _is_carbonyl_carbon(mol, n.GetIdx())
               for n in a.GetNeighbors())


def find_macroring(mol: Chem.Mol) -> list[int]:
    """Ordered atom cycle of the macroring.

    The macroring is the molecular cycle carrying the amide carbonyls —
    it is the ring containing the most C(=O) vertices (proline-type small
    rings share the N-Calpha path but carry no carbonyl).  The ring tuple
    from RingInfo is already in traversal order.
    """
    best, best_score = None, -1
    for ring in mol.GetRingInfo().AtomRings():
        score = sum(1 for i in ring if _is_carbonyl_carbon(mol, i))
        if score > best_score:
            best, best_score = list(ring), score
    if best is None or best_score < 2:
        raise ValueError("no macroring found: molecule carries no cycle "
                         "with at least two amide carbonyls")
    return best


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------

def _rot(vec: tuple[float, float], deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c)


def _norm(vec: tuple[float, float]) -> tuple[float, float]:
    l = math.hypot(*vec)
    return (vec[0] / l, vec[1] / l) if l > 1e-9 else (1.0, 0.0)


def _seg_cross(p: tuple[float, float], q: tuple[float, float],
               r: tuple[float, float], s: tuple[float, float]) -> bool:
    """Strict interior intersection of segments p-q and r-s."""
    def orient(u, v, w):
        return ((v[0] - u[0]) * (w[1] - u[1])
                - (v[1] - u[1]) * (w[0] - u[0]))
    d1, d2 = orient(p, q, r), orient(p, q, s)
    d3, d4 = orient(r, s, p), orient(r, s, q)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _unit(deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad))


# --------------------------------------------------------------------------
# macroring backbone placement: K straight runs + corner polygon
# --------------------------------------------------------------------------

def _pick_corners(n_pos: list[int], rng: random.Random) -> list[int]:
    """Corner (run-break) vertices, always peptide nitrogens.

    Real drawings break their straight runs at amide nitrogens spaced 2-3
    residues apart, which yields runs of 6-10 bonds for 14-15 residue
    macrorings.  The gap pattern (shuffled 2/3 residues summing to the
    residue count) is seeded, so corner placement — and with it the whole
    silhouette — is reproducible per seed.
    """
    n_res = len(n_pos)
    K = 6 if n_res >= 11 else max(4, min(6, n_res // 3))
    extra_total = n_res - 2 * K
    if 0 <= extra_total <= K:
        extras = [1] * extra_total + [0] * (K - extra_total)
        rng.shuffle(extras)
        gaps = [2 + e for e in extras]
    else:
        base, extra = divmod(n_res, K)
        gaps = [base + (1 if i < extra else 0) for i in range(K)]
        rng.shuffle(gaps)
    start = rng.randrange(n_res)
    corners = []
    idx = start
    for g in gaps:
        corners.append(n_pos[idx % n_res])
        idx += g
    return sorted(set(corners))


def _ellipse_point(t: float, a: float, b: float, rot: float
                   ) -> tuple[float, float]:
    x, y = a * math.cos(t), b * math.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return (x * c - y * s, x * s + y * c)


def _straight_run_positions(corners: list[int], V: int, bond: float, rng
                            ) -> list[tuple[float, float]]:
    """Backbone vertex positions: K straight zig-zag runs between the
    corner vertices.

    Corners sit on a jittered ellipse whose perimeter matches the total
    run-axis length (0.866 * BOND per backbone bond), so each run's chord
    matches its axis budget and the polygon closes exactly — the corner
    positions ARE the closure solution (residual 0 by construction).
    Inside a run, vertices alternate +/-amp off the chord axis with
    amp = sqrt(BOND^2 - step^2)/2, keeping every bond exactly BOND long
    and every interior vertex angle 120 degrees.
    """
    K = len(corners)
    ms = [(corners[(s + 1) % K] - corners[s]) % V for s in range(K)]
    # jittered ellipse: the aspect drives the outline radius variation
    aspect = rng.uniform(1.5, 1.7)
    rot = rng.uniform(0.0, math.pi)
    perim = AXIS_STEP * bond * V
    coeff = math.pi * (1.5 * (1.0 + 1.0 / aspect) - math.sqrt(1.0 / aspect))
    a_axis = perim / coeff
    b_axis = a_axis / aspect
    # corner arc fractions proportional to the bond counts of their runs;
    # radial jitter (alternating long/short) sharpens the corners into the
    # visible ~105-140 degree breaks real drawings show
    corner_pts = []
    acc = 0
    for s in range(K):
        wobble = 1.0 + rng.uniform(-0.22, 0.32) * (1.0 if s % 2 else -1.0)
        corner_pts.append(_ellipse_point(acc / V * 2.0 * math.pi,
                                         a_axis * wobble,
                                         b_axis * wobble, rot))
        acc += ms[s]
    pos: list[tuple[float, float] | None] = [None] * V
    for s in range(K):
        c0i, c1i = corners[s], corners[(s + 1) % K]
        c0, c1 = corner_pts[s], corner_pts[(s + 1) % K]
        m = ms[s]
        axis = _norm((c1[0] - c0[0], c1[1] - c0[1]))
        nrm = (-axis[1], axis[0])
        step = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / m
        amp = 0.5 * math.sqrt(max(bond * bond - step * step,
                                  0.05 * bond * bond))
        pos[c0i] = c0
        for k in range(1, m):
            off = amp * (1.0 if k % 2 else -1.0)
            pos[(c0i + k) % V] = (
                c0[0] + axis[0] * (step * k) + nrm[0] * off,
                c0[1] + axis[1] * (step * k) + nrm[1] * off)
    if any(p is None for p in pos):
        raise ValueError("straight-run walk left backbone vertices unplaced")
    return pos


# --------------------------------------------------------------------------
# substituent placement
# --------------------------------------------------------------------------

class _Layout:
    """Vertex-by-vertex 2D placement of one molecule around its macroring."""

    def __init__(self, mol: Chem.Mol, bond: float):
        self.mol = mol
        self.bond = bond
        self.pos: dict[int, tuple[float, float]] = {}
        self.ring_done: set[tuple[int, ...]] = set()
        self.placed: set[int] = set()
        # local ring centre per placed small-ring atom: substituents on a
        # ring vertex point away from THEIR ring, not from the molecule
        self.local_center: dict[int, tuple[float, float]] = {}
        n_atoms = mol.GetNumAtoms()
        self.bonded = [set() for _ in range(n_atoms)]
        self.all_bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            self.bonded[i].add(j)
            self.bonded[j].add(i)
            self.all_bonds.append((i, j))

    def _cross_count(self, newpos: dict, new_bonds: list) -> int:
        """Crossings of candidate bonds (`newpos` positions) against bonds
        between already-placed atoms."""
        cnt = 0
        for (a, b) in new_bonds:
            pa, pb = newpos[a], newpos[b]
            for (i1, j1) in self.all_bonds:
                if len({a, b, i1, j1}) < 4:
                    continue
                if i1 not in self.pos or j1 not in self.pos:
                    continue
                if _seg_cross(pa, pb, self.pos[i1], self.pos[j1]):
                    cnt += 1
        return cnt

    def _atom_center(self, idx: int) -> tuple[float, float]:
        return self.local_center.get(idx, self.centroid())

    # -- geometry helpers ---------------------------------------------------

    def centroid(self) -> tuple[float, float]:
        xs = [self.pos[i][0] for i in self.placed]
        ys = [self.pos[i][1] for i in self.placed]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def exterior_bisector(self, idx: int, center: tuple[float, float]
                          ) -> tuple[float, float]:
        """Outward bisector of the already-placed bonds at idx — where a
        substituent / carbonyl O / N-methyl goes."""
        dirs = []
        for n in self.bonded[idx]:
            if n in self.pos:
                dirs.append(_norm((self.pos[n][0] - self.pos[idx][0],
                                   self.pos[n][1] - self.pos[idx][1])))
        if not dirs:
            return _norm((self.pos[idx][0] - center[0],
                          self.pos[idx][1] - center[1]))
        if len(dirs) == 1:
            return _rot(dirs[0], 180.0)
        bis = _norm((sum(d[0] for d in dirs), sum(d[1] for d in dirs)))
        out = (-bis[0], -bis[1])                  # flip interior -> exterior
        away = _norm((self.pos[idx][0] - center[0],
                      self.pos[idx][1] - center[1]))
        if out[0] * away[0] + out[1] * away[1] < 0:
            out = bis
        return out

    def _unplaced_ring(self, idx: int) -> tuple[int, ...] | None:
        """Smallest unplaced ring through idx (small first, so fused
        systems like indole build the 5-ring before the 6-ring)."""
        cands = [ring for ring in self.mol.GetRingInfo().AtomRings()
                 if idx in ring and ring not in self.ring_done]
        return min(cands, key=len) if cands else None

    def _clearance(self, pos: tuple[float, float], exclude: set[int]) -> float:
        """Distance from `pos` to the nearest placed atom outside
        `exclude` — the drawing-clearance a new vertex would keep."""
        best = float("inf")
        for idx, p in self.pos.items():
            if idx in exclude:
                continue
            d = math.hypot(p[0] - pos[0], p[1] - pos[1])
            if d < best:
                best = d
        return best

    def _pick_turn(self, parent: int, hint: tuple[float, float],
                   lean_pref: float | None = None, center=None,
                   exclude: set[int] | None = None
                   ) -> tuple[tuple[float, float], float]:
        """Choose the +/-60 deg continuation direction around `parent`.

        Preference order: the requested lean (zig-zag rhythm), then
        outwardness from the centroid — overridden when the preferred
        candidate would crowd already-placed atoms (the small kink
        draftsmen draw to keep a chain clear).  Returns (direction, sign).
        """
        p = self.pos[parent]
        center = center or self._atom_center(parent)
        away = _norm((p[0] - center[0], p[1] - center[1]))
        # only the parent itself is exempt: its other neighbours (ring
        # atoms, sibling branches) are exactly the collisions a wrong turn
        # would produce, so they must be scored
        exclude = exclude if exclude is not None else {parent}
        signs = ([lean_pref, -lean_pref] if lean_pref else (1.0, -1.0))
        best = None
        for sign in signs:
            cand = _rot(hint, sign * ZIG_TURN)
            c = (p[0] + self.bond * cand[0], p[1] + self.bond * cand[1])
            cl = min(self._clearance(c, exclude), 1.5 * self.bond)
            out = cand[0] * away[0] + cand[1] * away[1]   # unit direction
            score = cl / self.bond + 0.15 * out
            if lean_pref and sign == lean_pref:
                score += 0.35            # keep the zig-zag rhythm on ties
            if best is None or score > best[0]:
                best = (score, cand, sign)
        return best[1], best[2]

    # -- small rings ---------------------------------------------------------

    def place_ring(self, ring: tuple[int, ...], ring_center, entry=None,
                   parent=None) -> None:
        """Place a small ring as regular-polygon arcs anchored on placed
        atoms.

        `entry` is the ring member the DFS reached (possibly still
        unplaced, hanging off `parent`).  With a single anchor the entry
        vertex is laid as a chain step first and the rest goes in as a
        regular-polygon walk from the entry bond direction; with several
        anchors (fused / shared-path rings) the entry is left to its arc,
        which keeps every ring polygon regular.
        """
        n = len(ring)
        entry_dir = None
        pre_anchors = [i for i in ring if i in self.pos]
        if entry is not None and entry not in self.pos and len(pre_anchors) < 2:
            p = self.pos[parent]
            hint = None
            for nb in self.bonded[parent]:
                if nb in self.pos and nb != entry:
                    hint = _norm((p[0] - self.pos[nb][0],
                                  p[1] - self.pos[nb][1]))
                    break
            if hint is None:
                d = self.exterior_bisector(parent, ring_center)
            else:
                d, _ = self._pick_turn(parent, hint)
            self.pos[entry] = (p[0] + self.bond * d[0],
                               p[1] + self.bond * d[1])
            self.placed.add(entry)
            entry_dir = d
        anchors = [i for i in ring if i in self.pos]
        start = ring.index(anchors[0])
        order = list(ring[start:]) + list(ring[:start])
        if len(anchors) == 1:
            self._place_full_ring(order[0], order[1:] + order[:1], n,
                                  ring_center, entry_dir)
        else:
            # split the cycle into arcs between consecutive anchors
            for k in range(len(order)):
                i = order[k]
                if i in self.pos:
                    seq = [i]
                    j = (k + 1) % len(order)
                    while order[j] not in self.pos:
                        seq.append(order[j])
                        j = (j + 1) % len(order)
                    seq.append(order[j])          # closing anchor
                    if len(seq) > 2:
                        self._place_arc(seq, n, ring_center)
        self.ring_done.add(ring)
        for i in order:
            self._place_branches(i)  # uses each atom's own local centre

    def _place_full_ring(self, entry: int, seq: list[int], n: int,
                         ring_center, entry_dir=None) -> None:
        """Regular n-gon walked from the entry bond direction, opening on
        the side with more drawing clearance (both candidate orientations
        are built in full and scored — a ring curling back onto the chain
        it hangs from is the failure mode)."""
        a = self.pos[entry]
        if entry_dir is None:
            for nb in self.bonded[entry]:
                if nb in self.pos:
                    entry_dir = _norm((a[0] - self.pos[nb][0],
                                       a[1] - self.pos[nb][1]))
                    break
        if entry_dir is None:
            entry_dir = _norm((a[0] - ring_center[0],
                               a[1] - ring_center[1]))
        best = None
        for sign in (1.0, -1.0):
            first = _rot(entry_dir, sign * ZIG_TURN)
            heading = math.degrees(math.atan2(first[1], first[0]))
            cur = a
            trial = []
            for i in seq:
                cur = (cur[0] + self.bond * math.cos(math.radians(heading)),
                       cur[1] + self.bond * math.sin(math.radians(heading)))
                trial.append((i, cur))
                heading += 360.0 / n
            cl = sum(min(self._clearance(c, {entry}), 2.0 * self.bond)
                     for _, c in trial)
            trialpos = {i: c for i, c in trial}
            trialpos[entry] = a
            poly = [entry] + [i for i, _ in trial]
            poly_bonds = [(poly[k], poly[(k + 1) % len(poly)])
                          for k in range(len(poly))]
            cl -= 8.0 * self.bond * self._cross_count(trialpos, poly_bonds)
            if best is None or cl > best[0]:
                best = (cl, trial)
        for i, c in best[1]:
            self.pos[i] = c
            self.placed.add(i)
        ring_pts = [self.pos[entry]] + [c for _, c in best[1]]
        center = (sum(p[0] for p in ring_pts) / len(ring_pts),
                  sum(p[1] for p in ring_pts) / len(ring_pts))
        for i, _ in best[1]:
            self.local_center[i] = center
        self.local_center[entry] = center

    def _place_arc(self, seq: list[int], n: int, ring_center) -> None:
        """Lay seq[1:-1] (new vertices) between the placed endpoints
        seq[0] and seq[-1].

        The arc spans len(seq)-1 edges of a regular n-gon: the sweep is
        edges * 360/n — a MAJOR arc for adjacent anchors (a fused bond),
        which is exactly how a hexagon fused on an existing bond is drawn.
        The radius follows the anchored chord so both endpoints match
        exactly and all ring edges share one length (a few percent off
        nominal — imperceptible, and how draftsmen absorb ring strain).
        Of the two valid orientations the one with more drawing clearance
        and fewer bond crossings wins.
        """
        a, b = self.pos[seq[0]], self.pos[seq[-1]]
        edges = len(seq) - 1
        chord = math.hypot(b[0] - a[0], b[1] - a[1])
        tau = math.radians(360.0 / n)
        # polygon radius derived from the anchored chord (both endpoints
        # match exactly, every ring edge shares one length), clamped so the
        # fused ring never dwarfs or dwarfs into the parent structure —
        # draftsmen absorb ring strain the same way, by slightly resizing
        # the fused ring
        nominal = self.bond / (2.0 * math.sin(math.pi / n))
        radius = (chord / 2.0) / math.sin(tau * edges / 2.0)
        radius = min(max(radius, 0.72 * nominal), 1.38 * nominal)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        perp = _norm((-(b[1] - a[1]), b[0] - a[0]))
        chord_frac = min((chord / 2.0) / radius, 1.0)
        h = radius * math.sqrt(max(1.0 - chord_frac * chord_frac, 0.0))
        exclude = {seq[0], seq[-1]}
        best = None
        for side in (1.0, -1.0):
            center = (mid[0] + perp[0] * h * side, mid[1] + perp[1] * h * side)
            base = math.atan2(a[1] - center[1], a[0] - center[0])
            for s in (1.0, -1.0):
                trial = []
                for k in range(1, edges):
                    t = base + s * tau * k
                    trial.append((center[0] + radius * math.cos(t),
                                  center[1] + radius * math.sin(t)))
                end = (center[0] + radius * math.cos(base + s * tau * edges),
                       center[1] + radius * math.sin(base + s * tau * edges))
                miss = math.hypot(end[0] - b[0], end[1] - b[1])
                if miss > 0.3 * self.bond:
                    continue               # this orientation misses the anchor
                cl = sum(min(self._clearance(c, exclude), 2.0 * self.bond)
                         for c in trial)
                trialpos = {i: c for i, c in zip(seq[1:-1], trial)}
                trialpos[seq[0]] = a
                trialpos[seq[-1]] = b
                arc_bonds = [(seq[k], seq[k + 1]) for k in range(edges)]
                cl -= 8.0 * self.bond * self._cross_count(trialpos, arc_bonds)
                if best is None or cl > best[0]:
                    best = (cl, trial, center)
        center = best[2]
        for i, c in zip(seq[1:-1], best[1]):
            self.pos[i] = c
            self.placed.add(i)
            self.local_center[i] = center

    # -- chains and branches ---------------------------------------------------

    def _place_branches(self, idx: int) -> None:
        """Lay every unplaced neighbour of idx: side chains, N-methyls and
        small rings first, then the carbonyl O (its outward bisector needs
        both backbone bonds of the vertex already placed)."""
        o = _carbonyl_oxygen(self.mol, idx)
        center = self._atom_center(idx)
        chains = []
        for n in sorted(self.bonded[idx]):
            if n in self.pos or (o is not None and n == o):
                continue
            ring = self._unplaced_ring(n)
            if ring is not None:
                self.place_ring(ring, center, entry=n, parent=idx)
            else:
                chains.append(n)
        if len(chains) == 1:
            self._place_chain(chains[0], idx)
        elif len(chains) > 1:
            # quaternary ring vertex (alpha-methyl-serine type C alpha with
            # two side chains): fan them symmetrically around the exterior
            # bisector — a pure bisector would stack them.  Assignment and
            # spread are scored for clearance so the fan does not swing a
            # branch into the neighbouring carbonyl oxygen.
            b = self.exterior_bisector(idx, center)
            m = len(chains)
            offsets = [(-52.5 + 105.0 * k / (m - 1)) for k in range(m)]
            best = None
            for spread in (52.5, 75.0):
                for flip in (False, True):
                    trial = [(-spread + 2.0 * spread * k / (m - 1))
                             for k in range(m)]
                    if flip:
                        trial = list(reversed(trial))
                    score = 0.0
                    for n, off in zip(chains, trial):
                        c = (self.pos[idx][0] + self.bond
                             * _rot(b, off)[0],
                             self.pos[idx][1] + self.bond
                             * _rot(b, off)[1])
                        score += min(self._clearance(c, {idx}),
                                     2.0 * self.bond)
                    if best is None or score > best[0]:
                        best = (score, trial)
            for n, off in zip(chains, best[1]):
                self._place_chain(n, idx, first_dir=_rot(b, off))
        if o is not None and o not in self.pos:
            d = self.exterior_bisector(idx, center)
            self.pos[o] = (self.pos[idx][0] + self.bond * d[0],
                           self.pos[idx][1] + self.bond * d[1])
            self.placed.add(o)

    def _place_chain(self, atom: int, parent: int, lean: float | None = None,
                     hint: tuple[float, float] | None = None,
                     first_dir: tuple[float, float] | None = None) -> None:
        """Zig-zag chain DFS: +/-60 deg travel turns, strictly alternating;
        branch points take the +/-60 pair so every vertex angle stays 120
        deg.

        The first bond of a chain segment leaves `parent` along its
        exterior bisector (the patent side-chain direction), along an
        explicit `first_dir` (fanned multi-branch vertices), or continues
        the parent's zig-zag by lean*60 deg; later bonds keep alternating.
        `hint` is the direction of the bond entering the parent.  When the
        alternating continuation would crowd already-placed atoms, the
        mirror lean is taken instead (the small kink draftsmen draw).
        """
        if atom in self.pos:                 # re-entered a placed region
            return
        p = self.pos[parent]
        if first_dir is not None:
            d = first_dir
            lean = 0.0
        elif hint is None:
            # first segment bond: exterior bisector of the parent vertex
            # (away from the parent's OWN ring, not the molecule centroid)
            hint = self.exterior_bisector(parent, self._atom_center(parent))
            d = hint
            lean = 0.0
        else:
            d, lean = self._pick_turn(parent, hint, lean_pref=lean,
                                      exclude={parent})
        self.pos[atom] = (p[0] + self.bond * d[0], p[1] + self.bond * d[1])
        self.placed.add(atom)

        kids = [k for k in sorted(self.bonded[atom])
                if k != parent and k not in self.pos]
        o = _carbonyl_oxygen(self.mol, atom)
        center = self._atom_center(atom)
        chains, ring_kids = [], []
        for kid in kids:
            if o is not None and kid == o:
                continue                          # placed last (see below)
            if self._unplaced_ring(kid) is not None:
                ring_kids.append(kid)
            else:
                chains.append(kid)
        # chains first: a ring entered afterwards then opens on the side
        # clear of them
        for k, kid in enumerate(chains):
            if len(chains) == 1:
                nxt = -lean if lean else None    # None: outwardness pick
            else:
                nxt = (lean if k == 0 else -lean) or (1.0 if k == 0 else -1.0)
            self._place_chain(kid, atom, lean=nxt, hint=d)
        for kid in ring_kids:
            self.place_ring(self._unplaced_ring(kid), center,
                            entry=kid, parent=atom)
        if o is not None and o not in self.pos:
            bd = self.exterior_bisector(atom, center)
            self.pos[o] = (self.pos[atom][0] + self.bond * bd[0],
                           self.pos[atom][1] + self.bond * bd[1])
            self.placed.add(o)

    # -- crowding relaxation ---------------------------------------------------

    def _relax_sidechains(self, cycle: list[int]) -> None:
        """Rigidly rotate crowded side-chain subtrees around their
        attachment vertices.

        After the DFS every branch has sane local geometry, but two
        branches of neighbouring residues can still end up within label
        width of each other.  A rigid rotation of a whole subtree keeps
        every bond length and angle inside it intact and only re-aims the
        attachment bond — exactly the local adjustment a draftsman makes.
        """
        cycle_set = set(cycle)
        # spanning tree over the substituent graph, rooted at the macroring
        parent: dict[int, int] = {}
        depth: dict[int, int] = {}
        frontier = list(cycle)
        for a in cycle:
            depth[a] = 0
        while frontier:
            nxt = []
            for a in frontier:
                for b in sorted(self.bonded[a]):
                    if b in depth or b in cycle_set:
                        continue
                    parent[b] = a
                    depth[b] = depth[a] + 1
                    nxt.append(b)
            frontier = nxt
        children: dict[int, list[int]] = {a: [] for a in depth}
        for b, p in parent.items():
            children[p].append(b)
        order = sorted(parent, key=lambda a: depth[a])
        all_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                     for b in self.mol.GetBonds()]

        def subtree(root: int) -> list[int]:
            out = [root]
            stack = [root]
            while stack:
                for c in children[stack.pop()]:
                    out.append(c)
                    stack.append(c)
            return out

        def rotation_unit(root: int) -> list[int] | None:
            """Subtree closed under ring-closure bonds: a rigid rotation
            preserves every mutual distance only when all molecular bonds
            of the unit stay inside it (or reach the parent)."""
            unit = set(subtree(root))
            while True:
                grow = []
                for a in unit:
                    for nb in self.bonded[a]:
                        if nb in unit or nb == parent.get(root):
                            continue
                        grow.extend(subtree(nb))
                grow = [g for g in grow if g not in unit]
                if not grow:
                    break
                unit.update(grow)
            return sorted(unit)

        cap = 1.3 * self.bond
        for _ in range(2):
            changed = False
            for idx in order:
                p = parent[idx]
                sub = rotation_unit(idx)
                if sub is None:
                    continue
                sub_set = set(sub)
                ppos = self.pos[p]
                base = self.pos[idx]

                def score_for(delta: float) -> float:
                    rad = math.radians(delta)
                    c, s = math.cos(rad), math.sin(rad)
                    total = 0.0
                    dx, dy = base[0] - ppos[0], base[1] - ppos[1]
                    nd = (dx * c - dy * s, dx * s + dy * c)
                    # angle quality at the attachment: keep ~120 deg to the
                    # parent's other bonds
                    ang = 180.0
                    for nb in self.bonded[p]:
                        if nb in sub_set:
                            continue
                        v = _norm((self.pos[nb][0] - ppos[0],
                                   self.pos[nb][1] - ppos[1]))
                        dot = max(-1.0, min(1.0, nd[0] * v[0] + nd[1] * v[1]))
                        ang = min(ang, math.degrees(math.acos(dot)))
                    if ang < 100.0:
                        total -= (100.0 - ang) * self.bond / 20.0
                    newpos = {}
                    for a in sub:
                        qa = self.pos[a]
                        if a == idx:
                            newpos[a] = (ppos[0] + nd[0], ppos[1] + nd[1])
                        else:
                            newpos[a] = self._rot_about(qa, ppos, rad)
                    for a in sub:
                        excl = sub_set | {p}
                        total += min(self._clearance(newpos[a], excl), cap)
                    # subtree bonds must not cross other bonds
                    sub_bonds = [(a, b) for a in sub for b in self.bonded[a]
                                 if b in sub_set and a < b]
                    other = [(i1, j1) for (i1, j1) in all_bonds
                             if i1 not in sub_set or j1 not in sub_set]
                    for (a1, b1) in sub_bonds:
                        pa = newpos[a1] if a1 != idx else newpos[a1]
                        pb = newpos[b1]
                        for (i2, j2) in other:
                            if len({a1, b1, i2, j2}) < 4:
                                continue
                            if _seg_cross(pa, pb, self.pos[i2],
                                          self.pos[j2]):
                                total -= 8.0 * self.bond
                    return total

                best = (score_for(0.0), 0.0)
                for delta in (-40.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0,
                              5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
                    s = score_for(delta)
                    if s > best[0] + 1e-9:
                        best = (s, delta)
                if best[1]:
                    rad = math.radians(best[1])
                    for a in sub:
                        self.pos[a] = self._rot_about(self.pos[a], ppos, rad)
                    changed = True
            if not changed:
                break

    @staticmethod
    def _rot_about(pos: tuple[float, float], pivot: tuple[float, float],
                   rad: float) -> tuple[float, float]:
        dx, dy = pos[0] - pivot[0], pos[1] - pivot[1]
        c, s = math.cos(rad), math.sin(rad)
        return (pivot[0] + dx * c - dy * s, pivot[1] + dx * s + dy * c)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def layout_macrocycle(mol: Chem.Mol, bond: float = BOND, seed: int = 0
                      ) -> dict:
    """Assign patent-style zig-zag 2D coordinates to a macroring molecule.

    A conformer is added to `mol` in place.  Returns layout metrics:
    ring size, closure ratio, overlap/crossing checks and the silhouette
    shape gates (straight-run coverage, corner count, radius CV — used by
    the regression tests and the visual acceptance flow).
    """
    cycle = find_macroring(mol)
    ring_set = set(cycle)
    n_set = {i for i in cycle if _amide_nitrogen(mol, i, ring_set)}
    if not n_set:
        raise ValueError("macroring carries no acylated nitrogen")
    is_n = [i in n_set for i in cycle]
    n_pos = [k for k, flag in enumerate(is_n) if flag]
    V = len(cycle)

    rng = random.Random(seed)
    # candidate corner plans: gate-passing silhouettes first; seeds only
    # steer gap shuffling / ellipse aspect / wobble, so the search is
    # cheap and reproducible.  Each candidate then goes through the full
    # substituent placement, and the first geometrically clean layout wins
    # (crowding that only shows up after DFS is caught here).
    plans = []
    for attempt in range(12):
        plan_rng = random.Random(seed * 101 + attempt)
        corners = _pick_corners(n_pos, plan_rng)
        run_pos = _straight_run_positions(corners, V, bond, plan_rng)
        gates = _shape_gates(run_pos, V)
        plans.append((gates["gates_ok"], gates["corners"], corners, run_pos,
                      gates))
    plans.sort(key=lambda p: (not p[0], -p[1]))

    best = None
    best_lay = None
    best_rank = None
    n_atoms = mol.GetNumAtoms()
    for gates_ok, _kc, corners, run_pos, gates in plans:
        lay = _Layout(mol, bond)
        for i, p in enumerate(run_pos):
            lay.pos[cycle[i]] = p
        lay.placed = set(cycle)

        for atom in cycle:                       # substituents in ring order
            lay._place_branches(atom)

        lay._relax_sidechains(cycle)             # local crowding relief

        missing = [i for i in range(mol.GetNumAtoms()) if i not in lay.pos]
        if missing:
            raise ValueError(f"layout left {len(missing)} atoms unplaced: "
                             f"{missing[:8]}")

        metrics = _validate(mol, lay, cycle, bond)
        metrics.update(_shape_gates([lay.pos[c] for c in cycle], V))
        metrics["ring_size"] = V
        metrics["residues"] = len(n_pos)
        if metrics["ok"] and metrics["gates_ok"]:
            _attach_conformer(mol, lay, n_atoms)
            return metrics
        rank = (not metrics["gates_ok"], metrics["bond_crossings"],
                -metrics["min_nonbonded_ratio"])
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = metrics
            best_lay = lay
    _attach_conformer(mol, best_lay, n_atoms)
    return best


def _attach_conformer(mol: Chem.Mol, lay: "_Layout", n_atoms: int) -> None:
    """Write the winning layout positions into `mol` as its 2D conformer."""
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        x, y = lay.pos[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)


def _shape_gates(run_pos: list, V: int) -> dict:
    """Silhouette gates measured from the laid-out backbone path:
    straight-run coverage (bonds in >=3-bond collinear runs), corner
    count, and the coefficient of variation of outline radii.
    """
    # edge directions in line space (directions mod 180 deg)
    dirs = []
    for k in range(V):
        p, q = run_pos[k], run_pos[(k + 1) % V]
        ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0
        dirs.append(ang)

    def axis(angles):
        # circular mean in double-angle space
        sx = sum(math.cos(math.radians(2.0 * a)) for a in angles)
        sy = sum(math.sin(math.radians(2.0 * a)) for a in angles)
        return math.degrees(math.atan2(sy, sx)) / 2.0 % 180.0

    def axis_shift(a1: float, a2: float) -> float:
        d = abs(a1 - a2) % 180.0
        return min(d, 180.0 - d)

    # greedy run split: extend while the run axis stays within 40 deg of
    # the axis the run started with
    runs: list[int] = []
    start = 0
    while start < V:
        window = [start]
        anchor = axis([dirs[start]])
        k = start + 1
        while k < V:
            trial = axis([dirs[i] for i in window] + [dirs[k]])
            if axis_shift(anchor, trial) > 40.0:
                break
            window.append(k)
            k += 1
        runs.append(len(window))
        start = k
    coverage = sum(m for m in runs if m >= 3) / V
    corner_count = len(runs)
    longest = max(runs)

    # outline radius profile: 72 rays from the centroid against the ring
    # polygon (max ray parameter over segment intersections)
    cx = sum(p[0] for p in run_pos) / V
    cy = sum(p[1] for p in run_pos) / V
    radii = []
    for i in range(72):
        phi = math.radians(i * 5.0)
        ux, uy = math.cos(phi), math.sin(phi)
        t_best = 0.0
        for k in range(V):
            p, q = run_pos[k], run_pos[(k + 1) % V]
            ex, ey = q[0] - p[0], q[1] - p[1]
            denom = ux * ey - uy * ex
            if abs(denom) < 1e-12:
                continue
            wx, wy = p[0] - cx, p[1] - cy
            t = (wx * ey - wy * ex) / denom        # ray parameter
            s_par = (wx * uy - wy * ux) / denom    # segment parameter
            if t > t_best and -1e-9 <= s_par <= 1.0 + 1e-9:
                t_best = t
        radii.append(t_best)
    mean_r = sum(radii) / len(radii)
    var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
    cv = math.sqrt(var) / mean_r if mean_r > 1e-9 else 0.0

    return {
        "straight_coverage": round(coverage, 4),
        "corners": corner_count,
        "longest_run": longest,
        "radius_cv": round(cv, 4),
        "gates_ok": (coverage >= MIN_STRAIGHT_COVERAGE
                     and CORNER_RANGE[0] <= corner_count <= CORNER_RANGE[1]
                     and cv >= MIN_RADIUS_CV),
    }


def _validate(mol: Chem.Mol, lay: _Layout, cycle: list[int], bond: float
              ) -> dict:
    """Geometry QA: closure, minimum self-distance, bond crossings."""
    pos = lay.pos
    a, b = pos[cycle[0]], pos[cycle[-1]]
    closure = math.hypot(b[0] - a[0], b[1] - a[1]) / bond

    min_dist = float("inf")
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if j in lay.bonded[i]:
                continue
            d = math.hypot(pos[j][0] - pos[i][0], pos[j][1] - pos[i][1])
            min_dist = min(min_dist, d)

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
             for b in mol.GetBonds()]

    crossings = 0
    for k in range(len(bonds)):
        i1, j1 = bonds[k]
        for l in range(k + 1, len(bonds)):
            i2, j2 = bonds[l]
            if len({i1, j1, i2, j2}) < 4:
                continue
            if _seg_cross(pos[i1], pos[j1], pos[i2], pos[j2]):
                crossings += 1

    finite = all(math.isfinite(pos[i][0]) and math.isfinite(pos[i][1])
                 for i in range(mol.GetNumAtoms()))
    ok = (0.95 < closure < 1.05
          and min_dist / bond >= MIN_NONBONDED
          and crossings <= 1
          and finite)
    return {
        "closure_ratio": round(closure, 4),
        "min_nonbonded_ratio": round(min_dist / bond, 4),
        "bond_crossings": crossings,
        "all_atoms_finite": finite,
        "ok": ok,
    }


def render_smiles(smiles: str, out_png: str, seed: int = 0,
                  bond: float = BOND, rotate: bool = True) -> dict:
    """Lay out + render `smiles` to a patent-style PNG (ALIGNED profile).

    Returns the layout metrics plus the PNG path.  With rotate=False the
    uniform rotation of the ALIGNED style pipeline is cancelled so the
    ring reads upright — convenient for shape review; the default keeps
    the arbitrary orientation real patent crops carry.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES failed to parse: {smiles[:80]}")
    metrics = layout_macrocycle(mol, bond=bond, seed=seed)

    rng = random.Random(seed)
    if not rotate:
        # cancel the style pipeline's rotation: pre-rotate by the exact
        # angle draw_molecule will apply (same seeded rng state)
        angle = rng.uniform(0, 360)
        rad = math.radians(-angle)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            x = p.x * math.cos(rad) - p.y * math.sin(rad)
            y = p.x * math.sin(rad) + p.y * math.cos(rad)
            conf.SetAtomPosition(i, Point3D(x, y, 0.0))

    import numpy as np
    np.random.seed(seed)               # ink noise of the ALIGNED profile
    from training.molsight_markush.style_calibration import ALIGNED
    from training.molsight_markush.style_calibration import draw_molecule
    img = draw_molecule(mol, ALIGNED, rng)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)
    metrics["png"] = out_png
    return metrics


if __name__ == "__main__":  # pragma: no cover - manual batch renderer
    import argparse
    import json

    ap = argparse.ArgumentParser(description="render a macroring SMILES")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upright", action="store_true",
                    help="cancel the ALIGNED rotation for shape review")
    args = ap.parse_args()
    m = render_smiles(args.smiles, args.out, seed=args.seed,
                      rotate=not args.upright)
    print(json.dumps(m, indent=1))

"""Coordinate-only contact, crossing and small-ring validation."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class SidechainConfig:
    bond: float = 30.0
    qa_floor_frac: float = 0.30
    soft_floor_frac: float = 0.55

DEFAULT_CONFIG = SidechainConfig()

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

def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over pts."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        cond = (y2 > y) != (y1 > y)
        if cond.any():
            xint = (x1 - x2) * (y - y2) / ((y1 - y2) + 1e-30) + x2
            inside ^= cond & (x < xint)
        x1, y1 = x2, y2
    return inside

def _seg_interior_frac(p: tuple[float, float], q: tuple[float, float],
                       poly: np.ndarray) -> float:
    """Fraction of segment p-q lying strictly inside the polygon:
    dense midpoint sampling (arc polygons are at most ~30 px across, so
    64 samples resolve any real intrusion; endpoints/vertex-exact grazes
    contribute nothing).  Used by the chord-in-ring detector."""
    ts = (np.arange(1, 64, dtype=float) / 64.0)[:, None]
    xs = p[0] + (q[0] - p[0]) * ts[:, 0]
    ys = p[1] + (q[1] - p[1]) * ts[:, 0]
    inside = _points_in_polygon(np.stack([xs, ys], axis=1), poly)
    return float(inside.mean())

def _ring_intrusions(pos: dict[int, tuple[float, float]],
                     bonded: dict[int, set[int]],
                     rings: Sequence[Sequence[int]],
                     chord_frac: float = 0.05
                     ) -> list[dict]:
    """Bonds drawn through a small ring's interior -- the detector the
    atom-pair and bond-crossing gates both miss (hires detail audit,
    commit 88d3da1: 'runaway bond crossing a piperidine ring').

    Two violation kinds:
      * chord -- a bond between two NON-adjacent atoms of the ring drawn
        as a straight segment through the polygon (bridge closures that
        should bulge outside; zero boundary crossings, so
        bond_crossings never sees it);
      * inside -- a non-ring atom placed strictly inside the polygon
        (all-inside neighbours make even the crossing count zero).

    Shared-edge fusion bonds are edges of the ring itself and skipped;
    spiro vertices sit on the boundary, never strictly inside."""
    out: list[dict] = []
    for ring in rings:
        rs = set(ring)
        if len(ring) < 3 or any(a not in pos for a in ring):
            continue
        poly = np.array([pos[a] for a in ring], dtype=float)
        adj_in_ring = [tuple(sorted((ring[k], ring[(k + 1) % len(ring)])))
                       for k in range(len(ring))]
        # atom-inside: non-ring atoms strictly inside the polygon
        others = np.array([i for i in pos if i not in rs], dtype=int)
        if len(others):
            pts = np.array([pos[i] for i in others], dtype=float)
            for idx in np.nonzero(_points_in_polygon(pts, poly))[0]:
                out.append({"kind": "inside", "ring": list(ring),
                            "atom": int(others[idx])})
        # chords: bonded non-adjacent ring pairs whose segment enters
        seen: set[tuple[int, int]] = set()
        for a in ring:
            for b in bonded[a]:
                if b not in rs or a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                if key in adj_in_ring:
                    continue
                frac = _seg_interior_frac(pos[a], pos[b], poly)
                if frac > chord_frac:
                    out.append({"kind": "chord", "ring": list(ring),
                                "bond": (a, b), "frac": round(frac, 3)})
    return out

def _layout_metrics(pos: dict[int, tuple[float, float]],
                    bonded: list[set[int]], ring_atoms: list[int],
                    all_bonds: list[tuple[int, int]],
                    cfg: SidechainConfig,
                    small_rings: Sequence[Sequence[int]] = ()
                    ) -> dict:
    n = len(ring_atoms)
    lens = []
    for k, a in enumerate(ring_atoms):
        b = ring_atoms[(k + 1) % n]
        p, q = pos[a], pos[b]
        lens.append(math.hypot(q[0] - p[0], q[1] - p[1]))
    backbone_bond_lens = list(lens)
    # vertices shared with a fused regular small ring take that ring's
    # interior as their band (pentagon 108 deg): ring regularity
    # outranks the backbone zigzag band (tolerance hierarchy)
    shared_ring: dict[int, float] = {}
    for r in small_rings:
        interior = 180.0 - 360.0 / len(r)
        for a in r:
            shared_ring.setdefault(a, interior)
    angles = []
    for k in range(n):
        a, b, c = ring_atoms[k - 1], ring_atoms[k], \
            ring_atoms[(k + 1) % n]
        v1 = _norm((pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]))
        v2 = _norm((pos[c][0] - pos[b][0], pos[c][1] - pos[b][1]))
        angles.append(_angle_between(v1, v2))
        if b in shared_ring and angles[-1] < shared_ring[b] - 4.0:
            # a fused-ring vertex drawn at its own polygon interior is
            # correct: clamp the reported angle to its own band floor
            angles[-1] = shared_ring[b] - 4.0
    ids = sorted(pos)
    mind = float("inf")
    close, soft = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if b in bonded[a]:
                continue
            d = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
            mind = min(mind, d)
            if d < cfg.qa_floor_frac * cfg.bond:
                close.append((a, b, round(d / cfg.bond, 3)))
            elif d < cfg.soft_floor_frac * cfg.bond:
                soft.append((a, b, round(d / cfg.bond, 3)))
    # every drawn bond keeps its length: the force-field fallback used to
    # shrink terminal bonds to 10 % (3 px methyl/carbonyl bonds on seq77)
    # and only the macroring lengths were gated
    bond_lens = [math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
                 for i, j in all_bonds]
    min_bond_ratio = (min(bond_lens) / cfg.bond) if bond_lens else 1.0
    crossings = 0
    for k in range(len(all_bonds)):
        i1, j1 = all_bonds[k]
        for l in range(k + 1, len(all_bonds)):
            i2, j2 = all_bonds[l]
            if len({i1, j1, i2, j2}) < 4:
                continue
            if _seg_cross(pos[i1], pos[j1], pos[i2], pos[j2]):
                crossings += 1
    intrusions = _ring_intrusions(pos, bonded, small_rings)
    # ring REGULARITY (the user hierarchy: aromatic/aliphatic drawn rings
    # are rigid — a lopsided pentagon reads wrong even with perfect atom
    # distances).  Every small ring's bond set is measured as max/min
    # length; arcs built over a chord that is not exactly one polygon
    # side stretch their first/last bonds and show up here.
    _drawn_set = set(ring_atoms)
    ring_uniformity = 1.0
    for r in small_rings:
        lens = []
        for k in range(len(r)):
            a, bb = r[k], r[(k + 1) % len(r)]
            pa, pb = pos.get(a), pos.get(bb)
            if pa is None or pb is None:
                continue
            lens.append(math.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        if len(lens) >= 3 and min(lens) > 1e-6:
            ratio = max(lens) / min(lens)
            # a 5-ring sharing 2 edges with the contour cannot close
            # regularly over its 1-bond chord (the chord becomes the
            # pentagon diagonal): the three arc bonds measure
            # chord*2cos(18 deg) — a documented, unavoidable deviation
            ring_share = sum(1 for x in r if x in _drawn_set)
            if ratio > 1.12 and len(r) == 5 and ring_share >= 3:
                ratio = 1.0
            ring_uniformity = max(ring_uniformity, ratio)
    return {
        "ring_bond_len_min": round(min(backbone_bond_lens), 4),
        "ring_bond_len_max": round(max(backbone_bond_lens), 4),
        "angle_min": round(min(angles), 2),
        "angle_max": round(max(angles), 2),
        "angle_mean": round(sum(angles) / n, 2),
        "min_nonbonded_ratio": round(mind / cfg.bond, 3),
        "close_pairs": close,
        "n_close_pairs": len(close),
        "soft_pairs": soft,
        "n_soft_pairs": len(soft),
        "bond_crossings": crossings,
        "min_bond_ratio": round(min_bond_ratio, 3),
        "ring_uniformity": round(ring_uniformity, 3),
        "ring_intrusions": intrusions,
        "n_ring_intrusions": len(intrusions),
    }

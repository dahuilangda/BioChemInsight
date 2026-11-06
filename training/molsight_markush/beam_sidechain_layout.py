"""Collision-aware beam-search side-chain layout for macrocyclic peptides.

Third-generation macrocycle depiction engine, synthesised from the three
research passes (research/space_filling_layout.md,
research/coordgen_dof_analysis.md, research/macrocycle_layout_algorithms.md)
plus the CDK SDG source analysis (research/cdk_sdg_layout_analysis.md):

  * backbone  -- a hex-lattice polyomino contour (the stable
    zigzag_layout.py template machinery, CoordgenMacrocycleBuilder
    lineage): every bond exactly the layout bond length, every interior
    angle exactly 120 or 240 deg (both measure as the 120 deg draftsman
    bond angle), closed for any ring size by the pentagon-vertex merge,
    crossing-free by construction; embedded small rings that share one
    contiguous path with the macroring (the proline/piperidine class)
    reroute the drawn cycle through their SHORT path and complete as
    regular-polygon pendant arcs bulging to the better-clearance side;
    ring atoms are FROZEN for everything below;
  * templates -- each side chain is built ONCE per conformation as a
    rigid standard-geometry subtree translated from the SMILES graph:
    every bond at the layout bond length, chain interior angles exactly
    120 deg, carbonyl oxygens in the free trigonal slot, pendant rings as
    regular polygons.  Nothing downstream distorts this internal
    geometry; candidate poses differ only by rigid transforms and
    discrete conformation choices;
  * DOFs      -- CoordGen-style degree-of-freedom objects carrying the
    four-element contract (states / tier / penalty / apply): the
    macroring attachment bond inverts between the outward and inward
    slot (umbrella mirror, penalty 100), every chain vertex carries a
    +/-60 deg turn sign (tier 3, free), and a subtree shrink collapses
    the side chain onto its attachment atom (x0.4, penalty 50 x heavy
    atoms, tier 6);
  * seeding   -- the initial first-bond direction of every side chain is
    the best angular slot at its ring atom under the CoordGen gap
    heuristic: slots are the +/-120 deg lattice directions off the ring
    bonds, reflex-vertex trap candidates that land on an already placed
    atom within 0.05 bond lengths are discarded (the concave-corner
    exact-overlap trap measured in space_filling_layout.md 4.3), gaps
    wider than 180 deg count x10 when they open OUTWARD of the drawn
    macroring and gaps pointing into the cavity or into a fused small
    ring are damped x0.2 -- zigzag contours have concave vertices whose
    wide gap faces the interior, so outwardness is checked geometrically
    (CoordGen's rule implicitly assumed convex template contours);
  * search    -- a JOINT beam search over all side chains' DOF vectors
    (beam width 7 = 6 x precision + 1, single-DOF mutations, global
    seen-set, solution-vector score memo, strict-improvement admission
    against the round-start best, tier += 3 escalation flip -> turn ->
    shrink on a zero-improvement round, <= 100 rounds / <= 10k scored
    solutions, final pose = argmin over the whole scored history --
    the CoordgenMinimizer growSolutions semantics verbatim);
  * scoring   -- a pluggable ScoreTerm list: CoordGen's linear atom-bond
    clash potential (rest distances 0.8/0.7/0.6 x bond by carbon
    neutrality) plus an atom-atom crowding term, DOF penalties, bond
    crossings (2500 x mult with macroring x8 / small ring x2 / terminal
    x0.5), small-ring press (50 + 100 x (1 - d/L), rings <= 8 atoms),
    the macrocycle-interior penalty CDK lacks (10 x heavy atoms of a
    side chain inside the drawn ring polygon) and a same-side angular
    spread term; the beam minimises the FULL score while the "clean"
    gate counts physical collisions only;
  * refine    -- a CDK LayoutRefiner port as the residual-congestion
    layer after the beam (<= 10 rounds): sigma-bond subtree reflection
    with a symmetric-bond blacklist and rollback, macroring side-chain
    inversion about the tangent through the ring atom, and a
    bend/stretch last resort; scored by an incrementally maintained
    Congestion (sum 1/d^2 over non-bonded, non-shared-ring pairs,
    collision = d < 0.75 bond or d < 2 bond with crossing adjacent
    bonds; acceptance thresholds rescaled from CDK's Angstrom squares
    to bond-length units);
  * fallback  -- CoordGen's two-step force field, only when the beam and
    the refiner leave physical collisions: terminal bonds in collision
    shrink to 10 % (a 90 % shrink), then <= 1000 rounds of full-field
    relaxation over the side-chain atoms with a lowest-energy-frame
    rollback;
  * QA        -- non-bonded collapse pairs (d < 0.30 bond) must be 0,
    ring bonds exact, ring angles 120 +/- 3, zero bond crossings, and
    the molblock wedge round-trip must reproduce the input isomeric
    SMILES; the ALIGNED black-and-white render path is shared with
    zigzag_layout_v2.

Entry points: `beam_layout_coords` / `beam_layout_with_retry` (layout),
`render_beam` (ALIGNED PNG), `validation_suite` (seq / KRAS / GPC3
renders + real-crop comparison sheets).
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

import zigzag_layout as zz1
import zigzag_layout_v2 as zzv2
from zigzag_layout_v2 import (
    BOND,
    _adjacency,
    _angle_between,
    _carbonyl_oxygen,
    _norm,
    _rot,
    _seg_cross,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SidechainConfig:
    """All tunables of the beam layout.  Defaults follow the research
    synthesis: CoordGen's beam/penalty magnitudes (7 : 10^4 : 10^2) and
    the space-filling report's acceptance geometry."""

    bond: float = BOND                     # layout bond length (px)
    # beam search
    beam_width: int = 7                    # 6 x precision + 1
    clean_energy: float = 10.0             # clashEnergyThreshold
    max_rounds: int = 100
    max_scored_solutions: int = 10_000
    tier_advance: int = 3                  # zero-improvement escalation
    max_tier: int = 6
    # DOF penalties (CoordGen sketcherMinimizerFragment.cpp:12-27)
    invert_penalty: float = 100.0          # side chain flipped inward
    shrink_factor: float = 0.4             # ScaleAtomsDOF contraction
    shrink_penalty_per_atom: float = 50.0
    # crossing penalty (CoordgenMinimizer.cpp:32-35)
    crossing_penalty: float = 2500.0
    macrocycle_cross_mult: float = 8.0
    small_ring_cross_mult: float = 2.0
    terminal_cross_mult: float = 0.5
    # clash potential (CoordgenMinimizer.cpp:222-232, k2 = 0.1)
    clash_rest_vk: float = 0.8             # x bond; -0.1 per neutral-C rule
    clash_linear_k2: float = 0.1
    atom_clash_frac: float = 0.65          # atom-atom crowding threshold
    # ring press (scoreAtomsInsideRings, rings < MACROCYCLE=9)
    small_ring_max_atoms: int = 8
    ring_press_base: float = 50.0
    ring_press_scale: float = 100.0
    # macrocycle interior penalty (the CDK defect we patch)
    in_ring_penalty_per_heavy: float = 10.0
    # central-void hard gate: moving atoms closer than cavity_frac x mean
    # ring-vertex radius are in the cavity real drawings keep empty
    cavity_frac: float = 0.45
    cavity_weight: float = 25.0
    in_ring_weight: float = 0.25
    # same-side angular spread of side chains around the ring centre
    sameside_min_angle_deg: float = 20.0
    sameside_penalty_per_pair: float = 1.0
    # seeding: gap heuristic + reflex trap
    gap_out_weight: float = 10.0           # gap > 180 deg opening outward
    gap_in_weight: float = 0.2             # inward / fused-ring pointing
    gap_probe_frac: float = 0.1            # CoordgenFragmentBuilder.cpp:705
    trap_hit_tol: float = 0.05             # x bond; reflex-vertex exact hit
    min_attach_angle: float = 55.0         # first bond vs ring bonds
    # CDK refiner (LayoutRefiner constants rescaled to bond units; CDK's
    # are in Angstrom^2 with bond ~1.5 A -> divide by 1.5^2)
    refine_rounds: int = 10
    refine_min_score: float = 1.0 / (0.75 ** 2)     # 1.78 severe overlap
    refine_maybe_crossed: float = 0.25              # d < 2 bond
    refine_rotate_delta: float = 5.0 / 2.25         # CDK delta > 5 A^-2
    refine_rotate_minor: float = 1.0 / 2.25
    refine_symmetric_delta: float = 0.1 / 2.25      # probable-symmetric set
    refine_invert_improvement: float = 0.02         # relative, >= 2 %
    refine_bend_min_gain: float = 0.5
    bend_step_deg: float = 10.0
    bend_max_attempts: int = 3
    stretch_step_frac: float = 0.2                  # x bond per attempt
    stretch_max_frac: float = 2.0                   # bond cap
    # force-field fallback (CoordgenMinimizer.cpp:72-151, 1397-1432)
    ff_rounds: int = 1000
    ff_shrink_keep: float = 0.1                     # 90 % terminal shrink
    ff_step: float = 0.1
    ff_step_cap: float = 3.0
    ff_stagnation_window: int = 100
    ff_stagnation_gain: float = 1.0

    # --- QA geometry (inherited from the v2 bars) ---
    qa_floor_frac: float = 0.30            # hard collapse threshold
    soft_floor_frac: float = 0.55          # reported crowding threshold
    angle_tol: float = 3.0


DEFAULT_CONFIG = SidechainConfig()


# ---------------------------------------------------------------------------
# vectorised geometry primitives
# ---------------------------------------------------------------------------

def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _seg_cross_arrays(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                      p4: np.ndarray) -> np.ndarray:
    """Segment (p1,p2) x (p3,p4) per row; exclusive-overlap test like
    CoordGen's intersectionOfSegments (collinear overlaps not reported --
    a documented upstream simplification we inherit)."""
    d1 = _cross2(p2 - p1, p3 - p1)
    d2 = _cross2(p2 - p1, p4 - p1)
    d3 = _cross2(p4 - p3, p1 - p3)
    d4 = _cross2(p4 - p3, p2 - p3)
    return ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))


def _point_seg_dist2(points: np.ndarray, a: np.ndarray,
                     b: np.ndarray) -> np.ndarray:
    """Squared point-to-segment distance per row (t clamped to [0, 1])."""
    ab = b - a
    denom = np.maximum(ab[:, 0] ** 2 + ab[:, 1] ** 2, 1e-12)
    ap = points - a
    t = np.clip((ap[:, 0] * ab[:, 0] + ap[:, 1] * ab[:, 1]) / denom, 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = points - proj
    return d[:, 0] ** 2 + d[:, 1] ** 2


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


def _filter_trapped_directions(candidates: Sequence[tuple[float, tuple[float, float]]],
                               root_pos: tuple[float, float],
                               placed_pts: Sequence[tuple[float, float]],
                               bond: float,
                               tol_frac: float
                               ) -> list[tuple[float, tuple[float, float]]]:
    """Reflex-vertex trap filter (space_filling_layout.md 4.3): at a
    concave ring vertex one of the +/-120 deg slot directions lands
    EXACTLY on the neighbouring ring atom (min distance 0.0).  Any
    candidate whose first-atom position falls within `tol_frac` bond
    lengths of a placed atom is discarded."""
    kept = []
    for weight, d in candidates:
        p = (root_pos[0] + bond * d[0], root_pos[1] + bond * d[1])
        hit = False
        for q in placed_pts:
            if abs(q[0] - p[0]) ** 2 + abs(q[1] - p[1]) ** 2 \
                    < (tol_frac * bond) ** 2:
                hit = True
                break
        if not hit:
            kept.append((weight, d))
    return kept


# ---------------------------------------------------------------------------
# rigid side-chain templates
# ---------------------------------------------------------------------------

class SidechainTemplate:
    """A side chain built ONCE per conformation in a local frame.

    Local frame: root at the origin, first bond along +X.  The builder is
    the SMILES translation of the draftsman's zig-zag alphabet (chain
    steps turn +/-60 deg giving exact 120 deg interior angles, carbonyl
    oxygens take the free trigonal slot, pendant rings complete as
    regular polygons anchored on already placed atoms -- the v1/v2
    lattice rules).  Chain-step vertices whose turn sign is a search DOF
    are discovered on the first build; later builds replay the recorded
    defaults with per-vertex overrides.  Every conformation a build can
    emit has exact bond lengths and exact 120 deg chain angles: the
    subtree is RIGID in the metric sense and only discrete choices vary.
    """

    def __init__(self, mol: Chem.Mol, root: int, entry: int, bond: float,
                 forbidden: set[int]):
        self.mol = mol
        self.root = root
        self.entry = entry
        self.bond = bond
        self.forbidden = set(forbidden)
        self.bonded = _adjacency(mol)
        self.parent: dict[int, int] = {}
        self.turn_atoms: list[int] = []        # DOF carriers, build order
        self.default_signs: dict[int, int] = {}
        local, _turns, _signs = self._build({}, discovering=True)
        self.turn_atoms = _turns
        self.default_signs = _signs
        self.atoms = sorted(i for i in local if i != self.root)
        self.atom_set = set(self.atoms)
        self.size = len(self.atoms)

    # -- local geometry rules -------------------------------------------------

    def _incoming(self, local: dict[int, tuple[float, float]],
                  atom: int) -> tuple[float, float]:
        p = self.parent.get(atom)
        if p is None or p not in local:
            return (1.0, 0.0)
        return _norm((local[atom][0] - local[p][0],
                      local[atom][1] - local[p][1]))

    def _third_slot(self, local: dict[int, tuple[float, float]],
                    atom: int) -> tuple[float, float]:
        """Direction of the free trigonal slot at `atom` (>= 2 placed
        bonds): bisector of the largest angular gap between the placed
        bond directions (the exterior-bisector rule; gap-based so 3+
        branches stay well-defined)."""
        dirs = [_norm((local[nb][0] - local[atom][0],
                       local[nb][1] - local[atom][1]))
                for nb in self.bonded[atom] if nb in local]
        if len(dirs) < 2:
            return (1.0, 0.0)
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

    def _unplaced_ring(self, kid: int, local: dict,
                       ring_done: set[tuple[int, ...]]) -> tuple[int, ...] | None:
        """Smallest ring through `kid` still to draw, touching no atom
        owned by another body (fused systems complete through shared
        anchors as pentagon+hexagon, never one stretched perimeter)."""
        cands = [r for r in self.mol.GetRingInfo().AtomRings()
                 if kid in r and tuple(sorted(r)) not in ring_done
                 and not any(a in self.forbidden for a in r)
                 and not all(a in local for a in r)]
        return min(cands, key=len) if cands else None

    def _local_clearance(self, local: dict, pos: dict[int, tuple[float, float]],
                         exclude: set[int]) -> float:
        """Min NON-bonded distance among candidate positions and against
        the atoms already drawn in this body's local frame."""
        best = float("inf")
        items = [(j, p) for j, p in local.items()]
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

    # -- chain steps ----------------------------------------------------------

    def _place_chain_step(self, kid: int, atom: int,
                          local: dict[int, tuple[float, float]],
                          queue: deque, lean: float,
                          overrides: Mapping[int, int], discovering: bool,
                          turn_atoms: list[int], signs: dict[int, int],
                          ) -> float:
        """One zig-zag step off `atom`; returns the next lean.

        `lean` carries the default sign for the step: a signed lean is a
        DOF-controlled turn (the sign is applied EXACTLY so rebuilds are
        deterministic), lean 0.0 (after branches / ring entries) selects
        between the +/-60 candidates by local clearance -- the draftsman's
        small kink, not a search DOF."""
        n_placed = [nb for nb in self.bonded[atom] if nb in local]
        if len(n_placed) >= 2:
            d = self._third_slot(local, atom)      # branch slot, no DOF
            new_lean = 0.0
        else:
            hint = self._incoming(local, atom) if n_placed else (1.0, 0.0)
            if lean == 0.0:
                # clearance-scored kink (non-DOF): ±60 candidates scored
                # against the atoms this body already drew
                best = None
                for s in (1.0, -1.0):
                    cand = _rot(hint, s * 60.0)
                    trial = {kid: (local[atom][0] + self.bond * cand[0],
                                   local[atom][1] + self.bond * cand[1])}
                    cl = self._local_clearance(local, trial, exclude={atom})
                    key = (min(cl, 2.0 * self.bond), 1.0 if s == 1.0 else 0.0)
                    if best is None or key > best[0]:
                        best = (key, cand)
                d = best[1]
                new_lean = 0.0
            else:
                if discovering:
                    self.default_signs[kid] = int(math.copysign(1.0, lean))
                    turn_atoms.append(kid)
                sign = int(overrides.get(kid, self.default_signs[kid]))
                signs[kid] = sign
                d = _rot(hint, sign * 60.0)
                new_lean = -float(sign)
        local[kid] = (local[atom][0] + self.bond * d[0],
                      local[atom][1] + self.bond * d[1])
        self.parent[kid] = atom
        queue.append((kid, new_lean))
        return new_lean

    # -- small rings (regular-polygon completion) ------------------------------

    def _place_ring(self, ring: tuple[int, ...], kid: int, atom: int,
                    local: dict, queue: deque, ring_done: set,
                    overrides: Mapping[int, int], discovering: bool,
                    turn_atoms: list[int], signs: dict[int, int]) -> None:
        n = len(ring)
        pre = [i for i in ring if i in local]
        if kid not in local and len(pre) < 2:
            self._place_chain_step(kid, atom, local, queue, 0.0,
                                   overrides, discovering,
                                   turn_atoms, signs)
        anchors = [i for i in ring if i in local]
        start = ring.index(anchors[0])
        order = list(ring[start:]) + list(ring[:start])
        if len(anchors) == 1:
            self._place_full_ring(order[0], order[1:] + order[:1], n, local,
                                  queue)
        else:
            for k in range(len(order)):
                i = order[k]
                if i in local:
                    seq = [i]
                    j = (k + 1) % len(order)
                    while order[j] not in local:
                        seq.append(order[j])
                        j = (j + 1) % len(order)
                    seq.append(order[j])
                    if len(seq) > 2:
                        self._place_arc(seq, n, local, queue)
        ring_done.add(tuple(sorted(ring)))
        for i in order:
            if i in local:
                queue.append((i, 0.0))

    def _place_full_ring(self, entry: int, seq: list[int], n: int,
                         local: dict, queue: deque) -> None:
        """Regular n-gon walked from the entry bond, opening on the side
        with more local clearance."""
        entry_dir = self._incoming(local, entry)
        best = None
        for sign in (1.0, -1.0):
            heading = math.degrees(math.atan2(entry_dir[1], entry_dir[0]))
            heading += sign * 60.0
            cur = local[entry]
            trial = []
            for _i in seq:
                cur = (cur[0] + self.bond * math.cos(math.radians(heading)),
                       cur[1] + self.bond * math.sin(math.radians(heading)))
                trial.append(cur)
                heading += 360.0 / n
            cand = {i: c for i, c in zip(seq, trial)}
            cl = self._local_clearance(local, cand, exclude={entry})
            if best is None or cl > best[0]:
                best = (cl, cand)
        for i, c in best[1].items():
            local[i] = c
            self.parent[i] = entry
            queue.append((i, 0.0))

    def _place_arc(self, seq: list[int], n: int, local: dict,
                   queue: deque) -> None:
        """Lay seq[1:-1] (new vertices) between placed endpoints as the
        arc of a regular n-gon (major arc for fused rings)."""
        a, b = local[seq[0]], local[seq[-1]]
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
            cl = self._local_clearance(local, cand,
                                       exclude={seq[0], seq[-1]})
            if best is None or cl > best[0]:
                best = (cl, cand)
        for i, c in best[1].items():
            local[i] = c
            self.parent[i] = seq[0]
            queue.append((i, 0.0))

    # -- construction driver ----------------------------------------------------

    def _build(self, overrides: Mapping[int, int], discovering: bool
               ) -> tuple[dict[int, tuple[float, float]],
                          list[int], dict[int, int]]:
        """Worklist construction off the attachment bond root->entry; the
        carbonyl oxygen of each carbon is placed after its heavy bonds
        (the bisector rule needs both)."""
        local: dict[int, tuple[float, float]] = {self.root: (0.0, 0.0)}
        local[self.entry] = (self.bond, 0.0)
        self.parent[self.entry] = self.root
        turn_atoms: list[int] = []
        signs: dict[int, int] = {}
        queue: deque = deque([(self.entry, 1.0)])
        ring_done: set[tuple[int, ...]] = set()
        while queue:
            atom, lean = queue.popleft()
            kids = sorted(k for k in self.bonded[atom]
                          if k not in local and k not in self.forbidden)
            o = _carbonyl_oxygen(self.mol, atom)
            if o is not None and o in kids:
                kids.remove(o)
            chains: list[int] = []
            ring_kids: list[tuple[int, tuple[int, ...]]] = []
            for kid in kids:
                rng_ = self._unplaced_ring(kid, local, ring_done)
                if rng_ is not None:
                    ring_kids.append((kid, rng_))
                else:
                    chains.append(kid)
            for k, kid in enumerate(chains):
                step_lean = lean if len(chains) == 1 \
                    else (lean or 1.0) * (1.0 if k == 0 else -1.0)
                self._place_chain_step(kid, atom, local, queue, step_lean,
                                       overrides, discovering,
                                       turn_atoms, signs)
            for kid, rng_ in ring_kids:
                self._place_ring(rng_, kid, atom, local, queue, ring_done,
                                 overrides, discovering, turn_atoms, signs)
            if o is not None and o not in local and o not in self.forbidden:
                n_placed = [nb for nb in self.bonded[atom] if nb in local]
                d = self._third_slot(local, atom) if len(n_placed) >= 2 \
                    else _rot(self._incoming(local, atom), 180.0)
                local[o] = (local[atom][0] + self.bond * d[0],
                            local[atom][1] + self.bond * d[1])
                self.parent[o] = atom
        return local, turn_atoms, signs

    def build(self, overrides: Mapping[int, int] | None = None
              ) -> dict[int, tuple[float, float]]:
        """Rebuild the local subtree with per-vertex turn-sign overrides
        (state 1 negates the recorded default sign)."""
        ov: dict[int, int] = {}
        for atom, state in (overrides or {}).items():
            if state:
                ov[atom] = -self.default_signs[atom]
        local, _t, _s = self._build(ov, discovering=False)
        return local

    # -- rigid placement --------------------------------------------------------

    def pose(self, origin: tuple[float, float], angle_deg: float,
             flip: bool = False, shrink: bool = False,
             shrink_factor: float = 0.4,
             overrides: Mapping[int, int] | None = None
             ) -> dict[int, tuple[float, float]]:
        """Rigid placement: optional umbrella mirror (x -> -x in the local
        frame reverses the first bond to the opposite slot exactly like
        CoordGen's InvertBondDOF mirror), optional contraction toward the
        root (ScaleAtomsDOF x0.4), then rotate by `angle_deg` and
        translate so the root sits at `origin`."""
        local = self.build(overrides)
        fx = -1.0 if flip else 1.0
        f = shrink_factor if shrink else 1.0
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        out: dict[int, tuple[float, float]] = {}
        for i, (x, y) in local.items():
            if i == self.root:
                out[i] = origin
                continue
            lx, ly = fx * f * x, f * y
            out[i] = (origin[0] + lx * c - ly * s,
                      origin[1] + lx * s + ly * c)
        return out


# ---------------------------------------------------------------------------
# degrees of freedom
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DegreeOfFreedom:
    """CoordGen's DOF contract: discrete states / search tier / current
    penalty / (apply realised through SidechainTemplate.pose rebuilds)."""

    body: int
    kind: str                 # "flip" | "turn" | "shrink"
    tier: int
    target: int = -1          # chain vertex for turns
    n_states: int = 2

    def penalty(self, state: int, cfg: SidechainConfig,
                n_heavy: int = 0) -> float:
        if state == 0:
            return 0.0
        if self.kind == "flip":
            return cfg.invert_penalty
        if self.kind == "shrink":
            return cfg.shrink_penalty_per_atom * n_heavy
        return 0.0


FLIP_TIER = 0
TURN_TIER = 3
SHRINK_TIER = 6


# ---------------------------------------------------------------------------
# pluggable scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreTerm:
    """One pluggable scoring item: a named, weighted function of the
    assembled pose.  The default list reproduces CoordGen's scoreClashes
    composition plus the two penalties the research passes add."""
    name: str
    weight: float
    fn: Callable[["LayoutScene", "PoseView"], float]

    physical: bool = True    # counts toward the "clean" collision gate


@dataclass
class PoseView:
    """Assembled coordinates of one solution vector."""
    coords: np.ndarray                      # (n_atoms, 2) full molecule
    moving: np.ndarray                      # side-chain atom indices
    body_states: tuple[tuple[int, ...], ...]
    body_centroids: tuple[np.ndarray, ...]


def _term_clash(scene: "LayoutScene", pv: PoseView) -> float:
    """Linear atom-bond potential (CoordGen ClashInteraction) plus an
    atom-atom crowding term at 0.65 x bond."""
    cfg = scene.cfg
    c = pv.coords
    e = 0.0
    if len(scene.aa_i):
        d2 = (c[scene.aa_i] - c[scene.aa_j]) ** 2
        d2 = d2[:, 0] + d2[:, 1]
        over = d2 < scene.aa_rest2
        if over.any():
            e += 0.5 * cfg.clash_linear_k2 * \
                float((scene.aa_rest2[over] - d2[over]).sum())
    if len(scene.ab_atom):
        d2 = _point_seg_dist2(c[scene.ab_atom], c[scene.ab_b0],
                              c[scene.ab_b1])
        over = d2 < scene.ab_rest2
        if over.any():
            e += 0.5 * cfg.clash_linear_k2 * \
                float((scene.ab_rest2[over] - d2[over]).sum())
    return e


def _term_dof(scene: "LayoutScene", pv: PoseView) -> float:
    """Sum of the DOF penalties of the current solution.  The umbrella
    flip penalty is GEOMETRIC: it applies to whatever pose points its
    centroid into the macroring cavity, not to a state index -- on the
    concave (valley) vertices of a sawtooth contour the natural 120 deg
    slot faces the interior, so the outward pose is the flipped state
    there and must stay unpenalised."""
    total = 0.0
    for bi, (lo, hi) in enumerate(scene.body_slices):
        n_heavy = scene.bodies[bi].size
        state = pv.body_states[bi]
        for dof, st in zip(scene.dofs[lo:hi], state):
            if dof.kind == "flip":
                if scene.body_inward(bi, state):
                    total += scene.cfg.invert_penalty
            else:
                total += dof.penalty(st, scene.cfg, n_heavy)
    return total


def _term_crossing(scene: "LayoutScene", pv: PoseView) -> float:
    """Bond crossings: 2500 x mult(b1) x mult(b2); macroring x8,
    small ring x2, terminal x0.5 (CoordgenMinimizer scoreCrossBonds)."""
    if not len(scene.cross_a):
        return 0.0
    c = pv.coords
    hit = _seg_cross_arrays(c[scene.cross_a0], c[scene.cross_a1],
                            c[scene.cross_b0], c[scene.cross_b1])
    if not hit.any():
        return 0.0
    m = scene.cross_mult[hit]
    return float(scene.cfg.crossing_penalty * m.sum())


def _term_ring_press(scene: "LayoutScene", pv: PoseView) -> float:
    """Atoms pressed into small fixed rings (< MACROCYCLE): 50 + 100 x
    (1 - d/L) within one bond length of the ring centroid."""
    cfg = scene.cfg
    c = pv.coords
    e = 0.0
    for centroid, _poly, cand_idx in scene.small_ring_probes:
        d = np.hypot(c[cand_idx, 0] - centroid[0],
                     c[cand_idx, 1] - centroid[1])
        hit = d < cfg.bond
        if hit.any():
            dd = d[hit]
            e += float((cfg.ring_press_base
                        + cfg.ring_press_scale * (1.0 - dd / cfg.bond)).sum())
    return e


def _term_in_ring(scene: "LayoutScene", pv: PoseView) -> float:
    """Side-chain heavy atoms inside the drawn macroring polygon (soft:
    valley-attached substituents legitimately occupy the concave notch,
    so this cannot be a feasibility gate)."""
    if scene.macro_poly is None:
        return 0.0
    inside = _points_in_polygon(pv.coords[pv.moving], scene.macro_poly)
    return float(scene.cfg.in_ring_penalty_per_heavy * inside.sum())


def _term_cavity(scene: "LayoutScene", pv: PoseView) -> float:
    """Moving atoms within the central void radius: 1 each, and the hard
    admission gate -- the cavity real patent drawings keep empty."""
    if scene.cavity_r <= 0.0:
        return 0.0
    d = pv.coords[pv.moving] - scene.ring_center
    return float((np.hypot(d[:, 0], d[:, 1]) < scene.cavity_r).sum())


def _term_same_side(scene: "LayoutScene", pv: PoseView) -> float:
    """Angular spread of the side-chain bodies around the ring centre:
    a flat penalty per pair of bodies clustering within
    sameside_min_angle_deg (the drawing convention of spreading
    substituents over the outline)."""
    angs = []
    for cent in pv.body_centroids:
        v = (cent[0] - scene.ring_center[0], cent[1] - scene.ring_center[1])
        angs.append(math.degrees(math.atan2(v[1], v[0])) % 360.0)
    pen = 0.0
    for a in range(len(angs)):
        for b in range(a + 1, len(angs)):
            sep = abs(angs[a] - angs[b])
            if sep > 180.0:
                sep = 360.0 - sep
            if sep < scene.cfg.sameside_min_angle_deg:
                pen += scene.cfg.sameside_penalty_per_pair
    return pen


DEFAULT_SCORE_TERMS: tuple[ScoreTerm, ...] = (
    ScoreTerm("clash", 1.0, _term_clash, physical=True),
    ScoreTerm("dof", 1.0, _term_dof, physical=False),
    ScoreTerm("crossing", 1.0, _term_crossing, physical=True),
    ScoreTerm("ring_press", 1.0, _term_ring_press, physical=True),
    ScoreTerm("cavity", 25.0, _term_cavity, physical=True),
    ScoreTerm("in_ring", 0.25, _term_in_ring, physical=False),
    ScoreTerm("sameside", 1.0, _term_same_side, physical=False),
)


# ---------------------------------------------------------------------------
# layout scene
# ---------------------------------------------------------------------------

class LayoutScene:
    """Fixed frame (backbone + pendant arcs) + side-chain bodies + the
    scorer.  All pair index arrays are STRUCTURAL (bond/participation
    patterns do not change between solutions), so they are precomputed
    once; per-solution work is a handful of vectorised gathers."""

    def __init__(self, mol: Chem.Mol, fixed_pos: dict[int, tuple[float, float]],
                 bodies: list[SidechainTemplate],
                 attach_angle: dict[int, float],
                 ring_atoms: list[int], ring_center: tuple[float, float],
                 cfg: SidechainConfig = DEFAULT_CONFIG,
                 terms: Sequence[ScoreTerm] | None = None):
        self.mol = mol
        self.cfg = cfg
        self.terms = tuple(terms) if terms is not None else \
            tuple(ScoreTerm(t.name,
                            cfg.cavity_weight if t.name == "cavity"
                            else cfg.in_ring_weight if t.name == "in_ring"
                            else t.weight,
                            t.fn, t.physical)
                  for t in DEFAULT_SCORE_TERMS)
        self.fixed_pos = dict(fixed_pos)
        self.bonded = _adjacency(mol)
        self.bodies = bodies
        self.attach_angle = dict(attach_angle)
        self.ring_atoms = list(ring_atoms)
        self.ring_center = tuple(ring_center)
        self.ring_set = set(ring_atoms)
        self.n = mol.GetNumAtoms()

        self._init_bodies()
        self._init_structure()
        self._pose_cache: dict[tuple[int, tuple[int, ...]],
                               tuple[np.ndarray, np.ndarray, int, bool]] = {}
        self._score_cache: dict[tuple[int, ...], tuple[float, dict[str, float]]] = {}

    # -- setup ---------------------------------------------------------------

    def _init_bodies(self) -> None:
        self.body_slices: list[tuple[int, int]] = []
        self.dofs: list[DegreeOfFreedom] = []
        for bi, body in enumerate(self.bodies):
            lo = len(self.dofs)
            self.dofs.append(DegreeOfFreedom(bi, "flip", FLIP_TIER))
            for v in body.turn_atoms:
                self.dofs.append(DegreeOfFreedom(bi, "turn", TURN_TIER,
                                                 target=v))
            self.dofs.append(DegreeOfFreedom(bi, "shrink", SHRINK_TIER))
            self.body_slices.append((lo, len(self.dofs)))
        self.initial_solution = tuple([0] * len(self.dofs))

    def _init_structure(self) -> None:
        cfg = self.cfg
        mol = self.mol
        moving_set: set[int] = set()
        for body in self.bodies:
            moving_set |= body.atom_set
        self.moving_set = moving_set
        self.moving = np.array(sorted(moving_set), dtype=int)
        fixed_set = set(self.fixed_pos)

        self.bonds: list[tuple[int, int]] = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        self.bond_orders = [
            float(b.GetBondTypeAsDouble()) for b in mol.GetBonds()]
        # crossing multipliers: macroring x8, small ring x2, terminal x0.5
        small_ring_bonds: set[tuple[int, int]] = set()
        self.small_ring_atom_sets: list[set[int]] = []
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) <= cfg.small_ring_max_atoms:
                self.small_ring_atom_sets.append(set(ring))
                for k in range(len(ring)):
                    small_ring_bonds.add(
                        tuple(sorted((ring[k], ring[(k + 1) % len(ring)]))))
        self.macro_bond_set: set[tuple[int, int]] = set()
        for k in range(len(self.ring_atoms)):
            self.macro_bond_set.add(tuple(sorted(
                (self.ring_atoms[k],
                 self.ring_atoms[(k + 1) % len(self.ring_atoms)]))))

        def mult(i: int, j: int) -> float:
            m = 1.0
            key = tuple(sorted((i, j)))
            if key in self.macro_bond_set:
                m *= cfg.macrocycle_cross_mult
            elif key in small_ring_bonds:
                m *= cfg.small_ring_cross_mult
            if len(self.bonded[i]) == 1 or len(self.bonded[j]) == 1:
                m *= cfg.terminal_cross_mult
            return m

        self.bond_mult = np.array([mult(i, j) for i, j in self.bonds])
        moving_bonds = [bi for bi, (i, j) in enumerate(self.bonds)
                        if i in moving_set or j in moving_set]
        mb_set = set(moving_bonds)
        cross_pairs: list[tuple[int, int]] = []
        for a in moving_bonds:
            ai, aj = self.bonds[a]
            for b in range(len(self.bonds)):
                if b == a or (b in mb_set and b < a):
                    continue
                bi, bj = self.bonds[b]
                if len({ai, aj, bi, bj}) < 4:
                    continue
                cross_pairs.append((a, b))
        self.cross_a = np.array([p[0] for p in cross_pairs], dtype=int)
        self.cross_b = np.array([p[1] for p in cross_pairs], dtype=int)
        self.cross_a0 = np.array([self.bonds[p[0]][0] for p in cross_pairs],
                                 dtype=int)
        self.cross_a1 = np.array([self.bonds[p[0]][1] for p in cross_pairs],
                                 dtype=int)
        self.cross_b0 = np.array([self.bonds[p[1]][0] for p in cross_pairs],
                                 dtype=int)
        self.cross_b1 = np.array([self.bonds[p[1]][1] for p in cross_pairs],
                                 dtype=int)
        self.cross_mult = (self.bond_mult[self.cross_a]
                           * self.bond_mult[self.cross_b]) \
            if len(cross_pairs) else np.zeros(0)

        # atom-atom crowding pairs (>= 1 moving endpoint, non-bonded)
        aa: list[tuple[int, int]] = []
        for i in sorted(moving_set):
            for j in range(self.n):
                if j == i or j in self.bonded[i]:
                    continue
                if j in moving_set and j < i:
                    continue
                aa.append((i, j))
        self.aa_i = np.array([p[0] for p in aa], dtype=int)
        self.aa_j = np.array([p[1] for p in aa], dtype=int)
        self.aa_rest2 = np.full(len(aa),
                                (cfg.atom_clash_frac * cfg.bond) ** 2)

        # atom-bond clash pairs, CoordGen's interaction set: any (atom,
        # bond) with the atom not an endpoint and not 1-3 to an endpoint,
        # where at least one of the two is side-chain owned
        charge = {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms()}
        anum = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
        neutral_c = lambda idx: anum[idx] == 6 and charge[idx] == 0
        ab: list[tuple[int, int]] = []
        seen_ab: set[tuple[int, int]] = set()

        def add_ab(atom: int, bnd: int) -> None:
            key = (atom, bnd)
            if key not in seen_ab:
                seen_ab.add(key)
                ab.append(key)

        for atom in sorted(moving_set):
            for bnd, (i, j) in enumerate(self.bonds):
                if atom in (i, j) or atom in self.bonded[i] \
                        or atom in self.bonded[j]:
                    continue
                add_ab(atom, bnd)
        for bnd in moving_bonds:
            i, j = self.bonds[bnd]
            for atom in sorted(fixed_set):
                if atom in (i, j) or atom in self.bonded[i] \
                        or atom in self.bonded[j]:
                    continue
                add_ab(atom, bnd)
        rest2: list[float] = []
        for atom, bnd in ab:
            vk = cfg.clash_rest_vk
            if neutral_c(atom):
                vk -= 0.1
            i, j = self.bonds[bnd]
            if neutral_c(i) and neutral_c(j):
                vk -= 0.1
            rest2.append((vk * cfg.bond) ** 2)
        self.ab_atom = np.array([p[0] for p in ab], dtype=int)
        self.ab_bond = np.array([p[1] for p in ab], dtype=int)
        self.ab_b0 = np.array([self.bonds[p[1]][0] for p in ab], dtype=int)
        self.ab_b1 = np.array([self.bonds[p[1]][1] for p in ab], dtype=int)
        self.ab_rest2 = np.array(rest2)

        # small fixed rings for the press term + probe candidate arrays
        self.small_ring_probes: list[tuple[tuple[float, float], np.ndarray,
                                           np.ndarray]] = []
        for ring_set in self.small_ring_atom_sets:
            if not ring_set <= fixed_set:
                continue          # body-internal rings move with their body
            pts = np.array([self.fixed_pos[a] for a in sorted(ring_set)])
            centroid = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            cand = np.array([i for i in self.moving], dtype=int)
            self.small_ring_probes.append((centroid, pts, cand))

        # macroring polygon (drawn cycle, fixed frame)
        poly = np.array([self.fixed_pos[a] for a in self.ring_atoms])
        self.macro_poly = poly if len(poly) >= 3 else None
        if self.macro_poly is not None:
            vertex_r = float(np.hypot(poly[:, 0] - self.ring_center[0],
                                      poly[:, 1] - self.ring_center[1])
                             .mean())
            self.cavity_r = self.cfg.cavity_frac * vertex_r
        else:
            self.cavity_r = 0.0

        base = np.zeros((self.n, 2))
        for i, p in self.fixed_pos.items():
            base[i] = p
        self._base_coords = base
        self.body_atom_arrays = [np.array(b.atoms, dtype=int)
                                 for b in self.bodies]

    # -- pose assembly ---------------------------------------------------------

    def _body_pose(self, bi: int, state: tuple[int, ...]
                   ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        key = (bi, state)
        hit = self._pose_cache.get(key)
        if hit is not None:
            return hit
        body = self.bodies[bi]
        flip = bool(state[0])
        shrink = bool(state[-1])
        turns = state[1:-1]
        overrides = {v: s for v, s in zip(body.turn_atoms, turns)}
        origin = self.fixed_pos[body.root]
        theta = self.attach_angle[bi]
        placed = body.pose(origin, theta, flip=flip, shrink=shrink,
                           shrink_factor=self.cfg.shrink_factor,
                           overrides=overrides)
        coords = np.zeros((len(body.atoms), 2))
        for k, i in enumerate(body.atoms):
            coords[k] = placed[i]
        centroid = coords.mean(axis=0)
        inside = 0
        centroid_in = False
        if self.macro_poly is not None:
            inside = int(_points_in_polygon(coords, self.macro_poly).sum())
            centroid_in = bool(_points_in_polygon(centroid[None, :],
                                                  self.macro_poly)[0])
        out = (coords, centroid, inside, centroid_in)
        self._pose_cache[key] = out
        return out

    def body_inward(self, bi: int, state: tuple[int, ...]) -> bool:
        """True when the body's centroid sits inside the drawn macroring
        polygon -- the geometric meaning of the umbrella flip penalty."""
        return self._body_pose(bi, state)[3]

    def pose_view(self, solution: tuple[int, ...]) -> PoseView:
        coords = self._base_coords.copy()
        states: list[tuple[int, ...]] = []
        centroids: list[np.ndarray] = []
        for bi, body in enumerate(self.bodies):
            lo, hi = self.body_slices[bi]
            state = tuple(solution[lo:hi])
            bc, cent, _inside, _cin = self._body_pose(bi, state)
            coords[self.body_atom_arrays[bi]] = bc
            states.append(state)
            centroids.append(cent)
        return PoseView(coords, self.moving, tuple(states),
                        tuple(centroids))

    # -- scoring ---------------------------------------------------------------

    def score_solution(self, solution: tuple[int, ...]
                       ) -> tuple[float, dict[str, float]]:
        hit = self._score_cache.get(solution)
        if hit is not None:
            return hit
        pv = self.pose_view(solution)
        breakdown: dict[str, float] = {}
        total = 0.0
        for term in self.terms:
            raw = float(term.fn(self, pv))
            breakdown[term.name] = raw
            total += term.weight * raw
        out = (total, breakdown)
        self._score_cache[solution] = out
        return out

    def clean_score(self, breakdown: dict[str, float]) -> float:
        """Physical-collision energy only (aesthetics excluded): the
        force-field fallback triggers on this, not on the full score."""
        return sum(breakdown.get(t.name, 0.0) for t in self.terms
                   if t.physical)

    # -- seeding ---------------------------------------------------------------

    def first_bond_candidates(self, body: SidechainTemplate
                              ) -> list[tuple[float, float, tuple[float, float]]]:
        """Scored first-bond slot directions at the attachment atom:
        the +/-120 deg lattice directions off the placed ring bonds,
        reflex-trap filtered, weighted by the CoordGen gap heuristic
        (outward wide gap x10, cavity/fused-ring pointing x0.2).  Returns
        (weight, angle_deg, unit_dir) sorted most-preferred first."""
        cfg = self.cfg
        root = body.root
        root_pos = self.fixed_pos[root]
        placed_dirs = [_norm((self.fixed_pos[nb][0] - root_pos[0],
                              self.fixed_pos[nb][1] - root_pos[1]))
                       for nb in self.bonded[root] if nb in self.fixed_pos]
        if not placed_dirs:
            away = _norm((root_pos[0] - self.ring_center[0],
                          root_pos[1] - self.ring_center[1]))
            return [(1.0, math.degrees(math.atan2(away[1], away[0])), away)]
        ring_dirs = placed_dirs[:2] if len(placed_dirs) >= 2 else placed_dirs
        # fixed atoms two bonds away (the pre-placed carbonyl O beyond
        # the backbone C(=O)) occupy real exterior sector space — slots
        # keep the same clearance from them as from the ring bonds
        near_fixed = list(ring_dirs)
        for nb1 in self.bonded[root]:
            if nb1 not in self.fixed_pos:
                continue
            for nb2 in self.bonded[nb1]:
                if nb2 == root or nb2 not in self.fixed_pos:
                    continue
                if self.mol.GetAtomWithIdx(nb2).GetSymbol() != "O":
                    continue
                near_fixed.append(_norm((
                    self.fixed_pos[nb2][0] - root_pos[0],
                    self.fixed_pos[nb2][1] - root_pos[1])))
        raw: list[tuple[float, float, tuple[float, float]]] = []
        seen: set[tuple[float, float]] = set()
        for u in ring_dirs:
            for sgn in (1.0, -1.0):
                d = _rot(u, sgn * 120.0)
                key = (round(d[0], 6), round(d[1], 6))
                if key in seen:
                    continue
                seen.add(key)
                ok = True
                for rd in near_fixed:
                    if _angle_between(d, rd) < cfg.min_attach_angle:
                        ok = False
                        break
                if not ok:
                    continue
                ang = math.degrees(math.atan2(d[1], d[0])) % 360.0
                raw.append((0.0, ang, d))
        placed_pts = [self.fixed_pos[i] for i in sorted(self.fixed_pos)]
        kept = _filter_trapped_directions([(0.0, d) for _w, _a, d in raw],
                                          root_pos, placed_pts,
                                          cfg.bond, cfg.trap_hit_tol)
        kept_dirs = [d for _w, d in kept]
        raw = [(w, a, d) for w, a, d in raw
               if any(d == kd or (abs(d[0] - kd[0]) < 1e-9
                                  and abs(d[1] - kd[1]) < 1e-9)
                      for kd in kept_dirs)]

        # gap structure at the root over ALL placed bond directions
        angs = sorted(math.degrees(math.atan2(v[1], v[0])) % 360.0
                      for v in placed_dirs)
        gaps: list[tuple[float, float]] = []   # (start, size)
        for j, a in enumerate(angs):
            b = angs[(j + 1) % len(angs)]
            gaps.append((a, (b - a) % 360.0))
        scored: list[tuple[float, float, tuple[float, float]]] = []
        for _w, ang, d in raw:
            weight = 0.0
            probe = (root_pos[0] + cfg.gap_probe_frac * cfg.bond * d[0],
                     root_pos[1] + cfg.gap_probe_frac * cfg.bond * d[1])
            for start, size in gaps:
                rel = (ang - start) % 360.0
                if rel <= size or size >= 359.999:
                    gap = size
                    weight = gap
                    if gap > 180.0:
                        outward = not _points_in_polygon(
                            np.array([probe]), self.macro_poly)[0] \
                            if self.macro_poly is not None else True
                        weight *= cfg.gap_out_weight if outward \
                            else cfg.gap_in_weight
                    elif self.macro_poly is not None and \
                            _points_in_polygon(np.array([probe]),
                                               self.macro_poly)[0]:
                        weight *= cfg.gap_in_weight
                    break
            for _centroid, poly, _cand in self.small_ring_probes:
                if _points_in_polygon(np.array([probe]), poly)[0]:
                    weight *= cfg.gap_in_weight
            scored.append((weight, ang, d))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored


# ---------------------------------------------------------------------------
# joint beam search (CoordgenMinimizer growSolutions semantics)
# ---------------------------------------------------------------------------

@dataclass
class BeamResult:
    solution: tuple[int, ...]
    score: float
    breakdown: dict[str, float]
    rounds: int
    scored: int
    tier_reached: int
    clean: bool


def run_beam_search(scene: LayoutScene,
                    cfg: SidechainConfig = DEFAULT_CONFIG) -> BeamResult:
    """Joint beam search over ALL side chains' DOF vectors.

    Semantics mirrored from CoordgenMinimizer.cpp:1105-1193: the seed is
    the all-default vector; each round expands up to beam_width+1 beam
    entries, mutating ONE DOF at a time to each of its other states;
    admission into the new beam requires STRICTLY improving the
    round-start best (prevents cycles); a zero-improvement round raises
    the tier by `tier_advance` (flip -> turn -> shrink) and restores the
    old beam; the final pose is the argmin over the WHOLE scored history
    and "clean" means the physical-collision energy sits under
    clean_energy with zero bond crossings."""
    dofs = scene.dofs
    seed = scene.initial_solution
    seed_score, seed_terms = scene.score_solution(seed)
    scored: dict[tuple[int, ...], float] = {seed: seed_score}
    seen: set[tuple[int, ...]] = {seed}
    beam: dict[tuple[int, ...], float] = {seed: seed_score}
    best_solution, best_score = seed, seed_score
    tier = 0
    rounds = 0
    while beam and rounds < cfg.max_rounds \
            and len(scored) < cfg.max_scored_solutions:
        rounds += 1
        old_beam = dict(beam)
        round_best = best_score
        ordered = sorted(beam.items(), key=lambda kv: (kv[1], kv[0]))
        candidates: dict[tuple[int, ...], float] = {}
        n = 0
        for sol, _sc in ordered:
            if n > cfg.beam_width:
                break                      # expands beam_width + 1 entries
            n += 1
            for di, dof in enumerate(dofs):
                if dof.tier > tier or dof.n_states < 2:
                    continue
                cur = sol[di]
                for k in range(1, dof.n_states):
                    st = (cur + k) % dof.n_states
                    new_sol = sol[:di] + (st,) + sol[di + 1:]
                    if new_sol in seen:
                        continue
                    seen.add(new_sol)
                    s, _terms = scene.score_solution(new_sol)
                    scored[new_sol] = s
                    if s < round_best:
                        candidates[new_sol] = s
                        if s < best_score:
                            best_score = s
                            best_solution = new_sol
        if candidates:
            beam = candidates
        elif tier + cfg.tier_advance <= cfg.max_tier:
            tier += cfg.tier_advance
            beam = old_beam
        else:
            beam = {}
    best_score, best_terms = scene.score_solution(best_solution)
    clean = (scene.clean_score(best_terms) < cfg.clean_energy)
    return BeamResult(best_solution, best_score, best_terms, rounds,
                      len(scored), tier, clean)


# ---------------------------------------------------------------------------
# CDK LayoutRefiner port: residual-congestion refinement layer
# ---------------------------------------------------------------------------

class Congestion:
    """CDK Congestion port: sum of 1/d^2 over non-bonded, non-shared-ring
    atom pairs in bond-normalised units (d < 1e-5 counts 1e5; bonded and
    same-ring-system pairs are -1 sentinels that never score).
    `update(moved)` maintains the score incrementally, recomputing only
    the pairs that involve a moved atom (Congestion.java semantics)."""

    def __init__(self, coords: np.ndarray, adj: list[set[int]],
                 ring_of: list[frozenset[int]], bond: float):
        self.coords = coords
        self.adj = adj
        self.ring_of = ring_of
        self.bond = bond
        n = len(adj)
        self.n = n
        self.c = np.zeros((n, n))
        for i in range(n):
            xi = coords[i]
            adj_i = adj[i]
            ring_i = ring_of[i]
            for j in range(i + 1, n):
                if j in adj_i:
                    continue
                if ring_i and (ring_i & ring_of[j]):
                    continue
                d = math.hypot(*(coords[j] - xi)) / bond
                v = 1e5 if d < 1e-5 else 1.0 / (d * d)
                self.c[i, j] = self.c[j, i] = v
        self.score = float(self.c.sum()) / 2.0

    def _pair(self, i: int, j: int) -> float:
        if j in self.adj[i]:
            return 0.0
        if self.ring_of[i] and (self.ring_of[i] & self.ring_of[j]):
            return 0.0
        d = math.hypot(*(self.coords[j] - self.coords[i])) / self.bond
        return 1e5 if d < 1e-5 else 1.0 / (d * d)

    def update(self, moved: Iterable[int]) -> None:
        m = sorted(set(moved))
        if not m:
            return
        mset = set(m)
        rest = [j for j in range(self.n) if j not in mset]
        marr = np.array(m, dtype=int)
        self.c[marr, :] = 0.0
        self.c[:, marr] = 0.0
        for i in m:
            for j in rest:
                v = self._pair(i, j)
                self.c[i, j] = self.c[j, i] = v
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                v = self._pair(m[a], m[b])
                self.c[m[a], m[b]] = self.c[m[b], m[a]] = v
        self.score = float(self.c.sum()) / 2.0


class CdkStyleRefiner:
    """Port of CDK LayoutRefiner.refine() as a post-beam layer.

    Three operator levels per round over the congested-pair queue
    (collision = d < 0.75 bond, or d < 2 bond with crossing adjacent
    bonds; same-ring-system pairs skipped; deterministic ordering):
      1. rotate  -- reflect a subtree about a sigma-bond line
        (single, non-ring, non-leaf bonds only; conservative acceptance
        with rollback; bonds whose reflection changes nothing beyond
        noise join the probable-symmetric blacklist);
      2. invert  -- macroring side-chain inversion about the tangent
        through the ring atom perpendicular to the side-chain bond
        (>= 2 % relative congestion improvement);
      3. bendOrStretch -- +/-10 deg x attempt subtree bends or
        +0.2 bond x attempt stretches along the congested path (last
        resort; recorded because they leave the 120 deg / exact-length
        bars).
    Only side-chain (body-owned) atoms are ever moved."""

    def __init__(self, mol: Chem.Mol, coords: np.ndarray,
                 movable: set[int], macro_set: set[int],
                 cfg: SidechainConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.mol = mol
        self.coords = coords
        self.n = len(coords)
        self.adj = _adjacency(mol)
        self.movable = set(movable)
        self.macro_set = set(macro_set)
        self.bond_obj = {tuple(sorted((b.GetBeginAtomIdx(),
                                       b.GetEndAtomIdx()))): b
                         for b in mol.GetBonds()}
        ring_of: list[frozenset[int]] = [frozenset() for _ in coords]
        for ring in mol.GetRingInfo().AtomRings():
            fs = frozenset(ring)
            for a in ring:
                ring_of[a] |= fs
        self.ring_of = ring_of
        self.congestion = Congestion(coords, self.adj, ring_of, cfg.bond)
        self.symmetric: set[tuple[int, int]] = set()
        self.tried: set[tuple[int, int, int, int]] = set()
        self.stats = {"rotations": 0, "inversions": 0, "bends": 0,
                      "stretches": 0, "rounds": 0, "pairs_initial": 0,
                      "pairs_final": 0}

    # -- helpers ---------------------------------------------------------------

    def _dist(self, i: int, j: int) -> float:
        return math.hypot(*(self.coords[j] - self.coords[i])) / self.cfg.bond

    def _bonds_cross(self, u: int, v: int) -> bool:
        """CDK haveCrossingBonds: any bond incident to u crossing any bond
        incident to v (distinct endpoints)."""
        pu, pv_ = self.coords[u], self.coords[v]
        for nu in self.adj[u]:
            for nv in self.adj[v]:
                if nu == v or nv == u or nu == nv:
                    continue
                if _seg_cross(pu, self.coords[nu], pv_, self.coords[nv]):
                    return True
        return False

    def _pair_congested(self, i: int, j: int) -> bool:
        d = self._dist(i, j)
        s = 1e5 if d < 1e-5 else 1.0 / (d * d)
        if s >= self.cfg.refine_min_score:
            return True
        return s >= self.cfg.refine_maybe_crossed and self._bonds_cross(i, j)

    def _subtree(self, start: int, blocked: int) -> list[int]:
        seen = {start}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for nb in self.adj[a]:
                if nb != blocked and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return sorted(seen)

    def _shortest_path(self, i: int, j: int) -> list[int]:
        prev = {i: None}
        queue = deque([i])
        while queue:
            a = queue.popleft()
            if a == j:
                break
            for nb in sorted(self.adj[a]):
                if nb not in prev:
                    prev[nb] = a
                    queue.append(nb)
        if j not in prev:
            return []
        path = []
        cur = j
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def find_congested_pairs(self
                             ) -> list[tuple[float, int, int, list[int]]]:
        out: list[tuple[float, int, int, list[int]]] = []
        for i in range(self.n):
            xi = self.coords[i]
            for j in range(i + 1, self.n):
                if j in self.adj[i]:
                    continue
                if self.ring_of[i] and (self.ring_of[i] & self.ring_of[j]):
                    continue
                if i not in self.movable and j not in self.movable:
                    continue    # two fixed atoms: nothing the refiner can move
                d = math.hypot(*(self.coords[j] - xi)) / self.cfg.bond
                s = 1e5 if d < 1e-5 else 1.0 / (d * d)
                if s >= self.cfg.refine_min_score or \
                        (s >= self.cfg.refine_maybe_crossed
                         and self._bonds_cross(i, j)):
                    path = self._shortest_path(i, j)
                    out.append((s, i, j, path))
        out.sort(key=lambda t: (-t[0], t[1], t[2]))
        return out

    def _center_first_bonds(self, path: list[int]) -> list[tuple[int, int]]:
        """Path bonds ordered center-first (CDK's bndAt ordering): bonds
        closest to the path midpoint are tried first."""
        n = len(path)
        if n < 2:
            return []
        mid = (n - 1) / 2.0
        bonds = [(path[k], path[k + 1]) for k in range(n - 1)]
        bonds.sort(key=lambda b: (min(abs(path.index(b[0]) - mid),
                                      abs(path.index(b[1]) - mid)),
                                  path.index(b[0])))
        return bonds

    def _reflect(self, mobile: Sequence[int], p1, p2) -> None:
        """Mirror the mobile atoms about the line p1-p2 (LayoutRefiner
        .reflect, closed form)."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        denom = max(dx * dx + dy * dy, 1e-12)
        a = (dx * dx - dy * dy) / denom
        b = 2.0 * dx * dy / denom
        for i in mobile:
            px = self.coords[i][0] - p1[0]
            py = self.coords[i][1] - p1[1]
            self.coords[i] = (a * px + b * py + p1[0],
                              b * px - a * py + p1[1])

    def _rotate_about(self, mobile: Sequence[int], pivot: int,
                      deg: float) -> None:
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        p = self.coords[pivot]
        for i in mobile:
            v = self.coords[i] - p
            self.coords[i] = (p[0] + v[0] * c - v[1] * s,
                              p[1] + v[0] * s + v[1] * c)

    def _translate(self, mobile: Sequence[int], dx: float, dy: float) -> None:
        for i in mobile:
            self.coords[i] = (self.coords[i][0] + dx,
                              self.coords[i][1] + dy)

    def _mobile_side(self, a: int, b: int) -> list[int] | None:
        """The subtree to move for bond (a, b): prefer the side that is
        entirely movable and smaller (move the small branch, keep the
        drawn backbone)."""
        best: list[int] | None = None
        for (start, blocked) in ((a, b), (b, a)):
            sub = self._subtree(start, blocked)
            if not all(x in self.movable for x in sub):
                continue
            if best is None or len(sub) < len(best):
                best = sub
        return best

    # -- operators ---------------------------------------------------------------

    def try_rotate(self, pairs) -> bool:
        """Level 1: sigma-bond subtree reflection with rollback and the
        probable-symmetric blacklist."""
        cfg = self.cfg
        self.tried.clear()
        for _s, i, j, path in pairs:
            for (a, b) in self._center_first_bonds(path):
                key = (i, j, a, b)
                if key in self.tried \
                        or tuple(sorted((a, b))) in self.symmetric:
                    continue
                self.tried.add(key)
                bond = self.bond_obj.get(tuple(sorted((a, b))))
                if bond is None:
                    continue
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue                  # sigma bonds only
                if self.ring_of[a] & self.ring_of[b]:
                    continue                  # ring bond
                if len(self.adj[a]) == 1 or len(self.adj[b]) == 1:
                    continue                  # leaf-end bond
                mobile = self._mobile_side(a, b)
                if not mobile:
                    continue
                backup = self.coords[mobile].copy()
                before = self.congestion.score
                self._reflect(mobile, self.coords[a], self.coords[b])
                self.congestion.update(mobile)
                delta = before - self.congestion.score
                cleared = not self._pair_congested(i, j)
                if delta > cfg.refine_rotate_delta or \
                        (delta > cfg.refine_rotate_minor and cleared):
                    self.stats["rotations"] += 1
                    return True
                if abs(delta) < cfg.refine_symmetric_delta:
                    self.symmetric.add(tuple(sorted((a, b))))
                self.coords[mobile] = backup
                self.congestion.update(mobile)
        return False

    def try_macro_invert(self, pairs) -> bool:
        """Level 2: macroring side-chain inversion about the tangent
        through the ring atom perpendicular to the side-chain bond
        (macroCycleInversion); relative improvement >= 2 %."""
        for _s, _i, _j, path in pairs:
            for v in path:
                if v not in self.macro_set:
                    continue
                ring_nbrs = [n for n in self.adj[v] if n in self.macro_set]
                if len(ring_nbrs) != 2:
                    continue                  # fusion point: skip
                before = self.congestion.score
                if before <= 0:
                    continue
                for w in sorted(self.adj[v]):
                    if w in self.macro_set:
                        continue
                    sub = self._subtree(w, v)
                    if not all(x in self.movable for x in sub):
                        continue
                    backup = self.coords[sub].copy()
                    b = self.coords[v]
                    t = _norm((-(self.coords[w][1] - b[1]),
                               self.coords[w][0] - b[0]))
                    self._reflect(sub, (b[0], b[1]),
                                  (b[0] + t[0], b[1] + t[1]))
                    self.congestion.update(sub)
                    gain = (before - self.congestion.score) / before
                    if gain >= self.cfg.refine_invert_improvement:
                        self.stats["inversions"] += 1
                        return True
                    self.coords[sub] = backup
                    self.congestion.update(sub)
        return False

    def try_bend_or_stretch(self, pairs) -> bool:
        """Level 3 (last resort): +/-10 deg x attempt bends or +0.2 bond x
        attempt stretches of an all-movable distal subtree along the
        congested path; the best trial is kept only on real gain."""
        cfg = self.cfg
        best: tuple[float, str, list[int], tuple] | None = None
        for _s, _i, _j, path in pairs:
            for (a, b) in self._center_first_bonds(path):
                mobile = self._mobile_side(a, b)
                if not mobile:
                    continue
                # pivot = the bond end OUTSIDE the mobile subtree; the
                # bend rotates / the stretch pushes the subtree away
                mset = set(mobile)
                pivot = a if b in mset else b
                far = b if pivot == a else a
                backup = self.coords[mobile].copy()
                v = self.coords[far] - self.coords[pivot]
                blen = math.hypot(*v) / cfg.bond
                unit = v / max(math.hypot(*v), 1e-9)
                for attempt in range(1, cfg.bend_max_attempts + 1):
                    for sgn in (1.0, -1.0):
                        self.coords[mobile] = backup
                        self._rotate_about(mobile, int(pivot),
                                           sgn * cfg.bend_step_deg * attempt)
                        self.congestion.update(mobile)
                        cand = (self.congestion.score, "bend",
                                list(mobile), (int(pivot),
                                               sgn * cfg.bend_step_deg
                                               * attempt))
                        if best is None or cand[0] < best[0]:
                            best = cand
                    self.coords[mobile] = backup
                    if blen + cfg.stretch_step_frac * attempt \
                            <= cfg.stretch_max_frac:
                        dist = cfg.stretch_step_frac * attempt * cfg.bond
                        self._translate(mobile, unit[0] * dist, unit[1] * dist)
                        self.congestion.update(mobile)
                        cand = (self.congestion.score, "stretch",
                                list(mobile), (float(unit[0]), float(unit[1]),
                                               dist))
                        if best is None or cand[0] < best[0]:
                            best = cand
                    self.coords[mobile] = backup
                    self.congestion.update(mobile)
        if best is None or best[0] >= self.congestion.score \
                - cfg.refine_bend_min_gain:
            return False
        _score, kind, mobile, args = best
        if kind == "bend":
            pivot, deg = args
            self._rotate_about(mobile, pivot, deg)
            self.stats["bends"] += 1
        else:
            ux, uy, dist = args
            self._translate(mobile, ux * dist, uy * dist)
            self.stats["stretches"] += 1
        self.congestion.update(mobile)
        return True

    # -- driver ---------------------------------------------------------------

    def run(self) -> dict:
        self.stats["pairs_initial"] = len(self.find_congested_pairs())
        for _ in range(self.cfg.refine_rounds):
            self.stats["rounds"] += 1
            pairs = self.find_congested_pairs()
            if not pairs:
                break
            if self.try_rotate(pairs):
                continue
            if self.try_macro_invert(pairs):
                continue
            if self.try_bend_or_stretch(pairs):
                continue
            break
        self.stats["pairs_final"] = len(self.find_congested_pairs())
        self.stats["congestion"] = round(self.congestion.score, 4)
        return self.stats


# ---------------------------------------------------------------------------
# force-field fallback (CoordGen two-step: terminal shrink + relaxation)
# ---------------------------------------------------------------------------

def forcefield_relax(coords: np.ndarray, movable: set[int],
                     congested: Sequence[tuple[int, int]],
                     adj: list[set[int]], ab_atom: np.ndarray,
                     ab_b0: np.ndarray, ab_b1: np.ndarray,
                     ab_rest2: np.ndarray, aa_i: np.ndarray,
                     aa_j: np.ndarray, aa_rest2: np.ndarray,
                     cfg: SidechainConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, dict]:
    """CoordgenMinimizer fallback: terminal bonds in collision shrink to
    10 % (the cheap avoidTerminalClashes move), then up to 1000 rounds of
    simultaneous repulsion relaxation over the side-chain atoms with a
    lowest-energy-frame rollback (100-round stagnation detection)."""
    marr = np.array(sorted(movable), dtype=int)
    mset = set(marr.tolist())
    stats: dict = {"terminal_shrunk": 0, "rounds": 0, "rolled_back": False,
                   "e_initial": 0.0, "e_final": 0.0}

    def energy(c: np.ndarray) -> float:
        e = 0.0
        if len(aa_i):
            d2 = (c[aa_i] - c[aa_j]) ** 2
            d2 = d2[:, 0] + d2[:, 1]
            over = d2 < aa_rest2
            if over.any():
                e += 0.5 * cfg.clash_linear_k2 * \
                    float((aa_rest2[over] - d2[over]).sum())
        if len(ab_atom):
            d2 = _point_seg_dist2(c[ab_atom], c[ab_b0], c[ab_b1])
            over = d2 < ab_rest2
            if over.any():
                e += 0.5 * cfg.clash_linear_k2 * \
                    float((ab_rest2[over] - d2[over]).sum())
        return e

    # 1) terminal shrink: degree-1 movable atoms named in a congested pair
    for (i, j) in congested:
        for a in (i, j):
            if a in mset and len(adj[a]) == 1:
                parent = next(iter(adj[a]))
                v = coords[a] - coords[parent]
                coords[a] = coords[parent] + cfg.ff_shrink_keep * v
                stats["terminal_shrunk"] += 1

    best = coords.copy()
    best_e = energy(coords)
    stats["e_initial"] = round(best_e, 4)
    window_best = best_e
    rounds_run = 0
    for rnd in range(cfg.ff_rounds):
        rounds_run = rnd + 1
        force = np.zeros_like(coords)
        if len(aa_i):
            diff = coords[aa_i] - coords[aa_j]
            d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
            hit = np.where(d2 < aa_rest2)[0]
            for k in hit:
                i, j = int(aa_i[k]), int(aa_j[k])
                d = math.sqrt(max(d2[k], 1e-9))
                rest = math.sqrt(aa_rest2[k])
                f = (rest - d)
                ux, uy = diff[k] / max(d, 1e-9)
                if i in mset:
                    force[i] += (ux * f, uy * f)
                if j in mset:
                    force[j] -= (ux * f, uy * f)
        if len(ab_atom):
            a = coords[ab_atom]
            b0 = coords[ab_b0]
            b1 = coords[ab_b1]
            ab = b1 - b0
            denom = np.maximum(ab[:, 0] ** 2 + ab[:, 1] ** 2, 1e-12)
            t = np.clip(((a - b0) * ab).sum(axis=1) / denom, 0.0, 1.0)
            proj = b0 + ab * t[:, None]
            diff = a - proj
            d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
            hit = np.where(d2 < ab_rest2)[0]
            for k in hit:
                atom = int(ab_atom[k])
                if atom not in mset:
                    continue
                d = math.sqrt(max(d2[k], 1e-9))
                rest = math.sqrt(ab_rest2[k])
                f = (rest - d)
                u = diff[k] / max(d, 1e-9)
                force[atom] += (u[0] * f, u[1] * f)
        step = force * cfg.ff_step
        norm = np.hypot(step[:, 0], step[:, 1])
        scale = np.where(norm > cfg.ff_step_cap, cfg.ff_step_cap /
                         np.maximum(norm, 1e-9), 1.0)
        step *= scale[:, None]
        mask = np.zeros(len(coords), dtype=bool)
        mask[marr] = True
        coords[mask] += step[mask]
        e = energy(coords)
        if e < best_e:
            best_e = e
            best = coords.copy()
        if (rnd + 1) % cfg.ff_stagnation_window == 0:
            if window_best - best_e < cfg.ff_stagnation_gain:
                break
            window_best = best_e
    stats["rounds"] = rounds_run
    if best_e < stats["e_initial"]:
        stats["rolled_back"] = True
        coords[:] = best
    stats["e_final"] = round(energy(coords), 4)
    return coords, stats


# ---------------------------------------------------------------------------
# layout orchestration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# odd-macroring contour: +/-60 sawtooth with Newton closure
# (the committed zigzag_layout_v2 machinery -- odd sizes cannot close on
# the pure hex lattice, and the pentagon-vertex merge leaves a sqrt(3)
# chord, so the drawn contour instead closes through a damped
# least-squares correction of the turns inside the 120 +/- 3 deg band;
# the closing bond carries the sub-pixel <= 0.02 bond deviation)
# ---------------------------------------------------------------------------

ZIG_TURN = 60.0
ANGLE_TOL = 3.0
FULL_TURN_BUDGET = 360.0
CLOSURE_TOL = 3e-8
ASPECT_TARGET = 1.15          # _ring_shape_score wide-flat target (v2)


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

def _backbone_crossings(pts: list[tuple[float, float]]) -> int:
    """Bond crossings of a closed chain (cycle-adjacent bonds share a
    vertex and never count)."""
    n = len(pts)
    cnt = 0
    for a in range(n):
        p, q = pts[a], pts[(a + 1) % n]
        for b in range(a + 1, n):
            if b == a or (a + 1) % n == b or (b + 1) % n == a:
                continue
            r, s = pts[b], pts[(b + 1) % n]
            if _seg_cross(p, q, r, s):
                cnt += 1
    return cnt

def _half_turns(m: int, rng: random.Random, target: float = 180.0,
                n_corners: int = 2, s0: float = 1.0) -> list[float] | None:
    """One half of a 2-fold-symmetric turn sequence: +/-60 alternation
    with base sign s0, CONVEX same-sign corner pairs at jittered sites,
    then the half sum pushed to `target` through the per-turn capacity of
    the +/-ANGLE_TOL band.  Convex-only corners keep the drawn ring round
    and its cavity open."""
    turns = [(s0 * ZIG_TURN if i % 2 == 0 else -s0 * ZIG_TURN)
             for i in range(m)]
    starts: list[int] = []
    occupied: set[int] = set()
    for k in range(n_corners):
        site = int(round((k + rng.uniform(0.15, 0.85)) * m
                         / n_corners)) % m
        probe = 0
        while any((site + d) % m in occupied for d in (-1, 0, 1, 2)):
            site = (site + 1) % m
            probe += 1
            if probe >= m:
                return None          # no room for another corner pair
        occupied.update({site, (site + 1) % m})
        starts.append(site)
    for site in starts:
        turns[site] = ZIG_TURN
        turns[(site + 1) % m] = ZIG_TURN
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



def _ring_shape_score(pts: list[tuple[float, float]], n: int,
                      bond: float) -> float:
    """Roundness score of a closed chain: rx/ry aspect near
    ASPECT_TARGET (wide-flat ~1:1), mean radius near the ideal circle
    radius n*bond/(2*pi) (the zigzag alphabet realises ~0.85x of it),
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
    r_ideal = n * bond / (2.0 * math.pi)
    circ = math.sqrt(sum((d - r_mean) ** 2 for d in ds) / n) / max(r_mean, 1e-9)
    aspect_pen = abs((w / max(h, 1e-9)) - ASPECT_TARGET)
    radius_pen = max(0.0, 0.80 - r_mean / r_ideal) * 6.0
    cavity_pen = max(0.0, 0.40 - min(ds) / r_ideal) * 3.0
    return 1.0 - aspect_pen * 1.5 - radius_pen - cavity_pen - circ * 1.0



def _bay_fit_score(pts: list[tuple[float, float]],
                   bay_need: list[int], bond: float) -> float:
    """Conditioned-shape term (CoordGen scorePath semantics, 0..1):
    a substituted vertex earns its bay — the outward bisector sector
    must clear the contour by a distance scaled to the substituent's
    heavy-atom count, so crowded residues locally bulge the outline the
    way real patent draftsmen draw them."""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    A = P
    Bv = np.roll(P, -1, axis=0)
    AB = Bv - A
    AB2 = np.maximum((AB * AB).sum(1), 1e-12)
    total = 0.0
    w_sum = 0
    for i, w in enumerate(bay_need):
        if w <= 0:
            continue
        w_sum += w
        d = P[i] - 0.5 * (P[(i - 1) % n] + P[(i + 1) % n])
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        u = d / norm
        need = min(0.4 + 0.15 * w, 1.6) * bond
        probe = P[i] + u * need
        t = np.clip(((probe - A) * AB).sum(1) / AB2, 0.0, 1.0)
        proj = A + t[:, None] * AB
        dist = np.linalg.norm(probe - proj, axis=1)
        dist[(i - 1) % n] = 1e9        # the bay's own two bonds don't count
        dist[i] = 1e9
        total += w * min(1.0, float(dist.min()) / (0.35 * bond))
    if w_sum == 0:
        return 1.0
    return total / w_sum


def build_backbone(n: int, rng: random.Random,
                   chord_idx_pairs: list[tuple] | None = None,
                   bond: float = BOND,
                   bay_need: list[int] | None = None
                   ) -> list[tuple[float, float]] | None:
    """Closed irregular zig-zag polygon: exact bond length bond, every
    interior angle within 120 +/- ANGLE_TOL, crossing-free.

    Base skeleton: `_round_turns` — the ±60 sawtooth with same-sign
    corner pairs budgeted to the 360 deg curl and spread evenly, closed
    exactly by the 2-fold turn pairing (odd sizes leave a ~1-bond gap to
    the Newton pass).  Candidates are scored by `_ring_shape_score`
    (wide-flat aspect ~1:1, mean radius near n*bond/(2*pi), visible
    interior cavity) and the best crossing-free chain wins — the round-2
    spatial-utilisation bar that replaced the folded polyomino
    contours."""
    best: tuple[float, list[tuple[float, float]]] | None = None
    fallback: list[tuple[float, float]] | None = None
    # tiny drawn cycles: the C2 half capacity cannot absorb the corner
    # curl and the small ring packs tighter than the round-oval gain is
    # worth — the round-1 polyomino net handles them directly
    if n <= 16:
        poly = _backbone_polyomino(n, rng, chord_idx_pairs, bond)
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
            g0 = math.hypot(*_positions_from_turns(n, turns, 0.0, bond)[-1])
            if g0 > 2.0 * bond:
                continue
            work = turns
        else:
            work = turns               # closed exactly by the C2 pairing
        pts2 = _close_positions(n, work, bond, rng,
                                gap_tol=0.02 * bond if n % 2 else None,
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
            best_local = _arc_min_clearance(pts2, i, j, ring_size, n, bond)
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
        score = _ring_shape_score(pts2, n, bond) - 0.5 * arc_pen
        if bay_need is not None:
            score += 2.0 * _bay_fit_score(pts2, bay_need, bond)
        if best is None or score > best[0]:
            best = (score, pts2)
    if best is not None:
        return best[1]
    return _backbone_polyomino(n, rng, chord_idx_pairs, bond)



def _backbone_polyomino(n: int, rng: random.Random,
                        chord_idx_pairs: list[tuple] | None = None,
                        bond: float = BOND
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
        pts = [(x * bond, y * bond) for (x, y) in pts]
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
                                                 heading, bond)
                gap = math.hypot(pts_walk[-1][0], pts_walk[-1][1])
                if best is None or gap < best[0]:
                    best = (gap, cand_work)
                if gap < 0.4 * bond:
                    break
            if best is None or best[0] > 1.1 * bond:
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
        pts2 = _close_positions(n, work, bond, rng,
                                gap_tol=0.02 * bond if odd else None)
        if pts2 is None:
            continue
        if _backbone_crossings(pts2) > 0:
            continue
        if any(abs(abs(t) - ZIG_TURN) > ANGLE_TOL + 1e-9 for t in work):
            continue
        if not _chords_fit(pts2, n, chord_idx_pairs, bond):
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
                       ring_size: int, n: int,
                       bond: float) -> float | None:
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
        best_local = max(best_local, mn / bond)
    return best_local



def _chords_fit(pts2: list[tuple[float, float]], n: int,
                chord_idx_pairs: list[tuple] | None,
                bond: float) -> bool:
    """Hard arc-fit gate for the PATH-ring chords: after the short-path
    reroute the anchors are ADJACENT drawn-cycle vertices and the small
    ring completes as a regular |R|-gon arc on one side of their bond —
    reject contours whose local cycle geometry wraps around the chord so
    tightly that BOTH sides collapse onto cycle atoms."""
    for ch in (chord_idx_pairs or []):
        hard = ch[3] if len(ch) > 3 else True
        if not hard:
            continue
        best_local = _arc_min_clearance(pts2, ch[0], ch[1], ch[2], n, bond)
        if best_local is not None and best_local < 0.45:
            return False
    return True



def _convex_flags(pts: list[tuple[float, float]]) -> list[bool]:
    """Convex (peak) vertex flags of a closed polygon via the cross
    product against the polygon orientation."""
    n = len(pts)
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    orient = 1.0 if area2 > 0 else -1.0
    flags = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = ((p1[0] - p0[0]) * (p2[1] - p1[1])
                 - (p1[1] - p0[1]) * (p2[0] - p1[0]))
        flags.append(cross * orient > 0)
    return flags


def _detect_path_rings(mol: Chem.Mol, order: list[int]) -> list[dict]:
    """Embedded rings whose atoms are ALL macroring atoms and contiguous
    along the macroring cycle (the piperidine of seq53/68/77).

    The ±60 hex-lattice contour cannot reproduce a full polygon curl and
    still close the loop, so the drawn cycle routes through the small
    ring's SHORT path: the shared run collapses to one chord edge and the
    new atoms come back as a pendant completion arc.  Returns per-ring
    dicts with the atoms to drop from the drawn cycle."""
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
        members = idxs[big + 1:] + idxs[:big + 1]
        out.append({"atoms": [order[m] for m in members[1:-1]],
                    "chord_atoms": (order[members[0]], order[members[-1]]),
                    "size": size})
    return out


def _substituent_bay_need(mol: Chem.Mol, drawn: list[int]) -> list[int]:
    """Per-drawn-vertex heavy-atom count of the off-cycle subtree: the
    conditioned-frame bay requirement (CoordGen scorePath's substituted-
    atom restraint in topological form), consumed by build_backbone."""
    bonded = _adjacency(mol)
    ring_set = set(drawn)
    need: list[int] = []
    for a in drawn:
        total = 0
        for nb in bonded[a]:
            if nb in ring_set:
                continue
            stack = [nb]
            seen = {a, nb}
            while stack:
                x = stack.pop()
                total += 1
                for y in bonded[x]:
                    if y not in seen and y not in ring_set:
                        seen.add(y)
                        stack.append(y)
        need.append(total)
    return need


def _detect_bridge_rings(mol: Chem.Mol, drawn: list[int]) -> list[dict]:
    """Small rings sharing >= 2 atoms with the drawn cycle but owning
    atoms outside it (seq53's fused lactam and bridged bicycle).

    The off-cycle run between its two shared-cycle neighbours returns as
    an anchored completion arc over that chord -- same dict shape as
    _detect_path_rings, so _complete_pendant_arcs consumes both.  Left
    undiscovered, the run is claimed as a rigid side-chain body whose
    ring closure back to the second anchor no DOF can satisfy, and the
    closing bond renders stretched."""
    pos = {a: i for i, a in enumerate(drawn)}
    aset = set(drawn)
    taken: set[int] = set()
    out: list[dict] = []
    for ring in sorted(mol.GetRingInfo().AtomRings(), key=len):
        rs = set(ring)
        size = len(ring)
        if size < 3 or size >= len(drawn) or rs <= aset:
            continue
        if rs & taken:
            continue              # overlapping an earlier bridge
        shared = rs & aset
        if not shared:
            continue                    # pure side-chain ring
        n = len(ring)
        start = next(k for k in range(n) if ring[k] in shared)
        seq = ring[start:] + ring[:start]   # rotates a shared atom to front
        runs: list[tuple[list[int], int, int]] = []
        cur: list[int] = []
        for k in range(1, n + 1):
            a = seq[k % n]
            if a not in shared:
                cur.append(a)
            else:
                if cur:
                    runs.append((cur, seq[k - len(cur) - 1], a))
                cur = []
        if len(runs) != 1:
            continue          # multiple off-cycle runs: not a simple bridge
        atoms, a_id, b_id = runs[0]
        if a_id not in pos or b_id not in pos:
            continue
        taken.update(atoms)
        out.append({"atoms": atoms, "chord_atoms": (a_id, b_id),
                    "size": size})
    return out


def _preplace_ring_carbonyl_oxygens(mol: Chem.Mol,
                                    fixed: dict[int, tuple[float, float]],
                                    ring_set: set[int], bond: float) -> None:
    """Backbone amide C=O oxygens protrude on the exterior bisector.

    Real patent drawings always draw the carbonyl oxygen pointing OUT of
    the macroring: the O-C bond bisects the reflex angle at the carbonyl
    carbon, making both O-C-N angles the obtuse ~120 deg.  Placement as
    a beam body lets clash pressure tuck O atoms into the cavity, which
    no real drawing does."""
    for c in ring_set:
        atom = mol.GetAtomWithIdx(c)
        if atom.GetSymbol() != "C":
            continue
        for nb in atom.GetNeighbors():
            b = mol.GetBondBetweenAtoms(c, nb.GetIdx())
            if (nb.GetSymbol() != "O" or nb.GetDegree() != 1
                    or b.GetBondType() != Chem.BondType.DOUBLE):
                continue
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetIdx() != nb.GetIdx() and n.GetIdx() in ring_set]
            if len(nbrs) != 2 or c not in fixed or nb.GetIdx() in fixed:
                continue
            pc = fixed[c]
            d = (2.0 * pc[0] - fixed[nbrs[0]][0] - fixed[nbrs[1]][0],
                 2.0 * pc[1] - fixed[nbrs[0]][1] - fixed[nbrs[1]][1])
            norm = math.hypot(*d)
            if norm < 1e-6:
                continue
            fixed[nb.GetIdx()] = (pc[0] + d[0] / norm * bond,
                                  pc[1] + d[1] / norm * bond)


def _backbone_frame(mol: Chem.Mol, seed: int, cfg: SidechainConfig
                    ) -> tuple[dict[int, tuple[float, float]], list[int],
                               list[dict], tuple[float, float]] | None:
    """Closed zig-zag contour for the drawn macroring cycle.

    The generator is zigzag_layout_v2's committed engine (round-2
    spatial tuning): +/-60 sawtooth with CONVEX corner pairs budgeted to
    the 360 deg curl, closed exactly for even sizes by the 2-fold turn
    pairing and Newton-closed inside the 120 +/- 3 deg band for odd
    sizes (closing bond deviation <= 0.02 bond, sub-pixel); the
    hex-lattice polyomino net handles tiny cycles; candidates are scored
    on roundness with PATH-ring chords arc-fit gated.  Bonds exact,
    angles in band, crossing-free.  Depiction-randomized (start vertex,
    walk direction, mirror) from `seed`, with the start PARITY
    following the draftsman rule: substituted ring atoms prefer CONVEX
    (peak) vertices -- at a sawtooth valley the only 120-deg slot faces
    the ring interior."""
    ring = zz1.find_macroring(mol)
    order = zz1.ordered_ring(mol, ring)
    rng = random.Random(seed * 2654435761 % (2 ** 31))
    path_rings = _detect_path_rings(mol, order)
    dropped: set[int] = set()
    for pr in path_rings:
        if dropped & set(pr["atoms"]):
            continue
        dropped |= set(pr["atoms"])
    drawn = [a for a in order if a not in dropped] if dropped else order
    n_drawn = len(drawn)

    # PATH-ring chords as drawn-cycle index pairs (hard arc-fit gates)
    pos_d = {a: i for i, a in enumerate(drawn)}
    chord_pairs: list[tuple] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pr in path_rings:
        ea, eb = pr["chord_atoms"]
        ia, ib = pos_d.get(ea), pos_d.get(eb)
        if ia is None or ib is None:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        chord_pairs.append((ia, ib, pr["size"], True))

    pts = build_backbone(n_drawn, rng, chord_pairs, bond=cfg.bond,
                         bay_need=_substituent_bay_need(mol, drawn))
    if pts is None:
        return None

    bonded_all = _adjacency(mol)
    macroring_set = set(order)
    substituted = {a for a in drawn
                   if any(nb not in macroring_set for nb in bonded_all[a])}
    conv_flags = _convex_flags(pts)

    def convex_count(off: int) -> int:
        return sum(1 for k, a in enumerate(drawn) if a in substituted
                   and conv_flags[(k + off) % n_drawn])

    start = rng.randrange(n_drawn)
    if convex_count(start + 1) > convex_count(start):
        start += 1
    mirror = rng.random() < 0.5
    fixed: dict[int, tuple[float, float]] = {}
    for k, aid in enumerate(drawn):
        x, y = pts[(k + start) % n_drawn]
        fixed[aid] = (-x, y) if mirror else (x, y)
    cx = sum(p[0] for p in fixed.values()) / n_drawn
    cy = sum(p[1] for p in fixed.values()) / n_drawn
    return fixed, drawn, path_rings, (cx, cy)


def _complete_pendant_arcs(mol: Chem.Mol, fixed: dict[int, tuple[float, float]],
                           drawn: list[int], path_rings: list[dict],
                           ring_center: tuple[float, float],
                           cfg: SidechainConfig) -> bool:
    """Complete each embedded path ring as a regular |R|-gon arc anchored
    on its (now adjacent) chord endpoints -- both bulge sides scored;
    collision-free exterior side preferred, interior side takes over when
    the exterior crosses or collapses (the way the real crops draw
    boundary small rings)."""
    bonded = _adjacency(mol)
    all_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in mol.GetBonds()]
    chord_to_ring: dict[tuple[int, int], dict] = {}
    for pr in path_rings:
        if pr["atoms"][0] in fixed:
            continue                     # already completed (nested drops)
        chord_to_ring[tuple(sorted(pr["chord_atoms"]))] = pr
    for chord, pr in chord_to_ring.items():
        a_id, b_id = chord
        new_atoms = list(pr["atoms"])
        pa, pb = fixed[a_id], fixed[b_id]
        edges = len(new_atoms) + 1
        ring_size = pr["size"]
        chord_len = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        radius = (chord_len / 2.0) / math.sin(math.pi * edges / ring_size)
        if chord_len / 2.0 > radius + 1e-9:
            return False
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        perp = _norm((-(pb[1] - pa[1]), pb[0] - pa[0]))
        h = math.sqrt(max(radius * radius - chord_len * chord_len / 4.0, 0.0))
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
            trial: dict[int, tuple[float, float]] = {}
            for k in range(1, edges):
                t = ang0 + sweep * k / edges
                trial[new_atoms[k - 1]] = \
                    (center[0] + radius * math.cos(t),
                     center[1] + radius * math.sin(t))
            # clearance of the trial atoms against the fixed frame
            clear = float("inf")
            for i, c in trial.items():
                for j, p in fixed.items():
                    if j in bonded[i]:
                        continue
                    clear = min(clear, math.hypot(p[0] - c[0], p[1] - c[1]))
            crossings = 0
            pos_all = dict(fixed)
            pos_all.update(trial)
            chain = [a_id] + new_atoms + [b_id]
            for k in range(len(chain) - 1):
                p1, q1 = pos_all[chain[k]], pos_all[chain[k + 1]]
                for (i1, j1) in all_bonds:
                    if len({chain[k], chain[k + 1], i1, j1}) < 4:
                        continue
                    if i1 not in fixed or j1 not in fixed:
                        continue
                    if _seg_cross(p1, q1, fixed[i1], fixed[j1]):
                        crossings += 1
            outward = _arc_outward(pa, pb, perp, side, ring_center)
            score = min(clear, 2.0 * cfg.bond) \
                - 3.0 * max(0.0, 0.55 * cfg.bond - clear) \
                - 12.0 * cfg.bond * crossings \
                + 0.8 * cfg.bond * (1.0 if outward else 0.0)
            sides.append((score, trial))
        if not sides:
            return False
        sides.sort(key=lambda s: -s[0])
        for i, p in sides[0][1].items():
            fixed[i] = p
    return True


def _arc_outward(pa, pb, perp, side, center) -> bool:
    mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    v = (mid[0] - center[0], mid[1] - center[1])
    return (v[0] * perp[0] * side + v[1] * perp[1] * side) >= 0.0


def _claim_bodies(mol: Chem.Mol, fixed_pos: dict[int, tuple[float, float]],
                  placed_order: list[int], bond: float
                  ) -> list[SidechainTemplate]:
    """One rigid template per unplaced neighbour subtree; largest subtrees
    are discovered first so they claim their atoms (long chains get the
    first choice of slots in the beam's DOF order)."""
    bonded = _adjacency(mol)
    claimed: set[int] = set(fixed_pos)
    bodies: list[SidechainTemplate] = []
    for root in placed_order:
        for kid in sorted(bonded[root]):
            if kid in claimed:
                continue
            body = SidechainTemplate(mol, root, kid, bond, claimed)
            for i in body.atoms:
                claimed.add(i)
            bodies.append(body)
    bodies.sort(key=lambda b: (-b.size, b.root))
    return bodies


def beam_layout_coords(mol: Chem.Mol, seed: int = 0,
                       cfg: SidechainConfig = DEFAULT_CONFIG
                       ) -> tuple[Chem.Mol | None, dict]:
    """Lay `mol` out as a zig-zag macroring with collision-aware beam
    search side-chain placement.  Returns (molecule with coordinates,
    report); the molecule is None only for hard failures."""
    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 0:
        return None, {"error": "empty"}
    if work.GetRingInfo().NumRings() == 0:
        Chem.FastFindRings(work)
    frame = _backbone_frame(work, seed, cfg)
    if frame is None:
        return None, {"error": "backbone"}
    fixed, drawn, path_rings, ring_center = frame
    bridge_rings = _detect_bridge_rings(work, drawn)
    if not _complete_pendant_arcs(work, fixed, drawn,
                                  path_rings + bridge_rings,
                                  ring_center, cfg):
        return None, {"error": "pendant_arc"}
    ring_atoms = list(drawn)
    ring_set = set(ring_atoms)
    _preplace_ring_carbonyl_oxygens(work, fixed, ring_set, cfg.bond)
    placed_order = list(ring_atoms) + \
        [i for i in sorted(fixed) if i not in ring_set]
    bodies = _claim_bodies(work, fixed, placed_order, cfg.bond)
    if not bodies:
        return None, {"error": "no_sidechain"}
    scene = LayoutScene(work, fixed, bodies, {}, ring_atoms,
                        ring_center, cfg=cfg)
    # seed every body's first-bond direction from the gap heuristic;
    # a seed pose reaching into the central void falls back to the most
    # radially-outward candidate (real drawings point substituents away
    # from the ring centre even at sawtooth valleys)
    def cavity_atoms(body: SidechainTemplate, theta: float) -> int:
        placed = body.pose(scene.fixed_pos[body.root], theta)
        pts = np.array([placed[i] for i in body.atoms])
        if scene.cavity_r <= 0.0:
            return 0
        d = pts - scene.ring_center
        return int((np.hypot(d[:, 0], d[:, 1]) < scene.cavity_r).sum())

    attach: dict[int, float] = {}
    for bi, body in enumerate(scene.bodies):
        cands = scene.first_bond_candidates(body)
        if not cands:
            return None, {"error": "candidates", "body": bi}
        angle = cands[0][1]
        if cavity_atoms(body, angle):
            rx = scene.fixed_pos[body.root][0] - scene.ring_center[0]
            ry = scene.fixed_pos[body.root][1] - scene.ring_center[1]
            r_norm = math.hypot(rx, ry)
            if r_norm > 1e-6:
                best_dot = -2.0
                for _w, ang, d in cands:
                    dot = (d[0] * rx + d[1] * ry) / r_norm
                    if dot > best_dot + 1e-2:
                        best_dot = dot
                        angle = ang
        attach[bi] = angle
    scene.attach_angle = attach

    # sibling bodies of one ring atom draw on OPPOSITE slots (wedge /
    # dash convention in the real crops): mirror every second sibling
    root_bodies: dict[int, list[int]] = {}
    for bi, body in enumerate(scene.bodies):
        root_bodies.setdefault(body.root, []).append(bi)
    seed_state = list(scene.initial_solution)
    for root, bis in root_bodies.items():
        for bi in bis[1:]:
            lo, _hi = scene.body_slices[bi]
            seed_state[lo] = 1
    scene.initial_solution = tuple(seed_state)

    beam = run_beam_search(scene, cfg)
    pv = scene.pose_view(beam.solution)
    coords = pv.coords.copy()

    refine_stats: dict = {}
    ff_stats: dict = {}
    final_clean = beam.clean
    if not final_clean:
        movable: set[int] = set()
        for body in scene.bodies:
            movable |= body.atom_set
        refiner = CdkStyleRefiner(work, coords, movable, scene.ring_set, cfg)
        refine_stats = refiner.run()
        final_clean = refine_stats["pairs_final"] == 0
        if not final_clean:
            congested = [(i, j) for _s, i, j, _p
                         in refiner.find_congested_pairs()]
            coords, ff_stats = forcefield_relax(
                coords, movable, congested, scene.bonded,
                scene.ab_atom, scene.ab_b0, scene.ab_b1, scene.ab_rest2,
                scene.aa_i, scene.aa_j, scene.aa_rest2, cfg)
            final_clean = ff_stats.get("e_final", 0.0) < cfg.clean_energy

    for i, p in enumerate(coords):
        if i not in fixed:
            fixed[i] = (float(p[0]), float(p[1]))
    all_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in work.GetBonds()]
    metrics = _layout_metrics(fixed, scene.bonded, ring_atoms,
                              all_bonds, cfg)
    qa_pass = _qa_pass(metrics, ring_atoms, cfg)
    orientation = {"out": 0, "in": 0, "mid": 0}
    per_body = []
    for bi, body in enumerate(scene.bodies):
        lo, hi = scene.body_slices[bi]
        state = beam.solution[lo:hi]
        flip = state[0]
        _coords, _cent, inside, _centroid_in = \
            scene._body_pose(bi, tuple(state))
        dc = math.hypot(_cent[0] - scene.ring_center[0],
                        _cent[1] - scene.ring_center[1])
        if dc < scene.cavity_r:
            tag = "in"
        elif scene.cavity_r > 0.0 and \
                (np.hypot(_coords[:, 0] - scene.ring_center[0],
                          _coords[:, 1] - scene.ring_center[1])
                 < scene.cavity_r).any():
            tag = "mid"
        else:
            tag = "out"
        orientation[tag] += 1
        per_body.append({
            "root": body.root, "size": body.size, "flip": int(flip),
            "shrunk": int(state[-1]),
            "turns_flipped": int(sum(1 for s in state[1:-1] if s)),
            "orientation": tag,
        })

    conf = Chem.Conformer(work.GetNumAtoms())
    for i in range(work.GetNumAtoms()):
        x, y = fixed[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    stereo_ok = zzv2.stereo_roundtrip_ok(work)

    rep: dict = {
        "ok": True,
        "seed_used": seed,
        "ring_atoms": len(ring_atoms),
        "bodies": len(scene.bodies),
        "dofs": len(scene.dofs),
        "beam": {"rounds": beam.rounds, "scored": beam.scored,
                 "tier_reached": beam.tier_reached,
                 "score": round(beam.score, 4),
                 "terms": {k: round(v, 4) for k, v in beam.breakdown.items()},
                 "clean": beam.clean},
        "orientation": orientation,
        "per_body": per_body,
        "refine": refine_stats,
        "forcefield": ff_stats,
        "physical_clean": final_clean,
        "metrics": metrics,
        "qa_pass": qa_pass,
        "stereo_roundtrip": stereo_ok,
    }
    return work, rep


def _qa_pass(metrics: dict, ring_atoms: list[int],
             cfg: SidechainConfig) -> bool:
    len_tol = 0.02 * cfg.bond if len(ring_atoms) % 2 else 1e-6
    return (abs(metrics["ring_bond_len_min"] - cfg.bond) < len_tol
            and abs(metrics["ring_bond_len_max"] - cfg.bond) < len_tol
            and metrics["angle_min"] >= 120.0 - cfg.angle_tol
            and metrics["angle_max"] <= 120.0 + cfg.angle_tol
            and metrics["min_nonbonded_ratio"] >= cfg.qa_floor_frac
            and metrics["bond_crossings"] == 0
            and metrics["n_close_pairs"] == 0)


def _layout_metrics(pos: dict[int, tuple[float, float]],
                    bonded: list[set[int]], ring_atoms: list[int],
                    all_bonds: list[tuple[int, int]],
                    cfg: SidechainConfig) -> dict:
    n = len(ring_atoms)
    lens = []
    for k, a in enumerate(ring_atoms):
        b = ring_atoms[(k + 1) % n]
        p, q = pos[a], pos[b]
        lens.append(math.hypot(q[0] - p[0], q[1] - p[1]))
    angles = []
    for k in range(n):
        a, b, c = ring_atoms[k - 1], ring_atoms[k], \
            ring_atoms[(k + 1) % n]
        v1 = _norm((pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]))
        v2 = _norm((pos[c][0] - pos[b][0], pos[c][1] - pos[b][1]))
        angles.append(_angle_between(v1, v2))
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
    crossings = 0
    for k in range(len(all_bonds)):
        i1, j1 = all_bonds[k]
        for l in range(k + 1, len(all_bonds)):
            i2, j2 = all_bonds[l]
            if len({i1, j1, i2, j2}) < 4:
                continue
            if _seg_cross(pos[i1], pos[j1], pos[i2], pos[j2]):
                crossings += 1
    return {
        "ring_bond_len_min": round(min(lens), 4),
        "ring_bond_len_max": round(max(lens), 4),
        "angle_min": round(min(angles), 2),
        "angle_max": round(max(angles), 2),
        "angle_mean": round(sum(angles) / n, 2),
        "min_nonbonded_ratio": round(mind / cfg.bond, 3),
        "close_pairs": close,
        "n_close_pairs": len(close),
        "soft_pairs": soft,
        "n_soft_pairs": len(soft),
        "bond_crossings": crossings,
    }


def beam_layout_with_retry(mol: Chem.Mol, seed: int = 0, seeds: int = 6,
                           cfg: SidechainConfig = DEFAULT_CONFIG
                           ) -> tuple[Chem.Mol | None, dict]:
    """Seed loop around `beam_layout_coords`: the first QA-passing layout
    wins; otherwise the best-effort layout (fewest collapse pairs, then
    crossings, then largest minimum non-bonded distance) is returned with
    its crowding report."""
    best: tuple[Chem.Mol | None, dict] | None = None

    def rank(rep: dict) -> tuple:
        m = rep.get("metrics", {})
        return (rep.get("qa_pass", False),
                -m.get("n_close_pairs", 99),
                -m.get("bond_crossings", 99),
                m.get("min_nonbonded_ratio", -9.0))

    last: dict = {"error": "no_seed_passed"}
    for k in range(seeds):
        laid, rep = beam_layout_coords(mol, seed=seed + k, cfg=cfg)
        if laid is None:
            last = dict(rep)
            last["seed_used"] = seed + k
            continue
        rep["seed_used"] = seed + k
        if rep.get("qa_pass") and rep.get("stereo_roundtrip"):
            return laid, rep
        if best is None or rank(rep) > rank(best[1]):
            best = (laid, rep)
    if best is not None:
        return best
    return None, last


# ---------------------------------------------------------------------------
# rendering + validation suite (ALIGNED black/white, shared with v2)
# ---------------------------------------------------------------------------

def render_beam(smiles: str, out_png: str, seed: int = 0, upright: bool = True,
                retries: int = 8,
                cfg: SidechainConfig = DEFAULT_CONFIG) -> dict:
    """Layout + ALIGNED-style render of one SMILES.  Pure black/white
    asserted (palette + single-channel ink); the accepted coordinates are
    only ever transformed globally (the renderer draws a COPY)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES failed to parse: {smiles[:80]}")
    laid, rep = beam_layout_with_retry(mol, seed=seed, seeds=retries, cfg=cfg)
    if laid is None:
        raise ValueError(f"beam side-chain layout failed: {rep}")
    profile = zzv2.render_profile()
    zzv2._assert_black_white(profile)
    rng_draw = random.Random(rep["seed_used"] + 101)
    if upright:
        angle = rng_draw.uniform(0, 360)
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
    np.random.seed(rep["seed_used"] + 101)
    img = zzv2.draw_molecule(render_mol, profile, rng_draw)
    zzv2._assert_grayscale(img)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)
    rep["png"] = out_png
    rep["shape"] = zzv2.shape_metrics(laid)
    return rep


def beam_validation_cases() -> list[tuple[str, str, str]]:
    """(case id, smiles, real crop path): the seq53/60/68/77 gold set plus
    one KRAS and one GPC3 verified SMILES for generality."""
    cases: list[tuple[str, str, str]] = []
    cases.append(("seq53_gold", zzv2._gold_smiles(zzv2.SEQ53_GOLD_KEY),
                  os.path.join(zzv2.REAL_CROP_ROOT,
                               "US20240287138A1/page_024_box_0.png")))
    for sid in (53, 60, 68, 77):
        smi, crop = zzv2._verified_smiles_for_seq(sid)
        cases.append((f"seq{sid}", smi, crop))
    cases.append(("seq68_gold", zzv2._gold_smiles("68"),
                  os.path.join(zzv2.REAL_CROP_ROOT,
                               "US20240287138A1/page_024_box_2.png")))
    for mid, smi, crop in zzv2._verified_pairs(zzv2.KRAS_MANIFEST, 1,
                                               "/tmp/crops_kras"):
        cases.append((f"kras_{mid}", smi, crop))
    for mid, smi, crop in zzv2._verified_pairs(zzv2.GPC3_MANIFEST, 1,
                                               "/tmp/crops_gpc3"):
        cases.append((f"gpc3_{mid}", smi, crop))
    return cases


def validation_suite(out_dir: str, seed: int = 0) -> dict:
    """Render every validation case + per-case real-vs-render comparison
    sheet; returns the JSON-able summary (printed by __main__)."""
    rows: dict[str, dict] = {}
    for case, smi, crop in beam_validation_cases():
        png = os.path.join(out_dir, f"{case}.png")
        try:
            rep = render_beam(smi, png, seed=seed)
            sheet = os.path.join(out_dir, f"compare_{case}.png")
            zzv2.comparison_sheet(
                crop, png,
                f"{case} - real patent crop (left) vs beam layout (right)",
                sheet)
            m = rep["metrics"]
            rows[case] = {
                "ok": True,
                "png": png,
                "compare": sheet,
                "seed": rep["seed_used"],
                "ring_atoms": rep["ring_atoms"],
                "bodies": rep["bodies"],
                "dofs": rep["dofs"],
                "beam_score": rep["beam"]["score"],
                "beam_clean": rep["beam"]["clean"],
                "orientation": rep["orientation"],
                "angle_min": m["angle_min"],
                "angle_max": m["angle_max"],
                "ring_bond_len_min": m["ring_bond_len_min"],
                "ring_bond_len_max": m["ring_bond_len_max"],
                "min_nonbonded_ratio": m["min_nonbonded_ratio"],
                "n_close_pairs": m["n_close_pairs"],
                "n_soft_pairs": m["n_soft_pairs"],
                "bond_crossings": m["bond_crossings"],
                "qa_pass": rep["qa_pass"],
                "physical_clean": rep["physical_clean"],
                "stereo_roundtrip": rep["stereo_roundtrip"],
                "refine": rep["refine"],
                "forcefield": rep["forcefield"],
                "shape": rep["shape"],
            }
        except Exception as exc:             # per-case isolation
            rows[case] = {"ok": False, "error": str(exc)[:200]}
    return rows


if __name__ == "__main__":  # pragma: no cover - validation CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="collision-aware beam side-chain layout validation")
    ap.add_argument("--out", default="/data/BioChemInsight/images/beam_sidechain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(validation_suite(args.out, seed=args.seed), indent=1))


def patent_label_style(mol: Chem.Mol) -> Chem.Mol:
    """Backbone amide N (deg-2, neutral, not in a small ring) gets an
    explicit H label — real patent drawings show 'NH'/'HN' on every
    backbone nitrogen; RDKit omits the implicit H by default."""
    out = Chem.Mol(mol)
    for a in out.GetAtoms():
        if (a.GetSymbol() == "N" and a.GetDegree() == 2
                and a.GetFormalCharge() == 0
                and not a.IsInRingSize(6) and not a.IsInRingSize(5)
                and a.GetNumExplicitHs() == 0):
            a.SetNumExplicitHs(1)
            a.SetNoImplicit(True)
    return out


def resolve_carbonyl_label_collisions(mol: Chem.Mol,
                                      min_frac: float = 0.55
                                      ) -> int:
    """Flip terminal carbonyl oxygens whose label position crowds the
    next residue's amide N (user-flagged defect: O and N labels nearly
    overlap at backbone C(=O)-N junctions).  Returns flips applied."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pts = np.array([[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y]
                    for a in range(n)])
    L = float(np.median([math.dist(pts[b.GetBeginAtomIdx()],
                                   pts[b.GetEndAtomIdx()])
                         for b in mol.GetBonds()]) or 30.0)
    min_d = min_frac * L
    flips = 0
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondTypeAsDouble() != 2.0:
            continue
        c = a2 if a1.GetSymbol() == "C" and a2.GetSymbol() == "O" else \
            a1 if a2.GetSymbol() == "C" and a1.GetSymbol() == "O" else None
        o = a1 if c is a2 else a2 if c is a1 else None
        if c is None or o is None or o.GetDegree() != 1:
            continue
        # nearest heteroatom label (N/O) that is not the carbonyl pair
        crowd = None
        for a in mol.GetAtoms():
            if (a.GetIdx() in (c.GetIdx(), o.GetIdx())
                    or a.GetSymbol() not in ("N", "O")):
                continue
            if math.dist(pts[a.GetIdx()], pts[o.GetIdx()]) < min_d:
                crowd = a
                break
        if crowd is None:
            continue
        # mirror O across the carbonyl C
        new = 2 * pts[c.GetIdx()] - pts[o.GetIdx()]
        conf.SetAtomPosition(o.GetIdx(),
                             [float(new[0]), float(new[1]), 0.0])
        pts[o.GetIdx()] = new
        flips += 1
    return flips


def native_zigzag_macrocycle(mol: Chem.Mol, amide_only: bool = True
                             ) -> Chem.Mol | None:
    """Patent-style macrocycle layout via open-chain zigzag + closure.

    Cuts one backbone amide bond in the macrocycle, draws the resulting
    open chain with RDKit's native depictor (which produces the standard
    zig-zag), then re-closes the ring by translating the tail along the
    cut axis to one bond length.  Returns None if no amide bond exists or
    the draw fails."""
    work = Chem.Mol(mol)
    if work.GetNumConformers() == 0:
        AllChem.Compute2DCoords(work)
    ri = work.GetRingInfo()
    try:
        ring = max(ri.AtomRings(), key=len)
    except ValueError:
        return None
    cut = None
    for b in work.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring and j in ring and (not amide_only or
                {work.GetAtomWithIdx(i).GetSymbol(),
                 work.GetAtomWithIdx(j).GetSymbol()} == {"N", "C"}):
            cut = (i, j)
            break
    if cut is None:
        return None
    em = Chem.RWMol(work)
    em.RemoveBond(*cut)
    f = em.GetMol()
    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(f, canonOrient=True)
    conf = f.GetConformer()
    p1 = conf.GetAtomPosition(cut[0])
    p2 = conf.GetAtomPosition(cut[1])
    gap = math.dist((p1.x, p1.y), (p2.x, p2.y))
    if gap < 1e-6:
        return None
    shrink = 30.0 / gap
    emconf = em.GetConformer()
    for a in range(em.GetNumAtoms()):
        p = emconf.GetAtomPosition(a)
        emconf.SetAtomPosition(a, [(p.x - p1.x) * shrink + p1.x,
                                   (p.y - p1.y) * shrink + p1.y, 0.0])
    em.AddBond(cut[0], cut[1], Chem.BondType.SINGLE)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out

"""Continuous layout optimizer: refine a CoordGen macroring depiction into
the spread, collision-free, zig-zag patent drawing style.

Motivation: CoordGen draws large peptide macrorings as round wheels whose
side chains collapse onto the ring core (circularity ~0, min non-bonded
near 0).  Real patent drawings spread the ring across the canvas, radiate
side chains outward and carry the +-radial zig-zag rhythm on the ring
contour.  This module casts that refinement as a smooth energy
minimization over the 2n coordinate vector and solves it with scipy
L-BFGS-B using exact analytic gradients.

Energy terms (all vectorized numpy, one closure returns (E, grad)):

  bond springs     wb  — every bond at the molecule's standard length L
                         (L grows mildly with atom count: bigger drawings
                         fill the canvas)
  angle springs    wa  — every neighbour pair around a two- or
                         three-neighbour skeleton atom targets 120 deg
                         (cosine form; terminal atoms excluded by degree).
                         Degree-2 ring atoms are the saw-tooth turns —
                         unconstrained they collapse into straight
                         collinear runs, which also breaks RDKit's wedge
                         rendering.  Around four-neighbour (quaternary)
                         centres a one-sided hinge instead bans narrow
                         bond sectors (< 45 deg), the geometry RDKit's
                         wedge heuristic needs.
  ring spread      wr  — macroring atoms pulled to per-atom radial targets
                         base_radius * (1 + 0.22 * subtree share): residues
                         carrying bulky side chains sit further out
  ring zig-zag     wz  — adjacent ring atoms alternate +-delta radial
                         offsets around their spread target (the patent
                         saw-tooth signature)
  side-chain spread ws  — exocyclic branches sharing an attachment atom
                         maximize their mutual angle (target 2*pi*k/m);
                         side-chain leaf atoms carry a weak outward hinge
                         spring against the ring centroid
  chirality barrier wc — at every stereocentre the rotational order of the
                         heavy neighbours is preserved from the start
                         layout (soft hinge on each neighbour-pair cross
                         product sign).  Without it the continuous solver
                         freely mirrors depicted stereocentres and the
                         rendered wedges contradict the SMILES tags.
  repulsion        wn  — every atom pair at graph distance >= 3 repels as
                         1/d softened at short range (the anti-overlap term)

The ring centroid is recomputed from current positions each evaluation and
its exact gradient contribution (a rank-1 correction over ring atoms) is
included, so the returned gradient is the true analytic gradient.

Solver: L-BFGS-B, maxiter 800, ftol/gtol convergence.  Multi-start —
CoordGen coordinates, CDK StructureDiagramGenerator coordinates (mapped
back to input atom order via canonical ranks) and three seeded Gaussian
perturbations — is ranked at a coarse tolerance, then only the winning
basin is re-solved to full convergence (phase 2).  Each start is solved
under its own depicted stereocentre parity.  A post-solve collision gate
recounts non-bond conflicts (graph distance >= 3 and euclidean distance
< 0.6 * L); any remaining conflict triggers one re-solve from the
rejected layout with the repulsion weight escalated.  A second repair
pass detects bond-segment crossings — which no atom-atom distance gate
can see — pushes the crossing bonds' atom pairs into the repulsion set
and re-solves locally, keeping the result only when the crossing count
drops without reintroducing conflicts.

Entry points: `optimize_layout(smiles_or_mol)` returning an
`OptimizeResult`, `collision_conflicts()` / `bond_crossings()` for the
gates, `layout_metrics()` for before/after numbers, and a CLI that
renders the seq53 CoordGen-vs-optimized comparison sheet and runs the
multi-molecule gate suite.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult as SciOptimizeResult
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style_calibration import ALIGNED, draw_molecule
from render_via_engine import EngineRenderError
from zigzag_layout import find_macroring, ordered_ring

RDLogger.DisableLog("rdApp.*")

# --- geometry tunables ------------------------------------------------------

BASE_BOND = 1.0          # standard bond length at the reference size
SPREAD_REF_ATOMS = 40    # atom count below which L stays at BASE_BOND
SPREAD_PER_ATOM = 0.0025 # L growth per atom above the reference size
SPREAD_MAX_SCALE = 1.35  # cap of the molecular bond-length scale
ZIGZAG_DELTA_FRAC = 0.30     # radial zig-zag amplitude, fraction of L
SUBTREE_RADIUS_FRAC = 0.22   # max extra radius for the bulkiest subtree
RING_RADIUS_MIN_BONDS = 1.8  # small-ring floor for the base radius (in L)
SIDE_OUTWARD_RADIUS = 1.05   # leaf hinge radius, multiple of base radius
REPULSION_SOFTENING = 0.25   # 1/d softening length (layout units)
CONFLICT_DISTANCE_FRAC = 0.6 # gate: conflict when d < frac * L
REPULSION_ESCALATION = 4.0   # repulsion weight multiplier on gate retry
CROSSING_ESCALATION = 8.0    # repulsion weight multiplier on crossing repair
CROSSING_REPAIR_ROUNDS = 5
QUATERNARY_NARROW_COS = math.cos(math.radians(45.0))  # banned bond sector
CHIRAL_ANGLE_MULT = 6.0  # angle-spring stiffness multiplier at stereocentres
COVERAGE_FLOOR_FRAC = 0.15   # radius-coverage denominator floor (mean share)

# --- solver tunables --------------------------------------------------------

LBFGS_MAXITER = 800
LBFGS_GTOL = 1e-6
# phase 1 ranks the starts at a coarse tolerance; only the winner is
# re-solved to full convergence in phase 2
LBFGS_RANK_FTOL = 1e-8
LBFGS_FINAL_FTOL = 1e-10
LBFGS_PERTURB_MAXITER = 60   # perturbation starts are tie-breakers only
LBFGS_PERTURB_FTOL = 1e-6    # their energies only rank starts, not output
LBFGS_CROSSING_MAXITER = 400  # bond-crossing repair re-solve cap
N_PERTURB_STARTS = 3
OUTPUT_MEDIAN_BOND = 1.5     # coordinate scale written to the output mol


class OptimizeLayoutError(Exception):
    """The layout optimization could not be performed on this input."""


@dataclass(frozen=True)
class OptimizeWeights:
    """Relative stiffness of the six energy terms."""

    bond: float = 20.0
    angle: float = 3.0
    ring_spread: float = 1.0
    ring_zigzag: float = 3.0
    side_chain: float = 1.0
    chirality: float = 25.0
    repulsion: float = 0.8


DEFAULT_WEIGHTS = OptimizeWeights()


@dataclass
class OptimizeResult:
    """Outcome of `optimize_layout`: optimized molecule plus diagnostics."""

    smiles: str
    mol: Chem.Mol                    # input atom order, optimized conformer
    energy: float
    seconds: float
    start_used: str
    start_energies: dict[str, float] = field(default_factory=dict)
    starts_skipped: dict[str, str] = field(default_factory=dict)
    n_conflicts_before: int = 0
    n_conflicts_after: int = 0
    gate_pass: bool = False
    reoptimized: bool = False
    n_crossings: int = 0
    crossing_repaired: bool = False
    n_iterations: int = 0


@dataclass
class LayoutMetrics:
    """Shape statistics used for before/after acceptance checks."""

    radius_coverage: float           # max/min ring radius (floored)
    ring_mean_radius_over_bond: float  # mean ring radius in units of L
    zigzag_amplitude: float          # mean |r_k - median(r)| in units of L
    subtree_min_gap_deg: float       # smallest angular gap between subtree centroids
    subtree_mean_gap_deg: float
    side_atom_min_gap_deg: float     # smallest cross-subtree atom-level angle
    branch_min_angle_deg: float      # smallest angle between co-attached branches
    n_conflicts: int


class _Topology:
    """Precomputed arrays consumed by the vectorized energy closure.

    Built once per molecule: bond list, skeleton angle triples (all
    two-/three-neighbour atoms), quaternary narrow-sector triples,
    ordered macroring with alternating zig-zag signs and per-atom radial
    targets, side-chain branch pairs at multi-branch attachments, leaf
    atoms for the outward spring, stereocentre neighbour pairs, and all
    graph-distance >= 3 pairs.
    """

    __slots__ = (
        "n", "bond_i", "bond_j", "ang_c", "ang_u", "ang_w", "ang_mult",
        "q4_c", "q4_u", "q4_w", "chi_c", "chi_u", "chi_w",
        "ring", "ring_sign", "ring_target",
        "subtree_size", "branch_pairs", "branch_pair_target",
        "leaves", "nb_i", "nb_j", "bond_length", "base_radius",
    )

    def __init__(self, mol: Chem.Mol) -> None:
        self.n = mol.GetNumAtoms()
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in mol.GetBonds()]
        if not bonds:
            raise OptimizeLayoutError("molecule has no bonds")
        self.bond_i = np.array([b[0] for b in bonds], dtype=np.intp)
        self.bond_j = np.array([b[1] for b in bonds], dtype=np.intp)

        nbrs: list[list[int]] = [[] for _ in range(self.n)]
        for i, j in bonds:
            nbrs[i].append(j)
            nbrs[j].append(i)

        # skeleton angle triples: every two- or three-neighbour heavy atom
        # targets 120 deg (terminal atoms excluded by degree, >= 4-neighbour
        # centres handled separately below)
        centers, us, ws_ = [], [], []
        for a, nb in enumerate(nbrs):
            if len(nb) == 3:
                for u, w in ((0, 1), (0, 2), (1, 2)):
                    centers.append(a)
                    us.append(nb[u])
                    ws_.append(nb[w])
            elif len(nb) == 2:
                centers.append(a)
                us.append(nb[0])
                ws_.append(nb[1])
        self.ang_c = np.array(centers, dtype=np.intp)
        self.ang_u = np.array(us, dtype=np.intp)
        self.ang_w = np.array(ws_, dtype=np.intp)
        # stereocentre triples are stiffened: RDKit's chiral-H placement
        # needs an open ~60-deg sector at the tag-mandated slot, which the
        # softer backbone springs alone do not guarantee
        chiral = {a for a in range(self.n)
                  if mol.GetAtomWithIdx(a).GetChiralTag()
                  != Chem.ChiralType.CHI_UNSPECIFIED}
        self.ang_mult = np.array(
            [CHIRAL_ANGLE_MULT if c in chiral else 1.0 for c in centers])

        # quaternary centres: all neighbour pairs, one-sided narrow-sector
        # hinge (no 120-deg target exists for four bonds)
        qc, qu, qw = [], [], []
        for a, nb in enumerate(nbrs):
            if len(nb) == 4:
                for u in range(4):
                    for w in range(u + 1, 4):
                        qc.append(a)
                        qu.append(nb[u])
                        qw.append(nb[w])
        self.q4_c = np.array(qc, dtype=np.intp)
        self.q4_u = np.array(qu, dtype=np.intp)
        self.q4_w = np.array(qw, dtype=np.intp)

        # stereocentre neighbour pairs: every unordered pair around an atom
        # with an assigned tetrahedral tag (depicted parity = signs of the
        # pair cross products; reference signs are taken per start)
        cc, cu, cw = [], [], []
        for a, nb in enumerate(nbrs):
            atom = mol.GetAtomWithIdx(a)
            if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            if len(nb) < 2:
                continue
            for u in range(len(nb)):
                for w in range(u + 1, len(nb)):
                    cc.append(a)
                    cu.append(nb[u])
                    cw.append(nb[w])
        self.chi_c = np.array(cc, dtype=np.intp)
        self.chi_u = np.array(cu, dtype=np.intp)
        self.chi_w = np.array(cw, dtype=np.intp)

        # macroring in bond order; alternating zig-zag sign per position
        try:
            ring = ordered_ring(mol, find_macroring(mol))
        except ValueError as exc:
            raise OptimizeLayoutError(f"no usable macroring: {exc}") from exc
        self.ring = np.array(ring, dtype=np.intp)
        n_ring = len(ring)
        ring_set = set(ring)
        self.ring_sign = np.where(np.arange(n_ring) % 2 == 0, 1.0, -1.0)

        # pendant side-chain subtree per ring atom (multi-source BFS; each
        # non-ring atom joins the first ring atom that discovers it)
        subtree_of: dict[int, int] = {}
        frontier: list[tuple[int, int]] = []
        for a in ring:
            for nb in nbrs[a]:
                if nb not in ring_set and nb not in subtree_of:
                    subtree_of[nb] = a
                    frontier.append((nb, a))
        while frontier:
            atom, root = frontier.pop()
            for nb in nbrs[atom]:
                if nb not in ring_set and nb not in subtree_of:
                    subtree_of[nb] = root
                    frontier.append((nb, root))
        sizes = np.zeros(n_ring)
        pos = {a: k for k, a in enumerate(ring)}
        for root in subtree_of.values():
            sizes[pos[root]] += 1
        self.subtree_size = sizes

        # per-atom radial targets: base radius times subtree share
        bond_length = standard_bond_length(self.n)
        self.bond_length = bond_length
        max_radius = n_ring * bond_length / (2.0 * math.pi)
        base_radius = max(max_radius, RING_RADIUS_MIN_BONDS * bond_length)
        self.base_radius = base_radius
        share = sizes / sizes.max() if sizes.max() > 0 else sizes
        self.ring_target = base_radius * (
            1.0 + SUBTREE_RADIUS_FRAC * share)

        # exocyclic branch pairs per attachment (target angle 2*pi*k/m)
        pairs, targets = [], []
        for a in ring:
            exo = [nb for nb in nbrs[a] if nb not in ring_set]
            m = len(exo)
            if m < 2:
                continue
            exo.sort()
            for u in range(m):
                for w in range(u + 1, m):
                    pairs.append((exo[u], exo[w], a))
                    targets.append(2.0 * math.pi * (w - u) / m)
        self.branch_pairs = (np.array([p[0] for p in pairs], dtype=np.intp),
                             np.array([p[1] for p in pairs], dtype=np.intp),
                             np.array([p[2] for p in pairs], dtype=np.intp))
        self.branch_pair_target = np.array(targets)

        # side-chain leaves for the outward hinge spring
        self.leaves = np.array(
            [a for a in range(self.n)
             if a not in ring_set and len(nbrs[a]) == 1], dtype=np.intp)

        # non-bonded pairs: graph distance >= 3
        dist = _graph_distance_matrix(nbrs, self.n)
        iu, ju = np.triu_indices(self.n, k=1)
        mask = dist[iu, ju] >= 3.0
        self.nb_i = iu[mask].astype(np.intp)
        self.nb_j = ju[mask].astype(np.intp)


def _graph_distance_matrix(nbrs: list[list[int]], n: int) -> np.ndarray:
    """All-pairs shortest path (hop count) of the molecular graph."""
    rows, cols = [], []
    for a, nb in enumerate(nbrs):
        rows.extend([a] * len(nb))
        cols.extend(nb)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(n, n))
    return shortest_path(graph, method="auto", unweighted=True)


def standard_bond_length(n_atoms: int) -> float:
    """Standard bond length L in layout units: BASE_BOND scaled up with
    molecule size so large macrorings fill proportionally more canvas."""
    scale = 1.0 + SPREAD_PER_ATOM * max(0, n_atoms - SPREAD_REF_ATOMS)
    return BASE_BOND * min(scale, SPREAD_MAX_SCALE)


def _energy_and_grad(x: np.ndarray, topo: _Topology, w: OptimizeWeights,
                     chi_sign: np.ndarray | None = None
                     ) -> tuple[float, np.ndarray]:
    """Energy and exact analytic gradient of the full layout objective.

    x is the flat 2n coordinate vector; chi_sign carries the reference
    cross-product signs at the stereocentre neighbour pairs (from the
    start being solved).  Every term below is vectorized and contributes
    closed-form derivatives via bincount scatter-adds.
    """
    n = topo.n
    p = x.reshape(n, 2)
    grad = np.zeros((n, 2))
    energy = 0.0

    def scatter(idx: np.ndarray, g: np.ndarray) -> None:
        grad[:, 0] += np.bincount(idx, weights=g[:, 0], minlength=n)
        grad[:, 1] += np.bincount(idx, weights=g[:, 1], minlength=n)

    # bond springs: w * sum (|d| - L)^2
    d = p[topo.bond_i] - p[topo.bond_j]
    r = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
    coeff = 2.0 * w.bond * (r - topo.bond_length) / r
    energy += w.bond * np.sum((r - topo.bond_length) ** 2)
    g = coeff[:, None] * d
    scatter(topo.bond_i, g)
    scatter(topo.bond_j, -g)

    # angle springs at two-/three-neighbour atoms: w * sum (cos - cos120)^2
    if topo.ang_c.size:
        du = p[topo.ang_u] - p[topo.ang_c]
        dw = p[topo.ang_w] - p[topo.ang_c]
        nu = np.sqrt(np.einsum("ij,ij->i", du, du)) + 1e-12
        nw = np.sqrt(np.einsum("ij,ij->i", dw, dw)) + 1e-12
        un = du / nu[:, None]
        wn = dw / nw[:, None]
        cos = np.einsum("ij,ij->i", un, wn)
        wt = w.angle * topo.ang_mult
        energy += float(np.sum(wt * (cos + 0.5) ** 2))
        coeff = (2.0 * wt * (cos + 0.5))[:, None]
        gu = coeff * (wn - cos[:, None] * un) / nu[:, None]
        gw = coeff * (un - cos[:, None] * wn) / nw[:, None]
        scatter(topo.ang_u, gu)
        scatter(topo.ang_w, gw)
        scatter(topo.ang_c, -(gu + gw))

    # quaternary centres: one-sided hinge banning narrow bond sectors
    # (w * sum max(0, cos - cos45)^2); keeps RDKit's wedge heuristic
    # unambiguous at four-bond stereocentres
    if topo.q4_c.size:
        du = p[topo.q4_u] - p[topo.q4_c]
        dw = p[topo.q4_w] - p[topo.q4_c]
        nu = np.sqrt(np.einsum("ij,ij->i", du, du)) + 1e-12
        nw = np.sqrt(np.einsum("ij,ij->i", dw, dw)) + 1e-12
        un = du / nu[:, None]
        wn = dw / nw[:, None]
        cos = np.einsum("ij,ij->i", un, wn)
        dev = np.maximum(0.0, cos - QUATERNARY_NARROW_COS)
        energy += w.angle * np.sum(dev ** 2)
        active = dev > 0.0
        coeff = (2.0 * w.angle * dev * active)[:, None]
        gu = coeff * (wn - cos[:, None] * un) / nu[:, None]
        gw = coeff * (un - cos[:, None] * wn) / nw[:, None]
        scatter(topo.q4_u, gu)
        scatter(topo.q4_w, gw)
        scatter(topo.q4_c, -(gu + gw))

    # ring spread + zig-zag: radial springs to per-atom targets, exact
    # gradient through the ring centroid (rank-1 correction)
    rc = p[topo.ring]
    centroid = rc.mean(axis=0)
    dv = rc - centroid
    rr = np.sqrt(np.einsum("ij,ij->i", dv, dv)) + 1e-12
    u = dv / rr[:, None]
    n_ring = len(topo.ring)
    g_spread = (2.0 * w.ring_spread * (rr - topo.ring_target))[:, None] * u
    zig_target = (topo.ring_target
                  + ZIGZAG_DELTA_FRAC * topo.bond_length * topo.ring_sign)
    g_zig = (2.0 * w.ring_zigzag * (rr - zig_target))[:, None] * u
    energy += (w.ring_spread * np.sum((rr - topo.ring_target) ** 2)
               + w.ring_zigzag * np.sum((rr - zig_target) ** 2))
    # exact centroid dependence: dE/dp_j = g_j - mean(g) over ring atoms
    g_ring = (g_spread + g_zig) - (g_spread + g_zig).mean(axis=0)

    # side-chain spread: branch angles + outward hinge on leaves
    bi, bj, ba = topo.branch_pairs
    if bi.size:
        du = p[bi] - p[ba]
        dw = p[bj] - p[ba]
        nu = np.sqrt(np.einsum("ij,ij->i", du, du)) + 1e-12
        nw = np.sqrt(np.einsum("ij,ij->i", dw, dw)) + 1e-12
        un = du / nu[:, None]
        wn = dw / nw[:, None]
        cos = np.einsum("ij,ij->i", un, wn)
        dev = cos - np.cos(topo.branch_pair_target)
        energy += w.side_chain * np.sum(dev ** 2)
        coeff = 2.0 * w.side_chain * dev
        gu = coeff[:, None] * (wn - cos[:, None] * un) / nu[:, None]
        gw = coeff[:, None] * (un - cos[:, None] * wn) / nw[:, None]
        scatter(bi, gu)
        scatter(bj, gw)
        scatter(ba, -(gu + gw))

    if topo.leaves.size:
        ldv = p[topo.leaves] - centroid
        lr = np.sqrt(np.einsum("ij,ij->i", ldv, ldv)) + 1e-12
        lu = ldv / lr[:, None]
        hinge = np.maximum(0.0, SIDE_OUTWARD_RADIUS * topo.base_radius - lr)
        active = hinge > 0.0
        energy += w.side_chain * np.sum(hinge ** 2)
        g_leaf = -(2.0 * w.side_chain * hinge * active)[:, None] * lu
        scatter(topo.leaves, g_leaf)
        # leaves' radius depends on the ring centroid -> ring-atom correction
        g_ring += (2.0 * w.side_chain / n_ring) * (
            (hinge * active)[:, None] * lu).sum(axis=0)

    grad[topo.ring] += g_ring

    # chirality barrier: hinge on each flipped neighbour-pair cross sign
    if chi_sign is not None and topo.chi_c.size:
        su = p[topo.chi_u] - p[topo.chi_c]
        sv = p[topo.chi_w] - p[topo.chi_c]
        cross = su[:, 0] * sv[:, 1] - su[:, 1] * sv[:, 0]
        flip = np.maximum(0.0, -chi_sign * cross)
        energy += w.chirality * np.sum(flip ** 2)
        active = flip > 0.0
        coeff = (-2.0 * w.chirality * flip * chi_sign * active)[:, None]
        # d cross / d u = (sv_y, -sv_x);  d cross / d v = (-su_y, su_x)
        gu = coeff * np.stack([sv[:, 1], -sv[:, 0]], axis=1)
        gw = coeff * np.stack([-su[:, 1], su[:, 0]], axis=1)
        scatter(topo.chi_u, gu)
        scatter(topo.chi_w, gw)
        scatter(topo.chi_c, -(gu + gw))

    # non-bonded repulsion: w * sum 1/sqrt(|d|^2 + soft^2)
    nd = p[topo.nb_i] - p[topo.nb_j]
    soft2 = REPULSION_SOFTENING ** 2
    d2 = np.einsum("ij,ij->i", nd, nd) + soft2
    energy += w.repulsion * np.sum(d2 ** -0.5)
    g = (-w.repulsion * d2 ** -1.5)[:, None] * nd
    scatter(topo.nb_i, g)
    scatter(topo.nb_j, -g)

    return float(energy), grad.ravel()


def _coords_array(mol: Chem.Mol) -> np.ndarray:
    """Flat 2n vector of the first conformer."""
    conf = mol.GetConformer()
    x = np.zeros(2 * mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        x[2 * i] = pos.x
        x[2 * i + 1] = pos.y
    return x


def _median_bond(x: np.ndarray, bond_i: np.ndarray, bond_j: np.ndarray
                 ) -> float:
    p = x.reshape(-1, 2)
    d = p[bond_i] - p[bond_j]
    lengths = np.sqrt(np.einsum("ij,ij->i", d, d))
    median = float(np.median(lengths))
    return median if median > 1e-6 else 1.0


def _scale_and_center(x: np.ndarray, bond_i: np.ndarray,
                      bond_j: np.ndarray) -> np.ndarray:
    """Center at the origin and scale to unit median bond length so all
    starts live in the same coordinate frame as the energy terms."""
    p = x.reshape(-1, 2) - x.reshape(-1, 2).mean(axis=0)
    unit = _median_bond(p.ravel(), bond_i, bond_j)
    return (p / unit).ravel()


def _chiral_reference_signs(x: np.ndarray, topo: _Topology) -> np.ndarray:
    """Cross-product signs of the stereocentre neighbour pairs at a start
    layout — the depicted parity the solve must preserve."""
    if topo.chi_c.size == 0:
        return np.zeros(0)
    p = x.reshape(-1, 2)
    su = p[topo.chi_u] - p[topo.chi_c]
    sv = p[topo.chi_w] - p[topo.chi_c]
    return np.sign(su[:, 0] * sv[:, 1] - su[:, 1] * sv[:, 0])


def _coordgen_start(mol: Chem.Mol) -> np.ndarray:
    """CoordGen depiction of the input molecule as a flat start vector.

    Atom order is preserved (no depiction shuffling — that is the render
    pipeline's concern, not the optimizer's)."""
    from rdkit.Chem import rdDepictor
    work = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    try:
        rdDepictor.Compute2DCoords(work, clearConfs=True)
    finally:
        rdDepictor.SetPreferCoordGen(False)
    return _coords_array(work)


def _cdk_start(smiles: str, ref_mol: Chem.Mol) -> np.ndarray:
    """CDK StructureDiagramGenerator coordinates mapped back to the input
    atom order via canonical ranks (verified: both graphs must canonically
    agree or the start is refused)."""
    from render_via_engine import _load_engine_coords
    cdk_mol = _load_engine_coords(smiles, "cdk")
    ref_canon = Chem.MolToSmiles(Chem.Mol(ref_mol))
    cdk_canon = Chem.MolToSmiles(Chem.Mol(cdk_mol))
    if ref_canon != cdk_canon:
        raise OptimizeLayoutError(
            "CDK output graph differs from input graph (canonical SMILES "
            "mismatch); CDK start refused")
    ref_rank = list(Chem.CanonicalRankAtoms(
        ref_mol, breakTies=True, includeChirality=True))
    cdk_rank = list(Chem.CanonicalRankAtoms(
        cdk_mol, breakTies=True, includeChirality=True))
    by_rank = {rank: i for i, rank in enumerate(cdk_rank)}
    perm = np.array([by_rank[rank] for rank in ref_rank], dtype=np.intp)
    if len(set(perm.tolist())) != cdk_mol.GetNumAtoms():
        raise OptimizeLayoutError(
            "canonical-rank atom mapping is not a bijection; CDK start "
            "refused")
    x = _coords_array(cdk_mol).reshape(-1, 2)[perm]
    return x.ravel()


def _perturbed_starts(base: np.ndarray, bond_length: float,
                      seed: int) -> list[np.ndarray]:
    """Three seeded Gaussian perturbations of a base start (sigma fractions
    of L, each with a random rotation) to escape shallow minima."""
    rng = np.random.default_rng(seed)
    starts = []
    n = len(base) // 2
    for sigma_frac in (0.25, 0.50, 0.75):
        p = base.reshape(n, 2).copy()
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        p = p @ rot.T
        p = p + rng.normal(0.0, sigma_frac * bond_length, p.shape)
        starts.append(p.ravel())
    return starts


def _solve_start(topo: _Topology, weights: OptimizeWeights, x0: np.ndarray,
                 maxiter: int, ftol: float = LBFGS_FINAL_FTOL,
                 chi_sign: np.ndarray | None = None) -> SciOptimizeResult:
    return minimize(
        _energy_and_grad, x0, args=(topo, weights, chi_sign), jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                 "ftol": ftol, "gtol": LBFGS_GTOL})


def collision_conflicts(x: np.ndarray, topo: _Topology) -> np.ndarray:
    """Non-bond conflict pairs under the acceptance gate: graph distance
    >= 3 with euclidean distance < CONFLICT_DISTANCE_FRAC * L.  `x` must
    be in topology units (median bond at L).  Returns a (k, 2) index array
    (empty when the layout is conflict-free)."""
    p = x.reshape(topo.n, 2)
    d = p[topo.nb_i] - p[topo.nb_j]
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    mask = dist < CONFLICT_DISTANCE_FRAC * topo.bond_length
    return np.stack([topo.nb_i[mask], topo.nb_j[mask]], axis=1)


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                    p4: np.ndarray) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return False
    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    u = (dp[0] * d1[1] - dp[1] * d1[0]) / denom
    return 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9


def bond_crossings(x: np.ndarray, topo: _Topology) -> list[tuple[int, int, int, int]]:
    """Proper intersections between bond segments that share no endpoint,
    as (i1, j1, i2, j2) tuples.  A drawn crossing survives every atom-atom
    distance gate, so it is checked (and repaired) separately."""
    p = x.reshape(topo.n, 2)
    bi, bj = topo.bond_i, topo.bond_j
    out: list[tuple[int, int, int, int]] = []
    for a in range(len(bi)):
        for b in range(a + 1, len(bi)):
            i1, j1 = int(bi[a]), int(bj[a])
            i2, j2 = int(bi[b]), int(bj[b])
            if i1 in (i2, j2) or j1 in (i2, j2):
                continue
            if _segments_cross(p[i1], p[j1], p[i2], p[j2]):
                out.append((i1, j1, i2, j2))
    return out


def _circular_gaps_deg(angles: list[float]) -> list[float]:
    """Gaps between consecutive angles on the circle, in degrees."""
    if len(angles) < 2:
        return []
    ordered = np.sort(np.array(angles))
    gaps = np.diff(ordered)
    wrap = 2.0 * np.pi - (ordered[-1] - ordered[0])
    return [math.degrees(g) for g in np.append(gaps, wrap)]


def _subtree_members(topo: _Topology, ring_set: set[int]
                     ) -> dict[int, set[int]]:
    """Ring-atom ring position -> set of pendant side-chain atom indices.

    Same multi-source BFS the topology counts derive from; the metrics
    path needs the membership, not just sizes."""
    nbrs: dict[int, list[int]] = {i: [] for i in range(topo.n)}
    for i, j in zip(topo.bond_i, topo.bond_j):
        nbrs[int(i)].append(int(j))
        nbrs[int(j)].append(int(i))
    member_map: dict[int, set[int]] = {}
    owner: dict[int, int] = {}
    frontier: list[tuple[int, int]] = []
    for k, a in enumerate(topo.ring.tolist()):
        for nb in nbrs[a]:
            if nb not in ring_set and nb not in owner:
                owner[nb] = k
                member_map.setdefault(k, set()).add(nb)
                frontier.append((nb, k))
    while frontier:
        atom, k = frontier.pop()
        for nb in nbrs[atom]:
            if nb not in ring_set and nb not in owner:
                owner[nb] = k
                member_map.setdefault(k, set()).add(nb)
                frontier.append((nb, k))
    return member_map


def layout_metrics(x: np.ndarray, topo: _Topology) -> LayoutMetrics:
    """Shape statistics of a layout in topology units (median bond at L):
    ring radius coverage, angular gaps between side-chain subtree
    centroids and between their atoms, minimum angle between branches
    sharing an attachment, and the gate conflict count."""
    p = x.reshape(topo.n, 2)
    centroid = p[topo.ring].mean(axis=0)
    rr = np.linalg.norm(p[topo.ring] - centroid, axis=1)
    coverage = float(rr.max() / max(rr.min(),
                                    COVERAGE_FLOOR_FRAC * rr.mean()))
    zigzag = float(np.mean(np.abs(rr - np.median(rr)))) / topo.bond_length

    ring_set = set(topo.ring.tolist())
    member_map = _subtree_members(topo, ring_set)
    centroid_angles: list[float] = []
    atom_angles: dict[int, list[float]] = {}
    for k, members in member_map.items():
        idx = np.array(sorted(members), dtype=np.intp)
        vec = p[idx].mean(axis=0) - centroid
        centroid_angles.append(math.atan2(vec[1], vec[0]))
        rel = p[idx] - centroid
        atom_angles[k] = [math.atan2(v[1], v[0]) for v in rel]
    gaps_deg = _circular_gaps_deg(centroid_angles)

    # atom-level cross-subtree angular separation: minimum circular angle
    # between any atom of one subtree and any atom of the subtree adjacent
    # to it (chains folded back onto a neighbour collapse this to ~0)
    side_atom_min = 180.0
    if len(atom_angles) >= 2:
        order = np.argsort(np.array(centroid_angles))
        keys = list(atom_angles.keys())
        for pos in range(len(order)):
            k_a = keys[int(order[pos])]
            k_b = keys[int(order[(pos + 1) % len(order)])]
            for a in atom_angles[k_a]:
                for b in atom_angles[k_b]:
                    delta = abs(a - b) % (2.0 * math.pi)
                    delta = min(delta, 2.0 * math.pi - delta)
                    side_atom_min = min(side_atom_min,
                                        math.degrees(delta))

    branch_min = 180.0
    bi, bj, ba = topo.branch_pairs
    if bi.size:
        v1 = p[bi] - p[ba]
        v2 = p[bj] - p[ba]
        cos = np.einsum("ij,ij->i", v1, v2) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
        branch_min = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).min())

    return LayoutMetrics(
        radius_coverage=coverage,
        ring_mean_radius_over_bond=float(rr.mean() / topo.bond_length),
        zigzag_amplitude=zigzag,
        subtree_min_gap_deg=float(min(gaps_deg)) if gaps_deg else 0.0,
        subtree_mean_gap_deg=float(np.mean(gaps_deg)) if gaps_deg else 0.0,
        side_atom_min_gap_deg=float(side_atom_min),
        branch_min_angle_deg=branch_min,
        n_conflicts=int(len(collision_conflicts(x, topo))),
    )


def _write_coords(mol: Chem.Mol, x: np.ndarray, median_bond: float) -> Chem.Mol:
    """Copy of mol carrying x as its (only) conformer, rescaled so the
    median bond length lands on OUTPUT_MEDIAN_BOND.  The conformer is
    flagged 2-D: a 3-D flag on planar coordinates silently misroutes
    RDKit's chiral-H placement and wedge logic (their 3-D branches
    degenerate on flat spiro stereocentres and the rendered wedges lose
    their red-box validity checks)."""
    from rdkit.Geometry import Point3D
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    conf.Set3D(False)
    p = x.reshape(-1, 2)
    scale = OUTPUT_MEDIAN_BOND / median_bond
    p = p - p.mean(axis=0)
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(p[i, 0] * scale),
                                        float(p[i, 1] * scale), 0.0))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def optimize_layout(mol: Chem.Mol,
                    weights: OptimizeWeights = DEFAULT_WEIGHTS,
                    seed: int = 0) -> OptimizeResult:
    """Optimize a molecule's 2D layout toward the patent macroring style.

    Multi-start (CoordGen, CDK, three perturbations) L-BFGS-B on the full
    energy, ranked coarsely then the winning basin solved to convergence;
    the collision gate escalates the repulsion weight and re-solves once
    if conflicts remain.  Raises OptimizeLayoutError when the molecule has
    no usable macroring.
    """
    started = time.perf_counter()
    smiles = Chem.MolToSmiles(mol)
    topo = _Topology(mol)

    starts: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    cg = _scale_and_center(_coordgen_start(mol), topo.bond_i, topo.bond_j)
    starts["coordgen"] = cg
    try:
        starts["cdk"] = _scale_and_center(
            _cdk_start(smiles, mol), topo.bond_i, topo.bond_j)
    except (OptimizeLayoutError, EngineRenderError) as exc:
        skipped["cdk"] = f"{type(exc).__name__}: {exc}"
    for k, x in enumerate(_perturbed_starts(cg, topo.bond_length, seed)):
        starts[f"perturb_{k + 1}"] = x

    # phase 1: rank every start at coarse tolerance; perturbation starts
    # get a hard iteration cap (they only need a comparable energy).  Each
    # solve preserves its own start's depicted stereocentre parity.
    ranked: list[tuple[str, np.ndarray, float, int, np.ndarray]] = []
    for name, x0 in starts.items():
        chi = _chiral_reference_signs(x0, topo)
        if name.startswith("perturb"):
            res = _solve_start(topo, weights, x0, LBFGS_PERTURB_MAXITER,
                               LBFGS_PERTURB_FTOL, chi)
        else:
            res = _solve_start(topo, weights, x0, LBFGS_MAXITER,
                               LBFGS_RANK_FTOL, chi)
        ranked.append((name, res.x.copy(), float(res.fun), int(res.nit), chi))
    ranked.sort(key=lambda item: item[2])

    # phase 2: the winning basin is solved to full convergence
    best_name, best_x, best_e, best_it, best_chi = ranked[0]
    res = _solve_start(topo, weights, best_x, LBFGS_MAXITER, LBFGS_FINAL_FTOL,
                       best_chi)
    best_x, best_e = res.x.copy(), float(res.fun)
    best_it += int(res.nit)

    conflicts = collision_conflicts(best_x, topo)
    reoptimized = False
    if len(conflicts) > 0:
        escalated = OptimizeWeights(
            bond=weights.bond, angle=weights.angle,
            ring_spread=weights.ring_spread, ring_zigzag=weights.ring_zigzag,
            side_chain=weights.side_chain, chirality=weights.chirality,
            repulsion=weights.repulsion * REPULSION_ESCALATION)
        res = _solve_start(topo, escalated, best_x, LBFGS_MAXITER,
                           LBFGS_FINAL_FTOL, best_chi)
        retry = collision_conflicts(res.x, topo)
        if len(retry) <= len(conflicts):
            best_x, best_e = res.x.copy(), float(res.fun)
            conflicts = retry
            reoptimized = True
            best_it += int(res.nit)

    # bond-crossing repair: the crossing bonds' atom pairs join the
    # repulsion set for a short local re-solve; accepted only when the
    # crossing count drops and no conflict reappears
    crossings = bond_crossings(best_x, topo)
    crossing_repaired = False
    escalated = OptimizeWeights(
        bond=weights.bond, angle=weights.angle,
        ring_spread=weights.ring_spread, ring_zigzag=weights.ring_zigzag,
        side_chain=weights.side_chain, chirality=weights.chirality,
        repulsion=weights.repulsion * CROSSING_ESCALATION)
    extra_i: list[np.ndarray] = []
    extra_j: list[np.ndarray] = []
    for _ in range(CROSSING_REPAIR_ROUNDS):
        if not crossings:
            break
        for c in crossings:
            for u, w in ((c[0], c[2]), (c[0], c[3]), (c[1], c[2]),
                         (c[1], c[3])):
                extra_i.append(np.array([u], dtype=np.intp))
                extra_j.append(np.array([w], dtype=np.intp))
        repair = copy.copy(topo)
        repair.nb_i = np.concatenate([topo.nb_i] + extra_i)
        repair.nb_j = np.concatenate([topo.nb_j] + extra_j)
        res = _solve_start(repair, escalated, best_x,
                           LBFGS_CROSSING_MAXITER, LBFGS_FINAL_FTOL, best_chi)
        new_cross = bond_crossings(res.x, repair)
        new_conf = collision_conflicts(res.x, repair)
        if len(new_cross) < len(crossings) and len(new_conf) == 0:
            best_x, best_e = res.x.copy(), float(res.fun)
            conflicts, crossings = new_conf, new_cross
            crossing_repaired = True
            best_it += int(res.nit)
        else:
            break

    out_mol = _write_coords(mol, best_x, topo.bond_length)
    return OptimizeResult(
        smiles=smiles,
        mol=out_mol,
        energy=best_e,
        seconds=time.perf_counter() - started,
        start_used=best_name,
        start_energies={name: float(_energy_and_grad(
            x, topo, weights, _chiral_reference_signs(x, topo))[0])
            for name, x in starts.items()},
        starts_skipped=skipped,
        n_conflicts_before=int(len(collision_conflicts(
            cg * topo.bond_length, topo))),
        n_conflicts_after=int(len(conflicts)),
        gate_pass=len(conflicts) == 0,
        reoptimized=reoptimized,
        n_crossings=len(bond_crossings(best_x, topo)),
        crossing_repaired=crossing_repaired,
        n_iterations=best_it,
    )


def optimize_smiles(smiles: str,
                    weights: OptimizeWeights = DEFAULT_WEIGHTS,
                    seed: int = 0) -> OptimizeResult:
    """Parse SMILES and run `optimize_layout`; parse failures raise
    OptimizeLayoutError."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise OptimizeLayoutError(f"RDKit could not parse SMILES: "
                                  f"{smiles[:80]}...")
    return optimize_layout(mol, weights, seed)


# ---------------------------------------------------------------------------
# rendering and validation suite
# ---------------------------------------------------------------------------

SEQ53_GOLD_KEY = "3"   # US20240287138A1 drawn "SEQ ID NO: 53" gold variant
GOLD_JSON = "/data/BioChemInsight/data/gold_smiles_289.json"
KRAS_MANIFEST = "/tmp/crops_kras/real_crops_manifest.json"
GPC3_MANIFEST = "/tmp/crops_gpc3/real_crops_manifest.json"
US_MANIFEST = ("/data/BioChemInsight/data/real_crops/"
               "real_crops_manifest.json")
COMPARE_SHEET_PATH = "/tmp/optimize_compare_seq53.png"


def _gold_smiles(key: str) -> str:
    gold = json.load(open(GOLD_JSON, encoding="utf-8"))
    if key not in gold:
        raise OptimizeLayoutError(f"gold key {key!r} missing from {GOLD_JSON}")
    return gold[key]["smiles"]


def _verified_smiles_for_seq(seq_id: int) -> str:
    """First verified SMILES drawn under a patent SEQ ID NO: label."""
    manifest = json.load(open(US_MANIFEST, encoding="utf-8"))
    hits = [r for r in manifest["rows"]
            if r.get("seq_id") == seq_id and r.get("verified") is True
            and r.get("smiles")]
    if not hits:
        raise OptimizeLayoutError(f"no verified SMILES for seq_id {seq_id}")
    return hits[0]["smiles"]


def _verified_smiles_from(manifest_path: str, count: int,
                          label: str) -> list[tuple[str, str]]:
    """Distinct verified SMILES from a crop manifest, (id, smiles) pairs."""
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    seen: set[str] = set()
    picked: list[tuple[str, str]] = []
    for row in manifest["rows"]:
        smi = row.get("smiles")
        if row.get("verified") is not True or not smi or smi in seen:
            continue
        seen.add(smi)
        picked.append((f"{label}_{len(picked) + 1}", smi))
        if len(picked) == count:
            break
    if len(picked) < count:
        raise OptimizeLayoutError(
            f"{manifest_path}: only {len(picked)} verified SMILES, "
            f"{count} requested")
    return picked


def validation_molecules() -> list[tuple[str, str, str]]:
    """The acceptance suite: seq53 gold + verified seq variants
    (seq53/60/68/77) plus two KRAS and two GPC3 verified SMILES — ten
    molecules, (id, smiles, source) tuples."""
    cases: list[tuple[str, str, str]] = [
        ("seq53_gold", _gold_smiles(SEQ53_GOLD_KEY),
         "gold_smiles_289.json key 3 (156 atoms)"),
        ("seq53_crop", _verified_smiles_for_seq(53),
         "verified crop US20240287138A1 p.24"),
        ("seq60_crop", _verified_smiles_for_seq(60),
         "verified crop WO2024163691A1 p.305"),
        ("seq68_gold", _gold_smiles("68"),
         "gold_smiles_289.json key 68 (150 atoms)"),
        ("seq68_crop", _verified_smiles_for_seq(68),
         "verified crop US20240287138A1 p.24"),
        ("seq77_crop", _verified_smiles_for_seq(77),
         "verified crop US20240287138A1 p.25"),
    ]
    cases += [(mid, smi, "KRAS verified manifest")
              for mid, smi in _verified_smiles_from(KRAS_MANIFEST, 2, "kras")]
    cases += [(mid, smi, "GPC3 verified manifest")
              for mid, smi in _verified_smiles_from(GPC3_MANIFEST, 2, "gpc3")]
    return cases


def _mol_with_coordgen(mol: Chem.Mol) -> Chem.Mol:
    """Copy of mol carrying fresh CoordGen coordinates (the before-layout
    baseline of every comparison render)."""
    from rdkit.Chem import rdDepictor
    work = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    try:
        rdDepictor.Compute2DCoords(work, clearConfs=True)
    finally:
        rdDepictor.SetPreferCoordGen(False)
    return work


def _render_pair(mol_in: Chem.Mol, mol_out: Chem.Mol, seed: int
                 ) -> tuple[object, object]:
    """ALIGNED-style renders of the input and optimized layouts."""
    rng_in = random.Random(seed)
    rng_out = random.Random(seed)
    return (draw_molecule(mol_in, ALIGNED, rng_in),
            draw_molecule(mol_out, ALIGNED, rng_out))


def render_compare_sheet(mol_in: Chem.Mol, mol_out: Chem.Mol, title: str,
                         path: str, seed: int = 0) -> None:
    """Side-by-side CoordGen-vs-optimized comparison PNG with a sheet
    title and one label above each panel."""
    from PIL import Image, ImageDraw, ImageFont
    img_in, img_out = _render_pair(mol_in, mol_out, seed)
    label_font = ImageFont.truetype(ALIGNED["font_file"], 26)
    pad, title_h = 24, 40
    height = max(img_in.height, img_out.height) + title_h + 3 * pad
    width = img_in.width + img_out.width + 3 * pad
    sheet = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, font=label_font, fill=0)
    panel_top = title_h + pad
    sheet.paste(img_in, (pad, panel_top))
    sheet.paste(img_out, (img_in.width + 2 * pad, panel_top))
    in_mid = pad + img_in.width // 2
    out_mid = img_in.width + 2 * pad + img_out.width // 2
    draw.text((in_mid - 60, panel_top - 34), "CoordGen input",
              font=label_font, fill=0)
    draw.text((out_mid - 40, panel_top - 34), "optimized",
              font=label_font, fill=0)
    sheet.save(path)


def run_validation(out_dir: str = "/tmp/macrocycle_optimize_validation",
                   seed: int = 0) -> int:
    """Full acceptance run: per-molecule before/after PNGs, the seq53
    comparison sheet at COMPARE_SHEET_PATH, and a metrics JSON.  Returns 0
    when every molecule passes the collision gate."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_rows = []
    all_pass = True
    for cid, smiles, source in validation_molecules():
        result = optimize_smiles(smiles, seed=seed)
        mol_in = _mol_with_coordgen(Chem.MolFromSmiles(smiles))
        assert mol_in is not None

        topo = _Topology(mol_in)
        before = layout_metrics(
            _scale_and_center(_coords_array(mol_in),
                              topo.bond_i, topo.bond_j) * topo.bond_length,
            topo)
        conf = result.mol.GetConformer()
        x_out = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y]
                          for i in range(result.mol.GetNumAtoms())]).ravel()
        after = layout_metrics(
            _scale_and_center(x_out, topo.bond_i, topo.bond_j)
            * topo.bond_length, topo)

        img_in, img_out = _render_pair(mol_in, result.mol, seed)
        img_in.save(str(out / f"{cid}_coordgen.png"))
        img_out.save(str(out / f"{cid}_optimized.png"))
        if cid == "seq53_gold":
            render_compare_sheet(mol_in, result.mol,
                                 "seq53 gold US20240287138A1 (156 atoms)",
                                 COMPARE_SHEET_PATH, seed)

        all_pass &= result.gate_pass
        report_rows.append({
            "id": cid,
            "source": source,
            "n_atoms": result.mol.GetNumAtoms(),
            "energy": round(result.energy, 4),
            "seconds": round(result.seconds, 3),
            "start_used": result.start_used,
            "start_energies": {k: round(v, 2)
                               for k, v in result.start_energies.items()},
            "starts_skipped": result.starts_skipped,
            "reoptimized": result.reoptimized,
            "conflicts_before": result.n_conflicts_before,
            "conflicts_after": result.n_conflicts_after,
            "gate_pass": result.gate_pass,
            "n_crossings": result.n_crossings,
            "crossing_repaired": result.crossing_repaired,
            "radius_coverage_before": round(before.radius_coverage, 3),
            "radius_coverage_after": round(after.radius_coverage, 3),
            "ring_mean_radius_before": round(before.ring_mean_radius_over_bond, 3),
            "ring_mean_radius_after": round(after.ring_mean_radius_over_bond, 3),
            "zigzag_amplitude_before": round(before.zigzag_amplitude, 3),
            "zigzag_amplitude_after": round(after.zigzag_amplitude, 3),
            "subtree_min_gap_deg_before": round(before.subtree_min_gap_deg, 1),
            "subtree_min_gap_deg_after": round(after.subtree_min_gap_deg, 1),
            "subtree_mean_gap_deg_before": round(before.subtree_mean_gap_deg, 1),
            "subtree_mean_gap_deg_after": round(after.subtree_mean_gap_deg, 1),
            "side_atom_min_gap_deg_before": round(before.side_atom_min_gap_deg, 1),
            "side_atom_min_gap_deg_after": round(after.side_atom_min_gap_deg, 1),
            "branch_min_angle_deg_before": round(before.branch_min_angle_deg, 1),
            "branch_min_angle_deg_after": round(after.branch_min_angle_deg, 1),
            "png_coordgen": str(out / f"{cid}_coordgen.png"),
            "png_optimized": str(out / f"{cid}_optimized.png"),
        })
        row = report_rows[-1]
        status = "PASS" if result.gate_pass else "FAIL"
        print(f"[{status}] {cid}: atoms={row['n_atoms']} "
              f"conflicts {row['conflicts_before']} -> "
              f"{row['conflicts_after']} "
              f"coverage {before.radius_coverage:.2f} -> "
              f"{after.radius_coverage:.2f} "
              f"ring_r/L {before.ring_mean_radius_over_bond:.2f} -> "
              f"{after.ring_mean_radius_over_bond:.2f} "
              f"gap {before.subtree_min_gap_deg:.1f} -> "
              f"{after.subtree_min_gap_deg:.1f} deg "
              f"crossings={result.n_crossings} "
              f"({result.seconds:.2f}s, start={result.start_used})",
              flush=True)

    metrics_path = out / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump({"all_gate_pass": bool(all_pass),
                   "n_molecules": len(report_rows),
                   "rows": report_rows}, fh, indent=2)
    print(f"metrics: {metrics_path}")
    print(f"seq53 comparison sheet: {COMPARE_SHEET_PATH}")
    return 0 if all_pass else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: no arguments runs the full validation suite; --smiles optimizes
    one molecule and writes a comparison sheet."""
    ap = argparse.ArgumentParser(
        description="Optimize CoordGen macroring layouts into the patent "
                    "zig-zag style (multi-start L-BFGS-B, analytic "
                    "gradients, collision gate).")
    ap.add_argument("--smiles", help="single molecule mode: SMILES string")
    ap.add_argument("--out-png", default=None,
                    help="comparison sheet path for --smiles mode")
    ap.add_argument("--out-dir", default="/tmp/macrocycle_optimize_validation",
                    help="validation-suite output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        if args.smiles:
            result = optimize_smiles(args.smiles, seed=args.seed)
            mol_in = _mol_with_coordgen(Chem.MolFromSmiles(args.smiles))
            sheet = args.out_png or "/tmp/optimize_compare_smiles.png"
            render_compare_sheet(mol_in, result.mol, "input", sheet,
                                 args.seed)
            print(f"energy={result.energy:.3f} conflicts="
                  f"{result.n_conflicts_after} gate_pass={result.gate_pass} "
                  f"({result.seconds:.2f}s) sheet={sheet}")
            return 0
        return run_validation(args.out_dir, args.seed)
    except OptimizeLayoutError as exc:
        print(f"macrocycle_optimize: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

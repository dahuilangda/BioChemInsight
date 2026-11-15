"""Side-chain overlap resolution for CoordGen 2D layouts.

CoordGen folds the side chains of the gold macrolide set onto each other
and onto the macroring; the rendered crops show crossed and overlapped
skeletons (measured: 37 of the 100 gold SMILES carry side-chain atom pairs
closer than 0.6x the standard bond length). This module post-processes an
existing conformer in four steps:

  1. detect    — every non-bonded atom pair closer than CONFLICT_RATIO_DEFAULT
                 x the layout's standard bond length (median bonded distance
                 of the conformer, the depiction's own scale) is a conflict.
  2. group     — conflicts are attributed to side-chain subtrees. The
                 scaffold rings (macroring backbones, SCAFFOLD_RING_MIN_SIZE
                 atoms and up) never move; a subtree is the BFS from the
                 non-scaffold end of a scaffold exit bond, blocked at
                 scaffold atoms, so a side chain travels as one rigid body
                 including its own aromatic rings (Trp indole, Tyr phenol,
                 ...). A subtree is swingable only when every one of its
                 atoms bonds outside the subtree to the pivot alone —
                 side chains fused/bridged to the scaffold at a second atom
                 (proline-type residue rings) would deform under a swing
                 and stay rigid. Pairs the model cannot move (scaffold
                 collisions, intra-subtree folds, bridged rings) are
                 report-only.
  3. resolve   — each conflicting swingable subtree, most severe first, is
                 swung rigidly around its pivot in ROTATION_STEP_DEG steps
                 over 360 deg. Candidate angles are ranked by the number of
                 clashes the swing leaves between the subtree and the rest
                 of the molecule, then by the sum over subtree atoms of
                 their minimum distance to every non-subtree atom (the
                 separation score), with ties keeping the angle nearest the
                 original pose. The clash term guards the scan — the raw
                 separation sum alone drives a subtree through a
                 neighbour's territory for a higher average clearance.
  4. iterate   — a swing can open a new collision elsewhere, so the
                 detect/resolve pass repeats up to MAX_ROUNDS_DEFAULT times.
                 Conflicts that survive all rounds raise
                 UnresolvedSidechainConflictsError (never silently accepted);
                 the exception carries the report and the best-effort
                 resolved molecule.

resolve_sidechain_overlaps() is the single resolution entry point. The CLI
renders CoordGen-initial vs resolved PNG pairs in the ALIGNED corpus style
(same seed, same orientation) for visual QA:

    python sidechain_resolve.py --smiles "..." --out-dir out/
    python sidechain_resolve.py --smiles-file names.smi --out-dir out/

The .smi file takes one molecule per line as "name<TAB>smiles" (or a bare
SMILES). Exit codes: 0 all clean, 1 hard layout failure, 2 at least one
molecule kept unresolved conflicts.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style_calibration import ALIGNED, coordgen_coords, draw_molecule  # noqa: E402

CONFLICT_RATIO_DEFAULT = 0.6   # conflict threshold as a fraction of the
                               # standard bond length
ROTATION_STEP_DEG = 15.0
MAX_ROUNDS_DEFAULT = 3
# scaffold vs side-chain ring split: side-chain rings in the patent-peptide
# corpus are aromatic/hetero cycles of at most 9 SSSR atoms (indole = 5+6),
# while every backbone macroring is 12 atoms and up — so a ring at this size
# or above is scaffold whose atoms anchor side chains and never swing
SCAFFOLD_RING_MIN_SIZE = 12
_TIE_EPS = 1e-9


class SidechainResolveError(Exception):
    """Base class: the side-chain overlap resolution path failed."""


class SidechainLayoutError(SidechainResolveError):
    """The input molecule has no usable conformer to screen or resolve."""


class UnresolvedSidechainConflictsError(SidechainResolveError):
    """Conflicts survived the full resolution loop.

    Carries `.report` (named unresolved list in `report.unresolved`) and
    `.mol`, the best-effort resolved molecule, so callers can still render
    or record the outcome instead of dropping it silently.
    """

    def __init__(self, report: "SidechainResolveReport"):
        shown = report.unresolved[:8]
        detail = "; ".join(
            f"{p.label} at {p.bond_length_ratio:.2f}x bond length"
            for p in shown)
        more = f" (+{len(report.unresolved) - len(shown)} more)" \
            if len(report.unresolved) > len(shown) else ""
        super().__init__(
            f"{report.conflicts_after} side-chain conflict(s) unresolved "
            f"after {report.rounds_used} round(s): {detail}{more}")
        self.report = report
        self.mol = report.molecule


@dataclass(frozen=True)
class ConflictPair:
    """A non-bonded atom pair rendered too close together."""
    atom_a: int
    atom_b: int
    symbol_a: str
    symbol_b: str
    distance: float            # layout units, same scale as the conformer
    shortfall: float           # standard_bond_length - distance
    bond_length_ratio: float   # distance / standard_bond_length

    @property
    def label(self) -> str:
        return f"{self.symbol_a}{self.atom_a}/{self.symbol_b}{self.atom_b}"


@dataclass
class SidechainSubtree:
    """One side chain as a rigid body: atoms swinging around the scaffold
    atom the exit bond leaves from."""
    pivot: int                 # scaffold atom the exit bond leaves from
    root: int                  # first atom off the scaffold
    atoms: tuple[int, ...]     # BFS order, root first


@dataclass
class SidechainResolveReport:
    """Outcome of one resolution run; mol carries the final coordinates."""
    molecule: Chem.Mol
    conflicts_before: int
    conflicts_after: int
    rounds_used: int
    rotations_applied: int
    standard_bond_length: float
    unresolved: list[ConflictPair] = field(default_factory=list)
    swings: list[tuple[int, int, float]] = field(default_factory=list)
    # swings: (pivot, root, applied angle in degrees)


def _conformer_xy(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                     for i in range(mol.GetNumAtoms())], dtype=np.float64)


def _standard_bond_length(mol: Chem.Mol, coords: np.ndarray) -> float:
    """Median bonded distance — the depiction's own length scale."""
    lengths = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lengths.append(float(np.linalg.norm(coords[i] - coords[j])))
    return float(np.median(lengths)) if lengths else 0.0


def conflict_pairs(mol: Chem.Mol,
                   ratio: float = CONFLICT_RATIO_DEFAULT) -> list[ConflictPair]:
    """Screen the conformer for non-bonded pairs closer than `ratio` x the
    standard bond length, most severe (largest shortfall) first."""
    if mol.GetNumConformers() == 0:
        raise SidechainLayoutError(
            "conflict screening needs an existing conformer")
    n = mol.GetNumAtoms()
    coords = _conformer_xy(mol)
    std = _standard_bond_length(mol, coords)
    if std <= 0.0:
        return []

    bonded = np.eye(n, dtype=bool)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    upper = np.triu_indices(n, k=1)
    close = (~bonded[upper]) & (dist[upper] < ratio * std)
    out = []
    for i, j in zip(upper[0][close], upper[1][close]):
        i, j = int(i), int(j)
        out.append(ConflictPair(
            atom_a=i, atom_b=j,
            symbol_a=mol.GetAtomWithIdx(i).GetSymbol(),
            symbol_b=mol.GetAtomWithIdx(j).GetSymbol(),
            distance=float(dist[i, j]),
            shortfall=float(std - dist[i, j]),
            bond_length_ratio=float(dist[i, j] / std)))
    out.sort(key=lambda p: p.shortfall, reverse=True)
    return out


def _scaffold_atoms(mol: Chem.Mol) -> set[int]:
    """Atoms of the scaffold (macroring) rings; the pivot set side chains
    hang from and the atoms a subtree BFS may never cross."""
    return {a for ring in mol.GetRingInfo().AtomRings()
            if len(ring) >= SCAFFOLD_RING_MIN_SIZE for a in ring}


def _sidechain_subtrees(mol: Chem.Mol) -> list[SidechainSubtree]:
    """One swingable subtree per scaffold exit bond: BFS from the
    non-scaffold end, blocked at scaffold atoms, keeping the whole side
    chain (own rings included) rigid around its pivot. Subtrees bonded to
    the scaffold anywhere besides the pivot (fused/bridged residue rings)
    would deform under a swing and are left out."""
    scaffold = _scaffold_atoms(mol)
    if not scaffold:
        return []
    adjacency: dict[int, list[int]] = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)

    subtrees = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (i in scaffold) == (j in scaffold):
            continue
        pivot, root = (i, j) if i in scaffold else (j, i)
        atoms, seen, queue = [root], {root}, deque([root])
        while queue:
            cur = queue.popleft()
            for nbr in adjacency[cur]:
                if nbr in seen or nbr in scaffold:
                    continue
                seen.add(nbr)
                atoms.append(nbr)
                queue.append(nbr)
        anchored = any(nbr != pivot and nbr not in seen
                       for a in atoms for nbr in adjacency[a])
        if anchored:
            continue
        subtrees.append(SidechainSubtree(pivot=pivot, root=root,
                                         atoms=tuple(atoms)))
    return subtrees


def _rotation_candidates(step_deg: float) -> list[float]:
    """0..360 deg in `step_deg` steps, ordered by displacement from the
    original pose so ties keep the smallest swing."""
    steps = int(round(360.0 / step_deg))
    if steps < 2:
        raise ValueError(f"rotation step {step_deg} deg coarser than 360 deg")
    return sorted((step_deg * k for k in range(steps)),
                  key=lambda a: (min(a, 360.0 - a), a))


def _separation_score(dist: np.ndarray) -> float:
    """Sum over subtree atoms of the minimum distance to any atom outside
    the subtree — the quantity the rotation scan maximizes."""
    return float(dist.min(axis=1).sum())


def _best_swing_angle(mol: Chem.Mol, subtree: SidechainSubtree,
                      ratio: float = CONFLICT_RATIO_DEFAULT,
                      step_deg: float = ROTATION_STEP_DEG) -> float:
    """Scan the swing angles and return the best one, 0.0 when the current
    pose already wins.

    Ranking key per candidate: fewest clashes the swing leaves between the
    subtree and the rest of the molecule first, mildest total clash
    shortfall second, then the separation score, then the angle nearest
    the original pose. The clash terms guard the scan — the raw separation
    sum alone drives a subtree through a neighbour's territory (measured
    on the gold set: it trades a few atoms into 0.2x-bond-length
    collisions for a higher average clearance).
    """
    coords = _conformer_xy(mol)
    std = _standard_bond_length(mol, coords)
    n = len(coords)
    sub_idx = np.asarray(subtree.atoms, dtype=int)
    rest_idx = np.nonzero(~np.isin(np.arange(n), sub_idx))[0]
    rest_xy = coords[rest_idx]

    bonded = np.eye(n, dtype=bool)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True
    sub_bonded = bonded[np.ix_(sub_idx, rest_idx)]
    rel = coords[sub_idx] - coords[subtree.pivot]
    pivot_xy = coords[subtree.pivot]
    threshold = ratio * std

    def evaluate(angle_deg: float) -> tuple[int, float, float]:
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        swung = rel @ rot.T + pivot_xy
        diff = swung[:, None, :] - rest_xy[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        clashing = (dist < threshold) & ~sub_bonded
        clashes = int(clashing.sum())
        severity = float((threshold - dist[clashing]).sum())
        return clashes, severity, _separation_score(dist)

    best_angle = 0.0
    best = evaluate(0.0)
    for angle in _rotation_candidates(step_deg):
        if angle == 0.0:
            continue
        cand = evaluate(angle)
        won = False
        if cand[0] != best[0]:
            won = cand[0] < best[0]
        elif cand[1] != best[1]:
            won = cand[1] < best[1] - _TIE_EPS
        else:
            won = cand[2] > best[2] + _TIE_EPS
        if won:
            best_angle, best = angle, cand
    return best_angle


def _apply_swing(mol: Chem.Mol, subtree: SidechainSubtree,
                 angle_deg: float) -> None:
    """Rotate the subtree atoms around the pivot by `angle_deg` in place."""
    conf = mol.GetConformer()
    coords = _conformer_xy(mol)
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    pivot_xy = coords[subtree.pivot]
    for atom in subtree.atoms:
        xy = rot @ (coords[atom] - pivot_xy) + pivot_xy
        conf.SetAtomPosition(atom, Point3D(float(xy[0]), float(xy[1]), 0.0))


def _plan_rotations(pairs: list[ConflictPair],
                    subtrees: list[SidechainSubtree]) -> dict[int, float]:
    """Map swingable subtree index -> its worst conflict shortfall, for
    pairs the subtree can actually move (exactly one atom of the pair
    inside it). Scaffold collisions, intra-subtree folds and pairs on
    bridged (non-swingable) rings are rigid and left out."""
    membership: dict[int, list[int]] = {}
    for si, subtree in enumerate(subtrees):
        for atom in subtree.atoms:
            membership.setdefault(atom, []).append(si)
    atom_sets = [set(s.atoms) for s in subtrees]
    severity: dict[int, float] = {}
    for pair in pairs:
        touched = set(membership.get(pair.atom_a, []))
        touched.update(membership.get(pair.atom_b, []))
        for si in touched:
            in_a = pair.atom_a in atom_sets[si]
            in_b = pair.atom_b in atom_sets[si]
            if in_a != in_b:
                severity[si] = max(severity.get(si, 0.0), pair.shortfall)
    return severity


def resolve_sidechain_overlaps(mol: Chem.Mol,
                               ratio: float = CONFLICT_RATIO_DEFAULT,
                               max_rounds: int = MAX_ROUNDS_DEFAULT
                               ) -> SidechainResolveReport:
    """Resolve side-chain overlaps on a copy of `mol`'s conformer.

    Raises UnresolvedSidechainConflictsError when conflicts survive
    `max_rounds` detect/resolve passes; the error carries the report and
    the best-effort molecule.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    work = Chem.Mol(mol)
    if work.GetNumConformers() == 0:
        raise SidechainLayoutError(
            "side-chain resolution needs an existing conformer "
            "(run the layout engine first)")
    Chem.FastFindRings(work)

    before = conflict_pairs(work, ratio)
    coords = _conformer_xy(work)
    std = _standard_bond_length(work, coords)
    subtrees = _sidechain_subtrees(work)
    swings: list[tuple[int, int, float]] = []
    rounds_used = 0

    if before and std > 0.0 and subtrees:
        atom_sets = [set(s.atoms) for s in subtrees]
        for round_no in range(max_rounds):
            rounds_used = round_no + 1
            severity = _plan_rotations(conflict_pairs(work, ratio), subtrees)
            if not severity:
                break
            # most crowded subtree first; index breaks ties deterministically
            order = sorted(severity, key=lambda si: (-severity[si], si))
            for si in order:
                live = conflict_pairs(work, ratio)
                still_hit = any(
                    (p.atom_a in atom_sets[si]) != (p.atom_b in atom_sets[si])
                    for p in live)
                if not still_hit:
                    continue
                angle = _best_swing_angle(work, subtrees[si], ratio=ratio)
                if angle != 0.0:
                    _apply_swing(work, subtrees[si], angle)
                    swings.append((subtrees[si].pivot, subtrees[si].root,
                                   angle))

    after = conflict_pairs(work, ratio)
    report = SidechainResolveReport(
        molecule=work,
        conflicts_before=len(before),
        conflicts_after=len(after),
        rounds_used=rounds_used,
        rotations_applied=len(swings),
        standard_bond_length=std,
        unresolved=after,
        swings=swings)
    if after:
        raise UnresolvedSidechainConflictsError(report)
    return report


# ---------------------------------------------------------------------------
# CLI: CoordGen-initial vs resolved render pairs for visual QA
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _coordgen_layout(smiles: str) -> Chem.Mol:
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        raise SidechainLayoutError(
            f"SMILES failed to parse: {smiles[:80]!r}")
    laid = coordgen_coords(ref)
    if laid is None:
        raise SidechainLayoutError("CoordGen layout failed for this molecule")
    return laid


def _read_smiles_records(path: str) -> list[tuple[str, str]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) >= 2:
                records.append((fields[0], fields[1]))
            else:
                records.append((f"mol{line_no:03d}", fields[0]))
    if not records:
        raise SidechainLayoutError(f"no SMILES records in {path}")
    return records


def _render_png(mol: Chem.Mol, path: str, seed: int) -> str:
    """Render in the ALIGNED corpus style; a fresh Random(seed) per call
    gives the before/after pair the identical orientation and margin."""
    img = draw_molecule(Chem.Mol(mol), ALIGNED, random.Random(seed))
    img.save(str(path))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Resolve CoordGen side-chain overlaps and render "
                    "before/after PNG pairs in the ALIGNED corpus style.")
    ap.add_argument("--smiles", help="single SMILES string")
    ap.add_argument("--smiles-file",
                    help=".smi file, one 'name<TAB>smiles' (or bare SMILES) "
                         "per line")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0,
                    help="render orientation seed (before/after share it)")
    ap.add_argument("--ratio", type=float, default=CONFLICT_RATIO_DEFAULT,
                    help="conflict threshold as a fraction of the standard "
                         f"bond length (default {CONFLICT_RATIO_DEFAULT})")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT,
                    help="resolution iteration cap (default "
                         f"{MAX_ROUNDS_DEFAULT})")
    args = ap.parse_args(argv)
    if bool(args.smiles) == bool(args.smiles_file):
        ap.error("pass exactly one of --smiles / --smiles-file")
    try:
        records = ([("molecule", args.smiles)] if args.smiles
                   else _read_smiles_records(args.smiles_file))
    except (SidechainResolveError, OSError) as exc:
        print(f"sidechain_resolve: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    hard_failures = unresolved_molecules = 0
    for name, smiles in records:
        try:
            laid = _coordgen_layout(smiles)
            try:
                report = resolve_sidechain_overlaps(
                    laid, ratio=args.ratio, max_rounds=args.max_rounds)
            except UnresolvedSidechainConflictsError as exc:
                report = exc.report
                unresolved_molecules += 1
            stem = _safe_name(name)
            before_png = _render_png(laid, os.path.join(out_dir,
                                                        f"{stem}_coordgen.png"),
                                     args.seed)
            after_png = _render_png(report.molecule,
                                    os.path.join(out_dir,
                                                 f"{stem}_resolved.png"),
                                    args.seed)
            status = "UNRESOLVED" if report.unresolved else "resolved"
            print(f"{name}: atoms={laid.GetNumAtoms()} "
                  f"conflicts={report.conflicts_before}"
                  f"->{report.conflicts_after} "
                  f"rounds={report.rounds_used} "
                  f"swings={report.rotations_applied} [{status}]")
            for pair in report.unresolved:
                print(f"  unresolved {pair.label} at "
                      f"{pair.bond_length_ratio:.2f}x bond length")
            print(f"  before {before_png}")
            print(f"  after  {after_png}")
        except SidechainResolveError as exc:
            print(f"sidechain_resolve: {name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            hard_failures += 1
    if hard_failures:
        return 1
    if unresolved_molecules:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

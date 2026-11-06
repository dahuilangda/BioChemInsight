"""Patent-style zig-zag macrocycle layout (hex-lattice template).

Port of the validated prototype (research/macrocycle_layout_algorithms.md,
section 6; /tmp/macrocycle_research/scripts/zigzag_template_prototype.py)
into the training render pipeline as the single maintained implementation,
generalized from the prototype's cyclo-Gly/Ala homopolymers to arbitrary
n-residue macrorings with in-ring heteroatoms, embedded small rings
(proline homologues) and arbitrary side chains.

Algorithm (CoordGen's own polyomino machinery, re-authored in Python):

  1. The macroring (largest ring; amide-carbonyl-count tie-break) is
     placed on the contour of a hex-lattice polyomino — ragged-box or
     skewed-box — whose contour carries exactly n vertices.  The plain
     box families only reach contour sizes n = 2 (mod 4); the other
     classes come from the two CoordGen-style moves, single-hex removal
     (listOfEquivalents, -2) and the pentagon merge of one contour vertex
     (markOneVertexAsPentagon, -1), combined where needed.  Boxes are
     preferred elongated (patent style: aspect 1.5-3.0, straight sawtooth
     runs on the long sides); the target aspect is randomized per render
     so the pool spans the real-crop shape distribution.
  2. Ring atoms are assigned to contour vertices in ring order (random
     start / direction / mirror for depiction diversity), giving all-120
     degree bond angles with zero collapse by construction — exact
     lattice coordinates, no solver.
  3. Side chains, carbonyl oxygens, N-methyls and small rings
     (pyrrolidine / piperidine / phenyl / indole, fused or sharing a path
     with the macroring) are placed by a DFS off the ring with the same
     lattice rules the prototype used: carbonyl O on the exterior
     bisector, chains as strict +/-60 deg zig-zags with clearance-scored
     turn choice, small rings as regular-polygon arcs anchored on the
     already-placed atoms.  A final pass rigidly rotates crowded
     side-chain subtrees around their attachment vertices.

`zigzag_coords()` is the render-pipeline entry point (drop-in for
style_calibration.coordgen_coords): it returns a depiction-randomized
molecule carrying layout coordinates, or None when the molecule has no
usable macroring / the layout fails geometry QA — callers fall back to
CoordGen.  The prototype's alternative finisher (hand the ring coords to
the RDKit native depictor as a coordMap hard constraint) is retained as
`finisher="native"` for regression parity, but it is not the default:
measured on the real-patent gold SMILES (seq53 etc.) the free side-chain
atoms it places crowd the constrained ring below label clearance
(min non-bonded 0.0-0.4 bond lengths), while the DFS finisher keeps
patent-style side chains collision-free.
"""
from __future__ import annotations

import math
import random

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D, Point3D

RDLogger.DisableLog("rdApp.*")

BOND = 1.0                    # lattice bond length in layout units
SQRT3H = 0.8660254037844386   # sqrt(3)/2 — hex vertex row spacing
ZIG_TURN = 60.0               # travel turn for a 120 deg chain vertex
MIN_RING_ATOMS = 9            # CoordGen's MACROCYCLE threshold
MAX_TEMPLATE_DIM = 30         # box enumeration covers contours to ~200 vertices
MIN_NONBONDED_RATIO = 0.42    # placement comfort target (bond units)
# hard QA floor: the incumbent pipeline's own quality on the same pool
# molecules measures min non-bonded 0.08-0.70 (median 0.61) with bond
# crossings in 13 of 15 depictions — and that corpus trained the 0.9116
# round-3 baseline — so the zigzag corpus is only rejected for GROSS
# failures (collapse / heavy self-overlap), never held to a bar the
# incumbent corpus itself does not meet
MIN_NONBONDED_FLOOR = 0.25
MAX_CROSSINGS = 6
PRESERVE_TOL = 1e-3           # QA: coordMap coordinates must survive verbatim

# lattice caches: template search is pure geometry, identical for every
# molecule with the same ring size
_TEMPLATE_CACHE: dict[int, list["MacrocycleTemplate"]] = {}


# ---------------------------------------------------------------------------
# hex lattice (axial hex coords; vertices in cube coords), ported from
# coordgenlibs Polyomino / CoordgenMacrocycleBuilder
# ---------------------------------------------------------------------------

def _hex_neighbors(h: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = h
    return [(x + 1, y), (x + 1, y - 1), (x, y - 1),
            (x - 1, y), (x - 1, y + 1), (x, y + 1)]


def _to_vertex(h: tuple[int, int]) -> tuple[int, int, int]:
    x, y = h
    return (x, y, -x - y)


class MacrocycleTemplate:
    """Polyomino on the hex lattice with its closed contour path.

    The contour vertices are the slots available to macroring atoms;
    ``vertex_xy`` maps a cube vertex to layout coordinates where every
    bond is exactly BOND long and every interior angle is 120 or 240
    degrees — the box corners are lattice turns, so no right angles
    appear anywhere on the contour.
    """

    def __init__(self, kind: str = "ragged") -> None:
        self.kind = kind                # "ragged" | "skewed"
        self.hexes: set[tuple[int, int]] = set()
        self.pentagons: list[tuple[int, int, int]] = []
        self._path: list[tuple[int, int, int]] | None = None

    # -- construction -------------------------------------------------------

    def build(self, x: int, y: int) -> None:
        """x-wide, y-tall box; ragged boxes shift every second row pair by
        one lattice unit so the long edges become straight sawtooth runs."""
        self.hexes = set()
        self.pentagons = []
        self._path = None
        if self.kind == "ragged":
            startx = 0
            yy = 0
            while yy < y:
                for xx in range(x):
                    self.hexes.add((startx + xx, yy))
                yy += 1
                if yy >= y:
                    break
                for xx in range(x):
                    self.hexes.add((startx + xx, yy))
                yy += 1
                startx -= 1
        else:
            for yy in range(y):
                for xx in range(x):
                    self.hexes.add((xx, yy))

    # -- vertex machinery (cube coords) --------------------------------------

    def _vertex_neighbors(self, v: tuple[int, int, int]
                          ) -> list[tuple[int, int]]:
        x, y, z = v
        d = x + y + z
        if d not in (1, -1):
            raise ValueError(f"bad vertex {v}")
        return [hh for hh in ((x - d, y), (x, y - d), (x, y))
                if hh in self.hexes]

    def _hexagons_at_vertex(self, v: tuple[int, int, int]) -> int:
        return len(self._vertex_neighbors(v))

    def _following_vertex(self, hx: tuple[int, int],
                          v: tuple[int, int, int]) -> tuple[int, int, int]:
        x, y = hx
        z = -x - y
        dx, dy, dz = v[0] - x, v[1] - y, v[2] - z
        if dx == 0 and dy == 0:
            dx, dy, dz = -dz, 0, 0
        elif dx == 0 and dz == 0:
            dx, dy, dz = 0, 0, -dy
        elif dy == 0 and dz == 0:
            dx, dy, dz = 0, -dx, 0
        return (x + dx, y + dy, z + dz)

    # -- contour path ---------------------------------------------------------

    def contour(self) -> list[tuple[int, int, int]]:
        """Closed contour walk (one vertex per macroring atom slot)."""
        if self._path is not None:
            return self._path
        first = None
        for h in self.hexes:
            x, y = h
            vert = (x + 1, y, -x - y)
            if self._hexagons_at_vertex(vert) == 1:
                first = vert
                break
        if first is None:
            raise ValueError("no outer vertex")
        cur = first
        curhex = self._vertex_neighbors(first)[0]
        nxt = self._following_vertex(curhex, cur)
        out: list[tuple[int, int, int]] = []
        while True:
            if cur not in self.pentagons:
                out.append(cur)
            cur = nxt
            nb = self._vertex_neighbors(cur)
            if len(nb) > 2:
                raise ValueError("contour walk hit a branch vertex")
            if len(nb) == 2:
                curhex = nb[0] if nb[0] != curhex else nb[1]
            nxt = self._following_vertex(curhex, cur)
            if cur == first:
                break
        self._path = out
        return out

    def merge_one_vertex(self) -> bool:
        """Fuse one contour vertex into a pentagon (odd contour sizes), the
        CoordgenMacrocycleBuilder.markOneVertexAsPentagon rule."""
        path = self.contour()

        def mult(v):
            return self._hexagons_at_vertex(v)

        for phase in ([2, 1, 2], [1, 2, 1]):
            pm, tm = mult(path[-1]), mult(path[0])
            nm = 0
            for i in range(len(path)):
                ni = 0 if i >= len(path) - 1 else i + 1
                nm = mult(path[ni])
                if pm == phase[0] and tm == phase[1] and nm == phase[2]:
                    self.pentagons.append(path[i])
                    self._path = None
                    return len(self.contour()) == len(path) - 1
                pm, tm = tm, nm
        return False

    @property
    def size(self) -> int:
        """Contour vertex count = number of macroring atoms placeable."""
        return len(self.contour())

    @property
    def box(self) -> tuple[int, int]:
        xs = [h[0] for h in self.hexes]
        ys = [h[1] for h in self.hexes]
        return (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

    # -- geometry ---------------------------------------------------------

    def vertex_xy(self, v: tuple[int, int, int]) -> tuple[float, float]:
        x, y, z = v
        return (BOND * SQRT3H * (x - z), BOND * (-0.5 * x + y - 0.5 * z))

    def substituent_vertex(self, pos: tuple[int, int, int]
                           ) -> tuple[int, int, int]:
        """Outward lattice vertex adjacent to contour vertex `pos` (the
        CoordgenMacrocycleBuilder.coordinatesOfSubstituent rule)."""
        nb = self._vertex_neighbors(pos)
        if not nb:
            raise ValueError("no hex at substituent vertex")
        if len(nb) == 1:
            pc = _to_vertex(nb[0])
            v = (pos[0] - pc[0], pos[1] - pc[1], pos[2] - pc[2])
            d = 1 if (v[0] + v[1] + v[2]) > 0 else -1
            if v[0] == 0:
                v = (v[0] - d, v[1], v[2])
            if v[1] == 0:
                v = (v[0], v[1] - d, v[2])
            if v[2] == 0:
                v = (v[0], v[1], v[2] - d)
            return (pc[0] + v[0], pc[1] + v[1], pc[2] + v[2])
        p1, p2 = _to_vertex(nb[0]), _to_vertex(nb[1])
        v = (pos[0] - p1[0], pos[1] - p1[1], pos[2] - p1[2])
        return (p2[0] - v[0], p2[1] - v[1], p2[2] - v[2])


# ---------------------------------------------------------------------------
# template selection
# ---------------------------------------------------------------------------

def _simply_connected(hexes: set[tuple[int, int]]) -> bool:
    start = next(iter(hexes))
    seen = {start}
    stack = [start]
    while stack:
        h = stack.pop()
        for nb in _hex_neighbors(h):
            if nb in hexes and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(hexes)


def _minus_one_hex(p: MacrocycleTemplate) -> list[MacrocycleTemplate]:
    """Single-hex equivalents (the CoordGen listOfEquivalents move):
    removing a boundary hex with exactly two neighbours shortens the
    contour by exactly two vertices and leaves an irregular notch on the
    edge — the shape grammar that reaches the contour sizes the plain box
    families cannot."""
    out: list[MacrocycleTemplate] = []
    target = len(p.contour()) - 2
    for h in list(p.hexes):
        if sum(1 for q in _hex_neighbors(h) if q in p.hexes) != 2:
            continue
        q = MacrocycleTemplate(p.kind)
        q.hexes = {g for g in p.hexes if g != h}
        if not _simply_connected(q.hexes):
            continue
        try:
            if q.size == target:
                out.append(q)
        except ValueError:
            continue
    return out


def _candidate_templates(n: int) -> list[MacrocycleTemplate]:
    """Every template whose contour carries exactly n vertices (all four
    size classes; cached — the search is identical for every molecule
    with the same ring size)."""
    if n in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[n]
    cands: list[MacrocycleTemplate] = []
    for kind in ("ragged", "skewed"):
        boxes: dict[int, list[MacrocycleTemplate]] = {}
        for x in range(1, MAX_TEMPLATE_DIM + 1):
            for y in range(1, MAX_TEMPLATE_DIM + 1):
                p = MacrocycleTemplate(kind)
                p.build(x, y)
                try:
                    boxes.setdefault(p.size, []).append(p)
                except ValueError:
                    continue
        cands.extend(boxes.get(n, []))
        for p in boxes.get(n + 1, []):            # odd n: pentagon merge
            if p.merge_one_vertex():
                cands.append(p)
        for base_size in (n + 2, n + 3):          # single-hex equivalents
            for p in boxes.get(base_size, []):
                for q in _minus_one_hex(p):
                    rem = base_size - 2           # n (even) or n+1 (odd)
                    if rem == n:
                        cands.append(q)
                    elif rem == n + 1 and q.merge_one_vertex():
                        cands.append(q)
    _TEMPLATE_CACHE[n] = cands
    return cands


def select_template(n: int, rng: random.Random
                    ) -> MacrocycleTemplate | None:
    """Pick the patent-style box template for an n-atom macroring.

    Preference: elongated boxes (long/short side ratio near a randomized
    1.5-3.0 target, short side >= 3 so real sawtooth runs exist), ragged
    over plain skewed.  Returns None when no template in the enumeration
    range carries exactly n contour vertices — callers fall back to
    CoordGen for those ring sizes.
    """
    if n < MIN_RING_ATOMS:
        return None
    cands = _candidate_templates(n)
    if not cands:
        return None
    target_aspect = rng.uniform(1.5, 3.0)
    min_dim = rng.choice((3, 4))

    def score(p: MacrocycleTemplate) -> tuple:
        bx, by = p.box
        lo, hi = min(bx, by), max(bx, by)
        return (0 if min(lo, hi) >= min_dim else 1,
                abs(hi / lo - target_aspect),
                0 if p.kind == "ragged" else 1,
                lo, hi)

    ranked = sorted(cands, key=score)
    best2 = score(ranked[0])[:2]
    ties = [p for p in ranked if score(p)[:2] == best2]
    return rng.choice(ties[:4])


# ---------------------------------------------------------------------------
# molecular bookkeeping
# ---------------------------------------------------------------------------

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


def find_macroring(mol: Chem.Mol) -> list[int]:
    """Atom cycle of the macroring: the largest ring; ties broken toward
    the ring carrying the most amide carbonyls (the peptide backbone ring,
    not a fused small ring)."""
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


def shuffle_atoms(mol: Chem.Mol, seed: int) -> Chem.Mol:
    """Depiction-randomizing atom renumbering (same trick as
    style_calibration.coordgen_coords) so repeated renders of one SMILES
    decorrelate."""
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.RenumberAtoms(mol, order)


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


# ---------------------------------------------------------------------------
# side-chain placement (DFS off the macroring, lattice rules)
# ---------------------------------------------------------------------------

class _SubstituentPlacer:
    """Vertex-by-vertex 2D placement of everything hanging off the
    macroring: side chains, carbonyl oxygens, N-methyls and small rings.

    Chain vertices keep the +/-60 deg zig-zag of the ring contour; turn
    choice is scored for drawing clearance so branches never fold back
    over placed atoms; small rings are laid as regular-polygon arcs
    anchored on already-placed atoms so fused systems (indole) and rings
    sharing a path with the macroring (proline homologues) close exactly.
    """

    def __init__(self, mol: Chem.Mol, bond: float):
        self.mol = mol
        self.bond = bond
        self.pos: dict[int, tuple[float, float]] = {}
        self.ring_done: set[tuple[int, ...]] = set()
        self.placed: set[int] = set()
        # centre of the macroring (set from the template coordinates):
        # the interior/exterior test for substituent directions — the
        # centroid of already-placed atoms drifts as branches grow
        self.ring_center: tuple[float, float] | None = None
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

    # -- geometry helpers ---------------------------------------------------

    def centroid(self) -> tuple[float, float]:
        xs = [self.pos[i][0] for i in self.placed]
        ys = [self.pos[i][1] for i in self.placed]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _atom_center(self, idx: int) -> tuple[float, float]:
        return self.local_center.get(idx, self.ring_center or self.centroid())

    def exterior_bisector(self, idx: int, center: tuple[float, float]
                          ) -> tuple[float, float]:
        """Outward bisector of the already-placed bonds at idx — where a
        substituent / carbonyl O / N-methyl goes.

        With two placed bonds the bisector of the SHORT sector between
        them is the interior bisector on a convex (120 deg) vertex but the
        exterior bisector on a reflex (240 deg) box corner; which of the
        two directions is interior is decided against the macroring
        centre, never against the drifting placed-atom centroid (at a
        jagged contour the centroid direction can point across the ring
        path, which would throw the substituent into the notch between
        the ring bonds)."""
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
        sx, sy = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
        away = _norm((self.pos[idx][0] - center[0],
                      self.pos[idx][1] - center[1]))
        if math.hypot(sx, sy) < 0.1:
            # fully coordinated vertex (three bonds at 120 deg): no bisector
            # exists — fall back to directly away from the centre
            return away
        bis = _norm((sx, sy))
        # the interior side is the one facing the macroring centre
        inward = (_norm((self.ring_center[0] - self.pos[idx][0],
                         self.ring_center[1] - self.pos[idx][1]))
                  if self.ring_center is not None else away)
        interior = bis if (bis[0] * inward[0] + bis[1] * inward[1] >=
                           -bis[0] * inward[0] - bis[1] * inward[1]) else \
            (-bis[0], -bis[1])
        return (-interior[0], -interior[1])

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
        center = center or self.centroid()
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
            cl -= 4.0 * self.bond * self._cross_count(trialpos, poly_bonds)
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

        The arc spans len(seq)-1 edges of a regular n-gon: its central
        sweep is exactly edges*360/n (which for fused rings such as
        indoles is the MAJOR arc — the unplaced path covers most of the
        ring — and taking the minor arc instead would crush all new
        vertices into a sliver of the chord).  The polygon radius is
        derived from the anchored chord so both endpoints match exactly
        and all edges of the ring share one length (a few percent off
        nominal — imperceptible, and how draftsmen draw fused rings).
        The arc opens on the side with more drawing clearance (the side
        toward a fused partner ring would fold onto it).
        """
        a, b = self.pos[seq[0]], self.pos[seq[-1]]
        edges = len(seq) - 1
        chord = math.hypot(b[0] - a[0], b[1] - a[1])
        radius = (chord / 2.0) / math.sin(math.pi * edges / n)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        perp = _norm((-(b[1] - a[1]), b[0] - a[0]))
        h = math.sqrt(max(radius * radius - chord * chord / 4.0, 0.0))
        exclude = {seq[0], seq[-1]}
        sweep_mag = edges * (2.0 * math.pi / n)
        best = None
        for side in (1.0, -1.0):
            center = (mid[0] + perp[0] * h * side, mid[1] + perp[1] * h * side)
            ang0 = math.atan2(a[1] - center[1], a[0] - center[0])
            ang1 = math.atan2(b[1] - center[1], b[0] - center[0])
            delta = (ang1 - ang0) % (2.0 * math.pi)   # minor arc angle
            if delta > math.pi:
                delta -= 2.0 * math.pi
            # sweep with |sweep| = sweep_mag that lands on ang1: for a
            # minor arc (< pi) it runs the same direction, for a major
            # arc it runs the opposite way (delta = 2pi - sweep_mag then)
            if abs(delta) <= math.pi and sweep_mag <= math.pi:
                sweep = math.copysign(sweep_mag, delta)
            else:
                sweep = -math.copysign(sweep_mag, delta)
            trial = []
            for k in range(1, edges):
                t = ang0 + sweep * k / edges
                trial.append((center[0] + radius * math.cos(t),
                              center[1] + radius * math.sin(t)))
            cl = sum(min(self._clearance(c, exclude), 2.0 * self.bond)
                     for c in trial)
            trialpos = {i: c for i, c in zip(seq[1:-1], trial)}
            trialpos[seq[0]] = a
            trialpos[seq[-1]] = b
            arc_bonds = [(seq[k], seq[k + 1]) for k in range(edges)]
            cl -= 4.0 * self.bond * self._cross_count(trialpos, arc_bonds)
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
        slot = self.sub_slot.get(idx) if hasattr(self, "sub_slot") else None
        if len(chains) == 1:
            if slot is not None and self._slot_free(idx, slot):
                self._place_chain(chains[0], idx,
                                  first_dir=_norm((slot[0] - self.pos[idx][0],
                                                   slot[1] - self.pos[idx][1])))
            else:
                self._place_chain(chains[0], idx)
        elif len(chains) > 1:
            # quaternary ring vertex (alpha-methyl type C alpha with two
            # side chains): the first chain takes the lattice slot, the
            # rest fan around it with clearance-scored offsets
            base = None
            if slot is not None and self._slot_free(idx, slot):
                base = _norm((slot[0] - self.pos[idx][0],
                              slot[1] - self.pos[idx][1]))
            if base is None:
                base = self.exterior_bisector(idx, center)
            m = len(chains)
            best = None
            for spread in (30.0, 52.5, 75.0):
                for flip in (False, True):
                    trial = [(-spread + 2.0 * spread * k / (m - 1))
                             for k in range(m)]
                    if flip:
                        trial = list(reversed(trial))
                    score = 0.0
                    for n, off in zip(chains, trial):
                        c = (self.pos[idx][0] + self.bond
                             * _rot(base, off)[0],
                             self.pos[idx][1] + self.bond
                             * _rot(base, off)[1])
                        score += min(self._clearance(c, {idx}),
                                     2.0 * self.bond)
                    if best is None or score > best[0]:
                        best = (score, trial)
            for n, off in zip(chains, best[1]):
                self._place_chain(n, idx, first_dir=_rot(base, off))
        if o is not None and o not in self.pos:
            if slot is not None and self._slot_free(idx, slot):
                self.pos[o] = slot
                self.placed.add(o)
            else:
                self._place_carbonyl_o(idx, o)

    def _slot_free(self, idx: int, slot: tuple[float, float]) -> bool:
        """True when no atom other than `idx` already occupies the lattice
        substituent slot (a neighbouring residue's branch may reach it)."""
        for other, (ox, oy) in self.pos.items():
            if other != idx and \
                    math.hypot(ox - slot[0], oy - slot[1]) < 0.5 * self.bond:
                return False
        return True

    def _place_carbonyl_o(self, idx: int, o: int) -> None:
        """Carbonyl O on the exterior bisector of its carbon; when a
        sibling branch already occupies that vertex, take the best
        clearance among small angular deviations (the C=O tilt draftsmen
        draw to dodge a crowded flank)."""
        center = self._atom_center(idx)
        base = self.exterior_bisector(idx, center)
        best = None
        for delta in (0.0, -20.0, 20.0, -40.0, 40.0, -60.0, 60.0):
            d = _rot(base, delta)
            c = (self.pos[idx][0] + self.bond * d[0],
                 self.pos[idx][1] + self.bond * d[1])
            cl = self._clearance(c, {idx})
            score = cl + 0.3 * (d[0] * base[0] + d[1] * base[1])
            if best is None or score > best[0]:
                best = (score, c)
            if cl >= 1.5 * self.bond:
                break
        self.pos[o] = best[1]
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
            # first segment bond: exterior bisector of the parent vertex,
            # tilted to the best-clearance candidate when that vertex is
            # crowded (fully coordinated ring atoms have no bisector at
            # all, and a naive bisector can land on an occupied vertex)
            base = self.exterior_bisector(parent, self.centroid())
            best = None
            for delta in (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0,
                          -120.0, 120.0, -150.0, 150.0, 180.0):
                cand = _rot(base, delta)
                c = (p[0] + self.bond * cand[0], p[1] + self.bond * cand[1])
                cl = min(self._clearance(c, {parent}), 1.5 * self.bond)
                score = cl + 0.3 * (cand[0] * base[0] + cand[1] * base[1])
                if best is None or score > best[0]:
                    best = (score, cand)
                if cl >= 1.5 * self.bond:
                    break
            hint = d = best[1]
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
            self._place_carbonyl_o(atom, o)

    # -- crowding relaxation ---------------------------------------------------

    def relax_sidechains(self, cycle: list[int]) -> None:
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

        cap = 1.3 * self.bond
        for _ in range(2):
            changed = False
            for idx in order:
                p = parent[idx]
                sub = subtree(idx)
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
                        dot = max(-1.0, min(1.0, nd[0] * v[0]
                                            + nd[1] * v[1]))
                        ang = min(ang, math.degrees(math.acos(dot)))
                    if ang < 100.0:
                        total -= (100.0 - ang) * self.bond / 20.0
                    newpos = {}
                    for a in sub:
                        qa = self.pos[a]
                        if a == idx:
                            newpos[a] = (ppos[0] + nd[0], ppos[1] + nd[1])
                        else:
                            dx, dy = qa[0] - ppos[0], qa[1] - ppos[1]
                            newpos[a] = (ppos[0] + dx * c - dy * s,
                                         ppos[1] + dx * s + dy * c)
                    for a in sub:
                        excl = sub_set | {p}
                        total += min(self._clearance(newpos[a], excl), cap)
                    # subtree bonds must not cross other bonds
                    sub_bonds = [(a, b) for a in sub for b in self.bonded[a]
                                 if b in sub_set and a < b]
                    other = [(i1, j1) for (i1, j1) in all_bonds
                             if i1 not in sub_set or j1 not in sub_set]
                    for (a1, b1) in sub_bonds:
                        pa, pb = newpos[a1], newpos[b1]
                        for (i2, j2) in other:
                            if len({a1, b1, i2, j2}) < 4:
                                continue
                            if _seg_cross(pa, pb, self.pos[i2],
                                          self.pos[j2]):
                                total -= 4.0 * self.bond
                    return total

                best = (score_for(0.0), 0.0)
                for delta in (-40.0, -30.0, -20.0, -10.0,
                              10.0, 20.0, 30.0, 40.0):
                    s = score_for(delta)
                    if s > best[0] + 1e-9:
                        best = (s, delta)
                if best[1]:
                    rad = math.radians(best[1])
                    c, s = math.cos(rad), math.sin(rad)
                    for a in sub:
                        qa = self.pos[a]
                        dx, dy = qa[0] - ppos[0], qa[1] - ppos[1]
                        self.pos[a] = (ppos[0] + dx * c - dy * s,
                                       ppos[1] + dx * s + dy * c)
                    changed = True
            if not changed:
                break

    # -- stochastic repair -----------------------------------------------------

    def violation_score(self, pinned: set[int], floor: float
                        ) -> tuple[float, int, float, int]:
        """Weighted geometry violations: bond crossings dominate, then
        non-bonded pairs closer than `floor` bond units.  Returns
        (score, crossings, min_distance, n_below_strict).  Crossing tests
        are bbox pre-filtered so the repair loop can afford thousands of
        attempts."""
        pts = self.pos
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in self.mol.GetBonds()]
        crossings = 0
        boxes = []
        for (i, j) in bonds:
            pi, pj = pts[i], pts[j]
            boxes.append((min(pi[0], pj[0]), min(pi[1], pj[1]),
                          max(pi[0], pj[0]), max(pi[1], pj[1])))
        for k in range(len(bonds)):
            i1, j1 = bonds[k]
            b1 = boxes[k]
            for l in range(k + 1, len(bonds)):
                i2, j2 = bonds[l]
                if len({i1, j1, i2, j2}) < 4:
                    continue
                b2 = boxes[l]
                if b1[0] > b2[2] or b2[0] > b1[2] or \
                        b1[1] > b2[3] or b2[1] > b1[3]:
                    continue
                if _seg_cross(pts[i1], pts[j1], pts[i2], pts[j2]):
                    crossings += 1
        close = 0.0
        mind = float("inf")
        n_strict = 0
        ids = list(pts)
        thr = floor * self.bond
        for a in range(len(ids)):
            i = ids[a]
            pi = pts[i]
            for b in range(a + 1, len(ids)):
                j = ids[b]
                if j in self.bonded[i]:
                    continue
                d = math.hypot(pts[j][0] - pi[0], pts[j][1] - pi[1])
                mind = min(mind, d)
                if d < thr:
                    close += (thr - d) / self.bond
                    if d < MIN_NONBONDED_RATIO * self.bond:
                        n_strict += 1
        return (10.0 * crossings + close, crossings, mind, n_strict)


# ---------------------------------------------------------------------------
# layout entry points
# ---------------------------------------------------------------------------

def _assign_ring_to_contour(poly: MacrocycleTemplate, ring: list[int],
                            rng: random.Random
                            ) -> dict[int, tuple[int, int, int]]:
    """Map ordered ring atoms to contour vertices, depiction-randomized by
    start vertex and walk direction (the mirror is applied when the
    coordinates are built)."""
    path = poly.contour()
    n = len(ring)
    start = rng.randrange(n)
    if rng.random() < 0.5:
        path = path[::-1]
    return {aid: path[(k + start) % len(path)] for k, aid in enumerate(ring)}


def zigzag_coords(mol: Chem.Mol, rng: random.Random | None = None,
                  seed: int = 0, finisher: str = "dfs",
                  ) -> Chem.Mol | None:
    """Lay `mol` out with the zig-zag macrocycle template.

    Returns the atom-shuffled molecule carrying the layout conformer, or
    None when the molecule has no usable macroring or the layout fails
    geometry QA (callers fall back to CoordGen).  finisher="dfs" (default)
    places side chains with the lattice-rule DFS; finisher="native" hands
    the ring coordinates to the RDKit native depictor as a coordMap
    constraint instead (prototype-faithful, but the free side-chain atoms
    it adds can crowd below label clearance on real cyclopeptides).
    """
    if finisher not in ("dfs", "native"):
        raise ValueError(f"unknown finisher: {finisher}")
    rng = rng or random.Random(seed)
    try:
        shuffled = shuffle_atoms(Chem.Mol(mol), seed=rng.randrange(2 ** 31))
        ring = ordered_ring(shuffled, find_macroring(shuffled))
    except ValueError:
        return None
    n = len(ring)
    poly = select_template(n, rng)
    if poly is None:
        return None

    mirrored = rng.random() < 0.5
    assign = _assign_ring_to_contour(poly, ring, rng)
    coords: dict[int, tuple[float, float]] = {}
    for aid, v in assign.items():
        x, y = poly.vertex_xy(v)
        coords[aid] = (-x if mirrored else x, y)

    if finisher == "native":
        cmap = {aid: Point2D(x, y) for aid, (x, y) in coords.items()}
        try:
            rdDepictor.SetPreferCoordGen(False)
            # baseline embedding first (prototype sequence), then the
            # constrained pass — the native engine honours coordMap exactly
            rdDepictor.Compute2DCoords(shuffled, clearConfs=True)
            conf = shuffled.GetConformer()
            for aid, p in cmap.items():
                conf.SetAtomPosition(aid, Point3D(float(p.x), float(p.y), 0.0))
            rdDepictor.Compute2DCoords(shuffled, coordMap=cmap)
        except Exception:
            rdDepictor.SetPreferCoordGen(False)
            return None
        if not _qa(shuffled, cmap, check_preserve=True):
            return None
        return shuffled

    placer = _SubstituentPlacer(shuffled, BOND)
    for aid in ring:
        placer.pos[aid] = coords[aid]
        placer.placed.add(aid)
    placer.ring_center = (
        sum(p[0] for p in coords.values()) / n,
        sum(p[1] for p in coords.values()) / n)
    # lattice substituent slots (the prototype's coordinatesOfSubstituent
    # rule): each contour vertex's outward lattice neighbour sits exactly
    # one bond from every nearby contour vertex, so a substituent placed
    # there can never crowd a ring bond — the guarantee the exterior
    # bisector cannot give at jagged contour corners
    sub_slot: dict[int, tuple[float, float]] = {}
    for aid, v in assign.items():
        try:
            x, y = poly.vertex_xy(poly.substituent_vertex(v))
            sub_slot[aid] = (-x if mirrored else x, y)
        except ValueError:
            continue
    placer.sub_slot = sub_slot
    for aid in ring:                       # substituents in ring order
        placer._place_branches(aid)
    placer.relax_sidechains(ring)
    placer.relax_sidechains(ring)          # second re-aim pass
    if any(i not in placer.pos for i in range(shuffled.GetNumAtoms())):
        return None
    _, crossings, mind, _ = placer.violation_score(set(ring), 0.45)
    if crossings > MAX_CROSSINGS or mind < MIN_NONBONDED_FLOOR * BOND:
        return None
    conf = Chem.Conformer(shuffled.GetNumAtoms())
    for i in range(shuffled.GetNumAtoms()):
        x, y = placer.pos[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    shuffled.RemoveAllConformers()
    shuffled.AddConformer(conf, assignId=True)
    if not _qa(shuffled, {aid: Point2D(*coords[aid]) for aid in ring},
               check_preserve=True):
        return None
    return shuffled


def _qa(mol: Chem.Mol, cmap: dict[int, Point2D],
        check_preserve: bool = False) -> bool:
    """Geometry gates: constrained coordinates preserved verbatim, no
    gross atom collapse (floor calibrated to the incumbent corpus's own
    quality — see MIN_NONBONDED_FLOOR), no heavy self-overlap, all
    coordinates finite."""
    conf = mol.GetConformer()
    if check_preserve:
        for aid, p in cmap.items():
            q = conf.GetAtomPosition(aid)
            if abs(q.x - float(p.x)) > PRESERVE_TOL or \
                    abs(q.y - float(p.y)) > PRESERVE_TOL:
                return False
    n = mol.GetNumAtoms()
    bonded = [set() for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded[i].add(j)
        bonded[j].add(i)
    pts = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
           for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j in bonded[i]:
                continue
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d < MIN_NONBONDED_FLOOR * BOND:
                return False
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    crossings = 0
    for k in range(len(bonds)):
        i1, j1 = bonds[k]
        for l in range(k + 1, len(bonds)):
            i2, j2 = bonds[l]
            if len({i1, j1, i2, j2}) < 4:
                continue
            if _seg_cross(pts[i1], pts[j1], pts[i2], pts[j2]):
                crossings += 1
                if crossings > MAX_CROSSINGS:
                    return False
    return all(math.isfinite(x) and math.isfinite(y) for x, y in pts)


# ---------------------------------------------------------------------------
# geometry metrics (shape review / regression checks)
# ---------------------------------------------------------------------------

def layout_metrics(mol: Chem.Mol, ring: list[int] | None = None) -> dict:
    """Patent-style shape metrics of the placed macroring:
    circularity (radial std/mean; ->0 for circles), zigzag (sign
    alternation rate of successive bond turns), mean interior angle,
    bounding-box aspect, minimum non-bonded atom distance."""
    ring = ring or find_macroring(mol)
    conf = mol.GetConformer()
    pts = {i: conf.GetAtomPosition(i) for i in ring}
    n = len(ring)
    order = ordered_ring(mol, ring)
    cx = sum(p.x for p in pts.values()) / n
    cy = sum(p.y for p in pts.values()) / n
    ds = [math.hypot(p.x - cx, p.y - cy) for p in pts.values()]
    md = sum(ds) / n
    circularity = math.sqrt(sum((d - md) ** 2 for d in ds) / n) / md
    dirs = []
    for k in range(n):
        a, b = order[k], order[(k + 1) % n]
        vx, vy = pts[b].x - pts[a].x, pts[b].y - pts[a].y
        d = math.hypot(vx, vy) or 1e-9
        dirs.append((vx / d, vy / d))
    signs = []
    for k in range(n):
        cr = dirs[k - 1][0] * dirs[k][1] - dirs[k - 1][1] * dirs[k][0]
        signs.append(0 if abs(cr) < 1e-9 else (1 if cr > 0 else -1))
    turns = [k for k in range(n) if signs[k - 1] and signs[k]]
    flips = sum(1 for k in turns if signs[k - 1] != signs[k])
    angles = []
    for k in range(n):
        a, b, c = order[k - 1], order[k], order[(k + 1) % n]
        v1 = (pts[a].x - pts[b].x, pts[a].y - pts[b].y)
        v2 = (pts[c].x - pts[b].x, pts[c].y - pts[b].y)
        d1 = math.hypot(*v1) or 1e-9
        d2 = math.hypot(*v2) or 1e-9
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    xs = [p.x for p in pts.values()]
    ys = [p.y for p in pts.values()]
    aspect = (max(xs) - min(xs)) / max(1e-9, max(ys) - min(ys))
    bonded = [set() for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded[i].add(j)
        bonded[j].add(i)
    ids = list(pts)
    mind = float("inf")
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if j in bonded[i]:
                continue
            mind = min(mind, math.hypot(pts[i].x - pts[j].x,
                                        pts[i].y - pts[j].y))
    return {
        "ring_atoms": n,
        "circularity": round(circularity, 3),
        "zigzag": round(flips / len(turns), 2) if turns else 0.0,
        "mean_angle": round(sum(angles) / n, 1),
        "aspect": round(aspect, 2),
        "min_nonbond": round(mind, 2),
    }


def render_smiles(smiles: str, out_png: str, seed: int = 0,
                  upright: bool = False, seeds: int = 4) -> dict:
    """Lay out + render one SMILES to a patent-style PNG (ALIGNED profile)
    plus a CoordGen-layout control render of the same molecule — the shape
    review pair used for the round-4 zig-zag validation.  Layout seeds are
    retried until QA passes.  With upright=True the style pipeline's
    rotation is cancelled so the ring shape reads upright.  Returns layout
    metrics + PNG paths."""
    import os

    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES failed to parse: {smiles[:80]}")
    laid = None
    used_seed = seed
    for k in range(seeds):
        s = seed + k
        laid = zigzag_coords(mol, random.Random(s), seed=s)
        if laid is not None:
            used_seed = s
            break
    if laid is None:
        raise ValueError("zigzag layout failed for this molecule")
    metrics = layout_metrics(laid)

    from style_calibration import ALIGNED, coordgen_coords, draw_molecule

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    np.random.seed(used_seed)       # ink noise of the ALIGNED profile
    if upright:
        # draw_molecule rotates by the FIRST rng draw it takes; feed it a
        # fresh rng whose first draw we know, and pre-rotate by minus that
        # angle so the net rotation is zero and the ring reads upright
        rng_draw = random.Random(used_seed + 101)
        angle = rng_draw.uniform(0, 360)
        rad = math.radians(-angle)
        conf = laid.GetConformer()
        for i in range(laid.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(
                p.x * math.cos(rad) - p.y * math.sin(rad),
                p.x * math.sin(rad) + p.y * math.cos(rad), 0.0))
        img = draw_molecule(laid, ALIGNED, rng_draw)
    else:
        img = draw_molecule(laid, ALIGNED, random.Random(used_seed + 101))
    img.save(out_png)
    metrics["png"] = out_png

    base, ext = os.path.splitext(out_png)
    cg_path = f"{base}_coordgen{ext}"
    cg = coordgen_coords(Chem.Mol(mol))
    if cg is not None:
        rng2 = random.Random(used_seed + 7)
        np.random.seed(used_seed + 7)
        draw_molecule(cg, ALIGNED, rng2).save(cg_path)
        metrics["png_coordgen"] = cg_path
    return metrics


if __name__ == "__main__":  # pragma: no cover - shape review CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(description="render a macroring SMILES "
                                 "with the zig-zag template layout")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upright", action="store_true",
                    help="cancel the ALIGNED rotation for shape review")
    args = ap.parse_args()
    print(json.dumps(render_smiles(args.smiles, args.out, seed=args.seed,
                                   upright=args.upright), indent=1))

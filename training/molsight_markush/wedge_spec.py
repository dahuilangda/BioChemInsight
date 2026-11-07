"""Patent-standard wedge rendering: slim glyph geometry + collision-aware
wedge-bond choice.

The renderer's auto-wedging (RDKit WedgeMolBonds inside
PrepareAndDrawMolecule) draws fat triangles (base ~0.3 bond) and picks
the wedge bond by internal heuristics: on the r7 seq68 render it wedged
a piperidine RING bond and pointed glyphs into neighbouring NH/SH label
crowds -- the general defect class the user flagged.  Real patent crops
instead wedge the C_alpha->side-chain first bond with a slim triangle
(base ~0.20 bond) or a 5-6 stroke dash run, and never wedge ring bonds.

Two halves, both driven by one plan:
  * plan_wedge_marks -- per stereocentre enumerate candidate marks
    (bond x {wedge, dash}), validate each by editing the bond-stereo
    flags of the RDKit-written molblock and re-parsing (the writer's own
    marks are the known-good baseline; any single-mark re-placement that
    round-trips to the same SMILES is stereo-faithful), then choose the
    candidate with the least glyph collision (atom label discs, bond
    segments, drawn ring interiors) under the patent conventions.
  * draw_wedge_marks -- render the planned marks as slim antialiased
    glyphs on the PIL image (solid triangle tapering from the
    stereocentre, or perpendicular dash strokes growing along the bond);
    the molecule itself is drawn wedge-less so no fat RDKit glyph
    appears under the overlay.

Measured standard (3x zoom of US20240287138A1 crops): solid base
~0.20 x bond, length = 1 bond; dashes 5-6, stroke = line width, growing
from ~0.1 x base at the stereocentre to the full base at the far end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

WEDGE_BASE_FRAC = 0.20          # base width / bond length
DASH_COUNT = 6
DASH_MIN_FRAC = 0.10            # first stroke length / base
# collision weights (px^2-free soft scores, tuned on seq68 render)
PEN_LABEL = 30.0                # glyph point inside an atom label disc
PEN_BOND = 6.0                  # glyph point near another bond segment
PEN_RING = 40.0                 # glyph point inside a drawn ring
PEN_RING_BOND = 8.0             # structural: wedge ON a ring bond edge.
                                # Dominates moderate label/bond grazing
                                # (a real draftman wedges the exocyclic
                                # C_alpha bond whenever one exists) but
                                # loses to a deep label hit (30 x ~10)
BONUS_EXOCYCLIC = 3.0           # macroring -> side-chain first bond
NEAR_BOND_PX = 1.6              # "near a segment" band (bond lw ~2.6)
RING_EDGE_PX = 1.0              # glyph point within this of a polygon
                                # edge counts as interior (edge-hugging
                                # glyphs used to slip the inside test)


@dataclass
class WedgeMark:
    bond_idx: int
    begin: int                  # stereocentre (glyph apex)
    end: int                    # far atom (glyph base)
    kind: str                   # "wedge" | "dash"
    trim0: float = 0.0          # apex trim as a FRACTION of bond length:
                                # labelled atoms stop the glyph at the
                                # text edge, exactly how RDKit trims
                                # plain bond lines
    trim1: float = 0.0          # base trim fraction


def _label_text(a: Chem.Atom) -> str:
    """Visible text for a drawn atom label (carbon stays silent)."""
    if a.GetAtomicNum() == 6:
        return ""
    s = a.GetSymbol()
    n_h = a.GetTotalNumHs()
    if a.GetFormalCharge():
        s += {"1": "+", "-1": "-", "2": "2+", "-2": "2-"}.get(
            str(a.GetFormalCharge()), "?")
    if n_h:
        s += "H" + (str(n_h) if n_h > 1 else "")
    return s


def _label_trim(a: Chem.Atom, font_px: float, bond_len_px: float
                ) -> float:
    """How far a glyph must stop short of a labelled atom (RDKit trims
    plain bonds at the text boundary; the overlay glyphs must do the
    same or they print through the label).  Rough text-half-width."""
    text = _label_text(a)
    if not text:
        return 0.0
    pad = 0.30 * font_px * len(text) + 1.5
    return min(pad, 0.45 * bond_len_px)


def _disc_radius(text: str, font_px: float) -> float:
    if not text:
        return 0.0
    return 0.45 * font_px * (0.75 + 0.45 * len(text))


def _tri_points(p0: np.ndarray, p1: np.ndarray, base: float,
                kind: str, trim0: float = 0.0, trim1: float = 0.0,
                n_samp: int = 14) -> np.ndarray:
    """Sample points covering the DRAWN glyph footprint: a filled
    triangle (wedge) or the dash strokes (dash), apex at p0, endpoints
    trimmed at labelled atoms (the text boundary)."""
    axis = p1 - p0
    length = float(np.hypot(*axis)) or 1.0
    u = axis / length
    perp = np.array([-u[1], u[0]])
    t_lo = min(max(trim0, 0.0), 0.45)
    t_hi = 1.0 - min(max(trim1, 0.0), 0.45)
    ts = np.linspace(t_lo, t_hi, n_samp)
    pts = []
    if kind == "wedge":
        for t in ts[1:]:
            w = 0.5 * base * (t - t_lo) / max(t_hi - t_lo, 1e-9)
            c = p0 + axis * t
            pts += [c + perp * w, c - perp * w]
        pts.append(p0 + axis * (t_lo + 0.5 * (t_hi - t_lo)))
    else:
        for k in range(DASH_COUNT):
            t = t_lo + (k + 0.5) / DASH_COUNT * (t_hi - t_lo)
            w = 0.5 * base * (DASH_MIN_FRAC
                              + (1.0 - DASH_MIN_FRAC)
                              * ((t - t_lo) / max(t_hi - t_lo, 1e-9)))
            c = p0 + axis * t
            steps = max(2, int(w / 1.2) + 1)
            for s in np.linspace(-w, w, steps):
                pts.append(c + perp * s)
    return np.array(pts)


def _apply_flag_edits(ls: list[str], n_atoms: int,
                      edits: dict[int, int]) -> list[str]:
    """Return bond-stereo flag replacements applied to a molblock LINE
    list (edits {bond_block_line: flag}, 0 clears)."""
    out = list(ls)
    for k, flag in edits.items():
        line = out[4 + n_atoms + k]
        out[4 + n_atoms + k] = line[:9] + f"{flag:3d}" + line[12:]
    return out


def plan_wedge_marks(mol: Chem.Mol, font_px: float = 16.0,
                     page_scale: float = 1.0
                     ) -> tuple[list[WedgeMark], dict]:
    """Choose the wedge mark per stereocentre.  Returns (marks, report);
    centres with no valid single-mark candidate fall back to the
    writer's own bond (flagged in the report).  `page_scale` converts
    the conformer's LAYOUT units to output pixels: the collision
    constants (disc radii, near-bond band, trims) live in page pixels,
    so the positions are scaled once here."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    try:
        marks, report = _plan_wedge_marks_inner(mol, font_px, page_scale)
    finally:
        RDLogger.EnableLog("rdApp.warning")
    return marks, report


def _plan_wedge_marks_inner(mol: Chem.Mol, font_px: float,
                            page_scale: float
                            ) -> tuple[list[WedgeMark], dict]:
    centers = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    if not centers:
        return [], {"n_centers": 0}
    try:
        orig_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        mb = Chem.MolToMolBlock(mol, includeStereo=True)
        mb_lines = mb.splitlines()
        n_atoms = int(mb_lines[3][0:3])
        n_bonds = int(mb_lines[3][3:6])
    except Exception:
        return [], {"error": "molblock", "n_centers": len(centers)}

    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                    for i in range(mol.GetNumAtoms())]) * page_scale
    # geometry helpers over the drawn structure (positions now in PAGE px,
    # matching the px constants below)
    ring_sets = [set(r) for r in mol.GetRingInfo().AtomRings()]
    exocyclic_pairs: set[tuple[int, int]] = set()
    biggest = max(ring_sets, key=len) if ring_sets else set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i in biggest) != (j in biggest):
            exocyclic_pairs.add(tuple(sorted((i, j))))
    discs = [(pos[a.GetIdx()], _disc_radius(_label_text(a), font_px))
             for a in mol.GetAtoms() if _label_text(a)]
    bond_segs = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_segs.append((i, j, pos[i], pos[j]))
    # greedy order: macroring C_alpha centres first (their exocyclic
    # side-chain bond is the standard's wedge-of-choice; deciding them
    # before crowded side-chain centres keeps the best candidates free)
    centers.sort(key=lambda c: (
        0 if any(tuple(sorted((c, n.GetIdx()))) in exocyclic_pairs
                 for n in mol.GetAtomWithIdx(c).GetNeighbors()) else 1,
        c))

    mb_bond_line = {}
    for k in range(n_bonds):
        line = mb_lines[4 + n_atoms + k]
        a1, a2 = int(line[0:3]) - 1, int(line[3:6]) - 1
        mb_bond_line[(min(a1, a2), max(a1, a2))] = k

    def collision_score(center: int, mark: WedgeMark) -> float:
        p0, p1 = pos[mark.begin], pos[mark.end]
        bond_px = float(np.hypot(*(p1 - p0))) or 1.0
        base = WEDGE_BASE_FRAC * bond_px
        pts = _tri_points(p0, p1, base, mark.kind, mark.trim0, mark.trim1)
        score = 0.0
        # already-chosen wedge glyphs are obstacles: two neighbouring
        # stereocentres' solid wedges once overlapped into one fat blob
        # (the seq53_gold bridgehead apex)
        for fp in chosen_footprints:
            for q in fp[::3]:
                d = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
                score += PEN_BOND * float((d < 1.6).sum())

        # atom label discs (skip the two endpoints: glyph lives between)
        for c, r in discs:
            if r <= 0:
                continue
            d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            score += PEN_LABEL * float((d < r + 0.5).sum())
        # other bond segments
        for i, j, pi, pj in bond_segs:
            if {i, j} == {mark.begin, mark.end}:
                continue
            if mark.begin in (i, j) and center in (i, j):
                continue          # sibling bonds at the stereocentre
            ab = pj - pi
            denom = max(float(ab @ ab), 1e-9)
            ap = pts - pi
            t = np.clip((ap @ ab) / denom, 0.0, 1.0)
            proj = pi + np.outer(t, ab)
            d = np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1])
            score += PEN_BOND * float((d < NEAR_BOND_PX).sum())
        # drawn ring interiors: a wedge slicing a ring is the q3 defect;
        # the edge band catches glyphs hugging the polygon boundary
        for rs in ring_sets:
            if len(rs) < 3 or not rs <= set(range(len(pos))):
                continue
            poly = pos[list(rs)]
            for p in pts[::4]:
                if _point_near_poly(p, poly, RING_EDGE_PX):
                    score += PEN_RING
                    break
        # conventions
        key = tuple(sorted((mark.begin, mark.end)))
        is_ring_bond = any(mark.begin in rs and mark.end in rs
                           for rs in ring_sets)
        if key in exocyclic_pairs and mark.begin in biggest:
            score -= BONUS_EXOCYCLIC
        if is_ring_bond:
            score += PEN_RING_BOND
        return score

    # Sequential conditional planning over ONE evolving molblock state:
    # a bond between two adjacent stereocentres may be marked by at most
    # ONE of them, and every later candidate validates against the marks
    # the earlier centres already chose (greedy per-centre validation
    # against the writer's original lines once produced contradictory
    # double marks -- the "bond wedging contradiction" RDKit warning).
    # Centres that keep no single-mark choice fall back to the writer's
    # own flags, which stay in the state untouched.
    lines_state = list(mb_lines)
    bond_line_of = {mol_bond.GetIdx(): mb_bond_line.get(
        tuple(sorted((mol_bond.GetBeginAtomIdx(),
                      mol_bond.GetEndAtomIdx()))))
        for mol_bond in mol.GetBonds()}
    bond_of_line = {k: b for b, k in bond_line_of.items()}
    # the writer's own marks per centre, from the ORIGINAL block: the
    # per-centre fallback base (a centre with no valid single-mark
    # candidate keeps exactly these)
    writer_flags: dict[int, list[tuple[int, int]]] = {}
    for k in range(n_bonds):
        flag = int(mb_lines[4 + n_atoms + k][9:12])
        if flag not in (1, 6):
            continue
        a1, a2 = int(mb_lines[4 + n_atoms + k][0:3]) - 1, \
            int(mb_lines[4 + n_atoms + k][3:6]) - 1
        for aid in (a1, a2):
            if aid in centers:
                writer_flags.setdefault(aid, []).append((k, flag))
    used_bonds: set[int] = set()

    def state_flags() -> dict[int, int]:
        return {kk: int(lines_state[4 + n_atoms + kk][9:12])
                for kk in range(n_bonds)
                if int(lines_state[4 + n_atoms + kk][9:12]) in (1, 6)}

    def centre_flag_bonds(center: int) -> list[int]:
        out = []
        for b in mol.GetAtomWithIdx(center).GetBonds():
            k = bond_line_of.get(b.GetIdx())
            if k is not None and k in state_flags():
                out.append(k)
        return out

    def candidate_edits(center: int, mark: WedgeMark) -> dict[int, int] \
            | None:
        k = bond_line_of.get(mark.bond_idx)
        if k is None or k in used_bonds:
            return None
        edits: dict[int, int] = {kk: 0
                                 for kk in centre_flag_bonds(center)
                                 if kk not in used_bonds}
        edits[k] = 1 if mark.kind == "wedge" else 6
        return edits

    def candidate_valid(edits: dict[int, int]) -> bool:
        text = "\n".join(_apply_flag_edits(lines_state, n_atoms, edits))
        back = Chem.MolFromMolBlock(text, sanitize=True)
        if back is None:
            return False
        Chem.AssignStereochemistry(back, cleanIt=True, force=True)
        return Chem.MolToSmiles(back, isomericSmiles=True) == orig_smi

    marks: list[WedgeMark] = []
    report = {"n_centers": len(centers), "fallbacks": []}
    chosen_footprints: list[np.ndarray] = []

    def _with_trims(mark: WedgeMark) -> WedgeMark:
        pa, pb = pos[mark.begin], pos[mark.end]
        bl = float(np.hypot(*(pb - pa))) or 1.0
        mark.trim0 = min(_label_trim(mol.GetAtomWithIdx(mark.begin),
                                     font_px, bl) / bl, 0.45)
        mark.trim1 = min(_label_trim(mol.GetAtomWithIdx(mark.end),
                                     font_px, bl) / bl, 0.45)
        return mark

    for center in centers:
        atom = mol.GetAtomWithIdx(center)
        # score FIRST (parse-free), then validate in score order and stop
        # at the first valid candidate — identical selection, ~4x fewer
        # molblock parses.  A V2000 wedge flag defines stereo only at the
        # flagged line's FIRST atom: candidates whose flag line begins at
        # another atom can never round-trip, so they are skipped outright
        # (35% of pairs on the gold set).
        scored: list[tuple[float, int, str, WedgeMark, dict[int, int]]] = []
        for b in atom.GetBonds():
            k = bond_line_of.get(b.GetIdx())
            if k is None or k in used_bonds:
                continue
            line = lines_state[4 + n_atoms + k]
            if int(line[0:3]) - 1 != center:
                continue
            other = b.GetOtherAtomIdx(center)
            for kind in ("wedge", "dash"):
                mark = _with_trims(WedgeMark(b.GetIdx(), center, other,
                                             kind))
                edits = candidate_edits(center, mark)
                if edits is None:
                    continue
                scored.append((collision_score(center, mark),
                               b.GetIdx(), kind, mark, edits))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        chosen = None
        for _score, _bi, _kind, mark, edits in scored:
            if candidate_valid(edits):
                chosen = (mark, edits)
                break
        if chosen is not None:
            mark, best_edits = chosen
            marks.append(mark)
            chosen_footprints.append(_tri_points(
                pos[mark.begin], pos[mark.end],
                WEDGE_BASE_FRAC * (float(np.hypot(
                    *(pos[mark.end] - pos[mark.begin]))) or 1.0),
                mark.kind, mark.trim0, mark.trim1))
            lines_state = _apply_flag_edits(lines_state, n_atoms,
                                            best_edits)
            used_bonds.add(bond_line_of.get(mark.bond_idx))
            continue
        # fallback: the writer's flags stay in the state AND its marks
        # are drawn as our own slim glyphs -- dropping them entirely
        # would silently LOSE that centre's stereo in the rendered image
        report["fallbacks"].append(center)
        for k, flag in writer_flags.get(center, []):
            if k in used_bonds:
                continue          # a neighbour's chosen mark owns this bond
            used_bonds.add(k)
            bond = mol.GetBondWithIdx(bond_of_line[k])
            other = bond.GetOtherAtomIdx(center)
            kind = "wedge" if flag == 1 else "dash"
            fb_mark = _with_trims(WedgeMark(bond.GetIdx(), center,
                                            other, kind))
            marks.append(fb_mark)
            chosen_footprints.append(_tri_points(
                pos[fb_mark.begin], pos[fb_mark.end],
                WEDGE_BASE_FRAC * (float(np.hypot(
                    *(pos[fb_mark.end] - pos[fb_mark.begin]))) or 1.0),
                fb_mark.kind, fb_mark.trim0, fb_mark.trim1))
    # end-to-end proof: the combined flag map round-trips
    text = "\n".join(lines_state)
    combined_ok = False
    back = Chem.MolFromMolBlock(text, sanitize=True)
    if back is not None:
        Chem.AssignStereochemistry(back, cleanIt=True, force=True)
        combined_ok = (Chem.MolToSmiles(back, isomericSmiles=True)
                       == orig_smi)
    report["combined_valid"] = combined_ok
    if not combined_ok:
        # never render an unproven mark set — but the WRITER'S OWN state
        # always round-trips (it is what RDKit itself would draw), so
        # emit slim glyphs for the writer's marks instead of dropping the
        # molecule to the stock fat path
        marks = []
        for k in range(n_bonds):
            flag = int(mb_lines[4 + n_atoms + k][9:12])
            if flag not in (1, 6):
                continue
            line = mb_lines[4 + n_atoms + k]
            a1 = int(line[0:3]) - 1
            a2 = int(line[3:6]) - 1
            bond = mol.GetBondWithIdx(bond_of_line[k])
            kind = "wedge" if flag == 1 else "dash"
            begin = a1 if a1 in centers else a2
            marks.append(_with_trims(WedgeMark(bond.GetIdx(), begin,
                                               bond.GetOtherAtomIdx(begin),
                                               kind)))
        report["writer_fallback"] = True
    report["n_marks"] = len(marks)
    return marks, report


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = p
    inside = False
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        if (y2 > y) != (y1 > y):
            xint = (x1 - x2) * (y - y2) / ((y1 - y2) + 1e-30) + x2
            if x < xint:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _point_near_poly(p: np.ndarray, poly: np.ndarray,
                     band: float) -> bool:
    """Inside the polygon OR within `band` px of its boundary: the
    edge-hugging wedge glyphs a strict inside-test slips through."""
    if _point_in_poly(p, poly):
        return True
    x, y = p
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        abx, aby = x2 - x1, y2 - y1
        denom = max(abx * abx + aby * aby, 1e-12)
        t = ((x - x1) * abx + (y - y1) * aby) / denom
        t = min(max(t, 0.0), 1.0)
        dx = x - (x1 + abx * t)
        dy = y - (y1 + aby * t)
        if dx * dx + dy * dy < band * band:
            return True
        x1, y1 = x2, y2
    return False


def draw_wedge_marks(img, marks: list[WedgeMark], coords_fn,
                     line_px: float = 2.6, scale: int = 4) -> None:
    """Overlay the planned marks as slim antialiased glyphs (drawn at
    `scale`x supersampling, then minified).  `coords_fn(atom_idx)` maps
    an atom to its (x, y) draw position; `line_px` is the profile's bond
    line width so dash strokes match the plain bonds.

    A wedge/dash REPLACES its bond line in real drawings: the plain line
    under the glyph is erased first (leaving a line + dashes would print
    '+'-shaped overlaps), and endpoints are trimmed at labelled atoms
    the way RDKit trims plain bond lines."""
    from PIL import Image, ImageDraw

    if not marks:
        return
    big = img.resize((img.width * scale, img.height * scale),
                     Image.NEAREST)
    draw = ImageDraw.Draw(big)
    lw = max(1, int(round(line_px * scale)))
    erase_w = max(1, int(round((line_px + 2.0) * scale)))
    ink = 0
    boxes = []
    for m in marks:
        p0 = np.array(coords_fn(m.begin), dtype=float) * scale
        p1 = np.array(coords_fn(m.end), dtype=float) * scale
        axis = p1 - p0
        length = float(np.hypot(*axis)) or 1.0
        u = axis / length
        perp = np.array([-u[1], u[0]])
        # trim fractions are plan-space; they apply identically here
        t_lo = min(max(m.trim0, 0.0), 0.45)
        t_hi = 1.0 - min(max(m.trim1, 0.0), 0.45)
        # hide the plain bond under the glyph (tight: t in [0.02, 0.98]
        # keeps the atom vertices of NEIGHBOURING bonds intact)
        ea = p0 + axis * max(t_lo, 0.02)
        eb = p0 + axis * min(t_hi, 0.98)
        draw.line((ea[0], ea[1], eb[0], eb[1]), fill=255, width=erase_w)
        base = WEDGE_BASE_FRAC * length
        span = t_hi - t_lo
        half_span = base * span / 2.0
        if m.kind == "wedge":
            a0 = p0 + axis * t_lo
            b0 = p0 + axis * t_hi + perp * half_span
            b1 = p0 + axis * t_hi - perp * half_span
            draw.polygon([tuple(a0), tuple(b0), tuple(b1)], fill=ink)
        else:
            for k in range(DASH_COUNT):
                t = t_lo + (k + 0.5) / DASH_COUNT * span
                w = base * (DASH_MIN_FRAC + (1.0 - DASH_MIN_FRAC)
                            * ((t - t_lo) / max(span, 1e-9))) / 2.0
                c = p0 + axis * t
                draw.line((c[0] - perp[0] * w, c[1] - perp[1] * w,
                           c[0] + perp[0] * w, c[1] + perp[1] * w),
                          fill=ink, width=lw)
        # region where the overlay changed ink (glyph + erase zone)
        margin = erase_w + half_span + 4.0 * scale
        xs = [ea[0], eb[0]]
        ys = [ea[1], eb[1]]
        boxes.append((int(max(0, min(xs) - margin)),
                      int(max(0, min(ys) - margin)),
                      int(min(big.width, max(xs) + margin)),
                      int(min(big.height, max(ys) + margin))))
    small = big.resize(img.size, Image.LANCZOS).convert(img.mode)
    # paste only the changed regions: a whole-image paste would resample
    # every line (and min() compositing could never lighten the erase).
    # `small` has the same pixel size as `img`; box coords live on the
    # `scale`x-supersampled canvas and come back down by 1/scale
    for x0, y0, x1, y1 in boxes:
        bx0 = max(0, int(x0 / scale) - 1)
        by0 = max(0, int(y0 / scale) - 1)
        bx1 = min(img.width, int(x1 / scale) + 1)
        by1 = min(img.height, int(y1 / scale) + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        region = small.crop((bx0, by0, bx1, by1))
        img.paste(region, (bx0, by0))

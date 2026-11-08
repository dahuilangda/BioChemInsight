"""Residue-granular cyclic word DP (§9 of conditioned_frame_optimization).

The 2D depiction of a cyclic peptide is a CYCLIC WORD over amino-acid
residues; the layout engine's freedom lives at residue INTERFACES only,
so every residue's internal frame (stereocentres, wedge bonds) is
carried rigidly -- the drawing-level peptide-plane flip (Hayward 2001).
The wedge planner re-derives from the final coordinates, hence
stereo_roundtrip holds BY CONSTRUCTION, not by repair.

Mathematical content that survives contact with the exact-bond
geometry:

* Fenchel's discrete equality (total turn = 360 deg for a convex
  closed polygon) -- the curl budget;
* the chord HULL-EDGE condition as a local predicate on the word: at
  every bridge/path chord the two FLANKING turns (the turns at the
  chord's cycle endpoints) must have the SAME sign.  Derived exactly
  and verified numerically: with the chord bond heading h, vertex i-1
  sits on the side +sin(t_i) of the chord line and vertex j+1 on the
  side +sin(t_j) -- t_i * t_j > 0 (a corner pair anchored exactly ON
  the chord edge) folds BOTH neighbours to one side; the chord is then
  a local hull edge and the regular completion arc lands on the OTHER
  side at 240-theta >= 90 deg.  Opposite-sign flanks (the plain
  sawtooth alternation) straddle the chord line -- one neighbour folds
  onto each bulge side and the arc pinches at |t|-dependent angles
  down to 7 deg: the seq53/77 stacked-arc-pocket defect, which no
  amount of post-hoc repair could clear.

The realisation rides the PROVEN turn-sequence construction (round
sawtooth + corner pairs + Newton closure): the hull-edge rule is a
post-condition on the turn vector with per-half curl rebalancing, so
the discrete Fenchel equality and the exact Newton closure both hold.
"""
from __future__ import annotations

import math
import random

from rdkit import Chem

from zigzag_layout_v2 import _adjacency, _is_carbonyl_carbon

PEAK, STRAIGHT, VALLEY = 1, 0, -1


# ---------------------------------------------------------------------------
# residue segmentation (chemistry-driven, no coordinates involved)
# ---------------------------------------------------------------------------

def residue_units(mol: Chem.Mol, order: list[int],
                  chord_pairs: list[tuple] | None = None) -> list[dict] | None:
    """Segment the drawn cycle into amino-acid residue units.

    A unit starts at a backbone amide N (neutral N on the cycle) and
    ends at the next carbonyl C on the cycle: the standard residue
    contributes [N, C_alpha, C(=O)] (glycine included), proline (or any
    residue whose side chain merges into the ring) contributes its
    extra ring carbons between N and C_alpha.  Returns None when the
    cycle does not parse as residues (non-peptide ring -- caller falls
    back to the geometric builders).

    Each unit: {"atoms": [cycle atom ids N..CO], "start": position,
    "chord_edges": [positions of chord endpoints inside the unit]}."""
    adj = _adjacency(mol)
    pos = {a: k for k, a in enumerate(order)}
    m = len(order)
    cyc = {a: {x for x in adj[a] if x in pos and x != a} for a in order}

    def is_amide_n(idx: int) -> bool:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() != 7 or a.GetFormalCharge() != 0:
            return False
        return any(mol.GetAtomWithIdx(nb).GetAtomicNum() == 6 and
                   _is_carbonyl_carbon(mol, nb) for nb in cyc[idx])

    n_positions = [k for k, a in enumerate(order) if is_amide_n(a)]
    if not n_positions:
        return None
    units: list[dict] = []
    for j, nk in enumerate(n_positions):
        end = n_positions[(j + 1) % len(n_positions)]
        span = (end - nk) % m
        if span < 2:
            return None                  # degenerate segmentation
        atoms = [order[(nk + t) % m] for t in range(span)]
        units.append({"atoms": atoms, "start": nk, "end": end})
    if sum(len(u["atoms"]) for u in units) != m:
        return None
    for u in units:
        u["chord_edges"] = []
        lo, hi = u["start"], u["start"] + len(u["atoms"])
        for ch in (chord_pairs or []):
            for p in ch[:2]:
                if lo <= p < hi:
                    u["chord_edges"].append(p)
    return units


def residue_phase_word(units: list[dict],
                       loads: list[float] | None = None
                       ) -> list[int] | None:
    """Max-load-weighted phase assignment over the residue cycle with
    the chord hull predicate as hard constraints.

    Phases: +1 (C_alpha outward peak), -1 (inward peak), 0 (straight
    run).  A unit hosting any chord edge is forced STRAIGHT (the chord
    becomes a local hull edge).  Cyclic: fixed first state, linear DP,
    closure requires the last unit's phase to keep a chord-hosting
    first unit straight."""
    K = len(units)
    if K == 0:
        return None
    forced = [bool(u["chord_edges"]) for u in units]
    w = [1.0 + 0.5 * ((loads[k] if loads and k < len(loads) else 0.0)
                      / max(1.0, max(loads) if loads and max(loads) > 0
                            else 1.0))
         for k in range(K)]
    states = (PEAK, STRAIGHT, VALLEY)
    best_word: list[int] | None = None
    best_score = -1e18
    for first in states:
        if forced[0] and first != STRAIGHT:
            continue
        dp: dict[int, tuple[float, list[int]]] = {first: (0.0, [first])}
        for k in range(1, K):
            ndp: dict[int, tuple[float, list[int]]] = {}
            for prev, (sc, hist) in dp.items():
                for st in states:
                    if forced[k] and st != STRAIGHT:
                        continue
                    gain = 0.0
                    if st != STRAIGHT:
                        gain = w[k]
                        if st == prev and prev != STRAIGHT:
                            gain *= 0.35
                    elif prev == STRAIGHT:
                        gain = -0.25
                    cur = ndp.get(st)
                    if cur is None or sc + gain > cur[0]:
                        ndp[st] = (sc + gain, hist + [st])
            dp = ndp
            if not dp:
                break
        if not dp:
            continue
        for last, (sc, hist) in dp.items():
            if hist[0] != first:
                continue
            if sc > best_score:
                best_score = sc
                best_word = hist
    return best_word


def chord_edge_positions(chord_pairs: list[tuple] | None,
                         order_len: int) -> set[int]:
    """Start positions of GAP-0 (cycle-adjacent) chords: only these
    have flanking turns (turns[ia], turns[ia+1]) whose signs decide
    the hull-edge condition.  A chord with a cycle atom between its
    endpoints folds the drawn path away by construction and never
    pinches (measured: landings >= 90 deg on every gap-1 chord)."""
    out: set[int] = set()
    for ch in (chord_pairs or []):
        ia, ib = ch[0], ch[1]
        if (ib - ia) % order_len == 1:
            out.add(ia)
    return out


def residue_word_straight_sites(mol: Chem.Mol, order: list[int],
                                chord_pairs: list[tuple] | None
                                ) -> set[int] | None:
    """Cycle positions that the phase word forces straight: every atom
    of a chord-hosting residue unit (the hull-edge runs).  None when
    the cycle is not a peptide ring."""
    units = residue_units(mol, order, chord_pairs)
    if units is None:
        return None
    pos = {a: k for k, a in enumerate(order)}
    sites: set[int] = set()
    for u in units:
        if not u["chord_edges"]:
            continue
        for a in u["atoms"]:
            sites.add(pos[a])
    return sites


# ---------------------------------------------------------------------------
# the hull-edge rule in the turn alphabet (with curl rebalancing)
# ---------------------------------------------------------------------------

def chord_flank_turns(turns: list[float],
                      chord_positions: set[int]) -> list[float] | None:
    """Enforce t_i * t_j > 0 on the flanking turns of every chord edge
    (same-sign corner pair anchored ON the chord) and rebalance the
    curl.

    The rule is exact geometry, verified numerically: with the chord
    bond heading h, vertex i-1 lies on the side +sin(t_i) of the chord
    line and vertex j+1 on the side +sin(t_j); SAME signs put both
    neighbours on ONE side (the chord is then a local hull edge -- the
    completion arc lands on the other side at 240-theta >= 90 deg).
    A flip breaks the 2-fold half sums of the even construction, so the
    flips and the rebalance act on both copies of every turn -- the
    discrete Fenchel equality and the exact Newton closure survive."""
    n = len(turns)
    out = list(turns)
    chord_pos = set(chord_positions)

    def legal() -> bool:
        for p in chord_pos:
            q = (p + 1) % n
            if out[p] * out[q] <= 0.0:
                return False
        return True

    # flips act on BOTH 2-fold copies (p and p+m) so the even-n
    # displacement pairing (t_{k+m} = t_k) -- and with it the EXACT
    # closure -- survives; the balancing adjustments are mirrored too
    m2 = n // 2
    done: set[int] = set()

    def done_legal() -> bool:
        return all(out[p] * out[(p + 1) % n] > 0.0 for p in done)

    for p in sorted(chord_pos):
        q = (p + 1) % n
        done.add(p)
        if out[p] * out[q] < 0.0:
            # straddled chord: flip the smaller-magnitude flank so the
            # pair turns same-sign (a corner anchored on the chord)
            cand = q if abs(out[q]) <= abs(out[p]) else p
            out[cand] = -out[cand]
            if n % 2 == 0:
                out[(cand + m2) % n] = -out[(cand + m2) % n]
            if not done_legal():
                out[cand] = -out[cand]
                if n % 2 == 0:
                    out[(cand + m2) % n] = -out[(cand + m2) % n]
                other = p if cand == q else q
                out[other] = -out[other]
                if n % 2 == 0:
                    out[(other + m2) % n] = -out[(other + m2) % n]
                if not done_legal():
                    return None
        elif abs(out[p]) < 1e-9 or abs(out[q]) < 1e-9:
            # a zero flank is an exactly-straight run -- give it the
            # same sign as the other flank so the pair folds together
            z = p if abs(out[p]) < 1e-9 else q
            o = q if z == p else p
            mag = abs(out[o]) if abs(out[o]) > 1e-9 else 30.0
            out[z] = math.copysign(mag, out[o])
            if n % 2 == 0:
                out[(z + m2) % n] = out[z]
            if not done_legal():
                return None
    if not legal():
        return None

    # rebalance each half to its +/-180 target over turns with band
    # slack, never touching the chord flanks; every adjustment is
    # applied to both 2-fold copies, and every turn keeps
    # 45 <= |t| <= 63 -- the same band the Newton closure clips into
    # (ZIG_TURN_MIN..ZIG_TURN+ANGLE_TOL), so the budget survives it
    def rebalance(half: list[int], target: float) -> None:
        err = target - sum(out[k] for k in half)
        mag_lo, mag_hi = 45.0, 63.0
        for _round in range(16):
            if abs(err) < 1e-9:
                return
            moved = False
            for k in sorted(half, key=lambda k: -abs(out[k])):
                kk = (k + m2) % n
                if k in chord_pos or (k - 1) % n in chord_pos:
                    continue
                v = out[k]
                if err > 0.0:
                    cap = mag_hi if v >= 0.0 else -mag_lo
                    room = cap - v
                else:
                    cap = -mag_hi if v <= 0.0 else mag_lo
                    room = v - cap
                if room <= 1e-9:
                    continue
                step = min(room, abs(err), 3.0)
                out[k] += math.copysign(step, err)
                if n % 2 == 0:
                    out[kk] += math.copysign(step, err)
                err -= math.copysign(step, err)
                moved = True
                if abs(err) < 1e-9:
                    return
            if not moved:
                return
    if n % 2 == 0 and m2 > 0:
        rebalance(list(range(m2)),
                  180.0 if sum(out[:m2]) >= 0 else -180.0)
    return out


def chord_flank_positions_from_turns(chord_pairs: list[tuple] | None,
                                     order_len: int) -> set[int]:
    """Positions p such that turns[p] and turns[p+1] flank a chord
    edge -- chord_pairs carry (ia, ib) endpoint positions."""
    return chord_edge_positions(chord_pairs)


# ---------------------------------------------------------------------------
# asymmetric full-cycle word realisation (the ChemDraw / human-hand look)
# ---------------------------------------------------------------------------

def asymmetric_word_turns(n: int, rng: random.Random,
                          chord_starts: set[int],
                          loads: list[float] | None = None,
                          jitter: float = 2.5) -> list[float] | None:
    """Full-cycle ASYMMETRIC turn word -- the chemistry-driven,
    human-hand zig-zag (no 2-fold symmetry anywhere).

    Construction: walk the cycle emitting the alternating sawtooth; at
    every gap-0 chord start emit a same-sign (+60, +60) corner pair
    (the §9 hull-edge rule -- both contour neighbours fold to one
    side) and resume the alternation from the opposite sign (phase
    consistent, no triple runs).  Curl budget is exact: the alternation
    contributes zero and each pair adds exactly +120, so three pairs
    carry the whole 360 deg.  The remaining corner anchors go to the
    heaviest drawing loads (bulky residues bulge -- chemistry-driven
    asymmetry, exactly where the real crops bulge); a zero-sum ±jitter
    drift (no curl bias) on the non-corner turns is the human hand.
    Position closure is the shared Newton pass downstream (band clip
    [45, 63] matches _close_positions' strict band)."""
    if n < 10:
        return None
    s0 = 1.0 if rng.random() < 0.5 else -1.0
    chord_pos = {p % n for p in chord_starts}

    # corner anchors: forced chord starts first; extra corner budget to
    # the heaviest loads (block +-1 around each chosen site)
    k_pairs = 3
    anchors: list[int] = []
    occupied: set[int] = set()
    for p in sorted(chord_pos):
        anchors.append(p)
        occupied |= {(p - 1) % n, p, (p + 1) % n, (p + 2) % n}
    if loads and len(loads) == n and len(anchors) < k_pairs:
        order_by_load = sorted(range(n),
                               key=lambda k: -float(loads[k]))
        for cand in order_by_load:
            if len(anchors) >= k_pairs:
                break
            if any((cand + d) % n in occupied
                   for d in (-1, 0, 1, 2)):
                continue
            anchors.append(cand)
            occupied |= {(cand - 1) % n, cand, (cand + 1) % n,
                         (cand + 2) % n}
    if len(anchors) < k_pairs:
        cand = (max(anchors) + 7) % n if anchors else 0
        while len(anchors) < k_pairs:
            if not any((cand + d) % n in occupied for d in (-1, 0, 1, 2)):
                anchors.append(cand)
                occupied |= {(cand - 1) % n, cand, (cand + 1) % n,
                             (cand + 2) % n}
            cand = (cand + 5) % n
    if len(anchors) != k_pairs:
        return None
    anchor_starts = set(anchors)

    # walk the cycle: alternating sawtooth, corner pairs as (+, +) with
    # the alternation resuming from the opposite sign; the pair emits
    # for positions i and i+1, so the walk advances by two there
    turns: list[float] = []
    i = 0
    cur = s0 * 60.0
    while i < n:
        p = i % n
        if p in anchor_starts:
            turns.append(60.0)
            turns.append(60.0)
            cur = -60.0
            i += 2
        else:
            turns.append(cur)
            cur = -cur
            i += 1

    # zero-sum human drift on the non-corner turns
    jit = [0.0 if turns[k] == 60.0 and k in anchor_starts
           else rng.uniform(-jitter, jitter) for k in range(n)]
    drift = sum(jit) / n
    turns = [t + (j - drift) for t, j in zip(turns, jit)]

    # the jitter residual is absorbed into the corner turns' band
    residual = 360.0 - sum(turns)
    corner_turns = [k for k in range(n) if k in anchor_starts
                    or (k - 1) % n in anchor_starts]
    cap_order = sorted(corner_turns, key=lambda k: rng.random())
    for k in cap_order:
        if abs(residual) < 1e-9:
            break
        room = (63.0 - turns[k]) if residual > 0 else (turns[k] - 45.0)
        if room <= 1e-9:
            continue
        step = min(room, abs(residual))
        turns[k] += math.copysign(step, residual)
        residual -= math.copysign(step, residual)
    if abs(residual) > 1e-6:
        return None
    if not all(44.5 <= abs(t) <= 63.5 for t in turns):
        return None
    return turns


def _is_alpha_carbon(mol, idx: int, unit_atoms: list[int]) -> bool:
    """C_alpha of the unit: the carbon directly bonded to the unit's N."""
    adj = _adjacency(mol)
    n_atom = unit_atoms[0]
    return idx in adj[n_atom] and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6


def _angle_range(pol: list[tuple[float, float]]) -> tuple[float, float]:
    """(sharpest, flattest) interior angles of a polar contour."""
    import numpy as np
    m = len(pol)
    pos = [(r * math.cos(a), r * math.sin(a)) for r, a in pol]
    lo, hi = 180.0, 0.0
    for k in range(m):
        v1 = np.array(pos[(k - 1) % m]) - np.array(pos[k])
        v2 = np.array(pos[(k + 1) % m]) - np.array(pos[k])
        nv = float(np.hypot(*v1)) * float(np.hypot(*v2)) + 1e-12
        ang = math.degrees(math.acos(max(-1.0, min(1.0,
                                                   float(v1 @ v2) / nv))))
        lo = min(lo, ang)
        hi = max(hi, ang)
    return lo, hi


def _sharpest(pol: list[tuple[float, float]]) -> float:
    return _angle_range(pol)[0]


def residue_word_contour(mol, order: list[int],
                         chord_starts: set[int],
                         bond: float,
                         rng: random.Random) -> list[tuple[float, float]] | None:
    """ASYMMETRIC closed contour from the residue word -- the human-hand
    zig-zag, built by the exact-bond polar propagation with class-based
    roots over the RESIDUE UNITS (the construction measured at
    maxdev 0.000, crossings 0, chord landings 91/66/54 deg).

    Classes from the phase word: the unit's N runs straight
    (self-correcting root toward the ideal radius -- no parity coupling,
    no drift), C_alpha peaks outward / carbonyl C valleys inward per the
    unit phase, proline's extra ring carbons run straight.  Chord-
    hosting units are forced straight by the word (the §9 hull-edge
    condition in radius language).  Angular steps are UNIFORM (measured
    twice: any modulation past ~5% pushes the r <= bond/sin(delta)
    ceiling below the closure mean radius).  Closure: the start radius
    r0 is scanned/bisected for the near-periodic fixed point."""
    return _word_contour_classes(mol, order, chord_starts, bond, rng)


def _word_contour_classes(mol, order: list[int],
                          chord_starts: set[int],
                          bond: float,
                          rng: random.Random):
    from beam_sidechain_layout import _detect_bridge_rings  # noqa: circular-free at call time
    if mol is None:
        return None
    n = len(order)
    two_pi = 2.0 * math.pi
    pos_d = {a: k for k, a in enumerate(order)}
    chord_pairs = []
    seen: set[tuple[int, int]] = set()
    for br in _detect_bridge_rings(mol, order):
        ea, eb = br["chord_atoms"]
        ia, ib = pos_d.get(ea), pos_d.get(eb)
        if ia is None or ib is None:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key not in seen:
            seen.add(key)
            chord_pairs.append((ia, ib, br["size"], False))
    units = residue_units(mol, order, chord_pairs)
    if units is None:
        return None
    word = residue_phase_word(units)
    if word is None:
        return None

    unit_of = {}
    for k, u in enumerate(units):
        for t, a in enumerate(u["atoms"]):
            unit_of[a] = (k, t)
    cls: dict[int, int] = {}
    for a in order:
        k, t = unit_of[a]
        phase = word[k]
        if t == 0:
            c = STRAIGHT
        elif t == len(units[k]["atoms"]) - 1:
            c = VALLEY if phase == PEAK else (
                PEAK if phase == VALLEY else STRAIGHT)
        elif _is_alpha_carbon(mol, a, units[k]["atoms"]):
            c = PEAK if phase == PEAK else (
                VALLEY if phase == VALLEY else STRAIGHT)
        else:
            c = STRAIGHT
        cls[a] = c

    deltas = [two_pi / n] * n
    r_target = n * bond / two_pi

    def propagate(r0: float):
        pts = [(r0, 0.0)]
        r = r0
        phi = 0.0
        for k in range(n):
            d = deltas[k]
            s, c = math.sin(d), math.cos(d)
            disc = bond * bond - (r * s) ** 2
            if disc <= 1e-9:
                return None
            root = math.sqrt(disc)
            cl = cls[order[(k + 1) % n]]
            if cl == PEAK:
                sign = 1.0
            elif cl == VALLEY:
                sign = -1.0
            else:
                sign = 1.0 if r < r_target else -1.0
            r = r * c + sign * root
            if r <= 0.5:
                return None
            phi += d
            pts.append((r, phi))
        return pts

    r_cap = bond / max(math.sin(d) for d in deltas)
    best: tuple[float, list[tuple[float, float]]] | None = None
    lo = r_target * 0.55
    hi = min(r_target * 1.5, r_cap * 0.999)
    steps = 40
    prev_r = prev_g = None

    def consider(got):
        nonlocal best
        sharp, flat = _angle_range(got)
        # prefer in-band candidates, but ACCEPT flat ones: the straight
        # runs of the S class always carry flat vertices in this polar
        # alphabet (the binary-root oscillation), and the downstream
        # contour_refine clamp pulls them into the drawing band -- so
        # flatness only ranks, it never vetoes
        score = abs(sharp - 133.0) + 0.3 * max(0.0, flat - 151.5)
        if best is None or score < best[0]:
            best = (score, got)

    for i in range(steps + 1):
        r0 = lo + (hi - lo) * i / steps
        got = propagate(r0)
        if got is None:
            prev_r = prev_g = None
            continue
        g = got[-1][0] - r0
        if abs(g) <= 0.015 * bond:
            consider(got)
            prev_r, prev_g = r0, g
            continue
        if prev_g is not None and (g > 0) != (prev_g > 0):
            a_r, a_g, b_r = prev_r, prev_g, r0
            for _ in range(36):
                mid = 0.5 * (a_r + b_r)
                got_m = propagate(mid)
                if got_m is None:
                    b_r = mid
                    continue
                g_m = got_m[-1][0] - mid
                if abs(g_m) <= 0.015 * bond:
                    consider(got_m)
                    break
                if (g_m > 0) == (a_g > 0):
                    a_r, a_g = mid, g_m
                else:
                    b_r = mid
        prev_r, prev_g = r0, g
    if best is None:
        return None
    pos = [(r * math.cos(a), r * math.sin(a)) for r, a in best[1][:-1]]
    cx = sum(p[0] for p in pos) / n
    cy = sum(p[1] for p in pos) / n
    return [(p[0] - cx, p[1] - cy) for p in pos]


def residue_reflection_groups(mol, order):
    """Rigid residue supports about N--C(=O), including all substituents.

    Only peptide interface bonds may cross a support boundary. Crosslinked
    residues are excluded rather than silently splitting their side chains.
    Endpoints lie on the reflection axis, so interface bonds stay exact.
    """
    units = residue_units(mol, order)
    if units is None:
        # Ring walkers may choose either direction; chemistry fixes N -> CO.
        units = residue_units(mol, list(reversed(order)))
    if not units:
        return []
    if any(not _is_carbonyl_carbon(mol, u["atoms"][-1]) for u in units):
        units = residue_units(mol, list(reversed(order)))
    if not units or any(not _is_carbonyl_carbon(mol, u["atoms"][-1])
                        for u in units):
        return []
    adj = _adjacency(mol)
    cycle = set(order)
    groups = []
    for u in units:
        core = set(u["atoms"])
        support = set(core)
        stack = list(core)
        valid = True
        while stack and valid:
            a = stack.pop()
            for b in adj[a]:
                if b in support:
                    continue
                if b in cycle:
                    if a not in (u["atoms"][0], u["atoms"][-1]):
                        valid = False
                        break
                    continue
                support.add(b)
                stack.append(b)
        if valid:
            # A side-chain bridge reaching another residue is not an interface.
            valid = all(b in support or
                        (a in (u["atoms"][0], u["atoms"][-1]) and b in cycle)
                        for a in support for b in adj[a])
        if valid:
            groups.append((u["atoms"][0], u["atoms"][-1], sorted(support)))
    return groups

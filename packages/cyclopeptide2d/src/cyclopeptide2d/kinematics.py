"""Generate initial residue poses from connection directions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .solver import PoseProblem


@dataclass
class PortSeed:
    score: float
    poses: np.ndarray
    signs: tuple[int, ...]


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _ports(problem: PoseProblem, residue: int, sign: int, adjacency):
    frame = problem.frames[residue]
    local = problem.local * [1, sign]
    ns = [a for a in adjacency[frame.n_atom] if problem.owner[a] == residue]
    cs = [a for a in adjacency[frame.c_atom] if problem.owner[a] == residue]
    vectors = local[cs] - local[frame.c_atom]
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    output = -vectors.sum(axis=0)
    outgoing = math.atan2(output[1], output[0])
    if not problem.interface_atoms[frame.n_atom]:
        incoming = ()
    elif len(ns) == 1:
        ray = local[ns[0]] - local[frame.n_atom]
        angle = math.acos(np.clip(frame.port_cosines[(frame.n_atom, ns[0])], -1, 1))
        base = math.atan2(ray[1], ray[0])
        incoming = (base + angle, base - angle)
    else:
        vectors = local[ns] - local[frame.n_atom]
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        incoming_vector = -vectors.sum(axis=0)
        incoming = (math.atan2(incoming_vector[1], incoming_vector[0]),)
    return incoming, outgoing


def _select_diverse(candidates, count):
    """Select candidates by score and differences in reflection signs and poses."""
    if len(candidates) <= count:
        return candidates
    pool = candidates[: max(count * 4, 16)]
    chosen = list(range(max(1, count // 2)))
    while len(chosen) < count:
        best_index, best_key = None, None
        for index, cand in enumerate(pool):
            if index in chosen:
                continue
            signs = sum(1 for c in chosen if pool[c].signs != cand.signs)
            poses = min(float(np.mean(np.abs(cand.poses - pool[c].poses))) for c in chosen)
            key = (signs, poses, -cand.score)
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        chosen.append(best_index)
    return [pool[i] for i in chosen]


def _segment_penalties(new_segments, placed_segments, margin=0.02, weight=500.0):
    """Crossing and clearance cost of new bonds against placed bonds.

    The seed loop's atom-overlap term cannot see an X crossing: two bonds
    may cross with every atom far from the other bond. Returns the summed
    penalty for one extension.
    """
    if not len(new_segments) or not len(placed_segments):
        return 0.0
    a = new_segments[:, 0]
    b = new_segments[:, 1]
    c = placed_segments[:, 0]
    d = placed_segments[:, 1]
    # bounding-box prefilter: only box-overlapping pairs reach the
    # distance machinery, which keeps the per-extension cost near the
    # atom-overlap term it accompanies
    new_lo = np.minimum(a, b).min(axis=0) - margin
    new_hi = np.maximum(a, b).max(axis=0) + margin
    placed_lo = np.minimum(c, d)
    placed_hi = np.maximum(c, d)
    near = (placed_lo <= new_hi).all(axis=1) & (placed_hi >= new_lo).all(axis=1)
    if not near.any():
        return 0.0
    c, d = c[near], d[near]
    a, b = a[:, None, :], b[:, None, :]
    c, d = c[None, :, :], d[None, :, :]

    def cross(v, w):
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]

    ab = b - a
    cd = d - c
    ac = c - a
    c1 = cross(ab, ac)
    c2 = cross(ab, d - a)
    c3 = cross(cd, -ac)
    c4 = cross(cd, b - c)
    crossing = ((c1 > 0) != (c2 > 0)) & ((c3 > 0) != (c4 > 0))

    # vertex-to-segment distances from both ends of every new bond
    def point_segment(p, u, v):
        vv = np.sum(v * v, axis=-1)
        t = np.clip(np.sum((p - u) * v, axis=-1) / np.maximum(vv, 1e-12), 0, 1)
        delta = p - (u + t[..., None] * v)
        return np.linalg.norm(delta, axis=-1)

    d_a = point_segment(a, c, cd)
    d_b = point_segment(b, c, cd)
    d_c = point_segment(c, a, ab)
    d_d = point_segment(d, a, ab)
    distance = np.minimum(np.minimum(d_a, d_b), np.minimum(d_c, d_d))
    distance = np.where(crossing, 0.0, distance)
    gap = np.maximum(margin - distance, 0)
    return weight * float(np.sum(gap**2))


def port_seeds(
    problem: PoseProblem,
    *,
    width: int = 512,
    count: int = 8,
    overlap: float = 0.88,
    diverse: bool = True,
    deadline: float | None = None,
    crossing_penalty: float = 0.0,
) -> list[PortSeed]:
    """Search residue connection directions and propose initial poses.

    Levels stop at the deadline and the surviving beam is used as is: a
    partial beam still beats an abandoned layout."""
    if width < 1 or count < 1 or overlap <= 0:
        raise ValueError("Beam width, seed count and overlap threshold must be positive")
    frames, native = problem.frames, problem.local
    adjacency = [[] for _ in range(problem.n)]
    for a, b in problem.bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    ports = {
        (k, sign): _ports(problem, k, sign, adjacency) for k in range(problem.k) for sign in (1, -1)
    }
    target = problem.initial.reshape(-1, 3)
    remaining = [sum(f.span + 1 for f in frames[k + 1 :]) for k in range(problem.k)]

    def local_bond_rows(atoms_list):
        lookup = {atom: row for row, atom in enumerate(map(int, atoms_list))}
        return np.array(
            [
                (lookup[int(b.GetBeginAtomIdx())], lookup[int(b.GetEndAtomIdx())])
                for b in problem.mol.GetBonds()
                if int(b.GetBeginAtomIdx()) in lookup and int(b.GetEndAtomIdx()) in lookup
            ],
            dtype=int,
        ).reshape(-1, 2)

    # Crossing penalties only consider the geometrically dominant fold-back
    # partners: recent frames and the closure region (the first frames).
    # A full sweep over all placed bonds costs more than the whole beam
    # budget on large rings.
    frame_bond_counts = [len(local_bond_rows(f.atoms)) for f in frames]
    row_offsets = []  # row index where frame j's bonds start in a state
    running = 0
    for j in range(problem.k):
        row_offsets.append(running + j)  # j inter-frame seams precede them
        running += frame_bond_counts[j]

    beam = []
    for sign in (1, -1):
        pose = target[0].copy()
        placed = (native[frames[0].atoms] * [1, sign]) @ _rotation(pose[2]).T + pose[:2]
        beam.append(
            (
                0.0,
                0.0,
                [pose],
                [sign],
                placed,
                list(frames[0].atoms),
                local_bond_rows(frames[0].atoms),
            )
        )
    for k in range(1, problem.k):
        if deadline is not None and time.monotonic() > deadline:
            break
        candidates = []
        frame_bonds = local_bond_rows(frames[k].atoms)
        window_lo = row_offsets[min(2, k)]
        window_hi = (
            len(local_bond_rows(frames[k - 1].atoms)) + row_offsets[k - 1] + 1 if k > 0 else 0
        )
        for score, contact_score, poses, signs, placed, ids, bond_rows in beam:
            previous, pose = frames[k - 1], poses[-1]
            carbonyl = (native[previous.c_atom] * [1, signs[-1]]) @ _rotation(pose[2]).T + pose[:2]
            heading = pose[2] + ports[k - 1, signs[-1]][1]
            nitrogen = carbonyl + [math.cos(heading), math.sin(heading)]
            first_n = (native[frames[0].n_atom] * [1, signs[0]]) @ _rotation(poses[0][2]).T + poses[
                0
            ][:2]
            for sign in (1, -1):
                for incoming in ports[k, sign][0]:
                    angle = heading + math.pi - incoming
                    rotation = _rotation(angle)
                    centre = nitrogen - (native[frames[k].n_atom] * [1, sign]) @ rotation.T
                    coordinates = (native[frames[k].atoms] * [1, sign]) @ rotation.T + centre
                    distances = np.linalg.norm(coordinates[:, None] - placed[None], axis=2)
                    gap = np.maximum(overlap - distances, 0)

                    ni = list(frames[k].atoms).index(frames[k].n_atom)
                    ci = ids.index(previous.c_atom)
                    gap[ni, ci] = 0
                    target_error = float(np.sum((centre - target[k, :2]) ** 2))
                    gap_cost = 80 * float(np.sum(gap**2))
                    contacts = contact_score + gap_cost
                    value = score + gap_cost + 0.10 * target_error + 0.015 * (sign == -1)
                    if (
                        crossing_penalty > 0.0
                        and len(placed)
                        and len(frame_bonds)
                        and len(bond_rows)
                    ):
                        window_rows = np.concatenate(
                            (
                                np.arange(0, min(window_lo, len(bond_rows))),
                                np.arange(window_lo, min(window_hi, len(bond_rows))),
                            )
                        ).astype(int)
                        if len(window_rows):
                            segment_cost = _segment_penalties(
                                coordinates[frame_bonds],
                                placed[bond_rows[window_rows]],
                                margin=crossing_penalty,
                            )
                            value += segment_cost
                            contacts += segment_cost
                    carbonyl = (native[frames[k].c_atom] * [1, sign]) @ rotation.T + centre
                    exit_angle = angle + ports[k, sign][1]
                    next_n = carbonyl + [math.cos(exit_angle), math.sin(exit_angle)]
                    if problem.ring:
                        excess = max(0.0, float(np.linalg.norm(next_n - first_n)) - remaining[k])
                        value += 300 * excess**2
                    candidates.append(
                        (
                            value,
                            contacts,
                            poses + [np.r_[centre, angle]],
                            signs + [sign],
                            np.vstack([placed, coordinates]),
                            ids + list(frames[k].atoms),
                            np.vstack(
                                (bond_rows, [[ci, len(placed) + ni]], frame_bonds + len(placed))
                            ),
                        )
                    )
        candidates.sort(
            key=(lambda state: state[0]) if problem.ring else (lambda state: (state[1], state[0]))
        )
        beam = candidates[:width]
    ranked = []
    target = problem.initial.reshape(-1, 3)
    for score, _contacts, poses, signs, _placed, _ids, _bond_rows in beam[: max(count * 4, 16)]:
        poses_arr = np.array(poses)
        # a deadline break leaves the beam covering only the placed frames;
        # the remainder starts from the template ring so the poses stay
        # full-length for the solver
        if poses_arr.shape[0] < problem.k:
            poses_arr = np.vstack([poses_arr, target[poses_arr.shape[0] :]])
            signs = tuple(signs) + (1,) * (problem.k - len(signs))
        ranked.append(PortSeed(float(score), poses_arr, tuple(signs)))
    if problem.crosslinks:
        # The beam only chains the macroring; candidates whose seed leaves a
        # crosslink far from its partner pay, so the constrained solves work
        # in basins where the second loop is closable.
        for seed in ranked:
            pose = seed.poses
            for a, b in problem.crosslinks:
                pa = (
                    _rotation(pose[problem.owner[a], 2]) @ problem.local[a]
                    + pose[problem.owner[a], :2]
                )
                pb = (
                    _rotation(pose[problem.owner[b], 2]) @ problem.local[b]
                    + pose[problem.owner[b], :2]
                )
                stretch = max(0.0, float(np.linalg.norm(pa - pb)) - 1.0)
                seed.score += 200.0 * stretch**2
        ranked.sort(key=lambda seed: seed.score)
    return _select_diverse(ranked, count) if diverse else ranked[:count]

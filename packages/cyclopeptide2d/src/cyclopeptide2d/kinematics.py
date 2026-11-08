"""Interface-direction beam search for rigid residue-chain initialization.

States choose complete-frame reflections and legal incoming port directions.
The carbonyl exit determines the next N position. A remaining-length bound
rejects open-chain trajectories that cannot return to the starting port.
These are candidate coordinates: the solver still enforces all final gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
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
    if len(ns) == 1:
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


def port_seeds(problem: PoseProblem, *, width: int = 512, count: int = 4,
               overlap: float = .88) -> list[PortSeed]:
    """Propose open-chain poses with exact placed links and rigid interiors.

    The final seam is deliberately left to joint constrained optimization.
    Seed scores are heuristics, never training-data acceptance criteria.
    The overlap threshold penalizes near-crossing placements, not just
    collisions: .88 cleared seed crossings on two diagnostic inputs where
    .72 could not, while 1.0 degraded a third and was rejected.
    """
    if width < 1 or count < 1 or overlap <= 0:
        raise ValueError('Beam width, seed count and overlap threshold must be positive')
    frames, native = problem.frames, problem.local
    adjacency = [[] for _ in range(problem.n)]
    for a, b in problem.bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    ports = {(k, sign): _ports(problem, k, sign, adjacency)
             for k in range(problem.k) for sign in (1, -1)}
    target = problem.initial.reshape(-1, 3)
    remaining = [sum(f.span + 1 for f in frames[k + 1:]) for k in range(problem.k)]
    beam = []
    for sign in (1, -1):
        pose = target[0].copy()
        placed = (native[frames[0].atoms] * [1, sign]) @ _rotation(pose[2]).T + pose[:2]
        beam.append((0., [pose], [sign], placed, list(frames[0].atoms)))
    for k in range(1, problem.k):
        candidates = []
        for score, poses, signs, placed, ids in beam:
            previous, pose = frames[k - 1], poses[-1]
            carbonyl = (native[previous.c_atom] * [1, signs[-1]]) @ _rotation(pose[2]).T + pose[:2]
            heading = pose[2] + ports[k - 1, signs[-1]][1]
            nitrogen = carbonyl + [math.cos(heading), math.sin(heading)]
            first_n = (native[frames[0].n_atom] * [1, signs[0]]) @ _rotation(poses[0][2]).T + poses[0][:2]
            for sign in (1, -1):
                for incoming in ports[k, sign][0]:
                    angle = heading + math.pi - incoming
                    rotation = _rotation(angle)
                    centre = nitrogen - (native[frames[k].n_atom] * [1, sign]) @ rotation.T
                    coordinates = (native[frames[k].atoms] * [1, sign]) @ rotation.T + centre
                    distances = np.linalg.norm(coordinates[:, None] - placed[None], axis=2)
                    gap = np.maximum(overlap - distances, 0)
                    # Exempt only the newly formed covalent interface.
                    ni = list(frames[k].atoms).index(frames[k].n_atom)
                    ci = ids.index(previous.c_atom)
                    gap[ni, ci] = 0
                    target_error = float(np.sum((centre - target[k, :2]) ** 2))
                    value = score + 80 * float(np.sum(gap ** 2)) + .10 * target_error + .015 * (sign == -1)
                    carbonyl = (native[frames[k].c_atom] * [1, sign]) @ rotation.T + centre
                    exit_angle = angle + ports[k, sign][1]
                    next_n = carbonyl + [math.cos(exit_angle), math.sin(exit_angle)]
                    excess = max(0., float(np.linalg.norm(next_n - first_n)) - remaining[k])
                    value += 300 * excess ** 2
                    candidates.append((value, poses + [np.r_[centre, angle]], signs + [sign],
                                       np.vstack([placed, coordinates]), ids + list(frames[k].atoms)))
        candidates.sort(key=lambda state: state[0])
        beam = candidates[:width]
    return [PortSeed(float(score), np.array(poses), tuple(signs))
            for score, poses, signs, _, _ in beam[:count]]

"""Resolve intersecting bonds through constrained separating half-planes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .solver import PoseProblem


@dataclass(frozen=True)
class HalfPlanes:
    atoms: np.ndarray
    sides: np.ndarray
    margin: float = 0.10


def crossing_pairs(bonds, xy):
    first, second = np.triu_indices(len(bonds), 1)
    distinct = np.all(bonds[first, :, None] != bonds[second, None, :], axis=(1, 2))
    first, second = first[distinct], second[distinct]
    a, b = xy[bonds[first, 0]], xy[bonds[first, 1]]
    c, d = xy[bonds[second, 0]], xy[bonds[second, 1]]

    def cross(v, w):
        return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

    hit = ((cross(b - a, c - a) > 0) != (cross(b - a, d - a) > 0)) & (
        (cross(d - c, a - c) > 0) != (cross(d - c, b - c) > 0)
    )
    return np.column_stack((first[hit], second[hit]))


def half_plane_constraints(problem: PoseProblem, x, planes: HalfPlanes, *, jacobian=False):
    xy, rotation = problem.coordinates(x)
    ia, ib, ip = planes.atoms.T
    a, b, p = xy[ia], xy[ib], xy[ip]
    v, w = b - a, p - a
    length = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    height = (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]) / length
    if not jacobian:
        return planes.sides * height - planes.margin

    def perpendicular(vector):
        return np.column_stack((-vector[:, 1], vector[:, 0]))

    ga = perpendicular(p - b) / length[:, None] + height[:, None] * v / length[:, None] ** 2
    gb = -perpendicular(w) / length[:, None] - height[:, None] * v / length[:, None] ** 2
    gp = perpendicular(v) / length[:, None]
    rows = np.tile(np.arange(len(ia)), 3)
    atoms = np.concatenate((ia, ib, ip))
    gradients = np.concatenate((ga, gb, gp)) * np.tile(planes.sides, 3)[:, None]
    base = rows * (3 * problem.k) + 3 * problem.owner[atoms]
    result = np.zeros(len(ia) * 3 * problem.k)
    np.add.at(
        result,
        np.concatenate((base, base + 1, base + 2)),
        np.concatenate(
            (gradients[:, 0], gradients[:, 1], (gradients * rotation[atoms]).sum(axis=1))
        ),
    )
    return result.reshape(len(ia), 3 * problem.k)


def separation_options(problem: PoseProblem, x, pairs, *, width=4):
    beam = [(0.0, [], [])]
    for first, second in pairs:
        a, b = problem.bonds[first]
        c, d = problem.bonds[second]
        options = []
        for start, end, points in ((a, b, (c, d)), (c, d, (a, b))):
            for side in (-1.0, 1.0):
                atoms = np.array([(start, end, point) for point in points])
                planes = HalfPlanes(atoms, np.full(2, side))
                slack = half_plane_constraints(problem, x, planes)
                jac = half_plane_constraints(problem, x, planes, jacobian=True)
                if np.any((slack < 0) & (np.linalg.norm(jac, axis=1) < 1e-10)):
                    continue
                cost = float(np.minimum(slack, 0) @ np.minimum(slack, 0))
                options.append((cost, atoms.tolist(), [side, side]))
        if not options:
            return []
        candidates = [
            (cost + value, atoms + extra, sides + signs)
            for cost, atoms, sides in beam
            for value, extra, signs in options
        ]
        beam = sorted(candidates, key=lambda item: item[0])[:width]
    return [HalfPlanes(np.array(atoms, dtype=int), np.array(sides)) for _, atoms, sides in beam]


def repair_crossings(problem: PoseProblem, x, *, rounds=4, maxiter=200, deadline=None):
    current = x.copy()
    rank = problem.defect_rank(current)[0]
    report = {"trials": [], "accepted": [], "converged": False}
    radius = max(10.0, sum(frame.span + 1 for frame in problem.frames))
    for round_index in range(rounds):
        xy, _ = problem.coordinates(current)
        pairs = crossing_pairs(problem.bonds, xy)
        if not len(pairs):
            report["converged"] = True
            break
        if deadline is not None and time.monotonic() >= deadline:
            report["error"] = "deadline_exceeded"
            break
        best = None
        for planes in separation_options(problem, current, pairs, width=8):
            if deadline is not None and time.monotonic() >= deadline:
                break
            reference = current.copy()

            def objective(z, reference=reference):
                energy, gradient = problem.objective(z)
                change = z - reference
                return energy + 0.1 * float(change @ change), gradient + 0.2 * change

            scaled_maxiter = maxiter
            if deadline is not None:
                scaled_maxiter = max(
                    40, min(maxiter, int(max(0.0, deadline - time.monotonic()) / 0.05))
                )
            fit = minimize(
                objective,
                reference,
                method="SLSQP",
                jac=True,
                bounds=[(-radius, radius), (-radius, radius), (None, None)] * problem.k,
                constraints=[
                    {"type": "eq", "fun": problem.closure, "jac": problem.closure_jac},
                    {"type": "ineq", "fun": problem.angle_slack, "jac": problem.angle_slack_jac},
                    {
                        "type": "ineq",
                        "fun": lambda z, planes=planes: half_plane_constraints(problem, z, planes),
                        "jac": lambda z, planes=planes: half_plane_constraints(
                            problem, z, planes, jacobian=True
                        ),
                    },
                ],
                options={"maxiter": scaled_maxiter, "ftol": 1e-9},
            )
            link_error = float(np.abs(problem.closure(fit.x)).max())
            angle_slack = float(problem.angle_slack(fit.x).min())
            plane_slack = float(half_plane_constraints(problem, fit.x, planes).min())
            # the half-plane margin keeps 0.08 clearance at the looser
            # tolerance, well above the zero distance a crossing needs
            feasible = link_error < 1e-7 and min(angle_slack, plane_slack) >= -0.02
            candidate_rank = problem.defect_rank(fit.x)[0]
            report["trials"].append(
                {
                    "round": round_index,
                    "iterations": int(fit.nit),
                    "feasible": feasible,
                    "max_link_error": link_error,
                    "min_angle_slack": angle_slack,
                    "min_half_plane_slack": plane_slack,
                    "defect_rank": list(candidate_rank),
                }
            )
            if feasible and candidate_rank < rank and (best is None or candidate_rank < best[0]):
                best = (candidate_rank, fit.x.copy(), planes)
                if candidate_rank[:3] == (0, 0, 0):
                    break
        if best is None:
            report["error"] = "no_feasible_topology_improvement"
            break
        rank, current, planes = best
        report["accepted"].append(
            {
                "round": round_index,
                "defect_rank": list(rank),
                "half_planes": planes.atoms.tolist(),
                "sides": planes.sides.tolist(),
            }
        )
    xy, _ = problem.coordinates(current)
    report["remaining_crossings"] = len(crossing_pairs(problem.bonds, xy))
    report["converged"] = report["remaining_crossings"] == 0
    return current, report

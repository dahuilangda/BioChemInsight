"""Joint atom refinement with fixed intrinsic ring geometry."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Geometry import Point2D
from scipy.optimize import brentq, minimize


class RefinementDeadline(TimeoutError):
    pass


class FlexibleGeometry:
    def __init__(self, problem):
        self.problem = problem
        self.n = problem.n
        reference = np.empty_like(problem.local)
        for frame in problem.frames:
            reference[frame.atoms] = frame.local
        edges = list(map(tuple, problem.bonds))
        distances = [1.0] * len(edges)
        seen = {tuple(sorted(pair)) for pair in edges}
        rigid_pairs, rigid_distances = [], []
        self.auxiliary_rings = []
        ring_groups = []
        for ring in problem.mol.GetRingInfo().AtomRings():
            if len(set(problem.owner[list(ring)])) != 1:
                if len(ring) <= 8:
                    self.auxiliary_rings.append(tuple(ring))
                continue
            group = set(ring)
            merged = []
            for previous in ring_groups:
                if group & previous:
                    group.update(previous)
                else:
                    merged.append(previous)
            ring_groups = merged + [group]
            for a, b in itertools.combinations(ring, 2):
                rigid_pairs.append((a, b))
                rigid_distances.append(float(np.sum((reference[a] - reference[b]) ** 2)))
            for index in range(2, len(ring) - 1):
                a, b = ring[0], ring[index]
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b))
                    distances.append(float(np.sum((reference[a] - reference[b]) ** 2)))
        self.edges = np.asarray(edges, dtype=int)
        self.distances = np.asarray(distances)
        self.rigid_pairs = np.asarray(rigid_pairs, dtype=int).reshape(-1, 2)
        self.rigid_distances = np.asarray(rigid_distances)
        self.ring_templates = [
            (sorted(group), reference[sorted(group)].copy()) for group in ring_groups
        ]
        adjacency = [set() for _ in range(self.n)]
        for a, b in problem.bonds:
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
        self.adjacency = adjacency
        ports = {
            tuple(triple): target
            for triple, target in zip(problem.angles, problem.angle_targets, strict=True)
        }
        angles, targets = [], []
        for b, neighbours in enumerate(adjacency):
            for a, c in itertools.combinations(sorted(neighbours), 2):
                if problem.owner[a] == problem.owner[b] == problem.owner[c]:
                    u, v = reference[a] - reference[b], reference[c] - reference[b]
                    target = float(u @ v / max(1e-12, np.linalg.norm(u) * np.linalg.norm(v)))
                else:
                    target = ports.get((a, b, c), ports.get((c, b, a)))
                    if target is None:
                        raise ValueError("missing_connection_angle")
                angles.append((a, b, c))
                targets.append(target)
        self.angles = np.asarray(angles, dtype=int).reshape(-1, 3)
        self.targets = np.asarray(targets)
        theta = np.arccos(np.clip(self.targets, -1.0, 1.0))
        self.low = np.cos(np.minimum(np.pi, theta + np.radians(18)))
        self.high = np.cos(np.maximum(0.1, theta - np.radians(12)))

    def equalities(self, x, *, jacobian=False):
        xy = x.reshape(-1, 2)
        a, b = self.edges.T
        delta = xy[a] - xy[b]
        if not jacobian:
            return np.sum(delta * delta, axis=1) - self.distances
        out = np.zeros((len(a), self.n, 2))
        rows = np.arange(len(a))
        out[rows, a], out[rows, b] = 2 * delta, -2 * delta
        return out.reshape(len(a), 2 * self.n)

    def rigid_error(self, xy):
        a, b = self.rigid_pairs.T
        squared = np.sum((xy[a] - xy[b]) ** 2, axis=1)
        return float(np.max(np.abs(squared - self.rigid_distances), initial=0.0))

    def cosines(self, x, *, jacobian=False):
        xy = x.reshape(-1, 2)
        a, b, c = self.angles.T
        u, v = xy[a] - xy[b], xy[c] - xy[b]
        un = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        vn = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        cosine = np.sum(u * v, axis=1) / (un * vn)
        if not jacobian:
            return cosine
        du = v / (un * vn)[:, None] - cosine[:, None] * u / un[:, None] ** 2
        dv = u / (un * vn)[:, None] - cosine[:, None] * v / vn[:, None] ** 2
        out = np.zeros((len(a), self.n, 2))
        rows = np.arange(len(a))
        out[rows, a], out[rows, b], out[rows, c] = du, -du - dv, dv
        return out.reshape(len(a), 2 * self.n)

    def angle_slack(self, x):
        cosine = self.cosines(x)
        return np.r_[cosine - self.low, self.high - cosine]

    def angle_jacobian(self, x):
        jac = self.cosines(x, jacobian=True)
        return np.vstack((jac, -jac))

    def separation(self, x, aa, ab, *, jacobian=False):
        xy, problem = x.reshape(-1, 2), self.problem
        a, b = problem.ai[aa], problem.aj[aa]
        delta = xy[a] - xy[b]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        atom_slack = distance - (problem.aa_rest[aa] - 0.08)
        p, q, r = problem.ab_a[ab], problem.ab_b[ab], problem.ab_c[ab]
        ray = xy[r] - xy[q]
        squared = np.maximum(np.sum(ray * ray, axis=1), 1e-10)
        fraction = np.clip(np.sum((xy[p] - xy[q]) * ray, axis=1) / squared, 0, 1)
        edge_delta = xy[p] - xy[q] - fraction[:, None] * ray
        edge_distance = np.maximum(np.linalg.norm(edge_delta, axis=1), 1e-9)
        edge_slack = edge_distance - (problem.ab_rest[ab] - 0.06)
        if not jacobian:
            return np.r_[atom_slack, edge_slack]
        result = np.zeros((len(a) + len(p), self.n, 2))
        rows = np.arange(len(a))
        result[rows, a], result[rows, b] = delta / distance[:, None], -delta / distance[:, None]
        rows = len(a) + np.arange(len(p))
        normal = edge_delta / edge_distance[:, None]
        result[rows, p] = normal
        result[rows, q] = -(1 - fraction)[:, None] * normal
        result[rows, r] = -fraction[:, None] * normal
        return result.reshape(len(result), 2 * self.n)

    def objective(self, x, reference):
        xy = x.reshape(-1, 2)
        gradient = 0.002 * (xy - reference)
        energy = 0.001 * float(np.sum((xy - reference) ** 2))
        error = self.cosines(x) - self.targets
        energy += 4 * float(error @ error)
        gradient += ((8 * error) @ self.cosines(x, jacobian=True)).reshape(-1, 2)
        problem = self.problem
        aa, ab = problem.contact_candidates(xy)
        a, b = problem.ai[aa], problem.aj[aa]
        delta = xy[a] - xy[b]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        gap = np.maximum(problem.aa_rest[aa] - distance, 0)
        energy += 100 * float(gap @ gap)
        force = (-200 * gap / distance)[:, None] * delta
        np.add.at(gradient, a, force)
        np.add.at(gradient, b, -force)
        a, b, c = problem.ab_a[ab], problem.ab_b[ab], problem.ab_c[ab]
        ray = xy[c] - xy[b]
        squared = np.maximum(np.sum(ray * ray, axis=1), 1e-10)
        fraction = np.clip(np.sum((xy[a] - xy[b]) * ray, axis=1) / squared, 0, 1)
        delta = xy[a] - xy[b] - fraction[:, None] * ray
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        gap = np.maximum(problem.ab_rest[ab] - distance, 0)
        energy += 100 * float(gap @ gap)
        force = (-200 * gap / distance)[:, None] * delta
        np.add.at(gradient, a, force)
        np.add.at(gradient, b, -(1 - fraction)[:, None] * force)
        np.add.at(gradient, c, -fraction[:, None] * force)
        return energy, gradient.ravel()

    def embedding(self, method, seed):
        method, *parameters = method.split(":")
        reflection = int(parameters[0]) if parameters else 0
        molecule = Chem.Mol(self.problem.mol)
        fixed = set()
        if method == "coordgen":
            rdCoordGen.AddCoords(molecule)
        else:
            coordinates = {}
            if method in ("auxiliary_ring", "theta", "theta_rigid"):
                ring = min(self.auxiliary_rings, key=lambda r: (len(r), r))
                radius = 1.5 / (2 * np.sin(np.pi / len(ring)))
                for index, atom in enumerate(ring):
                    theta = 2 * np.pi * index / len(ring)
                    coordinates[atom] = Point2D(radius * np.cos(theta), radius * np.sin(theta))
                if method in ("theta", "theta_rigid"):
                    self.outer_path(coordinates, ring)
                for atom in ring:
                    point = coordinates[atom]
                    length = math.hypot(point.x, point.y)
                    for neighbour in self.adjacency[atom] - set(ring):
                        if neighbour in coordinates:
                            continue
                        coordinates[neighbour] = Point2D(
                            point.x * (1 + 1.5 / length), point.y * (1 + 1.5 / length)
                        )
                fixed = set(ring)
            sampled = method in ("sampled", "auxiliary_ring", "theta", "theta_rigid")
            rdDepictor.Compute2DCoords(
                molecule,
                forceRDKit=True,
                useRingTemplates=not bool(coordinates),
                coordMap=coordinates,
                nFlipsPerSample=3 if sampled else 0,
                nSample=50 if sampled else 0,
                sampleSeed=int(seed % 2147483647),
            )
        xy = np.asarray(molecule.GetConformer().GetPositions())[:, :2].copy()
        a, b = self.problem.bonds.T
        xy /= np.median(np.linalg.norm(xy[a] - xy[b], axis=1))
        if reflection:
            core = set(self.problem.order)
            rings = [(ids, sorted(set(ids) & core)) for ids, _ in self.ring_templates
                     if len(set(ids) & core) == 2]
            for index, (ids, anchors) in enumerate(rings):
                if not reflection & (1 << index):
                    continue
                first, second = xy[anchors]
                axis = second - first
                normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
                moving = sorted(set(ids) - set(anchors))
                xy[moving] -= 2 * ((xy[moving] - first) @ normal)[:, None] * normal
        if len(parameters) == 3:
            path = self.backbone_bridge(ring)
            lo, hi = map(int, parameters[1:])
            moving = set(path[lo + 1 : hi])
            queue = list(moving)
            core = set(self.problem.order)
            for atom in queue:
                for neighbour in self.adjacency[atom] - moving - fixed:
                    if neighbour not in core:
                        moving.add(neighbour)
                        queue.append(neighbour)
            first, second = xy[[path[lo], path[hi]]]
            axis = second - first
            if np.linalg.norm(axis) > 1e-8:
                normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
                ids = sorted(moving)
                xy[ids] -= 2 * ((xy[ids] - first) @ normal)[:, None] * normal
        return xy, fixed

    def backbone_bridge(self, ring):
        core = set(self.problem.order)
        boundary = [atom for atom in ring if self.adjacency[atom] & (core - set(ring))]
        if len(boundary) != 2:
            return []
        start, end = boundary
        allowed = (core - set(ring)) | set(boundary)
        parents, queue = {start: None}, [start]
        for atom in queue:
            for neighbour in sorted(self.adjacency[atom] & allowed):
                if neighbour not in parents:
                    parents[neighbour] = atom
                    queue.append(neighbour)
        if end not in parents:
            return []
        path, atom = [], end
        while atom is not None:
            path.append(atom)
            atom = parents[atom]
        path.reverse()
        return path

    def bridge_reflections(self):
        ring = min(self.auxiliary_rings, key=lambda r: (len(r), r))
        path = self.backbone_bridge(ring)
        positions = {atom: index for index, atom in enumerate(path)}
        junctions = []
        for ids, _ in self.ring_templates:
            anchors = sorted(positions[atom] for atom in ids if atom in positions)
            if len(anchors) == 2:
                junctions.append(anchors)
        junctions.sort()
        intervals = [(first[1], second[0]) for first, second in zip(junctions, junctions[1:])]
        intervals += [(0, entry) for entry, _ in junctions]
        intervals += [(exit, len(path) - 1) for _, exit in junctions]
        return list(dict.fromkeys((lo, hi) for lo, hi in intervals if hi - lo >= 3))

    def outer_path(self, coordinates, ring):
        path = self.backbone_bridge(ring)
        if not path:
            return
        start, end = path[0], path[-1]
        count = len(path) - 3
        if count < 4:
            return
        a = np.array([coordinates[start].x, coordinates[start].y]) / 1.5
        b = np.array([coordinates[end].x, coordinates[end].y]) / 1.5
        a += a / np.linalg.norm(a)
        b += b / np.linalg.norm(b)
        chord = float(np.linalg.norm(b - a))
        low = max(0.5, chord / 2) * (1 + 1e-8)

        def curl(radius):
            return (
                count * 2 * np.arcsin(1 / (2 * radius))
                + 2 * np.arcsin(chord / (2 * radius))
                - 2 * np.pi
            )

        if curl(low) <= 0:
            return
        radius = brentq(curl, low, float(count + chord))
        midpoint = (a + b) / 2
        normal = midpoint / np.linalg.norm(midpoint)
        centre = midpoint + normal * np.sqrt(radius**2 - chord**2 / 4)
        first = math.atan2(*(a - centre)[::-1])
        last = math.atan2(*(b - centre)[::-1])
        sign = 1 if (last - first) % (2 * np.pi) > np.pi else -1
        step = sign * 2 * np.arcsin(1 / (2 * radius))
        for index, atom in enumerate(path[1:-1]):
            theta = first + index * step
            point = centre + radius * np.array([np.cos(theta), np.sin(theta)])
            coordinates[atom] = Point2D(float(1.5 * point[0]), float(1.5 * point[1]))

    def solve(self, reference, *, fixed=(), maxiter=250, deadline=None, rigid=True):
        from .coordinates import RigidCoordinates
        from .projection import project_constraints
        from .topology import crossing_pairs

        start = time.monotonic()
        chart = RigidCoordinates(reference, self.ring_templates if rigid else (), fixed=fixed)
        if not len(chart.initial):
            return None, {"error": "no_movable_coordinates"}
        reference = chart.coordinates(chart.initial).copy()
        fixed = chart.fixed
        eq_rows = np.asarray(
            [
                not set(pair) <= fixed and chart.owner[pair[0]] != chart.owner[pair[1]]
                for pair in self.edges
            ]
        )
        angle_rows = np.asarray(
            [
                not set(triple) <= fixed and len(set(chart.owner[list(triple)])) > 1
                for triple in self.angles
            ]
        )
        slack_rows = np.r_[angle_rows, angle_rows]
        if (
            np.max(np.abs(self.equalities(reference.ravel())[~eq_rows]), initial=0.0) >= 1e-7
            or np.min(self.angle_slack(reference.ravel())[~slack_rows], initial=0.0) < -1e-6
        ):
            return None, {"error": "inconsistent_fixed_geometry"}

        def unpack(values):
            return chart.coordinates(values).ravel()

        def equalities(values):
            return self.equalities(unpack(values))[eq_rows]

        def equality_jacobian(values):
            return chart.pullback(self.equalities(unpack(values), jacobian=True)[eq_rows], values)

        def slack(values):
            return self.angle_slack(unpack(values))[slack_rows]

        def slack_jacobian(values):
            return chart.pullback(self.angle_jacobian(unpack(values))[slack_rows], values)

        def objective(values):
            if deadline is not None and time.monotonic() >= deadline:
                raise RefinementDeadline
            energy, gradient = self.objective(unpack(values), reference)
            return energy, chart.pullback(gradient, values).ravel()

        def penalty(values):
            energy, gradient = objective(values)
            eq, angular = equalities(values), np.minimum(slack(values), 0.0)
            return (
                energy + 1000 * float(eq @ eq + angular @ angular),
                gradient
                + 2000 * (eq @ equality_jacobian(values) + angular @ slack_jacobian(values)),
            )

        best = None
        iterations = 0

        def remember(values):
            nonlocal best, iterations
            iterations += 1
            xy = unpack(values)
            if (
                np.max(np.abs(self.equalities(xy)), initial=0.0) >= 1e-7
                or np.min(self.angle_slack(xy), initial=0.0) < -1e-6
                or self.rigid_error(xy.reshape(-1, 2)) >= 1e-7
            ):
                return
            aa, ab = self.problem.contact_candidates(xy.reshape(-1, 2))
            gaps = self.separation(xy, aa, ab)
            negative = np.minimum(gaps, 0.0)
            key = (
                len(crossing_pairs(self.problem.bonds, xy.reshape(-1, 2))),
                int(np.count_nonzero(gaps < -1e-6)),
                float(negative @ negative),
                float(np.sum((xy.reshape(-1, 2) - reference) ** 2)),
            )
            if best is None or key < best[0]:
                best = (key, values.copy())

        def restore(values):
            if (
                np.max(np.abs(equalities(values)), initial=0.0) <= 0.01
                and np.min(slack(values), initial=0.0) >= -0.01
            ):
                projected, valid = project_constraints(
                    values, equalities, equality_jacobian, slack, slack_jacobian,
                    deadline=deadline,
                )
                if valid:
                    remember(projected)

        fit = None
        clearance_rounds = 0
        expired = False
        try:
            warm = minimize(
                penalty,
                chart.initial,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-10},
            )
            radius = max(10.0, float(np.max(np.abs(reference))) + 5.0)
            constraints = [
                {"type": "eq", "fun": equalities, "jac": equality_jacobian},
                {"type": "ineq", "fun": slack, "jac": slack_jacobian},
            ]
            fit = minimize(
                objective,
                warm.x,
                jac=True,
                method="SLSQP",
                bounds=chart.bounds(radius),
                constraints=constraints,
                callback=remember,
                options={"maxiter": maxiter, "ftol": 1e-9},
            )
            remember(fit.x)
            restore(fit.x)
            for _ in range(3):
                if (
                    np.max(np.abs(equalities(fit.x)), initial=0.0) > 1e-6
                    or np.min(slack(fit.x), initial=0.0) < -1e-5
                ):
                    break
                xy = unpack(fit.x)
                aa, ab = self.problem.contact_candidates(xy.reshape(-1, 2))
                gaps = self.separation(xy, aa, ab)
                if np.min(gaps, initial=0.0) >= -1e-6:
                    break
                atom_count = len(aa)
                aa = aa[gaps[:atom_count] < 0.02]
                ab = ab[gaps[atom_count:] < 0.02]
                aa = np.asarray(
                    [
                        row
                        for row in aa
                        if not {int(self.problem.ai[row]), int(self.problem.aj[row])} <= fixed
                        and chart.owner[self.problem.ai[row]] != chart.owner[self.problem.aj[row]]
                    ],
                    dtype=int,
                )
                ab = np.asarray(
                    [
                        row
                        for row in ab
                        if not {
                            int(self.problem.ab_a[row]),
                            int(self.problem.ab_b[row]),
                            int(self.problem.ab_c[row]),
                        }
                        <= fixed
                        and len(
                            set(
                                chart.owner[
                                    [
                                        self.problem.ab_a[row],
                                        self.problem.ab_b[row],
                                        self.problem.ab_c[row],
                                    ]
                                ]
                            )
                        )
                        > 1
                    ],
                    dtype=int,
                )

                def contacts(values, aa=aa, ab=ab):
                    return self.separation(unpack(values), aa, ab)

                def contact_jacobian(values, aa=aa, ab=ab):
                    return chart.pullback(
                        self.separation(unpack(values), aa, ab, jacobian=True), values
                    )

                fit = minimize(
                    objective,
                    fit.x,
                    jac=True,
                    method="SLSQP",
                    bounds=chart.bounds(radius),
                    constraints=constraints
                    + [{"type": "ineq", "fun": contacts, "jac": contact_jacobian}],
                    callback=remember,
                    options={"maxiter": maxiter, "ftol": 1e-9},
                )
                clearance_rounds += 1
                remember(fit.x)
                restore(fit.x)
        except RefinementDeadline:
            expired = True
            if best is None:
                return None, {"error": "deadline_exceeded", "seconds": time.monotonic() - start}
        values = best[1] if best is not None else fit.x
        retained = fit is None or not np.array_equal(values, fit.x)
        xy = unpack(values)
        return xy.reshape(-1, 2), {
            "converged": bool(not expired and not retained and fit.success),
            "message": "best_feasible_iterate" if retained else str(fit.message),
            "deadline_exceeded": expired,
            "iterations": int(fit.nit) if fit is not None else 0,
            "candidate_checks": iterations,
            "clearance_rounds": clearance_rounds,
            "variables": len(chart.initial),
            "seconds": time.monotonic() - start,
            "max_equality_error": float(np.max(np.abs(self.equalities(xy)), initial=0.0)),
            "min_angle_slack": float(np.min(self.angle_slack(xy), initial=0.0)),
            "max_rigid_error": self.rigid_error(xy.reshape(-1, 2)),
        }


def fragment_coordinates(problem, xy):
    local = np.empty_like(xy)
    poses = []
    for frame in problem.frames:
        origin = (xy[frame.n_atom] + xy[frame.c_atom]) / 2
        direction = xy[frame.c_atom] - xy[frame.n_atom]
        angle = math.atan2(direction[1], direction[0])
        co, si = math.cos(angle), math.sin(angle)
        local[frame.atoms] = (xy[frame.atoms] - origin) @ np.array([[co, -si], [si, co]])
        poses.append([*origin, angle])
    return local, np.asarray(poses).ravel()


def flexible_refine(problem, x, *, seed=0, deadline=None):
    geometry = FlexibleGeometry(problem)
    attempts = []
    initial_rank = problem.defect_rank(x)[0]
    methods = ["localized", "neighborhood", "current", "rdkit", "coordgen", "sampled"]
    if initial_rank[0] and not geometry.auxiliary_rings:
        methods = ["coordgen", "rdkit", "sampled", "localized", "neighborhood", "current"]
    if geometry.auxiliary_rings:
        core = set(problem.order)
        cyclic_junctions = sum(len(set(ids) & core) == 2 for ids, _ in geometry.ring_templates)
        masks = list(dict.fromkeys([0, (1 << cyclic_junctions) - 1,
                                   *(1 << index for index in range(cyclic_junctions))]))[:8]
        rigid_seeds = [f"theta_rigid:{mask}" for mask in masks]
        bridge_seeds = [f"theta_rigid:{mask}:{lo}:{hi}"
                        for lo, hi in geometry.bridge_reflections() for mask in masks]
        methods = [
            rigid_seeds[0],
            "theta",
            *rigid_seeds[1:],
            *bridge_seeds,
            "localized",
            "neighborhood",
            "coordgen",
            "rdkit",
            "current",
            "sampled",
            "auxiliary_ring",
        ]
    original = problem.local.copy()
    best = None
    for method in methods:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if method in ("current", "localized", "neighborhood"):
            reference, _ = problem.coordinates(x)
            fixed = set()
            if method in ("localized", "neighborhood"):
                _, implicated = problem.defect_rank(x)
                if method == "neighborhood":
                    owners = set(implicated)
                    for a, b in problem.bonds:
                        if problem.owner[a] in owners or problem.owner[b] in owners:
                            implicated.extend((int(problem.owner[a]), int(problem.owner[b])))
                slack = geometry.angle_slack(reference.ravel()).reshape(2, -1)
                bad = np.any(slack < -1e-6, axis=0)
                bad_bonds = np.abs(geometry.equalities(reference.ravel())) >= 1e-7
                inconsistent_atoms = np.r_[
                    geometry.angles[bad].ravel(), geometry.edges[bad_bonds].ravel()
                ]
                implicated.extend(map(int, problem.owner[inconsistent_atoms]))
                moving = set(np.flatnonzero(np.isin(problem.owner, implicated)))
                fixed = set(range(problem.n)) - moving
        else:
            reference, fixed = geometry.embedding(method, seed)
        attempt_seconds = (
            12.0 if initial_rank[0] and method in ("coordgen", "rdkit", "sampled") else 6.0
        )
        limit = (
            min(deadline, time.monotonic() + attempt_seconds)
            if deadline is not None
            else time.monotonic() + attempt_seconds
        )
        xy, report = geometry.solve(reference, fixed=fixed, deadline=limit, rigid=method != "theta")
        report["initialization"] = method
        attempts.append(report)
        if (
            xy is None
            or report["max_equality_error"] >= 1e-7
            or report["min_angle_slack"] < -1e-6
            or report["max_rigid_error"] >= 1e-7
        ):
            continue
        local, candidate = fragment_coordinates(problem, xy)
        problem.local[:] = local
        accepted = False
        try:
            rank, _ = problem.defect_rank(candidate)
            if rank[:2] == (0, 0) and rank[2] > 0:
                limit = time.monotonic() + 2.0
                if deadline is not None:
                    limit = min(limit, deadline)
                cleaned, cleanup = problem.conformer_refine(candidate, rounds=6, deadline=limit)
                report["conformer_cleanup"] = cleanup
                actual, _ = problem.coordinates(cleaned)
                if (
                    np.max(np.abs(geometry.equalities(actual.ravel())), initial=0.0) < 1e-7
                    and np.min(geometry.angle_slack(actual.ravel()), initial=0.0) >= -1e-6
                    and geometry.rigid_error(actual) < 1e-7
                ):
                    candidate = cleaned
                    local = problem.local.copy()
                    rank, _ = problem.defect_rank(candidate)
                else:
                    problem.local[:] = local
            report["defect_rank"] = list(rank)
            if rank[:3] == (0, 0, 0):
                accepted = True
                return candidate, {"accepted": True, "attempts": attempts}
            if rank < initial_rank and (best is None or rank < best[0]):
                best = (rank, candidate.copy(), local.copy(), method)
        finally:
            if not accepted:
                problem.local[:] = original
    if best is not None:
        _, candidate, local, method = best
        problem.local[:] = local
        return candidate, {
            "accepted": False,
            "adopted": True,
            "initialization": method,
            "attempts": attempts,
        }
    return x, {"accepted": False, "attempts": attempts}

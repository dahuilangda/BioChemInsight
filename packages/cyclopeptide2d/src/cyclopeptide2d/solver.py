"""Constrained optimization of cyclic peptide residue poses."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.optimize import brentq, minimize
from scipy.spatial import cKDTree

from .chemistry import stereo_roundtrip_ok
from .frames import ResidueFrame as ResidueFrame
from .frames import peptide_frames as peptide_frames

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]
Report = dict[str, Any]


class PoseProblem:
    """Optimize residue translations and rotations with analytic derivatives."""

    def __init__(
        self,
        mol: Chem.Mol,
        frames: list[ResidueFrame],
        order: list[int],
        seed: int = 0,
        crosslinks: list[tuple[int, int]] | None = None,
        segment_clearance: float = 0.0,
    ) -> None:
        self.mol, self.frames, self.order = mol, frames, order
        self.crosslinks = [tuple(pair) for pair in (crosslinks or [])]
        self.segment_clearance = segment_clearance
        self.k, self.n = len(frames), mol.GetNumAtoms()
        self.owner = np.empty(self.n, dtype=int)
        self.local = np.zeros((self.n, 2))
        for k, f in enumerate(frames):
            self.owner[f.atoms] = k
            self.local[f.atoms] = f.local
        self.bonds = np.array(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=int
        )
        self.links = np.array(
            [(f.c_atom, frames[(k + 1) % self.k].n_atom) for k, f in enumerate(frames)]
            + list(self.crosslinks),
            dtype=int,
        )
        adjacency = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
        self.junction_angles = {}
        for pivot, neighbours in enumerate(adjacency):
            for root in neighbours:
                for end in neighbours - {root}:
                    u, v = self.local[root] - self.local[pivot], self.local[end] - self.local[pivot]
                    cosine = float(u @ v) / max(1e-12, float(np.linalg.norm(u) * np.linalg.norm(v)))
                    self.junction_angles[root, pivot, end] = math.acos(np.clip(cosine, -1.0, 1.0))
        labels = np.array([a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms()])
        ai, aj = np.triu_indices(self.n, 1)
        mask = np.array([b not in adjacency[a] for a, b in zip(ai, aj, strict=True)])
        self.ai, self.aj = ai[mask], aj[mask]
        self.aa_rest = 0.72 + 0.16 * (labels[self.ai].astype(float) + labels[self.aj])
        atom, bi = np.indices((self.n, len(self.bonds)))
        atom, bi = atom.ravel(), bi.ravel()
        ba, bb = self.bonds[bi, 0], self.bonds[bi, 1]
        mask = (atom != ba) & (atom != bb)
        self.ab_a, self.ab_b, self.ab_c = atom[mask], ba[mask], bb[mask]
        self.ab_bond = bi[mask]
        self.ab_rest = 0.40 + 0.16 * labels[self.ab_a]

        self._aa_rows = np.full((self.n, self.n), -1, dtype=int)
        self._aa_rows[self.ai, self.aj] = np.arange(len(self.ai))
        self._ab_rows = np.full((len(self.bonds), self.n), -1, dtype=int)
        self._ab_rows[self.ab_bond, self.ab_a] = np.arange(len(self.ab_a))
        # A bond is owned by its frame only when both endpoints belong to it;
        # inter-frame peptide links have no fixed clearance.
        self.bond_frame = np.where(
            self.owner[self.bonds[:, 0]] == self.owner[self.bonds[:, 1]],
            self.owner[self.bonds[:, 0]],
            -1,
        )

        self.aa_fr_i, self.aa_fr_j = self.owner[self.ai], self.owner[self.aj]
        self.ab_fr_p = self.owner[self.ab_a]
        self.ab_fr_q, self.ab_fr_r = self.owner[self.ab_b], self.owner[self.ab_c]
        self.angles = []
        self.angle_targets = []
        for f in frames:
            for end in (f.n_atom, f.c_atom):
                external = [a for a in adjacency[end] if self.owner[a] != self.owner[end]]
                if len(external) != 1:
                    raise ValueError("non_chain_interface")

                for adjacent in adjacency[end] - set(external):
                    self.angles.append((adjacent, end, external[0]))
                    self.angle_targets.append(f.port_cosines[(end, adjacent)])
        # Crosslink junction angles: deliberately not constrained yet. The
        # closure of a second loop can force the bridge direction far from
        # its template, and a rigid window on top of an equality that the
        # seed cannot satisfy deadlocks the solve (verified on a disulfide
        # tetramer). The drawn bond still passes the crossing, contact and
        # bond-length gates; seed-aware angle targets remain future work.
        self.angles = np.array(self.angles, dtype=int)
        self.angle_targets = np.array(self.angle_targets)
        self.side_atoms = np.array([i for i in range(self.n) if i not in set(order)])
        self.rng = np.random.default_rng(seed)
        self.initial = self.circular_seed(seed)
        self.centres_target = self.initial.reshape(-1, 3)[:, :2].copy()

        self.cavity_r = 0.38 * np.mean(np.linalg.norm(self.centres_target, axis=1))
        self.last = None
        self.evaluations = 0

    def circular_seed(self, seed: int) -> FloatArray:
        lengths = np.array([x for f in self.frames for x in (f.span, 1.0)])
        lo = max(lengths) / 2 * (1 + 1e-9)

        def curl(r):
            return float((2 * np.arcsin(lengths / (2 * r))).sum() - 2 * np.pi)

        hi = float(lengths.sum())
        if curl(lo) < 0:
            raise ValueError("unrealizable_residue_polygon")
        r = brentq(curl, lo, hi)
        step = 2 * np.arcsin(lengths / (2 * r))
        phi = np.r_[0.0, np.cumsum(step)]
        points = r * np.stack((np.cos(phi), np.sin(phi)), axis=1)
        poses = np.zeros((self.k, 3))
        for k in range(self.k):
            a, b = points[2 * k : 2 * k + 2]
            poses[k, :2] = (a + b) / 2
            poses[k, 2] = math.atan2(b[1] - a[1], b[0] - a[0])

        theta = np.arctan2(poses[:, 1], poses[:, 0])
        phase = float(self.rng.uniform(0, 2 * np.pi))
        radial = 1 + 0.10 * np.cos(2 * theta + phase) + 0.055 * np.sin(3 * theta - phase)
        poses[:, :2] *= radial[:, None]
        return poses.ravel()

    def coordinates(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        pose = x.reshape(-1, 3)
        co, si = np.cos(pose[self.owner, 2]), np.sin(pose[self.owner, 2])
        lx, ly = self.local.T
        rot = np.stack((co * lx - si * ly, si * lx + co * ly), axis=1)
        return rot + pose[self.owner, :2], np.stack((-rot[:, 1], rot[:, 0]), axis=1)

    def pose_gradient(self, atom_gradient: FloatArray, rotation: FloatArray) -> FloatArray:
        idx = np.concatenate((3 * self.owner, 3 * self.owner + 1, 3 * self.owner + 2))
        weights = np.concatenate(
            (atom_gradient[:, 0], atom_gradient[:, 1], np.sum(atom_gradient * rotation, axis=1))
        )
        return np.bincount(idx, weights=weights, minlength=3 * self.k)

    def closure(self, x: FloatArray) -> FloatArray:
        xy, _ = self.coordinates(x)
        d = xy[self.links[:, 0]] - xy[self.links[:, 1]]
        return np.sum(d * d, axis=1) - 1.0

    def closure_jac(self, x: FloatArray) -> FloatArray:
        xy, rot = self.coordinates(x)
        out = np.zeros((len(self.links), 3 * self.k))
        for k, (a, b) in enumerate(self.links):
            v = 2 * (xy[a] - xy[b])
            for atom, sign in ((a, 1), (b, -1)):
                body = self.owner[atom]
                out[k, 3 * body : 3 * body + 2] += sign * v
                out[k, 3 * body + 2] += sign * float(v @ rot[atom])
        return out

    def interface_cosines(self, x, jacobian=False):
        xy, rot = self.coordinates(x)
        ia, ib, ic = self.angles.T
        u, v = xy[ia] - xy[ib], xy[ic] - xy[ib]
        un = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        vn = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        cosine = np.sum(u * v, axis=1) / (un * vn)
        if not jacobian:
            return cosine
        du = v / (un * vn)[:, None] - cosine[:, None] * u / un[:, None] ** 2
        dv = u / (un * vn)[:, None] - cosine[:, None] * v / vn[:, None] ** 2
        m = len(ia)
        atoms = np.concatenate((ia, ib, ic))
        gs = np.concatenate((du, -du - dv, dv), axis=0)
        rows3 = np.tile(np.arange(m), 3)
        base = rows3 * (3 * self.k) + 3 * self.owner[atoms]
        flat = np.zeros(m * 3 * self.k)
        np.add.at(
            flat,
            np.concatenate((base, base + 1, base + 2)),
            np.concatenate((gs[:, 0], gs[:, 1], np.sum(gs * rot[atoms], axis=1))),
        )
        return cosine, flat.reshape(m, 3 * self.k)

    def angle_slack(self, x: FloatArray) -> FloatArray:
        target = np.arccos(np.clip(self.angle_targets, -1, 1))
        low = np.cos(np.minimum(np.pi - 0.05, target + np.radians(18)))
        high = np.cos(np.maximum(0.1, target - np.radians(12)))
        cosine = self.interface_cosines(x)
        return np.r_[cosine - low, high - cosine]

    def angle_slack_jac(self, x: FloatArray) -> FloatArray:
        _, jac = self.interface_cosines(x, jacobian=True)
        return np.vstack((jac, -jac))

    def separation_constraints(self, x, aa_ids, ab_ids, jacobian=False):
        xy, rot = self.coordinates(x)
        a, b = self.ai[aa_ids], self.aj[aa_ids]
        d = xy[a] - xy[b]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)

        aa_rest = self.aa_rest[aa_ids] - 0.08
        p, q, r = self.ab_a[ab_ids], self.ab_b[ab_ids], self.ab_c[ab_ids]
        v = xy[r] - xy[q]
        vv = np.maximum(np.sum(v * v, axis=1), 1e-10)
        t = np.clip(np.sum((xy[p] - xy[q]) * v, axis=1) / vv, 0, 1)
        delta = xy[p] - xy[q] - t[:, None] * v
        length = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        if not jacobian:
            return np.r_[dist - aa_rest, length - (self.ab_rest[ab_ids] - 0.06)]
        out = np.zeros((len(a) + len(p), 3 * self.k))
        rows = np.arange(len(a))
        for ids, g in ((a, d / dist[:, None]), (b, -d / dist[:, None])):
            body = self.owner[ids]
            out[rows, 3 * body] += g[:, 0]
            out[rows, 3 * body + 1] += g[:, 1]
            out[rows, 3 * body + 2] += np.sum(g * rot[ids], axis=1)
        rows = np.arange(len(p)) + len(a)
        unit = delta / length[:, None]
        for ids, g in ((p, unit), (q, -(1 - t)[:, None] * unit), (r, -t[:, None] * unit)):
            body = self.owner[ids]
            out[rows, 3 * body] += g[:, 0]
            out[rows, 3 * body + 1] += g[:, 1]
            out[rows, 3 * body + 2] += np.sum(g * rot[ids], axis=1)
        return out

    def clearance_polish(self, start, rounds=4, deadline=None):
        """Add nearby contacts as constraints and check separation after each solve."""
        x = start.copy()
        active_aa, active_ab = set(), set()
        report = {}

        def scaled(maxiter):
            if deadline is None:
                return maxiter
            return max(40, min(maxiter, int(max(0.0, deadline - time.monotonic()) / 0.05)))

        for round_id in range(rounds):
            if deadline is not None and time.monotonic() > deadline:
                report["error"] = "deadline_exceeded"
                break
            xy, _ = self.coordinates(x)
            d = np.linalg.norm(xy[self.ai] - xy[self.aj], axis=1)
            active_aa.update(np.flatnonzero(d < self.aa_rest + 0.7))
            p, a, b = xy[self.ab_a], xy[self.ab_b], xy[self.ab_c]
            v = b - a
            t = np.clip(np.sum((p - a) * v, axis=1) / np.maximum(np.sum(v * v, axis=1), 1e-9), 0, 1)
            length = np.linalg.norm(p - a - t[:, None] * v, axis=1)
            active_ab.update(np.flatnonzero(length < self.ab_rest + 0.5))
            aa, ab = np.array(sorted(active_aa), dtype=int), np.array(sorted(active_ab), dtype=int)
            constraints = [
                {"type": "eq", "fun": self.closure, "jac": self.closure_jac},
                {"type": "ineq", "fun": self.angle_slack, "jac": self.angle_slack_jac},
                {
                    "type": "ineq",
                    "fun": lambda z, aa=aa, ab=ab: self.separation_constraints(z, aa, ab),
                    "jac": lambda z, aa=aa, ab=ab: self.separation_constraints(z, aa, ab, True),
                },
            ]

            def penalized(z, aa=aa, ab=ab):
                e, g = self.objective(z, 250.0, 250.0)
                slack = self.separation_constraints(z, aa, ab)
                bad = np.minimum(slack, 0)
                e += 250.0 * float(bad @ bad)
                g += 500.0 * (bad @ self.separation_constraints(z, aa, ab, True))
                return e, g

            relaxed = minimize(
                penalized,
                x,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": scaled(250), "ftol": 1e-10},
            )
            radius = max(10.0, sum(f.span + 1 for f in self.frames))
            fit = minimize(
                lambda z: self.objective(z),
                relaxed.x,
                jac=True,
                method="SLSQP",
                bounds=[(-radius, radius), (-radius, radius), (None, None)] * self.k,
                constraints=constraints,
                options={"maxiter": scaled(200), "ftol": 1e-8},
            )
            report = {
                "converged": bool(fit.success),
                "iterations": int(fit.nit),
                "rounds": round_id + 1,
                "active_contacts": len(aa) + len(ab),
            }
            if np.max(np.abs(self.closure(fit.x))) > 1e-7 or self.angle_slack(fit.x).min() < -1e-6:
                report["error"] = "infeasible_contact_projection"
                break
            x = fit.x
            all_sep = self.separation_constraints(
                x, np.arange(len(self.ai)), np.arange(len(self.ab_a))
            )
            report["min_global_separation_slack"] = float(all_sep.min())
            if all_sep.min() >= -1e-6:
                break
        return x, report

    def contact_repair(self, x, rounds=4, deadline=None):
        """Minimize contact violations subject to bond and interface constraints."""
        x = x.copy()
        report = {"rounds": 0, "converged": False}
        rank_before = self.defect_rank(x)[0]
        hard_aa, hard_ab = self.aa_rest - 0.08, self.ab_rest - 0.06
        radius = max(10.0, sum(f.span + 1 for f in self.frames))
        for round_id in range(rounds):
            if deadline is not None and time.monotonic() > deadline:
                report["error"] = "deadline_exceeded"
                break
            xy, _ = self.coordinates(x)
            d = np.linalg.norm(xy[self.ai] - xy[self.aj], axis=1)
            aa = np.flatnonzero(d < hard_aa + 0.10)
            p, a, b = xy[self.ab_a], xy[self.ab_b], xy[self.ab_c]
            v = b - a
            t = np.clip(np.sum((p - a) * v, axis=1) / np.maximum(np.sum(v * v, axis=1), 1e-9), 0, 1)
            length = np.linalg.norm(p - a - t[:, None] * v, axis=1)
            ab = np.flatnonzero(length < hard_ab + 0.10)
            if not len(aa) and not len(ab):
                report.update({"rounds": round_id, "converged": True})
                break

            def repair_fun(z, aa=aa, ab=ab):
                e, g = self.objective(z)
                slack = self.separation_constraints(z, aa, ab)
                bad = np.minimum(slack, 0)
                e += 5000.0 * float(bad @ bad)
                g += 10000.0 * (bad @ self.separation_constraints(z, aa, ab, True))
                return e, g

            fit = minimize(
                repair_fun,
                x,
                jac=True,
                method="SLSQP",
                bounds=[(-radius, radius), (-radius, radius), (None, None)] * self.k,
                constraints=[
                    {"type": "eq", "fun": self.closure, "jac": self.closure_jac},
                    {"type": "ineq", "fun": self.angle_slack, "jac": self.angle_slack_jac},
                ],
                options={"maxiter": 250, "ftol": 1e-9},
            )
            if np.max(np.abs(self.closure(fit.x))) > 1e-7 or self.angle_slack(fit.x).min() < -1e-6:
                report["error"] = "repair_projection_infeasible"
                break
            fit_rank = self.defect_rank(fit.x)[0]
            if not fit_rank < rank_before:
                report["error"] = "no_feasible_improvement"
                break
            rank_before = fit_rank
            x = fit.x
            report.update(
                {
                    "rounds": round_id + 1,
                    "iterations": int(fit.nit),
                    "active_contacts": int(len(aa) + len(ab)),
                    "min_separation_slack": float(
                        self.separation_constraints(
                            x, np.arange(len(self.ai)), np.arange(len(self.ab_a))
                        ).min()
                    ),
                }
            )
            if report["min_separation_slack"] >= -1e-6:
                report["converged"] = True
                break
        return x, report

    def _scatter(self, ids, vectors):
        """Sum Cartesian derivatives by atom index."""
        flat = np.bincount(
            np.concatenate((2 * ids, 2 * ids + 1)),
            weights=np.concatenate((vectors[:, 0], vectors[:, 1])),
            minlength=2 * self.n,
        )
        return flat[0::2], flat[1::2]

    def contact_candidates(
        self, xy: FloatArray, aa_margin: float = 0.0, ab_margin: float = 0.0
    ) -> tuple[IndexArray, IndexArray]:
        """Find contact rows with exact spatial queries.

        A point within r of a segment of length L lies within L/2 + r of its
        midpoint. Actual bond lengths keep this bound valid before ring closure.
        The index is rebuilt on every call because residue coordinates can change."""
        tree = cKDTree(xy)
        pairs = tree.query_pairs(
            np.nextafter(float(self.aa_rest.max()) + aa_margin, np.inf),
            eps=0,
            output_type="ndarray",
        )
        aa = self._aa_rows[pairs[:, 0], pairs[:, 1]]
        aa = np.sort(aa[aa >= 0])
        a, b = xy[self.bonds[:, 0]], xy[self.bonds[:, 1]]
        midpoint = (a + b) * 0.5
        reach = np.nextafter(
            0.5 * np.linalg.norm(b - a, axis=1) + float(self.ab_rest.max()) + ab_margin, np.inf
        )
        neighbours = tree.query_ball_point(midpoint, reach, eps=0)
        counts = np.fromiter(map(len, neighbours), dtype=int, count=len(neighbours))
        atoms = np.fromiter(
            (atom for group in neighbours for atom in group), dtype=int, count=int(counts.sum())
        )
        bonds = np.repeat(np.arange(len(self.bonds)), counts)
        ab = self._ab_rows[bonds, atoms]
        return aa, np.sort(ab[ab >= 0])

    def segment_candidates(self, xy: FloatArray, centres: FloatArray) -> tuple[IndexArray, IndexArray]:
        """Cross-frame bond pairs that could sit within the clearance margin.

        Every point of a bond lies within max(|end - centre|) of its frame
        centre, so the midpoint gap bounds the segment distance from below.
        Same-frame pairs are skipped: their clearance is a fixed property of
        the residue template, not something the solver can change.
        """
        q, r = xy[self.bonds[:, 0]], xy[self.bonds[:, 1]]
        mid = (q + r) * 0.5
        ext = np.linalg.norm(xy - centres[self.owner], axis=1)
        reach = np.maximum(ext[self.bonds[:, 0]], ext[self.bonds[:, 1]])
        margin = 0.05
        tree = cKDTree(mid)
        pairs = tree.query_pairs(float(reach.max()) * 2 + margin, eps=0, output_type="ndarray")
        if not len(pairs):
            empty = np.empty(0, dtype=int)
            return empty, empty
        first, second = pairs[:, 0], pairs[:, 1]
        fa, fb = self.bond_frame[first], self.bond_frame[second]
        bi, bj = self.bonds[first], self.bonds[second]
        distinct = (
            (bi[:, 0] != bj[:, 0]) & (bi[:, 0] != bj[:, 1])
            & (bi[:, 1] != bj[:, 0]) & (bi[:, 1] != bj[:, 1])
        )
        keep = (fa != fb) & (fa >= 0) & (fb >= 0) & distinct
        keep[keep] &= (
            np.linalg.norm(mid[first[keep]] - mid[second[keep]], axis=1)
            <= reach[first[keep]] + reach[second[keep]] + margin
        )
        return first[keep], second[keep]

    def segment_separation(
        self, xy: FloatArray, grad: FloatArray, i: IndexArray, j: IndexArray,
        *, margin: float = 0.05, weight: float = 200.0,
    ) -> float:
        """Penalise segment pairs closer than the margin, with gradients.

        The closest-point parameters sit at their constrained optimum on the
        active branch, so by the envelope theorem the derivative is just the
        endpoint chain of the unit separation vector — no differentiation of
        the clamp cascade. At exact contact the unit vector degenerates and
        a perpendicular of the first segment serves as the push direction.
        """
        a, b = xy[self.bonds[i, 0]], xy[self.bonds[i, 1]]
        c, d = xy[self.bonds[j, 0]], xy[self.bonds[j, 1]]
        d1, d2 = b - a, d - c
        r0 = a - c
        aa_ = np.sum(d1 * d1, axis=1)
        ee = np.sum(d2 * d2, axis=1)
        bb_ = np.sum(d1 * d2, axis=1)
        cc = np.sum(d1 * r0, axis=1)
        f = np.sum(d2 * r0, axis=1)
        # Exact closest-point cascade (Ericson, RTCD 5.1.9): the clamp order
        # and the boundary refinement matter — a partial sweep lands off the
        # constrained optimum, and the envelope-theorem gradient below is
        # only valid at an optimal (s, t).
        denom = aa_ * ee - bb_ * bb_
        s = np.where(
            (aa_ > 1e-15) & (ee > 1e-15) & (np.abs(denom) > 1e-15),
            np.clip((bb_ * f - cc * ee) / np.where(np.abs(denom) > 1e-15, denom, 1.0), 0, 1),
            0.0,
        )
        t = np.where(ee > 1e-15, np.clip((bb_ * s + f) / np.maximum(ee, 1e-15), 0, 1), 0.0)
        s = np.where(
            (t == 0) & (aa_ > 1e-15), np.clip(-cc / np.maximum(aa_, 1e-15), 0, 1), s
        )
        s = np.where(
            (t == 1) & (aa_ > 1e-15), np.clip((bb_ - cc) / np.maximum(aa_, 1e-15), 0, 1), s
        )
        delta = (a + s[:, None] * d1) - (c + t[:, None] * d2)
        dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
        gap = np.maximum(margin - dist, 0)
        if not (gap > 0).any():
            return 0.0
        energy = weight * float((gap[gap > 0] ** 2).sum())
        unit = delta / dist[:, None]
        degenerate = dist < 1e-9
        if degenerate.any():
            perp = np.stack((-d1[degenerate, 1], d1[degenerate, 0]), axis=1)
            unit[degenerate] = perp / np.maximum(
                np.linalg.norm(perp, axis=1), 1e-12
            )[:, None]
        gp = (-2 * weight * gap)[:, None] * unit
        for ids, coefficient in (
            (self.bonds[i, 0], 1 - s),
            (self.bonds[i, 1], s),
        ):
            gx, gy = self._scatter(ids, gp * coefficient[:, None])
            grad[:, 0] += gx
            grad[:, 1] += gy
        for ids, coefficient in (
            (self.bonds[j, 0], 1 - t),
            (self.bonds[j, 1], t),
        ):
            gx, gy = self._scatter(ids, -gp * coefficient[:, None])
            grad[:, 0] += gx
            grad[:, 1] += gy
        return energy

    def objective(
        self, x: FloatArray, closure_weight: float = 0.0, angle_weight: float = 0.0
    ) -> tuple[float, FloatArray]:
        xy, rot = self.coordinates(x)
        grad = np.zeros_like(xy)
        energy = 0.0
        pose = x.reshape(-1, 3)

        def distances(a, b, rest, weight):
            nonlocal energy
            v = xy[a] - xy[b]
            dist = np.maximum(np.linalg.norm(v, axis=1), 1e-10)
            gap = np.maximum(rest - dist, 0)
            energy += weight * float(gap @ gap)
            force = (-2 * weight * gap / dist)[:, None] * v
            for ids, g in ((a, force), (b, -force)):
                gx, gy = self._scatter(ids, g)
                grad[:, 0] += gx
                grad[:, 1] += gy

        aa, keep = self.contact_candidates(xy)
        distances(self.ai[aa], self.aj[aa], self.aa_rest[aa], 18.0)

        p, a, b = xy[self.ab_a[keep]], xy[self.ab_b[keep]], xy[self.ab_c[keep]]
        rest_ab = self.ab_rest[keep]
        v = b - a
        vv = np.maximum(np.sum(v * v, axis=1), 1e-10)
        t = np.clip(np.sum((p - a) * v, axis=1) / vv, 0, 1)
        delta = p - a - t[:, None] * v
        dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
        gap = np.maximum(rest_ab - dist, 0)
        energy += 25.0 * float(gap @ gap)
        force = (-50.0 * gap / dist)[:, None] * delta
        for ids, g in (
            (self.ab_a[keep], force),
            (self.ab_b[keep], -(1 - t)[:, None] * force),
            (self.ab_c[keep], -t[:, None] * force),
        ):
            gx, gy = self._scatter(ids, g)
            grad[:, 0] += gx
            grad[:, 1] += gy

        # Two crossing segments keep every endpoint clear of the other
        # segment, so the point-edge term above is blind to an X crossing.
        # Optional clearance term: across several margins it consistently
        # distorted legal dense packing on the diagnostic set, so it stays
        # off by default; formed crossings are repaired discretely instead.
        if self.segment_clearance > 0.0:
            si, sj = self.segment_candidates(xy, pose[:, :2])
            if len(si):
                energy += self.segment_separation(
                    xy, grad, si, sj, margin=self.segment_clearance
                )

        ia, ib, ic = self.angles.T
        u, v = xy[ia] - xy[ib], xy[ic] - xy[ib]
        un, vn = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        un, vn = np.maximum(un, 1e-9), np.maximum(vn, 1e-9)
        cosine = np.sum(u * v, axis=1) / (un * vn)
        err = cosine - self.angle_targets
        energy += 4.0 * float(err @ err)
        gu = 8 * err[:, None] * (v / (un * vn)[:, None] - cosine[:, None] * u / un[:, None] ** 2)
        gv = 8 * err[:, None] * (u / (un * vn)[:, None] - cosine[:, None] * v / vn[:, None] ** 2)
        np.add.at(grad, ia, gu)
        np.add.at(grad, ic, gv)
        np.add.at(grad, ib, -gu - gv)
        if closure_weight:
            a, b = self.links.T
            v = xy[a] - xy[b]
            err = np.sum(v * v, axis=1) - 1
            energy += closure_weight * float(err @ err)
            force = (4 * closure_weight * err)[:, None] * v
            np.add.at(grad, a, force)
            np.add.at(grad, b, -force)
        pg = self.pose_gradient(grad, rot).reshape(-1, 3)

        err = pose[:, :2] - self.centres_target
        energy += 0.025 * float(np.sum(err * err))
        pg[:, :2] += 0.05 * err
        if angle_weight:
            slack = self.angle_slack(x)
            bad = np.minimum(slack, 0)
            energy += angle_weight * float(bad @ bad)
            pg += (2 * angle_weight * (bad @ self.angle_slack_jac(x))).reshape(-1, 3)
        self.evaluations += 1
        return energy, pg.ravel()

    def solve(
        self,
        maxiter: int = 350,
        start: FloatArray | None = None,
        angle_constraints: bool = False,
        deadline: float | None = None,
    ) -> tuple[FloatArray, Report]:

        def out_of_time():
            return deadline is not None and time.monotonic() > deadline

        if start is None:
            warm = minimize(
                lambda x: self.objective(x, 40.0),
                self.initial,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
            start = warm.x
        if angle_constraints and not out_of_time():
            warm = minimize(
                lambda x: self.objective(x, 200.0, 200.0),
                start,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
            start = warm.x
        self.last = None

        def fun(x):
            if self.last is None or not np.array_equal(x, self.last[0]):
                e, g = self.objective(x)
                self.last = (x.copy(), e, g)
            return self.last[1]

        def jac(x):
            fun(x)
            return self.last[2]

        constraints = [{"type": "eq", "fun": self.closure, "jac": self.closure_jac}]
        if angle_constraints:
            constraints.append(
                {"type": "ineq", "fun": self.angle_slack, "jac": self.angle_slack_jac}
            )
        radius = max(10.0, sum(f.span + 1 for f in self.frames))
        bounds = [(-radius, radius), (-radius, radius), (None, None)] * self.k
        slsqp_maxiter = maxiter
        if deadline is not None:
            slsqp_maxiter = max(40, min(maxiter, int(max(0.0, deadline - time.monotonic()) / 0.05)))
        fit = minimize(
            fun,
            start,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": slsqp_maxiter, "ftol": 1e-9},
        )
        return fit.x, {
            "converged": bool(fit.success),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "evaluations": self.evaluations,
            "energy": float(fit.fun),
            "max_link_error": float(np.max(np.abs(self.closure(fit.x)))),
            "min_angle_slack": float(self.angle_slack(fit.x).min()),
        }

    def defect_rank(self, x):
        rank, implicated, _ = self._defects(x)
        return rank, implicated

    def _defects(self, x, *, collect_influence=False):
        from .geometry import _ring_intrusions

        xy, _ = self.coordinates(x)

        scaled_xy = xy * 30.0
        b = self.bonds
        i, j = np.triu_indices(len(b), 1)
        mask = (
            (b[i, 0] != b[j, 0])
            & (b[i, 0] != b[j, 1])
            & (b[i, 1] != b[j, 0])
            & (b[i, 1] != b[j, 1])
        )
        i, j = i[mask], j[mask]
        p, q, r, t = scaled_xy[b[i, 0]], scaled_xy[b[i, 1]], scaled_xy[b[j, 0]], scaled_xy[b[j, 1]]

        def cross(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        c1, c2 = cross(q - p, r - p), cross(q - p, t - p)
        c3, c4 = cross(t - r, p - r), cross(t - r, q - r)
        hits = ((c1 > 0) != (c2 > 0)) & ((c3 > 0) != (c4 > 0))
        crossing_atoms = np.concatenate((b[i[hits]].ravel(), b[j[hits]].ravel()))
        implicated = set(self.owner[crossing_atoms])
        influence = set(crossing_atoms) if collect_influence else set()
        adj = [set() for _ in xy]
        for a, c in self.bonds:
            adj[a].add(c)
            adj[c].add(a)
        rings = [list(r) for r in self.mol.GetRingInfo().AtomRings() if len(r) <= 8]
        intrusions = _ring_intrusions(dict(enumerate(map(tuple, scaled_xy))), adj, rings)
        for h in intrusions:
            if "atom" in h:
                implicated.add(self.owner[h["atom"]])
            implicated.update(self.owner[h["ring"]])
            if collect_influence:
                influence.update(h["ring"])
                if "atom" in h:
                    influence.add(h["atom"])

        aa, ab = self.contact_candidates(xy)
        sep = self.separation_constraints(x, aa, ab)
        violations_sep = sep < -1e-6
        aa_bad = aa[violations_sep[: len(aa)]]
        ab_bad = ab[violations_sep[len(aa) :]]
        implicated.update(self.owner[self.ai[aa_bad]])
        implicated.update(self.owner[self.aj[aa_bad]])
        implicated.update(self.owner[self.ab_a[ab_bad]])
        implicated.update(self.owner[self.ab_b[ab_bad]])
        implicated.update(self.owner[self.ab_c[ab_bad]])
        if collect_influence:
            aa_soft = aa[sep[: len(aa)] < 0.08 + 1e-12]
            ab_soft = ab[sep[len(aa) :] < 0.06 + 1e-12]
            for atoms in (
                self.ai[aa_soft],
                self.aj[aa_soft],
                self.ab_a[ab_soft],
                self.ab_b[ab_soft],
                self.ab_c[ab_soft],
            ):
                influence.update(atoms)
            influence.update(self.angles.ravel())
        deficit = np.minimum(sep, 0)
        slack = self.angle_slack(x)
        violations = slack < -1e-5
        for a, b, c in self.angles[violations[: len(self.angles)] | violations[len(self.angles) :]]:
            implicated.update((self.owner[a], self.owner[b], self.owner[c]))
        rank = (
            int(hits.sum()) + len(intrusions),
            int(violations.sum()),
            int(violations_sep.sum()),
            float(deficit @ deficit),
            float(self.objective(x)[0]),
        )
        return rank, sorted(implicated), frozenset(influence)

    def conformer_moves(self, implicated, fine=False):
        """Generate rotations at uncharged, nonchiral methylene junctions."""
        adj = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        for k in implicated:
            frame = self.frames[k]
            core = set(frame.backbone)
            support = set(map(int, frame.atoms))
            for anchor in frame.backbone[1:-1]:
                if self.mol.GetAtomWithIdx(anchor).GetAtomicNum() != 6:
                    continue
                for pivot in sorted(adj[anchor] - core):
                    atom = self.mol.GetAtomWithIdx(pivot)
                    if (
                        atom.GetAtomicNum() != 6
                        or atom.GetDegree() != 2
                        or atom.IsInRing()
                        or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                    ):
                        continue
                    end = next(iter(adj[pivot] - {anchor}))
                    bond = self.mol.GetBondBetweenAtoms(pivot, end)
                    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
                        continue
                    moving, stack = set(), [end]
                    while stack:
                        a = stack.pop()
                        if a == pivot or a in moving:
                            continue
                        moving.add(a)
                        stack.extend(adj[a] - moving - {pivot})
                    if not moving or not moving <= support or moving & core:
                        continue
                    a, b = (
                        self.local[anchor] - self.local[pivot],
                        self.local[end] - self.local[pivot],
                    )
                    theta = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
                    angle = (-2 * theta + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle) < 1e-6:
                        continue
                    candidates = [angle]
                    if fine:
                        candidates = [
                            math.radians(v) for v in (-12.0, -6.0, 6.0, 12.0)
                        ] + candidates
                    for delta in candidates:
                        resulting = abs((theta + delta + np.pi) % (2 * np.pi) - np.pi)
                        if (
                            fine
                            and not math.radians(108) - 1e-8
                            <= resulting
                            <= math.radians(138) + 1e-8
                        ):
                            continue
                        yield k, pivot, sorted(moving), delta

    def attachment_moves(self, implicated):
        """Generate side-chain rotations within the allowed attachment angles."""
        adj = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        for k in implicated:
            frame = self.frames[k]
            core = set(frame.backbone)
            support = set(map(int, frame.atoms))
            for pivot in sorted(support):
                atom = self.mol.GetAtomWithIdx(pivot)
                if (
                    atom.GetAtomicNum() != 6
                    or atom.GetDegree() not in (3, 4)
                    or atom.GetHybridization() != Chem.HybridizationType.SP3
                ):
                    continue
                for end in sorted(adj[pivot] - core):
                    bond = self.mol.GetBondBetweenAtoms(pivot, end)
                    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
                        continue
                    moving, stack = set(), [end]
                    while stack:
                        a = stack.pop()
                        if a == pivot or a in moving:
                            continue
                        moving.add(a)
                        stack.extend(adj[a] - moving - {pivot})
                    if not moving or not moving <= support or moving & core:
                        continue
                    origin = self.local[pivot]
                    ray = self.local[end] - origin
                    for degrees in (-12.0, -6.0, -3.0, 3.0, 6.0, 12.0):
                        angle = math.radians(degrees)
                        co, si = math.cos(angle), math.sin(angle)
                        rotated = np.array([[co, -si], [si, co]]) @ ray
                        legal = True
                        for other in adj[pivot] - {end}:
                            v = self.local[other] - origin
                            denominator = max(
                                1e-12, float(np.linalg.norm(v) * np.linalg.norm(rotated))
                            )
                            theta = math.acos(np.clip(float(v @ rotated) / denominator, -1, 1))
                            native = self.junction_angles[other, pivot, end]
                            before_cross = v[0] * ray[1] - v[1] * ray[0]
                            after_cross = v[0] * rotated[1] - v[1] * rotated[0]
                            if (
                                abs(theta - native) > math.radians(12) + 1e-9
                                or before_cross * after_cross < -1e-12
                            ):
                                legal = False
                                break
                        if legal:
                            yield k, pivot, sorted(moving), angle

    def subtree_moves(self, implicated):
        """Generate rotations about non-ring single bonds.

        Bond lengths and interface coordinates remain fixed. Junction angles
        must stay within 12 degrees of the original residue coordinates."""
        adjacency = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
        ring_bonds = {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in self.mol.GetBonds()
            if b.IsInRing()
        }
        for k in implicated:
            frame = self.frames[k]
            interface = set()
            for end in (frame.n_atom, frame.c_atom):
                interface.add(end)
                interface.update(a for a in adjacency[end] if self.owner[a] == k)
            support = set(map(int, frame.atoms))
            for a, b in sorted(tuple(pair) for pair in self.bonds):
                if self.owner[a] != k or self.owner[b] != k:
                    continue
                if (int(a), int(b)) in ring_bonds:
                    continue
                if (
                    self.mol.GetBondBetweenAtoms(int(a), int(b)).GetBondType()
                    != Chem.BondType.SINGLE
                ):
                    continue
                for root, side_root in ((int(a), int(b)), (int(b), int(a))):
                    moving, stack = set(), [side_root]
                    while stack:
                        atom = stack.pop()
                        if atom == root or atom in moving:
                            continue
                        moving.add(atom)

                        stack.extend((adjacency[atom] - moving) & support)

                    if len(moving) <= 1 or moving & interface:
                        continue
                    pivot = side_root
                    origin = self.local[pivot]

                    for angle in (math.radians(v) for v in range(-180, 180, 30)):
                        if abs(angle) < 1e-12:
                            continue

                        co, si = math.cos(angle), math.sin(angle)
                        rotation = np.array([[co, -si], [si, co]])
                        incoming = self.local[root] - origin
                        legal = True
                        for neighbour in adjacency[pivot] - {root}:
                            ray = self.local[neighbour] - origin
                            denominator = max(
                                1e-12, float(np.linalg.norm(incoming) * np.linalg.norm(ray))
                            )
                            native = self.junction_angles[root, pivot, neighbour]
                            changed = math.acos(
                                np.clip(float(incoming @ (rotation @ ray)) / denominator, -1.0, 1.0)
                            )
                            if abs(changed - native) > math.radians(12.0) + 1e-9:
                                legal = False
                                break
                        if not legal:
                            continue
                        yield k, sorted(moving), pivot, angle

    def template_refine(self, x, rounds=2, deadline=None, max_trials=500):
        """Choose subtree rotations that reduce the number or extent of defects."""
        rank, implicated = self.defect_rank(x)
        accepted, trials = [], 0
        for _ in range(rounds):
            best = None
            for k, ids, pivot, angle in self.subtree_moves(implicated):
                if trials >= max_trials or (deadline is not None and time.monotonic() > deadline):
                    break
                backup = self.local[ids].copy()
                origin = self.local[pivot].copy()
                co, si = math.cos(angle), math.sin(angle)
                rotation = np.array([[co, -si], [si, co]])
                self.local[ids] = (self.local[ids] - origin) @ rotation.T + origin
                candidate_rank, _ = self.defect_rank(x)
                trials += 1
                if candidate_rank < rank and (best is None or candidate_rank < best[0]):
                    best = (candidate_rank, ids, self.local[ids].copy(), k, pivot, angle)
                self.local[ids] = backup
                if best is not None and best[0][:3] == (0, 0, 0):
                    break
            if best is None:
                break
            rank, ids, local, k, pivot, angle = best
            self.local[ids] = local
            accepted.append(
                {
                    "residue": int(k),
                    "pivot": int(pivot),
                    "angle_deg": float(np.degrees(angle)),
                    "operation": "subtree_rotation",
                    "determinant": 1,
                }
            )
            rank, implicated = self.defect_rank(x)
            if rank[:3] == (0, 0, 0):
                break
        return x, {"template_trials": trials, "template_moves": accepted, "defect_rank": list(rank)}

    def conformer_refine(self, x, rounds=3, deadline=None):
        from .motions import subtree_reflections

        rank, implicated, influence = self._defects(x, collect_influence=True)
        accepted, trials, skipped = [], 0, 0
        for _ in range(rounds):
            best = None
            moves = []
            for generator, policy in (
                (self.conformer_moves(implicated, fine=True), {"absolute_band_deg": [108, 138]}),
                (
                    self.attachment_moves(implicated),
                    {"max_template_deviation_deg": 12, "preserve_branch_order": True},
                ),
            ):
                for k, pivot, ids, angle in generator:
                    co, si = math.cos(angle), math.sin(angle)
                    moves.append(
                        (
                            k,
                            pivot,
                            ids,
                            np.array([[co, -si], [si, co]]),
                            {
                                "angle_deg": float(np.degrees(angle)),
                                "operation": "proper_rotation",
                                "determinant": 1,
                                "junction_policy": policy,
                            },
                        )
                    )
            for k, root, pivot, ids, reflection in subtree_reflections(self, implicated):
                moves.append(
                    (
                        k,
                        pivot,
                        ids,
                        reflection,
                        {
                            "axis": [int(root), int(pivot)],
                            "operation": "subtree_reflection",
                            "determinant": -1,
                            "junction_policy": "preserve_angles",
                        },
                    )
                )
            for k, pivot, ids, transform, metadata in moves:
                if influence.isdisjoint(ids):
                    skipped += 1
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    break
                backup = self.local[ids].copy()
                origin = self.local[pivot].copy()
                self.local[ids] = (self.local[ids] - origin) @ transform.T + origin
                candidate_rank, _ = self.defect_rank(x)
                trials += 1
                if candidate_rank < rank and (best is None or candidate_rank < best[0]):
                    best = (candidate_rank, ids, self.local[ids].copy(), k, pivot, metadata)
                self.local[ids] = backup
                if best is not None and best[0][:3] == (0, 0, 0):
                    break
            if best is None:
                break
            rank, ids, local, k, pivot, metadata = best
            self.local[ids] = local
            accepted.append({"residue": int(k), "hinge": int(pivot), **metadata})
            rank, implicated, influence = self._defects(x, collect_influence=True)
            if rank[:3] == (0, 0, 0):
                break
        return x, {"trials": trials, "skipped_inactive_moves": skipped, "accepted": accepted}

    def discrete_refine(self, x, rounds=4, deadline=None, max_trials=14):
        """Reoptimize single and coupled residue reflections."""
        from itertools import combinations, zip_longest

        rank, implicated = self.defect_rank(x)
        trials, accepted, groups = 0, [], []
        for _ in range(rounds):
            best = None
            xy, _ = self.coordinates(x)
            aa, ab = self.contact_candidates(xy)
            slack = self.separation_constraints(x, aa, ab)
            priorities = {}
            for ids, violation in zip(
                zip(self.ai[aa], self.aj[aa], strict=True), slack[: len(aa)], strict=True
            ):
                if violation < 0:
                    owners = tuple(sorted(set(map(int, self.owner[list(ids)]))))
                    if len(owners) == 2:
                        priorities[owners] = priorities.get(owners, 0.0) + violation**2
            for ids, violation in zip(
                zip(self.ab_a[ab], self.ab_b[ab], self.ab_c[ab], strict=True),
                slack[len(aa) :],
                strict=True,
            ):
                if violation < 0:
                    for pair in combinations(sorted(set(map(int, self.owner[list(ids)]))), 2):
                        priorities[pair] = priorities.get(pair, 0.0) + violation**2
            singles = [(k,) for k in implicated]
            pairs = sorted(
                combinations(implicated, 2), key=lambda group: -priorities.get(group, 0.0)
            )
            candidates = [
                group for pair in zip_longest(singles, pairs) for group in pair if group is not None
            ]
            for group in candidates:
                if trials >= max_trials or (deadline is not None and time.monotonic() > deadline):
                    break
                ids = np.concatenate([self.frames[k].atoms for k in group])
                self.local[ids, 1] *= -1
                candidate, _ = self.solve(
                    maxiter=180, start=x, angle_constraints=True, deadline=deadline
                )
                candidate_rank, candidate_bad = self.defect_rank(candidate)
                if candidate_rank[:2] == (0, 0) and candidate_rank[2] > 0:
                    contact_x, _ = self.clearance_polish(candidate, rounds=2, deadline=deadline)
                    contact_rank, contact_bad = self.defect_rank(contact_x)
                    if contact_rank < candidate_rank:
                        candidate, candidate_rank, candidate_bad = (
                            contact_x,
                            contact_rank,
                            contact_bad,
                        )
                trials += 1
                if (
                    np.max(np.abs(self.closure(candidate))) < 1e-7
                    and self.angle_slack(candidate).min() >= -1e-6
                    and candidate_rank < rank
                    and (best is None or candidate_rank < best[0])
                ):
                    best = (candidate_rank, candidate.copy(), group, candidate_bad)
                self.local[ids, 1] *= -1
                if best is not None and best[0][:3] == (0, 0, 0):
                    break
            if best is None:
                break
            rank, x, group, implicated = best
            for k in group:
                self.local[self.frames[k].atoms, 1] *= -1
            accepted.extend(map(int, group))
            groups.append(list(map(int, group)))
            if rank[:3] == (0, 0, 0):
                break
        return x, {
            "reflection_trials": trials,
            "whole_residue_reflections": accepted,
            "reflection_groups": groups,
            "defect_rank": list(rank),
        }


def residue_pose_layout(
    mol: Chem.Mol, seed: int = 0, maxiter: int = 350, deadline: float | None = None
) -> tuple[Chem.Mol | None, Report]:
    work = Chem.Mol(mol)
    try:
        frames, order, crosslinks = peptide_frames(work, backbone_template="auto")
        problem = PoseProblem(work, frames, order, seed, crosslinks=crosslinks)
    except ValueError as e:
        return None, {"ok": False, "error": str(e)}

    def expired():
        return deadline is not None and time.monotonic() > deadline

    budget_start = time.monotonic()

    def rung(fraction):
        if deadline is None:
            return None
        return budget_start + fraction * (deadline - budget_start)

    def before(limit):
        return deadline is None or time.monotonic() < limit

    phase_seconds = {}

    def phase(name, fn, *args, **kwargs):
        t0 = time.monotonic()
        out = fn(*args, **kwargs)
        phase_seconds[name] = round(time.monotonic() - t0, 3)
        return out

    if problem.crosslinks and problem.k < 16:
        # The circular seed ignores the second loop; cheap trial solves rank
        # the seed's asymmetry phase so a closable one drives the ladder.
        best_seed = None
        for attempt_seed in (seed, seed + 1, seed + 2):
            if expired():
                break
            trial = PoseProblem(work, frames, order, attempt_seed, crosslinks=crosslinks)
            trial_x, trial_rep = trial.solve(min(maxiter, 200))
            rank = trial.defect_rank(trial_x)[0]
            key = (rank, trial_rep["max_link_error"])
            if best_seed is None or key < best_seed[0]:
                best_seed = (key, attempt_seed)
        if best_seed is not None and best_seed[1] != seed:
            seed = best_seed[1]
            problem = PoseProblem(work, frames, order, seed, crosslinks=crosslinks)

    if problem.k >= 16:
        from .kinematics import port_seeds

        beam_deadline = rung(0.40)
        candidates = port_seeds(problem, deadline=beam_deadline)
        native = problem.local.copy()
        best = None
        seed_reports = []
        skipped_candidates = 0
        trial_maxiter = min(maxiter, 200)
        for candidate in candidates:
            if best is not None and beam_deadline is not None and time.monotonic() > beam_deadline:
                skipped_candidates += 1
                continue
            problem.local[:] = native
            for k, sign in enumerate(candidate.signs):
                problem.local[frames[k].atoms, 1] *= sign
            problem.initial = candidate.poses.ravel().copy()
            problem.centres_target = candidate.poses[:, :2].copy()
            x, initial_report = problem.solve(trial_maxiter, deadline=beam_deadline)
            x, polish = problem.solve(
                trial_maxiter, start=x, angle_constraints=True, deadline=beam_deadline
            )
            rank = problem.defect_rank(x)[0]
            feasible_ports = (
                polish["max_link_error"] < 1e-7 and problem.angle_slack(x).min() >= -1e-6
            )
            key = (not feasible_ports, *rank)
            seed_reports.append(
                {
                    "score": candidate.score,
                    "defect_rank": list(rank),
                    "feasible_ports": bool(feasible_ports),
                }
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    x.copy(),
                    problem.local.copy(),
                    candidate.poses.copy(),
                    initial_report,
                    polish,
                )
        _, x, local, poses, initial_report, polish = best
        problem.local[:] = local
        problem.initial = poses.ravel().copy()
        problem.centres_target = poses[:, :2].copy()
        report = {
            "initial_solve": initial_report,
            "polish": polish,
            "initialization": "interface_direction_beam",
            "seed_candidates": seed_reports,
        }
        if skipped_candidates:
            report["skipped_seed_candidates"] = skipped_candidates
    else:
        x, initial_report = problem.solve(maxiter, deadline=deadline)
        report = {"initial_solve": initial_report, "initialization": "asymmetric_polygon"}
        if not expired():
            polished, polish = problem.solve(
                maxiter, start=x, angle_constraints=True, deadline=deadline
            )
            if (
                polish["max_link_error"] < 1e-7
                and problem.defect_rank(polished)[0] < problem.defect_rank(x)[0]
            ):
                x = polished
            report["polish"] = polish
        else:
            report["polish"] = {"error": "deadline_exceeded"}
    if before(rung(0.40)):
        x, report["early_conformer_refine"] = phase(
            "early_conformer_refine", problem.conformer_refine, x, rounds=6, deadline=rung(0.55)
        )
    else:
        report["early_conformer_refine"] = {
            "trials": 0,
            "accepted": [],
            "error": "deadline_exceeded",
        }
    entry_rank = problem.defect_rank(x)[0]

    contacts_only = entry_rank[0] == 0 and entry_rank[1] == 0
    if entry_rank[:3] != (0, 0, 0) and not contacts_only and before(rung(0.58)):
        x, discrete = phase("discrete_refine", problem.discrete_refine, x, deadline=rung(0.75))
        report.update(discrete)
        if problem.defect_rank(x)[0][0] > 0 and before(rung(0.76)):
            # Bonds can survive every atom-level term while crossed; the
            # half-plane stage decides sides explicitly and enforces them
            # as constrained inequalities.
            from .topology import repair_crossings

            x, crossing_rep = phase(
                "crossing_repair", repair_crossings, problem, x, deadline=rung(0.86)
            )
            report["crossing_repair"] = {
                "converged": crossing_rep.get("converged"),
                "remaining_crossings": crossing_rep.get("remaining_crossings"),
                "accepted": crossing_rep.get("accepted"),
                "error": crossing_rep.get("error"),
            }
        if problem.angle_slack(x).min() >= -1e-6 and before(rung(0.78)):
            contact_x, contact_rep = phase(
                "contact_polish", problem.clearance_polish, x, deadline=rung(0.90)
            )
            if problem.defect_rank(contact_x)[0] < problem.defect_rank(x)[0]:
                x = contact_x
            report["contact_polish"] = contact_rep
    elif contacts_only and entry_rank[2] > 0 and before(rung(0.60)):
        contact_x, contact_rep = phase(
            "contact_polish", problem.clearance_polish, x, deadline=rung(0.85)
        )
        if problem.defect_rank(contact_x)[0] < problem.defect_rank(x)[0]:
            x = contact_x
        report["contact_polish"] = contact_rep
    else:
        report.update({"reflection_trials": 0, "whole_residue_reflections": []})
    if before(rung(0.88)):
        x, report["internal_conformer_refine"] = phase(
            "internal_conformer_refine", problem.conformer_refine, x, rounds=6, deadline=rung(0.88)
        )
    else:
        report["internal_conformer_refine"] = {
            "trials": 0,
            "accepted": [],
            "error": "deadline_exceeded",
        }
    rank = problem.defect_rank(x)[0]
    if rank[:3] != (0, 0, 0) and before(rung(0.88)):
        x, report["template_refine"] = phase(
            "template_refine", problem.template_refine, x, rounds=2, deadline=rung(0.94)
        )
        rank = problem.defect_rank(x)[0]
    if rank[0] > 0 and before(rung(0.90)):
        from .topology import repair_crossings

        x, crossing_rep = phase(
            "crossing_repair_tail", repair_crossings, problem, x, deadline=rung(0.97)
        )
        report["crossing_repair"] = {
            "converged": crossing_rep.get("converged"),
            "remaining_crossings": crossing_rep.get("remaining_crossings"),
            "accepted": crossing_rep.get("accepted"),
            "error": crossing_rep.get("error"),
        }
        rank = problem.defect_rank(x)[0]

    tail = []
    tail_round = 0
    while rank[:2] == (0, 0) and rank[2] > 0 and not expired() and tail_round < 4:
        previous_rank = rank
        if before(rung(0.94 + 0.02 * tail_round)):
            x, tref = problem.template_refine(x, rounds=1, deadline=rung(0.96 + 0.01 * tail_round))
            tail.append(
                {
                    "template_trials": tref["template_trials"],
                    "template_moves": len(tref["template_moves"]),
                }
            )
        if not expired():
            x, rep_r = problem.contact_repair(x, deadline=deadline)
            tail.append({"repair": rep_r.get("error") or rep_r.get("converged")})
        rank = problem.defect_rank(x)[0]
        if rank >= previous_rank and not expired():
            x, reflections = phase(
                f"contact_reflections_{tail_round}",
                problem.discrete_refine,
                x,
                rounds=2,
                max_trials=8,
                deadline=deadline,
            )
            tail.append({"reflections": reflections})
            rank = problem.defect_rank(x)[0]
            if rank >= previous_rank:
                break
        tail_round += 1
    if tail:
        report["tail_alternation"] = tail
    report["defect_rank"] = list(rank)
    if phase_seconds:
        report["phase_seconds"] = phase_seconds
    report["max_link_error"] = float(np.max(np.abs(problem.closure(x))))
    xy, _ = problem.coordinates(x)
    conf = Chem.Conformer(work.GetNumAtoms())
    conf.Set3D(False)
    for i, (a, b) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(a * 30), float(b * 30), 0.0))
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    from .chemistry import _adjacency
    from .geometry import DEFAULT_CONFIG, _layout_metrics

    positions = {i: (float(a * 30), float(b * 30)) for i, (a, b) in enumerate(xy)}
    small_rings = [list(r) for r in work.GetRingInfo().AtomRings() if len(r) <= 8]
    metrics = _layout_metrics(
        positions,
        _adjacency(work),
        order,
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in work.GetBonds()],
        DEFAULT_CONFIG,
        small_rings=small_rings,
    )
    link_error = float(np.max(np.abs(problem.closure(x))))
    angle_slack = float(problem.angle_slack(x).min())
    all_slack = problem.separation_constraints(
        x, np.arange(len(problem.ai)), np.arange(len(problem.ab_a))
    )
    separation_slack = float(all_slack.min())
    contact_slack = float(all_slack[: len(problem.ai) + len(problem.ab_a)].min())
    centre_diagnostic = xy[order].mean(axis=0)
    cavity_slack = float(
        (
            np.linalg.norm(xy[problem.side_atoms] - centre_diagnostic, axis=1) - problem.cavity_r
        ).min()
    )
    lengths = np.linalg.norm(xy[problem.bonds[:, 0]] - xy[problem.bonds[:, 1]], axis=1)
    bond_error = float(np.abs(lengths - 1).max())
    centre = xy[order].mean(axis=0)
    contour = xy[order] - centre
    asymmetry = None
    if len(order) % 2 == 0:
        half = len(order) // 2
        asymmetry = float(
            np.sqrt(np.mean(np.sum((contour[:half] + contour[half:]) ** 2, axis=1)))
            / max(1e-9, np.linalg.norm(contour, axis=1).mean())
        )
    stereo_ok = stereo_roundtrip_ok(work)
    feasible = (
        link_error < 1e-7
        and angle_slack >= -1e-6
        and separation_slack >= -1e-6
        and bond_error < 1e-6
        and metrics["bond_crossings"] == 0
        and metrics["n_ring_intrusions"] == 0
        and stereo_ok
    )
    report.update(
        {
            "ok": True,
            "engine": "residue_pose",
            "seed": seed,
            "residues": len(frames),
            "crosslinks": [list(pair) for pair in problem.crosslinks],
            "backbone_templates": [frame.backbone_template for frame in frames],
            "stereo_roundtrip": stereo_ok,
            "metrics": metrics,
            "constraints": {
                "max_link_error": link_error,
                "min_angle_slack": angle_slack,
                "min_global_separation_slack": separation_slack,
                "min_contact_slack": contact_slack,
                "central_occupancy_diagnostic": cavity_slack,
                "central_occupancy_is_gate": False,
                "max_bond_relative_error": bond_error,
            },
            "half_turn_asymmetry": asymmetry,
            "energy": float(problem.objective(x)[0]),
            "converged": feasible,
            "qa_pass": feasible,
            "solution_pose": x.tolist(),
            "solution_local": problem.local.tolist(),
            "training_approved": False,
            "training_note": "Geometry validation does not assess label readability or recognition accuracy.",
        }
    )
    return work, report

"""Joint 2D cyclic-peptide layout on rigid amino-acid frames.

Each frame is an independently depicted chemical fragment obtained by cutting
ONLY the macrocycle's amide bonds. Its pose is in SE(2), with an optional
whole-frame reflection. All intra-residue distances and small rings are exact
invariants. Inter-residue peptide bonds are equality constraints; collisions,
interface angles and load-dependent spacing are optimized
on the same variables, including the backbone. No atom springs or bond shrink.

The solver is a 2D depiction method, not a physical 3D conformation method.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D
from scipy.optimize import brentq, minimize

from . import chemistry as zz
from .chemistry import _is_carbonyl_carbon, stereo_roundtrip_ok, residue_units


@dataclass
class ResidueFrame:
    atoms: np.ndarray
    local: np.ndarray
    n_atom: int
    c_atom: int
    backbone: list[int]
    span: float
    port_cosines: dict


def peptide_frames(mol: Chem.Mol) -> tuple[list[ResidueFrame], list[int]]:
    """Cut true peptide interfaces, retaining complete proline/ACPC rings.

    Reject crosslinks that join two supports: a rigid-frame chain is not a
    valid representation of those graphs. They require a multiloop solver.
    Original atom indices and stereochemical tags are never changed.
    """
    ring = zz.find_macroring(mol)
    order = zz.ordered_ring(mol, ring)
    units = None
    for candidate in (order, order[::-1]):
        trial = residue_units(mol, candidate)
        if trial and all(_is_carbonyl_carbon(mol, u['atoms'][-1]) for u in trial):
            units, order = trial, candidate
            break
    if not units or len(units) < 3:
        raise ValueError('not_a_head_to_tail_peptide')
    cuts = []
    for k, u in enumerate(units):
        a, b = u['atoms'][-1], units[(k + 1) % len(units)]['atoms'][0]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError('invalid_peptide_interface')
        cuts.append(bond.GetIdx())
    cut = Chem.FragmentOnBonds(mol, cuts, addDummies=True)
    mappings = []
    fragments = Chem.GetMolFrags(cut, asMols=True, fragsMolAtomMapping=mappings)
    original_n = mol.GetNumAtoms()
    by_atom = {}
    for fragment, mapping in zip(fragments, mappings):
        for a in mapping:
            if a < original_n:
                by_atom[a] = (fragment, mapping)
    frames = []
    claimed = set()
    for u in units:
        n, c = u['atoms'][0], u['atoms'][-1]
        fragment, mapping = by_atom[n]
        support = [a for a in mapping if a < original_n]
        if c not in support or claimed.intersection(support) or \
                not set(u['atoms']).issubset(support):
            raise ValueError('crosslinked_residue_support')
        coord_map = {}
        if len(u['atoms']) == 3:
            na, ca, co = [mapping.index(a) for a in u['atoms']]
            if not fragment.GetBondBetweenAtoms(na, ca).IsInRing() and \
                    not fragment.GetBondBetweenAtoms(ca, co).IsInRing():
                from rdkit.Geometry import Point2D
                coord_map = {na: Point2D(-1.299038105676658, 0.),
                             ca: Point2D(0., -.75),
                             co: Point2D(1.299038105676658, 0.)}
        rdDepictor.Compute2DCoords(fragment, canonOrient=False, forceRDKit=True,
                                  coordMap=coord_map)
        xyz = np.asarray(fragment.GetConformer().GetPositions())[:, :2]
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in fragment.GetBonds()]
        scale = np.median([np.linalg.norm(xyz[a] - xyz[b]) for a, b in bonds])
        xy = xyz / scale
        ni, ci = mapping.index(n), mapping.index(c)
        centre = (xy[ni] + xy[ci]) / 2
        tangent = xy[ci] - xy[ni]
        span = float(np.linalg.norm(tangent))
        if span < 0.1:
            raise ValueError('degenerate_residue_chord')
        tangent /= span
        normal = np.array([-tangent[1], tangent[0]])
        local = (xy - centre) @ np.stack((tangent, normal), axis=1)
        ids = [mapping.index(a) for a in support]
        local = local[ids]
        # Frame y<0 is the outside of a counterclockwise contour. Prefer
        # the complete substituent mass outside, not a side-chain-only flip.
        side = [i for i, a in enumerate(support) if a not in u['atoms'] and
                mol.GetAtomWithIdx(a).GetAtomicNum() != 8]
        score = float(local[side, 1].sum()) if side else float(local[:, 1].sum())
        if score > 0:
            local[:, 1] *= -1
        port_cosines = {}
        for end in (n, c):
            ei = mapping.index(end)
            dummy = [nb.GetIdx() for nb in fragment.GetAtomWithIdx(ei).GetNeighbors()
                     if mapping[nb.GetIdx()] >= original_n]
            if len(dummy) != 1:
                raise ValueError('invalid_residue_port')
            port = xy[dummy[0]] - xy[ei]
            for nb in fragment.GetAtomWithIdx(ei).GetNeighbors():
                ni = nb.GetIdx()
                if mapping[ni] >= original_n:
                    continue
                v = xy[ni] - xy[ei]
                port_cosines[(end, mapping[ni])] = float(v @ port /
                    (np.linalg.norm(v)*np.linalg.norm(port)))
        frames.append(ResidueFrame(np.array(support), local, n, c,
                                   list(u['atoms']), span, port_cosines))
        claimed.update(support)
    if claimed != set(range(original_n)):
        raise ValueError('disconnected_or_uncovered_atoms')
    return frames, order


class PoseProblem:
    """Analytic atom-to-pose gradients for joint collision/closure solving."""
    def __init__(self, mol, frames, order, seed=0):
        self.mol, self.frames, self.order = mol, frames, order
        self.k, self.n = len(frames), mol.GetNumAtoms()
        self.owner = np.empty(self.n, dtype=int)
        self.local = np.zeros((self.n, 2))
        for k, f in enumerate(frames):
            self.owner[f.atoms] = k
            self.local[f.atoms] = f.local
        self.bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                               for b in mol.GetBonds()], dtype=int)
        self.links = np.array([(f.c_atom, frames[(k+1) % self.k].n_atom)
                               for k, f in enumerate(frames)], dtype=int)
        adjacency = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adjacency[a].add(b); adjacency[b].add(a)
        labels = np.array([a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms()])
        ai, aj = np.triu_indices(self.n, 1)
        mask = np.array([b not in adjacency[a] for a,b in zip(ai,aj)])
        self.ai, self.aj = ai[mask], aj[mask]
        self.aa_rest = .72 + .16 * (labels[self.ai].astype(float) + labels[self.aj])
        atom, bi = np.indices((self.n, len(self.bonds)))
        atom, bi = atom.ravel(), bi.ravel()
        ba, bb = self.bonds[bi, 0], self.bonds[bi, 1]
        mask = (atom != ba) & (atom != bb)
        self.ab_a, self.ab_b, self.ab_c = atom[mask], ba[mask], bb[mask]
        self.ab_bond = bi[mask]
        self.ab_rest = .40 + .16 * labels[self.ab_a]
        # Exact pair pruning bounds. Rigid frames fix every atom's distance
        # to its pose centre, so pair distances are bounded below by the
        # frame-centre gap minus the two body radii. Radii are recomputed
        # per evaluation: hinge rotations change them.
        self.aa_fr_i, self.aa_fr_j = self.owner[self.ai], self.owner[self.aj]
        self.ab_fr_p = self.owner[self.ab_a]
        self.ab_fr_q, self.ab_fr_r = self.owner[self.ab_b], self.owner[self.ab_c]
        self.angles = []
        self.angle_targets = []
        for f in frames:
            for end in (f.n_atom, f.c_atom):
                external = [a for a in adjacency[end] if self.owner[a] != self.owner[end]]
                if len(external) != 1:
                    raise ValueError('non_chain_interface')
                # Every occupied valence slot matters. Using only the
                # backbone neighbour admits the other 120-degree solution,
                # placing the peptide bond ON TOP OF the carbonyl oxygen.
                for adjacent in adjacency[end] - set(external):
                    self.angles.append((adjacent, end, external[0]))
                    self.angle_targets.append(f.port_cosines[(end, adjacent)])
        self.angles = np.array(self.angles, dtype=int)
        self.angle_targets = np.array(self.angle_targets)
        self.side_atoms = np.array([i for i in range(self.n) if i not in set(order)])
        self.rng = np.random.default_rng(seed)
        self.initial = self.circular_seed(seed)
        self.centres_target = self.initial.reshape(-1, 3)[:, :2].copy()
        # Diagnostic radius only: inward-facing sidechains are valid.
        # Central occupancy never constrains or penalizes the layout.
        self.cavity_r = .38 * np.mean(np.linalg.norm(self.centres_target, axis=1))
        self.last = None
        self.evaluations = 0

    def circular_seed(self, seed):
        lengths = np.array([x for f in self.frames for x in (f.span, 1.)])
        lo = max(lengths) / 2 * (1 + 1e-9)
        def curl(r): return float((2 * np.arcsin(lengths/(2*r))).sum() - 2*np.pi)
        hi = float(lengths.sum())
        if curl(lo) < 0:
            raise ValueError('unrealizable_residue_polygon')
        r = brentq(curl, lo, hi)
        step = 2 * np.arcsin(lengths / (2*r))
        phi = np.r_[0., np.cumsum(step)]
        points = r * np.stack((np.cos(phi), np.sin(phi)), axis=1)
        poses = np.zeros((self.k, 3))
        for k in range(self.k):
            a, b = points[2*k:2*k+2]
            poses[k, :2] = (a+b)/2
            poses[k, 2] = math.atan2(b[1]-a[1], b[0]-a[0])
        # Low frequency asymmetry, no independent atom jitter. Chemical
        # crowding can redistribute these frame poses jointly afterwards.
        theta = np.arctan2(poses[:, 1], poses[:, 0])
        phase = float(self.rng.uniform(0, 2*np.pi))
        radial = 1 + .10*np.cos(2*theta+phase) + .055*np.sin(3*theta-phase)
        poses[:, :2] *= radial[:, None]
        return poses.ravel()

    def coordinates(self, x):
        pose = x.reshape(-1, 3)
        co, si = np.cos(pose[self.owner, 2]), np.sin(pose[self.owner, 2])
        lx, ly = self.local.T
        rot = np.stack((co*lx-si*ly, si*lx+co*ly), axis=1)
        return rot + pose[self.owner, :2], np.stack((-rot[:, 1], rot[:, 0]), axis=1)

    def pose_gradient(self, atom_gradient, rotation):
        idx = np.concatenate((3*self.owner, 3*self.owner + 1, 3*self.owner + 2))
        weights = np.concatenate((atom_gradient[:, 0], atom_gradient[:, 1],
                                  np.sum(atom_gradient * rotation, axis=1)))
        return np.bincount(idx, weights=weights, minlength=3*self.k)

    def closure(self, x):
        xy, _ = self.coordinates(x)
        d = xy[self.links[:, 0]] - xy[self.links[:, 1]]
        return np.sum(d*d, axis=1) - 1.

    def closure_jac(self, x):
        xy, rot = self.coordinates(x)
        out = np.zeros((self.k, 3*self.k))
        for k, (a, b) in enumerate(self.links):
            v = 2*(xy[a] - xy[b])
            for atom, sign in ((a, 1), (b, -1)):
                body = self.owner[atom]
                out[k, 3*body:3*body+2] += sign*v
                out[k, 3*body+2] += sign * float(v @ rot[atom])
        return out

    def interface_cosines(self, x, jacobian=False):
        xy, rot = self.coordinates(x)
        ia, ib, ic = self.angles.T
        u, v = xy[ia]-xy[ib], xy[ic]-xy[ib]
        un = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        vn = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        cosine = np.sum(u*v, axis=1)/(un*vn)
        if not jacobian:
            return cosine
        du = v/(un*vn)[:, None] - cosine[:, None]*u/un[:, None]**2
        dv = u/(un*vn)[:, None] - cosine[:, None]*v/vn[:, None]**2
        m = len(ia)
        atoms = np.concatenate((ia, ib, ic))
        gs = np.concatenate((du, -du-dv, dv), axis=0)
        rows3 = np.tile(np.arange(m), 3)
        base = rows3*(3*self.k) + 3*self.owner[atoms]
        flat = np.zeros(m*3*self.k)
        np.add.at(flat, np.concatenate((base, base+1, base+2)),
                  np.concatenate((gs[:, 0], gs[:, 1],
                                  np.sum(gs*rot[atoms], axis=1))))
        return cosine, flat.reshape(m, 3*self.k)

    def angle_slack(self, x):
        target = np.arccos(np.clip(self.angle_targets, -1, 1))
        low = np.cos(np.minimum(np.pi-.05, target+np.radians(18)))
        high = np.cos(np.maximum(.1, target-np.radians(12)))
        cosine = self.interface_cosines(x)
        return np.r_[cosine-low, high-cosine]

    def angle_slack_jac(self, x):
        _, jac = self.interface_cosines(x, jacobian=True)
        return np.vstack((jac, -jac))

    def separation_constraints(self, x, aa_ids, ab_ids, jacobian=False):
        xy, rot = self.coordinates(x)
        a, b = self.ai[aa_ids], self.aj[aa_ids]
        d = xy[a]-xy[b]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        # Lower than soft budgets, higher than the old hard-collapse gate.
        aa_rest = self.aa_rest[aa_ids] - .08
        p, q, r = self.ab_a[ab_ids], self.ab_b[ab_ids], self.ab_c[ab_ids]
        v = xy[r]-xy[q]
        vv = np.maximum(np.sum(v*v, axis=1), 1e-10)
        t = np.clip(np.sum((xy[p]-xy[q])*v, axis=1)/vv, 0, 1)
        delta = xy[p]-xy[q]-t[:, None]*v
        length = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        if not jacobian:
            return np.r_[dist-aa_rest,length-(self.ab_rest[ab_ids]-.06)]
        out = np.zeros((len(a)+len(p),3*self.k))
        rows = np.arange(len(a))
        for ids, g in ((a, d/dist[:, None]), (b, -d/dist[:, None])):
            body = self.owner[ids]
            out[rows, 3*body] += g[:, 0]
            out[rows, 3*body+1] += g[:, 1]
            out[rows, 3*body+2] += np.sum(g*rot[ids], axis=1)
        rows = np.arange(len(p))+len(a)
        unit = delta/length[:, None]
        for ids, g in ((p, unit), (q, -(1-t)[:, None]*unit), (r, -t[:, None]*unit)):
            body = self.owner[ids]
            out[rows, 3*body] += g[:, 0]
            out[rows, 3*body+1] += g[:, 1]
            out[rows, 3*body+2] += np.sum(g*rot[ids], axis=1)
        return out

    def clearance_polish(self, start, rounds=4, deadline=None):
        """Constraint generation: solve contacts, verify globally, add misses."""
        x = start.copy()
        active_aa, active_ab = set(), set()
        report = {}
        def scaled(maxiter):
            if deadline is None:
                return maxiter
            return max(40, min(maxiter,
                int(max(0., deadline - time.monotonic()) / .05)))
        for round_id in range(rounds):
            if deadline is not None and time.monotonic() > deadline:
                report['error'] = 'deadline_exceeded'
                break
            xy, _ = self.coordinates(x)
            d = np.linalg.norm(xy[self.ai]-xy[self.aj], axis=1)
            active_aa.update(np.flatnonzero(d < self.aa_rest+.7))
            p, a, b = xy[self.ab_a], xy[self.ab_b], xy[self.ab_c]
            v = b-a
            t = np.clip(np.sum((p-a)*v, axis=1)/np.maximum(np.sum(v*v, axis=1), 1e-9), 0, 1)
            length = np.linalg.norm(p-a-t[:, None]*v, axis=1)
            active_ab.update(np.flatnonzero(length < self.ab_rest+.5))
            aa, ab = np.array(sorted(active_aa), dtype=int), np.array(sorted(active_ab), dtype=int)
            constraints = [
                {'type': 'eq', 'fun': self.closure, 'jac': self.closure_jac},
                {'type': 'ineq', 'fun': self.angle_slack, 'jac': self.angle_slack_jac},
                {'type': 'ineq',
                 'fun': lambda z: self.separation_constraints(z, aa, ab),
                 'jac': lambda z: self.separation_constraints(z, aa, ab, True)}]
            def penalized(z):
                e, g = self.objective(z, 250., 250.)
                slack = self.separation_constraints(z, aa, ab)
                bad = np.minimum(slack, 0)
                e += 250.*float(bad@bad)
                g += 500.*(bad @ self.separation_constraints(z, aa, ab, True))
                return e, g
            relaxed = minimize(penalized, x, jac=True, method='L-BFGS-B',
                               options={'maxiter': scaled(250), 'ftol': 1e-10})
            radius = max(10., sum(f.span+1 for f in self.frames))
            fit = minimize(lambda z: self.objective(z), relaxed.x, jac=True, method='SLSQP',
                           bounds=[(-radius, radius), (-radius, radius), (None, None)]*self.k,
                           constraints=constraints,
                           options={'maxiter': scaled(200), 'ftol': 1e-8})
            report = {'converged': bool(fit.success), 'iterations': int(fit.nit),
                      'rounds': round_id+1, 'active_contacts': len(aa)+len(ab)}
            if np.max(np.abs(self.closure(fit.x))) > 1e-7 or self.angle_slack(fit.x).min() < -1e-6:
                report['error'] = 'infeasible_contact_projection'
                break
            x = fit.x
            all_sep = self.separation_constraints(x, np.arange(len(self.ai)),
                                                 np.arange(len(self.ab_a)))
            report['min_global_separation_slack'] = float(all_sep.min())
            if all_sep.min() >= -1e-6:
                break
        return x, report

    def contact_repair(self, x, rounds=4, deadline=None):
        """Lean endgame for layouts whose only remaining defects are contacts.

        Imposing the active separation rows as SLSQP constraints makes the
        QP inconsistent on tight packings and the solve wanders far off
        closure. Here the violated rows only penalize the objective while
        closure and interface angles stay exact constraints, which keeps
        every iterate a feasible ring layout.
        """
        x = x.copy()
        report = {'rounds': 0, 'converged': False}
        rank_before = self.defect_rank(x)[0]
        hard_aa, hard_ab = self.aa_rest - .08, self.ab_rest - .06
        radius = max(10., sum(f.span+1 for f in self.frames))
        for round_id in range(rounds):
            if deadline is not None and time.monotonic() > deadline:
                report['error'] = 'deadline_exceeded'
                break
            xy, _ = self.coordinates(x)
            d = np.linalg.norm(xy[self.ai]-xy[self.aj], axis=1)
            aa = np.flatnonzero(d < hard_aa + .10)
            p, a, b = xy[self.ab_a], xy[self.ab_b], xy[self.ab_c]
            v = b-a
            t = np.clip(np.sum((p-a)*v, axis=1)/np.maximum(np.sum(v*v, axis=1), 1e-9), 0, 1)
            length = np.linalg.norm(p-a-t[:, None]*v, axis=1)
            ab = np.flatnonzero(length < hard_ab + .10)
            if not len(aa) and not len(ab):
                report.update({'rounds': round_id, 'converged': True})
                break
            def repair_fun(z, aa=aa, ab=ab):
                e, g = self.objective(z)
                slack = self.separation_constraints(z, aa, ab)
                bad = np.minimum(slack, 0)
                e += 5000.*float(bad@bad)
                g += 10000.*(bad @ self.separation_constraints(z, aa, ab, True))
                return e, g
            fit = minimize(repair_fun, x, jac=True, method='SLSQP',
                           bounds=[(-radius, radius), (-radius, radius), (None, None)]*self.k,
                           constraints=[
                {'type': 'eq', 'fun': self.closure, 'jac': self.closure_jac},
                {'type': 'ineq', 'fun': self.angle_slack, 'jac': self.angle_slack_jac}],
                           options={'maxiter': 250, 'ftol': 1e-9})
            if np.max(np.abs(self.closure(fit.x))) > 1e-7 or \
                    self.angle_slack(fit.x).min() < -1e-6:
                report['error'] = 'repair_projection_infeasible'
                break
            fit_rank = self.defect_rank(fit.x)[0]
            if not fit_rank < rank_before:
                report['error'] = 'no_feasible_improvement'
                break
            rank_before = fit_rank
            x = fit.x
            report.update({'rounds': round_id + 1, 'iterations': int(fit.nit),
                           'active_contacts': int(len(aa)+len(ab)),
                           'min_separation_slack': float(self.separation_constraints(
                               x, np.arange(len(self.ai)), np.arange(len(self.ab_a))).min())})
            if report['min_separation_slack'] >= -1e-6:
                report['converged'] = True
                break
        return x, report

    def _scatter(self, ids, vectors):
        """Column-wise scatter-add via bincount; orders faster than np.add.at."""
        flat = np.bincount(np.concatenate((2 * ids, 2 * ids + 1)),
                           weights=np.concatenate((vectors[:, 0], vectors[:, 1])),
                           minlength=2 * self.n)
        return flat[0::2], flat[1::2]

    def objective(self, x, closure_weight=0., angle_weight=0.):
        xy, rot = self.coordinates(x)
        grad = np.zeros_like(xy)
        energy = 0.
        pose = x.reshape(-1, 3)
        centres = pose[:, :2]
        radius = np.linalg.norm(self.local, axis=1)
        centre_gap = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
        def distances(a, b, rest, weight):
            nonlocal energy
            v = xy[a] - xy[b]
            dist = np.maximum(np.linalg.norm(v, axis=1), 1e-10)
            gap = np.maximum(rest-dist, 0)
            energy += weight*float(gap@gap)
            force = (-2*weight*gap/dist)[:, None] * v
            for ids, g in ((a, force), (b, -force)):
                gx, gy = self._scatter(ids, g)
                grad[:, 0] += gx; grad[:, 1] += gy
        # A pair cannot violate its budget when even the frame-centre gap
        # minus both body radii clears it; those rows contribute exactly zero.
        bound = centre_gap[self.aa_fr_i, self.aa_fr_j] - radius[self.ai] - radius[self.aj]
        keep = bound < self.aa_rest
        distances(self.ai[keep], self.aj[keep], self.aa_rest[keep], 18.)
        # Point-edge distances include glyph budgets and detect bond
        # crossings missed by an atom-atom-only congestion score. Any point
        # of a bond stays within (bond length + end radius) of its end's
        # centre, which bounds the point-edge distance from below.
        ext = np.linalg.norm(xy - centres[self.owner], axis=1)
        reach = np.linalg.norm(xy[self.bonds[:, 0]] - xy[self.bonds[:, 1]],
                               axis=1)[self.ab_bond]
        bound = np.minimum(
            centre_gap[self.ab_fr_p, self.ab_fr_q] - radius[self.ab_a] - reach - ext[self.ab_b],
            centre_gap[self.ab_fr_p, self.ab_fr_r] - radius[self.ab_a] - reach - ext[self.ab_c])
        keep = bound < self.ab_rest
        p, a, b = xy[self.ab_a[keep]], xy[self.ab_b[keep]], xy[self.ab_c[keep]]
        rest_ab = self.ab_rest[keep]
        v = b-a
        vv = np.maximum(np.sum(v*v, axis=1), 1e-10)
        t = np.clip(np.sum((p-a)*v, axis=1)/vv, 0, 1)
        delta = p-a-t[:, None]*v
        dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
        gap = np.maximum(rest_ab-dist, 0)
        energy += 25.*float(gap@gap)
        force = (-50.*gap/dist)[:, None]*delta
        for ids, g in ((self.ab_a[keep], force),
                       (self.ab_b[keep], -(1-t)[:, None]*force),
                       (self.ab_c[keep], -t[:, None]*force)):
            gx, gy = self._scatter(ids, g)
            grad[:, 0] += gx; grad[:, 1] += gy
        # Trigonal interface angles, differentiated in Cartesian space.
        ia, ib, ic = self.angles.T
        u, v = xy[ia]-xy[ib], xy[ic]-xy[ib]
        un, vn = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        un, vn = np.maximum(un, 1e-9), np.maximum(vn, 1e-9)
        cosine = np.sum(u*v, axis=1)/(un*vn)
        err = cosine - self.angle_targets
        energy += 4.*float(err@err)
        gu = 8*err[:, None] * (v/(un*vn)[:, None] - cosine[:, None]*u/un[:, None]**2)
        gv = 8*err[:, None] * (u/(un*vn)[:, None] - cosine[:, None]*v/vn[:, None]**2)
        np.add.at(grad, ia, gu); np.add.at(grad, ic, gv)
        np.add.at(grad, ib, -gu-gv)
        if closure_weight:
            a, b = self.links.T
            v = xy[a]-xy[b]
            err = np.sum(v*v, axis=1)-1
            energy += closure_weight*float(err@err)
            force = (4*closure_weight*err)[:, None]*v
            np.add.at(grad, a, force); np.add.at(grad, b, -force)
        pg = self.pose_gradient(grad, rot).reshape(-1, 3)
        # Weak shape tether fixes the translational/rotational gauge while
        # leaving large collective motions free. It is not a ring template.
        err = pose[:, :2] - self.centres_target
        energy += .025*float(np.sum(err*err))
        pg[:, :2] += .05*err
        if angle_weight:
            slack = self.angle_slack(x)
            bad = np.minimum(slack, 0)
            energy += angle_weight*float(bad@bad)
            pg += (2*angle_weight*(bad @ self.angle_slack_jac(x))).reshape(-1, 3)
        self.evaluations += 1
        return energy, pg.ravel()

    def solve(self, maxiter=350, start=None, angle_constraints=False, deadline=None):
        # Relax initially infeasible poses, then impose every peptide bond
        # as a simultaneous equality (no post-hoc open-chain closing seam).
        def out_of_time():
            return deadline is not None and time.monotonic() > deadline
        if start is None:
            warm = minimize(lambda x: self.objective(x, 40.), self.initial,
                            jac=True, method='L-BFGS-B',
                            options={'maxiter': maxiter, 'ftol': 1e-10})
            start = warm.x
        if angle_constraints and not out_of_time():
            warm = minimize(lambda x: self.objective(x, 200., 200.), start,
                            jac=True, method='L-BFGS-B',
                            options={'maxiter': maxiter, 'ftol': 1e-10})
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
        constraints = [{'type': 'eq', 'fun': self.closure, 'jac': self.closure_jac}]
        if angle_constraints:
            constraints.append({'type': 'ineq', 'fun': self.angle_slack,
                                'jac': self.angle_slack_jac})
        radius = max(10., sum(f.span+1 for f in self.frames))
        bounds = [(-radius, radius), (-radius, radius), (None, None)] * self.k
        slsqp_maxiter = maxiter
        if deadline is not None:
            # SLSQP is a single C call; scale its iteration cap to the
            # remaining wall clock (conservative ~.05 s per iteration).
            slsqp_maxiter = max(40, min(maxiter,
                int(max(0., deadline - time.monotonic()) / .05)))
        fit = minimize(fun, start, jac=jac, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': slsqp_maxiter, 'ftol': 1e-9})
        return fit.x, {'converged': bool(fit.success), 'message': str(fit.message),
                       'iterations': int(fit.nit), 'evaluations': self.evaluations,
                       'energy': float(fit.fun),
                       'max_link_error': float(np.max(np.abs(self.closure(fit.x)))),
                       'min_angle_slack': float(self.angle_slack(fit.x).min())}


    def defect_rank(self, x):
        from .geometry import _ring_intrusions
        xy, _ = self.coordinates(x)
        # Crossings and intrusions are censused at the acceptance gate's
        # render scale (bond length 30) so the refinement loop and the
        # final gate can never disagree on a grazing bond pair: near-zero
        # cross products round differently at different scales.
        xy_gate = xy * 30.
        b = self.bonds
        i, j = np.triu_indices(len(b), 1)
        mask = (b[i, 0] != b[j, 0]) & (b[i, 0] != b[j, 1]) & \
               (b[i, 1] != b[j, 0]) & (b[i, 1] != b[j, 1])
        i, j = i[mask], j[mask]
        p, q, r, t = xy_gate[b[i, 0]], xy_gate[b[i, 1]], xy_gate[b[j, 0]], xy_gate[b[j, 1]]
        def cross(a, b): return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        # Same predicate as the acceptance gate's _seg_cross: sign flips
        # count, and a zero (bond endpoint exactly on the other bond)
        # counts too. The old strict-product test missed T-contacts and
        # near-touches that the gate then rejected after refinement had
        # already stopped.
        c1, c2 = cross(q-p, r-p), cross(q-p, t-p)
        c3, c4 = cross(t-r, p-r), cross(t-r, q-r)
        hits = ((c1 > 0) != (c2 > 0)) & ((c3 > 0) != (c4 > 0))
        implicated = set(self.owner[b[i[hits]].ravel()]) | set(self.owner[b[j[hits]].ravel()])
        adj = [set() for _ in xy]
        for a, c in self.bonds:
            adj[a].add(c); adj[c].add(a)
        rings = [list(r) for r in self.mol.GetRingInfo().AtomRings() if len(r) <= 8]
        intrusions = _ring_intrusions(dict(enumerate(map(tuple, xy_gate))), adj, rings)
        for h in intrusions:
            if 'atom' in h: implicated.add(self.owner[h['atom']])
            implicated.update(self.owner[h['ring']])
        # Rank exactly the same atom/label/bond clearances that the final
        # feasibility gate checks. An atom-atom-only rank can improve while
        # placing a carbonyl O on an aromatic bond (verified SEQ53 defect).
        sep = self.separation_constraints(x,np.arange(len(self.ai)),np.arange(len(self.ab_a)))
        violations_sep = sep < -1e-6
        aa_bad = violations_sep[:len(self.ai)]
        ab_bad = violations_sep[len(self.ai):len(self.ai)+len(self.ab_a)]
        implicated.update(self.owner[self.ai[aa_bad]])
        implicated.update(self.owner[self.aj[aa_bad]])
        implicated.update(self.owner[self.ab_a[ab_bad]])
        implicated.update(self.owner[self.ab_b[ab_bad]])
        implicated.update(self.owner[self.ab_c[ab_bad]])
        deficit = np.minimum(sep,0)
        slack = self.angle_slack(x)
        violations = slack < -1e-5
        for a, b, c in self.angles[violations[:len(self.angles)] | violations[len(self.angles):]]:
            implicated.update((self.owner[a], self.owner[b], self.owner[c]))
        rank = (int(hits.sum()) + len(intrusions), int(violations.sum()),
                int(violations_sep.sum()), float(deficit@deficit), float(self.objective(x)[0]))
        return rank, sorted(implicated)

    def conformer_moves(self, implicated, fine=False):
        """Proper rotations at achiral methylene hinges, never reflections.

        Reflection remains a WHOLE-residue operation. This separate local
        coordinate move rotates an attached rigid group around an achiral
        degree-two CH2, preserving every bond length, the hinge angle and
        all stereocentres. It supplies different internal residue conformers
        instead of treating a single native depiction as chemically rigid.
        """
        adj = [set() for _ in range(self.n)]
        for a,b in self.bonds:
            adj[int(a)].add(int(b));adj[int(b)].add(int(a))
        for k in implicated:
            frame = self.frames[k]
            core = set(frame.backbone)
            support = set(map(int,frame.atoms))
            for anchor in frame.backbone[1:-1]:
                if self.mol.GetAtomWithIdx(anchor).GetAtomicNum()!=6:
                    continue
                for pivot in sorted(adj[anchor]-core):
                    atom = self.mol.GetAtomWithIdx(pivot)
                    if atom.GetAtomicNum()!=6 or atom.GetDegree()!=2 or atom.IsInRing() or \
                            atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:
                        continue
                    end = next(iter(adj[pivot]-{anchor}))
                    bond = self.mol.GetBondBetweenAtoms(pivot,end)
                    if bond.GetBondType()!=Chem.BondType.SINGLE or bond.IsInRing():
                        continue
                    moving, stack = set(), [end]
                    while stack:
                        a = stack.pop()
                        if a==pivot or a in moving:
                            continue
                        moving.add(a)
                        stack.extend(adj[a]-moving-{pivot})
                    if not moving or not moving<=support or moving&core:
                        continue
                    a,b = self.local[anchor]-self.local[pivot],self.local[end]-self.local[pivot]
                    theta = math.atan2(a[0]*b[1]-a[1]*b[0],float(a@b))
                    angle = (-2*theta+np.pi)%(2*np.pi)-np.pi
                    if abs(angle)<1e-6:
                        continue
                    candidates = [angle]
                    if fine:
                        candidates = [math.radians(v) for v in (-12.,-6.,6.,12.)] + candidates
                    for delta in candidates:
                        resulting = abs((theta+delta+np.pi)%(2*np.pi)-np.pi)
                        if fine and not math.radians(108)-1e-8 <= resulting <= math.radians(138)+1e-8:
                            continue
                        yield k,pivot,sorted(moving),delta

    def attachment_moves(self,implicated):
        """Small proper rotations at an attachment, within valence-angle bounds.

        The backbone frame stays fixed. A chiral side chain moves as one
        oriented rigid group; there is no side-chain reflection. All contacts,
        including within the residue, are checked by the same hard gate.
        """
        adj=[set() for _ in range(self.n)]
        for a,b in self.bonds:
            adj[int(a)].add(int(b));adj[int(b)].add(int(a))
        for k in implicated:
            frame=self.frames[k];core=set(frame.backbone);support=set(map(int,frame.atoms))
            for pivot in frame.backbone[1:-1]:
                atom=self.mol.GetAtomWithIdx(pivot)
                if atom.GetAtomicNum()!=6 or atom.GetDegree()!=3 or \
                        atom.GetHybridization()!=Chem.HybridizationType.SP3:
                    continue
                for end in sorted(adj[pivot]-core):
                    bond=self.mol.GetBondBetweenAtoms(pivot,end)
                    if bond.GetBondType()!=Chem.BondType.SINGLE or bond.IsInRing():
                        continue
                    moving,stack=set(),[end]
                    while stack:
                        a=stack.pop()
                        if a==pivot or a in moving:continue
                        moving.add(a);stack.extend(adj[a]-moving-{pivot})
                    if not moving or not moving<=support or moving&core:continue
                    origin=self.local[pivot]
                    ray=self.local[end]-origin
                    for degrees in (-12.,-6.,-3.,3.,6.,12.):
                        angle=math.radians(degrees);co,si=math.cos(angle),math.sin(angle)
                        rotated=np.array([[co,-si],[si,co]])@ray
                        legal=True
                        for other in adj[pivot]-{end}:
                            v=self.local[other]-origin
                            cosine=float(v@rotated)/max(1e-12,float(np.linalg.norm(v)*np.linalg.norm(rotated)))
                            theta=math.degrees(math.acos(np.clip(cosine,-1,1)))
                            if not 105.-1e-6<=theta<=150.+1e-6:
                                legal=False;break
                        if legal:yield k,pivot,sorted(moving),angle

    def subtree_moves(self, implicated):
        """Whole-subtree rotations about acyclic bonds: template escapes.

        Rotating one side of an acyclic bond about its pivot preserves every
        intra-frame distance and angle exactly, and interface atoms (the
        frame ports and their bonded neighbours) never move, so closure and
        interface constraints are exact invariants — no re-solve needed.
        Backbone bonds are skipped because both of their sides contain an
        interface. This reaches placements small hinge perturbations cannot:
        whole-ring flips and sidechain chain U-turns.
        """
        adjacency = [set() for _ in range(self.n)]
        for a, b in self.bonds:
            adjacency[a].add(b); adjacency[b].add(a)
        ring_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                      for b in self.mol.GetBonds() if b.IsInRing()}
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
                for root, side_root in ((int(a), int(b)), (int(b), int(a))):
                    moving, stack = set(), [side_root]
                    while stack:
                        atom = stack.pop()
                        if atom == root or atom in moving:
                            continue
                        moving.add(atom)
                        # Stay inside this frame: the peptide links to other
                        # residues are not part of any rotatable subtree.
                        stack.extend((adjacency[atom] - moving) & support)
                    # A single-atom "subtree" rotated about its own position
                    # is a no-op; sides containing interface atoms are
                    # untouchable.
                    if len(moving) <= 1 or moving & interface:
                        continue
                    pivot = side_root
                    origin = self.local[pivot]
                    # A 30-degree grid: coarse swings miss wedged-pair
                    # escapes (a -60 degree rotation cleared a packing that
                    # 90/180-degree moves could not), while each trial costs
                    # only one defect rank.
                    for angle in (math.radians(v) for v in range(-180, 180, 30)):
                        yield k, sorted(moving), pivot, angle

    def template_refine(self, x, rounds=2, deadline=None, max_trials=500):
        """Descent over subtree-template moves with exact invariant checks.

        Poses never change; only a residue's internal drawing does, so every
        trial costs one defect rank and cannot touch closure or interface
        feasibility. Tries whole-sidechain rotations when the smaller hinge
        and attachment moves have already failed.
        """
        rank, implicated = self.defect_rank(x)
        accepted, trials = [], 0
        for _ in range(rounds):
            best = None
            for k, ids, pivot, angle in self.subtree_moves(implicated):
                if trials >= max_trials or (deadline is not None and
                                            time.monotonic() > deadline):
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
            accepted.append({'residue': int(k), 'pivot': int(pivot),
                             'angle_deg': float(np.degrees(angle)),
                             'operation': 'subtree_rotation', 'determinant': 1})
            rank, implicated = self.defect_rank(x)
            if rank[:3] == (0, 0, 0):
                break
        return x, {'template_trials': trials, 'template_moves': accepted,
                   'defect_rank': list(rank)}

    def conformer_refine(self,x,rounds=3,deadline=None):
        rank, implicated = self.defect_rank(x)
        accepted, trials = [],0
        for _ in range(rounds):
            best = None
            for k,pivot,ids,angle in (list(self.conformer_moves(implicated,fine=True)) +
                                      list(self.attachment_moves(implicated))):
                if deadline is not None and time.monotonic() > deadline:
                    break
                backup = self.local[ids].copy()
                origin = self.local[pivot].copy()
                co,si = math.cos(angle),math.sin(angle)
                rotation = np.array([[co,-si],[si,co]])
                self.local[ids] = (self.local[ids]-origin)@rotation.T+origin
                candidate = x.copy()
                candidate_rank,_ = self.defect_rank(candidate)
                # Port atoms and their local neighbours stay fixed, so
                # closure/interface feasibility is unchanged. Evaluate the
                # exact contact objective locally before a global re-solve.
                trials += 1
                if candidate_rank<rank and (best is None or candidate_rank<best[0]):
                    best = (candidate_rank,candidate.copy(),ids,self.local[ids].copy(),k,pivot,angle)
                self.local[ids] = backup
                if best is not None and best[0][:3]==(0,0,0):
                    break
            if best is None:
                break
            rank,x,ids,local,k,pivot,angle = best
            self.local[ids] = local
            accepted.append({'residue':int(k),'hinge':int(pivot),
                             'angle_deg':float(np.degrees(angle)),
                             'operation':'proper_rotation','determinant':1,
                             'hinge_angle_band_deg':[108,138]})
            rank,implicated = self.defect_rank(x)
            if rank[:3]==(0,0,0):
                break
        return x,{'trials':trials,'accepted':accepted}

    def discrete_refine(self, x, rounds=4, deadline=None, max_trials=14):
        """Mixed discrete/continuous descent; reflect COMPLETE frames.

        A reflection is an exact isometry about the residue N--CO chord;
        the closed link endpoints remain unchanged. Each trial re-solves
        all residue poses jointly, retaining a lexicographic defect minimum.
        Trials are capped: a heavily implicated large ring can otherwise
        spend the whole layout budget re-solving every residue.
        """
        rank, implicated = self.defect_rank(x)
        trials, accepted = 0, []
        for _ in range(rounds):
            best = None
            for k in implicated:
                if trials >= max_trials or (deadline is not None and
                                            time.monotonic() > deadline):
                    break
                ids = self.frames[k].atoms
                self.local[ids, 1] *= -1
                candidate, rep = self.solve(maxiter=180, start=x, angle_constraints=True,
                                            deadline=deadline)
                candidate_rank, candidate_bad = self.defect_rank(candidate)
                if candidate_rank[0] == 0 and candidate_rank[1] == 0 and candidate_rank[2] > 0:
                    contact_x, _ = self.clearance_polish(candidate, rounds=2)
                    contact_rank, contact_bad = self.defect_rank(contact_x)
                    if contact_rank < candidate_rank:
                        candidate, candidate_rank, candidate_bad = contact_x, contact_rank, contact_bad
                trials += 1
                if rep['max_link_error'] < 1e-7 and candidate_rank < rank and \
                        (best is None or candidate_rank < best[0]):
                    best = (candidate_rank, candidate.copy(), k, candidate_bad)
                self.local[ids, 1] *= -1
            if best is None:
                break
            rank, x, k, implicated = best
            self.local[self.frames[k].atoms, 1] *= -1
            accepted.append(int(k))
            if rank[0] == 0 and rank[1] == 0 and rank[2] == 0:
                break
        return x, {'reflection_trials': trials, 'whole_residue_reflections': accepted,
                   'defect_rank': list(rank)}


def residue_pose_layout(mol: Chem.Mol, seed: int = 0, maxiter: int = 350, deadline=None):
    work = Chem.Mol(mol)
    try:
        frames, order = peptide_frames(work)
        problem = PoseProblem(work, frames, order, seed)
    except ValueError as e:
        return None, {'ok': False, 'error': str(e)}
    def expired():
        return deadline is not None and time.monotonic() > deadline
    # Share ladder over the total budget. Every stage stops at its rung, so
    # load spikes shrink every phase proportionally instead of letting the
    # last, cheap, pass-deciding stages starve.
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
    if problem.k >= 16:
        from .kinematics import port_seeds
        candidates = port_seeds(problem)
        native = problem.local.copy()
        best = None
        seed_reports = []
        skipped_candidates = 0
        # Seed search may only claim part of the budget: its candidates are
        # heuristics, and the feasibility work happens in the refine stages.
        beam_deadline = rung(.40)
        trial_maxiter = min(maxiter, 200)
        for candidate in candidates:
            if best is not None and beam_deadline is not None and \
                    time.monotonic() > beam_deadline:
                skipped_candidates += 1
                continue
            problem.local[:] = native
            for k, sign in enumerate(candidate.signs):
                problem.local[frames[k].atoms, 1] *= sign
            problem.initial = candidate.poses.ravel().copy()
            problem.centres_target = candidate.poses[:, :2].copy()
            x, initial_report = problem.solve(trial_maxiter, deadline=beam_deadline)
            x, polish = problem.solve(trial_maxiter, start=x, angle_constraints=True,
                                      deadline=beam_deadline)
            rank = problem.defect_rank(x)[0]
            feasible_ports = (polish['max_link_error'] < 1e-7 and problem.angle_slack(x).min() >= -1e-6)
            key = (not feasible_ports, *rank)
            seed_reports.append({'score': candidate.score, 'defect_rank': list(rank),
                                 'feasible_ports': bool(feasible_ports)})
            if best is None or key < best[0]:
                best = (key, x.copy(), problem.local.copy(), candidate.poses.copy(),
                        initial_report, polish)
        _, x, local, poses, initial_report, polish = best
        problem.local[:] = local
        problem.initial = poses.ravel().copy()
        problem.centres_target = poses[:, :2].copy()
        report = {'initial_solve': initial_report, 'polish': polish,
                  'initialization': 'interface_direction_beam', 'seed_candidates': seed_reports}
        if skipped_candidates:
            report['skipped_seed_candidates'] = skipped_candidates
    else:
        x,initial_report = problem.solve(maxiter, deadline=deadline)
        report = {'initial_solve':initial_report, 'initialization': 'asymmetric_polygon'}
        if not expired():
            polished,polish = problem.solve(maxiter,start=x,angle_constraints=True,
                                            deadline=deadline)
            if polish['max_link_error']<1e-7 and problem.defect_rank(polished)[0]<problem.defect_rank(x)[0]:
                x = polished
            report['polish'] = polish
        else:
            report['polish'] = {'error': 'deadline_exceeded'}
    if before(rung(.40)):
        x,report['early_conformer_refine'] = phase('early_conformer_refine',
            problem.conformer_refine, x, rounds=6, deadline=rung(.55))
    else:
        report['early_conformer_refine'] = {'trials':0,'accepted':[],
                                            'error':'deadline_exceeded'}
    entry_rank = problem.defect_rank(x)[0]
    # Whole-residue reflections exist to untangle crossings and interface
    # angles; a contacts-only state gains nothing from their re-solves and
    # the contact stages below are the purpose-built tool for it.
    contacts_only = entry_rank[0] == 0 and entry_rank[1] == 0
    if entry_rank[:3] != (0,0,0) and not contacts_only and before(rung(.58)):
        x,discrete = phase('discrete_refine',
            problem.discrete_refine, x, deadline=rung(.75))
        report.update(discrete)
        if problem.angle_slack(x).min() >= -1e-6 and before(rung(.78)):
            contact_x,contact_rep = phase('contact_polish',
                problem.clearance_polish, x, deadline=rung(.90))
            if problem.defect_rank(contact_x)[0]<problem.defect_rank(x)[0]:
                x = contact_x
            report['contact_polish'] = contact_rep
    elif contacts_only and before(rung(.60)):
        # Straight to the contact stages with the time reflections would
        # have burned.
        contact_x,contact_rep = phase('contact_polish',
            problem.clearance_polish, x, deadline=rung(.85))
        if problem.defect_rank(contact_x)[0]<problem.defect_rank(x)[0]:
            x = contact_x
        report['contact_polish'] = contact_rep
    else:
        report.update({'reflection_trials':0,'whole_residue_reflections':[]})
    if before(rung(.88)):
        x,report['internal_conformer_refine'] = phase('internal_conformer_refine',
            problem.conformer_refine, x, rounds=6, deadline=rung(.88))
    else:
        report['internal_conformer_refine'] = {'trials':0,'accepted':[],
                                               'error':'deadline_exceeded'}
    rank = problem.defect_rank(x)[0]
    if rank[:3] != (0, 0, 0) and before(rung(.88)):
        # Whole-sidechain rotations: the escape family for packings the
        # hinge/attachment moves cannot rearrange. Poses stay fixed, so
        # trials cost one defect rank each.
        x,report['template_refine'] = phase('template_refine',
            problem.template_refine, x, rounds=2, deadline=rung(.94))
        rank = problem.defect_rank(x)[0]
    # Alternating tail: each repair shifts which pairs are wedged, so a
    # failed repair often leaves a state a fresh template pass can escape.
    tail = []
    tail_round = 0
    while rank[:2] == (0, 0) and rank[2] > 0 and not expired() and tail_round < 4:
        if before(rung(.94 + .02 * tail_round)):
            x,tref = problem.template_refine(x, rounds=1, deadline=rung(.96 + .01 * tail_round))
            tail.append({'template_trials': tref['template_trials'],
                         'template_moves': len(tref['template_moves'])})
        if not expired():
            x,rep_r = problem.contact_repair(x, deadline=deadline)
            tail.append({'repair': rep_r.get('error') or rep_r.get('converged')})
        rank = problem.defect_rank(x)[0]
        tail_round += 1
    if tail:
        report['tail_alternation'] = tail
    report['defect_rank'] = list(rank)
    if phase_seconds:
        report['phase_seconds'] = phase_seconds
    report['max_link_error'] = float(np.max(np.abs(problem.closure(x))))
    xy, _ = problem.coordinates(x)
    conf = Chem.Conformer(work.GetNumAtoms())
    conf.Set3D(False)
    for i, (a, b) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(a*30), float(b*30), 0.))
    work.RemoveAllConformers(); work.AddConformer(conf, assignId=True)
    from .geometry import _layout_metrics, DEFAULT_CONFIG
    from .chemistry import _adjacency
    positions = {i:(float(a*30),float(b*30)) for i,(a,b) in enumerate(xy)}
    small_rings = [list(r) for r in work.GetRingInfo().AtomRings() if len(r)<=8]
    metrics = _layout_metrics(positions, _adjacency(work), order,
        [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in work.GetBonds()],
        DEFAULT_CONFIG, small_rings=small_rings)
    link_error = float(np.max(np.abs(problem.closure(x))))
    angle_slack = float(problem.angle_slack(x).min())
    all_slack=problem.separation_constraints(x,np.arange(len(problem.ai)),np.arange(len(problem.ab_a)))
    separation_slack=float(all_slack.min())
    contact_slack=float(all_slack[:len(problem.ai)+len(problem.ab_a)].min())
    centre_diagnostic=xy[order].mean(axis=0)
    cavity_slack=float((np.linalg.norm(xy[problem.side_atoms]-centre_diagnostic,axis=1)-problem.cavity_r).min())
    lengths = np.linalg.norm(xy[problem.bonds[:,0]]-xy[problem.bonds[:,1]],axis=1)
    bond_error = float(np.abs(lengths-1).max())
    centre = xy[order].mean(axis=0)
    contour = xy[order]-centre
    asymmetry = None
    if len(order)%2 == 0:
        half = len(order)//2
        asymmetry = float(np.sqrt(np.mean(np.sum((contour[:half]+contour[half:])**2,axis=1))) /
                          max(1e-9,np.linalg.norm(contour,axis=1).mean()))
    stereo_ok = stereo_roundtrip_ok(work)
    feasible = (link_error < 1e-7 and angle_slack >= -1e-6 and
                separation_slack >= -1e-6 and bond_error < 1e-6 and
                metrics['bond_crossings']==0 and metrics['n_ring_intrusions']==0 and stereo_ok)
    report.update({'ok': True, 'engine': 'residue_pose', 'seed': seed,
        'residues':len(frames), 'stereo_roundtrip':stereo_ok, 'metrics':metrics,
        'constraints':{'max_link_error':link_error,'min_angle_slack':angle_slack,
                       'min_global_separation_slack':separation_slack,
                       'min_contact_slack':contact_slack,
                       'central_occupancy_diagnostic':cavity_slack,
                       'central_occupancy_is_gate':False,
                       'max_bond_relative_error':bond_error},
        'half_turn_asymmetry':asymmetry,
        'energy':float(problem.objective(x)[0]),
        'converged':feasible, 'qa_pass':feasible,
        'solution_pose':x.tolist(), 'solution_local':problem.local.tolist(),
        'training_approved':False,
        'training_note':'Geometry feasibility alone does not verify rendered glyph clearance or recognition accuracy.'})
    return work, report

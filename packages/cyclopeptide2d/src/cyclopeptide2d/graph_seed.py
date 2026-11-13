"""Fit registered rigid fragments to a full molecular graph embedding."""

import math
import time
from dataclasses import replace

import numpy as np
from rdkit.Chem import rdDepictor


def open_chain_refine(reference, *, record=None, seed, deadline=None, target_poses=None):
    from .frames import registered_frames
    from .kinematics import port_seeds
    from .solver import PoseProblem

    if reference.ring or reference.crosslinks:
        raise ValueError("open_chain_refine_requires_one_acyclic_chain")
    if record is None:
        frames = [
            replace(frame, local=frame.local.copy(), port_cosines=dict(frame.port_cosines))
            for frame in reference.frames
        ]
    else:
        frames, graph = registered_frames(record)
        if graph.cycle_rank or len(graph.paths) != 1:
            raise ValueError("open_chain_refine_requires_one_acyclic_chain")
    problem = PoseProblem(
        reference.mol, frames, reference.order, seed=seed, ring=False, links=reference.links
    )
    targets = [("extended", problem.initial.copy())]
    if target_poses is not None:
        targets.append(("embedding", np.asarray(target_poses).reshape(-1).copy()))
    native = problem.local.copy()
    attempts = []
    for family, target in targets:
        if deadline is not None and time.monotonic() >= deadline:
            break
        problem.local[:] = native
        problem.initial = target
        family_deadline = deadline
        if deadline is not None and len(targets) > 1 and family == "extended":
            family_deadline = time.monotonic() + 0.65 * (deadline - time.monotonic())
        candidates = port_seeds(problem, width=256, count=6, crossing_penalty=0.3,
                                deadline=family_deadline)
        ranked = []
        for candidate in candidates:
            problem.local[:] = native
            for frame, sign in zip(frames, candidate.signs, strict=True):
                problem.local[frame.atoms, 1] *= sign
            rank, _ = problem.defect_rank(candidate.poses.ravel())
            ranked.append((rank, candidate))
        ranked.sort(key=lambda item: item[0])
        for index, (_, candidate) in enumerate(ranked):
            if family_deadline is not None and time.monotonic() >= family_deadline:
                break
            problem.local[:] = native
            for frame, sign in zip(frames, candidate.signs, strict=True):
                problem.local[frame.atoms, 1] *= sign
            x = candidate.poses.ravel().copy()
            problem.initial = x.copy()
            problem.centres_target = candidate.poses[:, :2].copy()
            trial_deadline = family_deadline
            if family_deadline is not None:
                trial_deadline = time.monotonic() + (family_deadline - time.monotonic()) / math.sqrt(
                    len(ranked) - index
                )
            rank, _ = problem.defect_rank(x)
            if rank[:3] != (0, 0, 0):
                x, _ = problem.conformer_refine(x, rounds=6, deadline=trial_deadline)
            rank, _ = problem.defect_rank(x)
            if rank[:3] != (0, 0, 0):
                x, _ = problem.clearance_polish(x, deadline=trial_deadline)
            rank, _ = problem.defect_rank(x)
            error = float(np.max(np.abs(problem.closure(x)), initial=0.0))
            angle_slack = float(np.min(problem.angle_slack(x), initial=0.0))
            attempts.append(
                {"family": family, "defect_rank": list(rank), "max_link_error": error,
                 "min_angle_slack": angle_slack}
            )
            if rank[:3] == (0, 0, 0) and error < 1e-7 and angle_slack >= -1e-6:
                return problem, x, {"accepted": True, "attempts": attempts}
    return None, None, {"accepted": False, "attempts": attempts}


def graph_seed(molecule, frames, links, *, preserve_embedding=False, sample_seed=None):
    rdDepictor.Compute2DCoords(
        molecule,
        forceRDKit=True,
        useRingTemplates=True,
        nFlipsPerSample=3 if sample_seed is not None else 0,
        nSample=100 if sample_seed is not None else 0,
        sampleSeed=0 if sample_seed is None else int(sample_seed % 2147483647),
    )
    xy = np.asarray(molecule.GetConformer().GetPositions())[:, :2]
    lengths = [
        np.linalg.norm(xy[bond.GetBeginAtomIdx()] - xy[bond.GetEndAtomIdx()])
        for bond in molecule.GetBonds()
    ]
    xy = xy / np.median(lengths)
    if preserve_embedding:
        from .frames import _unit_tree_bonds

        xy = _unit_tree_bonds(molecule, xy, set())
    anchors = {atom for pair in links for atom in pair}
    poses = []
    for frame in frames:
        if preserve_embedding:
            origin = (xy[frame.n_atom] + xy[frame.c_atom]) / 2
            direction = xy[frame.c_atom] - xy[frame.n_atom]
            angle = math.atan2(direction[1], direction[0])
            co, si = math.cos(angle), math.sin(angle)
            frame.local[:] = (xy[frame.atoms] - origin) @ np.array([[co, -si], [si, co]])
            support = set(frame.atoms)
            for end, adjacent in frame.port_cosines:
                external = [
                    atom.GetIdx()
                    for atom in molecule.GetAtomWithIdx(end).GetNeighbors()
                    if atom.GetIdx() not in support
                ]
                if len(external) != 1:
                    raise ValueError("non_chain_interface")
                u, v = xy[adjacent] - xy[end], xy[external[0]] - xy[end]
                frame.port_cosines[end, adjacent] = float(
                    u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                )
            poses.append([*origin, angle])
            continue
        weights = np.array(
            [4.0 if atom in anchors or atom in frame.backbone else 1.0 for atom in frame.atoms]
        )
        target = xy[frame.atoms]
        centre = np.average(target, axis=0, weights=weights)
        best = None
        for sign in (1, -1):
            local = frame.local * [1, sign]
            origin = np.average(local, axis=0, weights=weights)
            u, _, vt = np.linalg.svd((local - origin).T @ ((target - centre) * weights[:, None]))
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            rotation = u @ vt
            translation = centre - origin @ rotation
            error = float(
                np.sum(weights * ((local @ rotation + translation - target) ** 2).sum(axis=1))
            )
            if best is None or error < best[0]:
                best = error, local, rotation, translation
        _, local, rotation, translation = best
        frame.local[:] = local
        poses.append([*translation, math.atan2(rotation[0, 1], rotation[0, 0])])
    return np.array(poses)


def sampled_graph_refine(problem, *, seed, maxiter, deadline=None):
    attempts = []
    for sample_seed in dict.fromkeys((seed, *range(8))):
        if deadline is not None and time.monotonic() >= deadline:
            break
        frames = [
            replace(frame, local=frame.local.copy(), port_cosines=dict(frame.port_cosines))
            for frame in problem.frames
        ]
        poses = graph_seed(
            problem.mol,
            frames,
            problem.links,
            preserve_embedding=True,
            sample_seed=sample_seed,
        )
        candidate = type(problem)(
            problem.mol,
            frames,
            problem.order,
            seed=seed,
            ring=False,
            links=problem.links,
            crosslinks=problem.crosslinks,
            initial_poses=poses,
        )
        x = candidate.initial
        rank, _ = candidate.defect_rank(x)
        if rank[:3] != (0, 0, 0):
            x, _ = candidate.solve(
                maxiter=min(maxiter, 200), start=x, angle_constraints=True, deadline=deadline
            )
            rank, _ = candidate.defect_rank(x)
        closed = np.max(np.abs(candidate.closure(x)), initial=0.0) < 1e-7
        attempts.append({"sample_seed": int(sample_seed), "defect_rank": list(rank)})
        if rank[:3] == (0, 0, 0) and closed:
            return candidate, x, {"accepted": True, "attempts": attempts}
    return problem, None, {"accepted": False, "attempts": attempts}

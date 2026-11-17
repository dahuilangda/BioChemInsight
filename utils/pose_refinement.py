"""Bounded 2D sketch refinement; source coordinates are a constraint, not a seed.

A sparse molecular distance/angle objective with analytical gradients is solved
inside a per-atom trust region. No atom permutation, reflection, canonical
orientation, graph edit, or macrocycle template substitution is performed.
"""
from __future__ import annotations
import itertools
import math
import time
import threading
import numpy as np


def refine_pose(mol, *, max_iterations=48, max_shift=0.32):
    from rdkit import Chem
    from scipy.optimize import minimize
    started = time.perf_counter()
    report = {'algorithm': 'bounded_source_pose_v1', 'accepted': False}
    if mol is None or not mol.GetNumConformers() or mol.GetNumAtoms() < 2:
        return {**report, 'status': 'missing_coordinates'}
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    if not np.isfinite(coordinates).all():
        return {**report, 'status': 'nonfinite_coordinates'}
    if np.any(np.abs(coordinates[:, 2]) > 1e-6):
        return {**report, 'status': 'not_a_planar_source_pose'}
    reference = coordinates[:, :2].copy()
    if not np.isfinite(reference).all():
        return {**report, 'status': 'nonfinite_coordinates'}
    edges = np.asarray([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=int).reshape(-1, 2)
    if not len(edges): return {**report, 'status': 'no_bonds'}
    lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    positive = lengths[lengths > 1e-6]
    if not len(positive): return {**report, 'status': 'degenerate_coordinates'}
    bond_scale = float(np.median(positive))
    if np.any(lengths < bond_scale * .05):
        return {**report, 'status': 'collapsed_bond_requires_source_review'}
    origin = reference.mean(axis=0)
    source = (reference - origin) / bond_scale
    count = len(source)
    if count > 500: return {**report, 'status': 'size_budget_exceeded'}
    try: rings = list(Chem.GetSymmSSSR(mol))
    except Exception: rings = []
    ring_sets = [set(ring) for ring in rings]
    ring_atoms = set().union(*ring_sets) if ring_sets else set()
    macro_atoms = set().union(*(ring for ring in ring_sets if len(ring) > 8)) if any(len(ring) > 8 for ring in ring_sets) else set()
    pairs, targets, weights = list(map(tuple, edges)), [1.] * len(edges), [90.] * len(edges)
    turns = []; angle_specs = []
    for atom in mol.GetAtoms():
        centre = atom.GetIdx()
        neighbours = [a.GetIdx() for a in atom.GetNeighbors()]
        for left, right in itertools.combinations(neighbours, 2):
            v, w = source[left] - source[centre], source[right] - source[centre]
            area = float(v[0] * w[1] - v[1] * w[0])
            if abs(area) > 0.08: turns.append((left, centre, right, area))
            if len(neighbours) > 3 or centre in macro_atoms: continue
            common_rings = [len(ring) for ring in ring_sets if {left, centre, right} <= ring]
            smallest = min(common_rings, default=0)
            if smallest > 8:
                # Macrocycle conformation belongs to the source/model. Do not
                # force cyclic peptides into a regular polygon or a small-ring template.
                continue
            if smallest:
                angle = math.pi * (smallest - 2) / smallest
            elif atom.GetHybridization() == Chem.HybridizationType.SP:
                angle = math.pi
            else:
                angle = 2 * math.pi / 3
            angle_specs.append((left, centre, right, math.cos(angle)))
    angle_indices = np.asarray([spec[:3] for spec in angle_specs], dtype=int).reshape(-1, 3)
    angle_cosines = np.asarray([spec[3] for spec in angle_specs])
    pairs = np.asarray(pairs, dtype=int); targets = np.asarray(targets); weights = np.asarray(weights)
    adjacency = np.eye(count, dtype=bool)
    adjacency[edges[:, 0], edges[:, 1]] = True; adjacency[edges[:, 1], edges[:, 0]] = True
    aa, bb = np.tril_indices(count, -1)
    nearby_pairs = ~adjacency[aa, bb] & (np.linalg.norm(source[aa] - source[bb], axis=1) < .62 + 2 * math.sqrt(2) * max_shift)
    repel = np.column_stack((aa[nearby_pairs], bb[nearby_pairs]))
    anchors = np.asarray([3.5 if i in ring_atoms else 2. for i in range(count)])
    # Spatial broad phase: points farther away cannot enter the clearance
    # region under the bounded displacement, so need not enter the solver.
    a = np.repeat(np.arange(count), len(edges)); i = np.tile(edges[:, 0], count); j = np.tile(edges[:, 1], count)
    eligible = ~adjacency[a, i] & ~adjacency[a, j]
    a, i, j = a[eligible], i[eligible], j[eligible]
    vector = source[j] - source[i]
    projection = np.clip(np.sum((source[a] - source[i]) * vector, axis=1) / np.maximum(np.sum(vector**2, axis=1), 1e-9), 0, 1)
    distance = np.linalg.norm(source[a] - source[i] - projection[:, None] * vector, axis=1)
    nearby = distance < .24 + 2 * math.sqrt(2) * max_shift
    atom_edges = np.column_stack((a[nearby], i[nearby], j[nearby]))
    def objective(flat, gradient=True):
        xy = flat.reshape(-1, 2); grad = np.zeros_like(xy)
        displacement = xy - source
        energy = float(np.sum(anchors[:, None] * displacement ** 2))
        grad += 2 * anchors[:, None] * displacement
        delta = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        residual = distance - targets
        energy += float(np.sum(weights * residual ** 2))
        force = (2 * weights * residual / distance)[:, None] * delta
        np.add.at(grad, pairs[:, 0], force); np.add.at(grad, pairs[:, 1], -force)
        if len(angle_indices):
            left, centre, right = angle_indices.T
            u, v = xy[left] - xy[centre], xy[right] - xy[centre]
            un = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
            vn = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            cosine = np.sum(u * v, axis=1) / (un * vn)
            error = cosine - angle_cosines
            energy += float(24 * np.dot(error, error))
            du = v / (un * vn)[:, None] - cosine[:, None] * u / (un ** 2)[:, None]
            dv = u / (un * vn)[:, None] - cosine[:, None] * v / (vn ** 2)[:, None]
            gu, gv = 48 * error[:, None] * du, 48 * error[:, None] * dv
            np.add.at(grad, left, gu); np.add.at(grad, right, gv); np.add.at(grad, centre, -gu-gv)
        if len(repel):
            delta = xy[repel[:, 0]] - xy[repel[:, 1]]
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            gap = np.minimum(distance - .62, 0)
            energy += float(8 * np.dot(gap, gap))
            force = (16 * gap / distance)[:, None] * delta
            np.add.at(grad, repel[:, 0], force); np.add.at(grad, repel[:, 1], -force)
        if len(atom_edges):
            a, i, j = atom_edges.T
            vector = xy[j] - xy[i]
            projection = np.clip(np.sum((xy[a] - xy[i]) * vector, axis=1) / np.maximum(np.sum(vector ** 2, axis=1), 1e-9), 0, 1)
            delta = xy[a] - xy[i] - projection[:, None] * vector
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            gap = np.minimum(distance - .24, 0)
            energy += float(5 * np.dot(gap, gap))
            force = (10 * gap / distance)[:, None] * delta
            np.add.at(grad, a, force); np.add.at(grad, i, -(1 - projection)[:, None] * force); np.add.at(grad, j, -projection[:, None] * force)
        return energy, grad.ravel()
    aa, bb = np.tril_indices(len(edges), -1)
    disjoint = (edges[aa, 0] != edges[bb, 0]) & (edges[aa, 0] != edges[bb, 1]) & (edges[aa, 1] != edges[bb, 0]) & (edges[aa, 1] != edges[bb, 1])
    edge_pairs = np.column_stack((aa[disjoint], bb[disjoint]))
    def crossings(xy):
        if not len(edge_pairs): return 0
        first, second = edges[edge_pairs[:, 0]], edges[edge_pairs[:, 1]]
        a, b = xy[first[:, 0]], xy[first[:, 1]]
        c, d = xy[second[:, 0]], xy[second[:, 1]]
        cross = lambda v, w: v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
        return int(np.sum((cross(b-a, c-a) * cross(b-a, d-a) < -1e-9) & (cross(d-c, a-c) * cross(d-c, b-c) < -1e-9)))
    initial, initial_grad = objective(source.ravel())
    if initial < 1e-5 or len(macro_atoms) == count:
        candidate = source.copy(); report['status'] = 'macrocycle_pose_fixed' if macro_atoms else 'already_clean'
    else:
        radius = np.asarray([0. if i in macro_atoms else max_shift for i in range(count)])[:, None]
        solution = minimize(objective, source.ravel(), jac=True, method='L-BFGS-B', bounds=list(zip((source - radius).ravel(), (source + radius).ravel())), options={'maxiter': max_iterations, 'maxls': 12, 'ftol': 1e-8, 'gtol': 1e-5})
        candidate = solution.x.reshape(-1, 2)
        report.update(status='refined', iterations=int(getattr(solution, 'nit', 0)))
    if not macro_atoms:
        # Remove only incidental rigid drift using a proper rotation (det +1).
        centred = candidate - candidate.mean(axis=0)
        u, _, vt = np.linalg.svd(centred.T @ source)
        correction = np.eye(2); correction[-1, -1] = np.linalg.det(u @ vt)
        candidate = centred @ u @ correction @ vt
    report['fixed_macrocycle_atoms'] = len(macro_atoms)
    before_crossings = crossings(source)
    full_step = candidate - source
    accepted = False
    for fraction in (1., .5, .25, .125, .0625, .03125):
        candidate = source + fraction * full_step
        displacement = np.linalg.norm(candidate - source, axis=1)
        energy = float(objective(candidate.ravel())[0])
        after_crossings = crossings(candidate)
        turns_preserved = True
        for left, centre, right, area in turns:
            v, w = candidate[left] - candidate[centre], candidate[right] - candidate[centre]
            if float(v[0] * w[1] - v[1] * w[0]) * area <= 0:
                turns_preserved = False; break
        if turns_preserved and after_crossings <= before_crossings and displacement.max() <= max_shift * math.sqrt(2) + 1e-6 and energy <= initial + 1e-7:
            accepted = True; break
    report.update(rms_shift_bonds=float(np.sqrt(np.mean(displacement ** 2))), max_shift_bonds=float(displacement.max()), objective_before=float(initial), objective_after=energy, crossings_before=before_crossings, crossings_after=after_crossings, step_fraction=fraction)
    if not accepted:
        return {**report, 'status': 'rejected_pose_or_topology_budget'}
    # Coordinate updates on a copy; never partially mutate on rejection.
    edited = Chem.Mol(mol)
    conf = edited.GetConformer(); conf.Set3D(False)
    for i, point in enumerate(candidate * bond_scale + origin): conf.SetAtomPosition(i, (float(point[0]), float(point[1]), 0.))
    try:
        before = Chem.MolToSmiles(mol, isomericSmiles=True)
        roundtrip = Chem.MolFromMolBlock(Chem.MolToMolBlock(edited), removeHs=False)
        if roundtrip is None or Chem.MolToSmiles(roundtrip, isomericSmiles=True) != before:
            return {**report, 'status': 'rejected_stereochemical_roundtrip'}
    except Exception:
        return {**report, 'status': 'rejected_unverifiable_stereochemistry'}
    destination = mol.GetConformer(); destination.Set3D(False)
    for i in range(count): destination.SetAtomPosition(i, conf.GetAtomPosition(i))
    report.update(accepted=True, elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
    return report


def transfer_source_pose(smiles, reference_mol, *, minimum_coverage=.65):
    """Transfer exact or sufficiently large common-core image coordinates.

    The graph always comes from the requested SMILES. A timeout or inadequate
    correspondence returns None; canonical re-layout is not called a source pose.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor, rdFMCS
    from rdkit.Geometry import Point2D
    target = Chem.MolFromSmiles(smiles)
    if target is None or reference_mol is None or not reference_mol.GetNumConformers(): return None
    reference = Chem.Mol(reference_mol)
    try: Chem.SanitizeMol(reference)
    except Exception:
        # Reference chemistry is not adopted; malformed/aromatic bond orders
        # may still carry valid image coordinates for element/connectivity mapping.
        try: reference.UpdatePropertyCache(strict=False); Chem.GetSymmSSSR(reference)
        except Exception: return None
    xy = np.asarray(reference.GetConformer().GetPositions())[:, :2]
    if not np.isfinite(xy).all(): return None
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in reference.GetBonds()]
    lengths = [np.linalg.norm(xy[a] - xy[b]) for a, b in edges]
    lengths = [value for value in lengths if value > 1e-6]
    if not lengths: return None
    xy = (xy - xy.mean(axis=0)) * (1.5 / np.median(lengths))
    direct = reference.GetSubstructMatch(target)
    if len(direct) == target.GetNumAtoms():
        mapping = list(enumerate(direct))
    else:
        if max(target.GetNumAtoms(), reference.GetNumAtoms()) > 500: return None
        mapping = []
        modes = [(rdFMCS.BondCompare.CompareOrderExact, True), (rdFMCS.BondCompare.CompareAny, True)]
        from collections import Counter
        if any(len(ring) > 8 for ring in Chem.GetSymmSSSR(target)) and Counter(a.GetAtomicNum() for a in target.GetAtoms()) == Counter(a.GetAtomicNum() for a in reference.GetAtoms()):
            # A corrected macrocycle closure can reuse the coordinate head's
            # atoms even if that head missed the closure edge. Chemistry still
            # comes entirely from the accepted graph, with strict pose validation.
            modes.append((rdFMCS.BondCompare.CompareOrderExact, False))
        for bond_comparison, ring_matching in modes:
            parameters = rdFMCS.MCSParameters()
            parameters.Timeout = 1
            parameters.BondTyper = bond_comparison
            parameters.AtomCompareParameters.MatchIsotope = True
            parameters.AtomCompareParameters.MatchFormalCharge = True
            parameters.AtomCompareParameters.RingMatchesRingOnly = ring_matching
            parameters.BondCompareParameters.RingMatchesRingOnly = ring_matching
            parameters.BondCompareParameters.CompleteRingsOnly = ring_matching
            match = rdFMCS.FindMCS([target, reference], parameters)
            if match.canceled or not match.smartsString: continue
            pattern = Chem.MolFromSmarts(match.smartsString)
            tm, rm = target.GetSubstructMatch(pattern), reference.GetSubstructMatch(pattern)
            coverage = minimum_coverage if ring_matching else max(.9, minimum_coverage)
            if tm and len(tm) == len(rm) and len(tm) >= max(3, target.GetNumAtoms() * coverage):
                mapping = list(zip(tm, rm)); break
        if not mapping: return None
    if len(mapping) == target.GetNumAtoms():
        conf = Chem.Conformer(target.GetNumAtoms()); conf.Set3D(False)
        for ti, ri in mapping: conf.SetAtomPosition(ti, (float(xy[ri, 0]), float(xy[ri, 1]), 0.))
        target.AddConformer(conf)
    else:
        rdDepictor.Compute2DCoords(target, canonOrient=False, coordMap={ti: Point2D(float(xy[ri, 0]), float(xy[ri, 1])) for ti, ri in mapping}, forceRDKit=True, bondLength=1.5)
        points = target.GetConformer()
        if any(np.linalg.norm(np.asarray([points.GetAtomPosition(ti).x, points.GetAtomPosition(ti).y]) - xy[ri]) > 1e-3 for ti, ri in mapping): return None
        # Optional attachment labels retain the direction of the corresponding
        # source dummy even when an incorrect intervening atom was removed.
        mapped = dict(mapping)
        for atom in target.GetAtoms():
            if atom.GetAtomicNum() or atom.GetIdx() in mapped or atom.GetDegree() != 1: continue
            originals = [a for a in reference.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() == atom.GetIsotope() and a.GetAtomMapNum() == atom.GetAtomMapNum()]
            neighbour = atom.GetNeighbors()[0].GetIdx()
            if len(originals) == 1 and neighbour in mapped:
                origin = xy[mapped[neighbour]]; direction = xy[originals[0].GetIdx()] - origin
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    point = origin + direction / norm * 1.5
                    points.SetAtomPosition(atom.GetIdx(), (float(point[0]), float(point[1]), 0.))
    positions = np.asarray(target.GetConformer().GetPositions())[:, :2]
    target_lengths = [np.linalg.norm(positions[b.GetBeginAtomIdx()] - positions[b.GetEndAtomIdx()]) for b in target.GetBonds()]
    if any(value < .75 or value > 2.7 for value in target_lengths): return None
    try:
        roundtrip = Chem.MolFromMolBlock(Chem.MolToMolBlock(target), removeHs=False)
        if roundtrip is None or Chem.MolToSmiles(roundtrip, isomericSmiles=True) != Chem.MolToSmiles(target, isomericSmiles=True): return None
    except Exception: return None
    target.SetProp('_BCIPoseSource', 'exact_source_mapping' if len(mapping) == target.GetNumAtoms() else 'constrained_source_core')
    return target


# Used at recognition time only, never during thumbnail rendering or editor open.
# The chemistry comes from the accepted recognizer; this auxiliary decoder supplies
# source image coordinates, which must pass the common-core and stereo gates.
_source_coordinate_cache = {}
_source_coordinate_lock = threading.RLock()


def recover_source_coordinates(smiles, image_path):
    import hashlib
    from pathlib import Path
    from rdkit import Chem
    digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    key = (smiles, digest)
    with _source_coordinate_lock:
        if key in _source_coordinate_cache: return _source_coordinate_cache[key]
        try:
            from utils.structure_recognition import _load_molnextr, extract_molblock
            from utils.pose_preserving_render import mol_from_molblock_tolerant
            model = _load_molnextr()
            prediction = model.predict_image_files([str(image_path)], return_atoms_bonds=True, return_confidence=False, batch_size=1, beam_size=1, n_best=1)[0]
            reference = mol_from_molblock_tolerant(extract_molblock(prediction))
            aligned = transfer_source_pose(smiles, reference, minimum_coverage=.9)
            result = Chem.MolToMolBlock(aligned) if aligned is not None else None
        except Exception:
            result = None
        if len(_source_coordinate_cache) >= 64: _source_coordinate_cache.pop(next(iter(_source_coordinate_cache)))
        _source_coordinate_cache[key] = result
        return result

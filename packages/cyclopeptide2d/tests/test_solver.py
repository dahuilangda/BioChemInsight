"""Chemical segmentation, rigid-frame invariants, closure and true gradients."""

import math

import numpy as np
from rdkit import Chem
from scipy.optimize._numdiff import approx_derivative

from cyclopeptide2d import validate_input
from cyclopeptide2d.solver import PoseProblem, peptide_frames, residue_pose_layout

SMI = "O=C1N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC)C(=O)N[C@@H](CO)1"


def test_chemical_frames_partition_all_atoms_and_preserve_chemistry():
    m = Chem.MolFromSmiles(SMI)
    before = Chem.MolToSmiles(m)
    frames, order, _ = peptide_frames(m)
    assert len(frames) == 3
    assert sorted(a for f in frames for a in f.atoms) == list(range(m.GetNumAtoms()))
    for f in frames:
        assert m.GetAtomWithIdx(f.n_atom).GetAtomicNum() == 7
        assert any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(m.GetAtomWithIdx(f.c_atom)).GetAtomicNum() == 8
            for b in m.GetAtomWithIdx(f.c_atom).GetBonds()
        )
    assert Chem.MolToSmiles(m) == before


def test_frame_motion_and_reflection_are_isometries():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order, seed=2)
    x = p.initial.copy()
    xy, _ = p.coordinates(x)
    ids = frames[0].atoms
    before = np.linalg.norm(xy[ids, None] - xy[ids], axis=2)
    ports = xy[[frames[0].n_atom, frames[0].c_atom]].copy()
    p.local[ids, 1] *= -1
    reflected, _ = p.coordinates(x)
    assert np.allclose(ports, reflected[[frames[0].n_atom, frames[0].c_atom]])
    assert np.allclose(before, np.linalg.norm(reflected[ids, None] - reflected[ids], axis=2))
    assert not np.allclose(xy[ids], reflected[ids])
    x.reshape(-1, 3)[0] += [2.0, -1.0, 0.7]
    moved, _ = p.coordinates(x)
    assert np.allclose(before, np.linalg.norm(moved[ids, None] - moved[ids], axis=2))


def test_analytic_derivatives_match_numerical_gradients():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x = p.initial + np.random.default_rng(3).normal(0, 0.03, len(p.initial))
    _, g = p.objective(x, 40.0)
    fd = approx_derivative(lambda z: p.objective(z, 40.0)[0], x, method="3-point").ravel()
    assert np.allclose(g, fd, atol=2e-6, rtol=1e-5)
    assert np.allclose(p.closure_jac(x), approx_derivative(p.closure, x), atol=1e-7)
    assert np.allclose(p.angle_slack_jac(x), approx_derivative(p.angle_slack, x), atol=1e-7)


def test_pruned_objective_matches_numerical_gradients_at_scattered_poses():
    """Pair pruning must stay exact wherever it drops rows, not only near seeds."""
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    rng = np.random.default_rng(9)

    ids = frames[0].atoms
    pivot = int(frames[0].backbone[1])
    co, si = np.cos(0.7), np.sin(0.7)
    rotation = np.array([[co, -si], [si, co]])
    origin = p.local[pivot].copy()
    p.local[ids] = (p.local[ids] - origin) @ rotation.T + origin
    pose = p.initial.reshape(-1, 3)
    pose[0::3] += rng.uniform(-6, 6, len(frames))
    pose[1::3] += rng.uniform(-6, 6, len(frames))
    x = pose.ravel()
    centres = pose[:, :2]
    radius = np.linalg.norm(p.local, axis=1)
    gap = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
    bound = gap[p.aa_fr_i, p.aa_fr_j] - radius[p.ai] - radius[p.aj]
    assert (bound >= p.aa_rest).sum() > 0, "scatter must exercise pruning"
    _, g = p.objective(x, 40.0, 40.0)
    fd = approx_derivative(lambda z: p.objective(z, 40.0, 40.0)[0], x, method="3-point").ravel()
    assert np.allclose(g, fd, atol=2e-6, rtol=1e-5)


def _place_crossing_bonds(p, angle_offset=math.pi * 0.45, parallel=False, offset=0.0):
    """Pose two frames so one bond of each shares a midpoint (plus offset).

    Returns (bond_i, bond_j, x). The bonds cross when parallel is False and
    offset is zero; a positive offset slides the second bond sideways so the
    closest-point branch stays smooth for finite differences."""
    i = j = None
    for a in range(len(p.bonds)):
        for b in range(a + 1, len(p.bonds)):
            fa, fb = p.bond_frame[a], p.bond_frame[b]
            if fa >= 0 and fb >= 0 and fa != fb:
                if not set(map(int, p.bonds[a])) & set(map(int, p.bonds[b])):
                    i, j = a, b
                    break
        if i is not None:
            break
    assert i is not None
    x = p.initial.copy()
    xy, _ = p.coordinates(x)
    pose = x.reshape(-1, 3)
    mid_i = (xy[p.bonds[i, 0]] + xy[p.bonds[i, 1]]) / 2
    theta_i = math.atan2(*(xy[p.bonds[i, 1]] - xy[p.bonds[i, 0]])[::-1])
    frame_b = int(p.bond_frame[j])
    mid_local_j = (p.local[p.bonds[j, 0]] + p.local[p.bonds[j, 1]]) / 2
    theta_local_j = math.atan2(*(p.local[p.bonds[j, 1]] - p.local[p.bonds[j, 0]])[::-1])
    theta = theta_i - theta_local_j + (0.0 if parallel else angle_offset)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    target = mid_i + offset * np.array([-math.sin(theta_i), math.cos(theta_i)])
    pose[frame_b, 2] = theta
    pose[frame_b, :2] = target - rotation @ mid_local_j
    return i, j, x


def test_segment_separation_crossing_energy_and_smooth_gradient():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    # exact contact: the energy is at (or near) the full margin penalty and
    # non-smooth there, so only the value is asserted — a crossing X keeps
    # dist at zero under any small endpoint perturbation, and finite
    # differences cannot see the subgradient
    i, j, x = _place_crossing_bonds(p, offset=0.0)
    xy, _ = p.coordinates(x)
    si, sj = p.segment_candidates(xy, x.reshape(-1, 3)[:, :2])
    analytic = np.zeros_like(xy)
    energy = p.segment_separation(xy, analytic, si, sj)
    import cyclopeptide2d.solver as _solver
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(
        _solver.PoseProblem.segment_separation).parameters.items()}
    full_cost = defaults["weight"] * defaults["margin"] ** 2
    assert energy > full_cost * 0.5, "a crossing must pay near-full margin cost"
    # gradient check on the crafted pair alone: the full candidate list
    # always contains pairs sitting exactly on the margin boundary, where
    # finite differences are meaningless. The configuration must also keep
    # the pair strictly inside the margin without crossing: the closest
    # point then sits at an interior vertex-edge branch — parallel
    # overlaps and X crossings are both kinks of the value function where
    # finite differences are meaningless.
    i, j, x = _place_crossing_bonds(p, angle_offset=math.pi * 0.5, offset=0.52)
    xy, _ = p.coordinates(x)
    pair_i, pair_j = np.array([i]), np.array([j])
    a, b = xy[p.bonds[i, 0]], xy[p.bonds[i, 1]]
    c, d = xy[p.bonds[j, 0]], xy[p.bonds[j, 1]]
    analytic = np.zeros_like(xy)
    gap_energy = p.segment_separation(xy, analytic, pair_i, pair_j)
    assert gap_energy > 0, "crafted pair must sit inside the clearance margin"
    assert gap_energy < 60.0 * 0.30**2 * 0.5, "crafted pair must not cross"
    _, rot = p.coordinates(x)
    pose_grad = p.pose_gradient(analytic, rot).ravel()
    fd = approx_derivative(
        lambda z: p.segment_separation(
            p.coordinates(z)[0], np.zeros_like(p.coordinates(z)[0]), pair_i, pair_j
        ),
        x, method="3-point",
    ).ravel()
    assert np.allclose(pose_grad, fd, atol=1e-6, rtol=1e-4)


def test_segment_separation_subgradient_pushes_crossing_apart():
    """A few descent steps from an exact crossing must separate the bonds."""
    from scipy.optimize import minimize

    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    i, j, x = _place_crossing_bonds(p, offset=0.0)
    xy, _ = p.coordinates(x)
    pair_i, pair_j = np.array([i]), np.array([j])
    start = p.segment_separation(xy, np.zeros_like(xy), pair_i, pair_j)

    def energy_grad(z):
        xy, rot = p.coordinates(z)
        grad = np.zeros_like(xy)
        value = p.segment_separation(xy, grad, pair_i, pair_j)
        return value, p.pose_gradient(grad, rot).ravel()

    result = minimize(energy_grad, x, jac=True, method="L-BFGS-B", options={"maxiter": 60})
    assert result.fun < start * 0.5, "the subgradient must drive the crossing apart"


def test_deadline_skips_optional_phases_but_still_reports():
    import time

    from cyclopeptide2d.solver import residue_pose_layout

    m = Chem.MolFromSmiles(SMI)
    laid, report = residue_pose_layout(m, seed=0, deadline=time.monotonic() - 1.0)
    assert laid is not None and report["ok"]
    assert report["early_conformer_refine"]["trials"] == 0
    assert report["internal_conformer_refine"]["trials"] == 0
    assert report["reflection_trials"] == 0
    assert "solution_pose" in report and "defect_rank" in report


def test_subtree_moves_preserve_rigid_and_interface_invariants():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x = p.initial.copy()
    closure_before = p.closure(x).copy()
    angles_before = p.angle_slack(x).copy()
    moves = list(p.subtree_moves(range(len(frames))))
    assert moves
    for k, ids, pivot, angle in moves:
        before = p.local.copy()
        frame_atoms = set(map(int, frames[k].atoms))
        moved, fixed = set(ids), frame_atoms - set(ids)
        assert moved <= frame_atoms, "a subtree never leaves its frame"
        assert pivot in moved
        assert not ({frames[k].n_atom, frames[k].c_atom} & moved), (
            "a move must never relocate an interface port atom"
        )
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.array([[co, -si], [si, co]])
        origin = before[pivot]
        p.local[ids] = (before[ids] - origin) @ rotation.T + origin

        d_moved_before = np.linalg.norm(before[ids][:, None] - before[ids], axis=2)
        d_moved_after = np.linalg.norm(p.local[ids][:, None] - p.local[ids], axis=2)
        assert np.allclose(d_moved_before, d_moved_after, atol=1e-12)
        assert np.allclose(
            np.linalg.norm(before[ids] - origin, axis=1),
            np.linalg.norm(p.local[ids] - origin, axis=1),
            atol=1e-12,
        )
        fixed_ids = sorted(fixed)
        assert np.allclose(before[fixed_ids], p.local[fixed_ids], atol=0)
        assert np.allclose(p.closure(x), closure_before, atol=1e-12)
        assert np.allclose(p.angle_slack(x), angles_before, atol=1e-12)
        p.local[:] = before


def test_template_refine_never_degrades_layout():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x, rep = p.solve(maxiter=180)
    rank_before = p.defect_rank(x)[0]
    x2, report = p.template_refine(x, rounds=2)
    assert report["defect_rank"][0] <= list(rank_before)[0]
    assert report["template_trials"] >= 0

    assert np.abs(p.closure(x2)).max() < 1e-7


def test_beam_selection_spreads_sign_patterns():
    from cyclopeptide2d.kinematics import port_seeds

    m = Chem.MolFromSmiles("N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O")
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    seeds = port_seeds(p, width=32, count=4)
    assert len(seeds) == 4
    sign_patterns = {s.signs for s in seeds}
    assert len(sign_patterns) >= 2, "selection should not cluster one pattern"


def test_native_templates_keep_acyclic_quaternary_backbone_angle():
    m = Chem.MolFromSmiles("O=C1N[C@](C)(CO)C(=O)N[C@@H](CC)C(=O)NCC1")
    frames, order, _ = peptide_frames(m)
    for f in frames:
        if len(f.backbone) != 3:
            continue
        a, b, c = [list(f.atoms).index(i) for i in f.backbone]
        u, v = f.local[a] - f.local[b], f.local[c] - f.local[b]
        assert np.isclose(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -0.5)


def test_link_solver_closes_the_entire_cycle_without_stretch():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x, report = p.solve(maxiter=180)
    assert report["max_link_error"] < 1e-7
    xy, _ = p.coordinates(x)
    for a, b in p.bonds:
        assert abs(np.linalg.norm(xy[a] - xy[b]) - 1.0) < 1e-6


def test_contact_constraint_gradients_match_numerical_derivatives():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x = p.initial + np.random.default_rng(4).normal(0, 0.03, len(p.initial))
    aa, ab = np.arange(len(p.ai)), np.arange(len(p.ab_a))
    jac = p.separation_constraints(x, aa, ab, True)
    fd = approx_derivative(lambda z: p.separation_constraints(z, aa, ab), x)
    assert np.allclose(jac, fd, atol=1e-7)


def test_internal_conformer_moves_are_proper_rotations_with_fixed_bond_geometry():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    moves = list(p.conformer_moves(range(len(frames))))
    assert moves
    for k, pivot, ids, angle in moves:
        before = p.local.copy()
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.array([[co, -si], [si, co]])
        assert np.isclose(np.linalg.det(rotation), 1.0)
        origin = before[pivot]
        p.local[ids] = (before[ids] - origin) @ rotation.T + origin
        for a, b in p.bonds:
            if p.owner[a] == p.owner[b]:
                assert np.isclose(
                    np.linalg.norm(before[a] - before[b]), np.linalg.norm(p.local[a] - p.local[b])
                )
        assert np.allclose(before[frames[k].backbone], p.local[frames[k].backbone])
        assert np.allclose(
            np.linalg.norm(before[ids, None] - before[ids], axis=2),
            np.linalg.norm(p.local[ids, None] - p.local[ids], axis=2),
        )

        assert m.GetAtomWithIdx(pivot).GetDegree() == 2
        assert m.GetAtomWithIdx(pivot).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        neighbors = [a.GetIdx() for a in m.GetAtomWithIdx(pivot).GetNeighbors()]

        def cosine(xy, neighbors=neighbors, pivot=pivot):
            a, b = xy[neighbors] - xy[pivot]
            return a @ b / np.linalg.norm(a) / np.linalg.norm(b)

        assert np.isclose(cosine(before), cosine(p.local))
        p.local[:] = before


def test_contact_constraints_are_global_translation_invariant():
    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x = p.initial.copy()
    shifted = x.copy()
    shifted.reshape(-1, 3)[:, :2] += [19.0, -11.0]
    aa, ab = np.arange(len(p.ai)), np.arange(len(p.ab_a))
    assert np.allclose(
        p.separation_constraints(x, aa, ab), p.separation_constraints(shifted, aa, ab)
    )
    assert np.allclose(p.closure(x), p.closure(shifted))


def test_central_occupancy_is_neither_a_constraint_nor_an_energy_penalty():

    m = Chem.MolFromSmiles(SMI)
    frames, order, _ = peptide_frames(m)
    p = PoseProblem(m, frames, order)
    x = p.initial.copy()
    aa, ab = np.arange(len(p.ai)), np.arange(len(p.ab_a))
    constraints = p.separation_constraints(x, aa, ab)
    energy, gradient = p.objective(x)
    assert len(constraints) == len(aa) + len(ab)
    p.cavity_r = 1e6
    assert np.array_equal(p.separation_constraints(x, aa, ab), constraints)
    assert p.objective(x)[0] == energy
    assert np.array_equal(p.objective(x)[1], gradient)


def test_crosslinked_macrocycle_closes_and_passes_gates():
    """cyclo(Cys-Gly3-Cys-Gly3) with an S-S bridge: one closure per loop."""
    smiles = "O=C1NC(CS2)C(=O)NCC(=O)NCC(=O)NCC(=O)NC(CS2)C(=O)NCC(=O)NCC(=O)NCC(=O)N1"
    m = validate_input(smiles)
    laid, report = residue_pose_layout(m, seed=0, maxiter=350)
    assert report.get("ok"), report
    assert len(report["crosslinks"]) == 1
    assert report.get("qa_pass"), (report.get("defect_rank"), report.get("constraints"))
    assert laid is not None
    conf = laid.GetConformer()
    for bond in laid.GetBonds():
        a = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        b = conf.GetAtomPosition(bond.GetEndAtomIdx())
        length = np.hypot(a.x - b.x, a.y - b.y)
        assert abs(length / 30.0 - 1.0) < 1e-5, "every bond including the S-S closes at length 1"

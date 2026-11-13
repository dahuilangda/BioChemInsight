import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from scipy.optimize._numdiff import approx_derivative

from cyclopeptide2d import depict_record
from cyclopeptide2d.coordinates import RigidCoordinates
from cyclopeptide2d.flexible import FlexibleGeometry, flexible_refine
from cyclopeptide2d.frames import registered_frames
from cyclopeptide2d.graph_seed import graph_seed
from cyclopeptide2d.solver import PoseProblem

peptidecore = pytest.importorskip("peptidecore")
CASES = json.loads((Path(__file__).parent / "fixtures/difficult_peptides.json").read_text())["records"]


def make_problem(helm):
    record = peptidecore.from_helm(helm)
    frames, graph = registered_frames(record)
    molecule = record.molecule
    poses = graph_seed(molecule, frames, graph.links)
    return record, PoseProblem(
        molecule,
        frames,
        graph.order,
        links=graph.links,
        crosslinks=graph.crosslinks,
        ring=graph.ring,
        initial_poses=poses,
    )


def test_flexible_derivatives_match_finite_differences():
    _, problem = make_problem("PEPTIDE1{C.F.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0")
    geometry = FlexibleGeometry(problem)
    xy, _ = geometry.embedding("rdkit", 0)
    x = xy.ravel() + np.random.default_rng(31).normal(0, 0.01, 2 * problem.n)
    assert np.allclose(
        geometry.equalities(x, jacobian=True), approx_derivative(geometry.equalities, x), atol=2e-7
    )
    assert np.allclose(
        geometry.angle_jacobian(x), approx_derivative(geometry.angle_slack, x), atol=2e-7
    )
    _, gradient = geometry.objective(x, xy)
    numerical = approx_derivative(lambda z: geometry.objective(z, xy)[0], x).ravel()
    assert np.allclose(gradient, numerical, atol=2e-5, rtol=1e-5)
    empty = np.array([], dtype=int)
    assert geometry.separation(x, empty, empty, jacobian=True).shape == (0, 2 * problem.n)
    aa = np.arange(min(64, len(problem.ai)))
    ab = np.arange(min(64, len(problem.ab_a)))
    assert np.allclose(
        geometry.separation(x, aa, ab, jacobian=True),
        approx_derivative(lambda z: geometry.separation(z, aa, ab), x),
        atol=2e-6,
    )


def test_rigid_coordinates_preserve_rings_fixed_atoms_and_derivatives():
    _, problem = make_problem("PEPTIDE1{C.F.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0")
    geometry = FlexibleGeometry(problem)
    xy, _ = geometry.embedding("rdkit", 0)
    ring_atoms = set(atom for ids, _ in geometry.ring_templates for atom in ids)
    fixed = {next(atom for atom in range(problem.n) if atom not in ring_atoms)}
    chart = RigidCoordinates(xy, geometry.ring_templates, fixed=fixed)
    values = chart.initial + np.random.default_rng(7).normal(0, 0.1, len(chart.initial))
    result = chart.coordinates(values).copy()
    assert np.array_equal(result[sorted(fixed)], xy[sorted(fixed)])
    assert geometry.rigid_error(result) < 1e-12
    jacobian = chart.pullback(np.eye(2 * problem.n), values)
    numerical = approx_derivative(lambda v: chart.coordinates(v).ravel(), values)
    assert np.allclose(jacobian, numerical, atol=2e-7)


def test_expired_refinement_preserves_problem_state():
    _, problem = make_problem("PEPTIDE1{C.A.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0")
    before = problem.local.copy()
    x, report = flexible_refine(problem, problem.initial, deadline=0.0)
    assert not report["accepted"]
    assert report["attempts"] == []
    assert np.array_equal(x, problem.initial)
    assert np.array_equal(problem.local, before)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"][:12])
def test_difficult_peptides_pass_chemical_and_geometric_checks(case):
    record, reference = make_problem(case["helm"])
    model = FlexibleGeometry(reference)
    result = depict_record(record, seed=case["seed"], time_budget=case.get("time_budget"))
    assert result.accepted, result.report["constraints"]
    xy = np.asarray(result.molecule.GetConformer().GetPositions())[:, :2] / 30.0
    assert model.rigid_error(xy) < 1e-7
    assert result.report["defect_rank"][:3] == [0, 0, 0]
    assert Chem.MolToSmiles(result.molecule) == record.smiles
    if any(
        result.report.get(name, {}).get("accepted")
        for name in ("flexible_geometry", "flexible_geometry_tail")
    ):
        assert np.min(model.angle_slack(xy.ravel()), initial=0.0) >= -1e-6
        assert result.report["constraints"]["max_rigid_ring_error"] < 1e-7


def test_native_linear_angles_remain_feasible():
    _, problem = make_problem("PEPTIDE1{[deamino-Gly_CN].A.G.A}$$$$V2.0")
    geometry = FlexibleGeometry(problem)
    linear = np.flatnonzero(geometry.targets < -1 + 1e-10)
    assert len(linear)
    xy, _ = problem.coordinates(problem.initial)
    assert np.min(geometry.angle_slack(xy.ravel())[linear]) >= -1e-12


def test_fixed_geometry_uses_the_final_feasibility_tolerance():
    import time

    case = json.loads((Path(__file__).parent / "fixtures/localized_geometry.json").read_text())
    _, problem = make_problem(case["helm"])
    problem.local[:] = np.asarray(case["local"])
    poses = np.asarray(case["poses"]).ravel()
    reference, _ = problem.coordinates(poses)
    geometry = FlexibleGeometry(problem)
    _, implicated = problem.defect_rank(poses)
    fixed = set(np.flatnonzero(~np.isin(problem.owner, implicated)))
    frozen_edges = np.asarray([set(pair) <= fixed for pair in geometry.edges])
    residual = np.abs(geometry.equalities(reference.ravel())[frozen_edges])
    assert 1e-8 < residual.max() < 1e-7
    _, report = geometry.solve(reference, fixed=fixed, deadline=time.monotonic() - 1)
    assert report.get("error") != "inconsistent_fixed_geometry"
    invalid = reference.copy()
    invalid[geometry.edges[frozen_edges][0, 0], 0] += 0.01
    _, report = geometry.solve(invalid, fixed=fixed, deadline=time.monotonic() - 1)
    assert report["error"] == "inconsistent_fixed_geometry"


def test_residue_reflections_resolve_a_refined_aromatic_contact():
    import time

    case = json.loads((Path(__file__).parent / "fixtures/aromatic_geometry.json").read_text())
    _, problem = make_problem(case["helm"])
    geometry = FlexibleGeometry(problem)
    problem.local[:] = np.asarray(case["local"])
    poses = np.asarray(case["poses"])
    assert problem.defect_rank(poses)[0][:3] == (0, 0, 1)
    result, report = problem.discrete_refine(poses, rounds=2, max_trials=8, deadline=time.monotonic() + 6)
    assert report["whole_residue_reflections"]
    assert problem.defect_rank(result)[0][:3] == (0, 0, 0)
    coordinates, _ = problem.coordinates(result)
    assert np.max(np.abs(geometry.equalities(coordinates.ravel()))) < 1e-7
    assert np.min(geometry.angle_slack(coordinates.ravel())) >= -1e-6
    assert geometry.rigid_error(coordinates) < 1e-7

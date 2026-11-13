import numpy as np
import pytest
from rdkit import Chem

from cyclopeptide2d import depict_record
from cyclopeptide2d.frames import registered_frames
from cyclopeptide2d.graph_seed import graph_seed
from cyclopeptide2d.projection import retract
from cyclopeptide2d.solver import PoseProblem

peptidecore = pytest.importorskip("peptidecore")


@pytest.mark.parametrize(
    "helm,cycles,paths,closed",
    [
        ("PEPTIDE1{A.G.K}$$$$V2.0", 0, 1, False),
        ("PEPTIDE1{[ac].A.G.K.[am]}$$$$V2.0", 0, 1, False),
        ("PEPTIDE1{C.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-4:R3$$$V2.0", 1, 1, False),
        (
            "PEPTIDE1{C.A.A.C}$PEPTIDE1,PEPTIDE1,4:R2-1:R1|PEPTIDE1,PEPTIDE1,1:R3-4:R3$$$V2.0",
            2,
            1,
            True,
        ),
        ("PEPTIDE1{A.K.G}|PEPTIDE2{A.G}$PEPTIDE1,PEPTIDE2,2:R3-2:R2$$$V2.0", 0, 2, False),
    ],
)
def test_registered_graph_retains_exact_connections_and_owners(helm, cycles, paths, closed):
    record = peptidecore.from_helm(helm)
    frames, graph = registered_frames(record)
    assert graph.cycle_rank == cycles
    assert len(graph.paths) == paths
    assert graph.ring is closed
    assert sorted(atom for frame in frames for atom in frame.atoms) == list(
        range(record.molecule.GetNumAtoms())
    )
    declared = {
        tuple(sorted((edge.source_atom, edge.target_atom)))
        for edge in record.connections
        if graph.owners[edge.source_atom] != graph.owners[edge.target_atom]
    }
    assert {tuple(sorted(pair)) for pair in graph.links} == declared
    for instance in set(record.atom_instances):
        assert (
            len(
                {
                    graph.owners[index]
                    for index, owner in enumerate(record.atom_instances)
                    if owner == instance
                }
            )
            == 1
        )
    assert len(graph.links) - len(frames) + 1 == cycles


def test_branched_solver_has_no_artificial_chain_link():
    record = peptidecore.from_helm(
        "PEPTIDE1{A.K.G}|PEPTIDE2{A.G}$PEPTIDE1,PEPTIDE2,2:R3-2:R2$$$V2.0"
    )
    frames, graph = registered_frames(record)
    poses = graph_seed(record.molecule, frames, graph.links)
    problem = PoseProblem(
        record.molecule,
        frames,
        graph.order,
        links=graph.links,
        ring=False,
        initial_poses=poses,
        crosslinks=graph.crosslinks,
    )
    assert len(problem.links) == 4
    assert set(map(tuple, problem.links)) == set(graph.links)
    result = depict_record(record, maxiter=200)
    assert result.accepted
    assert result.report["connection_graph"]["registered_boundaries"]
    assert result.report["connection_graph"]["cycle_rank"] == 0
    assert Chem.MolToSmiles(result.molecule) == record.smiles


def test_single_registered_residue_does_not_require_empty_fragmentation():
    record = peptidecore.from_sequence("A")
    result = depict_record(record)
    assert result.accepted
    assert result.report["initialization"] == "registered_fragment"
    assert result.report["constraints"]["max_link_error"] == 0
    assert Chem.MolToSmiles(result.molecule) == record.smiles


def test_crosslink_branches_remain_flexible_and_projection_closes_all_links():
    record = peptidecore.from_helm("PEPTIDE1{C.A.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0")
    molecule = record.molecule
    frames, graph = registered_frames(record)
    initial = graph_seed(molecule, frames, graph.links)
    problem = PoseProblem(
        molecule,
        frames,
        graph.order,
        links=graph.links,
        ring=False,
        initial_poses=initial,
        crosslinks=graph.crosslinks,
    )
    layout = depict_record(record, maxiter=200)
    assert layout.accepted
    problem.local[:] = layout.report["solution_local"]
    x = np.array(layout.report["solution_pose"])
    assert np.abs(problem.closure(x)).max() < 1e-7
    candidates = list(problem.conformer_moves(range(problem.k), fine=True))
    candidates += list(problem.attachment_moves(range(problem.k)))
    assert any(problem.interface_atoms[ids].any() for _, _, ids, _ in candidates)
    projected = False
    for _, pivot, ids, angle in candidates:
        if not problem.interface_atoms[ids].any():
            continue
        original = problem.local[ids].copy()
        origin = problem.local[pivot].copy()
        co, si = np.cos(angle), np.sin(angle)
        problem.local[ids] = (original - origin) @ np.array([[co, -si], [si, co]]).T + origin
        candidate, valid, attempted = problem.project_local_motion(x, ids, original)
        if attempted and valid:
            assert np.abs(problem.closure(candidate)).max() < 1e-7
            assert problem.angle_slack(candidate).min() >= -1e-6
            projected = True
            problem.local[ids] = original
            break
        problem.local[ids] = original
    assert projected


def test_retraction_restores_all_links_and_angles_without_mutating_start():
    record = peptidecore.from_helm("PEPTIDE1{C.A.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-6:R3$$$V2.0")
    frames, graph = registered_frames(record)
    problem = PoseProblem(
        record.molecule,
        frames,
        graph.order,
        links=graph.links,
        ring=False,
        initial_poses=graph_seed(record.molecule, frames, graph.links),
    )
    layout = depict_record(record, maxiter=200)
    assert layout.accepted
    problem.local[:] = layout.report["solution_local"]
    reference = np.array(layout.report["solution_pose"])
    constrained_atoms = set(problem.angles[:, 1])
    assert set(problem.links.ravel()) <= constrained_atoms
    rng = np.random.default_rng(13)
    for _ in range(5):
        start = reference + rng.normal(0, 0.02, len(reference))
        before = start.copy()
        candidate, valid = retract(problem, start)
        assert valid
        assert np.array_equal(start, before)
        assert np.abs(problem.closure(candidate)).max() < 1e-8
        assert problem.angle_slack(candidate).min() >= -1e-8
    candidate, valid = retract(problem, start, deadline=0.0)
    assert not valid
    assert np.array_equal(candidate, start)


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{V.K.P.W.N}|PEPTIDE2{P.G}$PEPTIDE1,PEPTIDE2,2:R3-2:R2$$$V2.0",
        "PEPTIDE1{N.K.K.Q.I.K}|PEPTIDE2{Y.K.E}$PEPTIDE1,PEPTIDE2,3:R3-3:R2$$$V2.0",
        "PEPTIDE1{H.W.K.W.Q.F.H}|PEPTIDE2{A.N.H.A}$PEPTIDE1,PEPTIDE2,3:R3-4:R2$$$V2.0",
        "PEPTIDE1{A.E.F.K.F.Q.I.V}|PEPTIDE2{K.T}$PEPTIDE1,PEPTIDE2,4:R3-2:R2$$$V2.0",
        "PEPTIDE1{C.W.A.K.F.S.F.Y.S}|PEPTIDE2{D.D.S}$PEPTIDE1,PEPTIDE2,4:R3-3:R2$$$V2.0",
        "PEPTIDE1{G.E.L.E.K.T.E.W.T.Y}|PEPTIDE2{Q.R.K.C}$PEPTIDE1,PEPTIDE2,5:R3-4:R2$$$V2.0",
        "PEPTIDE1{G.D.Y.W.K.D.E.F.I.I.S}|PEPTIDE2{S.E}$PEPTIDE1,PEPTIDE2,5:R3-2:R2$$$V2.0",
        "PEPTIDE1{D.K.P.I.L.K.H.V.A.Q.T.K}|PEPTIDE2{L.Q.Y}$PEPTIDE1,PEPTIDE2,6:R3-3:R2$$$V2.0",
    ],
)
def test_branched_embeddings_preserve_unit_bonds_and_pass_validation(helm):
    record = peptidecore.from_helm(helm)
    layout = depict_record(record)
    assert layout.accepted, layout.report.get("constraints")
    assert layout.report["constraints"]["max_bond_relative_error"] < 1e-7
    assert Chem.MolToSmiles(layout.molecule) == record.smiles

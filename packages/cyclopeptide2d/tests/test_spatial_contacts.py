"""Differential checks against exhaustive contacts, including mutable frames."""

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from cyclopeptide2d.solver import PoseProblem, peptide_frames


@pytest.mark.parametrize("key", ["53", "68", "83"])
def test_spatial_contacts_match_exhaustive_objective_and_defects(key):
    refs = json.loads((Path(__file__).parents[1] / "examples/reviewed_references.json").read_text())
    mol = Chem.MolFromSmiles(refs[key]["smiles"])
    frames, order, _ = peptide_frames(mol)
    p = PoseProblem(mol, frames, order)
    spatial = p.contact_candidates

    def exhaustive(xy):
        return np.arange(len(p.ai)), np.arange(len(p.ab_a))

    rng = np.random.default_rng(84)
    for scale in (0.0, 0.1, 2.0, 12.0):
        x = p.initial + rng.normal(0, scale, p.initial.shape)

        p.local[frames[0].atoms, 1] *= -1
        p.local[frames[1].atoms] += [1.3, -0.7]
        p.contact_candidates = spatial
        e, g = p.objective(x, 40.0, 200.0)
        rank, implicated = p.defect_rank(x)
        xy, _ = p.coordinates(x)
        aa, ab = spatial(xy)
        all_slack = p.separation_constraints(x, *exhaustive(xy))
        assert set(np.flatnonzero(all_slack[: len(p.ai)] < 0)) <= set(aa)
        assert set(np.flatnonzero(all_slack[len(p.ai) :] < 0)) <= set(ab)
        p.contact_candidates = exhaustive
        full_e, full_g = p.objective(x, 40.0, 200.0)
        full_rank, full_implicated = p.defect_rank(x)
        assert np.isclose(e, full_e, rtol=1e-13, atol=1e-10)
        assert np.allclose(g, full_g, rtol=1e-12, atol=1e-9)
        assert rank[:3] == full_rank[:3]
        assert np.allclose(rank[3:], full_rank[3:], rtol=1e-13, atol=1e-10)
        assert implicated == full_implicated


def test_subtree_moves_keep_native_junction_angles_after_repeated_moves():
    mol = Chem.MolFromSmiles("O=C1N[C@@H](CCc2ccccc2)C(=O)N[C@@H](CC)C(=O)N[C@@H](CO)1")
    frames, order, _ = peptide_frames(mol)
    p = PoseProblem(mol, frames, order)
    for _ in range(3):
        moves = list(p.subtree_moves(range(p.k)))
        assert moves

        for _, ids, pivot, angle in moves:
            before = p.local.copy()
            co, si = np.cos(angle), np.sin(angle)
            rotation = np.array([[co, -si], [si, co]])
            p.local[ids] = (before[ids] - before[pivot]) @ rotation.T + before[pivot]
            for (a, b, c), target in p.junction_angles.items():
                u, v = p.local[a] - p.local[b], p.local[c] - p.local[b]
                actual = np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
                assert abs(actual - target) <= np.radians(12) + 1e-8
            p.local[:] = before
        _, ids, pivot, angle = moves[0]
        co, si = np.cos(angle), np.sin(angle)
        origin = p.local[pivot].copy()
        p.local[ids] = (p.local[ids] - origin) @ np.array([[co, si], [-si, co]]) + origin


def test_inactive_branch_moves_cannot_improve_existing_defects():
    molecule = Chem.MolFromSmiles(
        "O=C1N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)1"
    )
    frames, order, _ = peptide_frames(molecule)
    problem = PoseProblem(molecule, frames, order)
    skipped = 0
    for scale in (1.0, 4.0):
        x = problem.initial.copy()
        x.reshape(-1, 3)[:, :2] *= scale
        rank, _, influence = problem._defects(x, collect_influence=True)
        moves = list(problem.attachment_moves(range(problem.k)))
        moves += list(problem.conformer_moves(range(problem.k), fine=True))
        for _, pivot, ids, angle in moves:
            if not influence.isdisjoint(ids):
                continue
            skipped += 1
            before = problem.local[ids].copy()
            origin = problem.local[pivot].copy()
            co, si = np.cos(angle), np.sin(angle)
            problem.local[ids] = (before - origin) @ np.array([[co, -si], [si, co]]).T + origin
            candidate, _ = problem.defect_rank(x)
            problem.local[ids] = before
            assert candidate[:3] >= rank[:3]
            if candidate[:3] == rank[:3]:
                assert candidate[3] >= rank[3] - 1e-10
                if np.isclose(candidate[3], rank[3], atol=1e-10, rtol=0):
                    assert candidate[4] >= rank[4] - 1e-10
    assert skipped > 0

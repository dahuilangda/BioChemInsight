import numpy as np
from rdkit import Chem
from scipy.optimize._numdiff import approx_derivative

from cyclopeptide2d.frames import peptide_frames
from cyclopeptide2d.motions import subtree_reflections
from cyclopeptide2d.solver import PoseProblem
from cyclopeptide2d.topology import HalfPlanes, crossing_pairs, half_plane_constraints

SMILES = "O=C1N[C@@H](CC(C)C)C(=O)N[C@@H](CCO)C(=O)N[C@@H](CCC(=O)O)1"


def problem():
    molecule = Chem.MolFromSmiles(SMILES)
    frames, order, _ = peptide_frames(molecule)
    return PoseProblem(molecule, frames, order)


def test_half_plane_jacobian_matches_finite_differences():
    p = problem()
    f = p.frames
    planes = HalfPlanes(
        np.array(
            [
                [f[0].n_atom, f[0].c_atom, f[1].n_atom],
                [f[0].n_atom, f[1].n_atom, f[2].n_atom],
                [f[1].c_atom, f[2].n_atom, f[0].n_atom],
            ]
        ),
        np.array([1.0, -1.0, 1.0]),
    )
    x = p.initial + np.random.default_rng(54).normal(0, 0.05, p.initial.shape)
    expected = approx_derivative(lambda z: half_plane_constraints(p, z, planes), x)
    actual = half_plane_constraints(p, x, planes, jacobian=True)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_crossing_pairs_exclude_incident_bonds():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    bonds = np.array([[0, 1], [2, 3], [1, 4]])
    assert crossing_pairs(bonds, xy).tolist() == [[0, 1]]


def test_pendant_reflections_preserve_lengths_angles_and_interface_coordinates():
    p = problem()
    before = p.local.copy()
    moves = list(subtree_reflections(p, range(p.k)))
    assert moves
    x = p.initial
    closure, angles = p.closure(x), p.angle_slack(x)
    for _, root, pivot, ids, reflection in moves:
        p.local[:] = before
        assert np.isclose(np.linalg.det(reflection), -1)
        origin = before[pivot]
        p.local[ids] = (before[ids] - origin) @ reflection.T + origin
        np.testing.assert_allclose(p.local[[root, pivot]], before[[root, pivot]], atol=1e-12)
        np.testing.assert_allclose(p.local[p.order], before[p.order], atol=1e-12)
        np.testing.assert_allclose(p.closure(x), closure, atol=1e-12)
        np.testing.assert_allclose(p.angle_slack(x), angles, atol=1e-12)
        for a, b in p.bonds:
            assert np.isclose(
                np.linalg.norm(p.local[a] - p.local[b]), np.linalg.norm(before[a] - before[b])
            )
        for (a, b, c), _ in p.junction_angles.items():
            u, v = p.local[a] - p.local[b], p.local[c] - p.local[b]
            old_u, old_v = before[a] - before[b], before[c] - before[b]
            if p.owner[a] == p.owner[b] == p.owner[c]:
                assert np.isclose(
                    u @ v / np.linalg.norm(u) / np.linalg.norm(v),
                    old_u @ old_v / np.linalg.norm(old_u) / np.linalg.norm(old_v),
                )

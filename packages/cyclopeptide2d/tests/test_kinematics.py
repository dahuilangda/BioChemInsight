import numpy as np
import pytest
from rdkit import Chem

from cyclopeptide2d.kinematics import port_seeds
from cyclopeptide2d.solver import PoseProblem, peptide_frames


def problem():
    mol = Chem.MolFromSmiles("N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O")
    frames, order, _ = peptide_frames(mol)
    return PoseProblem(mol, frames, order)


def test_port_candidates_preserve_complete_frames_and_placed_peptide_bonds():
    p = problem()
    native = p.local.copy()
    for candidate in port_seeds(p, width=32, count=3):
        p.local[:] = native
        for k, sign in enumerate(candidate.signs):
            p.local[p.frames[k].atoms, 1] *= sign
        xy, _ = p.coordinates(candidate.poses.ravel())
        assert np.allclose(p.closure(candidate.poses.ravel())[:-1], 0, atol=1e-10)
        for frame in p.frames:
            ids = frame.atoms
            assert np.allclose(
                np.linalg.norm(xy[ids, None] - xy[ids], axis=2),
                np.linalg.norm(native[ids, None] - native[ids], axis=2),
            )
    p.local[:] = native


def test_seed_generation_is_deterministic_and_does_not_mutate_input_problem():
    p = problem()
    initial, local = p.initial.copy(), p.local.copy()
    before = Chem.MolToSmiles(p.mol)
    first = port_seeds(p, width=32, count=2)
    second = port_seeds(p, width=32, count=2)
    import time

    lapsed = port_seeds(p, width=32, count=2, deadline=time.monotonic() - 1.0)
    assert len(lapsed) == 2, 'an expired deadline must still return the current beam'
    assert len(first) == len(second) == 2
    assert all(
        a.signs == b.signs and np.array_equal(a.poses, b.poses)
        for a, b in zip(first, second, strict=True)
    )
    assert np.array_equal(p.initial, initial)
    assert np.array_equal(p.local, local)
    assert Chem.MolToSmiles(p.mol) == before


def test_seed_parameters_are_validated():
    with pytest.raises(ValueError):
        port_seeds(problem(), count=0)
    with pytest.raises(ValueError):
        port_seeds(problem(), overlap=0.0)

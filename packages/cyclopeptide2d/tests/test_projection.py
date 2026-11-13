import numpy as np

from cyclopeptide2d.projection import project_constraints


def test_projection_handles_redundant_equalities_and_an_active_boundary():
    def equalities(x):
        residual = x @ x - 1
        return np.array([residual, 2 * residual])

    def jacobian(x):
        return np.array([2 * x, 4 * x])

    start = np.array([-0.02, 1.03])
    projected, valid = project_constraints(
        start, equalities, jacobian,
        lambda x: x[:1], lambda x: np.array([[1.0, 0.0]]),
    )
    assert valid
    assert np.max(np.abs(equalities(projected))) < 1e-9
    assert projected[0] >= -1e-9
    assert np.allclose(start, [-0.02, 1.03])


def test_projection_respects_an_expired_deadline():
    start = np.array([0.1])
    projected, valid = project_constraints(
        start, lambda x: x - 1, lambda x: np.ones((1, 1)),
        lambda x: np.empty(0), lambda x: np.empty((0, 1)), deadline=0.0,
    )
    assert not valid
    assert np.array_equal(projected, start)

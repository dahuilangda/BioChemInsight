from __future__ import annotations

import time

import numpy as np


def project_constraints(
    start, equalities, equality_jacobian, slack, slack_jacobian, *, maxiter=24, deadline=None
):
    """Retract a coordinate chart onto equality and active inequality constraints."""
    values = np.array(start, dtype=float, copy=True)
    for _ in range(maxiter):
        if deadline is not None and time.monotonic() >= deadline:
            break
        eq, gaps = equalities(values), slack(values)
        if np.max(np.abs(eq), initial=0.0) < 1e-9 and np.min(gaps, initial=0.0) >= -1e-9:
            return values, True
        active = gaps < 0.0
        residual = np.r_[eq, gaps[active]]
        jacobian = np.vstack((equality_jacobian(values), slack_jacobian(values)[active]))
        scale = np.maximum(np.linalg.norm(jacobian, axis=1), 1e-8)
        step = np.linalg.lstsq(
            jacobian / scale[:, None], -residual / scale, rcond=1e-10
        )[0]
        if not np.isfinite(step).all():
            break
        step /= max(1.0, np.max(np.abs(step), initial=0.0) / 0.25)
        merit = float(eq @ eq + np.minimum(gaps, 0.0) @ np.minimum(gaps, 0.0))
        for power in range(10):
            trial = values + step * 0.5**power
            trial_eq, trial_gaps = equalities(trial), np.minimum(slack(trial), 0.0)
            if float(trial_eq @ trial_eq + trial_gaps @ trial_gaps) < merit:
                values = trial
                break
        else:
            break
    return values, False


def retract(problem, start, *, maxiter=16, deadline=None):
    """Restore connection constraints with damped minimum-norm pose steps."""
    x = np.array(start, dtype=float, copy=True)
    for _ in range(maxiter):
        if deadline is not None and time.monotonic() >= deadline:
            break
        closure = problem.closure(x)
        slack = problem.angle_slack(x)
        if np.max(np.abs(closure), initial=0.0) < 1e-8 and np.min(slack, initial=0.0) >= -1e-8:
            return x, True
        active = slack < 0.0
        residual = np.r_[closure, slack[active]]
        jac = np.vstack((problem.closure_jac(x), problem.angle_slack_jac(x)[active]))
        scale = np.maximum(np.linalg.norm(jac, axis=1), 1e-8)
        jac = jac / scale[:, None]
        residual = residual / scale
        step = np.linalg.lstsq(jac, -residual, rcond=1e-10)[0]
        if not np.isfinite(step).all():
            break
        poses = step.reshape(-1, 3)
        step /= max(
            1.0,
            np.max(np.linalg.norm(poses[:, :2], axis=1)) / 0.5,
            np.max(np.abs(poses[:, 2])) / 0.35,
        )
        merit = np.sum(closure**2) + np.sum(np.minimum(slack, 0.0) ** 2)
        for power in range(8):
            trial = x + step * 0.5**power
            value = np.sum(problem.closure(trial) ** 2) + np.sum(
                np.minimum(problem.angle_slack(trial), 0.0) ** 2
            )
            if value < merit:
                x = trial
                break
        else:
            break
    return x, False

"""Cartesian atoms and rigid ring systems in one coordinate chart."""

import math

import numpy as np


class RigidCoordinates:
    def __init__(self, reference, templates, *, fixed=()):
        self.reference = reference.copy()
        self.fixed = set(fixed)
        self.owner = np.arange(len(reference))
        self.groups = []
        ring_atoms = set()
        for ids, template in templates:
            order = np.argsort(ids)
            ids = np.asarray(ids, dtype=int)[order]
            template = np.asarray(template)[order]
            ring_atoms.update(ids)
            self.owner[ids] = ids[0]
            if self.fixed.intersection(ids):
                self.fixed.update(ids)
            target = reference[ids]
            centre = target.mean(axis=0)
            native = template - template.mean(axis=0)
            best = None
            for sign in (1, -1):
                local = native * [1, sign]
                u, _, vt = np.linalg.svd(local.T @ (target - centre))
                if np.linalg.det(u @ vt) < 0:
                    u[:, -1] *= -1
                rotation = u @ vt
                error = float(np.sum((local @ rotation + centre - target) ** 2))
                if best is None or error < best[0]:
                    best = error, local, rotation
            _, local, rotation = best
            if self.fixed.intersection(ids):
                continue
            self.reference[ids] = local @ rotation + centre
            self.groups.append((ids, local, centre, math.atan2(rotation[0, 1], rotation[0, 0])))
        self.free = np.asarray(
            sorted(set(range(len(reference))) - ring_atoms - self.fixed), dtype=int
        )
        values = list(self.reference[self.free].ravel())
        self.bodies = []
        for ids, local, centre, angle in self.groups:
            self.bodies.append((ids, local, len(values)))
            values.extend((*centre, angle))
        self.initial = np.asarray(values)
        self.last = None

    def coordinates(self, values):
        if self.last is None or not np.array_equal(self.last, values):
            self.xy = self.reference.copy()
            self.xy[self.free] = values[: 2 * len(self.free)].reshape(-1, 2)
            self.rotated = []
            for ids, local, offset in self.bodies:
                x, y, angle = values[offset : offset + 3]
                co, si = math.cos(angle), math.sin(angle)
                rotated = local @ np.array([[co, si], [-si, co]])
                self.xy[ids] = rotated + [x, y]
                self.rotated.append(rotated)
            self.last = values.copy()
        return self.xy

    def pullback(self, jacobian, values):
        self.coordinates(values)
        rows = jacobian.reshape(-1, len(self.reference), 2)
        result = np.zeros((len(rows), len(values)))
        result[:, : 2 * len(self.free)] = rows[:, self.free].reshape(len(rows), 2 * len(self.free))
        for (ids, _, offset), rotated in zip(self.bodies, self.rotated, strict=True):
            gradients = rows[:, ids]
            result[:, offset : offset + 2] = gradients.sum(axis=1)
            derivative = np.column_stack((-rotated[:, 1], rotated[:, 0]))
            result[:, offset + 2] = np.sum(gradients * derivative, axis=(1, 2))
        return result

    def bounds(self, radius):
        bounds = [(-radius, radius)] * (2 * len(self.free))
        bounds.extend(
            bound
            for _ in self.bodies
            for bound in ((-radius, radius), (-radius, radius), (None, None))
        )
        return bounds

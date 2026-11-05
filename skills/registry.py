"""Skill registry: atomic operations, exposed level by level.

A skill entry: name, kind (model | chemistry), one-line description, and
an async handler taking a JSON payload -> JSON result. The API layer
serves /v1/skills (listing) and /v1/skills/{name}/invoke (atomic call)
directly from this registry; the harness composes them by name."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Skill:
    name: str
    kind: str
    description: str
    handler: Callable[[dict], Awaitable[dict]]
    input_schema: dict = field(default_factory=dict)


_REGISTRY: dict[str, Skill] = {}


def register(name: str, kind: str, description: str):
    def deco(fn):
        _REGISTRY[name] = Skill(name, kind, description, fn)
        return fn
    return deco


def get(name: str) -> Skill | None:
    return _REGISTRY.get(name)


def listing() -> list[dict]:
    return [{"name": s.name, "kind": s.kind, "description": s.description}
            for s in sorted(_REGISTRY.values(), key=lambda s: s.name)]


def load_skills() -> None:
    """Import all skill modules so their handlers self-register."""
    from skills.peptide_notation import skill as _peptide  # noqa: F401
    from skills.cross_validate import skill as _xval  # noqa: F401
    from skills.antibody_numbering import skill as _abnum  # noqa: F401
    from skills.variant_sites import skill as _sites  # noqa: F401

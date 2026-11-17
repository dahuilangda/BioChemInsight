"""Registration and discovery of atomic skills."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    kind: str
    description: str
    handler: Callable[[dict], Awaitable[dict]]
    input_schema: dict = field(default_factory=dict)


_REGISTRY: dict[str, Skill] = {}


def register(
    name: str, kind: str, description: str, *, input_schema: dict | None = None
):
    if (
        not isinstance(name, str)
        or not name.isascii()
        or not name.isidentifier()
        or name != name.lower()
    ):
        raise ValueError("Skill name must be a lowercase ASCII identifier")
    if kind not in {"model", "chemistry", "external"}:
        raise ValueError(f"Unknown skill kind: {kind!r}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Skill description is required")
    if input_schema is not None and not isinstance(input_schema, dict):
        raise TypeError("Skill input schema must be an object")

    def decorator(handler):
        if name in _REGISTRY:
            raise ValueError(f"Skill already registered: {name}")
        if not inspect.iscoroutinefunction(handler):
            raise TypeError("Skill handler must be asynchronous")
        _REGISTRY[name] = Skill(
            name,
            kind,
            description,
            handler,
            deepcopy(input_schema) if input_schema is not None else {},
        )
        return handler

    return decorator


def get(name: str) -> Skill | None:
    return _REGISTRY.get(name)


def listing() -> list[dict]:
    return [
        {
            "name": skill.name,
            "kind": skill.kind,
            "description": skill.description,
            "input_schema": deepcopy(skill.input_schema),
        }
        for skill in sorted(_REGISTRY.values(), key=lambda skill: skill.name)
    ]


def load_skills() -> None:
    from skills.activity_tables import skill as _activity  # noqa: F401
    from skills.antibody_numbering import skill as _numbering  # noqa: F401
    from skills.cross_validate import skill as _cross_validate  # noqa: F401
    from skills.markush_assemble import skill as _markush  # noqa: F401
    from skills.name_resolution import skill as _identity  # noqa: F401
    from skills.peptide_notation import skill as _peptide  # noqa: F401
    from skills.seq_id_vision import skill as _sequence_id  # noqa: F401
    from skills.sequence_assembler import skill as _sequences  # noqa: F401
    from skills.variant_sites import skill as _variants  # noqa: F401

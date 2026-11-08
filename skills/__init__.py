"""Atomic capability registry.

Each skill is a self-contained directory: prompt.md (the model contract,
stored separately from code), skill.py (the atomic operation), and the
registry entry that exposes it. The harness composes skills; nothing
below a skill's level knows about orchestration."""

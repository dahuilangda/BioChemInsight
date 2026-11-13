"""Shared release gate for optional Markush and fragment processing."""
import constants
import os


def markush_assembly_enabled():
    override = os.environ.get('BCI_MARKUSH_ASSEMBLY')
    if override is not None: return override.strip().lower() in {'1', 'true', 'yes'}
    return bool(getattr(constants, 'MARKUSH_ASSEMBLY_ENABLED', False))

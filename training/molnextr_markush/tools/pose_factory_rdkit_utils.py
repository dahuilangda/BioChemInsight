"""Backward-compatibility shim.

The real implementation now lives in
``training.molnextr_markush.src.rdkit_utils``.
"""
from training.molnextr_markush.src.rdkit_utils import *  # noqa: F401,F403
from training.molnextr_markush.src.rdkit_utils import (  # noqa: F401
    BACKEND_REFERENCES,
    atom_label,
    bond_records,
    can_accept_attachment_dummy,
    choose_anchor,
    configure_draw_options,
    graph_consistency,
    import_rdkit,
    normalize_coord,
    perturb_image,
    side_from_endpoint,
    stable_id,
)

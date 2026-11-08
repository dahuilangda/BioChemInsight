"""Per-box refs expansion: teacher-forcing supervision for multi-box pages.

The MoE trainer's ``refs`` batch is **per sample** (per page in fusion
terms): sequence targets ``refs[fmt] = [tensor(B, T, ...), lengths(B, 1)]``,
``coords``/``attachment_*`` padded per-sample tensors, and
``decoder_task_mask`` per page (dataset.py collate; entry.py:844). The
fused recognizer feeds the decoder ``(sum(K), 144, C)`` features where K_i
is the number of boxes on page i — so refs must be expanded **per box, in
RoICollector order (page-major, boxes within a page in detection order)**.

Two regimes are supported:

* single-structure pages (Stage P1: MG2 one-structure pages, K may exceed 1
  through duplicated detections) — every box of a page inherits that page's
  (only) structure target;
* multi-structure pages (Stage P2) — the caller supplies
  ``box_structure_ids[page][box] = structure index`` and per-page structure
  refs built by stacking the individual structure refs along dim 0 first.

The adapter is strict: any key whose batch dimension does not match the page
count, or an assignment out of range, raises ``ValueError`` naming the key —
no silent misalignment between boxes and labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

SEQ_KEYS = {"atomtok", "inchi", "nodes", "atomtok_coords", "chartok_coords", "atom_indices"}
PADDED_KEYS = {
    "coords", "attachment_points", "attachment_point_mask",
    "attachment_bonded", "attachment_bond_type", "attachment_dummy_indices",
    # production bms_collate/entry keys (round-2 audit): same per-sample
    # padded layout, expand identically
    "attachment_anchor_points", "attachment_anchor_mask",
    "attachment_anchor_indices", "attachment_atom_coords",
    "attachment_set_complete", "attachment_count", "edges",
}


def _flat_index(per_page_counts: list[int], box_structure_ids: list[list[int]] | None,
                structure_counts: list[int] | None, batch_size: int, key: str) -> torch.Tensor:
    """Build the gather index over the structure axis for every emitted box."""
    if box_structure_ids is None:
        ids = [[0] * k for k in per_page_counts]
    else:
        ids = box_structure_ids
    if len(ids) != batch_size:
        raise ValueError(
            f"{key}: box_structure_ids covers {len(ids)} pages but refs batch is {batch_size}"
        )
    flat: list[int] = []
    for page, page_ids in enumerate(ids):
        if len(page_ids) != per_page_counts[page]:
            raise ValueError(
                f"{key}: page {page} emitted {per_page_counts[page]} boxes but "
                f"assignment lists {len(page_ids)}"
            )
        limit = structure_counts[page] if structure_counts else 1
        for structure in page_ids:
            if not 0 <= structure < limit:
                raise ValueError(
                    f"{key}: page {page} assignment {structure} out of range [0, {limit})"
                )
            # without structure_counts the refs axis IS the page axis
            flat.append(page if structure_counts is None else
                        _page_offset(structure_counts, page) + structure)
    return torch.tensor(flat, dtype=torch.long)


def _page_offset(structure_counts: list[int], page: int) -> int:
    return sum(structure_counts[:page])


@dataclass
class ExpandedRefs:
    refs: dict[str, Any]
    structure_labels: torch.Tensor | None
    per_page_counts: list[int]
    unknown_keys_expanded: list[str] = field(default_factory=list)


def expand_refs_for_boxes(
    refs: dict[str, Any],
    per_page_counts: list[int],
    structure_labels: torch.Tensor | None = None,
    box_structure_ids: list[list[int]] | None = None,
    structure_counts: list[int] | None = None,
) -> ExpandedRefs:
    """Expand per-page refs to per-box refs in RoICollector order.

    Parameters
    ----------
    refs : trainer-style refs dict (dataset.py collate + entry.py additions)
    per_page_counts : K_i per page; must be non-negative, at least one total
    structure_labels : optional (B,) routing supervision -> (sum(K),)
    box_structure_ids : per page, per box structure index (None = all page's
        boxes use structure 0 — the explicit form of the old coincidence)
    structure_counts : structures per page; required when any page has >1
        structure, in which case refs' batch axis is interpreted as
        concatenated per-structure rows (page-major)
    """
    refs_axis = _infer_batch_size(refs)
    total_boxes = sum(per_page_counts)
    if total_boxes == 0:
        raise ValueError("cannot expand refs for zero boxes")
    pages = len(structure_counts) if structure_counts else len(per_page_counts)
    total_structures = sum(structure_counts) if structure_counts else pages
    if len(per_page_counts) != pages:
        raise ValueError(
            f"per_page_counts covers {len(per_page_counts)} pages but the run "
            f"has {pages} pages (from structure_counts)"
        )
    if refs_axis != total_structures:
        raise ValueError(
            f"refs batch axis is {refs_axis} but the "
            f"{'concatenated structure rows' if structure_counts else 'page count'} "
            f"imply {total_structures}"
        )
    gather = _flat_index(per_page_counts, box_structure_ids, structure_counts,
                         pages, "<assignments>").to(_refs_device(refs))

    expanded: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in refs.items():
        if key in SEQ_KEYS and isinstance(value, list) and len(value) == 2:
            sequences, lengths = value
            _check_batch(key, sequences.shape[0], total_structures)
            expanded[key] = [sequences.index_select(0, gather),
                             lengths.index_select(0, gather)]
        elif key in PADDED_KEYS or key == "decoder_task_mask":
            _check_batch(key, value.shape[0], total_structures)
            expanded[key] = value.index_select(0, gather)
        elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == total_structures:
            expanded[key] = value.index_select(0, gather)
            unknown.append(key)
        else:
            expanded[key] = value  # non-batched metadata passes through untouched
    labels = None
    if structure_labels is not None:
        _check_batch("structure_labels", structure_labels.shape[0], total_structures)
        labels = structure_labels.to(gather.device).index_select(0, gather)
    return ExpandedRefs(expanded, labels, list(per_page_counts), unknown)


def _refs_device(refs: dict[str, Any]) -> torch.device:
    for value in refs.values():
        target = value[0] if isinstance(value, list) and value else value
        if torch.is_tensor(target):
            return target.device
    return torch.device("cpu")


def _infer_batch_size(refs: dict[str, Any]) -> int:
    for key in SEQ_KEYS:
        value = refs.get(key)
        if isinstance(value, list) and len(value) == 2 and torch.is_tensor(value[0]):
            return int(value[0].shape[0])
    for key in PADDED_KEYS | {"decoder_task_mask"}:
        value = refs.get(key)
        if torch.is_tensor(value) and value.dim() > 0:
            return int(value.shape[0])
    raise ValueError("refs carries no batched key to infer the page count from")


def _check_batch(key: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{key}: batch axis {actual} does not match page count {expected}")

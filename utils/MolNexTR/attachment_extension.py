"""Strict decoding for MolNexTR's native fragment attachment extension."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


ATTACHMENT_EXTENSION_RE = re.compile(r"\[(0|[1-9][0-9]*):\*\]\Z")
ATTACHMENT_EXTENSION_SCHEMA = "molnextr_single_attachment_extension_v1"


class AttachmentExtensionError(ValueError):
    """A model-emitted attachment extension cannot be decoded as a graph."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        message = self.code if not self.detail else f"{self.code}:{self.detail}"
        super().__init__(message)


def parse_attachment_extension(extension: str) -> int:
    """Return the zero-based anchor index from exactly ``[anchor:*]``."""

    if not isinstance(extension, str) or not extension:
        raise AttachmentExtensionError("missing")
    match = ATTACHMENT_EXTENSION_RE.fullmatch(extension)
    if match is None:
        raise AttachmentExtensionError("malformed", extension[:64])
    return int(match.group(1))


def allowed_extension_token_ids(tokenizer, emitted_sequence: Sequence[int]):
    """Return the legal next-token ids after ``<sep>``, or ``None`` before it.

    This is a representation grammar, not a repair: the network still decides
    whether to emit ``<sep>`` and which anchor digits to emit. Once it enters the
    extension, syntactically impossible continuations are excluded.
    """

    sep_id = getattr(tokenizer, "sep_id", None)
    if sep_id is None:
        return None
    sequence = [int(token_id) for token_id in emitted_sequence]
    if int(sep_id) not in sequence:
        return None
    sep_position = sequence.index(int(sep_id))
    suffix = sequence[sep_position + 1 :]
    stoi = getattr(tokenizer, "stoi", {})

    def token_id(character: str) -> int:
        value = stoi.get(character)
        if value is None:
            raise AttachmentExtensionError("vocabulary_missing", character)
        return int(value)

    left = token_id("[")
    colon = token_id(":")
    star = token_id("*")
    right = token_id("]")
    digits = {token_id(str(value)) for value in range(10)}
    eos_id = int(stoi.get("<eos>", 2))

    if not suffix:
        return {left}
    if suffix[0] != left:
        return set()
    anchor_tokens = []
    cursor = 1
    while cursor < len(suffix) and suffix[cursor] in digits:
        anchor_tokens.append(suffix[cursor])
        cursor += 1
    if not anchor_tokens:
        return digits if cursor == len(suffix) else set()
    zero_id = token_id("0")
    if anchor_tokens[0] == zero_id and len(anchor_tokens) > 1:
        return set()
    if cursor == len(suffix):
        return {colon} if anchor_tokens[0] == zero_id else digits | {colon}
    if suffix[cursor] != colon:
        return set()
    cursor += 1
    if cursor == len(suffix):
        return {star}
    if suffix[cursor] != star:
        return set()
    cursor += 1
    if cursor == len(suffix):
        return {right}
    if suffix[cursor] != right:
        return set()
    cursor += 1
    if cursor == len(suffix):
        return {eos_id}
    return set()


def _validated_coords(coords: Sequence[Sequence[float]], atom_count: int):
    if not isinstance(coords, Sequence) or len(coords) != atom_count:
        raise AttachmentExtensionError(
            "coordinate_count", f"{len(coords) if isinstance(coords, Sequence) else -1}!={atom_count}"
        )
    output = []
    for atom_index, coord in enumerate(coords):
        if not isinstance(coord, Sequence) or len(coord) < 2:
            raise AttachmentExtensionError("invalid_coordinate", str(atom_index))
        x, y = float(coord[0]), float(coord[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise AttachmentExtensionError("nonfinite_coordinate", str(atom_index))
        output.append([x, y])
    return output


def _validated_edges(edges: Sequence[Sequence[int]], atom_count: int):
    if not isinstance(edges, Sequence) or len(edges) != atom_count:
        raise AttachmentExtensionError(
            "edge_shape", f"{len(edges) if isinstance(edges, Sequence) else -1}!={atom_count}"
        )
    output = []
    for row_index, row in enumerate(edges):
        if not isinstance(row, Sequence) or len(row) != atom_count:
            raise AttachmentExtensionError("edge_shape", f"row={row_index}")
        output.append([int(value) for value in row])
    return output


def _dummy_coordinate(coords, edges, anchor_index: int) -> list[float]:
    """Place the abstract dummy deterministically for MolBlock serialization."""

    anchor = coords[anchor_index]
    neighbor_indices = [
        index
        for index in range(len(coords))
        if index != anchor_index
        and (edges[anchor_index][index] != 0 or edges[index][anchor_index] != 0)
    ]
    bond_lengths = []
    for begin in range(len(coords)):
        for end in range(begin + 1, len(coords)):
            if edges[begin][end] == 0 and edges[end][begin] == 0:
                continue
            distance = math.hypot(
                coords[begin][0] - coords[end][0],
                coords[begin][1] - coords[end][1],
            )
            if distance > 1.0e-6:
                bond_lengths.append(distance)
    if bond_lengths:
        bond_lengths.sort()
        length = bond_lengths[len(bond_lengths) // 2]
    else:
        length = 0.12
    length = min(0.25, max(0.04, float(length)))

    if neighbor_indices:
        center_x = sum(coords[index][0] for index in neighbor_indices) / len(neighbor_indices)
        center_y = sum(coords[index][1] for index in neighbor_indices) / len(neighbor_indices)
    else:
        center_x = sum(coord[0] for coord in coords) / len(coords)
        center_y = sum(coord[1] for coord in coords) / len(coords)
    direction_x = anchor[0] - center_x
    direction_y = anchor[1] - center_y
    norm = math.hypot(direction_x, direction_y)
    if norm <= 1.0e-6:
        direction_x, direction_y, norm = 1.0, 0.0, 1.0
    return [
        anchor[0] + length * direction_x / norm,
        anchor[1] + length * direction_y / norm,
    ]


def materialize_attachment_extension(
    atom_data: dict,
    edges: Sequence[Sequence[int]],
    *,
    sep_count: int,
    invalid_extension_tokens: bool = False,
    attachment_confidence: float | None = None,
):
    """Decode the extension into one dummy node and one bonded graph edge.

    The operation is atomic: validation finishes before ``atom_data`` is
    changed. Invalid model output raises ``AttachmentExtensionError`` and must
    be surfaced as an inference failure by the caller.
    """

    if int(sep_count) != 1:
        raise AttachmentExtensionError("separator_count", str(int(sep_count)))
    if invalid_extension_tokens:
        raise AttachmentExtensionError("non_character_token")
    anchor_index = parse_attachment_extension(atom_data.get("extension"))
    symbols = list(atom_data.get("symbols") or [])
    atom_count = len(symbols)
    if atom_count == 0:
        raise AttachmentExtensionError("empty_backbone")
    if any("*" in str(symbol) for symbol in symbols):
        raise AttachmentExtensionError("backbone_contains_dummy")
    if not 0 <= anchor_index < atom_count:
        raise AttachmentExtensionError(
            "anchor_out_of_range", f"{anchor_index}>={atom_count}"
        )
    coords = _validated_coords(atom_data.get("coords") or [], atom_count)
    edge_rows = _validated_edges(edges, atom_count)
    dummy_index = atom_count
    dummy_coord = _dummy_coordinate(coords, edge_rows, anchor_index)

    new_edges = [row + [0] for row in edge_rows]
    new_edges.append([0] * (atom_count + 1))
    new_edges[anchor_index][dummy_index] = 1
    new_edges[dummy_index][anchor_index] = 1
    new_symbols = symbols + ["*"]
    new_coords = coords + [dummy_coord]

    atom_data["symbols"] = new_symbols
    atom_data["coords"] = new_coords
    atom_scores = atom_data.get("atom_scores")
    if isinstance(atom_scores, list):
        atom_data["atom_scores"] = atom_scores + [
            float(attachment_confidence)
            if attachment_confidence is not None
            else 0.0
        ]
    metadata = {
        "schema_version": ATTACHMENT_EXTENSION_SCHEMA,
        "extension": atom_data["extension"],
        "anchor_atom_index": anchor_index,
        "dummy_atom_index": dummy_index,
        "bond_type": 1,
        "dummy_coordinate_policy": "deterministic_outward_graph_serialization_v1",
    }
    return new_edges, metadata

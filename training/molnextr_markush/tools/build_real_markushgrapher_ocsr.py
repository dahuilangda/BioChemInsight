"""Build pose-verified OCSR rows from original MarkushGrapher2 images.

The official ``cxsmiles_dataset`` field contains depiction coordinates, but
those coordinates live in the depiction frame rather than page-image pixels.
For training rows we recover the page pose with matched R-label OCR anchors,
then independently verify the transformed graph against ink along every bond.
Rows that fail this geometric contract are excluded from training instead of
being converted to coordinate MASK tokens. Official test rows remain complete:
pose recovery is diagnostic there and never changes the evaluation selection.
"""
from __future__ import annotations

import argparse
import html
import hashlib
import io
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.molnextr_markush.src.moe_dataset import (
    MOE_DATA_CONTRACT_VERSION,
    _strip_cxsmiles,
    fragment_attachment_contract,
    implant_markush_attachment_isotopes,
    parse_cxsmiles_dummy_labels,
)
from utils.MolNexTR.tokenization import atomwise_tokenizer
from utils.MolNexTR.utils import FORMAT_INFO


DEFAULT_RAW_ROOT = Path("training/molnextr_markush/data/raw/markushgrapher2")
DEFAULT_OUTPUT_ROOT = Path(
    "training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2"
)
TRAIN_GLOBS = ("uspto-mol-m-54k/train-*.parquet",)
EVAL_GLOBS = (
    "uspto-mol-m-54k/test-*.parquet",
    "ip5-markush/test-*.parquet",
    "m2s/test-*.parquet",
    "uspto-markush/test-*.parquet",
)
BOND_TYPE_TO_INT = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
SHARD_RE = re.compile(r"(?:train|test)-(\d+)-of-(\d+)")
POSE_DRAW_SIZE = 512
POSE_DRAW_PADDING = 0.05
POSE_ANCHOR_INLIER_DISTANCE = 0.04
POSE_MAX_ANCHOR_RESIDUAL = 0.06
POSE_MIN_BOND_INK_HIT_RATE = 0.45
POSE_MAX_BOND_INK_P90 = 0.035


def iter_source_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({path.resolve() for path in paths})


def choose_cxsmiles(row: dict[str, Any]) -> str:
    for key in ("cxsmiles", "cxsmiles_dataset", "cxsmiles_opt"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalized_label(value: Any) -> str:
    return "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", str(value or ""))
        if character.isalnum()
    )


def attachment_cell_targets(
    row: dict[str, Any],
    *,
    dummy_count: int,
) -> tuple[list[list[float]], list[int], bool]:
    """Extract high-precision real variable centers from MG2 OCR cells.

    The annotation contains one ``<r>...</r>`` entry per graph dummy. OCR cells
    carry normalized image bboxes. Ambiguous duplicate text is deliberately
    left unmatched; partial sets remain useful through positive-unlabeled set
    loss and the exact graph dummy count still supplies cardinality supervision.
    """

    annotation = html.unescape(str(row.get("annotation") or ""))
    labels = [
        re.sub(r"<[^>]+>", "", value).strip()
        for value in re.findall(
            r"<r>(.*?)</r>", annotation, flags=re.IGNORECASE | re.DOTALL
        )
    ]
    labels = [value for value in labels if value]
    cells = []
    for cell in row.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        bbox = cell.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
            continue
        cells.append(
            {
                "text": str(cell.get("text") or "").strip(),
                "normalized": _normalized_label(cell.get("text")),
                "point": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            }
        )

    used: set[int] = set()
    points: list[list[float]] = []
    matched_label_indices: list[int] = []
    for label_index, label in enumerate(labels):
        normalized = _normalized_label(label)
        if not normalized:
            continue
        candidates = []
        for index, cell in enumerate(cells):
            if index in used or not cell["normalized"]:
                continue
            exact = cell["normalized"] == normalized
            affix = (
                len(normalized) >= 2
                and (
                    normalized in cell["normalized"]
                    or cell["normalized"] in normalized
                )
            )
            if exact or affix:
                candidates.append((index, exact))
        exact_candidates = [item for item in candidates if item[1]]
        chosen = None
        if len(exact_candidates) == 1:
            chosen = exact_candidates[0][0]
        elif not exact_candidates and len(candidates) == 1:
            chosen = candidates[0][0]
        if chosen is not None:
            used.add(chosen)
            points.append(list(cells[chosen]["point"]))
            matched_label_indices.append(label_index)
    return (
        points,
        matched_label_indices,
        bool(len(points) == int(dummy_count) and len(labels) == int(dummy_count)),
    )


def target_length(smiles: str) -> tuple[int, int]:
    length = 2
    atoms = 0
    for token in atomwise_tokenizer(smiles):
        is_atom = token.isalpha() or token.startswith("[") or token == "*"
        length += len(token) + (2 if is_atom else 0)
        atoms += int(is_atom)
    return length, atoms


def graph_edges(mol: Chem.Mol) -> list[list[int]]:
    edges = []
    for bond in mol.GetBonds():
        bond_type = 4 if bond.GetIsAromatic() else BOND_TYPE_TO_INT.get(
            bond.GetBondType(),
            1,
        )
        edges.append(
            [
                int(bond.GetBeginAtomIdx()),
                int(bond.GetEndAtomIdx()),
                int(bond_type),
            ]
        )
    return edges


def _edge_signature(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    return sorted(
        (min(begin, end), max(begin, end), bond_type)
        for begin, end, bond_type in graph_edges(mol)
    )


def source_pose_molecule(row: dict[str, Any], target_mol: Chem.Mol) -> Chem.Mol:
    """Load the official coordinate-bearing graph without changing atom order."""

    source = str(row.get("cxsmiles_dataset") or "").strip()
    if not source:
        raise ValueError("missing_cxsmiles_depiction_coordinates")
    pose_mol = Chem.MolFromSmiles(source, sanitize=False)
    if pose_mol is None or pose_mol.GetNumConformers() != 1:
        raise ValueError("invalid_cxsmiles_depiction_coordinates")
    if pose_mol.GetNumAtoms() != target_mol.GetNumAtoms():
        raise ValueError("pose_target_atom_count_mismatch")
    source_atomic_numbers = [atom.GetAtomicNum() for atom in pose_mol.GetAtoms()]
    target_atomic_numbers = [atom.GetAtomicNum() for atom in target_mol.GetAtoms()]
    if source_atomic_numbers != target_atomic_numbers:
        raise ValueError("pose_target_atom_order_mismatch")
    if _edge_signature(pose_mol) != _edge_signature(target_mol):
        raise ValueError("pose_target_edge_order_mismatch")
    return pose_mol


def canonical_draw_coords(mol: Chem.Mol) -> np.ndarray:
    """Map official depiction coordinates into RDKit's normalized draw frame."""

    drawer = rdMolDraw2D.MolDraw2DSVG(POSE_DRAW_SIZE, POSE_DRAW_SIZE)
    drawer.drawOptions().padding = POSE_DRAW_PADDING
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    coords = np.asarray(
        [
            [
                float(drawer.GetDrawCoords(index).x) / POSE_DRAW_SIZE,
                float(drawer.GetDrawCoords(index).y) / POSE_DRAW_SIZE,
            ]
            for index in range(mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )
    if coords.shape != (mol.GetNumAtoms(), 2) or not np.isfinite(coords).all():
        raise ValueError("invalid_canonical_draw_coordinates")
    return coords


def fit_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares orientation-preserving 2D similarity transform."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("invalid_similarity_correspondences")
    if len(source) < 2:
        raise ValueError("insufficient_similarity_correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denominator = float(np.square(source_centered).sum())
    if denominator <= 1e-10:
        raise ValueError("degenerate_similarity_correspondences")
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(singular_values.sum() / denominator)
    translation = target_mean - scale * source_mean @ rotation
    if not np.isfinite(scale) or not np.isfinite(rotation).all():
        raise ValueError("nonfinite_similarity_transform")
    return scale, rotation, translation


def apply_similarity(
    coords: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return float(scale) * np.asarray(coords, dtype=np.float64) @ rotation + translation


def robust_anchor_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC over known OCR correspondences, followed by an inlier refit."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 2:
        raise ValueError("insufficient_pose_ocr_anchors")
    candidates = []
    pairs = list(itertools.combinations(range(len(source)), 2))
    if len(pairs) > 96:
        pairs = pairs[:96]
    for first, second in pairs:
        if float(np.linalg.norm(source[first] - source[second])) <= 1e-4:
            continue
        try:
            scale, rotation, translation = fit_similarity_transform(
                source[[first, second]],
                target[[first, second]],
            )
        except ValueError:
            continue
        residuals = np.linalg.norm(
            apply_similarity(source, scale, rotation, translation) - target,
            axis=1,
        )
        inliers = residuals <= POSE_ANCHOR_INLIER_DISTANCE
        candidates.append(
            (
                int(inliers.sum()),
                -float(np.median(residuals[inliers])) if bool(inliers.any()) else -1e9,
                scale,
                rotation,
                translation,
                inliers,
            )
        )
    if not candidates:
        raise ValueError("pose_anchor_similarity_fit_failed")
    _count, _score, scale, rotation, translation, inliers = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    minimum_inliers = max(2, int(np.ceil(0.75 * len(source))))
    if int(inliers.sum()) < minimum_inliers:
        raise ValueError("pose_anchor_inlier_fraction_failed")
    scale, rotation, translation = fit_similarity_transform(
        source[inliers],
        target[inliers],
    )
    residuals = np.linalg.norm(
        apply_similarity(source, scale, rotation, translation) - target,
        axis=1,
    )
    return scale, rotation, translation, residuals


def _foreground_geometry(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.asarray(gray < 245, dtype=bool)
    rows, columns = np.nonzero(foreground)
    if len(columns) < 32:
        raise ValueError("pose_image_has_insufficient_ink")
    x_bounds = np.quantile(columns, [0.001, 0.999]) / float(gray.shape[1])
    y_bounds = np.quantile(rows, [0.001, 0.999]) / float(gray.shape[0])
    return x_bounds, y_bounds


def _bond_sample_points(coords: np.ndarray, mol: Chem.Mol) -> np.ndarray:
    points = []
    for bond in mol.GetBonds():
        begin = coords[bond.GetBeginAtomIdx()]
        end = coords[bond.GetEndAtomIdx()]
        for fraction in np.linspace(0.12, 0.88, 7):
            points.append(begin * (1.0 - fraction) + end * fraction)
    if not points:
        raise ValueError("pose_graph_has_no_bonds")
    return np.asarray(points, dtype=np.float64)


def bond_ink_metrics(
    coords: np.ndarray,
    mol: Chem.Mol,
    gray: np.ndarray,
) -> tuple[float, float, float]:
    """Measure transformed bond coverage against source-image ink."""

    ink = np.asarray(gray < 230, dtype=np.uint8)
    distance = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)
    distance = distance / float(max(gray.shape))
    points = _bond_sample_points(coords, mol)
    columns = np.clip(
        np.rint(points[:, 0] * (gray.shape[1] - 1)),
        0,
        gray.shape[1] - 1,
    ).astype(np.int64)
    rows = np.clip(
        np.rint(points[:, 1] * (gray.shape[0] - 1)),
        0,
        gray.shape[0] - 1,
    ).astype(np.int64)
    values = distance[rows, columns]
    return (
        float(np.mean(values <= 0.012)),
        float(np.median(values)),
        float(np.quantile(values, 0.90)),
    )


def single_anchor_similarity(
    canonical: np.ndarray,
    anchor_index: int,
    anchor_point: np.ndarray,
    mol: Chem.Mol,
    gray: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the one-anchor scale ambiguity with bond-to-ink registration."""

    x_bounds, y_bounds = _foreground_geometry(gray)
    span = np.ptp(canonical, axis=0)
    base_scale = min(
        float(np.ptp(x_bounds)) / max(float(span[0]), 1e-8),
        float(np.ptp(y_bounds)) / max(float(span[1]), 1e-8),
    )
    best = None
    for multiplier in (0.78, 0.88, 0.96, 1.0, 1.04, 1.12, 1.22):
        for angle_degrees in (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0):
            angle = np.deg2rad(angle_degrees)
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=np.float64,
            )
            scale = float(base_scale * multiplier)
            translation = np.asarray(anchor_point) - scale * canonical[anchor_index] @ rotation
            transformed = apply_similarity(canonical, scale, rotation, translation)
            hit_rate, _median, p90 = bond_ink_metrics(transformed, mol, gray)
            bounds_penalty = max(0.0, -float(transformed.min())) + max(
                0.0,
                float(transformed.max()) - 1.0,
            )
            score = (hit_rate, -p90, -bounds_penalty)
            if best is None or score > best[0]:
                best = (score, scale, rotation, translation, transformed)
    if best is None:
        raise ValueError("single_anchor_pose_search_failed")
    _score, scale, rotation, translation, transformed = best
    residuals = np.asarray([0.0], dtype=np.float64)
    return scale, rotation, translation, residuals


def recover_image_pose(
    mol: Chem.Mol,
    raw_image: bytes,
    attachment_points: list[list[float]],
    attachment_point_dummy_indices: list[int],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Recover and independently validate normalized page-image atom centers."""

    if not attachment_points or not attachment_point_dummy_indices:
        raise ValueError("pose_missing_matched_ocr_anchor")
    if len(attachment_points) != len(attachment_point_dummy_indices):
        raise ValueError("pose_ocr_anchor_alignment_mismatch")
    canonical = canonical_draw_coords(mol)
    anchor_indices = np.asarray(attachment_point_dummy_indices, dtype=np.int64)
    if bool((anchor_indices < 0).any()) or bool((anchor_indices >= len(canonical)).any()):
        raise ValueError("pose_ocr_anchor_index_out_of_range")
    anchor_targets = np.asarray(attachment_points, dtype=np.float64)
    with Image.open(io.BytesIO(raw_image)) as image:
        image.load()
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if len(anchor_targets) >= 2 and float(np.ptp(canonical[anchor_indices], axis=0).max()) > 1e-4:
        scale, rotation, translation, anchor_residuals = robust_anchor_similarity(
            canonical[anchor_indices],
            anchor_targets,
        )
        transformed = apply_similarity(canonical, scale, rotation, translation)
    else:
        scale, rotation, translation, anchor_residuals = single_anchor_similarity(
            canonical,
            int(anchor_indices[0]),
            anchor_targets[0],
            mol,
            gray,
        )
        transformed = apply_similarity(canonical, scale, rotation, translation)
    hit_rate, ink_median, ink_p90 = bond_ink_metrics(transformed, mol, gray)
    inlier_residuals = anchor_residuals[
        anchor_residuals <= POSE_ANCHOR_INLIER_DISTANCE
    ]
    max_anchor_residual = float(
        inlier_residuals.max() if len(inlier_residuals) else anchor_residuals.max()
    )
    minimum_hit_rate = (
        0.50 if len(anchor_targets) == 1 else POSE_MIN_BOND_INK_HIT_RATE
    )
    maximum_ink_p90 = 0.030 if len(anchor_targets) == 1 else POSE_MAX_BOND_INK_P90
    if not 0.15 <= float(scale) <= 2.5:
        raise ValueError("pose_similarity_scale_out_of_range")
    if float(transformed.min()) < -0.05 or float(transformed.max()) > 1.05:
        raise ValueError("pose_coordinates_out_of_image_range")
    if max_anchor_residual > POSE_MAX_ANCHOR_RESIDUAL:
        raise ValueError("pose_anchor_residual_failed")
    if hit_rate < minimum_hit_rate:
        raise ValueError("pose_bond_ink_hit_rate_failed")
    if ink_p90 > maximum_ink_p90:
        raise ValueError("pose_bond_ink_distance_failed")
    transformed = np.clip(transformed, 0.0, 1.0)
    diagnostics = {
        "method": (
            "ocr_anchor_ransac_similarity"
            if len(anchor_targets) >= 2
            else "single_ocr_anchor_bond_ink_search"
        ),
        "anchor_count": int(len(anchor_targets)),
        "anchor_inlier_count": int(
            np.sum(anchor_residuals <= POSE_ANCHOR_INLIER_DISTANCE)
        ),
        "anchor_residual_mean": float(anchor_residuals.mean()),
        "anchor_residual_max": float(max_anchor_residual),
        "anchor_residual_all_max": float(anchor_residuals.max()),
        "_anchor_inlier_mask": (
            anchor_residuals <= POSE_ANCHOR_INLIER_DISTANCE
        ).tolist(),
        "similarity_scale": float(scale),
        "similarity_rotation_degrees": float(
            np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
        ),
        "bond_ink_hit_rate": float(hit_rate),
        "bond_ink_distance_median": float(ink_median),
        "bond_ink_distance_p90": float(ink_p90),
    }
    return transformed.tolist(), diagnostics


def image_payload(value: Any) -> tuple[bytes, str, int, int, float]:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("missing_page_image_bytes")
    raw = bytes(value)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        image_format = str(image.format or "PNG").lower()
    if width < 64 or height < 64:
        raise ValueError(f"image_too_small:{width}x{height}")
    if width > 4096 or height > 4096:
        raise ValueError(f"image_too_large:{width}x{height}")
    contrast = float(gray.std())
    if not np.isfinite(contrast) or contrast < 3.0:
        raise ValueError(f"image_low_contrast:{contrast:.3f}")
    extension = ".jpg" if image_format in {"jpeg", "jpg"} else ".png"
    return raw, extension, int(width), int(height), contrast


def shard_name(path: Path) -> str:
    match = SHARD_RE.search(path.stem)
    if match:
        return f"s{int(match.group(1)):03d}"
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"s_{digest}"


def build_split(
    *,
    split: str,
    source_paths: list[Path],
    output_root: Path,
    vocab: set[str],
    max_rows: int,
    skip_existing: bool,
    progress_every: int = 1000,
) -> dict[str, Any]:
    split_root = output_root / split
    images_root = split_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    max_target_length = int(FORMAT_INFO["chartok_coords"]["max_len"])
    require_verified_pose = split == "train"

    for source_path in source_paths:
        source_shard = shard_name(source_path)
        parquet = pq.ParquetFile(source_path)
        wanted = [
            name
            for name in (
                "id",
                "image_name",
                "page_image_path",
                "page_image",
                "annotation",
                "cxsmiles",
                "cxsmiles_dataset",
                "cxsmiles_opt",
                "cells",
            )
            if name in parquet.schema_arrow.names
        ]
        source_row = 0
        for batch in parquet.iter_batches(columns=wanted, batch_size=512):
            for row in batch.to_pylist():
                counts["scanned"] += 1
                source_row += 1
                if progress_every > 0 and counts["scanned"] % progress_every == 0:
                    print(
                        f"{split}: scanned={counts['scanned']} accepted={len(records)} "
                        f"pose_verified={counts['pose_verified_rows']} "
                        f"pose_unavailable={counts['pose_unavailable_rows']}",
                        flush=True,
                    )
                if max_rows > 0 and len(records) >= max_rows:
                    break
                try:
                    cxsmiles = choose_cxsmiles(row)
                    smiles = _strip_cxsmiles(cxsmiles)
                    if not smiles or "*" not in smiles:
                        raise ValueError("missing_dummy_target")
                    # Preserve the scaffold's real variable identity (R13 -> [13*])
                    # from the CXSMILES $...$ block, which _strip_cxsmiles drops.
                    # Without this the real-image training targets carry sequential
                    # isotopes and the decoder never learns the patent's R-numbers.
                    smiles = implant_markush_attachment_isotopes(
                        smiles, parse_cxsmiles_dummy_labels(cxsmiles)
                    )
                    if any(character not in vocab for token in atomwise_tokenizer(smiles) for character in token):
                        raise ValueError("target_out_of_vocab")
                    sequence_length, atom_count = target_length(smiles)
                    if sequence_length > max_target_length:
                        raise ValueError(
                            f"target_too_long:{sequence_length}>{max_target_length}"
                        )
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is None:
                        raise ValueError("rdkit_parse_failed")
                    if int(mol.GetNumAtoms()) != atom_count:
                        raise ValueError(
                            f"atom_count_mismatch:{mol.GetNumAtoms()}!={atom_count}"
                        )
                    edges = graph_edges(mol)
                    fragment_ok, _fragment_reason = fragment_attachment_contract(
                        smiles,
                        edges,
                    )
                    dummy_count = int(smiles.count("*"))
                    (
                        attachment_points,
                        attachment_point_label_indices,
                        attachment_points_complete,
                    ) = attachment_cell_targets(row, dummy_count=dummy_count)
                    dummy_atom_indices = [
                        int(atom.GetIdx())
                        for atom in mol.GetAtoms()
                        if atom.GetAtomicNum() == 0
                    ]
                    if len(dummy_atom_indices) != dummy_count:
                        raise ValueError("dummy_atom_index_count_mismatch")
                    attachment_point_dummy_indices = [
                        dummy_atom_indices[label_index]
                        for label_index in attachment_point_label_indices
                        if 0 <= label_index < len(dummy_atom_indices)
                    ]
                    if len(attachment_point_dummy_indices) != len(attachment_points):
                        raise ValueError("attachment_point_dummy_alignment_failed")
                    # MarkushGrapher images remain semantic Markush layouts even
                    # when their graph happens to contain one terminal dummy
                    # (for example a polymer bracket with X and n). Dummy count
                    # alone must never relabel these real images as fragments.
                    label = 1
                    bucket = "markush_layout"
                    raw_image, extension, width, height, contrast = image_payload(
                        row.get("page_image")
                    )
                    node_coords: list[list[float]] = []
                    pose_diagnostics: dict[str, Any] = {}
                    try:
                        pose_mol = source_pose_molecule(row, mol)
                        node_coords, pose_diagnostics = recover_image_pose(
                            pose_mol,
                            raw_image,
                            attachment_points,
                            attachment_point_dummy_indices,
                        )
                        anchor_inliers = list(
                            pose_diagnostics.pop("_anchor_inlier_mask", [])
                        )
                        if anchor_inliers and not all(anchor_inliers):
                            if len(anchor_inliers) != len(attachment_points):
                                raise ValueError("pose_anchor_inlier_mask_mismatch")
                            removed = int(len(anchor_inliers) - sum(anchor_inliers))
                            attachment_points = [
                                point
                                for point, keep in zip(
                                    attachment_points,
                                    anchor_inliers,
                                )
                                if keep
                            ]
                            attachment_point_dummy_indices = [
                                atom_index
                                for atom_index, keep in zip(
                                    attachment_point_dummy_indices,
                                    anchor_inliers,
                                )
                                if keep
                            ]
                            attachment_points_complete = False
                            counts["pose_ocr_anchor_outliers_removed"] += removed
                        counts["pose_verified_candidates"] += 1
                        counts[
                            f"pose_method:{pose_diagnostics['method']}"
                        ] += 1
                    except ValueError as pose_error:
                        reason = str(pose_error).split(":", 1)[0]
                        counts["pose_unavailable_rows"] += 1
                        counts[f"pose_rejected:{reason}"] += 1
                        if require_verified_pose:
                            raise ValueError(reason) from pose_error
                    digest = hashlib.sha256(
                        raw_image + b"\0" + smiles.encode("utf-8")
                    ).hexdigest()
                    if digest in seen:
                        rejected["duplicate_image_target"] += 1
                        continue
                    seen.add(digest)
                    image_dir = images_root / source_shard
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"{digest[:24]}{extension}"
                    if not image_path.exists() or not skip_existing:
                        image_path.write_bytes(raw_image)
                    records.append(
                        {
                            "file_path": str(image_path.resolve()),
                            "SMILES": smiles,
                            "node_coords": (
                                json.dumps(node_coords, separators=(",", ":"))
                                if node_coords
                                else ""
                            ),
                            "node_coords_space": (
                                "normalized_image_cxsmiles_ocr_similarity"
                                if node_coords
                                else "unavailable_real_image"
                            ),
                            "edges": repr(edges),
                            "source_arrow": (
                                "markushgrapher2:uspto_original_image:"
                                f"{source_path.parent.name}"
                            ),
                            "structure_type_label": int(label),
                            "structure_type_bucket": bucket,
                            "attachment_render_mode": "",
                            "attachment_render_geometry": "original_patent_pixels",
                            "real_tight_crop_style": False,
                            "terminal_wavy_externality_passed": False,
                            "image_domain": "real_original",
                            "coordinate_targets_available": bool(node_coords),
                            "coordinate_pose_verified": bool(node_coords),
                            "coordinate_pose_method": str(
                                pose_diagnostics.get("method") or ""
                            ),
                            "coordinate_pose_anchor_count": int(
                                pose_diagnostics.get("anchor_count") or 0
                            ),
                            "coordinate_pose_anchor_residual_max": (
                                float(pose_diagnostics["anchor_residual_max"])
                                if pose_diagnostics
                                else None
                            ),
                            "coordinate_pose_bond_ink_hit_rate": (
                                float(pose_diagnostics["bond_ink_hit_rate"])
                                if pose_diagnostics
                                else None
                            ),
                            "coordinate_pose_bond_ink_distance_p90": (
                                float(pose_diagnostics["bond_ink_distance_p90"])
                                if pose_diagnostics
                                else None
                            ),
                            "data_contract_version": MOE_DATA_CONTRACT_VERSION,
                            "source_dataset": "markushgrapher2",
                            "source_subset": str(source_path.parent.name),
                            "source_parquet": str(source_path),
                            "source_row": int(source_row - 1),
                            "source_id": str(row.get("id") or ""),
                            "source_image_name": str(
                                row.get("image_name") or row.get("page_image_path") or ""
                            ),
                            "target_length": int(sequence_length),
                            "atom_count": int(atom_count),
                            "dummy_count": int(smiles.count("*")),
                            "attachment_points": json.dumps(
                                attachment_points, separators=(",", ":")
                            ),
                            "attachment_points_complete": bool(
                                attachment_points_complete
                            ),
                            "attachment_point_dummy_indices": json.dumps(
                                attachment_point_dummy_indices,
                                separators=(",", ":"),
                            ),
                            "attachment_point_source": "mg2_r_label_ocr_cell_bbox",
                            "single_terminal_dummy_graph": bool(fragment_ok),
                            "image_width": width,
                            "image_height": height,
                            "image_contrast_std": contrast,
                        }
                    )
                    counts[f"accepted_label_{label}"] += 1
                    if node_coords:
                        counts["pose_verified_rows"] += 1
                        counts["pose_verified_atoms"] += len(node_coords)
                    counts["attachment_points"] += len(attachment_points)
                    counts["attachment_points_complete_rows"] += int(
                        attachment_points_complete
                    )
                    counts["attachment_points_partial_rows"] += int(
                        bool(attachment_points) and not attachment_points_complete
                    )
                except ValueError as exc:
                    rejected[str(exc).split(":", 1)[0]] += 1
                except Exception as exc:
                    rejected[f"unexpected_{exc.__class__.__name__}"] += 1
            if max_rows > 0 and len(records) >= max_rows:
                break
        if max_rows > 0 and len(records) >= max_rows:
            break

    frame = pd.DataFrame(records)
    output_path = split_root / "data.parquet"
    frame.to_parquet(output_path, index=False)
    passed = bool(len(frame)) and not frame["file_path"].duplicated().any()
    if require_verified_pose and len(frame):
        passed = passed and bool(
            frame["coordinate_targets_available"].astype(bool).all()
            and frame["coordinate_pose_verified"].astype(bool).all()
        )
    pose_frame = (
        frame[frame["coordinate_targets_available"].astype(bool)]
        if len(frame)
        else frame
    )
    pose_summary = {"rows": int(len(pose_frame))}
    for column in (
        "coordinate_pose_anchor_residual_max",
        "coordinate_pose_bond_ink_hit_rate",
        "coordinate_pose_bond_ink_distance_p90",
    ):
        raw_values = (
            pose_frame[column]
            if column in pose_frame.columns
            else pd.Series(dtype=float)
        )
        values = pd.to_numeric(raw_values, errors="coerce").dropna()
        pose_summary[column] = {
            "mean": float(values.mean()),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
        } if len(values) else {}
    report = {
        "schema_version": "real_markushgrapher_ocsr_v2",
        "split": split,
        "passed": passed,
        "source_paths": [str(path) for path in source_paths],
        "rows": int(len(frame)),
        "counts": dict(counts),
        "rejected": dict(rejected),
        "label_counts": {
            str(key): int(value)
            for key, value in frame["structure_type_label"].value_counts().sort_index().items()
        } if len(frame) else {},
        "pose_summary": pose_summary,
        "policy": {
            "original_image_pixels_preserved": True,
            "coordinate_targets_are_source_derived": True,
            "coordinate_source": (
                "official_cxsmiles_depiction_plus_r_label_ocr_similarity_"
                "with_bond_ink_verification"
            ),
            "training_requires_verified_pose": require_verified_pose,
            "evaluation_selection_is_pose_independent": split == "eval",
            "coordinate_targets_available": bool(
                len(frame)
                and frame["coordinate_targets_available"].astype(bool).any()
            ),
            "pose_quality_gate": {
                "anchor_inlier_distance_max": POSE_ANCHOR_INLIER_DISTANCE,
                "anchor_residual_max": POSE_MAX_ANCHOR_RESIDUAL,
                "bond_ink_hit_rate_min": POSE_MIN_BOND_INK_HIT_RATE,
                "bond_ink_distance_p90_max": POSE_MAX_BOND_INK_P90,
            },
            "symbol_and_edge_targets_from_official_cxsmiles": True,
            "markush_semantics_are_not_inferred_from_dummy_count": True,
            "single_terminal_dummy_is_recorded_as_an_attribute_only": True,
            "official_test_parquets_are_eval_only": split == "eval",
            "data_contract_version": MOE_DATA_CONTRACT_VERSION,
        },
        "output": str(output_path),
    }
    report_path = split_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vocab",
        default="utils/MolNexTR/vocab/vocab_chars.json",
    )
    args = parser.parse_args()

    vocab = set(json.loads(Path(args.vocab).read_text(encoding="utf-8")).keys())
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    reports = {}
    if args.split in {"train", "all"}:
        reports["train"] = build_split(
            split="train",
            source_paths=iter_source_paths(raw_root, TRAIN_GLOBS),
            output_root=output_root,
            vocab=vocab,
            max_rows=max(0, int(args.max_rows)),
            skip_existing=bool(args.skip_existing),
            progress_every=max(0, int(args.progress_every)),
        )
    if args.split in {"eval", "all"}:
        reports["eval"] = build_split(
            split="eval",
            source_paths=iter_source_paths(raw_root, EVAL_GLOBS),
            output_root=output_root,
            vocab=vocab,
            max_rows=max(0, int(args.max_rows)),
            skip_existing=bool(args.skip_existing),
            progress_every=max(0, int(args.progress_every)),
        )
    print(json.dumps(reports, indent=2, sort_keys=True))
    if not reports or not all(report.get("passed") is True for report in reports.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

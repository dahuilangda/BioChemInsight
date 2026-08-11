from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (
    POSE_FACTORY_SCHEMA_VERSION,
    formal_sim_dataset_lineage,
    formal_simulation_policy,
)


GENERATOR_VERSION = "molgrapher_synthetic_pose_convert_v1"
BACKEND_REFERENCES = [
    {
        "name": "MolGrapher-Synthetic-300K",
        "url": "https://huggingface.co/datasets/docling-project/MolGrapher-Synthetic-300K",
        "role": "Large PubChem-derived RDKit/MolDepictor synthetic OCSR source with image, molfile, SMILES, and keypoints.",
    },
    {
        "name": "MolDepictor",
        "url": "https://github.com/DS4SD/MolDepictor",
        "role": "Original generation pipeline for MolGrapher-Synthetic-300K.",
    },
]


def parse_keypoints(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    raw = value if isinstance(value, list) else []
    coords = []
    for atom_index, offset in enumerate(range(0, len(raw), 3)):
        if offset + 1 >= len(raw):
            break
        x = float(raw[offset]) / float(width)
        y = float(raw[offset + 1]) / float(height)
        coords.append(
            {
                "atom_index": atom_index,
                "token": "",
                "x": max(0.0, min(1.0, x)),
                "y": max(0.0, min(1.0, y)),
            }
        )
    return coords


def mol_bonds_and_symbols(mol_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(mol_text, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("invalid mol block")
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "begin_atom_index": int(bond.GetBeginAtomIdx()),
                "end_atom_index": int(bond.GetEndAtomIdx()),
                "bond_order": str(bond.GetBondType()),
                "bond_dir": str(bond.GetBondDir()),
                "is_aromatic": bool(bond.GetIsAromatic()),
            }
        )
    return bonds, symbols


def canonical_smiles_pair(mol_text: str, row_smiles: str) -> tuple[str, str, bool]:
    from rdkit import Chem

    mol_from_block = Chem.MolFromMolBlock(mol_text, sanitize=True, removeHs=False)
    if mol_from_block is None:
        raise ValueError("invalid mol block for canonical comparison")
    if any(atom.GetAtomicNum() == 0 for atom in mol_from_block.GetAtoms()):
        raise ValueError("ordinary negative mol block contains dummy/query atoms")
    mol_from_block = Chem.RemoveHs(mol_from_block, sanitize=True)
    mol_from_smiles = Chem.MolFromSmiles(row_smiles)
    if mol_from_smiles is None:
        raise ValueError("invalid row SMILES for canonical comparison")
    if any(atom.GetAtomicNum() == 0 for atom in mol_from_smiles.GetAtoms()):
        raise ValueError("ordinary negative row SMILES contains dummy/query atoms")
    mol_from_smiles = Chem.RemoveHs(mol_from_smiles, sanitize=True)
    mol_canonical = Chem.MolToSmiles(mol_from_block, canonical=True, isomericSmiles=True)
    row_canonical = Chem.MolToSmiles(mol_from_smiles, canonical=True, isomericSmiles=True)
    return mol_canonical, row_canonical, mol_canonical == row_canonical


def load_heldout_canonical_smiles(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    from rdkit import Chem

    heldout: set[str] = set()
    for path_text in paths:
        path = Path(path_text)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "").strip()
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    heldout.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return heldout


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def sha1_bytes(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_ordinary_document_context(
    image: Any,
    atom_coordinates: list[dict[str, Any]],
    *,
    seed: int,
    enabled: bool,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from io import BytesIO

    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

    rng = random.Random(int(seed))
    image = image.convert("RGB")
    old_w, old_h = image.size
    operations: list[str] = []
    pad_left = pad_top = pad_right = pad_bottom = 0
    coordinate_canvas_w, coordinate_canvas_h = old_w, old_h
    resize_scale = 1.0
    paper_profile = "none"
    paper_background = 255
    paper_noise_sigma = 0.0

    if enabled:
        pad_left = int(round(old_w * rng.uniform(0.02, 0.22)))
        pad_right = int(round(old_w * rng.uniform(0.03, 0.28)))
        pad_top = int(round(old_h * rng.uniform(0.02, 0.24)))
        pad_bottom = int(round(old_h * rng.uniform(0.03, 0.30)))
        profile_specs = {
            "clean_white": ([253, 254, 255], (0.04, 0.35), (0.00, 0.14)),
            "white_scan": ([250, 251, 252, 253, 254], (0.30, 1.15), (0.08, 0.40)),
            "gray_scan": ([246, 247, 248, 249, 250, 251, 252], (0.75, 2.00), (0.18, 0.78)),
            "aged_scan": ([246, 247, 248, 249, 250], (0.45, 1.35), (0.10, 0.52)),
        }
        paper_profile = rng.choices(
            ["clean_white", "white_scan", "gray_scan", "aged_scan"],
            weights=[0.26, 0.34, 0.30, 0.10],
            k=1,
        )[0]
        background_values, noise_range, wave_range = profile_specs[paper_profile]
        paper_background = int(rng.choice(background_values))
        paper_noise_sigma = float(rng.uniform(*noise_range))
        canvas_size = (old_w + pad_left + pad_right, old_h + pad_top + pad_bottom)
        coordinate_canvas_w, coordinate_canvas_h = canvas_size
        arr_bg = np.full((canvas_size[1], canvas_size[0]), paper_background, dtype=np.int16)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        arr_bg = np.clip(arr_bg + np_rng.normal(0, paper_noise_sigma, arr_bg.shape), 0, 255).astype("uint8")
        yy, xx = np.mgrid[0 : canvas_size[1], 0 : canvas_size[0]]
        paper_wave = (
            np.sin((xx + rng.uniform(0, 1000)) / rng.uniform(42.0, 92.0))
            + np.cos((yy + rng.uniform(0, 1000)) / rng.uniform(46.0, 118.0))
        ) * rng.uniform(*wave_range)
        arr_bg = np.clip(arr_bg.astype(np.float32) + paper_wave, 0, 255).astype("uint8")

        arr_struct = np.asarray(image.convert("L")).astype(np.float32)
        alpha = np.clip((252.0 - arr_struct) / 42.0, 0.0, 1.0)
        alpha = np.power(alpha, 0.82)
        ink_values = np.clip(arr_struct * rng.uniform(0.84, 0.97), 0, 255)
        patch = arr_bg[pad_top : pad_top + old_h, pad_left : pad_left + old_w].astype(np.float32)
        patch = patch * (1.0 - alpha) + np.minimum(patch, ink_values) * alpha
        arr_bg[pad_top : pad_top + old_h, pad_left : pad_left + old_w] = np.clip(patch, 0, 255).astype("uint8")
        image = Image.fromarray(arr_bg, mode="L").convert("RGB")
        operations.extend(
            [
                "moldepictor_source_ink_only_composite",
                "asymmetric_white_document_margin",
                "alpha_antialiased_transparent_white_structure_composite",
                "diverse_patent_paper_before_structure_composite",
                f"paper_profile:{paper_profile}",
            ]
        )

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        ink = rng.choice([(0, 0, 0), (18, 18, 18), (35, 35, 35), (52, 52, 52)])
        if rng.random() < 0.22 and pad_top >= 10:
            label = rng.choice(["(a)", "(b)", "A", "B", "Example", "Scheme"])
            draw.text((rng.randint(2, max(2, image.width // 3)), max(1, pad_top // 3)), label, fill=ink, font=font)
            operations.append("small_margin_caption")
        if rng.random() < 0.13:
            y_candidates = []
            if pad_top >= 18:
                y_candidates.append(max(2, pad_top // 3))
            if pad_bottom >= 18:
                y_candidates.append(old_h + pad_top + max(2, (2 * pad_bottom) // 3))
            if y_candidates:
                y = rng.choice(y_candidates)
                draw.line(
                    (rng.randint(0, max(0, image.width // 10)), y, image.width - rng.randint(0, max(0, image.width // 10)), y),
                    fill=ink,
                    width=1,
                )
                operations.append("thin_document_rule")
        if rng.random() < 0.10:
            for _ in range(rng.randint(1, 2)):
                x = rng.randint(0, max(0, image.width - 1))
                draw.line((x, 0, x + rng.choice([-1, 0, 1]), image.height), fill=(230, 230, 230), width=1)
            operations.append("faint_scan_column_artifact")

        current_long_edge = max(image.size)
        target_long_edge = int(
            round(
                min(
                    current_long_edge,
                    rng.choice([640, 704, 768, 832, 896, 960, 1024]) * rng.uniform(0.92, 1.06),
                )
            )
        )
        if current_long_edge > target_long_edge:
            resize_scale = float(target_long_edge) / float(current_long_edge)
            resample = Image.Resampling.BICUBIC if resize_scale > 0.58 else Image.Resampling.LANCZOS
            image = image.resize(
                (
                    max(64, int(round(image.width * resize_scale))),
                    max(64, int(round(image.height * resize_scale))),
                ),
                resample=resample,
            )
            operations.append("patent_crop_long_edge_scale_normalization")
    else:
        operations.append("moldepictor_source_image_copy")

    if rng.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.06, 0.34)))
        operations.append("light_scan_blur")
    if rng.random() < 0.62:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        noise_sigma = rng.uniform(0.45, 2.20)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, noise_sigma, arr.shape)
        image = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"), mode="L").convert("RGB")
        operations.append("light_scan_noise")
    if rng.random() < 0.68:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        paper_mask = arr > 210
        dark_speckles = (np_rng.random(arr.shape) < rng.uniform(0.00016, 0.00095)) & paper_mask
        gray_speckles = (np_rng.random(arr.shape) < rng.uniform(0.00032, 0.00150)) & paper_mask
        if dark_speckles.any() or gray_speckles.any():
            arr[dark_speckles] = np.minimum(arr[dark_speckles], np_rng.integers(98, 186, size=int(dark_speckles.sum())))
            arr[gray_speckles] = np.minimum(arr[gray_speckles], np_rng.integers(178, 228, size=int(gray_speckles.sum())))
            image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L").convert("RGB")
            operations.append("fragment_like_sparse_scan_speckles")
    if rng.random() < 0.30:
        arr = np.asarray(image.convert("L")).astype(np.int16)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        ink_mask = arr < 120
        dropout = (np_rng.random(arr.shape) < rng.uniform(0.00010, 0.00065)) & ink_mask
        if dropout.any():
            arr[dropout] = np_rng.integers(176, 232, size=int(dropout.sum()))
            image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L").convert("RGB")
            operations.append("tiny_ink_dropout")
    if rng.random() < 0.46:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(76, 93))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        operations.append("jpeg_roundtrip")
    if rng.random() < 0.52:
        gray = image.convert("L")
        if rng.random() < 0.46:
            gray = ImageOps.autocontrast(gray, cutoff=rng.choice([0, 1]))
            operations.append("post_document_domain_autocontrast")
        if rng.random() < 0.62:
            gray = ImageEnhance.Contrast(gray).enhance(rng.uniform(1.03, 1.16))
            operations.append("post_document_domain_contrast")
        if rng.random() < 0.36:
            gray = ImageEnhance.Sharpness(gray).enhance(rng.uniform(1.04, 1.20))
            operations.append("post_document_domain_sharpen")
        arr = np.asarray(gray).astype(np.int16)
        ink_mask = arr < 210
        if ink_mask.any() and rng.random() < 0.62:
            arr[ink_mask] = np.clip(arr[ink_mask] * rng.uniform(0.88, 0.97), 0, 255)
            operations.append("post_document_domain_dark_strokes")
        if rng.random() < 0.46:
            arr = np.clip(
                arr + np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(0.12, 0.55), arr.shape),
                0,
                255,
            )
            operations.append("post_document_domain_fine_noise")
        gray = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L")
        if rng.random() < 0.24:
            buffer = BytesIO()
            gray.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(86, 96))
            buffer.seek(0)
            gray = Image.open(buffer).convert("L")
            operations.append("post_document_domain_jpeg_roundtrip")
        image = gray.convert("RGB")

    def xform_point(x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) * old_w + pad_left) / max(1.0, float(coordinate_canvas_w)),
            (float(y) * old_h + pad_top) / max(1.0, float(coordinate_canvas_h)),
        )

    updated_atoms = []
    for atom in atom_coordinates:
        updated = dict(atom)
        updated["x"], updated["y"] = xform_point(float(atom["x"]), float(atom["y"]))
        updated_atoms.append(updated)

    arr_final = np.asarray(image.convert("L"))
    dark_ratio = float((arr_final < 245).mean())
    ink_ratio = float((arr_final < 220).mean())
    blockers = []
    if dark_ratio < 0.001:
        blockers.append("ordinary_document_context_blank")
    if dark_ratio > 0.62:
        blockers.append("ordinary_document_context_too_dense")
    if min(image.size) < 80:
        blockers.append("ordinary_document_context_too_small")

    contract = {
        "schema_version": "ordinary_document_domain_policy_v2",
        "policy": "molgrapher_complete_molecule_coordinate_preserving_patent_literature_input_domain_v1",
        "enabled": bool(enabled),
        "operations": operations,
        "allowed_operations": operations,
        "seed": int(seed),
        "padding_px": [int(pad_left), int(pad_top), int(pad_right), int(pad_bottom)],
        "old_image_size": [int(old_w), int(old_h)],
        "new_image_size": [int(coordinate_canvas_w), int(coordinate_canvas_h)],
        "final_image_size": [int(image.width), int(image.height)],
        "resize_scale_after_padding": float(resize_scale),
        "paper_profile": paper_profile,
        "paper_background_gray": int(paper_background),
        "paper_noise_sigma": float(paper_noise_sigma),
        "transparent_white_structure_composite": "alpha_antialiased" if enabled else "not_applied",
        "geometry_mutation_allowed": False,
        "graph_topology_mutation_allowed": False,
        "coordinate_mutation_after_render": bool(enabled),
        "atom_coordinates_synchronized": True,
        "image_to_graph_orientation_alignment_preserved": True,
        "attachment_like_rows_allowed": False,
        "complete_path": "direct_original_molnextr",
        "dark_pixel_ratio": dark_ratio,
        "ink_pixel_ratio": ink_ratio,
        "blank": dark_ratio < 0.001,
        "dense": dark_ratio > 0.62,
        "molnextr_input_quality_passed": not blockers,
        "blockers": blockers,
    }
    return image, updated_atoms, contract


def load_heldout_image_sha1(paths: list[str]) -> set[str]:
    heldout: set[str] = set()
    for path_text in paths:
        csv_path = Path(path_text)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_path = row_image_path(row, csv_path)
                if image_path.exists():
                    heldout.add(sha1_file(image_path))
    return heldout


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_id",
        "source_arrow",
        "file_path",
        "SMILES",
        "smiles",
        "structure_type_bucket",
        "structure_type_label",
        "image_width",
        "image_height",
        "render_quality",
        "reliable_training_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_parquet_rows(
    parquet_path: Path,
    columns: list[str],
    *,
    start_row: int,
    max_source_rows: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    row_offset = 0
    row_limit = start_row + max_source_rows if max_source_rows > 0 else None
    for row_group_index in range(parquet_file.num_row_groups):
        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        row_group_end = row_offset + row_group_rows
        if row_group_end <= start_row:
            row_offset = row_group_end
            continue
        if row_limit is not None and row_offset >= row_limit:
            break
        table = parquet_file.read_row_group(row_group_index, columns=columns)
        for local_index, item in enumerate(table.to_pylist()):
            row_index = row_offset + local_index
            if row_index < start_row:
                continue
            if row_limit is not None and row_index >= row_limit:
                break
            yield row_index, item
        row_offset = row_group_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MolGrapher-Synthetic parquet rows into pose-factory ordinary negatives.")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--disable-document-context", action="store_true")
    parser.add_argument("--heldout-csv", action="append", default=[])
    parser.add_argument("--heldout-image-csv", action="append", default=[])
    args = parser.parse_args()

    from PIL import Image
    import io
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    heldout_canonical = load_heldout_canonical_smiles(list(args.heldout_csv))
    heldout_image_sha1 = load_heldout_image_sha1(list(args.heldout_image_csv))
    rows = []
    failures = []
    failure_reasons: Counter[str] = Counter()
    source_rows_seen = 0
    source_rows_considered = 0
    columns = ["id", "image", "mol", "smiles", "keypoints"]
    for index, item in iter_parquet_rows(
        parquet_path,
        columns,
        start_row=int(args.start_row),
        max_source_rows=int(args.max_source_rows),
    ):
        source_rows_seen += 1
        source_rows_considered += 1
        if 0 < int(args.max_rows) <= len(rows):
            break
        try:
            source_id = f"molgrapher_synthetic:{item['id']}"
            image_bytes = item["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            saved_image = io.BytesIO()
            image.save(saved_image, format="PNG")
            saved_image_bytes = saved_image.getvalue()
            image_hash = sha1_bytes(saved_image_bytes)
            if image_hash in heldout_image_sha1:
                raise ValueError("row image SHA1 overlaps held-out benchmark")
            image_rel = Path("images") / f"{source_id.replace(':', '_')}.png"
            width, height = image.size
            mol_text = str(item.get("mol") or "")
            row_smiles = str(item.get("smiles") or "")
            atom_coordinates = parse_keypoints(item.get("keypoints"), width, height)
            bonds, symbols = mol_bonds_and_symbols(mol_text)
            if len(atom_coordinates) != len(symbols):
                raise ValueError(f"keypoint atom count mismatch: {len(atom_coordinates)} != {len(symbols)}")
            for coord, symbol in zip(atom_coordinates, symbols):
                coord["token"] = symbol
            mol_canonical, row_canonical, row_matches_mol = canonical_smiles_pair(mol_text, row_smiles)
            if not row_matches_mol:
                raise ValueError(f"row SMILES/mol mismatch: row={row_canonical} mol={mol_canonical}")
            if row_canonical in heldout_canonical:
                raise ValueError("row SMILES canonical overlaps held-out benchmark")
            image, atom_coordinates, ordinary_document_domain_policy = apply_ordinary_document_context(
                image,
                atom_coordinates,
                seed=int(hashlib.sha1(f"ordinary_document_context:{source_id}".encode("utf-8")).hexdigest()[:8], 16),
                enabled=not bool(args.disable_document_context),
            )
            if ordinary_document_domain_policy.get("molnextr_input_quality_passed") is not True:
                raise ValueError(
                    "ordinary document context quality failed: "
                    + ",".join(str(item) for item in ordinary_document_domain_policy.get("blockers") or ["unknown"])
                )
            final_image = io.BytesIO()
            image.save(final_image, format="PNG")
            final_image_bytes = final_image.getvalue()
            (output_dir / image_rel).write_bytes(final_image_bytes)
            pose_alignment = {
                "image_to_graph_orientation_alignment": True,
                "coordinate_mutation_after_render": not bool(args.disable_document_context),
                "orientation_policy": "molgrapher_keypoints_normalized_to_generated_document_context_image",
                "synchronized_after_augmentation": True,
                "alignment_evidence": "source_image_ink_composited_to_patent_paper_with_normalized_keypoints_synchronized_by_padding_scale_no_rotation_flip_perspective",
            }
            render_quality = {
                "schema_version": POSE_FACTORY_SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "backend": "molgrapher_synthetic_300k",
                "backend_references": BACKEND_REFERENCES,
                "dataset_lineage": formal_sim_dataset_lineage(
                    branch="ordinary_complete_router_negative",
                    source_dataset="docling-project/MolGrapher-Synthetic-300K",
                    source_file=str(parquet_path),
                    source_record_id=str(item["id"]),
                    parent_source_group=f"molgrapher_synthetic:{item['id']}",
                    generation_stage="source_preserving_conversion_formal_sim_router_negative",
                    shard_id=output_dir.name,
                ),
                "simulation_policy": formal_simulation_policy(
                    branch="ordinary_complete_router_negative",
                    allowed_operations=list(ordinary_document_domain_policy.get("operations") or []),
                    coordinate_mutation_policy="synchronized_normalized_padding_then_uniform_image_scale_no_rotation_flip_perspective",
                    geometry_contract="molgrapher_keypoints_normalized_to_generated_document_context_image_pose_contract_v2",
                    formal_capable=True,
                    image_synchronized=True,
                    atom_coordinates_synchronized=True,
                    endpoint_coordinates_synchronized=False,
                    ocr_boxes_synchronized=False,
                    svg_or_connector_anchors_synchronized=False,
                ),
                "ordinary_document_domain_policy": ordinary_document_domain_policy,
                "source_dataset": "docling-project/MolGrapher-Synthetic-300K",
                "source_record_id": str(item["id"]),
                "structure_type": "complete_compound",
                "image_width": image.width,
                "image_height": image.height,
                "atom_coordinates": atom_coordinates,
                "bonds": bonds,
                "layout_seed": int(item["id"]),
                "render_style": "moldepictor_synthetic_patent_context",
                "coord_policy": "molgrapher_keypoints_normalized_to_generated_document_context_image",
                "pose_alignment": pose_alignment,
                "render_provenance": {
                    "source_kind": "external_dataset_conversion",
                    "source_dataset": "docling-project/MolGrapher-Synthetic-300K",
                    "source_parquet": str(parquet_path),
                    "source_record_id": str(item["id"]),
                    "image_field": "image.bytes",
                    "mol_field": "mol",
                    "smiles_field": "smiles",
                    "keypoints_field": "keypoints",
                    "image_sha1": sha1_bytes(final_image_bytes),
                    "source_image_sha1": sha1_bytes(image_bytes),
                    "pre_document_context_png_sha1": image_hash,
                    "conversion_policy": "copy_source_image_verify_smiles_mol_canonical_then_coordinate_synchronized_document_domain_generation",
                },
                "quality_gates": {
                    "image_readable": True,
                    "atom_coordinates_present": True,
                    "external_backend_referenced": True,
                    "smiles_graph_consistent": row_matches_mol,
                    "image_to_graph_orientation_alignment": True,
                    "molnextr_pose_synchronized_after_augmentation": True,
                    "ordinary_molnextr_input_quality_passed": bool(
                        ordinary_document_domain_policy.get("molnextr_input_quality_passed")
                    ),
                    "ordinary_document_context_present": not bool(args.disable_document_context),
                },
                "graph_consistency": {
                    "canonical_smiles": row_canonical,
                    "mol_block_canonical_smiles": mol_canonical,
                    "row_smiles_canonical_matches_mol": row_matches_mol,
                    "atom_count": len(symbols),
                    "bond_count": len(bonds),
                    "dummy_atom_indices": [],
                    "single_dummy_atom": False,
                    "anchor_dummy_bond_present": False,
                },
                "style_augmentations": list(ordinary_document_domain_policy.get("operations") or []),
            }
            rows.append(
                {
                    "source_id": source_id,
                    "source_arrow": "pose_factory:ordinary_molgrapher_synthetic",
                    "file_path": str(image_rel),
                    "SMILES": row_smiles,
                    "smiles": row_smiles,
                    "structure_type_bucket": "ordinary_structure",
                    "structure_type_label": "complete_molecule",
                    "image_width": str(image.width),
                    "image_height": str(image.height),
                    "render_quality": json.dumps(render_quality, sort_keys=True),
                    "reliable_training_label": "true",
                }
            )
        except Exception as exc:
            error = str(exc)
            failure_reasons[error.split(":", 1)[0]] += 1
            failures.append({"index": index, "id": str(item.get("id")), "error": error})

    csv_path = output_dir / "ordinary_negative.csv"
    write_csv(csv_path, rows)
    manifest = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_parquet": str(parquet_path),
        "start_row": int(args.start_row),
        "max_source_rows": int(args.max_source_rows),
        "source_rows_seen": int(source_rows_seen),
        "source_rows_considered": int(source_rows_considered),
        "csv": str(csv_path),
        "row_count": len(rows),
        "failure_count": len(failures),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "heldout_canonical_blacklist_size": len(heldout_canonical),
        "heldout_image_sha1_blacklist_size": len(heldout_image_sha1),
        "failures": failures[:50],
        "status": "candidate_large_ordinary_negative_requires_validation_and_review",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

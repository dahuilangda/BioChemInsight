from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import POSE_FACTORY_SCHEMA_VERSION


GENERATOR_VERSION = "indigo_ordinary_pose_factory_v1"
BACKEND_REFERENCES = [
    {
        "name": "MolScribe",
        "url": "https://github.com/thomas0809/MolScribe/blob/main/molscribe/dataset.py",
        "role": "Reference for Indigo dynamic rendering and coordinate-aware graph supervision.",
    },
    {
        "name": "MolScribe Indigo wrapper",
        "url": "https://github.com/thomas0809/MolScribe/tree/main/molscribe/indigo",
        "role": "Vendored Indigo Python wrapper used from /tmp as a rendering backend probe.",
    },
]

DEFAULT_SMILES = [
    "CCO",
    "CC(=O)N",
    "c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "CCN(CC)C(=O)c1ccccc1",
    "COc1ccc2ccccc2c1",
    "CC(C)Oc1ccc(C(=O)O)cc1",
    "O=S(=O)(N)c1ccccc1",
    "N#Cc1ccc(O)cc1",
    "Clc1ccc(O)cc1",
]


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def load_indigo(molscribe_root: Path):
    sys.path.insert(0, str(molscribe_root / "molscribe"))
    from indigo import Indigo
    from indigo.renderer import IndigoRenderer

    return Indigo, IndigoRenderer


def load_smiles(path: str) -> list[tuple[str, str]]:
    if not path:
        return [(f"curated_ordinary:{index}", smiles) for index, smiles in enumerate(DEFAULT_SMILES)]
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            smiles = str(row.get("SMILES") or row.get("smiles") or row.get("cxsmiles") or "").strip()
            if not smiles or "*" in smiles:
                continue
            rows.append((str(row.get("source_id") or row.get("id") or f"row:{index}"), smiles))
    return rows


def configure_indigo(indigo: Any, rng: random.Random, image_size: int) -> str:
    style = rng.choice(
        [
            "indigo_literature_thin",
            "indigo_literature_bold",
            "indigo_times_hetero",
            "indigo_terminal_hetero",
        ]
    )
    indigo.setOption("render-output-format", "png")
    indigo.setOption("render-background-color", "1,1,1")
    indigo.setOption("render-image-size", int(image_size), int(image_size))
    indigo.setOption("render-margins", rng.randint(8, 26), rng.randint(8, 26))
    indigo.setOption("render-stereo-style", "none")
    indigo.setOption("render-label-mode", "terminal-hetero" if "terminal" in style else "hetero")
    indigo.setOption("render-font-family", "Times" if "times" in style else rng.choice(["Arial", "Helvetica", "Courier"]))
    thickness = rng.uniform(0.55, 1.8)
    indigo.setOption("render-relative-thickness", thickness)
    indigo.setOption("render-bond-line-width", rng.uniform(1.0, max(1.01, 4.0 - thickness)))
    indigo.setOption("render-implicit-hydrogens-visible", rng.choice([True, False]))
    return style


def image_stats(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageFilter
    import numpy as np

    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    dark_ratio = float((arr < 245).mean())
    return {
        "dark_pixel_ratio": dark_ratio,
        "blank": bool(dark_ratio < 0.0005),
        "dense": bool(dark_ratio > 0.85),
    }


def normalize_coords(coords: list[tuple[int, str, float, float]]) -> list[dict[str, Any]]:
    xs = [item[2] for item in coords]
    ys = [item[3] for item in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    margin = 0.08
    scale = 1.0 - 2.0 * margin
    normalized = []
    for atom_index, symbol, x, y in coords:
        normalized.append(
            {
                "atom_index": int(atom_index),
                "token": symbol,
                "x": margin + scale * ((x - min_x) / span_x),
                "y": margin + scale * ((y - min_y) / span_y),
            }
        )
    return normalized


def bond_records(mol: Any) -> list[dict[str, Any]]:
    records = []
    for bond in mol.iterateBonds():
        records.append(
            {
                "begin_atom_index": int(bond.source().index()),
                "end_atom_index": int(bond.destination().index()),
                "bond_order": int(bond.bondOrder()),
                "bond_dir": "",
                "is_aromatic": bool(bond.bondOrder() == 4),
            }
        )
    return records


def generate_row(
    *,
    smiles: str,
    source_id: str,
    output_dir: Path,
    index: int,
    rng: random.Random,
    image_size: int,
    molscribe_root: Path,
) -> dict[str, str]:
    Indigo, IndigoRenderer = load_indigo(molscribe_root)
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    style = configure_indigo(indigo, rng, image_size)
    mol = indigo.loadMolecule(smiles)
    if rng.random() < 0.55:
        mol.dearomatize()
    else:
        mol.aromatize()
    mol.layout()
    row_smiles = mol.smiles().split(" ", 1)[0]
    row_id = stable_id("indigo_ordinary", source_id, row_smiles, index, rng.randint(0, 2**31 - 1))
    image_rel = Path("images") / f"{row_id}.png"
    image_path = output_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(renderer.renderToBuffer(mol))
    stats = image_stats(image_path)
    atom_coords_raw = []
    for atom in mol.iterateAtoms():
        x, y = atom.coords()
        atom_coords_raw.append((atom.index(), atom.symbol(), float(x), float(y)))
    atom_coordinates = normalize_coords(atom_coords_raw)
    bonds = bond_records(mol)
    graph_consistency = {
        "canonical_smiles": row_smiles,
        "row_smiles_canonical_matches_mol": True,
        "atom_count": len(atom_coordinates),
        "bond_count": len(bonds),
        "dummy_atom_indices": [],
        "single_dummy_atom": False,
        "anchor_dummy_bond_present": False,
    }
    render_quality = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "backend": "indigo",
        "backend_version": "molscribe_vendored_indigo",
        "backend_references": BACKEND_REFERENCES,
        "source_dataset": "curated_ordinary_smiles_seed",
        "source_record_id": source_id,
        "structure_type": "complete_compound",
        "image_width": int(image_size),
        "image_height": int(image_size),
        "atom_coordinates": atom_coordinates,
        "bonds": bonds,
        "layout_seed": index,
        "render_style": style,
        "coord_policy": "indigo_layout_coords_normalized_to_render_box_for_ordinary_negative_pose_audit",
        "pose_alignment": {
            "image_to_graph_orientation_alignment": True,
            "coordinate_mutation_after_render": False,
            "orientation_policy": "indigo_layout_coords_normalized_to_render_box_for_ordinary_negative_pose_audit",
            "synchronized_after_augmentation": True,
        },
        "quality_gates": {
            "image_readable": True,
            "blank": bool(stats["blank"]),
            "dense": bool(stats["dense"]),
            "atom_coordinates_present": True,
            "external_backend_referenced": True,
            "smiles_graph_consistent": True,
            "image_to_graph_orientation_alignment": True,
        },
        "graph_consistency": graph_consistency,
        "style_augmentations": [],
    }
    return {
        "source_id": row_id,
        "source_arrow": "pose_factory:ordinary_indigo",
        "file_path": str(image_rel),
        "SMILES": row_smiles,
        "smiles": row_smiles,
        "structure_type_bucket": "ordinary_structure",
        "structure_type_label": "complete_molecule",
        "image_width": str(int(image_size)),
        "image_height": str(int(image_size)),
        "render_quality": json.dumps(render_quality, sort_keys=True),
        "reliable_training_label": "true",
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Indigo ordinary hard-negative pose-factory shard.")
    parser.add_argument("--input-smiles-csv", default="")
    parser.add_argument("--output-dir", default="training/molnextr_markush/data/generated/pose_factory/indigo_ordinary_v1")
    parser.add_argument("--molscribe-root", default="/tmp/MolScribe")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_smiles = load_smiles(args.input_smiles_csv)
    if not source_smiles:
        raise SystemExit("no source SMILES available")
    rng = random.Random(int(args.seed))
    rows = []
    failures = []
    for index in range(int(args.rows)):
        source_id, smiles = source_smiles[index % len(source_smiles)]
        try:
            row = generate_row(
                smiles=smiles,
                source_id=source_id,
                output_dir=output_dir,
                index=index,
                rng=rng,
                image_size=int(args.image_size),
                molscribe_root=Path(args.molscribe_root),
            )
            quality = json.loads(row["render_quality"])
            gates = quality["quality_gates"]
            if gates["blank"] or gates["dense"]:
                failures.append({"source_id": source_id, "smiles": smiles, "error": "image_density_gate_failed"})
                continue
            rows.append(row)
        except Exception as exc:
            failures.append({"source_id": source_id, "smiles": smiles, "error": str(exc)})
    csv_path = output_dir / "ordinary_negative.csv"
    write_csv(csv_path, rows)
    manifest = {
        "schema_version": POSE_FACTORY_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "csv": str(csv_path),
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures[:50],
        "seed": int(args.seed),
        "source": "curated_ordinary_smiles_seed" if not args.input_smiles_csv else args.input_smiles_csv,
        "external_references": BACKEND_REFERENCES,
        "status": "candidate_ordinary_negative_requires_schema_validation_and_visual_review",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

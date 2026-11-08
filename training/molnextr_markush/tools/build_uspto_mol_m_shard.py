"""Production adapter: USPTO-MOL-M 54k parquets -> pose_factory aggregate CSVs.

Emits aggregate rows (render_quality JSON with atom_coordinates/bonds) that
`build_moe_df` consumes, so the shard flows through the standard contract —
turning the 33,792 symbol-only dead rows (§44.23) into label-dense markush
training rows with real patent pixels and grounded variable isotopes.

Placement: each page render is a single-markush layout, so the ink bounding
box (+margin) is the structure crop (prototype validated, §44.23). Atom
coordinates: the CXSMILES |(x,y,...)| block mapped affinely onto the ink
bbox, then normalized by the crop size. Isotopes: $...$ labels implanted via
implant_markush_attachment_isotopes (R13 -> [13*]).

Usage:
  python tools/build_uspto_mol_m_shard.py \
    --parquet-glob '.../uspto-mol-m-54k/train-*.parquet' \
    --images-dir .../uspto_mol_m_images --out-csv .../uspto_mol_m_shard.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rdkit import Chem, RDLogger  # noqa: E402

from training.molnextr_markush.src.moe_dataset import (  # noqa: E402
    implant_markush_attachment_isotopes,
    parse_cxsmiles_dummy_labels,
    _strip_cxsmiles,
)
from training.molnextr_markush.tools.prototype_uspto_placement import (  # noqa: E402
    ink_crop_from_page_bytes,
)

RDLogger.DisableLog("rdApp.*")

BOND_ORDER_TO_INT = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
INK_MARGIN = 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    return parser.parse_args()


def mol_atom_graph(mol):
    """Per-atom mol-space coords + edge triples (u, v, bond-order-int)."""
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    coords = []
    for atom in mol.GetAtoms():
        if conf is not None:
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append((float(pos.x), float(pos.y)))
        else:
            coords.append((0.0, 0.0))
    edges = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.GetIsAromatic():
            order = "AROMATIC"
        else:
            order = str(bond.GetBondType()).split(".")[-1].upper()
        edges.append((u, v, BOND_ORDER_TO_INT.get(order, 1)))
    return coords, edges


def main():
    args = parse_args()
    import glob as globlib

    import pyarrow.parquet as pq

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = no_ink = no_dummy = parse_failed = coord_failed = 0
    handle = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=[
        "file_path", "SMILES", "node_coords", "node_coords_space", "edges",
        "source_arrow", "structure_type_bucket", "render_quality",
    ])
    writer.writeheader()

    for parquet_path in sorted(globlib.glob(args.parquet_glob)):
        parquet = pq.ParquetFile(parquet_path)
        shard = Path(parquet_path).stem
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                scanned += 1
                if args.limit and written >= args.limit:
                    handle.close()
                    print(f"limit reached: wrote {written} rows "
                          f"(scanned {scanned}, no_ink {no_ink}, no_dummy {no_dummy}, "
                          f"parse {parse_failed}, coords {coord_failed})", flush=True)
                    return
                try:
                    # cxsmiles_dataset carries the coordinate block (conformer)
                    # AND the $...$ variable labels; plain cxsmiles has neither.
                    cxsmiles = str(row.get("cxsmiles_dataset") or "")
                    base_smiles = _strip_cxsmiles(cxsmiles)
                    if not base_smiles or "*" not in base_smiles:
                        no_dummy += 1
                        continue
                    params = Chem.SmilesParserParams()
                    params.allowCXSMILES = True
                    params.strictCXSMILES = False
                    params.removeHs = False
                    mol = Chem.MolFromSmiles(cxsmiles, params)
                    if mol is None or mol.GetNumConformers() == 0:
                        parse_failed += 1
                        continue
                    labels = parse_cxsmiles_dummy_labels(cxsmiles)
                    smiles = implant_markush_attachment_isotopes(base_smiles, labels)
                    if not smiles or "*" not in smiles:
                        no_dummy += 1
                        continue
                    if Chem.MolFromSmiles(smiles, sanitize=False) is None:
                        parse_failed += 1
                        continue
                    mol_coords, edges = mol_atom_graph(mol)
                    xs = [c[0] for c in mol_coords]
                    ys = [c[1] for c in mol_coords]
                    span_x = (max(xs) - min(xs)) or 1.0
                    span_y = (max(ys) - min(ys)) or 1.0
                    min_x, min_y = min(xs), min(ys)

                    crop, bbox = ink_crop_from_page_bytes(
                        row["page_image"]["bytes"], margin=INK_MARGIN)
                    if crop is None:
                        no_ink += 1
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    bx0, by0, bx1, by1 = bbox
                    span_px_x = max(1, bx1 - bx0)
                    span_px_y = max(1, by1 - by0)

                    node_coords = []
                    atom_coordinates = []
                    ok = True
                    # crop origin is clamped at 0 when the ink bbox sits near
                    # the page edge; subtract the clamped origin, not the
                    # unclamped (bx0 - INK_MARGIN), or edge rows shift.
                    origin_x = max(0, bx0 - INK_MARGIN)
                    origin_y = max(0, by0 - INK_MARGIN)
                    for atom_index, ((mx, my), symbol) in enumerate(
                            zip(mol_coords, [
                                str(a.GetSymbol()) for a in mol.GetAtoms()])):
                        try:
                            nx_ = (mx - min_x) / span_x
                            ny_ = (my - min_y) / span_y
                        except (ZeroDivisionError, TypeError):
                            ok = False
                            break
                        # map into the ink bbox inside the crop, normalized
                        px = (bx0 + nx_ * span_px_x - origin_x) / crop_w
                        py = (by0 + ny_ * span_px_y - origin_y) / crop_h
                        px = min(1.0, max(0.0, px))
                        py = min(1.0, max(0.0, py))
                        node_coords.append([round(px, 4), round(py, 4)])
                        atom_coordinates.append({
                            "atom_index": atom_index,
                            "token": symbol,
                            "x": round(px, 4),
                            "y": round(py, 4),
                        })
                    if not ok or not node_coords:
                        coord_failed += 1
                        continue
                    image_name = Path(str(row.get("image_name") or "page")).stem
                    file_path = str(images_dir / f"usptomolm_{shard}_{image_name}_{written}.png")
                    if not cv2.imwrite(file_path, crop):
                        no_ink += 1
                        continue
                    edges_json = json.dumps(
                        [[u, v, t] for (u, v, t) in edges], separators=(",", ":"))
                    render_quality = json.dumps({
                        "atom_coordinates": atom_coordinates,
                        "bonds": [
                            {"begin_atom_index": u, "end_atom_index": v,
                             "bond_order": {1: "SINGLE", 2: "DOUBLE",
                                            3: "TRIPLE", 4: "AROMATIC"}.get(t, "SINGLE"),
                             "is_aromatic": t == 4}
                            for (u, v, t) in edges
                        ],
                    }, separators=(",", ":"))
                    writer.writerow({
                        "file_path": file_path,
                        "SMILES": smiles,
                        "node_coords": json.dumps(node_coords, separators=(",", ":")),
                        "node_coords_space": "normalized_image",
                        "edges": edges_json,
                        "source_arrow": f"markushgrapher2:uspto_mol_m_54k:{shard}",
                        "structure_type_bucket": "markush_layout",
                        "render_quality": render_quality,
                    })
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    parse_failed += 1
                    if parse_failed <= 3:
                        print(f"row error: {exc!r}", flush=True)
        print(f"{Path(parquet_path).name}: scanned={scanned} written={written}", flush=True)

    handle.close()
    print(f"DONE: scanned={scanned} written={written} no_ink={no_ink} "
          f"no_dummy={no_dummy} parse={parse_failed} coords={coord_failed}", flush=True)


if __name__ == "__main__":
    main()

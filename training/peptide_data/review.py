from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import signal
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cyclopeptide2d import depict_record
from peptidecore import core_library, from_helm
from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from molsight.cpu import available_cpus, initialize_worker

from training.peptide_data.quality import chemical_identity, geometry_report, stereo_roundtrip
from training.peptide_data.svg_quality import drawing_report

logger = logging.getLogger(__name__)

STYLES = {
    "serif_clean": {
        "font": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "font_size": 16,
        "line_width": 2.0,
        "bond_length": 24,
        "padding": 0.08,
    },
    "sans_clean": {
        "font": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "font_size": 14,
        "line_width": 1.6,
        "bond_length": 24,
        "padding": 0.08,
    },
}


class LayoutTimeout(TimeoutError):
    pass


def _timeout(*_):
    raise LayoutTimeout("layout_time_limit")


def _draw(molecule, style, size, *, vector):
    drawer_type = rdMolDraw2D.MolDraw2DSVG if vector else rdMolDraw2D.MolDraw2DCairo
    drawer = drawer_type(*size)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fontFile = style["font"]
    options.fixedFontSize = style["font_size"]
    options.fixedBondLength = style["bond_length"]
    options.bondLineWidth = style["line_width"]
    options.additionalAtomLabelPadding = 0.14
    options.padding = style["padding"]
    options.prepareMolsBeforeDrawing = False
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    positions = np.asarray(
        [
            (drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y)
            for i in range(molecule.GetNumAtoms())
        ]
    )
    return drawer.GetDrawingText(), positions


def _render(molecule, directory, style_name, sizes):
    from molsight.dataset import get_transforms

    style = STYLES[style_name]
    font = Path(style["font"])
    if not font.is_file():
        raise FileNotFoundError(font)
    positions = np.asarray(molecule.GetConformer().GetPositions())[:, :2]
    bonds = np.asarray(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()]
    )
    bond_length = np.median(
        np.linalg.norm(positions[bonds[:, 0]] - positions[bonds[:, 1]], axis=1)
    )
    extent = np.ptp(positions, axis=0) * style["bond_length"] / bond_length
    canvas = tuple(max(220, int(value + 6 * style["bond_length"])) for value in extent)
    if max(canvas) > 8192:
        raise ValueError("native_canvas_exceeds_limit")
    attempts = []
    for attempt in range(4):
        svg, svg_positions = _draw(molecule, style, canvas, vector=True)
        quality = drawing_report(svg, molecule, svg_positions)
        attempts.append({"canvas": canvas, "reasons": quality["reasons"]})
        if not set(quality["reasons"]) & {"glyph_ink_clearance", "clipped_drawing"}:
            break
        if attempt < 3:
            canvas = tuple(int(np.ceil(value * 1.2)) for value in canvas)
            if max(canvas) > 8192:
                raise ValueError("native_canvas_exceeds_limit")
    raster, draw_positions = _draw(molecule, style, canvas, vector=False)
    if not np.allclose(svg_positions, draw_positions, atol=1e-5):
        raise RuntimeError("vector_raster_transform_mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "drawing.svg").write_text(svg)
    (directory / "native.png").write_bytes(raster)
    image = np.asarray(Image.open(io.BytesIO(raster)).convert("RGB"))
    report = {
        "style": style_name,
        "settings": style,
        "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "canvas": canvas,
        "canvas_attempts": attempts,
        "atom_positions": draw_positions.tolist(),
        "inputs": [],
    }
    native_bonds = np.linalg.norm(
        draw_positions[bonds[:, 0]] - draw_positions[bonds[:, 1]], axis=1
    )
    for size in sizes:
        args = SimpleNamespace(input_size=size, encoder="efficientvit")
        transformed = get_transforms(args, augment=False, rotate=False, debug=True)(
            image=image.copy(), keypoints=draw_positions.tolist()
        )
        final_image = transformed["image"]
        final_positions = np.asarray(transformed["keypoints"])[:, :2]
        final_bonds = np.linalg.norm(
            final_positions[bonds[:, 0]] - final_positions[bonds[:, 1]], axis=1
        )
        scale = float(np.median(final_bonds / native_bonds))
        font_height = style["font_size"] * scale
        stroke_width = style["line_width"] * scale
        reasons = []
        if font_height < 8:
            reasons.append("nominal_font_below_8_pixels")
        if float(np.median(final_bonds)) < 10:
            reasons.append("median_bond_below_10_pixels")
        if stroke_width < 0.7:
            reasons.append("nominal_stroke_below_0.7_pixels")
        resolution_reasons = list(reasons)
        vector_quality = drawing_report(svg, molecule, svg_positions, scale=scale)
        reasons.extend(vector_quality["reasons"])
        path = directory / f"input_{size}.png"
        Image.fromarray(final_image).save(path)
        report["inputs"].append(
            {
                "size": size,
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "atom_positions": final_positions.tolist(),
                "median_bond_pixels": float(np.median(final_bonds)),
                "nominal_font_pixels": font_height,
                "nominal_stroke_pixels": stroke_width,
                "resolution_screen_pass": not resolution_reasons,
                "input_quality_pass": not reasons,
                "reasons": reasons,
                "drawing_quality": vector_quality,
                "glyph_review_required": True,
                "training_release_approved": False,
            }
        )
    return report


def _review(job):
    candidate, output, seed, maxiter, budget, sizes, engine = job
    start = time.monotonic()
    result = {
        "id": candidate["id"],
        "stratum": candidate["stratum"],
        "seed": seed,
        "status": "pending",
        "training_release_approved": False,
    }
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(budget)
    try:
        record = from_helm(candidate["helm"])
        if record.as_dict() != candidate["chemical_record"]:
            raise ValueError("chemical_record_mismatch")
        if engine == "coordgen":
            molecule = record.molecule
            rdCoordGen.AddCoords(molecule)
            result["layout_method"] = "coordgen"
        elif engine == "rdkit":
            molecule = record.molecule
            rdDepictor.Compute2DCoords(molecule, forceRDKit=True, useRingTemplates=True)
            result["layout_method"] = "rdkit_2d"
        else:
            layout = depict_record(
                record, seed=seed, maxiter=maxiter, time_budget=budget * 0.95
            )
            result["layout_method"] = "cyclopeptide2d"
            result["layout"] = layout.report
            if not layout.accepted:
                result["status"] = "layout_rejected"
                return result
            molecule = layout.molecule
        result["geometry"] = geometry_report(molecule)
        if not result["geometry"]["accepted"]:
            result["status"] = "geometry_rejected"
            return result
        molecule = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)
        if chemical_identity(molecule) != record.smiles or not stereo_roundtrip(
            molecule
        ):
            result["status"] = "stereo_rejected"
            return result
        directory = Path(output) / "images" / candidate["id"]
        directory.mkdir(parents=True, exist_ok=False)
        Chem.MolToMolFile(molecule, str(directory / "drawing.mol"), includeStereo=True)
        result["images"] = [
            _render(molecule, directory / style, style, sizes) for style in STYLES
        ]
        result["status"] = "rendered_for_review"
    except LayoutTimeout:
        result["status"] = "timeout"
    except Exception as error:
        logger.exception("Peptide review failed for %s", candidate["id"])
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        result["seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--budget-seconds", type=int, default=60)
    parser.add_argument("--maxiter", type=int, default=350)
    parser.add_argument("--sizes", type=int, nargs="+", default=[384, 512, 768])
    parser.add_argument("--strata", nargs="+")
    parser.add_argument(
        "--layout-engine", choices=("residue", "rdkit", "coordgen"), default="residue"
    )
    args = parser.parse_args()
    if min(args.workers, args.budget_seconds, args.maxiter, *args.sizes) < 1:
        parser.error("Counts, dimensions, iterations and budgets must be positive")
    design = json.loads(args.design.read_text())
    if design["library"]["digest"] != core_library().digest:
        raise ValueError("monomer_library_mismatch")
    candidates = [
        record
        for record in design["records"]
        if not args.strata or record["stratum"] in args.strata
    ]
    if not candidates:
        raise ValueError("No matching candidate strata")
    args.out.mkdir(parents=True, exist_ok=False)
    source_directories = [
        Path(__file__).parent,
        Path(__import__("peptidecore").__file__).parent,
        Path(__import__("cyclopeptide2d").__file__).parent,
        Path(__import__("molsight").__file__).parent,
    ]
    source_paths = sorted(
        {path for root in source_directories for path in root.rglob("*.py")}
    )
    sources = {
        str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    for path in source_paths:
        target = args.out / "sources" / path.relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    jobs = [
        (
            candidate,
            str(args.out),
            int(candidate["id"][:7], 16),
            args.maxiter,
            args.budget_seconds,
            args.sizes,
            args.layout_engine,
        )
        for candidate in candidates
    ]
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker, initargs=(0,)
    ) as pool:
        for result in pool.map(_review, jobs):
            results.append(result)
            (args.out / f"{result['id']}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {key: result[key] for key in ("id", "stratum", "status", "seconds")}
                ),
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
        "rdkit_version": rdBase.rdkitVersion,
        "library_digest": core_library().digest,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "sources_unchanged": all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
            for path, value in sources.items()
        ),
        "layout_engine": args.layout_engine,
        "workers": args.workers,
        "budget_seconds": args.budget_seconds,
        "maxiter": args.maxiter,
        "counts": dict(Counter(result["status"] for result in results)),
        "training_release_approved": False,
        "records": results,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()

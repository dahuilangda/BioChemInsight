"""External-engine 2D layout path into the calibrated ALIGNED render style.

SMILES -> external engine 2D coordinates -> V2000 MOL block -> RDKit read
-> style_calibration.draw_molecule(ALIGNED) PNG, the same single render
path as the training corpus. Everything here runs offline except where a
function's docstring says otherwise.

Engine study record (per-engine install -> render seq53 (US20240287138A1
"SEQ ID NO: 53", 156 atoms, 44-membered peptide macroring, gold SMILES
data/gold_smiles_289.json key "3") -> compare against the real crop
data/real_crops/US20240287138A1/page_024_box_0.png, shape verdict; no
verdict was silently passed, /tmp/engine_comparison.png is the visual
record):

  openbabel gen2D  — OBBuilder.Build + the gen2D OBOp (verified identical
      coords from the obabel CLI and the openbabel-wheel bindings, max
      atom-coordinate diff 0.0).  Layout: coherent closed contour with
      side chains radiating outward, but the macroring reads ROUND, not
      the patent zigzag polygon; a few crowded runs.  Verdict: FAIL for
      zigzag, CLOSEST of the four candidates — selected.
  cdk 2.9 SDG      — StructureDiagramGenerator.generateCoordinates().
      Layout: the 44-ring collapses into a serpentine blob with dense
      bond/label collisions in the core.  Verdict: FAIL.
  cdk 2.8 SDG      — same collapsed serpentine as 2.9.  Verdict: FAIL.
  cdk 2.9 experimental — generateExperimentalCoordinates().  Same blob.
      Verdict: FAIL.  (Maven Central cdk-bundle 2.7.1/2.8/2.9 are 2.2-2.6
      KB placeholder jars; the working bundles come from GitHub releases.)
  nih cactus image — https://cactus.nci.nih.gov/chemical/structure/
      {url-encoded SMILES}/image returns a real PNG (Content-Type header
      misleadingly says image/gif; trust the magic bytes) but the layout
      is a perfect CIRCLE with sunburst backbone spokes and an unreadable
      ring core; image_width/height params are ignored and no coordinate
      output exists, so it cannot feed the RDKit ALIGNED render.
      Verdict: FAIL.
  pubchem 3d image — PUG-REST /compound/smiles/{smiles}/cids returns
      CID=0 for this novel patent compound: no database entry, no 3D
      conformer, no image.  Cannot render arbitrary gold SMILES.
      Verdict: UNAVAILABLE.

Consequence: NO external engine reproduces the patent zigzag macroring
(straight runs + corners, side chains outward) — that layout remains
zigzag_layout.py's job, with style_calibration.coordgen_coords as the
CoordGen fallback.  This module is the selected engine's coordinate path
(closest candidate) so engine layouts can be rendered in the exact
ALIGNED corpus style for pool mixing and ablations.
"""
from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

from style_calibration import ALIGNED, draw_molecule

RDLogger.DisableLog("rdApp.*")

ENGINES = ("openbabel",)


class EngineRenderError(Exception):
    """Base class: the engine-coordinate render path failed."""


class EngineUnsupportedError(EngineRenderError):
    """Requested engine is not one of the implemented engines."""


class EngineNotInstalledError(EngineRenderError):
    """The selected engine's libraries are missing from this machine."""


class EngineInvalidInputError(EngineRenderError):
    """The SMILES input did not parse."""


class EngineLayoutFailedError(EngineRenderError):
    """The engine ran but did not return a usable all-2D layout of the
    input graph."""


def openbabel_molblock(smiles: str) -> str:
    """SMILES -> OpenBabel 2D layout -> V2000 MOL block text.

    Uses the same code path as `obabel -ismi -omol --gen2D`: OBBuilder
    for the layout plus the gen2D OBOp (verified to produce coordinates
    identical to the CLI, max diff 0.0 on the seq53 study molecule).
    """
    try:
        from openbabel import openbabel as ob
    except ImportError as exc:
        raise EngineNotInstalledError(
            "openbabel python bindings not importable; install with "
            "`pip install openbabel-wheel` (or `conda install -c "
            "conda-forge openbabel`)") from exc

    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    mol = ob.OBMol()
    if not conv.ReadString(mol, smiles):
        raise EngineInvalidInputError("openbabel failed to read SMILES")

    # mirror `obabel -ismi -omol --gen2D` exactly: the gen2D OBOp does its
    # own build internally (a standalone OBBuilder().Build may return
    # False yet gen2D still yields a clean all-2D layout — verified on the
    # seq53 study molecule; planarity/atom-count checks below are the
    # actual guards, not the builder return code)
    op = ob.OBOp.FindType("gen2D")
    if op is None:
        raise EngineNotInstalledError(
            "openbabel 'gen2D' OBOp plugin not found in this build")
    op.Do(mol, "")

    out = ob.OBConversion()
    out.SetOutFormat("mol")
    return out.WriteString(mol)


def engine_molblock(smiles: str, engine: str) -> str:
    """Dispatch SMILES to the selected engine's 2D layout -> MOL block."""
    if engine not in ENGINES:
        raise EngineUnsupportedError(
            f"engine {engine!r} not implemented; available: {ENGINES}")
    if engine == "openbabel":
        return openbabel_molblock(smiles)
    raise EngineUnsupportedError(
        f"engine {engine!r} has no dispatch branch")  # guarded by ENGINES


def _load_engine_coords(smiles: str, engine: str) -> Chem.Mol:
    """Run the engine and return the RDKit molecule carrying its 2D
    coordinates, validated against the input graph (named failures)."""
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        raise EngineInvalidInputError("RDKit could not parse the SMILES input")

    block = engine_molblock(smiles, engine)
    mol = Chem.MolFromMolBlock(block)
    if mol is None or mol.GetNumConformers() == 0:
        raise EngineLayoutFailedError(
            "RDKit could not read the engine MOL block (no conformer)")
    if mol.GetNumAtoms() != ref.GetNumAtoms():
        raise EngineLayoutFailedError(
            f"engine layout atom count {mol.GetNumAtoms()} != input "
            f"graph {ref.GetNumAtoms()} — engine mangled the molecule")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        if abs(conf.GetAtomPosition(i).z) > 1e-3:
            raise EngineLayoutFailedError(
                f"engine coordinate for atom {i} has z != 0 — layout is "
                "not planar 2D")
    return mol


def render_smiles_via_engine(smiles: str, engine: str = "openbabel",
                             seed: int = 0) -> "PIL.Image.Image":
    """SMILES -> engine 2D coords -> ALIGNED-style tight-crop grayscale
    PIL image (the corpus render path; rotation / print-gray / noise as
    in style_calibration.draw_molecule)."""
    mol = _load_engine_coords(smiles, engine)
    return draw_molecule(mol, ALIGNED, random.Random(seed))


def render_file_via_engine(smiles_file: str, engine: str = "openbabel",
                           seed: int = 0):
    """Read the first field of a .smi line and render it."""
    with open(smiles_file, "r", encoding="utf-8") as fh:
        line = fh.readline().strip()
    smiles = line.split("\t")[0].split()[0] if line else ""
    if not smiles:
        raise EngineInvalidInputError(f"no SMILES in {smiles_file}")
    return render_smiles_via_engine(smiles, engine, seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Render SMILES via an external 2D-layout engine in the "
                    "ALIGNED corpus style (engine study: openbabel gen2D).")
    ap.add_argument("--smiles", help="SMILES string")
    ap.add_argument("--smiles-file", help=".smi file; first field used")
    ap.add_argument("--engine", default="openbabel",
                    help=f"layout engine, one of {ENGINES} "
                         "(default: openbabel)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-png", required=True, help="output PNG path")
    ap.add_argument("--out-mol", help="optional output path for the engine "
                                      "coordinate MOL block")
    args = ap.parse_args(argv)

    try:
        if bool(args.smiles) == bool(args.smiles_file):
            raise EngineInvalidInputError(
                "pass exactly one of --smiles / --smiles-file")
        mol = _load_engine_coords(
            args.smiles if args.smiles else
            open(args.smiles_file, encoding="utf-8").readline()
            .split("\t")[0].strip(), args.engine)
        img = draw_molecule(mol, ALIGNED, random.Random(args.seed))
        if args.out_mol:
            sys.stdout.flush()
            with open(args.out_mol, "w", encoding="utf-8") as fh:
                fh.write(Chem.MolToMolBlock(mol))
        img.save(args.out_png)
    except EngineRenderError as exc:
        print(f"render_via_engine: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(f"render_via_engine: wrote {args.out_png} "
          f"({img.width}x{img.height}, engine={args.engine})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""External-engine 2D layout path into the calibrated ALIGNED render style.

SMILES -> external engine 2D coordinates -> V2000 MOL block -> RDKit read
-> style_calibration.draw_molecule(ALIGNED) PNG, the same single render
path as the training corpus.  Everything here runs offline except the
one-time CDK bundle fetch (see cdk_molblock) and the engines whose
docstrings say otherwise.

Engine study record (per-engine install -> render the seq53 gold
macrocyclopeptide (US20240287138A1 "SEQ ID NO: 53", 44-membered peptide
macroring; both gold SMILES variants — data/gold_smiles_289.json key "3",
156 atoms, and data/real_crops/US20240287138A1/rows.jsonl, 150 atoms —
plus an 8-molecule probe set spanning 42-45 ring atoms) -> compare the
macroring against the real crop
data/real_crops/US20240287138A1/page_024_box_0.png; macroring skeletons
and shape metrics (circularity / zig-zag turn alternation / mean interior
angle) in experiments/engine_layout_eval/, visual record
experiments/engine_layout_eval/comparison_sheet.png; no verdict was
silently passed):

  cdk SDG (2.9 and 2.13 alike, GitHub-release bundles) — SMILES ->
      StructureDiagramGenerator -> V2000 MOL block.  The macroring comes
      out as the REAL-PATENT SAW-TOOTH POLYGON: irregular elongated
      contour, straight +-60 deg runs with sharp corners, circularity
      0.39-0.45, zig-zag alternation 0.45-0.49, mean interior angle
      ~120 deg, min non-bonded >= 1.49 bond units, identical shape on
      both seq53 SMILES variants and all 8 probes.  VERDICT: MATCH —
      the only mature engine reproducing the patent zigzag; default
      engine of this module.  (An earlier round recorded a "serpentine
      blob" for cdk 2.9; that was an evaluation artifact — re-measured
      2.9 and 2.13 reproduce the clean saw-tooth contour, see
      experiments/engine_layout_eval/skeleton_cdk29.png /
      skeleton_cdk213.png.  The Maven Central cdk-bundle 2.7.1-2.9
      artifacts are 2.2-2.6 KB placeholder jars and must not be used;
      working bundles come from GitHub releases, sha256-pinned here.)
  openbabel gen2D — OBBuilder.Build + the gen2D OBOp (verified identical
      coords from the obabel CLI and the openbabel-wheel bindings, max
      atom-coordinate diff 0.0).  Coherent closed contour with side
      chains radiating outward, but the macroring is a NEAR-PERFECT
      CIRCLE: circularity 0.00-0.03, mean interior angle 170-172 deg
      (same wheel style as PubChem PUG-REST).  Verdict: FAIL for the
      patent zigzag; retained for comparison studies and small-ring
      molecules.
  rdkit coordgen — on these macropetides CoordGen OPENS the macrocycle
      (atom-shuffled and unshuffled alike) and closes it with one
      runaway straight bond ~8 bond lengths long crossing the drawing
      (experiments/engine_layout_eval/skeleton_coordgen*.png); for
      cleaner inputs it selects the roundish polyomino wheel instead,
      with no public parameter to steer the choice.  Verdict: FAIL
      (either a broken contour or the circular wheel).
  nih cactus image — https://cactus.nci.nih.gov/chemical/structure/
      {url-encoded SMILES}/image returns a real raster (Content-Type
      header misleadingly says image/gif; trust the magic bytes) but the
      layout is a perfect CIRCLE with sunburst backbone spokes and an
      unreadable ring core; image_width/height params are ignored and no
      coordinate output exists, so it cannot feed the RDKit ALIGNED
      render.  Verdict: FAIL, archived.
  pubchem 3d image — PUG-REST /compound/smiles/{smiles}/cids returns
      CID=0 for these novel patent compounds: no database entry, no 3D
      conformer, no image.  Cannot render arbitrary gold SMILES.
      Verdict: UNAVAILABLE.

Consequence: `cdk` is the recommended engine and this module's default —
it is the only external layout engine whose macroring reproduces the
patent saw-tooth polygon.  The incumbent zigzag_layout.py template stays
in the pipeline for depiction-randomized saw-tooth variety (its own QA
rejects some gold molecules, e.g. seq53 across seeds 0-8, where the CDK
path renders fine), with style_calibration.coordgen_coords as the local
CoordGen fallback; `obabel` remains available for ablations and
small-ring inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger

from style_calibration import ALIGNED, draw_molecule

RDLogger.DisableLog("rdApp.*")

ENGINES = ("cdk", "obabel")
ENGINE_ALIASES = {"openbabel": "obabel"}     # historic study name

CDK_BUNDLE_VERSION = "2.13"
CDK_BUNDLE_URL = (f"https://github.com/cdk/cdk/releases/download/"
                  f"cdk-{CDK_BUNDLE_VERSION}/cdk-{CDK_BUNDLE_VERSION}.jar")
CDK_BUNDLE_SHA256 = ("5a42221d412dca53e48fdfc0bfae6453"
                     "10f9de80082800af801d06a87df32411")
CDK_BUNDLE_ENV_VAR = "BIOCHEM_CDK_JAR"
CDK_BUNDLE_CACHE = (Path.home() / ".cache" / "biocheminsight" / "cdk" /
                    f"cdk-{CDK_BUNDLE_VERSION}.jar")
CDK_DRIVER_JAVA = Path(__file__).resolve().parent / "engines" / "CdkLayout.java"
CDK_TIMEOUT_SECONDS = 300
OBABEL_TIMEOUT_SECONDS = 120


class EngineRenderError(Exception):
    """Base class: the engine-coordinate render path failed."""


class EngineUnsupportedError(EngineRenderError):
    """Requested engine is not one of the implemented engines."""


class EngineNotInstalledError(EngineRenderError):
    """The selected engine's runtime (bundle jar / JDK / CLI / bindings)
    is missing from this machine, or the bundle failed verification."""


class EngineInvalidInputError(EngineRenderError):
    """The SMILES input did not parse (in RDKit or in the engine)."""


class EngineLayoutFailedError(EngineRenderError):
    """The engine ran but did not return a usable all-2D layout of the
    input graph."""


# ---------------------------------------------------------------------------
# cdk — StructureDiagramGenerator via the JEP 330 single-file Java driver
# ---------------------------------------------------------------------------

def cdk_bundle_jar() -> Path:
    """Locate (or fetch once, sha256-verified) the CDK bundle jar.

    Resolution order: BIOCHEM_CDK_JAR -> ~/.cache/biocheminsight/cdk ->
    one-time download from the CDK GitHub release.  Maven Central
    cdk-bundle artifacts are stub-sized placeholders and are never used.
    """
    env_jar = os.environ.get(CDK_BUNDLE_ENV_VAR)
    if env_jar:
        jar = Path(env_jar)
        if jar.is_file():
            return jar
        raise EngineNotInstalledError(
            f"{CDK_BUNDLE_ENV_VAR}={env_jar} does not exist")
    if CDK_BUNDLE_CACHE.is_file() and \
            _sha256(CDK_BUNDLE_CACHE) == CDK_BUNDLE_SHA256:
        return CDK_BUNDLE_CACHE
    return _download_cdk_bundle(CDK_BUNDLE_CACHE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_cdk_bundle(cached: Path) -> Path:
    if shutil.which("java") is None:
        raise EngineNotInstalledError(
            "java not on PATH and no CDK bundle jar cached — install a "
            f"JDK 11+ or set {CDK_BUNDLE_ENV_VAR}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(".jar.part")
    try:
        request = urllib.request.Request(CDK_BUNDLE_URL,
                                         headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request,
                                    timeout=CDK_TIMEOUT_SECONDS) as resp:
            partial.write_bytes(resp.read())
    except OSError as exc:
        raise EngineNotInstalledError(
            f"cannot download the CDK bundle from {CDK_BUNDLE_URL}: {exc}; "
            f"set {CDK_BUNDLE_ENV_VAR} to an existing cdk jar") from exc
    if _sha256(partial) != CDK_BUNDLE_SHA256:
        partial.unlink(missing_ok=True)
        raise EngineNotInstalledError(
            "downloaded CDK bundle failed the sha256 check "
            f"(expected {CDK_BUNDLE_SHA256})")
    partial.replace(cached)
    return cached


def cdk_molblock(smiles: str) -> str:
    """SMILES -> CDK StructureDiagramGenerator layout -> V2000 MOL block.

    Runs engines/CdkLayout.java through the JEP 330 single-file source
    launcher (JDK 11+, no build step) against the sha256-pinned bundle
    jar.  This is the engine whose macroring layout reproduces the
    real-patent saw-tooth polygon (see the study record above).
    """
    jar = cdk_bundle_jar()
    if shutil.which("java") is None:
        raise EngineNotInstalledError("java not on PATH — install a JDK 11+")
    if not CDK_DRIVER_JAVA.is_file():
        raise EngineNotInstalledError(
            f"CDK driver source missing: {CDK_DRIVER_JAVA}")
    with tempfile.TemporaryDirectory(prefix="cdk_layout_") as workdir:
        smiles_file = Path(workdir) / "in.smi"
        mol_file = Path(workdir) / "out.mol"
        smiles_file.write_text(smiles.strip() + "\n", encoding="utf-8")
        result = subprocess.run(
            ["java", "-cp", str(jar), str(CDK_DRIVER_JAVA),
             str(smiles_file), str(mol_file)],
            capture_output=True, text=True,
            timeout=CDK_TIMEOUT_SECONDS, check=False)
        if result.returncode != 0 or not mol_file.is_file():
            reason = (result.stderr or result.stdout
                      or f"exit {result.returncode}").strip()
            if "CDK_INVALID_SMILES" in reason:
                raise EngineInvalidInputError(
                    f"cdk rejected the SMILES: {reason}")
            raise EngineLayoutFailedError(f"cdk layout failed: {reason}")
        return mol_file.read_text()


# ---------------------------------------------------------------------------
# obabel — gen2D OBOp (python bindings; verified == obabel CLI --gen2D)
# ---------------------------------------------------------------------------

def obabel_molblock(smiles: str) -> str:
    """SMILES -> Open Babel gen2D layout -> V2000 MOL block text.

    Uses the same code path as `obabel -ismi -omol --gen2D`: the gen2D
    OBOp does its own build internally (a standalone OBBuilder().Build
    may return False yet gen2D still yields a clean all-2D layout —
    verified on the seq53 study molecule; planarity/atom-count checks in
    _load_engine_coords are the actual guards, not the builder return
    code).  Measured circular for macrorings — kept for comparison
    studies and small-ring molecules.
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
    engine = ENGINE_ALIASES.get(engine, engine)
    if engine == "cdk":
        return cdk_molblock(smiles)
    if engine == "obabel":
        return obabel_molblock(smiles)
    raise EngineUnsupportedError(
        f"engine {engine!r} not implemented; available: {ENGINES} "
        f"(aliases: {sorted(ENGINE_ALIASES)})")


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


def render_smiles_via_engine(smiles: str, engine: str = "cdk",
                             seed: int = 0) -> "PIL.Image.Image":
    """SMILES -> engine 2D coords -> ALIGNED-style tight-crop grayscale
    PIL image (the corpus render path; rotation / print-gray / noise as
    in style_calibration.draw_molecule).  Default engine is `cdk`, the
    only engine matching the real-patent macroring shape."""
    mol = _load_engine_coords(smiles, engine)
    return draw_molecule(mol, ALIGNED, random.Random(seed))


def render_file_via_engine(smiles_file: str, engine: str = "cdk",
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
                    "ALIGNED corpus style (engine study: cdk saw-tooth "
                    "matches the real patent crop; openbabel is circular).")
    ap.add_argument("--smiles", help="SMILES string")
    ap.add_argument("--smiles-file", help=".smi file; first field used")
    ap.add_argument("--engine", default="cdk",
                    help=f"layout engine, one of {ENGINES} "
                         f"(alias {sorted(ENGINE_ALIASES)}; default: cdk)")
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

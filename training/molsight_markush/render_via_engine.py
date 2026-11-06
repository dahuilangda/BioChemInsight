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

Render-quality record (visual-reject round): a diff of this module
against experiments/engine_layout_eval/eval_engine_mol.py (the path that
produced the visually matched comparison_sheet.png CDK panel) showed the
two funnel into the SAME style_calibration.draw_molecule(ALIGNED) call —
their seq53 outputs agree pixel-level (identical ink fraction, >93
percent pixels equal), so the sheet quality was already reproduced and
the rejected "sparse / thin-bond / coloured-atom" artifacts came from
ad-hoc RGB debug renders using the RDKit CPK colour defaults, never from
this path.  Two latent defects were fixed anyway: (1) CDK SDG can land
two terminal atoms on one spot (seq53's opposing backbone carbonyl O
atoms overlapped at 0.005 bond units, drawing as a single broken glyph
in the sheet itself) — _declash_terminal_atoms now separates such pairs
without touching ring/backbone coordinates; (2) patent black-white is
now enforced explicitly (apply_patent_bw_palette through
draw_molecule's opts_hook) and verified post-render by
assert_black_and_white, so the CPK red-O / blue-N defaults can never
reach an engine render (tests/test_skills.py pins both checks).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from style_calibration import ALIGNED, draw_molecule

RDLogger.DisableLog("rdApp.*")

# Patent style: pure black ink on white — every atom label, every bond and
# wedge is black regardless of element (the CPK red-O / blue-N defaults are
# a visual reject).  Applied on top of the ALIGNED profile (which already
# carries a black palette) so the guarantee holds even if the profile
# drifts, and enforced post-render by assert_black_and_white.
PATENT_BW_MAX_SATURATION = 8      # RGB channel spread tolerated (anti-alias)
DECLASH_MIN_BOND_RATIO = 0.15     # non-bonded pairs closer than this are
                                  # engine layout defects (glyph-level
                                  # collapse), not accepted tight joints

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


class RenderNotBlackWhiteError(EngineRenderError):
    """The rendered PNG failed the patent black-and-white check."""


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


def _declash_terminal_atoms(mol: Chem.Mol,
                            min_ratio: float = DECLASH_MIN_BOND_RATIO,
                            max_rounds: int = 64) -> Chem.Mol:
    """Push apart engine-collapsed non-bonded atom pairs.

    CDK SDG occasionally lands two terminal groups on the same spot (on
    seq53 the two opposing backbone carbonyl O atoms overlap at 0.005
    bond units and render as one broken glyph — the same defect exists in
    the archived comparison_sheet molblock).  Only degree-1 atoms are
    moved, each re-placed at one standard bond length from its neighbor
    pointing directly away from the atom it collides with, so ring and
    backbone coordinates (the saw-tooth macroring the engine study
    verdict rests on) are untouched.  Pairs at >= min_ratio are left
    alone: the fused-bicyclic tight joints (~0.21-0.25) the layout gates
    accept are not layout defects.
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    bonded = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
    neighbor = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        neighbor.setdefault(i, []).append(j)
        neighbor.setdefault(j, []).append(i)

    def median_bond() -> float:
        lens = []
        for i, j in bonded:
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            lens.append(math.hypot(pj.x - pi.x, pj.y - pi.y))
        return float(np.median(lens)) if lens else 1.0

    for _ in range(max_rounds):
        unit = median_bond() or 1.0
        worst = None            # (ratio, terminal_idx, nbr_idx, other_idx)
        for i in range(n):
            pi = conf.GetAtomPosition(i)
            for j in range(i + 1, n):
                if (i, j) in bonded or (j, i) in bonded:
                    continue
                pj = conf.GetAtomPosition(j)
                ratio = math.hypot(pj.x - pi.x, pj.y - pi.y) / unit
                if ratio >= min_ratio:
                    continue
                if len(neighbor.get(i, [])) == 1:
                    cand = (ratio, i, neighbor[i][0], j)
                elif len(neighbor.get(j, [])) == 1:
                    cand = (ratio, j, neighbor[j][0], i)
                else:
                    continue          # both interior: do not distort
                if worst is None or cand[0] < worst[0]:
                    worst = cand
        if worst is None:
            break
        _, term, nbr, other = worst
        pn = conf.GetAtomPosition(nbr)
        po = conf.GetAtomPosition(other)
        dx, dy = pn.x - po.x, pn.y - po.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            dx, dy, norm = 1.0, 0.0, 1.0
        conf.SetAtomPosition(term, Point3D(pn.x + dx / norm * unit,
                                           pn.y + dy / norm * unit, 0.0))
    return mol


def apply_patent_bw_palette(opts) -> None:
    """Force the patent black-and-white look on a MolDraw2D options
    object: white background, black palette for every element (covers
    atomic numbers -1..120 plus the DUMMY implicit case), black wedge
    bonds.  Belt-and-braces on top of style_calibration.ALIGNED so the
    CPK colour defaults can never leak into an engine render."""
    opts.useBWAtomPalette = True
    opts.setAtomPalette({k: (0.0, 0.0, 0.0) for k in range(-1, 121)})
    opts.singleColourWedgeBonds = True
    opts.backgroundColour = (1.0, 1.0, 1.0)


def assert_black_and_white(image: "PIL.Image.Image",
                           max_saturation: int = PATENT_BW_MAX_SATURATION
                           ) -> None:
    """Assert a rendered PNG carries no coloured pixels (patent style is
    pure black ink on white).  Saturation = per-pixel RGB channel spread;
    grayscale anti-aliasing and the print-gray noise stay far below the
    threshold, while a CPK red-O / blue-N leak saturates at 255."""
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    worst = int(spread.max()) if spread.size else 0
    if worst > max_saturation:
        raise RenderNotBlackWhiteError(
            f"render carries coloured pixels: max RGB channel spread "
            f"{worst} > {max_saturation} (patent style must be pure "
            "black ink on white)")


def render_smiles_via_engine(smiles: str, engine: str = "cdk",
                             seed: int = 0,
                             declash: bool = True) -> "PIL.Image.Image":
    """SMILES -> engine 2D coords -> ALIGNED-style tight-crop grayscale
    PIL image (the corpus render path; rotation / print-gray / noise as
    in style_calibration.draw_molecule, patent black-white palette
    enforced).  Default engine is `cdk`, the only engine matching the
    real-patent macroring shape.  `declash` separates engine-collapsed
    terminal atom pairs before drawing (see _declash_terminal_atoms)."""
    mol = _load_engine_coords(smiles, engine)
    if declash:
        mol = _declash_terminal_atoms(mol)
    img = draw_molecule(mol, ALIGNED, random.Random(seed),
                        opts_hook=apply_patent_bw_palette)
    assert_black_and_white(img)
    return img


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
        mol = _declash_terminal_atoms(mol)
        img = draw_molecule(mol, ALIGNED, random.Random(args.seed),
                            opts_hook=apply_patent_bw_palette)
        assert_black_and_white(img)
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

# Cyclopeptide2D

Two-dimensional structure generation for cyclic peptides, using joint optimization of residue positions and orientations.

Requires Python 3.10+, RDKit, NumPy, and SciPy. Version 0.1.0 is experimental. The package runs independently of the surrounding repository and does not require model weights or network access.

## Installation

```bash
python -m pip install ./packages/cyclopeptide2d
```

For development, run from this directory:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Python API

```python
from pathlib import Path

from rdkit import Chem

from cyclopeptide2d import depict, render_svg

smiles = "N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O"
result = depict(smiles, seed=0, maxiter=350)

if result.accepted:
    Path("structure.svg").write_text(render_svg(result), encoding="utf-8")
    Chem.MolToMolFile(result.molecule, "structure.mol")
else:
    print(result.report)
```

`depict` returns a `LayoutResult` containing a candidate molecule and a validation report. `accepted` is true only when the candidate passes validation. Invalid SMILES, radicals, wildcard atoms, and disconnected components raise `ValueError`. Unsupported ring structures and unsuccessful layouts return a rejected result with an error or measurements in the report.

`render_svg` requires an accepted result and uses its existing coordinates. Drawing options include `width` and `height`, both defaulting to 1000 pixels. Layout coordinates use a bond length of 30 units. Specified stereochemistry is preserved; unspecified stereochemistry remains unspecified.

### Sequence input

Install the local `peptidecore` package and the `sequence` extra to use `depict_sequence` or `depict_record`:

```bash
python -m pip install ./packages/peptidecore './packages/cyclopeptide2d[sequence]'
```

```python
from cyclopeptide2d import depict_sequence

result = depict_sequence("PEPTIDE1{G.G.G.G.G.G.G.G}$PEPTIDE1,PEPTIDE1,8:R2-1:R1$$$V2.0")
```

The assembled record is included in the report, with monomer ownership and atom indices retained. This interface uses the current cyclic peptide solver; it does not add linear, branched, or general multicycle layout support. See [PeptideCore](../peptidecore/README.md) for supported notation features.

## Command line

```bash
cyclopeptide2d \
  --smiles 'N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O' \
  --seed 0 --maxiter 350 --out ./result
```

`python -m cyclopeptide2d` provides the same interface.

| Output | Contents |
| --- | --- |
| `report.json` | Validation measurements and solver status |
| `structure.mol` | Molecule with two-dimensional coordinates |
| `structure.svg` | Monochrome structure drawing |

Exit code 0 indicates acceptance. Exit code 2 indicates invalid input, unsupported structure, or failed validation. Rejected layouts produce a report only. Use a separate output directory for each run; files from previous runs are not removed.

## Method

The molecular graph is cut at peptide connections to form complete residues. Each residue receives local coordinates and a pose with two translation variables and one rotation variable. Small rings remain part of their residue.

Initialization uses an asymmetric polygon for fewer than 16 residues. Larger rings use a search over connection directions and complete-residue reflections. Candidates are selected by score and differences in reflection signs and positions.

L-BFGS relaxation and SLSQP optimize all residue poses together. Peptide bond lengths are equality constraints. Interface angles and nonbonded separation provide additional constraints. Local rotations and residue reflections supply alternative arrangements when continuous optimization leaves intersections or close contacts.

Subtree rotations preserve bond lengths and connection coordinates. Each changed junction angle must remain within 12 degrees of its original value. Repeated rotations use the same original reference.

Intersections that survive continuous optimization are repaired discretely: for every crossing, one bond is assigned to each side of the other bond's line, and the side assignment is enforced as a constrained inequality with analytic derivatives. A beam over side assignments keeps the first assignment that closes, holds interface angles, and improves the defect rank.

If contact repair stops improving the layout, the solver tests a limited number of complete-residue reflections. An unchanged result ends the repair loop.

Exact spatial queries reduce the contact rows evaluated by the objective and candidate ranking. For atom–bond contacts, a bond of length L is queried within L/2 plus the maximum clearance from its midpoint. Actual bond lengths make this bound valid during initial relaxation. The spatial index is rebuilt for each evaluation.

Final validation independently checks every bond length, connection angle, nonbonded contact, bond crossing, small-ring intersection, and stereochemical MOL round trip. Solver convergence alone does not establish acceptance.

## Supported structures

The solver requires a cyclic chain of complete residues. Reviewed examples include head-to-tail peptides and one ClAc–Cys thioether closure. Single-bond crosslinks between residues (disulfides, thioethers, lactam bridges) are supported: each bridge becomes its own closure equality, and the macroring is located by walking the amide chain rather than by ring perception, which crosslinks confuse. Adjacent residues inside small rings can still fail when their depicted sidechain directions leave the bridge unclosable; junction-angle gates for bridges remain future work.

The output is a chemical diagram, not a three-dimensional conformation. Side chains may face inward. Ring-center clearance is reported for inspection and does not affect acceptance.

Geometry validation does not check text-label readability or establish that an input matches a patent structure. The reviewed input structures are listed in `examples/reviewed_references.json`.

The three reviewed structures pass validation at seed 0. A diagnostic set of eight synthetic peptides, containing 14–28 residues, currently has five accepted layouts (14, 18, 20, 26, and 28 residues) within a 100-second budget per input. The remaining three retain one crossing or a few close contacts, each reported with an exact defect rank. This set is too small to establish general coverage. A segment-clearance penalty for the continuous optimizer exists behind `PoseProblem(..., segment_clearance=...)` but stays off by default: across several margins it distorted legal dense packing more often than it removed crossings.

## Performance

The package uses the CPU. A single-thread comparison against the preceding implementation measured 4.0–4.7 times shorter layout times on the three reviewed structures, with all three accepted. These measurements are limited to one seed and one run per structure.

Run the numerical and timing comparison from this directory:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python examples/profile_contacts.py \
  --layout --budget 60 --out /tmp/contact_profile.json
```

The script compares spatial queries with exhaustive contact evaluation in the current solver. It records dependency versions, objective timings, numerical differences, and optional layout results. Topology tables still require quadratic memory; dense overlaps and long unclosed bonds reduce the benefit of spatial queries.

The internal `residue_pose_layout` function accepts an absolute `time.monotonic()` deadline. Stages share the remaining budget and report skipped work. This is a scheduling limit rather than a hard timeout: an optimization call can finish after the deadline.

A GPU implementation would require a batched solver with coordinates, derivatives, and iteration state kept on the device. Moving distance calculations alone to the GPU would retain CPU optimization and add synchronization. GPU performance has not been measured. See the [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) and [CUDA performance guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## Development

| Module | Responsibility |
| --- | --- |
| `api.py` | Input validation, depiction, and SVG output |
| `cli.py` | Command-line interface |
| `chemistry.py` | Graph traversal and stereochemistry checks |
| `frames.py` | Residue coordinates and connection geometry |
| `geometry.py` | Geometric measurements and intersection checks |
| `kinematics.py` | Initial pose search |
| `solver.py` | Continuous optimization and candidate refinement |

Tests cover residue partitioning, rigid transformations, analytic derivatives, spatial queries against exhaustive evaluation, junction-angle limits, rendering, and failure reports. Tests and examples are included in the source distribution; generated reports and build artifacts are excluded.

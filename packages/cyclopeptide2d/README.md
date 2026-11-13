# Cyclopeptide2D

Two-dimensional structure generation for peptides, using explicit monomer connections and joint optimization of residue positions and orientations.

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

The assembled record retains monomer ownership, attachment atoms, and source atom indices. The solver uses these connections directly for linear, cyclic, branched, and crosslinked peptides. Each chain keeps its own backbone path. Caps remain attached to their residue. See [PeptideCore](../peptidecore/README.md) for supported notation features.

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

Local rotations preserve internal bond lengths. When a complete residue is too rigid for a closure, joint atom refinement preserves intrinsic ring geometry and adjusts the remaining connections with analytic derivatives. Rigid ring coordinates reduce the number of optimization variables. Shared-ring paths receive an explicit planar initialization that places the long path outside the short auxiliary ring. Local repair expands from implicated residues to neighboring residues before using a complete graph refinement. A bounded active set enforces residual contacts as inequalities. A bond can remain flexible inside its residue even when an inter-residue crosslink places it in a larger molecular ring. Moves that change an attachment position restore all connection lengths and interface angles jointly. A minimum-norm step with analytic derivatives provides a fast feasible candidate; candidates that do not improve the layout use the full constrained optimizer. Each changed local junction angle stays within 12 degrees of its original reference.

Intersections that survive continuous optimization are repaired discretely: for every crossing, one bond is assigned to each side of the other bond's line, and the side assignment is enforced as a constrained inequality with analytic derivatives. A beam over side assignments keeps the first assignment that closes, holds interface angles, and improves the defect rank.

If contact repair stops improving the layout, the solver tests a limited number of complete-residue reflections. An unchanged result ends the repair loop.

Exact spatial queries reduce the contact rows evaluated by the objective and candidate ranking. For atom–bond contacts, a bond of length L is queried within L/2 plus the maximum clearance from its midpoint. Actual bond lengths make this bound valid during initial relaxation. The spatial index is rebuilt for each evaluation.

Final validation independently checks every bond length, connection angle, nonbonded contact, bond crossing, small-ring intersection, and stereochemical MOL round trip. Solver convergence alone does not establish acceptance.

## Supported structures

Sequence-derived records support linear chains, head-to-tail cycles, disulfides, lactam closures, branches, and multiple closures through registered attachment atoms. Raw SMILES input recovers an amide backbone before segmentation; explicit records provide more reliable boundaries for modified residues and branches.

Coverage remains incomplete. Short constrained loops, crowded long chains, and some branch arrangements can fail. Multiple connections on the same attachment atom, single-atom connectors, and disconnected component packing require further work. Failed structures return their measurements and remain in evaluation totals.

The output is a chemical diagram, not a three-dimensional conformation. Side chains may face inward. Ring-center clearance is reported for inspection and does not affect acceptance.

Geometry validation does not check text-label readability or establish that an input matches a patent structure. The reviewed input structures are listed in `examples/reviewed_references.json`.

The fixed review matrix accepts 96 of 96 structures within a 60-second budget per input. It includes eight inputs in each of twelve groups: short, medium, long, and modified linear peptides; small, medium, large, and modified head-to-tail rings; disulfides; lactams/lariats; branches; and multiple closures.

A second synthetic matrix, generated with a new seed after implementation, accepts 24 of 24 structures and has no exact structure overlap with the first. The earlier eight-case diagnostic with 14–28 residues now accepts all eight within a 100-second budget. The three reviewed patent references also pass. These are bounded evaluations, not a guarantee for arbitrary chemistry. The second matrix uses the same generator and is not an independent real-image benchmark.

The former eleven matrix failures are retained as regression fixtures. Flexible refinement checks all local angles and every pair distance inside intrinsic ring templates. Final validation repeats these checks, alongside bond lengths, contacts, crossings, ring intrusions, molecular identity, and stereo reconstruction. Reported angles are measured directly without ring-dependent clamping.

Local experiment outputs are under `output/peptide_flexible_review_v2` and `output/peptide_flexible_holdout_v1`, including HTML reviews, manifests, source snapshots, input identities, and image hashes. Across the 120 matrix structures and two drawing styles, 15 of 240 variants fail the preliminary resolution screen at 384 pixels, three fail at 512, and none fail at 768. These images remain review data; geometry acceptance and resolution screening do not establish training readiness or recognition accuracy.

## Performance

The package uses the CPU. Earlier single-thread contact-search measurements found 4.0–4.7 times shorter layout times on the three reviewed structures. Those results precede the current general-graph changes and are not current end-to-end speedup claims. Beam segment costs evaluate each segment pair once and include explicit crossing costs. Runtime comparisons report acceptance and include initialization, discrete search, refinement, and validation.

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

The registered branch benchmark accepts all eight inputs. Seven core layout calls took 0.14–0.27 seconds and one took 2.53 seconds in a local single-thread diagnostic. A separate five-perturbation connection-restoration benchmark measured median times of 1.09 ms for minimum-norm retraction and 51.35 ms for the full solver, with both satisfying closure and angular tolerances. The full solver also optimizes layout energy; these timings compare feasibility restoration and do not imply the same speedup for complete depiction.

A GPU backend is not implemented. The current solver uses small, irregular constrained systems and discrete candidate decisions. A useful GPU implementation needs batched solves and persistent device data, followed by acceptance and end-to-end latency comparisons with the CPU implementation. Porting only distance calculations would leave optimizer synchronization and candidate search on the CPU.

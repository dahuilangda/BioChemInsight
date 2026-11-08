# Peptide image review

Build reproducible peptide chemistry, render candidate figures, and inspect the actual model inputs before creating a training release. The [plan](PLAN.md) defines the coverage matrix and release requirements.

`design.py` constructs 96 distinct structures across twelve strata using pinned HELM monomer definitions. HELM and BILN reassembly must reproduce each complete atomic structure. Records include source notation, atom ownership, connection endpoints, registry identity, seeds, and synthetic provenance. All records initially belong to the review set.

`review.py` provides three explicit layout modes:

| Mode | Method |
| --- | --- |
| `residue` | RDKit for linear peptides; Cyclopeptide2D for head-to-tail cycles; unsupported topologies reported |
| `rdkit` | RDKit depiction for every topology, assessed independently |
| `coordgen` | CoordGen depiction for every topology, assessed independently |

There is no automatic fallback between modes. Each attempt records its method, rejection reason, geometry checks, stereo serialization check, and timings. Accepted geometry is drawn in two clean styles with the same vector/raster coordinates. Files include MOL, SVG, native PNG, font hashes, and per-atom image coordinates. Source files are archived with checksums.

Model input previews use the existing MolSight crop/resize/pad transforms, with rotation, noise, blur and tensor normalization disabled. Resolution screens flag nominal text below 8 pixels, median bonds below 10 pixels, or nominal strokes below 0.7 pixels. These are preliminary screens. They do not prove glyph clearance, stereo visibility, realistic style, or suitability for training. No generated image is automatically approved.

## Commands

Run from the workspace root with PeptideCore, Cyclopeptide2D, and the MolSight runtime available on `PYTHONPATH`. The existing MolSight environment contains the rendering and preprocessing dependencies.

```bash
PYTHONPATH=packages/peptidecore/src:. python -m training.peptide_data.design \
  --out output/peptide_review/design.json

NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=packages/peptidecore/src:packages/cyclopeptide2d/src:training/molsight_runtime:. \
/data/molsight_venv/bin/python -m training.peptide_data.review \
  --design output/peptide_review/design.json --out output/peptide_review/residue

python -m training.peptide_data.report \
  --design output/peptide_review/design.json \
  --manifest output/peptide_review/residue/manifest.json \
  --out output/peptide_review/review.html
```

Design and render commands refuse to overwrite existing output. `--strata` selects explicit design strata; `--layout-engine` selects an independent method. The report verifies image hashes, retains rejected cases, and links final-resolution previews.

## Current review

The artifacts in `output/peptide_review_v1` contain the 96-record design, a complete registry snapshot and license, the first diagnostic run, the revised residue-mode run, and a CoordGen comparison on 48 difficult records.

| Attempt | Rendered structures | Geometry rejected | Layout rejected | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Revised residue mode, 96 records | 47 | 7 | 10 | 32 |
| CoordGen comparison, 48 records | 29 | 19 | 0 | 0 |

The two recorded methods produced review images for 73 distinct structures. There are 152 style/layout representations and 456 resolution previews. Seventeen CoordGen style/layout representations fail the 384-pixel screen, eleven fail at 512, and three still fail at 768. Fully expanded long peptides are a principal resolution problem. The residue solver still fails all eight small head-to-tail examples in this design; those failures remain part of its coverage report even when a separate engine produces a review drawing.

These figures describe this synthetic review set. They do not establish independent chemical coverage or model accuracy. Further work includes general layout support, small-cycle constraints, readable long-peptide layouts, glyph and wedge inspection, patent-style calibration, family-separated evaluation, and full trainer validation.

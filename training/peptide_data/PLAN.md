# Peptide recognition data

## Scope

MolSight training combines complete molecules, Markush structures, attachment fragments, and peptides. Peptide records retain their resolved atomic structure, HELM/BILN notation, pinned monomer definitions, atom ownership, and explicit connection endpoints. The notation contract is defined in [PeptideCore](../../packages/peptidecore/NOTATION.md).

The chemistry matrix covers twelve strata: short, medium, long, and modified linear peptides; small, medium, large, and modified head-to-tail rings; disulfides; lactams and lariats; branched chains; and multiple closures. Modified records include D residues, N-methylation, and registered terminal groups.

## Current matrix

The frozen pilot contains 2,004 distinct structures, with 167 per stratum. Full generation and subsequent validation of all 148 affected long-chain records cover the complete matrix. The current render manifest contains 2,004 chemically and geometrically accepted records.

`output/peptide_training_pilot_v1/redraw_v7` contains two font styles at 512, 768, and 1024 pixels: 12,024 model input images. The independent vector audit checks emitted glyph ink, bond and atom-label coverage, visible wedge geometry, stereo reconstruction, image hashes, and the final coordinate transform. Each structure has at least one input passing its resolution and drawing checks. Rejected image variants remain recorded with their reasons.

These results establish coverage of this matrix. The selected catalog contains 4,003 style variants, each at its minimum passing resolution. Grouped admission assigns 3,173 images to training, 464 to validation, and 366 to test; peptide compositions do not cross splits. Recorded visual inspection covers four selected final inputs and complements the full automated audit. It is not exhaustive human review. Broader non-natural monomer coverage and measured acquisition variants remain extensions beyond this pilot.

## Generation

`design.py` constructs resolved chemical graphs and checks HELM/BILN round trips. Parallel design derives an independent seed for each stratum, so worker count does not change its output. Serial `build_design` uses a shared random stream and has a separate reproducibility contract.

`review.py` generates coordinates, validates geometry and stereo, and draws SVG/PNG pairs with matching atom transforms. Native canvas spacing adapts to measured glyph interference. Font files, drawing settings, coordinates, final inputs, and hashes accompany every record.

`audit_render.py` independently reads frozen render manifests and checks the emitted drawings at each model input size. `redraw.py` regenerates images from verified molecular drawings without repeating layout searches. `artifacts.py` verifies manifest coverage and explicit replacement order.

Run frozen modules from their snapshot directory with absolute input paths. Rendering and auditing share the host CPU allocation; native math libraries use one thread per worker.

## Image quality

Atomic drawings use the complete resolved structure. Condensed residue labels require a separate expansion contract before they can become atomic recognition targets. Markush variables and attachment identifiers retain their visible meaning.

Image admission combines chemical identity, geometric validity, visible stereo reconstruction, glyph clearance, crop integrity, and final-resolution legibility. A source vector image and a readable native raster do not establish that a smaller model input is readable.

Style calibration compares source-verified patent crops with synthetic images using glyph/bond ratios, stroke widths, spacing, ink coverage, and scale. Scan variants derive from measured acquisition properties and retain their clean parent. Geometry, visible stereo, and attachment information remain recoverable after preprocessing.

## Splits and release

All images, styles, scan variants, and notation traversals of a structure share one split. Exact atomic identity groups cyclic rotations and reversed traversal. Composition, modification, closure topology, and available publication identifiers provide additional family guards. Official evaluation records and their matching families are excluded from training.

A release records its source manifests, code and definition hashes, split decisions, duplicate/conflict audit, rejected cases, selected input resolutions, and review results. Coordinate supervision additionally requires a verified mapping between drawing atoms and the target SMILES traversal.

New designs group compositions by canonical monomer-template identity, independent of stratum, sequence order, and chain partition. The stricter audit of the first release found 2,000 composition groups and zero cross-split groups. The audit is recorded under `/home/dahuilangda/molsight_work/peptide_composition_audit_v2`.

## Model validation

The deployed R3 checkpoint has a twelve-layer decoder and a frozen 64-by-64 Fourier position grid. High-resolution inputs retain the original grid and evaluate its recovered frequency basis at the new feature coordinates. Native-resolution positions remain unchanged.

The production BF16 preflight covers real image/label pairs and long synthetic peptides at 1024 pixels. The two-GPU check compares distributed accumulation with the same global supervised batch and verifies unchanged parameters after a zero-learning-rate optimizer step. Reports are under `output/molsight_training_audit`.

The image-only diagnostic baseline reveals a substantial loss when existing 512-input models are fed uniformly enlarged 1024 inputs. Initial training therefore requires native-resolution replay, resolution buckets, and separate high-resolution evaluation. A shape-compatible forward pass is insufficient.

Training uses a protected complete-molecule retention benchmark and separate Markush, attachment, and peptide reports. Supervised comparisons precede reinforcement learning. Reinforcement learning starts from the supervised checkpoint with a fixed reference and is compared with supervised continuation at matched compute. The first merged release has passed admission. Broader retention evaluation, exact optimizer/RNG restart verification, and production policy-update validation remain outstanding.
## Coordinate-bearing patent corpus

`engine_corpus.py` assembles each HELM record before layout and checks its chemical identity against the design. Output directories are immutable. Records are written as workers finish; accepted layouts retain MolBlock coordinates, SVG coordinates, artifact hashes, and the full engine report. Generation records the package sources and checks that they remained unchanged.

`render_corpus.py` verifies the corpus record and artifact hashes, then produces serif and sans-serif drawings at each requested model input size. It uses the existing glyph, bond, clipping, and visible-stereo checks. Per-image quality and resolution results remain explicit. Neither generation nor rendering grants training admission.

The original 16 rejected patent-profile layouts were rerun from their saved HELM records using the current registered-graph path. All 16 passed geometry and an independent 1024-pixel SVG audit. This is a regression-cohort result, not a guarantee for unseen structures or an assessment of patent recognition.

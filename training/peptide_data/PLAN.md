# Peptide image data and training

## Objective

Train MolSight to read atomic structures in peptide and cyclic-peptide figures while retaining complete-molecule, Markush, and attachment-bearing fragment recognition. Chemical identity, monomer interpretation, and visual appearance have separate verification requirements.

## Initial review set

Build 96 distinct chemical records, with eight records in each row below. Render two independently specified styles per accepted record, giving at most 192 review images. These are review allocations, not a production mixture or an accuracy benchmark.

| Stratum | Chemistry | Layout requirement |
| --- | --- | --- |
| Linear, short | 2–5 residues, ordinary termini | Explicit termini and non-overlapping bonds |
| Linear, medium | 6–12 residues, diverse side chains | Readable final-resolution labels |
| Linear, long | 13–30 residues | Demonstrate sufficient resolution or reject |
| Linear, modified | D residues, N-methylation, registered caps | Exact registered definitions and stereo |
| Head-to-tail, small | 3–5 residues | Correct closure and ring strain-independent 2D geometry |
| Head-to-tail, medium | 6–10 residues | Residue geometry and side-chain clearance |
| Head-to-tail, large | 11–20 residues | Closure, crossings, and training-resolution readability |
| Head-to-tail, modified | Registered noncanonical residues | Retained substituent and stereo relationships |
| Disulfide | Explicit cysteine side-chain connections | Correct sulfur endpoints |
| Lactam/lariat | Side-chain or side-chain-to-terminus closure | Correct attachment occupancy and endpoints |
| Branched | Explicit multi-chain peptide graph | Complete ownership and all junctions |
| Multiple closures | Bicyclic and multiply crosslinked records | Every closure represented and independently checked |

Construct source-qualified chemistry through PeptideCore and a pinned monomer registry. Store HELM/BILN, complete isomeric SMILES, atom ownership, and connection endpoints. Do not infer structures from mass shifts or ambiguous residue names. Registry extensions require explicit structures and provenance.

The current cyclic layout solver does not support this entire matrix. Unsupported topologies remain visible in the coverage report. They require solver development before their images can enter a release. A general depiction engine may be evaluated as a separate named method, with the same acceptance checks; it must not silently replace failed layouts.

## Image contracts

Start with fully expanded atomic drawings. Separately develop condensed residue and mixed representations after a typed label/expansion contract is available. A residue label may be expanded to an atomic target only when the visible label and supplied dictionary determine the structure. Preserve ambiguous variables and attachment labels explicitly.

For each image, retain:

- Source record and definition hashes; canonical atomic identity; original notation.
- Layout method, version, seed, atom coordinates, and validation report.
- Drawing settings, font file/hash, text labels, stereo marks, and atom/bond-to-image transforms.
- Clean vector output, native raster, final model input, image hashes, and augmentation history.
- Category, source-family and structure-family groups, split assignment, and review decisions.

Use chemistry-aware rendering for bonds, glyphs, wedges, and connection points. Do not use generative image models to draw chemical truth. Realism comes from measured drawing and acquisition properties with traceable labels.

## Acceptance

1. Reassemble every chemical record and verify complete graph identity, stereo, charge, isotope, components, and connection endpoints. An ambiguous reverse decomposition is not a unique sequence target.
2. Validate layout coordinates and bond geometry. Reject intersections, collapsed bonds, side-chain collisions, or unrepresented closures. Record all attempts and failures; never lower a threshold to increase yield.
3. Verify stereo marks in the final drawing against the source graph. Rotations, reflections, bond rearrangements, and atom renumbering must preserve the encoded configuration.
4. Validate glyph clearance, bond endpoints, crop margins, and final-resolution legibility. Inspect the actual preprocessing result, initially at the current 384-pixel input size and at proposed larger sizes. Native high-resolution images alone do not pass this requirement.
5. Compare styles with source-verified patent crops outside the final test families. Measure font/bond ratios, stroke widths, label spacing, ring shapes, ink and scale distributions. Review paired crops and diverse failure cases visually.
6. Publish an explicit review decision. Parser success and model agreement are insufficient to approve an image. Keep rejected, unsupported, and uncertain cases outside supervised labels and report their rates by stratum.

Maintain clean-vector, clean-raster, and degraded versions separately. Apply scan effects only after a clean image passes. Recheck the final degraded model input. Use mild, measured effects; exclude cases whose stereo or attachment information becomes unreadable. A scan should not invent or erase chemical information.

## Splits

Assign chemistry and source families before rendering or augmentation. All rotations, styles, atom orderings, SMILES enumerations, and scan variants of a structure stay in one split. Group cyclic rotations and reversed traversal by atomic identity. Also group close peptide families by residue composition, modification pattern, and closure topology.

Create separate diagnostic sets for unseen combinations of known monomers, held-out monomer families, ring sizes, closure types, and long sequences. A random image split does not test these capabilities. Reject exact structure and image duplicates across splits; inspect near-duplicate structures and perceptual image matches.

Use held-out real patent families for final evaluation. Figures already used to calibrate typography or layout are development data. Store original crops, source locations, manual label verification, and uncertainty records. Keep the existing complete-molecule baseline set frozen.

## Scaling

Advance from the 96-record review matrix to a 2,000-record chemistry pilot only after the supported strata pass review. This is an initial capacity target, not an automatic release size. Measure generation time, acceptance by topology and length, sequence lengths, vocabulary coverage, and review rejection reasons before expanding.

Choose the training mixture from real-domain performance and capability-retention measurements. Do not count multiple styles of one structure as independent chemistry coverage. Maintain ordinary molecules and existing supported outputs as replay data with verified provenance.

## Two-GPU execution

The host has two RTX 4070 Ti SUPER cards with 16 GB each. The two workspace Compose projects were stopped without removing volumes. A standalone MolSight service still occupies approximately 1.3 GB on GPU 1; leave it intact unless its memory is needed and stopping it is authorized.

Use one training process per GPU with DistributedDataParallel. Sixteen gigabytes per card remains the memory limit for a replicated model; two cards do not become a single 32 GB allocation. Measure usable per-rank batch size and image/sequence length before choosing accumulation. Report samples per second and optimizer updates per second at matched effective batch and precision; do not assume a twofold speedup.

Image synthesis primarily uses CPU chemistry, layout, and rendering. Profile those stages before adding GPU kernels. Generate and cache accepted images independently of GPU training so rendering cannot stall the trainer. Reserve GPU work for model validation, training, and explicitly measured tensor workloads.

First validate distributed gradients, effective-batch normalization, reference-policy behavior, BF16, checkpointing, and resumability. Then run supervised comparisons against the unchanged baseline and replay-only control. Start reinforcement learning only after the dataset, reward contract, and supervised evaluation pass their checks. Compare reinforcement learning with supervised continuation at matched compute.

## Release evidence

A release needs its chemical and image manifests, source hashes, split audit, rejected-case report, final-resolution review, independent real-image results, unchanged-baseline comparison, and capability-retention results. Decoder regression tests and synthetic recognition scores do not substitute for these artifacts.

# MolSight data preparation

Source-preserving preparation of complete molecules, named Markush structures, attachment fragments, and peptide drawings.

## Label contract

`molsight.structure` validates SMILES, indexed E-SMILES, and CXSMILES. Exact comparison retains stereochemistry, isotope and atom-map identifiers, variable names, polymer groups, and position-variation candidate sets. Display coordinates are excluded from recognition targets. Symbolic labels participate in stereochemical identity.

MarkushGrapher sources use `cxsmiles_opt`. Its named variables and position-variation declarations are converted to CXSMILES. `cxsmiles` and numeric-isotope conversions can omit this information and are not interchangeable training labels.

## Preparation

`prepare.py` reads explicit source manifests, checks labels and tokenization, verifies image decoding, and stores embedded images by content hash. CPU workers use bounded queues. Each record retains its source file, row, original label, normalized target, image hashes, dimensions, and rejection reasons.

`normalize.py` revalidates these records against a frozen label implementation. It reuses verified image artifacts and recovers images for newly supported records. Its report records code hashes and verifies that those sources remained unchanged during execution.

Complete-molecule labels with extreme atomic charges, extreme radical counts, or highly fragmented annotations are quarantined by `chemical_quality.py`. These conservative flags request source review; they do not repair labels or reject ordinary salts and radicals. Existing admitted releases are immutable and predate these checks.

`index.py` combines frozen candidate files and explicit corrections to previously rejected records. It keeps drawing variants distinct, groups identical pixel images, and compares conflicting targets by typed chemical identity. Source publication identifiers are added where available. The index uses a pinned SQLite driver independently of the host Python build. Conflicts and parser rejects remain separate from release selections.

`resolve.py` computes a separate chemical interpretation using the pinned abbreviation registry. Original targets remain in the source index. The current run validates 748,927 distinct targets and expands 13,194; restricted or ambiguous contexts remain explicit.

`split.py` joins images, resolved structure families, known publications, and peptide compositions into connected components. Source tests take priority over validation and training. Every font variant and closure traversal follows its component. An independent SQL audit checks all four relations for cross-split overlap. Identical-image deduplication retains a deterministic representative only when the resolved labels agree.

`export.py` writes memory-mapped Arrow splits with cached character tokens, input resolution, image hashes, source record identifiers, and component assignments. `audit_export.py` rereads the tokens and cached images, checks byte and decoded-pixel hashes, and rejects blank or resized verified inputs. Export alone does not approve admission.

Run frozen modules with their snapshot as the working directory and absolute input paths. This prevents the workspace's `training` namespace from overriding the snapshot.

## CPU execution

Preparation and normalization default to the CPU count available to the process. Use `--workers` to share that capacity with concurrent rendering jobs. Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before starting Python to prevent nested math-library thread pools. Queues hold at most two batches per worker.

On a 12-CPU host, run one bulk preparation job with 12 workers, or divide the workers between simultaneous jobs. During two-GPU training, the default loader allocation is five workers per rank, reserving two CPUs for the training processes. Prepare chemical structures and render images before training so those jobs do not compete with batch loading.

## Training release

Preparation produces candidates. Admission to training also requires:

- Conflicting-image annotation checks and image/structure deduplication.
- Protected evaluation splits and source/publication overlap checks.
- Final-input crop, glyph, bond, stereo, and attachment visibility checks.
- Recorded drawing settings and source-qualified chemistry for synthetic images.
- Stratified review of real and synthetic images.

Rejected records remain in the manifests. Training and evaluation selections are separate artifacts; a parser-success count is not a recognition score.

Admission verifies every peptide catalog record against its indexed payload, in addition to catalog hashes, export image identity, and resolved supervision. Catalog counts alone do not establish provenance.

The combined index contains 815,071 records, including 4,003 peptide images. The grouped selection contains 691,863 training images, 15,563 validation images, and 81,711 test images. It excludes 1,203 records with conflicting image labels and 21,768 equivalent duplicates. All four audited relations have zero cross-split overlap. Publication metadata is available for a subset of sources; unrecorded provenance remains unknown.

Artifacts are under `output/molsight_merge_v1`, `molsight_resolved_v1`, `molsight_split_v1`, and `molsight_release_v1`. New large artifacts use `/home/dahuilangda/molsight_work`, with links from the project output directory. Source downloads and checkpoints remain unchanged.

The quality-reviewed preparation under `output/molsight_patent_merge_v2/admitted_quality_v3` contains 689,938 training, 15,691 validation, and 81,745 test images. All 787,374 images passed the recorded image/token audit. The release includes 6,002 synthetic peptide images representing 3,004 chemical records. It quarantines 3,688 distinct suspect complete-molecule annotations and protects prior evaluation assignments. All four split relations have zero overlap. These automated checks do not establish exhaustive source-label correctness or independent patent generalization.

Short supervised runs have tested updates and checkpoint restart behavior. Main capability training and reinforcement learning remain pending. Image-only baselines measure recognition separately from execution checks.

## Patent recognition

`peptide_tags.py` discovers connected peptide-backbone candidates in validated labels, including alpha, beta, and gamma backbones. These tags support sampling and evaluation; they do not establish a unique sequence or verify the source annotation. The frozen scan contains 13,297 candidates, including 3,758 cyclic candidates, across 748,927 distinct labels.

`sampling_tags.py` recomputes the backbone features against resolved targets and binds candidate indices to the admitted release and Arrow hashes. The resulting file covers 9,321 training, 552 validation, and 5,653 test records. It changes sampling metadata, not labels, images, or split assignments.

`patent_benchmark.py` binds reviewed figures to crop and source-PDF hashes. It records known historical figure and publication overlap. A reviewed reference from a previously trained publication is a diagnostic, not evidence of independent patent generalization.

`patent_errors.py` separates unparsed predictions, connectivity or annotation differences, and stereochemistry differences. Removing stereo for this diagnostic retains attachment identifiers and symbolic labels. Backbone counts are reported separately and never substitute for full chemical identity. Invalid references stop analysis rather than becoming model errors.

## Frozen capability panel

`freeze_capability_panel.py` freezes a test panel before candidate inference. It checks admitted Arrow hashes, verifies test component separation from train and validation, selects one image per component with declared category/peptide-stratum budgets and source balancing, and verifies selected image hashes. Sampling tags affect grouping without changing the original recognition category. It writes separate pair files for each admitted resolution, because the current recognition benchmark takes a fixed input size per invocation.

```bash
PYTHONPATH=training python -m molsight_data.freeze_capability_panel \
  --manifest /path/to/admitted/manifest.json \
  --tags /path/to/sampling.json --out /path/to/new-panel
```

This establishes separation within the adaptation release. Historical pretraining overlap and source annotation correctness require separate evidence. Keep panel predictions out of training, sampling and checkpoint selection. The first panel has 882 unique components, including 384 complete molecules and 32 large patent-profile cyclic peptides; one component was subsequently exposed during a CPU worker smoke check. The frozen panel remains unchanged; use its `development_exposures.json` registry to report all 882 and the 881 unexposed components separately. Its report is under `/home/dahuilangda/molsight_work/molsight_capability_panel_v1/report.json`.

`training/molsight_runtime/compare_capability_panel.py` compares before/after benchmark directories containing one subdirectory per resolution. It rejects changed pair files, incomplete or misidentified predictions, different decoding limits, non-FP32 runs and repeated components. It reports literal and resolved exact paired gains/losses, confidence intervals, parse and termination counts by peptide stratum, resolution and source. Group intervals are exploratory and do not provide an automatic deployment gate.

## Chemical graph supervision alignment

`graph_supervision.align_smiles` maps unchanged literal SMILES atom tokens to RDKit lexical atom indices and exports complete chemical bonds, charges, isotopes and stereo references. It checks the tokenizer round trip and SOS-relative query offsets, then reconstructs a molecule only from the exported graph and verifies canonical isomeric identity. Explicit hydrogens are retained. The mapping does not export drawing coordinates or invent rendered wedge directions from chemical SMILES. SMILES bond-direction metadata is explicitly distinct from drawn wedges.

The initial version rejects CXSMILES/E-SMILES annotations rather than discarding them. Such samples must retain character supervision until their semantic mapping is implemented; rejection from this pilot is not removal from the admitted corpus. In the train-only pilot, 257 of 288 selected records exported successfully, including all 240 peptide records across 15 strata and all 16 selected complete molecules. The remaining 31 use unsupported extensions. Report: `output/molsight_training_audit/graph_alignment_pilot_v2/report.json`. This proves a chemical-graph serialization/alignment step, not drawing alignment, graph-head training or capability gains. The running training runtime was not changed.

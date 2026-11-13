# MolSight spatial adaptation

Experimental image → atom queries → chemical graph adaptation of the existing
MolSight encoder. Its pretrained character decoder and 104-token vocabulary are
retained. The default freezes the entire backbone while learning the spatial
head; `--backbone-mode joint` enables complete-molecule and peptide replay. No SELFIES or MolNexTR
backbone is used here. The shared `ocsr_reboot` imports provide only graph,
objective, checkpoint and protocol utilities.

The detector uses distinct deformable sampling offsets, actual Fourier position
features, iterative coordinate refinement and low-rank chemical pair heads.
Hungarian matching, denoising masks, five-class focal weights and visible stereo
are supervised separately. All declared atoms, isotopes, charges, explicit H,
radicals and disconnected components are retained. Unsupported graphs fail
explicitly. Inference has no gold count, coordinates or atom ordering, and no
valence repair, hydrogen search or AR fallback.

## Development data

`prepare` creates a synthetic three-domain pilot from existing admitted training
and validation identities. Splits are checked by identity, component, image and
available family identifiers. Every admitted drawing must round-trip its
coordinate-aligned labels through strict graph assembly. This verifies labels;
it does not establish image legibility or real-patent generalization.

`pilot_v6` has 97 training and 26 validation drawings, after one severe
atom-collision drawing was excluded. Training includes a
390-atom cyclic peptide and validation includes 383- and 363-atom cyclic
peptides. This deliberately small pilot checks feasibility; its sampling weights
and development exact scores are not deployment acceptance criteria.

## Run

From the repository root, use the MolSight environment:

```bash
export PYTHONPATH=.:training:training/molsight_runtime
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/data/molsight_venv/bin/python -m training.molsight_spatial.train \
  --manifest /home/dahuilangda/molsight_work/molsight_spatial_v2/pilot_v6/manifest.json \
  --checkpoint /home/dahuilangda/molsight_work/molsight_graph_joint_focus_v2/run/update_000500.pth \
  --vocabulary output/molsight_data_v4/vocab.json \
  --admitted-manifest /home/dahuilangda/molsight_work/molsight_patent_merge_v5/admitted/manifest.json \
  --out /path/to/new-run --updates 500 --device cuda:0
```

Add `--preflight` for three real backward/AdamW steps at zero learning rate,
one per domain, including the largest training cyclic peptide. It checks finite
nonzero gradients, byte-unchanged model state and strict checkpoint reload.
Normal training requires real deltas in every declared trainable module. Frozen
mode verifies every backbone parameter and buffer is byte-unchanged, and omits
repeated AR inference. Joint mode requires encoder/decoder/detector deltas and
stops if a development domain's retained AR exact count drops below its baseline.

`--resume /path/to/update_000250.pth` restores optimizer, scheduler, RNG and
sampling cursor with strict source/config/data hashes. Keep all training arguments
identical; output directory and stop boundary may change. Existing update logs
must end at the restored step. CUDA `grid_sample` backward can be nondeterministic,
so future GPU trajectories are not claimed to be bit-identical. A CPU integration
test verifies the next real spatial update exactly.

## Artifact and image-only prediction

```python
from training.molsight_spatial.artifact import export_artifact
export_artifact('/path/to/update_000500.pth', '/path/to/new-artifact')
```

```bash
/data/molsight_venv/bin/python -m training.molsight_spatial.predict \
  --artifact /path/to/new-artifact --image /path/to/image.png \
  --device cuda:0 --out /path/to/prediction.json
```

The artifact contains the full model and vocabulary and binds runtime source
hashes. It does not need the original base checkpoint to load. Inference does not
restore training RNG or optimizer state. Artifacts remain experimental pending
independent, image-only small-molecule, linear-peptide and cyclic-peptide results.

## Verification

```bash
/data/molsight_venv/bin/python -m pytest training/molsight_spatial/tests \
  training/molsight_runtime/tests/test_chemical_focal.py -q
```

Audit evidence and the true baseline comparison are documented in
`docs/molsight_joint_capability_audit_2026-11-12.md`. Conditional gold-coordinate
29/63 topology and 28/63 stereo results are not image-only recognition accuracy.
The public USPTO benchmark has only two candidate cyclic peptides, both missed
by the existing baselines; it cannot validate large cyclic-peptide capability.

## Reserved image cohort evaluation

`training.evaluate_molsight_spatial` accepts a frozen JSON list containing
`record_key`, `component`, `file_path`, `image_sha256` and `SMILES`. It rejects
invalid gold and changed images before inference rather than removing them from
the denominator. Every image is passed to the model without gold fields. Outputs
include per-record raw predictions, exact/topology/parse counts, cohort and
artifact hashes, and an explicit terminal status. Add `--with-ar` to separately
score the retained character decoder at its native 512 input size.

```bash
/data/molsight_venv/bin/python -m training.evaluate_molsight_spatial \
  --artifact /path/to/new-artifact --pairs /path/to/frozen-pairs.json \
  --out /path/to/new-evaluation --device cuda:1 --with-ar
```

The evaluator checks model state remains unchanged and all image hashes still
match at completion. Real patent peptide candidate flags stay explicitly marked
as unreviewed. Use component-level paired uncertainty when comparing models;
multiple images from one patent component are not independent evidence.

The larger `expanded_v2` corpus has 770 training and 98 validation drawings,
with 256/256/258 and 32/32/34 small/linear/cyclic counts. This remains synthetic
development data. The first joint pilot stopped at update 250 for AR regression,
with zero spatial exact on both development validation and 664 reserved real
patent components; it is a failed experiment, not a release candidate.

## Explicit continuation and hard-negative experiments

`--spatial-warmstart /path/to/spatial-checkpoint.pth` starts a new objective
phase with the complete spatial model restored and a fresh optimizer. It first
verifies all original MolSight backbone bytes still match the declared base.
The provenance records this reset, the source checkpoint and the accumulated
prior update count. Data augmentation/sampling advances past prior updates.
`--resume` restores that phase's optimizer/scheduler/RNG and sampling cursor;
retain the original `--spatial-warmstart` argument when resuming such a phase.

`--edge-negative-ratio 4` keeps every known chemical positive and the hardest
`max(32, 4 × positive-pair-count)` known negatives per image, bounded by available
negatives. `--stereo-negative-ratio 4` applies the same principle independently
to wedge/double heads. Unknown pairs remain unknown. This addresses dilution of
local false chords or stereo cues by quadratically many easy non-bonds. It does
not modify inference graphs or repair valence.

Optional `--balance-objectness`, `--objectness-weight` and `--count-weight` expose
separately tested experimental objectives. They remain disabled/default-scaled
in the reported mining runs, so their effectiveness has not been established.

The larger requested 1024-per-domain release failed admission: only 660 unique,
round-trip-admissible linear-peptide identities were available. Its partial
files have no approved manifest and must not be used for training. All current
small-molecule preparation is still capped at 120 atoms, an explicit pilot
coverage limit rather than evidence for every complete molecular graph.

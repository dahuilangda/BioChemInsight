# Universal OCSR restart: execution and independent review

Goal: genuinely recognize complete small molecules, linear peptides and cyclic/modified peptides from images, with correct chemical graph and stereochemistry. This goal remains active. Passing the infrastructure or memorization gates is not completion.

## Requirements and evidence ledger

| Requirement | Current evidence | State |
|---|---|---|
| Preserve failed experiments before replacing them | 84 source/log/report/checkpoint files and SHA256 inventory at `/home/dahuilangda/molsight_work/ocsr_reboot_v1/archive/stage5d_20261111/archive.json`; original PID3019091 terminated | Done for original process |
| Prevent accidental mutation of production weights/runtime | New code under `training/ocsr_reboot`, new artifacts under separate `ocsr_reboot_v1`; no production model/config replacement | Maintained |
| Independent architecture, training and evaluation reviews | Three independent agents; concrete data, optimizer, EOS, probability and checkpoint bugs found, followed by executable regressions and cross-review | Ongoing at each new stage |
| Correct focal/DN/matching math | `objectives.py`, shape/gradient/direction/missing-supervision tests | CPU verified; these DETR helpers are not used to mislabel the new AR warm-start as DETR |
| Real warm-start architecture | Plain pretrained MolNexTR encoder and decoder, strict load, explicit SEP tail migration; no MoE or native chemical repair | Implemented |
| Source/vocabulary/data binding | Full model source snapshots, source/base-weight/manifest/vocabulary hashes, stoi/coordinate bins/SEP mapping, RDKit/albumentations/torch versions | Implemented in current run |
| Atomic optimizer/RNG/scheduler checkpoint and exact reload | Schema2 binds algorithm classes, parameters, groups, step, state and provenance; CPU next-step restart tests; actual GPU parameter reload | Verified; true interrupted large-run resume remains to be exercised |
| Every module really learns | Actual GPU finite nonzero gradients in encoder, decoder and graph head; module parameter deltas and byte-exact strict reload | Preflight verified, monitored during gate training |
| Native/high-resolution execution | Actual BF16 backward at384 and768;390 atoms/1670tokens at768 peak allocated about2.88GB in preflight | Execution verified, accuracy not implied |
| Correct pixel→atom correspondence | Actual metadata, admitted image hashes, one transform for image+coords, indexed stereo-aware graph equality;17 independent data tests | Verified for diagnostic cohort |
| No split leakage | Frozen gate_v5:16 training structures and5 validation structures; complete molecular identity/component/image guards; no random per-image heldout | Gate verified; broad publication-family independence remains unproven |
| Model can memorize difficult complete structures | `memorize768_v2`,1000 updates, all16 images including390atom cyclic peptide; actual raw image-only EOS and predicted graphs retained | Running; no pass asserted |
| Explicit graph state/stereo supervision and raw assembly | Separate raw graph/state/stereo contract being implemented and tested | Incomplete |
| Independent three-domain generalization | Current5 validation samples are development diagnostics only | Incomplete |
| Actual production service and resource acceptance | No new recognizer released | Incomplete |

## Independent review findings beyond the original audit

- Old stage5 evaluation supplied seven-class probabilities to an assembler expecting merged five-class probabilities; single-bond ranking differed.
- Old canonical comparison labeled `phase1_last.pth` as `focal_last`; the28/63 oracle score cannot be bound to the latter filename without a hash.
- Old deformable cross-attention's identically zero-initialized sampling points remain symmetric in a CPU training probe; four nominal points effectively collapse to one per head.
- New data importer originally checked canonical identity after atom renumbering, which is permutation invariant. Independent adversarial tests exposed element/topology/stereo mapping failures. The importer now compares uniquely atom-index-labeled stereochemical identity and original authoritative image hashes.
- Independent cross-review of new checkpoints caught optimizer/scheduler algorithm mismatch and whole-payload finite-state holes; schema2 rejects both. Missing EOS defaults to false, rather than inventing termination.
- Plain raw decode bypasses production bond-count clipping and graph repair. Sequence EOS/exact is independent of graph quality; raw graph classes are retained, with reverse-direction-aware seven-class probability symmetrization.

## Gate cohort

`/home/dahuilangda/molsight_work/ocsr_reboot_v1/gate_v5/manifest.json`:

- Train16:4 complete synthetic diagnostic drawings (including stereo, charge/components and isotope),4 linear peptide strata,4 ring/multiclosure strata and4 patent-style large cyclic peptides spanning their size range.
- Validation5:2 distinct small diagnostic drawings,1 admitted linear peptide,1 small cyclic peptide and1 admitted large cyclic peptide.
- All image/coordinate/structure transformations are checked. Token targets cannot silently receive random missing coordinates, unknown tokens, or truncated EOS.
- Synthetic drawings and already-admitted validation images are development evidence, not a blind real-patent product benchmark.

## Current run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. \
/home/dahuilangda/miniconda3/envs/llm/bin/python -m training.ocsr_reboot.train \
  --manifest /home/dahuilangda/molsight_work/ocsr_reboot_v1/gate_v5/manifest.json \
  --out /home/dahuilangda/molsight_work/ocsr_reboot_v1/memorize768_v2 \
  --updates 1000 --eval-every 250 --warmup 25 --input-size 768
```

The initial768 image-only raw-sequence baseline is0/16 train and0/5 validation. This is a starting measurement, not proof that the pretrained model is intrinsically bad: resolution adaptation and the diagnostic drawing distribution differ from its native384 inputs. Actual training uses the same preprocessing at train/evaluation and has nonzero gradients in the visual backbone.

This first stage jointly supervises sequence/coordinates and chemical bonds without fabricated drawing wedge labels. It deliberately does not claim full graph/stereo completion. Next stages must add explicitly verified visible-stereo/state supervision and compare raw graph reconstruction and the sequence path on the same image-only inputs.

Earlier `memorize768_v1` stopped before any update to add independent-review-required vocabulary provenance; its state records this reason. Earlier preflight_v1 found and corrected a TorchVersion metadata serialization issue; native_v2 and large768_v1 completed actual updates and byte-exact reloading. None of these runs is renamed into a successful product experiment.

An external Zcode session restarted the old stage5d in its old directory on GPU1 after the archived process was terminated. The new experiment uses GPU0 and its own immutable update filenames. External activities are not counted as this run's results.

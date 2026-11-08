# MolSight data and training

Complete and validate the structure generators before publishing a synthetic training release. Then run supervised fine-tuning, followed by a separately evaluated reinforcement-learning experiment.

## Capabilities

Preserve complete-molecule recognition while adding reliable Markush, attachment-bearing fragment, peptide, and cyclic-peptide recognition. Evaluate overlapping categories separately; a fragment or Markush structure may also contain peptide chemistry.

Retain an unchanged baseline checkpoint and frozen, source-checked evaluation sets for each category. Preserve the existing output contract until tokenizer, decoder, graph annotations, and postprocessing have been checked together. Atomic recognition and sequence interpretation remain separate: ambiguous monomer decompositions are not unique sequence labels.

The peptide-specific review matrix, image contracts, split rules, and two-GPU execution plan are in [Peptide image data and training](../peptide_data/PLAN.md).

## Data release requirements

- Retain source structures, definition versions, atom maps, rendering settings, seeds, and image hashes.
- Check connectivity, charge, isotopes, stereo, terminal groups, variable-group labels, attachment labels, and their bond endpoints.
- Check the final image at training resolution for readable text, wedges, bonds, connection points, cropping, and collisions.
- Compare typography, line widths, spacing, layout, resolution, and scan effects with held-out real patent figures. Keep visual-review records and measured style distributions.
- Split by patent family, structure family, monomer combinations, and rendering variants before augmentation. Detect structure and image duplicates across splits.
- Keep uncertain source assignments and unreadable or rejected images outside the deterministic-label training release.
- Treat model agreement and fingerprint similarity as diagnostic signals, not independent verification of a source label.
- Keep synthetic provenance in every release, including visually realistic images.

The existing residue renderer identifies its output as a pilot and does not approve it for training. A new release must independently satisfy chemical, image, and held-out evaluation requirements.

## Training audit

Audit the actual trainer, dataset, tokenizer, objectives, optimizer, accumulation, precision, sampling, generation, rewards, reference policy, and evaluation. Record source hashes and reproduce defects before changing behavior.

The initial audit of `/data/molsight_repo` reproduced three defects:

| Component | Observation | Required correction |
| --- | --- | --- |
| Smoothed KL loss | The smoothed target mass is below one | Define and test a normalized distribution and padding policy |
| Sequence loss | An extra ignored target still produces a gradient | Apply all masks and test gradients at ignored positions |
| Stereo reward | Different attachment-bearing structures obtain the same exact-match reward because both InChIKeys are empty | Use a validated structure comparison that preserves attachment semantics; never equate failed conversions |

`audit_training.py` runs the original objective and reward definitions on CPU and records their source locations and hashes. An exit code of 2 reports reproduced defects. It does not modify the training checkout.

```bash
python training/molsight_markush/audit_training.py \
  --repository /data/molsight_repo \
  --out output/molsight_training_audit/report.json
```

Further checks include token truncation and EOS handling, batch packing and loss denominators, FP32 objective calculations under mixed precision, missing annotations, augmentation effects on stereo and labels, and adapter/reference state during reinforcement learning and validation. The current mean of microbatch token means is not equivalent to a combined token mean when token counts differ; choose the intended objective explicitly.

Review supervised label smoothing, decoder capacity, learning rates, dropout, length handling, and task weighting using controlled experiments. Prefer measured improvements to changes made solely because an approach is newer.

## Supervised fine-tuning

Start with a small, fully reviewed release and a fixed real-image validation set. Mix new samples with existing complete-molecule and extended-structure data. Select replay proportions from per-category performance and gradient statistics. Report failures and long-sequence behavior in addition to average loss.

Compare the unchanged baseline, replay-only control, and new-data training at matched compute and evaluation settings. Expand the dataset only after the pilot improves independently checked real-image results without a material regression in protected capabilities.

## Reinforcement learning

Begin only after supervised objectives, rewards, generation, and evaluation are validated. Use a fixed reference policy, valid rollout masks, consistent sampling probabilities, controlled update ratios, and finite, correctly normalized gradients. Check policy/reference adapter state and frozen encoder behavior explicitly.

Reward exact atomic structure, specified stereo, visible attachment identifiers, and connection endpoints. Partial credit must distinguish incomplete structures from incorrect substitutions. Parsing success, atom count, fingerprint similarity, or string length alone must not make an incorrect structure preferable. Test deliberately wrong stereo, changed attachments, missing components, duplicate groups, invalid structures, and unsupported notation before using a reward.

Compare reinforcement learning with a supervised continuation at matched compute. Keep it only if the independent real-image and capability-retention measurements improve.

## Evaluation

Report valid-parse rate, exact connectivity, exact structure including stereo, attachment/variable-label correctness, connection-endpoint correctness, and complete sample correctness. Fingerprint similarity is secondary. Include all failures, invalid outputs, truncations, and timeouts in denominators.

Separate raw model decoding from postprocessing and assisted recovery. Evaluate each output contract without replacing unknown labels, dropping molecular components, or discarding attachment semantics. Use confidence intervals and per-category results; a small mean fingerprint gain does not establish broader capability.

Do not reuse training examples or repeated source-family variants to select checkpoints. Preserve the baseline and all candidate checkpoints, dataset manifests, source versions, configurations, and evaluation results.

## Runtime verification

Corrections are implemented in `training/molsight_runtime`, a source snapshot of the original working tree. Its `SOURCE.json` records provenance. The original checkout and weights remain unchanged.

Decoder-level checks now cover normalized objectives, masked gradients, strict structure rewards, raw evaluation, rollout versus teacher-forced probabilities, clipped policy ratios, adapter state restoration, fixed-reference output after an update, cache cleanup, SCST baseline grouping, and checkpoint recomputation. GPU 1 tests cover FP32/BF16 and attention/position/checkpoint variants. GPU BF16 testing reproduced and then corrected detached gradients caused by autocast weight reuse across sampling and training passes.

These checks are prerequisites. They do not approve the training data release or establish full-model training quality. Effective-batch normalization, coordinate objectives, full trainer integration, structured Markush labels, independent real-image evaluation, and baseline retention remain open.

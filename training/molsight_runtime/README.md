# MolSight training runtime

Workspace implementation of the MolSight trainer. `SOURCE.json` records the source checkout and hashes; `UPSTREAM_README.md` and `LICENSE` retain upstream documentation and licensing. Original checkpoints and datasets are read-only inputs.

## Structure labels

`molsight.structure` provides typed SMILES, E-SMILES, and CXSMILES comparison. Identity includes stereo, components, charge, isotope, attachment maps, variable names, repeat groups, and position-variation sets. Numeric R-group identifiers and attachment maps use distinct namespaces.

Native serialization verifies candidate bond notations against the source graph. A valid structure that cannot be serialized losslessly remains available for comparison, while its recognition-target export fails explicitly. Unknown characters, malformed targets, discarded input, and excessive target lengths also fail explicitly.

`molsight.abbreviations` supplies a versioned single-port interpretation for defined groups such as Ph, CF3, and Boc. The registry records explicit attachment atoms. Expansion retains stereo and remaining annotations; conflicting ports, mapped dummies, and position-variation anchors remain unresolved. Document definitions can reserve otherwise standard labels. Ambiguous labels, including Py, Tol, NC, and NBoc, are preserved. Literal identity remains available independently of resolved chemical identity. This follows the abbreviation/substituent distinction in [MolParser](https://github.com/dptech-corp/MolParser#post-process-e-smiles).

Rewards and evaluation use this same contract. Invalid predictions receive zero exact reward and remain in accuracy denominators. Fingerprint similarity, weighted by annotation agreement, is a secondary shaping reward. Model selection uses raw exact structure accuracy; chemical repair is reported separately.

## Objectives and updates

Supervised objectives use FP32 reductions. Effective-batch denominators count valid sequence tokens, weighted annotated edges, and valid coordinate components across microbatches and ranks. Coordinate likelihoods are normalized truncated Laplace mixtures. Missing supervision has a differentiable zero contribution.

Policy accumulation weights per-completion objectives by image count across ranks. Clipping metrics use sampled-token counts. Nonfinite objectives fail before backward, and learning-rate schedules advance only after successful optimizer steps. BF16 execution does not use dynamic gradient scaling.

Policy updates suppress automatic DDP reduction and synchronize gradients once after accumulation. Parameters used on only some ranks receive the same global gradient; parameters unused everywhere retain a missing gradient and skip optimizer decay.

GRPO stores rollout probabilities and applies the clipped old-policy ratio. Sampling, probability calculation, and the frozen reference use the same distribution. Completion masks include EOS and an invalid sampled PAD action, while excluding padding after termination. Adapter state, evaluation mode, checkpoint recomputation, and KV hooks are restored after each operation.

`fit.py --objective grpo --policy-reference CHECKPOINT` uses full-parameter policy optimization with an independent frozen supervised reference and an identical recorded vocabulary. Student image features are recomputed with gradients; reference features come from the reference encoder. Encoder and decoder gradient tests cover checkpointed and ordinary execution. This path shares the admitted data loader, deterministic sampling, checkpoint restart state, image-only evaluation, and complete-molecule regression checks with supervised training. It currently excludes supervised anchor replay. Production policy validation remains required. The legacy `post_train.py` driver still uses adapters. Checkpoint loading requires complete parameter coverage and preserves existing token IDs during vocabulary extension.

## Image resolution

Checkpoint configuration includes the original image-position grid. Higher-resolution features evaluate the same random Fourier basis at new normalized coordinates; the recovered basis is checked against the stored grid. Native-resolution buffers are returned unchanged. This preserves the encoding function rather than replacing it with a new random grid. The encoding follows the same normalized-coordinate construction as [SAM's random position encoder](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py).

Positional compatibility does not establish recognition accuracy. On the frozen diagnostic sample, both existing checkpoints lose complete-molecule accuracy when every input is enlarged from 512 to 1024. Training must retain native-resolution replay and validate high-resolution adaptation separately.

`molsight.prepared` reads memory-mapped, pretokenized Arrow records. It verifies target-file and image hashes and preserves validated peptide frames. Character-only records contain no fabricated graph or coordinate targets. `molsight.sampling.DomainBatchSampler` samples domains per optimizer update, keeping domain weights independent of caption length. Peptide strata are sampled uniformly; microbatches share a resolution across ranks. Sampling resumes deterministically at an update boundary.

`fit.py --encoder-lr RATE` enables encoder weight updates at a separate learning rate. Zero keeps the encoder frozen. Encoder normalization statistics remain fixed during adaptation. A trainable encoder uses an independent reference encoder for replay; student image features retain gradients. The decoder uses `--lr`. Both rates are recorded in the resume contract.

The supervised sequence objective is KL divergence to a smoothed token distribution, so its minimum is zero. Update diagnostics separately report unsmoothed token NLL and teacher-forced token accuracy, weighted by valid token counts across ranks. Neither measures unassisted whole-molecule recognition. GRPO also reports the fractions of nonconstant reward groups and zero-reward completions: a centered surrogate near zero can have nonzero gradients, while constant reward groups contribute no advantage signal.

`--peptide-group-weights '{"synthetic":0.5,"discovered_cyclic":0.35,"discovered_linear":0.15}'` sets sampling proportions within the peptide domain. Synthetic strata remain uniform within their group. The weights must cover every available group; omitted weights preserve the original uniform stratum sampler. Domain proportions, resolution selection, and deterministic restart remain separate from these group proportions. Discovered candidates retain their source labels and require source review before use as a benchmark.

The 250-update peptide/replay experiment improved exact recognition from 1,273/1,800 to 1,303/1,800 on a source-balanced complete-molecule validation cohort: 54 gains and 24 losses. The paired bootstrap interval for the accuracy change is +0.72 to +2.61 percentage points. These are validation diagnostics with unknown historical pretraining overlap; large-molecule coverage is limited. Synthetic peptide exact recognition increased from 11/64 to 24/64 in the separate training validation panel. Neither result establishes independent patent generalization.

## CPU allocation

Loaders share available CPUs across `LOCAL_WORLD_SIZE`, reserving one CPU per local training process. On a 12-CPU host, two GPU ranks use five data workers each. `--num_workers` overrides this allocation; zero selects serial loading. Workers use spawn, persistent processes, two-batch prefetch, and single-threaded OpenCV/PyTorch execution.

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before launching Python. Preparation, rendering, and auditing use bounded process pools and cached image artifacts. Divide workers between concurrent jobs.

TorchInductor defaults to one compile thread per process. `TORCHINDUCTOR_COMPILE_THREADS` overrides this setting. This avoids independent compiler pools consuming the CPU allocation reserved for data loading.

## Validation

From the workspace root:

```bash
PYTHONPATH=training:training/molsight_runtime python -m pytest -q training/molsight_runtime/tests
```

`preflight.py` checks a complete checkpoint against actual image/label pairs without optimizer updates. `distributed_preflight.py` compares production DDP gradients with the same global supervised batch. It executes one zero-learning-rate optimizer step and verifies unchanged parameter hashes.

`benchmark.py` decodes images without target tokens or chemical repair. It reports literal and abbreviation-resolved exact identity, parse success, genuine EOS termination, category results, timings, and checkpoint/data hashes. Existing source-test overlap with historical checkpoints is unverified; the small diagnostic sample is not a final generalization benchmark. Frozen comparisons are under `output/molsight_baseline_v1`.

The deployed R3 preflight covers its twelve-layer decoder, EfficientViT encoder, 1024-pixel inputs, BF16, and targets up to 658 tokens. The two-GPU accumulated gradient differs from the global-batch reference by approximately 0.164% in relative L2 norm; loss error is below 3e-8. Reports are `output/molsight_training_audit/production_r3_preflight_1024.json` and `production_r3_ddp_1024.json`. These checks establish execution and gradient behavior, not recognition accuracy, training stability, or inference throughput.

## Training release

The first merged release has passed admission. Broader capability-retention comparisons, exact optimizer/RNG restart verification, and production policy updates remain required. Coordinate supervision needs verified drawing-to-token atom alignment. Reward weights and policy objectives require controlled comparisons with supervised continuation.

`fit.py --peptide-tags PATH` adds verified backbone candidates to peptide sampling while retaining their original recognition domains. Additional discovered-peptide validation records are selected by independent components and reported separately. The protected complete-molecule cohort remains unchanged. The current release provides 6,210 additional training records and 24 additional validation components through this route.

`scalars.jsonl` records finite training and validation metrics. Short runs have exercised supervised updates and restart behavior. Main capability fine-tuning and reinforcement learning remain pending.

## Strict workspace checkpoint worker

`utils/molsight_runtime_worker.py` is an opt-in JSON-line worker using the same workspace model, image preprocessing and image-only decoder as training evaluation. It requires the checkpoint's recorded architecture and exact vocabulary mapping/file hash, loads all model parameters strictly, and rejects nonfinite model tensors. Configure `MOLSIGHT_WORKER` to its absolute path and set `MOLSIGHT_CHECKPOINT`, `MOLSIGHT_VOCAB`, `MOLSIGHT_INPUT_SIZE` (512, 768 or 1024) and `MOLSIGHT_MAX_LEN` (up to the recorded limit). The existing service passes `MOLSIGHT_DEVICE` as its device argument. The legacy worker and active service configuration are unchanged.

The new worker returns the unmodified decoded text, genuine EOS termination and parse status. `ok` requires both parse success and termination, so incomplete but syntactically valid fragments cannot silently become successful recognitions. It performs no chemical repair. Readiness includes checkpoint/vocabulary hashes, precision, input size, decode limit and training update. Requests normally contain `image_path`; `model_input_preprocessed: true` is reserved for already verified square model inputs and must match the evaluation preprocessing contract.

Actual update125 CPU smoke checks covered strict loading, 512 and 768 image inputs, and EOS termination on one existing diagnostic image. They do not establish GPU throughput, cyclic-peptide accuracy or production readiness. Keep the legacy service active until independent capability comparisons and GPU serving checks justify a release.

For capability-panel comparisons, pass `--exposures /path/to/development_exposures.json` to `compare_capability_panel.py`. The registry binds exposed components to the frozen report hash. Full-panel results remain visible, and separate `unexposed` results exclude components used for engineering diagnostics. Version 1 has one such component (the worker CPU smoke image has a different path but shares its component with a selected image). This is why path-only separation is insufficient; do not describe all 882 records as prediction-unseen. The original pair files are unchanged.

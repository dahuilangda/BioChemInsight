# MolSight training runtime

A working source snapshot of the existing MolSight trainer. `SOURCE.json` records the original working tree, base commit, and file hashes. The original checkout, checkpoints, and datasets are unchanged. `UPSTREAM_README.md` contains the upstream documentation; `LICENSE` retains its license.

This directory is under validation. It is not an approved training release.

## Verified corrections

- Label smoothing distributes unit probability over the target vocabulary, excluding ignored classes. Objective calculations use FP32.
- Sequence loss excludes all configured ignored tokens and handles batches with no supervised tokens without NaNs or gradients.
- Edge loss computes a weighted mean over annotated edges and handles missing annotations without hiding numerical failures.
- Exact structure rewards compare canonical isomeric graphs, retaining components, charges, isotopes, and atom-map identifiers. Failed InChI conversion cannot yield an exact reward. Atom-count bonuses were removed.
- Rewards consume original predictions and targets. Chemical repair does not determine reinforcement rewards.
- Invalid reward targets fail explicitly; invalid predictions receive zero. Unequal list lengths fail instead of silently dropping samples.
- Tokenization rejects unknown characters and discarded input. Datasets reject targets exceeding the length limit instead of truncating molecular labels.
- Evaluation rejects invalid gold structures, retains failed predictions in the denominator, reports parse success, and preserves unknown-label failures.
- Evaluation CLI uses the requested prediction column without automatic chemical repair and rejects duplicate image identifiers.
- Model selection uses raw exact structure accuracy, including specified stereo. TensorBoard records it as accuracy.
- GRPO samples from the same full distribution used for policy probabilities, records rollout probabilities, and applies a clipped old-policy ratio.
- Reference calculations restore adapter state; policy calculations restore module training modes. Cached features require a frozen encoder and base model.
- EOS is included in the completion objective; padding and tokens after termination are excluded. Constant-reward groups have finite zero advantages.
- Sampling and reference passes disable autocast weight caching, preventing detached BF16 weights from contaminating policy gradients.
- KV hooks are removed after failures. Non-reentrant gradient checkpoints preserve forward adapter and training states during recomputation.
- SCST expands the greedy baseline per image and uses summed sequence log probabilities under the sampled policy.

## Validation

Run in the existing MolSight environment from the workspace root:

```bash
PYTHONPATH=training/molsight_runtime /data/molsight_venv/bin/python -m pytest -q training/molsight_runtime/tests
```

These are objective, gradient, tokenizer, and metric regression tests. They do not establish model accuracy or training stability.

## Remaining validation

| Area | Required work |
| --- | --- |
| Structured labels | A shared typed parser for named Markush variables and extended labels; explicit attachment-notation equivalence |
| Supervised training | Effective-batch loss denominators, distributed normalization, optimizer and scheduler behavior, overflow handling |
| Coordinate objectives | Distribution parameterization, valid annotations, numerical stability, and augmentation consistency |
| Reinforcement learning | Distributed execution, production checkpoint integration, long-sequence stability, and controlled reward/objective ablations |
| Evaluation | Independent real-image splits, source-family leakage checks, capability retention, uncertainty estimates, and category-specific reports |
| Data | Chemistry and image quality gates, length/vocabulary preflight, human review, and reproducible release manifests |

Canonical isomeric SMILES comparison currently covers RDKit-readable structures. Arbitrary Markush labels and extended notation are not yet a complete supported reward contract. Isotope-labelled and atom-map-labelled dummy atoms are not automatically treated as interchangeable. Fingerprint reward remains a secondary shaping term whose weights require controlled experiments. The reinforcement objectives have decoder-level gradient tests; full trainer and production-checkpoint validation remain incomplete.

No new fine-tuning or reinforcement run has been started from this snapshot.

## GPU validation

The workspace sandbox does not expose NVIDIA devices. Running outside it with `CUDA_VISIBLE_DEVICES=GPU-dadf9245-17d3-7050-f678-e155fe4ace48` selects physical GPU 1, an RTX 4070 Ti SUPER. It appears as logical `cuda:0` inside that process. Physical GPU 0 is excluded.

The GPU regression suite runs the actual MolSight decoder with a small configuration and supplied image features. It covers FP32 and BF16, checkpointing on/off, rotary and learned positions, and SDPA and explicit attention. This establishes decoder/objective gradient behavior, not full-model accuracy or production throughput. The GPU test report is `output/molsight_training_audit/gpu_policy.xml` relative to the workspace root.

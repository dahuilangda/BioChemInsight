---
name: biocheminsight-model-common
description: Shared runtime and schema conventions for all BioChemInsight language and vision model calls, including default system prompts, retry policy, temperature, and reusable output-contract patterns.
model: inherit
effort: medium
context: auto
---

# BioChemInsight Model Common

Use this skill as the shared base convention for model-facing tasks.

## Shared goals
- Keep language and vision model calls deterministic by default
- Standardize retry behavior and output contracts
- Make task-specific skills override only what is truly task-specific
- Route production model calls through the harness so transport errors, timeouts,
  and output-contract errors are classified and auditable
- Keep prompts progressively disclosed: short task prompt first, schema/reference
  files for output shape, examples only when the task needs them

## Shared references
- `references/runtime.json`
- `references/output_schemas.json`

## Contract rules
1. Do not silently coerce malformed model output into a successful result.
2. Do not maintain duplicate truth fields. If compatibility fields exist, derive
   runtime decisions from the strict `decisions`/object contract.
3. Recover at task boundaries with explicit warning/audit records, not guessed
   model answers.

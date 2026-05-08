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

## Shared references
- `references/runtime.json`
- `references/output_schemas.json`

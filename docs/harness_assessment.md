# Harness and skill assessment

Three independent audits examined model orchestration, evaluation, and interface behavior. Findings below distinguish implemented checks from proposed work. Passing a response contract does not establish that the extracted science is correct.

## Changes

| Area | Change | Validation |
| --- | --- | --- |
| Page extraction | Shared vision harness; strict JSON and entity contracts; preserved response text | Malformed, empty, escaped, and incomplete response tests |
| Document orchestration | Required page and ledger failures propagate; missing or repeated pages are rejected | Failure injection and page coverage tests |
| Measurement discovery | Candidate order follows source pages rather than completion order | Reordered concurrent completions |
| Evaluation | Missing entities count as failures; unresolved references remain inconclusive; stereochemistry participates in comparison | Empty output, denominator, enantiomer, and core-only cases |
| Submission | Cancellation intent survives the interval before a task ID arrives | Deferred submission and cancellation responses |
| Interface | Visual extraction options, compact source replacement, explicit capability errors and retry | Capability states, keyboard input, narrow layouts, long filenames |
| Skill catalog | Duplicate names cannot overwrite handlers; schemas are copied; registration requires asynchronous handlers | Registry mutation and registration tests |
| Peptide skills | Removed recursive decoder call; input contracts and validation scope exposed | Decoder dispatch and malformed-input tests |

The peptide comparison reports residue-code agreement. It explicitly does not certify stereochemistry or identity against a source figure. All thirteen registered skills now declare input schemas. These catalog declarations are not a claim that every legacy handler enforces a complete schema. Output contracts and consistent invocation validation still need work.

## Remaining gaps

1. A valid source citation can still accompany an unsupported keep, merge, or exclude decision. Line-bound checks establish reference validity, not entailment. Different targets, endpoints, conditions, units, and experimental versus calculated measurements need separate semantic evaluation.
2. Legacy atomic skills are not uniformly integrated with the shared harness. The activity-table skill now uses model extraction and independent review instead of an endpoint grammar. Sequence-ID vision now uses strict structured responses through the shared multi-image transport; uncertain annotations raise evidence errors. The sequence-listing assembler still has a legacy text grammar and needs separate semantic migration and evaluation.
3. Peptide helpers have different representation and coverage limits. Formula equality and residue-code agreement cannot replace graph identity with stereochemistry. A successful round trip cannot establish agreement with the original image.
4. The current evaluation corpus cannot support a broad accuracy claim. Thirty-two live fixtures reduce to six normalized textual templates; they are not thirty-two independent document tasks.

No semantic keyword filter or alternate extraction result was introduced to close these gaps. Unsupported or incomplete evidence must remain visible as an unresolved result or a failure.

## Corpus coverage

The inspected local inventory contains thirteen literature PDFs. The literature reference file covers two papers and thirty-eight activity values, both for α-glucosidase. One paper has eighteen structure names; the other supplies a scaffold rather than complete member structures.

The patent manifest contains 282 unique patents and 507 picks. Human verdicts contain 505 labels: 499 positive and six negative. Accepting every pick would score 98.8% raw accuracy on these labels. Rejection quality therefore needs a separate, deliberately sampled test set.

Split evaluation by document and patent family; keep duplicate figures and derived crops in the same split. Maintain distinct scores for entity recall, measurement ownership, units and conditions, source support, full graph identity, stereochemistry, and unresolved evidence. Report completeness alongside precision. Never shrink denominators because reference resolution failed.

## Skill organization

Group skills by the evidence or representation they consume and produce:

| Group | Responsibility | Required distinction |
| --- | --- | --- |
| Document evidence | Pages, regions, tables, and source references | Read failure versus absence |
| Chemical identity | Names, identifiers, and source-linked candidates | Resolved name versus verified source identity |
| Graph operations | Assembly, peptide decoding, and depiction | Valid graph versus correct interpretation |
| Measurements | Endpoint, value, unit, condition, and compound ownership | Reported versus calculated values |
| Verification | Source support and graph comparisons | Scope of each verification result |

Keep stable public skill names while correcting implementation names locally. Add contracts before exposing more composition paths. A coordinator should schedule operations whose prerequisites are satisfied and preserve their evidence; adding more skills or agent loops alone does not establish reliability.

Independent semantic review is implemented for the measurement skill; broader use still requires evaluation. It should inspect the source and proposed decision without the generator's rationale, identify supporting facts, and report unsupported claims explicitly. Calibrate it against human judgments on held-out documents before making it a release gate. Repeated runs should measure all-run success as well as best-run success, latency, cost, and failure classes.

## Evidence and references

The focused regression suite passed 941 checks on the host and in the worker container. The production frontend build passed. Browser checks passed for extraction options at five widths, cancellation before submission acknowledgment, save feedback and dialog focus in both review views, and three workflow stages at seven resolutions. These checks use controlled fixtures; no new full extraction accuracy result is claimed.

Local artifacts are in `output/harness_review/`. `evaluation_before.json` and `interface_audit.md` are audit snapshots from before the fixes. Regression tests exercise the corrected behavior. Two existing Markush assertions also fail against committed code; their baseline reproduction is recorded separately in `skill_baseline.xml`.

The separation between citation existence and support follows the evaluation problem studied by [ALCE](https://arxiv.org/abs/2305.14627). Outcome and trace checks serve different purposes; see [Anthropic's agent evaluation guidance](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) and [OpenAI trace grading](https://developers.openai.com/api/docs/guides/trace-grading). [PubTables-1M](https://arxiv.org/abs/2110.00061) and [MMSci](https://arxiv.org/abs/2501.13042) provide broader document and figure research context, but do not replace chemistry-specific source annotations.

## Current correction workflow

The measurement skill can revise omissions, conflicts, unsupported assignments, and transcription errors from source evidence. The reviewer sees the source and candidate without generator rationale. It also receives scoped source questions retained by the host; these are provisional claims requiring explicit adjudication. Every revision is a complete candidate and must receive a fresh review. Original candidates and review issues remain in the returned history; model audit files retain stage responses when auditing is enabled. An unresolved conflict or an exhausted revision budget fails explicitly.

This workflow is implemented for the atomic measurement skill. It is not yet a release-qualified semantic gate for the main extraction pipeline. The initial ten-case timeout run remains archived. Service generation recovered; source-linked RMSD extraction and real-source ownership conflict detection now have repeated live checks. Full-document accuracy and long-term availability remain unmeasured.

Source references now select line ranges; the host preserves exact source excerpts. This removes model quotation-copy errors without accepting normalized substitutes. Errors preserve actionable review details. Contract repairs and semantic revisions share one deadline; decoder output is bounded, and truncated generations cannot be accepted. Local capacity expiry does not poison provider health.

The original nine-page reconciliation changed from 21 to 26 retained labels. Independent source review found a docking/MD referent change, representation-dependent MD system granularity, an unverified GlideScore unit, and a genuine 6h/6j source contradiction. The PDF image confirms the conflicting 6h text. Neither retained-name count is a gold reference. See `output/harness_review/source_identity_audit.json` and `source_ownership.png`.

Sequence labels now distinguish detected, absent, and uncertain annotations; malformed responses cannot become absence. Listing sequences are no longer truncated at 400 characters. Antibody inputs are rejected rather than stripped or uppercased. Empty numbering and variant parsing cannot report success. Skill API errors distinguish insufficient evidence (422), timeout (504), circuit state (503), and model contract/provider failure (502).

The source-question ledger survives contract repairs, including invalid neighboring
proposals. Every pending question requires an explicit source-supported disposition.
Response hashes and decision history preserve provenance. Dynamic request schemas
constrain question IDs without changing scientific identifiers or reported values.

The latest field-by-field review check passed both literature cases twice, preserving
the docking comparison reference and unresolved 6h/6j numeric ownership conflict.
Its report is under
`output/measurement_review/context_verification/469e3bc387354ff19237897dac9c937b`.
Earlier incomplete metadata remains recorded as a failure. These two excerpts do not
establish broad accuracy or qualify the main pipeline. Correct values and valid
citations alone cannot establish complete scientific context.

The controlled measurement run completed all twenty cases twice: forty successes,
twenty all-run-consistent cases, and no execution failures. It covers corrections,
omissions, condition swaps, namespace and provenance distinctions, missing values,
source conflicts, and refutation of a seeded false source-conflict claim. Report:
`output/measurement_review/controlled_final/93e811e8134b44e6b363ae0842de8d55/report.json`.
These are controlled cases, not forty independent documents.

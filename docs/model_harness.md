# Model harness

Assay reconciliation uses model decisions with candidate IDs and source line references. The parser accepts a complete validated response or raises an error. It does not rewrite names, accept partial decisions, substitute results, or apply endpoint patterns.

## Layout

| Path | Responsibility |
| --- | --- |
| `utils/assay_reconciliation.py` | Candidate references, source references, response validation |
| `utils/model_harness.py` | Retries, circuit state, call audit |
| `scripts/check_model_harness.py` | Validation command |
| `tests/test_assay_reconciliation.py` | Assay response tests |
| `tests/test_model_harness.py` | Model call tests |
| `utils/extraction_metrics.py` | Completeness and structure scoring |
| `docs/harness_assessment.md` | Independent audit findings and remaining gaps |

## Contract

Each candidate receives one action:

- `keep`: retain the candidate and reference its own ID.
- `merge`: reference another candidate with an explicit `keep` decision.
- `exclude`: provide a source-based exclusion reason and a null canonical ID.

Every action requires evidence with `source_id`, `line_start`, and `line_end`. Line numbers are 1-based and inclusive. The model receives the complete source text with numbered lines. The parser checks IDs, coverage, action consistency, confidence, and line bounds. Source text is preserved, including PDF line breaks and whitespace.

Missing evidence produces `ModelEvidenceError`. Invalid output produces `ModelContractError`. Neither produces replacement results. Semantic judgments remain model decisions; reference validation does not establish scientific accuracy.

The provider receives a request-specific JSON Schema. Unsupported structured output fails explicitly. There is no automatic switch to unstructured output. Structured responses reach the parser unchanged; JSON wrappers, duplicate keys, and non-finite constants are rejected.

## Calls and audit

The harness owns retries; SDK retries are disabled. Contract retries include validation feedback. Authentication, quota, context-size, and other permanent provider errors stop immediately. Missing source evidence is terminal.

Queueing and execution share the outer timeout. Timed-out workers retain their concurrency slot until they exit. Circuit recovery admits one probe. Concurrency limits and circuits are process-local.

Audit records include a call ID, actual attempt count, provider identity, usage, and a response hash. Response files use owner-only permissions and a 1 MiB limit. Truncation is recorded. Audit I/O errors are logged without replaying model calls. Response files are stored beside the audit; the current results ZIP does not include them.

## Checks

```bash
python scripts/check_model_harness.py
python scripts/check_model_harness.py --model-cases 32 --workers 2
python scripts/check_model_harness.py --request request.json --model-timeout 120
```

The first command is offline. The other commands contact the configured model. `request.json` contains `assay_names` and a `page_markdowns` object. Use `--out` to select the report directory.

| Check | Result |
| --- | --- |
| Host focused regression suite | 941 passed |
| Worker container, application Python | 941 passed |
| Generated response mutations | 32,768 checked |
| Nested type corruption | 4,096 checked |
| Retry sequences | 1,024 checked |
| Concurrent independent adjudications | 1,024 checked |
| Configured-provider fixtures | 32 passed: 28 completed, 4 expected evidence errors |
| Original failed document | 27 candidates, 9 pages, one successful attempt |

The original document check used 29,119 source characters. The latest call completed in 105.967 seconds and retained 26 names, including calculated RMSD. An earlier call retained 21. The source audit found that a generic RMSD label changed scientific referent between docking validation and MD; neither count is a certified reference. These calls reran name reconciliation, not full extraction. See `output/harness_review/source_identity_audit.json`.

Extended regression has five existing failures: four structure-review state assertions and one missing-image composite check. The same failures were reproduced using the committed harness, pipeline, and model-call code in a separate process. They are not counted as passes.

Local results:

- `output/model_harness/report.json`
- `output/model_harness/container.xml`
- `output/model_harness/model_results.json`
- `output/model_harness/document.json`
- `output/model_harness/regression.xml`
- `output/model_harness/baseline.xml`

The 32 live fixtures reduce to six normalized single-page textual templates. These checks do not measure long-term provider availability, end-to-end extraction accuracy, or accuracy across a literature corpus. Existing tasks were not restarted. New task subprocesses load the mounted source files.

The expanded focused regression reports are `output/harness_review/final/host.xml` and `container.xml`. They cover document coverage, page response parsing, concurrent candidate ordering, evaluation scoring, and skill registration checks. Browser checks cover extraction options, submission cancellation, and three workflow stages at seven resolutions. These are controlled HTTP fixtures, not production extraction jobs.

## Measurement review

The `activity_tables` skill extracts compound measurements, reviews them against the source, revises source-supported errors, and requires a fresh review. It covers omissions, conflicting assignments, unsupported claims, and transcription errors. This atomic skill has not replaced the main pipeline's separate assay reconciliation or visual table reader.

Scientific identity includes identifier namespace, compound, endpoint, target, conditions, unit, and provenance. Display labels do not establish identity. Conditions include stated species and sample matrix. Unknown metadata remains null; a reviewer cannot require facts absent from the source. Printed missingness markers remain records with a null value, without losing the stated column provenance.

Model references contain inclusive line ranges. The host attaches excerpts from the original text, preserving line endings and Unicode spacing. Models select evidence rather than retype it. Reference validity alone does not prove that a scientific claim follows; the review checks support separately.

Each stage allows two contract attempts with validation feedback. The default permits one semantic revision; two are available explicitly. All stages share one monotonic deadline, including retry delays, queueing, SDK timeouts, and response validation. No earlier candidate is returned after a later stage fails. Error details preserve issues, source references, and revision history.

Text generation defaults to 8192 output tokens; this measurement workflow uses 4096. Reaching a provider output limit is terminal even if the returned text looks like valid JSON. Raw output and available usage remain in the audit. Local queue expiry is a capacity error and does not count against provider health. Expired callers cannot open the provider circuit.

```bash
python scripts/check_measurement_review.py --workers 2 --repeats 2
python scripts/check_measurement_review.py --cases evals/literature_measurement_cases.json --repeats 2
python scripts/check_measurement_review.py --rescore output/measurement_review/RUN_DIRECTORY
```

Each invocation creates a UUID run directory. Inputs, references, trials, results, and audits are separate. Rescoring creates a derived run, preserves original input snapshots, and refuses changes to model inputs or incomplete trial sets. Reference-defined wording equivalents apply only to scoring; they do not rewrite model results. Reference denominators retain every original observation.

Twenty controlled cases exercise activity, ADME, provenance, missing values, condition swaps, namespace collisions, conflicting sources, and seeded extraction defects. Separate literature cases use actual PDF excerpts with page links and document hashes: docking RMSD and a 6h/6j ownership conflict. These are excerpt tests, not whole-document accuracy measurements.

Overall case success includes execution failures. Reference agreement is reported separately for completed judgments. All-run consistency requires every repetition of a case to pass; a best run cannot hide failures.

Model clients bind the configured proxy per URL scheme. Empty proxy configuration means direct transport; ambient proxy variables are not inherited. Requests do not mutate process proxy variables or change routes after failures.

## Source questions

Uncertain drafts propose scoped source questions rather than establishing ambiguity.
The host assigns question IDs and retains immutable issue snapshots, response hashes,
and all adjudications. Questions specify affected subjects, endpoint, method, and
reported values. A separate source review must confirm or refute every pending ID
with evidence. Refutation of an uncertain draft requires a complete revision and
another review; it cannot turn an empty draft into an accepted result.

Valid questions survive unrelated contract failures and invalid neighboring questions.
The parser inspects every proposal from a complete JSON response, records independently
valid concerns, then rejects any contract errors. It never salvages malformed JSON.
Contract retries refresh both the question list and request schema. Scientific fields
do not share the question-ID enum. Confirmed conflicts stop the workflow; unresolved
questions cannot disappear through repair or revision.

The reviewer checks typed fields independently of display labels and evidence excerpts.
Both docking trials retained the crystallographic comparison reference, and both MD
trials preserved the unresolved numeric ownership conflict. The four-trial report is
`output/measurement_review/context_verification/469e3bc387354ff19237897dac9c937b`.
Earlier incomplete metadata remains recorded in the derived scoring run
`output/measurement_review/reference_review/45c67b797dc842ba824f1c4ad261876c`.
Missing comparison context remains a failure. These excerpt checks do not establish
general document accuracy or qualify the main extraction pipeline.

The final deadline check also covers audit and circuit bookkeeping before returning
from the shared harness. Audit delay cannot yield a late successful return or count as
a model service failure.

The controlled measurement run completed all twenty cases twice: forty successes,
twenty all-run-consistent cases, and no execution failures. It covers corrections,
omissions, condition swaps, namespace and provenance distinctions, missing values,
source conflicts, and refutation of a seeded false source-conflict claim. Report:
`output/measurement_review/controlled_final/93e811e8134b44e6b363ae0842de8d55/report.json`.
These are controlled cases, not forty independent documents.

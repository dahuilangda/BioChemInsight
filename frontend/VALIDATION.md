# Workbench validation — 2026-11-10

## Automated regression checks

- `npm run build --prefix frontend/ui`: TypeScript and production build passed.
- `python -m pytest -q frontend/tests/test_api_harness.py`: five tests passed. Coverage includes JSON and multipart HTTP contracts, HTTP/domain failures, observation timeout with retained task ID, resuming without another submission, failed-task reporting, and ZIP download.
- `NODE_PATH=/tmp/biochem-ui-qa/node_modules node frontend/tests/workbench.browser.cjs`: fixture-backed layout checks at widths 320, 375, 768, 1280 and 1920; assay values; card 20 lazy-loading; actual JSME assets at six viewport sizes; Molfile header preservation when saving.

Browser fixture checks validate UI behavior, not inference quality.

## Live API checks

The running structure service on port 8011 accepted skill discovery, `variant_sites`, image detection and MolSight recognition. Detection returned one bounding box; recognition returned `ok: true` and nonempty SMILES. These checks prove invocation and response handling, not chemical correctness against a curated reference.

A real image-derived PDF was uploaded through the browser to the workbench on port 8000 and submitted through its Redis dispatcher and Celery worker. Task `029bca190ed947acbe7fce01a3aa5eee` reached the vision page-analysis step, then failed because the configured `gemma4-31b` gateway returned an upstream connection error. The harness report preserves that failed task and the backend error. Successful completion of the full automatic PDF pipeline remains unverified; do not report this as an end-to-end inference pass.

To verify save/export independently, the actual MolSight response was imported into a dedicated task in an isolated test Redis instance. This was an imported recognition artifact, not a completed full-pipeline run. In the browser:

- Changing a compound ID persisted through the real PUT endpoint and a page reload.
- JSME opened with usable Save/Cancel controls at 320×640, 375×812, 768×700, 1280×720, 1920×1080 and 812×375.
- Saving the molecule produced a valid Molfile. RDKit canonical isomeric SMILES confirmed that the saved SMILES and Molfile retained the original prediction's molecular graph and stereochemistry.
- The browser downloaded an intact ZIP; `structures.csv` contained the saved ID and the same molecule.
- Structure/assay page selection, Enter/Space toggling, actual PDF rendering and Jobs were checked at all five widths without document overflow or JavaScript page errors.
- Reopening the failed real task showed a concise error, expandable technical details, and a working dismiss button at width 320.

Local evidence is under `/tmp/biochem-ui-qa/` (workflow report, recognition/detection JSON, saved records, ZIP and screenshots). It is temporary runtime evidence, not a permanent production artifact.

## Persistence and cancellation follow-up

The configured vision endpoint was probed again with one minimal inference request and still returned HTTP 500 with an upstream connection error. No replacement model or fabricated successful pipeline result was used.

Additional implementation and verification:

- Full-pipeline structure results now support the same PUT editing API as standalone structure tasks.
- Saving an empty list writes a readable, empty CSV. Legacy parameter fallbacks no longer resurrect rows from an existing empty CSV.
- Updating full-pipeline structures invalidates its derived `merged.csv`, so exports use the revised data.
- The UI serializes saves, retains unsaved edits after errors, offers an explicit retry, saves before downloading or replacing the current workspace, and ignores stale responses from another task.
- Failed cancellation requests retain task state and polling. Accepted cancellation retains the canceled task's handle.
- `test_workbench_persistence.py`: six FastAPI regression cases passed for standalone and full-pipeline tasks, including last-row deletion, current-data ZIP export, and completed tasks without a stored artifact path.
- `workbench.persistence.browser.cjs`: both task types passed asynchronous save retry, ordered writes, reopening before a pending save completes, export flush, and deletion/reload/export checks; failed cancellation continued polling and then succeeded on retry.

A dedicated full-pipeline-type persistence fixture (`9cbe6aa979b447c993ebab280a2b86e7`) was seeded from the live recognition artifact in the isolated QA store. Through the real browser and backend, editing immediately followed by download produced a valid ZIP with the new ID; deleting the last row, reloading, and downloading produced a valid empty ZIP without old raw records. This fixture verifies editing/export for that task type, not automatic inference completion. Local evidence: `full-saved.zip` and `full-empty.zip` under `/tmp/biochem-ui-qa/`.

Task isolation checks also passed: a delayed poll from a previous job cannot replace the current task; completed metadata cannot enable an empty save/export before rows load; restarting the pipeline clears the old results and sends subsequent saves only to the new task. Save ownership is bound to the task whose rows were actually loaded. Browser and HTTP fixtures test these races without model calls.

The CLI `workflow` command was also exercised directly against the live workbench: it uploaded a one-page PDF, submitted task `dcabc018dc1d4d3cb14caa85d87b65b9`, reached its observation timeout with the task handle preserved in `cli-live-submit.json`, and the same task was then canceled through the real cancellation endpoint. This submission/timeout/cancellation smoke check does not claim successful inference. Final Python regression run: 11 passed across the harness and backend persistence suites; production build and both browser suites passed.

## Unified workspace and resource use

The source, structure selection, bioactivity selection, Jobs and result surfaces now share restrained slate/blue tokens, compact controls and consistent focus states. Results use independent PDF/data scrolling, remembered pointer/keyboard pane resizing, cards/table linked navigation, mobile reading/data switching and a focus mode. Manual page changes clear the selected source region; selection never submits inference implicitly.

Latest verification (2026-11-10):

- Production TypeScript/Vite build passed; 31 text assets precompressed at build time, approximately 4,592 KiB → 1,261 KiB with Brotli. These are aggregate asset sizes, not first-load transfer measurements.
- `workbench.navigation.browser.cjs` passed against both Vite and the production static server with a real PDF.js worker: high-DPI canvas ratio 2; no repaint for same-page record changes; one sharpen pass after a 14-step resize; cached recent-page reuse; one PDF fetch across desktop/mobile switches; zero duplicate page-PNG requests; each selected box JSON fetched once; late metadata cannot override a newer selection. API results in this test are synthetic fixtures.
- `workbench.range.browser.cjs` uses real HTTP Range responses and a synthetic 64-page PDF with contiguous page metadata (6,421,812 bytes). Opening page 1 transferred 195,892 bytes; jumping to page 32 reached 392,500 bytes total through six range requests. The initial full GET was canceled after headers. These figures establish demand-driven reading for this fixture, not a guarantee for every PDF: scattered page metadata can require more ranges.
- `workbench.stages.browser.cjs` passed source/structure/bioactivity surfaces at 320×640, 375×812, 768×900, 1280×800, 1920×1080, 3840×1600 and 812×375: no document overflow; most available height goes to the gallery; Auto/manual selection and keyboard toggles agree; no implicit submissions; Jobs focuses search and Escape returns focus at every stage; a delayed magnification response cannot reopen a dismissed preview.
- `workbench.browser.cjs` passed the production bundle at five result widths, actual JSME at six sizes, image-error retry and lazy loading. A 200-record page requested 16 original images during the exercised scroll path rather than eagerly fetching all 200.
- `test_static_delivery.py`: five real HTTP server tests passed for precompression, conditional 304, cache rules, HEAD, SPA/asset 404 handling, byte ranges, immediate streaming headers, abandoned-download upstream closure and unchanged streamed upload bytes.
- Final Python regression set: 16 cases across HTTP harness, backend persistence and static delivery. Browser persistence races, retry/export/deletion and cancellation were rechecked after the layout change.

A live API review of the imported MolSight artifact confirmed independent panes use 754 of 900 vertical pixels (about 84%) at 1280 and 1920 widths, and mobile has no document overflow. Screenshots are in `/tmp/biochem-ui-qa/unified/` and `/tmp/biochem-ui-qa/workspace-*.png`. Full automatic inference still has the vision endpoint failure recorded above.

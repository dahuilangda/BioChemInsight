# PDF search

The PDF toolbar opens document search. Enter and Shift Enter move between matches; Escape closes it. Ctrl/⌘ F searches the PDF while focus is inside the PDF pane, and results elsewhere.

## Indexing

A nonempty query starts indexing after 250 ms, current page first. Native text is fetched in batches of up to eight pages. Each batch shares an open PDF document. Scanned pages are then processed individually through the existing PaddleOCR service, using temporary single-page PDFs. Native matches highlight words; OCR matches highlight layout blocks. The source PDF is not rewritten.

Closing search or clearing the query aborts the current request and stops scheduling pages. A submitted OCR job may still finish and populate the disk cache. Retries retain successful pages. Index progress and unavailable pages remain visible; incomplete searches never report “No matches”. Input method composition does not trigger match navigation.

## Limits

Page caches are keyed by file size, modification time, index version, OCR endpoint and language. Writes use atomic replacement; invalid cache files are rebuilt. A cross-process file lock permits one search OCR request at a time. Busy requests return 429 with Retry-After. OCR failures are not cached. The OCR submission timeout is 15 seconds; completion wait is 90 seconds.

The browser caches at most 300 pages with an estimated 8 MiB budget, in addition to the current document's search index. Normalized text and spans are retained through weak references to page results. Match lookup walks spans in order and stops at 5,000 occurrences. Query changes are debounced by 150 ms. Only nearby PDF pages mount highlights; moving between matches updates the affected page components.

## API

- `GET /api/pdfs/{id}/search/pages?numbers=1,2,3`: up to eight native text pages; failures are reported individually.
- `GET /api/pdfs/{id}/search/pages/{number}?ocr=true`: permits OCR for a scanned page.

Coordinates are normalized to the displayed page and rounded to six decimal places. Large images with little embedded text identify scanned pages; this heuristic does not classify every possible mixed image/text layout. OCR recognition accuracy depends on the source scan.

## Validation

- `docker compose exec -T web python < frontend/tests/pdf_search_test.py`
- `node frontend/tests/pdf_search_test.cjs`
- `NODE_PATH=/tmp/biochem-ui-qa/node_modules node frontend/tests/workbench.pdf-search.browser.cjs`
- `python frontend/tests/run_workbench_qa.py`

Tests cover cache recovery, partial failures, OCR locking, rotation, finite coordinates, cancellation, keyboard controls, mobile layout, match correctness against a baseline, and 300,000 words. Live validation uses the retained SR task and a synthetic image-only page through the actual OCR service. Search does not write result rows or edit history.

# PDF loading and workspace switching review

The loading unit is the PDF session. The reader, page gallery, and result source previews share that session's bytes and page rasters. RDKit previews are keyed by molecular input and can be reused across jobs.

## Loading order

1. **Upload:** send the file once, distinguish upload progress from server preparation, then use slices of the browser's original `File` for PDF.js range requests. No second download or full-file main-thread `arrayBuffer()` is needed.
2. **Completed job:** fetch source metadata, source PDF data, and results in parallel. Prepare the PDF while resolving the job; wait for its evidence-page intent before painting the reader. Structures, assay observations, scaffold pages, JSON page lists, and selected task pages contribute to that intent.
3. **Reading:** the visible page wins. Nearby reading pages follow, ordered by distance from the viewport. Scrolling and resizing postpone background work.
4. **Results:** copy available page canvases into source previews. One background page is prepared at a time; duplicate consumers share its promise. At idle, prepare at most one next known evidence page, considering only a small lookahead. Save-data mode disables this speculative preparation.
5. **Switching:** retain the current workspace while a different job is prepared. Commit its document and records together; failed or superseded opens cannot replace the current workspace. An already open job does not reload its data.

Downloading bytes and drawing pages are separate. Keeping a PDF session avoids repeated file parsing and range downloads for bytes already held by PDF.js. Page rasters and source clips have explicit memory limits; an evicted raster can be redrawn from that same document when needed. Hundreds of full-resolution canvases are never prepared together.

## Review findings addressed

| Finding | Change |
| --- | --- |
| Uploading a PDF was followed by another server download for the reader. | A `PDFDataRangeTransport` reads local `File.slice()` ranges and transfers those chunks to the worker. |
| Upload completion, server preparation, PDF transfer, and page preparation were conflated. | Separate status text and measured PDF byte progress; no fabricated percentage for patent fetching or page rendering. Short viewports receive a compact waiting state. |
| Restored result pages could arrive after the cover was already painted, causing an avoidable jump. | Defer the initial page paint until restoration resolves; retain an explicit user navigation intent when later metadata arrives. |
| Reader and background-preview caches only reused pages in one direction. | Either consumer can immediately use an existing page raster. A sufficiently sharp raster also avoids a reader re-render during resizing. |
| Gallery navigation could evict small source crops. | Separate source and gallery cache budgets. |
| Background work remained queued after its preview or job disappeared. | Abortable consumers, ref-counted shared requests, cancelable idle waits, and cancellation of unowned PDF loading tasks. Hidden tabs wait for visibility events rather than polling. |
| An old request's cleanup could remove a newer request for the same document. | Request-identity checks govern cleanup and progress publication. |
| Opening a job cleared the old view before the replacement was ready. | Parallel fetches followed by one guarded commit; current content is retained under a delayed, stationary waiting layer. |
| Late URL-restoration responses could overwrite a manually opened job. | Restoration and explicit opens share a generation guard. |
| Molecular previews were discarded on every job switch. | Reuse successful input-keyed renders, capped at 128 entries. |
| A legacy page-PNG download and image-crop pipeline remained beside the PDF pipeline. | Remove its queues, bitmap state, crop state, and duplicate loading indicators. Explicit legacy image artifacts remain a fallback when no usable PDF source exists. |
| A late upload-progress timer could affect the next import, and concurrent imports could race. | Remove the delayed reset and serialize source imports through one guard. |
| Small surface translations made otherwise stable views appear to move. | Use opacity-only transitions; suppress brief job-opening indicators with a short display delay. Reduced-motion settings remain supported. |
| Programmatic source positioning was mistaken for manual scrolling, clearing selection near page boundaries. | Track the commanded position separately; use the viewport center for manual page detection. Manual navigation invalidates pending source coordinates. |
| Hover enlargement clipped image borders inside table cells. | Keep image bounds fixed and draw focus and hover borders inside the cell. |
| Pointer selection moved both the PDF and the result list. | Preserve the list position on pointer selection; reveal rows for keyboard navigation and view changes. |
| Every PDF page and preview created its own visibility observer. | Pool observers by scroll root and margin. Memoized card components and stable callbacks avoid unrelated card updates. |

## Resource limits

- At most two cached parsed documents; inactive documents are also evicted when their known file sizes exceed 64 MiB in total.
- Reader raster cache: at most three rasters and 12 million pixels; each raster is capped at 8 million pixels.
- Background page cache: 6 million pixels; each page is capped at 2 million pixels.
- Source clip cache: 3 million pixels. Gallery thumbnail cache: 1 million pixels.
- Only nearby display canvases retain their buffers. Background pages are serialized and yield to the reader.

These budgets describe application-owned caches, rather than total PDF.js worker/browser memory. The currently open document stays pinned so normal reading and same-document job switches keep reusing it. A preload lease is handed to the reader only after the reader mounts, preventing a cancellation/re-download gap during commit.

## Verification

Browser checks use the actual PDF.js worker and valid PDF fixtures:

- `workbench.loading.browser.cjs`: local upload range reuse and evidence-first completed-job restoration.
- `workbench.switching.browser.cjs`: stable geometry during opening, parallel requests, rapid B/C switching, failed-open recovery, shared PDF and RDKit reuse.
- `workbench.pdf-priority.browser.cjs`: source copies without a new PDF render, canceled background requests, collapse/resize reuse, rapid jumps, and zoom position.
- `workbench.navigation.browser.cjs`: source links, high-DPI reading, manual region recognition, result editing, and mobile navigation.
- `workbench.review.browser.cjs`: page-edge selection, native wheel/touch/keyboard scrolling, delayed coordinates, input focus, source context during editing, image borders, independent scaffold/fragment links, and drawing-layer cleanup.
- `workbench.range.browser.cjs`: actual HTTP partial-content requests and limited transfer for distant-page navigation.
- `workbench.layout.browser.cjs`, `workbench.stages.browser.cjs`, `workbench.experience.browser.cjs`: long documents, desktop/phone/landscape layout, source pixels, bounded offscreen buffers, keyboard interactions, and reduced motion.
- `workbench.persistence.browser.cjs`: save, restoration, cross-job edit isolation, and restart/cancel behavior.

Read-only checks against an existing completed job are also used to inspect actual chemical source crops and RDKit images in the built frontend.

Recorded checks from this review:

- A 6,421,812-byte uploaded fixture opened through three local range reads totaling 392,500 bytes; server PDF downloads: **0**.
- A completed structure job landed on page **37**; an activity job linked to a pipeline landed on its activity evidence page **45**, rather than the linked structure page 37.
- Same-page source copies caused **0** additional PDF.js renders. Shrinking the splitter reused the existing high-DPI raster with **0** additional reader page renders.
- **20** abandoned preview requests rejected with `AbortError`.
- Rapid B/C job switching retained geometry and ignored B's late result. A failed subsequent open preserved C. Same-document jobs used one shared PDF download and identical molecules used one RDKit batch.
- A 6.4 MB remote range fixture needed three range requests, with 261,428 bytes transferred by first-page readiness and 392,500 bytes after a jump to page 32.
- Real completed-job checks observed **0** page-image requests, **0** source-crop image requests, and no page errors or mobile document overflow. The built frontend is served on the existing application port.
- The 500-page layout fixture used **2** intersection observers. Navigation preserved the PDF mount, page raster, zoom, and scroll position.

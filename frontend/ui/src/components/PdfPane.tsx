import React from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export interface PdfTarget {
  page: number;
  /** normalized fractions (0-1, x-first) of the highlight rect */
  rect?: { x: number; y: number; w: number; h: number };
}

interface PdfPaneProps {
  pdfId: string;
  target: PdfTarget | null;
}

const RENDER_SCALE = 2;
const NEIGHBOUR_PAGES = 1;
const PDF_DATA_CACHE_LIMIT = 3;
const PDF_DOC_CACHE_LIMIT = 2;
const ZOOM_STEP = 0.25;
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 2;

/**
 * Raw file bytes per PDF. Toggling Table<->PDF (or jumping between rows)
 * remounts this pane; the cache guarantees the document is downloaded once.
 * pdf.js detaches the ArrayBuffer it receives, so the cache keeps the master
 * copy and every mount hands pdf.js its own copy.
 */
const pdfDataCache = new Map<string, ArrayBuffer>();

/** Parsed documents per PDF so re-opening a pane skips download and parse. */
const pdfDocCache = new Map<string, pdfjsLib.PDFDocumentProxy>();

function rememberData(pdfId: string, buffer: ArrayBuffer): void {
  pdfDataCache.delete(pdfId);
  pdfDataCache.set(pdfId, buffer);
  while (pdfDataCache.size > PDF_DATA_CACHE_LIMIT) {
    const oldest = pdfDataCache.keys().next().value;
    if (oldest === undefined) break;
    pdfDataCache.delete(oldest);
  }
}

function rememberDoc(pdfId: string, doc: pdfjsLib.PDFDocumentProxy): void {
  pdfDocCache.delete(pdfId);
  pdfDocCache.set(pdfId, doc);
  while (pdfDocCache.size > PDF_DOC_CACHE_LIMIT) {
    const oldest = pdfDocCache.keys().next().value;
    if (oldest === undefined || oldest === pdfId) break;
    const evicted = pdfDocCache.get(oldest);
    pdfDocCache.delete(oldest);
    void evicted?.destroy().catch(() => undefined);
  }
}

/**
 * Left-hand PDF viewer for the review screen. Renders a window of pages
 * around the target and overlays the highlighted structure box. Coordinates
 * arrive as fractions of the page, so raster and PDF DPI mismatch is
 * irrelevant.
 */
export default function PdfPane({ pdfId, target }: PdfPaneProps) {
  const [doc, setDoc] = React.useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [error, setError] = React.useState('');
  const pageRefs = React.useRef<Map<number, HTMLDivElement>>(new Map());
  const canvasRefs = React.useRef<Map<number, HTMLCanvasElement>>(new Map());
  const renderTasks = React.useRef<Map<number, pdfjsLib.RenderTask>>(new Map());
  const renderJobs = React.useRef<Map<number, Promise<unknown>>>(new Map());
  const paneRef = React.useRef<HTMLDivElement | null>(null);
  const [pageCount, setPageCount] = React.useState(0);
  const [basePageWidth, setBasePageWidth] = React.useState(0);
  /** multiplier on RENDER_SCALE; 1 = default sharpness, "Fit width" normalizes it */
  const [zoom, setZoom] = React.useState(1);

  const scrollToFocus = React.useCallback(
    (focus: number) => {
      const holder = pageRefs.current.get(focus);
      if (holder) holder.scrollIntoView({ behavior: 'auto', block: 'start' });
    },
    [],
  );

  React.useEffect(() => {
    let cancelled = false;
    const cachedDoc = pdfDocCache.get(pdfId);
    if (cachedDoc) {
      setDoc(cachedDoc);
      setPageCount(cachedDoc.numPages);
      setError('');
      return () => { cancelled = true; };
    }
    setDoc(null);
    setPageCount(0);
    setError('');

    const load = async () => {
      try {
        let buffer = pdfDataCache.get(pdfId);
        if (!buffer) {
          const response = await fetch(`/api/pdfs/${encodeURIComponent(pdfId)}/file`);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          buffer = await response.arrayBuffer();
          if (cancelled) return;
          rememberData(pdfId, buffer);
        } else if (cancelled) {
          return;
        }
        const parsed = await pdfjsLib.getDocument({ data: buffer.slice(0) }).promise;
        if (cancelled) {
          void parsed.destroy().catch(() => undefined);
          return;
        }
        rememberDoc(pdfId, parsed);
        setDoc(parsed);
        setPageCount(parsed.numPages);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [pdfId]);

  // width of a page at scale 1, used by "fit width"
  React.useEffect(() => {
    if (!doc) return;
    let cancelled = false;
    void doc.getPage(1).then((p) => {
      if (cancelled) return;
      setBasePageWidth(p.getViewport({ scale: 1 }).width);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [doc]);

  const invalidateRenders = React.useCallback(() => {
    renderTasks.current.forEach((task) => task.cancel());
    renderTasks.current.clear();
    renderJobs.current.clear();
    canvasRefs.current.forEach((canvas) => {
      const context = canvas.getContext('2d');
      context?.clearRect(0, 0, canvas.width, canvas.height);
    });
  }, []);

  const renderPage = React.useCallback((page: number) => {
    if (!doc || page < 1 || page > doc.numPages) return Promise.resolve();
    const running = renderJobs.current.get(page);
    if (running) return running;
    const job = doc.getPage(page)
      .then((p) => {
        const canvas = canvasRefs.current.get(page);
        if (!canvas) return;
        const viewport = p.getViewport({ scale: RENDER_SCALE * zoom });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        // keep the raster at its natural size so zooming in stays sharp and
        // the page scrolls horizontally instead of blurring
        canvas.style.width = zoom === 1 ? '' : `${viewport.width}px`;
        const task = p.render({ canvasContext: canvas.getContext('2d')!, viewport });
        renderTasks.current.set(page, task);
        return task.promise;
      })
      .finally(() => {
        renderJobs.current.delete(page);
        renderTasks.current.delete(page);
      })
      .catch(() => undefined);
    renderJobs.current.set(page, job);
    return job;
  }, [doc, zoom]);

  // re-render the window when the zoom level changes
  React.useEffect(() => {
    if (!doc) return;
    invalidateRenders();
    const focus = Math.min(Math.max(target?.page ?? 1, 1), doc.numPages);
    const ordered: number[] = [];
    for (let offset = 0; offset <= NEIGHBOUR_PAGES + 6; offset += 1) {
      if (focus - offset >= 1) ordered.push(focus - offset);
      if (offset > 0 && focus + offset <= doc.numPages) ordered.push(focus + offset);
    }
    ordered.forEach((p) => void renderPage(p));
  }, [doc, zoom, invalidateRenders, renderPage, target]);

  // one window definition drives both the mounted DOM pages and the
  // rendered canvases — mounting a page without rendering it leaves a
  // permanently blank canvas
  const pages = React.useMemo(() => {
    if (!doc) return [];
    const focus = Math.min(Math.max(target?.page ?? 1, 1), doc.numPages);
    const list: number[] = [];
    for (let p = Math.max(1, focus - NEIGHBOUR_PAGES - 2);
         p <= Math.min(doc.numPages, focus + NEIGHBOUR_PAGES + 6); p++) list.push(p);
    return list;
  }, [doc, target]);

  // render the focus page first, then the rest of the window in pass order
  React.useEffect(() => {
    if (!doc || pages.length === 0) return;
    const focus = Math.min(Math.max(target?.page ?? 1, 1), doc.numPages);
    const ordered = [...pages].sort(
      (a, b) => Math.abs(a - focus) - Math.abs(b - focus) || a - b,
    );
    ordered.forEach((p) => void renderPage(p));
  }, [pages, renderPage, doc, target]);

  // scroll to the target once it has actually been painted, then re-align
  // after the whole window settles (earlier pages changing height would
  // otherwise leave the highlight out of view)
  React.useEffect(() => {
    if (!doc || !target) return;
    let cancelled = false;
    const focus = Math.min(Math.max(target.page, 1), doc.numPages);
    void renderPage(focus).then(() => {
      if (!cancelled) scrollToFocus(focus);
    });
    void Promise.all(pages.map((p) => renderPage(p))).then(() => {
      if (!cancelled) scrollToFocus(focus);
    });
    return () => { cancelled = true; };
  }, [doc, target, pages, renderPage, scrollToFocus]);

  // release per-page raster memory when the pane goes away; the parsed
  // document itself stays cached for the next mount
  React.useEffect(() => {
    return () => {
      renderTasks.current.forEach((task) => task.cancel());
      renderTasks.current.clear();
      void doc?.cleanup().catch(() => undefined);
    };
  }, [doc]);

  const fitWidth = React.useCallback(() => {
    const paneWidth = paneRef.current?.clientWidth ?? 0;
    if (!basePageWidth || !paneWidth) return;
    const targetScale = (paneWidth - 28) / basePageWidth;
    const multiplier = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, targetScale / RENDER_SCALE));
    setZoom(Math.round(multiplier * 100) / 100);
  }, [basePageWidth]);

  const zoomPercent = Math.round(RENDER_SCALE * zoom * 100);

  if (error) {
    return <div className="pdf-pane pdf-pane--error">PDF failed to load: {error}</div>;
  }
  if (!doc) {
    return <div className="pdf-pane pdf-pane--loading">Loading PDF…</div>;
  }
  const focus = Math.min(Math.max(target?.page ?? 1, 1), pageCount);
  const scrollable = zoom !== 1;

  return (
    <div className="pdf-pane" ref={paneRef}>
      <div className="pdf-pane__toolbar">
        <button
          type="button"
          className="pdf-pane__zoom-btn"
          onClick={() => setZoom((z) => Math.max(ZOOM_MIN, Math.round((z - ZOOM_STEP) * 100) / 100))}
          aria-label="Zoom out"
          title="Zoom out"
        >
          −
        </button>
        <span className="pdf-pane__zoom-level" aria-live="polite">{zoomPercent}%</span>
        <button
          type="button"
          className="pdf-pane__zoom-btn"
          onClick={() => setZoom((z) => Math.min(ZOOM_MAX, Math.round((z + ZOOM_STEP) * 100) / 100))}
          aria-label="Zoom in"
          title="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          className="pdf-pane__zoom-btn pdf-pane__zoom-btn--fit"
          onClick={fitWidth}
          aria-label="Fit page width"
          title="Fit page width"
        >
          Fit
        </button>
      </div>
      <div className="pdf-pane__pages">
        {pages.map((p) => (
          <div className="pdf-pane__page" key={p}
               style={scrollable ? { overflowX: 'auto' } : undefined}
               ref={(el) => { if (el) pageRefs.current.set(p, el); else pageRefs.current.delete(p); }}>
            <div className="pdf-pane__pageno">{p} / {pageCount}</div>
            <div className="pdf-pane__canvas-wrap">
              <canvas ref={(el) => { if (el) canvasRefs.current.set(p, el); else canvasRefs.current.delete(p); }} />
              {p === focus && target?.rect && (
                <span className="pdf-pane__highlight" style={{
                  left: `${target.rect.x * 100}%`,
                  top: `${target.rect.y * 100}%`,
                  width: `${target.rect.w * 100}%`,
                  height: `${target.rect.h * 100}%`,
                }} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

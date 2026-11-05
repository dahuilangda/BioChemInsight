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

const RENDER_SCALE = 1.6;
const NEIGHBOUR_PAGES = 1;

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
  const renderJobs = React.useRef<Map<number, Promise<unknown>>>(new Map());
  const [pageCount, setPageCount] = React.useState(0);

  React.useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError('');
    pdfjsLib.getDocument(`/api/pdfs/${encodeURIComponent(pdfId)}/file`)
      .promise.then((d) => { if (!cancelled) { setDoc(d); setPageCount(d.numPages); } })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [pdfId]);

  const renderPage = React.useCallback((page: number) => {
    if (!doc || page < 1 || page > doc.numPages) return Promise.resolve();
    const existing = renderJobs.current.get(page);
    if (existing) return existing;
    const job = doc.getPage(page)
      .then((p) => {
        const canvas = canvasRefs.current.get(page);
        if (!canvas) return;
        const viewport = p.getViewport({ scale: RENDER_SCALE });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        return p.render({ canvasContext: canvas.getContext('2d')!, viewport }).promise;
      })
      .finally(() => renderJobs.current.delete(page));
    renderJobs.current.set(page, job);
    return job;
  }, [doc]);

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

  React.useEffect(() => {
    pages.forEach((p) => void renderPage(p));
  }, [pages, renderPage]);

  // scroll to the target and reveal the highlight
  React.useEffect(() => {
    if (!doc || !target) return;
    const focus = Math.min(Math.max(target.page, 1), doc.numPages);
    const holder = pageRefs.current.get(focus);
    if (holder) holder.scrollIntoView({ behavior: 'smooth', block: 'start' });
    void renderPage(focus);
  }, [doc, target, renderPage]);

  if (error) {
    return <div className="pdf-pane pdf-pane--error">PDF failed to load: {error}</div>;
  }
  if (!doc) {
    return <div className="pdf-pane pdf-pane--loading">Loading PDF…</div>;
  }
  const focus = Math.min(Math.max(target?.page ?? 1, 1), pageCount);

  return (
    <div className="pdf-pane">
      <div className="pdf-pane__pages">
        {pages.map((p) => (
          <div className="pdf-pane__page" key={p}
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

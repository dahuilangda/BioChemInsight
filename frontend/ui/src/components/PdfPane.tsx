import React from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { getPdfFileUrl, getPdfPreviewUrl } from '../api/client';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export interface PdfTarget {
  page: number;
  requestId?: number;
  rect?: { x: number; y: number; w: number; h: number };
}
type Region = NonNullable<PdfTarget['rect']>;
interface PdfPaneProps { selectedCompoundLabel?: string; pdfId: string; target: PdfTarget | null; onPageChange?: (page: number) => void; onRecognizeRegion?: (page: number, rect: Region, purpose?: 'structure' | 'table' | 'fragment', replaceSelected?: boolean) => Promise<void> }

// Nearby pages are painted on demand. Same-page selections move the
// highlight without reinterpreting the PDF or repainting its canvas.
const documents = new Map<string, pdfjsLib.PDFDocumentProxy>();
const loadingDocuments = new Map<string, Promise<pdfjsLib.PDFDocumentProxy>>();
const documentUsers = new Map<string, number>();
interface Raster { canvas: HTMLCanvasElement; pixels: number; pdfId: string }
const rasters = new Map<string, Raster>();
const MAX_RASTER_PIXELS = 8_000_000;
const MAX_CACHED_PIXELS = 12_000_000;

function evictDocuments() {
  for (const [id, document] of documents) {
    if (documents.size <= 2) break;
    if (documentUsers.get(id)) continue;
    documents.delete(id);
    void document.destroy().catch(() => undefined);
    for (const [key, raster] of rasters) {
      if (raster.pdfId !== id) continue;
      raster.canvas.width = raster.canvas.height = 0;
      rasters.delete(key);
    }
  }
}
function loadDocument(pdfId: string): Promise<pdfjsLib.PDFDocumentProxy> {
  const existing = documents.get(pdfId);
  if (existing) {
    documents.delete(pdfId); documents.set(pdfId, existing);
    return Promise.resolve(existing);
  }
  const pending = loadingDocuments.get(pdfId);
  if (pending) return pending;
  // Let PDF.js request the ranges needed for the current page instead of
  // downloading every page of a long patent before the user can read it.
  const task = pdfjsLib.getDocument({
    url: getPdfFileUrl(pdfId), disableAutoFetch: true, disableStream: true,
    rangeChunkSize: 64 * 1024,
  });
  const promise = task.promise.then((document) => {
    documents.set(pdfId, document); evictDocuments(); return document;
  }).finally(() => { loadingDocuments.delete(pdfId); });
  loadingDocuments.set(pdfId, promise);
  return promise;
}
function rememberRaster(key: string, raster: Raster) {
  rasters.set(key, raster);
  let pixels = [...rasters.values()].reduce((sum, item) => sum + item.pixels, 0);
  for (const [oldKey, item] of rasters) {
    if (rasters.size <= 3 && pixels <= MAX_CACHED_PIXELS) break;
    if (oldKey === key) continue;
    rasters.delete(oldKey); pixels -= item.pixels;
    item.canvas.width = item.canvas.height = 0;
  }
}

interface PageProps {
  document: pdfjsLib.PDFDocumentProxy;
  pdfId: string;
  number: number;
  width: number;
  zoom: number | 'fit';
  root: React.RefObject<HTMLDivElement>;
  onSize: (number: number, width: number, height: number) => void;
  onPaint: () => void;
  onRegister: (number: number, element: HTMLDivElement | null) => void;
  estimatedSize: { width: number; height: number };
  rect?: Region;
  drawing: boolean;
  recognizing: boolean;
  draft?: Region;
  onDraft: (number: number, rect: Region | null) => void;
}

function ContinuousPage({ document, pdfId, number, width, zoom, root, onSize, onPaint, onRegister,
  estimatedSize, rect, drawing, recognizing, draft, onDraft }: PageProps) {
  const element = React.useRef<HTMLDivElement>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const [nearby, setNearby] = React.useState(false);
  const [page, setPage] = React.useState<pdfjsLib.PDFPageProxy | null>(null);
  const [painted, setPainted] = React.useState(false);
  const [error, setError] = React.useState('');
  const [retry, setRetry] = React.useState(0);
  const startPoint = React.useRef<{ x: number; y: number } | null>(null);
  const scale = zoom === 'fit' ? Math.max(.1, (width - 24) / estimatedSize.width) : zoom;
  const paperWidth = estimatedSize.width * scale;
  const paperHeight = estimatedSize.height * scale;

  React.useEffect(() => {
    onRegister(number, element.current);
    const observer = new IntersectionObserver(([entry]) => setNearby(entry.isIntersecting), { root: root.current, rootMargin: '450px 0px' });
    if (element.current) observer.observe(element.current);
    return () => { observer.disconnect(); onRegister(number, null); };
  }, [number, onRegister, root]);
  React.useEffect(() => {
    if (!nearby || page) return;
    let canceled = false;
    void document.getPage(number).then((value) => {
      if (canceled) return;
      const size = value.getViewport({ scale: 1 });
      onSize(number, size.width, size.height); setPage(value); setError('');
    }).catch(() => { if (!canceled) setError('Page unavailable'); });
    return () => { canceled = true; };
  }, [document, number, nearby, page, retry, onSize]);
  React.useEffect(() => {
    if (!nearby || !page || !canvas.current || !width) return;
    let canceled = false;
    const cssViewport = page.getViewport({ scale });
    const outputScale = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(MAX_RASTER_PIXELS / (cssViewport.width * cssViewport.height)));
    const viewport = page.getViewport({ scale: scale * outputScale });
    const key = `${pdfId}|${number}|${Math.ceil(viewport.width)}x${Math.ceil(viewport.height)}`;
    const paint = (raster: Raster) => {
      if (canceled || !canvas.current) return;
      canvas.current.width = raster.canvas.width; canvas.current.height = raster.canvas.height;
      canvas.current.getContext('2d', { alpha: false })!.drawImage(raster.canvas, 0, 0);
      setPainted(true); setError('');
    };
    const cached = rasters.get(key);
    if (cached) { rasters.delete(key); rasters.set(key, cached); paint(cached); return; }
    const offscreen = window.document.createElement('canvas');
    offscreen.width = Math.ceil(viewport.width); offscreen.height = Math.ceil(viewport.height);
    const task = page.render({ canvasContext: offscreen.getContext('2d', { alpha: false })!, viewport });
    void task.promise.then(() => {
      if (canceled) { offscreen.width = offscreen.height = 0; return; }
      const raster = { canvas: offscreen, pixels: offscreen.width * offscreen.height, pdfId };
      rememberRaster(key, raster); paint(raster); onPaint();
    }).catch((cause: { name?: string }) => {
      offscreen.width = offscreen.height = 0;
      if (!canceled && cause.name !== 'RenderingCancelledException') setError('Page unavailable');
    });
    return () => { canceled = true; task.cancel(); };
  }, [page, nearby, scale, width, pdfId, number, retry, onPaint]);
  // Release off-screen display buffers; recent pages remain in the bounded cache.
  React.useEffect(() => {
    if (!nearby && canvas.current) { canvas.current.width = canvas.current.height = 0; setPainted(false); }
  }, [nearby]);
  const point = (event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)) };
  };
  return <div ref={element} className="pdf-page" data-pdf-page={number}>
    <div className="pdf-page__label">Page {number}</div>
    <div className="pdf-pane__paper" style={{ width: paperWidth, height: paperHeight }}>
      {nearby && !painted && <img className="pdf-pane__raster-preview" src={getPdfPreviewUrl(pdfId, number)} alt={`PDF page ${number} preview`} />}
      <canvas ref={canvas} style={{ width: paperWidth, height: paperHeight }} aria-label={`PDF page ${number}`} />
      {error && <div className="pdf-pane__message" role="status">{error}<button type="button" onClick={() => { setError(''); setRetry((value) => value + 1); }}>Retry</button></div>}
      {rect && <span className="pdf-pane__highlight" aria-label="Selected source region" style={{ left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` }} />}
      {(drawing || draft) && <div className="pdf-region-layer" style={{ touchAction: drawing ? 'none' : undefined }}
        onPointerDown={(event) => { if (!drawing || recognizing || event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); startPoint.current = point(event); onDraft(number, null); }}
        onPointerMove={(event) => { const start = startPoint.current; if (!start || recognizing) return; const end = point(event); onDraft(number, { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), w: Math.abs(end.x - start.x), h: Math.abs(end.y - start.y) }); }}
        onPointerUp={() => { startPoint.current = null; }} onPointerCancel={() => { startPoint.current = null; onDraft(number, null); }}>
        {draft && <span className="pdf-region-box" style={{ left: `${draft.x * 100}%`, top: `${draft.y * 100}%`, width: `${draft.w * 100}%`, height: `${draft.h * 100}%` }} />}
      </div>}
    </div>
  </div>;
}

export default function PdfPane({ pdfId, target, onPageChange, onRecognizeRegion, selectedCompoundLabel }: PdfPaneProps) {
  const [document, setDocument] = React.useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [error, setError] = React.useState('');
  const [retry, setRetry] = React.useState(0);
  const [pageNumber, setPageNumber] = React.useState(target?.page ?? 1);
  const currentPageRef = React.useRef(pageNumber);
  const [pageDraft, setPageDraft] = React.useState(String(pageNumber));
  const [sizes, setSizes] = React.useState<Record<number, { width: number; height: number }>>({});
  const [width, setWidth] = React.useState(0);
  const [zoom, setZoom] = React.useState<number | 'fit'>('fit');
  const [renderCount, setRenderCount] = React.useState(0);
  const [drawing, setDrawing] = React.useState(false);
  const [selectionPurpose, setSelectionPurpose] = React.useState<'structure' | 'table' | 'fragment'>('structure');
  const [replaceSelected, setReplaceSelected] = React.useState(false);
  const [draft, setDraft] = React.useState<{ page: number; rect: Region | null } | null>(null);
  const [recognizing, setRecognizing] = React.useState(false);
  const [regionError, setRegionError] = React.useState('');
  const viewportRef = React.useRef<HTMLDivElement>(null);
  const pagesRef = React.useRef(new Map<number, HTMLDivElement>());
  const navigationRef = React.useRef('');
  const viewportMetricsRef = React.useRef({ width: 0, height: 0 });
  const resizeAnchorRef = React.useRef<{ page: number; fraction: number } | null>(null);
  const scrollTimer = React.useRef<number>();
  const pageChangeRef = React.useRef(onPageChange); pageChangeRef.current = onPageChange;
  const onRegister = React.useCallback((number: number, element: HTMLDivElement | null) => {
    if (element) pagesRef.current.set(number, element); else pagesRef.current.delete(number);
  }, []);
  const onSize = React.useCallback((number: number, w: number, h: number) => {
    setSizes((previous) => previous[number]?.width === w && previous[number]?.height === h ? previous : { ...previous, [number]: { width: w, height: h } });
  }, []);
  const onPaint = React.useCallback(() => setRenderCount((value) => value + 1), []);
  const onDraft = React.useCallback((page: number, rect: Region | null) => setDraft({ page, rect }), []);

  React.useEffect(() => {
    let canceled = false;
    documentUsers.set(pdfId, (documentUsers.get(pdfId) ?? 0) + 1);
    setDocument(null); setSizes({}); setError(''); setZoom('fit'); setDraft(null); setDrawing(false);
    navigationRef.current = ''; currentPageRef.current = target?.page ?? 1; setPageNumber(currentPageRef.current);
    void loadDocument(pdfId).then(async (value) => {
      const page = await value.getPage(Math.min(value.numPages, Math.max(1, target?.page ?? 1)));
      if (canceled) return;
      const size = page.getViewport({ scale: 1 });
      setSizes({ 0: { width: size.width, height: size.height }, [page.pageNumber]: { width: size.width, height: size.height } });
      setDocument(value);
    }).catch(() => { if (!canceled) setError('PDF unavailable'); });
    return () => { canceled = true; documentUsers.set(pdfId, Math.max(0, (documentUsers.get(pdfId) ?? 1) - 1)); evictDocuments(); };
  }, [pdfId, retry]);
  React.useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    let timer: number | undefined;
    const measure = () => {
      const previous = viewportMetricsRef.current;
      if (previous.width && previous.width !== element.clientWidth) {
        // Keep the same source point at the viewport centre while fit width
        // changes. Retaining an absolute scrollTop jumps several PDF pages.
        const centre = element.scrollTop + previous.height / 2;
        let anchor: { page: number; fraction: number } | null = null;
        for (const [number, node] of pagesRef.current) {
          if (node.offsetTop > centre) break;
          anchor = { page: number, fraction: Math.max(0, Math.min(1, (centre - node.offsetTop) / Math.max(1, node.offsetHeight))) };
        }
        resizeAnchorRef.current = anchor;
        window.clearTimeout(scrollTimer.current);
      }
      viewportMetricsRef.current = { width: element.clientWidth, height: element.clientHeight };
      element.style.setProperty('--pdf-fit-width', `${Math.max(1, element.clientWidth - 24)}px`);
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => setWidth(element.clientWidth), 120);
    };
    measure(); setWidth(element.clientWidth);
    const observer = new ResizeObserver(measure); observer.observe(element);
    return () => { observer.disconnect(); if (timer) window.clearTimeout(timer); };
  }, []);
  React.useLayoutEffect(() => {
    const anchor = resizeAnchorRef.current, viewport = viewportRef.current;
    if (!anchor || !viewport) return;
    const node = pagesRef.current.get(anchor.page);
    if (node) viewport.scrollTop = Math.max(0, node.offsetTop + anchor.fraction * node.offsetHeight - viewport.clientHeight / 2);
    resizeAnchorRef.current = null;
  }, [width]);
  React.useEffect(() => { setPageDraft(String(pageNumber)); }, [pageNumber]);
  React.useEffect(() => () => window.clearTimeout(scrollTimer.current), []);

  const jump = React.useCallback((page: number, rect?: Region) => {
    const element = pagesRef.current.get(page), viewport = viewportRef.current;
    if (!element || !viewport) return;
    const paper = element.querySelector<HTMLElement>('.pdf-pane__paper');
    const offset = element.offsetTop;
    const top = rect && paper ? offset + paper.offsetTop + (rect.y + rect.h / 2) * paper.offsetHeight - viewport.clientHeight / 2 : offset;
    viewport.scrollTo({ top: Math.max(0, top), left: 0, behavior: 'auto' });
    currentPageRef.current = page; setPageNumber(page);
  }, []);
  React.useEffect(() => {
    if (!document || !target) return;
    const page = Math.min(document.numPages, Math.max(1, target.page));
    const key = `${pdfId}|${page}|${target.requestId ?? ''}|${JSON.stringify(target.rect ?? null)}`;
    if (navigationRef.current === key) return;
    navigationRef.current = key;
    if (page === currentPageRef.current && !target.rect && !target.requestId && viewportRef.current?.scrollTop) return;
    jump(page, target.rect);
  }, [document, pdfId, target?.page, target?.requestId, target?.rect?.x, target?.rect?.y, target?.rect?.w, target?.rect?.h, jump]);
  const onScroll = () => {
    window.clearTimeout(scrollTimer.current);
    scrollTimer.current = window.setTimeout(() => {
      const viewport = viewportRef.current;
      if (!viewport) return;
      const anchor = viewport.getBoundingClientRect().top + Math.min(100, viewport.clientHeight / 3);
      let current = 1;
      for (const [number, element] of pagesRef.current) {
        if (element.getBoundingClientRect().top <= anchor) current = number;
        else break;
      }
      if (current !== currentPageRef.current) {
        currentPageRef.current = current; setPageNumber(current);
        pageChangeRef.current?.(current);
      }
    }, 120);
  };
  const changePage = (next: number) => {
    if (!document || !Number.isFinite(next) || recognizing) return;
    const page = Math.min(document.numPages, Math.max(1, Math.round(next)));
    jump(page); onPageChange?.(page); setDraft(null); setDrawing(false); setRegionError('');
  };
  const commitPage = () => pageDraft.trim() ? changePage(Number(pageDraft)) : setPageDraft(String(pageNumber));
  const recognize = async () => {
    if (!draft?.rect || !onRecognizeRegion || recognizing) return;
    setRecognizing(true); setRegionError('');
    try { await onRecognizeRegion(draft.page, draft.rect, selectionPurpose, selectionPurpose === 'structure' && replaceSelected && Boolean(selectedCompoundLabel)); setDraft(null); setDrawing(false); }
    catch (cause) { setRegionError(cause instanceof Error ? cause.message : 'Recognition failed. Try again.'); }
    finally { setRecognizing(false); }
  };
  const size = sizes[pageNumber] ?? sizes[0] ?? { width: 612, height: 792 };
  const scale = zoom === 'fit' ? Math.max(.1, (width - 24) / size.width) : zoom;
  return <div className="pdf-pane" data-page={pageNumber} data-fit={zoom === 'fit'} data-render-count={renderCount} aria-label="Source PDF">
    <div className="pdf-pane__toolbar">
      <div className="pdf-pane__navigation">
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Previous PDF page" disabled={!document || recognizing || pageNumber <= 1} onClick={() => changePage(pageNumber - 1)}>‹</button>
        <input aria-label="PDF page" inputMode="numeric" value={pageDraft} disabled={!document || recognizing} onChange={(event) => setPageDraft(event.target.value)} onBlur={commitPage}
          onKeyDown={(event) => { if (event.key === 'Enter') { commitPage(); event.currentTarget.blur(); } else if (event.key === 'Escape') { event.stopPropagation(); setPageDraft(String(pageNumber)); } }} />
        <span className="pdf-pane__page-count">/ {document?.numPages ?? '—'}</span>
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Next PDF page" disabled={!document || recognizing || pageNumber >= document.numPages} onClick={() => changePage(pageNumber + 1)}>›</button>
      </div>
      <div className="pdf-pane__zoom-controls">
        {onRecognizeRegion && <button type="button" className="pdf-pane__zoom-btn" aria-pressed={drawing} disabled={!document || recognizing} title="Select a PDF region to extract a structure, scaffold, fragment or result table" onClick={() => { setDrawing(!drawing); setDraft(null); setRegionError(''); }}>Draw box</button>}
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Zoom out" disabled={!document || scale <= .25} onClick={() => setZoom(Math.max(.25, scale - .2))}>−</button>
        <span className="pdf-pane__zoom-level">{Math.round(scale * 100)}%</span>
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Zoom in" disabled={!document || scale >= 4} onClick={() => setZoom(Math.min(4, scale + .2))}>+</button>
        <button type="button" className={`pdf-pane__zoom-btn pdf-pane__zoom-btn--fit${zoom === 'fit' ? ' is-active' : ''}`} aria-label="Fit page width" onClick={() => setZoom('fit')}>Fit</button>
        <a className="pdf-pane__zoom-btn pdf-pane__open" href={getPdfFileUrl(pdfId)} target="_blank" rel="noreferrer" aria-label="Open original PDF" title="Open original PDF">↗</a>
      </div>
    </div>
    {(drawing || draft || regionError || recognizing) && <div className="pdf-region-actions" role="status">
      <select aria-label="Selection purpose" value={selectionPurpose} disabled={recognizing} onChange={(event) => setSelectionPurpose(event.target.value as 'structure' | 'table' | 'fragment')}><option value="structure">Structure</option><option value="table">Result table</option><option value="fragment">Scaffold / fragment</option></select>
      {selectionPurpose === 'structure' && selectedCompoundLabel && <label className="pdf-region-destination"><input type="checkbox" checked={replaceSelected} disabled={recognizing} onChange={(event) => setReplaceSelected(event.target.checked)} />Replace structure for {selectedCompoundLabel}</label>}
      <span>{recognizing ? (selectionPurpose === 'table' ? 'Extracting result rows…' : 'Recognizing structure…') : regionError || (draft?.rect ? (selectionPurpose === 'table' ? 'Check that the green box includes headers and row identifiers.' : 'Check the green box, then recognize and review the structure.') : (selectionPurpose === 'table' ? 'Include the result columns, headers and compound identifiers.' : 'Drag around one structure. Green = your selection.'))}</span>
      {draft?.rect && <button type="button" disabled={recognizing || draft.rect.w < .005 || draft.rect.h < .005} onClick={() => void recognize()}>{selectionPurpose === 'table' ? 'Extract rows' : 'Recognize'}</button>}
      <button type="button" disabled={recognizing} onClick={() => { setDrawing(false); setDraft(null); setRegionError(''); }}>Cancel</button>
    </div>}
    <div className="pdf-pane__viewport" ref={viewportRef} tabIndex={0} aria-label="PDF page contents" onScroll={onScroll}>
      {error && <div className="pdf-pane__message" role="status">{error}<button type="button" onClick={() => setRetry((value) => value + 1)}>Retry</button></div>}
      {!document && !error && <img className="pdf-pane__first-preview" src={getPdfPreviewUrl(pdfId, target?.page ?? 1)} alt="PDF preview" />}
      {document && Array.from({ length: document.numPages }, (_, index) => index + 1).map((number) => <ContinuousPage key={`${pdfId}-${number}`} document={document} pdfId={pdfId} number={number} width={width} zoom={zoom} root={viewportRef} onSize={onSize} onPaint={onPaint} onRegister={onRegister}
        estimatedSize={sizes[number] ?? sizes[0] ?? size} rect={target?.page === number ? target.rect : undefined} drawing={drawing} recognizing={recognizing} draft={draft?.page === number ? draft.rect ?? undefined : undefined} onDraft={onDraft} />)}
    </div>
  </div>;
}

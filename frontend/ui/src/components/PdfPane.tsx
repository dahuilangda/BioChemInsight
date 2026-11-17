import React from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { cachedPageRaster, getPreviewPageRaster, prioritizeReader, scheduleReaderRender, trackReaderRender, type Raster, loadDocument, retainDocument, releaseDocument, rasters, rememberRaster, MAX_RASTER_PIXELS } from '../utils/pdfDocument';
import { getPdfFileUrl } from '../api/client';
import { updatePdfLoad } from '../utils/pdfSession';
import DocumentLoading from './DocumentLoading';
import PdfSearch from './PdfSearch';
import type { PdfMatch } from '../utils/pdfSearch';
import { observeVisibility } from '../utils/visibility';
import {adjustRegion, type RegionAction} from '../utils/pdfRegion';

export interface PdfTarget {
  page: number;
  requestId?: number;
  rect?: { x: number; y: number; w: number; h: number };
}
type Region = NonNullable<PdfTarget['rect']>;
const NO_MATCHES: PdfMatch[] = [];
interface PdfPaneProps { onUserNavigate?: () => void; onDocumentMount?: (pdfId: string) => void; priorityPages?: number[]; deferPage?: boolean; selectedCompoundLabel?: string; pdfId: string; target: PdfTarget | null; onPageChange?: (page: number) => void; onRecognizeRegion?: (page: number, rect: Region, purpose?: 'structure' | 'table' | 'fragment', replaceSelected?: boolean) => Promise<void> }

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
  matches: PdfMatch[];
  activeMatch?: PdfMatch;
  drawing: boolean;
  recognizing: boolean;
  draft?: Region;
  onDraft: (number: number, rect: Region | null) => void;
}

const ContinuousPage = React.memo(function ContinuousPage({ document, pdfId, number, width, zoom, root, onSize, onPaint, onRegister,
  estimatedSize, rect, matches, activeMatch, drawing, recognizing, draft, onDraft }: PageProps) {
  const element = React.useRef<HTMLDivElement>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const [nearby, setNearby] = React.useState(false);
  const [page, setPage] = React.useState<pdfjsLib.PDFPageProxy | null>(null);
  const [painted, setPainted] = React.useState(false);
  const [error, setError] = React.useState('');
  const [retry, setRetry] = React.useState(0);
  const startPoint = React.useRef<{ x: number; y: number; action: RegionAction; initial?: Region } | null>(null);
  const scale = zoom === 'fit' ? Math.max(.1, (width - 24) / estimatedSize.width) : zoom;
  const paperWidth = estimatedSize.width * scale;
  const paperHeight = estimatedSize.height * scale;

  React.useEffect(() => {
    onRegister(number, element.current);
    const stop = element.current ? observeVisibility(element.current, setNearby, root.current, '450px 0px') : undefined;
    return () => { stop?.(); onRegister(number, null); };
  }, [number, onRegister, root]);
  React.useEffect(() => {
    if (!nearby || page) return;
    let canceled = false;
    prioritizeReader();
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
      setPainted(true); setError(''); updatePdfLoad(pdfId, { phase: 'ready' });
    };
    const cached = rasters.get(key);
    if (cached) { rasters.delete(key); rasters.set(key, cached); paint(cached); return; }
    const shared = cachedPageRaster(pdfId, number);
    if (shared) {
      paint(shared);
      if (shared.canvas.width >= viewport.width && shared.canvas.height >= viewport.height) return;
    }
    let task: pdfjsLib.RenderTask | undefined;
    const queued = scheduleReaderRender(async () => {
      if (canceled) return null;
      const offscreen = window.document.createElement('canvas');
      offscreen.width = Math.ceil(viewport.width); offscreen.height = Math.ceil(viewport.height);
      task = page.render({ canvasContext: offscreen.getContext('2d', { alpha: false })!, viewport });
      task.onContinue = (resume: () => void) => requestAnimationFrame(() => { if (!canceled) resume(); });
      try {
        await task.promise;
        if (canceled) { offscreen.width = offscreen.height = 0; return null; }
        const raster = { canvas: offscreen, pixels: offscreen.width * offscreen.height, pdfId };
        rememberRaster(key, raster); paint(raster); onPaint(); return raster;
      } catch (cause) {
        offscreen.width = offscreen.height = 0;
        if (!canceled && (cause as { name?: string }).name !== 'RenderingCancelledException') setError('Page unavailable');
        return null;
      }
    }, () => {
      const bounds = element.current?.getBoundingClientRect(), viewportBounds = root.current?.getBoundingClientRect();
      return bounds && viewportBounds ? Math.abs((bounds.top + bounds.bottom) / 2 - (viewportBounds.top + viewportBounds.bottom) / 2) : Infinity;
    });
    trackReaderRender(pdfId, number, queued.promise);
    return () => { canceled = true; queued.cancel(); task?.cancel(); };
  }, [page, nearby, scale, width, pdfId, number, retry, onPaint]);
  React.useEffect(() => {
    if (!nearby && canvas.current) { canvas.current.width = canvas.current.height = 0; setPainted(false); }
  }, [nearby]);
  const point = (event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)) };
  };
  return <div ref={element} className="pdf-page" data-pdf-page={number}>
    <div className="pdf-page__label">Page {number}</div>
    <div className="pdf-pane__paper" style={{ width: paperWidth, height: paperHeight, '--pdf-page-ratio': `${estimatedSize.width} / ${estimatedSize.height}` } as React.CSSProperties}>
      {nearby && !painted && <div className="pdf-page__loading" role="status"><span />Rendering page {number}…</div>}
      <canvas ref={canvas} width={0} height={0} style={{ width: paperWidth, height: paperHeight }} aria-label={`PDF page ${number}`} />
      {error && <div className="pdf-pane__message" role="status">{error}<button type="button" onClick={() => { setError(''); setRetry((value) => value + 1); }}>Retry</button></div>}
      {nearby && matches.flatMap((match, index) => match.rects.map((hit, part) => <span key={`${index}-${part}`} className={`pdf-search-hit${match === activeMatch ? ' is-active' : ''}`} style={{ left: `${hit.x * 100}%`, top: `${hit.y * 100}%`, width: `${hit.w * 100}%`, height: `${hit.h * 100}%` }} />))}
      {rect && <span className="pdf-pane__highlight" aria-label="Selected source region" style={{ left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` }} />}
      {(drawing || draft) && <div className="pdf-region-layer" style={{ touchAction: drawing && !recognizing ? 'none' : undefined, pointerEvents: recognizing ? 'none' : undefined }}
        onPointerDown={(event) => {
          if (!drawing || recognizing || event.button !== 0) return;
          event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
          const target = event.target as HTMLElement;
          const handle = target.closest<HTMLElement>('[data-region-handle]')?.dataset.regionHandle as RegionAction | undefined;
          const action: RegionAction = handle || (target.closest('.pdf-region-box') && draft ? 'move' : 'new');
          startPoint.current = {...point(event), action, initial: draft};
          if(action === 'new') onDraft(number, null);
        }}
        onPointerMove={(event) => { const start = startPoint.current; if (!start || recognizing) return; onDraft(number, adjustRegion(start.action, start, point(event), start.initial)); }}
        onPointerUp={(event) => { if(!startPoint.current) return; startPoint.current = null; if(event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); root.current?.focus({preventScroll:true}); }}
        onPointerCancel={() => { if(startPoint.current) {startPoint.current = null; onDraft(number, null);} }}>
        {draft && <span className="pdf-region-box" aria-label="Your PDF selection" style={{ left: `${draft.x * 100}%`, top: `${draft.y * 100}%`, width: `${draft.w * 100}%`, height: `${draft.h * 100}%` }}>
          {(['nw','ne','sw','se'] as const).map(corner => <span key={corner} className={`pdf-region-handle pdf-region-handle--${corner}`} data-region-handle={corner} />)}
        </span>}
      </div>}
    </div>
  </div>;
});

export default function PdfPane({ pdfId, target, onPageChange, onRecognizeRegion, selectedCompoundLabel, priorityPages = [], deferPage = false, onDocumentMount, onUserNavigate }: PdfPaneProps) {
  const [loadedDocument, setLoadedDocument] = React.useState<{ pdfId: string; proxy: pdfjsLib.PDFDocumentProxy } | null>(null);
  const document = loadedDocument?.pdfId === pdfId ? loadedDocument.proxy : null;
  const setDocument = (proxy: pdfjsLib.PDFDocumentProxy | null) => setLoadedDocument(proxy ? { pdfId, proxy } : null);
  const [error, setError] = React.useState('');
  const [retry, setRetry] = React.useState(0);
  const [searchOpen, setSearchOpen] = React.useState(false);
  const [search, setSearch] = React.useState<{ matches: PdfMatch[]; active: number }>({ matches: [], active: -1 });
  const matchesByPage = React.useMemo(() => {
    const pages = new Map<number, PdfMatch[]>();
    for (const match of search.matches) {
      const group = pages.get(match.page);
      if (group) group.push(match); else pages.set(match.page, [match]);
    }
    return pages;
  }, [search.matches]);
  const setMatches = React.useCallback((matches: PdfMatch[], active: number) => setSearch({ matches, active }), []);
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
  const paneRef = React.useRef<HTMLDivElement>(null);
  const viewportRef = React.useRef<HTMLDivElement>(null);
  const pagesRef = React.useRef(new Map<number, HTMLDivElement>());
  const navigationRef = React.useRef('');
  const viewportMetricsRef = React.useRef({ width: 0, height: 0 });
  const layoutAnchorRef = React.useRef<{ page: number; fraction: number } | null>(null);
  const scrollTimer = React.useRef<number>();
  const commandedScroll = React.useRef<{ top: number; left: number } | null>(null);
  const userNavigateRef = React.useRef(onUserNavigate); userNavigateRef.current = onUserNavigate;
  const scrollToPosition = React.useCallback((top: number, left = 0) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.scrollTo({ top: Math.max(0, top), left, behavior: 'auto' });

    commandedScroll.current = { top: viewport.scrollTop, left: viewport.scrollLeft };
  }, []);
  const pageChangeRef = React.useRef(onPageChange); pageChangeRef.current = onPageChange;
  const onRegister = React.useCallback((number: number, element: HTMLDivElement | null) => {
    if (element) pagesRef.current.set(number, element); else pagesRef.current.delete(number);
  }, []);
  const onSize = React.useCallback((number: number, w: number, h: number) => {
    setSizes((previous) => previous[number]?.width === w && previous[number]?.height === h ? previous : { ...previous, [number]: { width: w, height: h } });
  }, []);
  const onPaint = React.useCallback(() => { setRenderCount(value => value + 1); updatePdfLoad(pdfId, { phase: 'ready' }); }, [pdfId]);
  const onDraft = React.useCallback((page: number, rect: Region | null) => setDraft({ page, rect }), []);

  React.useLayoutEffect(() => { retainDocument(pdfId); onDocumentMount?.(pdfId); return () => releaseDocument(pdfId); }, [pdfId]);
  React.useEffect(() => {
    let canceled = false;
    prioritizeReader(250);
    setSearchOpen(false); setSearch({ matches: [], active: -1 });
    setDocument(null); setSizes({}); setError(''); setZoom('fit'); setDraft(null); setDrawing(false);
    navigationRef.current = ''; currentPageRef.current = target?.page ?? 1; setPageNumber(currentPageRef.current);
    void loadDocument(pdfId).then(async (value) => {
      if (canceled || deferPage) return;
      const page = await value.getPage(Math.min(value.numPages, Math.max(1, target?.page ?? 1)));
      if (canceled) return;
      const size = page.getViewport({ scale: 1 });
      setSizes({ 0: { width: size.width, height: size.height }, [page.pageNumber]: { width: size.width, height: size.height } });
      setDocument(value);
    }).catch(() => { if (!canceled) setError('PDF unavailable'); });
    return () => { canceled = true; };
  }, [pdfId, retry, deferPage]);
  const canRecognize = Boolean(onRecognizeRegion);
  React.useEffect(() => {
    if (canRecognize) return;
    setDrawing(false);
    if (!recognizing) { setDraft(null); setRegionError(''); }
  }, [canRecognize, recognizing]);
  const priorityKey = priorityPages.join(',');
  React.useEffect(() => {
    if (!document || deferPage || (window.navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData) return;
    const index = priorityPages.indexOf(pageNumber);
    const candidates = index >= 0 ? priorityPages.slice(index + 1, index + 4) : priorityPages.slice(0, 3);
    const next = candidates.find(page => page > 0 && page <= document.numPages && !cachedPageRaster(pdfId, page));
    if (!next) return;
    const controller = new AbortController();
    void getPreviewPageRaster(pdfId, next, controller.signal).catch(() => undefined);
    return () => controller.abort();
  }, [document, pdfId, pageNumber, priorityKey, deferPage]);
  React.useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    let timer: number | undefined;
    const measure = () => {
      if (!element.clientWidth || !element.clientHeight) return;
      const previous = viewportMetricsRef.current;
      if (previous.width && previous.width !== element.clientWidth) {

        const centre = element.scrollTop + previous.height / 2;
        let anchor: { page: number; fraction: number } | null = null;
        for (const [number, node] of pagesRef.current) {
          if (node.offsetTop > centre) break;
          anchor = { page: number, fraction: Math.max(0, Math.min(1, (centre - node.offsetTop) / Math.max(1, node.offsetHeight))) };
        }
        layoutAnchorRef.current = anchor;
        window.clearTimeout(scrollTimer.current);
      }
      viewportMetricsRef.current = { width: element.clientWidth, height: element.clientHeight };
      element.style.setProperty('--pdf-fit-width', `${Math.max(1, element.clientWidth - 24)}px`);

      const anchor = layoutAnchorRef.current;
      const anchorNode = anchor && pagesRef.current.get(anchor.page);
      if (anchor && anchorNode?.isConnected) {
        scrollToPosition(anchorNode.offsetTop + anchor.fraction * anchorNode.offsetHeight - element.clientHeight / 2, element.scrollLeft);
      }
      commandedScroll.current = { top: element.scrollTop, left: element.scrollLeft };
      if (timer) window.clearTimeout(timer);
      prioritizeReader(250);
      const commitWidth = () => {
        if (element.closest('.review-split--resizing')) { timer = window.setTimeout(commitWidth, 100); return; }
        if (element.clientWidth) setWidth(element.clientWidth);
      };
      timer = window.setTimeout(commitWidth, 120);
    };
    measure(); setWidth(element.clientWidth);
    const observer = new ResizeObserver(measure); observer.observe(element);
    return () => { observer.disconnect(); if (timer) window.clearTimeout(timer); };
  }, []);
  React.useLayoutEffect(() => {
    const anchor = layoutAnchorRef.current, viewport = viewportRef.current;
    if (!anchor || !viewport) return;
    const node = pagesRef.current.get(anchor.page);
    if (node?.isConnected) scrollToPosition(node.offsetTop + anchor.fraction * node.offsetHeight - viewport.clientHeight / 2, viewport.scrollLeft);
    layoutAnchorRef.current = null;
  }, [width, zoom, scrollToPosition]);
  React.useEffect(() => { setPageDraft(String(pageNumber)); }, [pageNumber]);
  React.useEffect(() => () => window.clearTimeout(scrollTimer.current), []);

  const jump = React.useCallback((page: number, rect?: Region) => {
    const element = pagesRef.current.get(page), viewport = viewportRef.current;
    if (!element || !viewport) return;
    prioritizeReader();
    layoutAnchorRef.current = null;
    window.clearTimeout(scrollTimer.current);
    const paper = element.querySelector<HTMLElement>('.pdf-pane__paper');
    const offset = element.offsetTop;
    const top = rect && paper ? offset + paper.offsetTop + (rect.y + rect.h / 2) * paper.offsetHeight - viewport.clientHeight / 2 : offset;
    scrollToPosition(top);
    currentPageRef.current = page; setPageNumber(page);
  }, [scrollToPosition]);
  React.useEffect(() => {
    if (!document || !target) return;
    const page = Math.min(document.numPages, Math.max(1, target.page));
    const key = `${pdfId}|${page}|${target.requestId ?? ''}|${JSON.stringify(target.rect ?? null)}`;
    if (navigationRef.current === key) return;
    navigationRef.current = key;
    if (target.requestId && !recognizing) { setDrawing(false); setDraft(null); setRegionError(''); }
    if (page === currentPageRef.current && !target.rect && !target.requestId && viewportRef.current?.scrollTop) return;
    jump(page, target.rect);
  }, [document, pdfId, target?.page, target?.requestId, target?.rect?.x, target?.rect?.y, target?.rect?.w, target?.rect?.h, jump]);
  const navigateMatch = React.useCallback((match: PdfMatch) => {
    userNavigateRef.current?.();
    jump(match.page, match.rects[0]);
    pageChangeRef.current?.(match.page);
  }, [jump]);
  const closeSearch = () => {
    setSearchOpen(false); setSearch({ matches: [], active: -1 });
    paneRef.current?.querySelector<HTMLButtonElement>('[data-pdf-search]')?.focus();
  };
  const onScroll = () => {
    const viewport = viewportRef.current;
    if (!document || !viewport?.clientHeight) return;
    const command = commandedScroll.current;
    commandedScroll.current = null;
    if (command && Math.abs(viewport.scrollTop - command.top) <= 1 && Math.abs(viewport.scrollLeft - command.left) <= 1) return;

    userNavigateRef.current?.();
    prioritizeReader();
    window.clearTimeout(scrollTimer.current);
    scrollTimer.current = window.setTimeout(() => {
      const viewport = viewportRef.current;
      if (!viewport) return;
      const anchor = viewport.getBoundingClientRect().top + viewport.clientHeight / 2;
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
  const changeZoom = (next: number | 'fit') => {
    const viewport = viewportRef.current;
    if (!viewport || next === zoom) return;
    const centre = viewport.scrollTop + viewport.clientHeight / 2;
    for (const [page, node] of pagesRef.current) {
      if (node.offsetTop > centre) break;
      layoutAnchorRef.current = { page, fraction: Math.max(0, Math.min(1, (centre - node.offsetTop) / Math.max(1, node.offsetHeight))) };
    }
    prioritizeReader(250); window.clearTimeout(scrollTimer.current);
    setZoom(next);
  };
  const changePage = (next: number) => {
    if (!document || !Number.isFinite(next) || recognizing) return;
    const page = Math.min(document.numPages, Math.max(1, Math.round(next)));
    userNavigateRef.current?.();
    jump(page); onPageChange?.(page); setDraft(null); setDrawing(false); setRegionError('');
  };
  const commitPage = () => pageDraft.trim() ? changePage(Number(pageDraft)) : setPageDraft(String(pageNumber));
  const recognize = async () => {
    if (!draft?.rect || !onRecognizeRegion || recognizing) return;
    setRecognizing(true); setRegionError('');
    try { await onRecognizeRegion(draft.page, draft.rect, selectionPurpose, selectionPurpose === 'structure' && replaceSelected && Boolean(selectedCompoundLabel)); setDraft(null); setDrawing(false); }
    catch (cause) {
      setRegionError(cause instanceof Error ? cause.message : 'Recognition failed. Try again.');
      if(window.document.activeElement === window.document.body || paneRef.current?.contains(window.document.activeElement)) viewportRef.current?.focus({preventScroll:true});
    }
    finally { setRecognizing(false); }
  };
  const cancelSelection = () => { setDrawing(false); setDraft(null); setRegionError(''); paneRef.current?.querySelector<HTMLButtonElement>('[data-draw-box]')?.focus(); };
  const toggleDrawing = () => {
    if (!document || recognizing || !onRecognizeRegion) return;
    setDrawing(!drawing); setDraft(null); setRegionError('');
    if(!drawing) requestAnimationFrame(() => {if(!window.document.querySelector('[aria-modal="true"]')) viewportRef.current?.focus({preventScroll:true});});
  };
  const drawCommand = React.useRef(toggleDrawing); drawCommand.current = toggleDrawing;
  React.useEffect(() => {const run = () => drawCommand.current(); window.addEventListener('bci:draw-region',run); return () => window.removeEventListener('bci:draw-region',run);}, []);
  React.useEffect(() => setReplaceSelected(false), [selectedCompoundLabel, pdfId]);
  const selectionKeyDown = (event: React.KeyboardEvent) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'f' && !event.shiftKey) {
      event.preventDefault(); event.stopPropagation(); setSearchOpen(true);
      paneRef.current?.querySelector<HTMLInputElement>('[aria-label="Search PDF text"]')?.focus();
      return;
    }
    if(event.defaultPrevented || event.nativeEvent.isComposing || event.repeat) return;
    if((event.metaKey || event.ctrlKey) && event.key === 'Enter' && draft?.rect) {event.preventDefault(); event.stopPropagation(); if(draft.rect.w >= .005 && draft.rect.h >= .005) void recognize(); return;}
    if((event.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')) return;
    if(event.key === 'Escape' && (drawing || draft)) {event.preventDefault(); event.stopPropagation(); if(!recognizing) cancelSelection(); return;}

    if(!recognizing && draft?.rect && event.target === viewportRef.current && ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key) && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault(); event.stopPropagation();
      const paper = pagesRef.current.get(draft.page)?.querySelector<HTMLElement>('.pdf-pane__paper');
      if(!paper) return;
      const pixels = event.shiftKey ? 10 : 1;
      setDraft({...draft,rect:adjustRegion('move',{x:0,y:0},{x:event.key==='ArrowLeft' ? -pixels/paper.clientWidth : event.key==='ArrowRight' ? pixels/paper.clientWidth : 0,y:event.key==='ArrowUp' ? -pixels/paper.clientHeight : event.key==='ArrowDown' ? pixels/paper.clientHeight : 0},draft.rect)});
    }
  };
  const size = sizes[pageNumber] ?? sizes[0] ?? { width: 612, height: 792 };
  const scale = zoom === 'fit' ? Math.max(.1, (width - 24) / size.width) : zoom;
  return <div ref={paneRef} onKeyDown={selectionKeyDown} className="pdf-pane" data-drawing={drawing} data-page={pageNumber} data-fit={zoom === 'fit'} data-render-count={renderCount} aria-label="Source PDF">
    <div className="pdf-pane__toolbar">
      <div className="pdf-pane__navigation">
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Previous PDF page" disabled={!document || recognizing || pageNumber <= 1} onClick={() => changePage(pageNumber - 1)}>‹</button>
        <input aria-label="PDF page" inputMode="numeric" value={pageDraft} disabled={!document || recognizing} onChange={(event) => setPageDraft(event.target.value)} onBlur={commitPage}
          onKeyDown={(event) => { if (event.key === 'Enter') { commitPage(); event.currentTarget.blur(); } else if (event.key === 'Escape') { event.stopPropagation(); setPageDraft(String(pageNumber)); } }} />
        <span className="pdf-pane__page-count">/ {document?.numPages ?? '—'}</span>
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Next PDF page" disabled={!document || recognizing || pageNumber >= document.numPages} onClick={() => changePage(pageNumber + 1)}>›</button>
      </div>
      <div className="pdf-pane__zoom-controls">
        <button type="button" className="pdf-pane__zoom-btn" data-pdf-search aria-label="Search PDF" title="Search PDF (Ctrl/⌘ F within PDF)" aria-expanded={searchOpen} disabled={!document} onClick={() => searchOpen ? closeSearch() : setSearchOpen(true)}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></svg></button>
        {onRecognizeRegion && <button type="button" className="pdf-pane__zoom-btn" data-draw-box aria-keyshortcuts="B" aria-pressed={drawing} disabled={!document || recognizing} title="Draw a box to capture missing structures or results (B)" onClick={toggleDrawing}>Draw box</button>}
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Zoom out" disabled={!document || scale <= .25} onClick={() => changeZoom(Math.max(.25, scale - .2))}>−</button>
        <span className="pdf-pane__zoom-level">{Math.round(scale * 100)}%</span>
        <button type="button" className="pdf-pane__zoom-btn" aria-label="Zoom in" disabled={!document || scale >= 4} onClick={() => changeZoom(Math.min(4, scale + .2))}>+</button>
        <button type="button" className={`pdf-pane__zoom-btn pdf-pane__zoom-btn--fit${zoom === 'fit' ? ' is-active' : ''}`} aria-label="Fit page width" onClick={() => changeZoom('fit')}>Fit</button>
        <a className="pdf-pane__zoom-btn pdf-pane__open" href={getPdfFileUrl(pdfId)} target="_blank" rel="noreferrer" aria-label="Open original PDF" title="Open original PDF">↗</a>
      </div>
    </div>
    {searchOpen && document && <PdfSearch key={pdfId} pdfId={pdfId} count={document.numPages} currentPage={pageNumber} onMatches={setMatches} onNavigate={navigateMatch} onClose={closeSearch} />}
    {(drawing || draft || regionError || recognizing) && <div className="pdf-region-actions" role="status">
      <select aria-label="Selection purpose" value={selectionPurpose} disabled={recognizing} onChange={(event) => setSelectionPurpose(event.target.value as 'structure' | 'table' | 'fragment')}><option value="structure">Structure</option><option value="table">Result table</option><option value="fragment">Scaffold / fragment</option></select>
      {selectionPurpose === 'structure' && selectedCompoundLabel && <label className="pdf-region-destination"><input type="checkbox" checked={replaceSelected} disabled={recognizing} onChange={(event) => setReplaceSelected(event.target.checked)} />Replace structure for {selectedCompoundLabel}</label>}
      {selectionPurpose !== 'table' && !replaceSelected && <small className="pdf-region-destination-hint">Adds a new record</small>}
      <span role={regionError ? 'alert' : undefined}>{recognizing ? (selectionPurpose === 'table' ? 'Extracting result rows…' : 'Recognizing structure…') : regionError || (draft?.rect ? (selectionPurpose === 'table' ? 'Check that the green box includes headers and row identifiers.' : 'Drag the box or its corners to adjust. Recognize when ready.') : (selectionPurpose === 'table' ? 'Include the result columns, headers and compound identifiers.' : 'Drag around one structure. Your box is green; source evidence is amber.'))}</span>
      {draft?.rect && <button type="button" title="Parse this selection (Ctrl/⌘ Enter)" aria-keyshortcuts="Control+Enter Meta+Enter" disabled={recognizing || draft.rect.w < .005 || draft.rect.h < .005} onClick={() => void recognize()}>{selectionPurpose === 'table' ? 'Extract rows' : 'Recognize'}</button>}
      <button type="button" disabled={recognizing} title="Cancel selection (Esc)" onClick={cancelSelection}>Cancel</button>
    </div>}
    <div className="pdf-pane__viewport" ref={viewportRef} tabIndex={0} aria-label="PDF page contents" onScroll={onScroll}>
      {error && <div className="pdf-pane__message" role="status">{error}<button type="button" onClick={() => setRetry((value) => value + 1)}>Retry</button></div>}
      {!document && !error && <DocumentLoading pdfId={pdfId} />}
      {document && Array.from({ length: document.numPages }, (_, index) => index + 1).map((number) => <ContinuousPage key={`${pdfId}-${number}`} document={document} pdfId={pdfId} number={number} width={width} zoom={zoom} root={viewportRef} onSize={onSize} onPaint={onPaint} onRegister={onRegister}
        matches={matchesByPage.get(number) ?? NO_MATCHES} activeMatch={search.matches[search.active]?.page === number ? search.matches[search.active] : undefined} estimatedSize={sizes[number] ?? sizes[0] ?? size} rect={target?.page === number ? target.rect : undefined} drawing={drawing} recognizing={recognizing} draft={draft?.page === number ? draft.rect ?? undefined : undefined} onDraft={onDraft} />)}
    </div>
  </div>;
}

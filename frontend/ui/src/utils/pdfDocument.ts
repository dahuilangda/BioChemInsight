import * as pdfjsLib from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { getPdfFileUrl } from '../api/client';
import { localPdf, pdfLoadState, updatePdfLoad } from './pdfSession';
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
const documents = new Map<string, pdfjsLib.PDFDocumentProxy>();
interface DocumentRequest { promise: Promise<pdfjsLib.PDFDocumentProxy>; task?: pdfjsLib.PDFDocumentLoadingTask; aborted: boolean }
const loadingDocuments = new Map<string, DocumentRequest>();
const documentUsers = new Map<string, number>();
export function retainDocument(pdfId: string) { documentUsers.set(pdfId, (documentUsers.get(pdfId) ?? 0) + 1); }
export function releaseDocument(pdfId: string) {
  const users = Math.max(0, (documentUsers.get(pdfId) ?? 1) - 1);
  if (users) documentUsers.set(pdfId, users); else documentUsers.delete(pdfId);
  const pending = loadingDocuments.get(pdfId);
  if (!documentUsers.get(pdfId) && pending) {
    pending.aborted = true; void pending.task?.destroy().catch(() => undefined);
  }
  evictDocuments();
}
export interface Raster { canvas: HTMLCanvasElement; pixels: number; pdfId: string }
export const rasters = new Map<string, Raster>();
export const MAX_RASTER_PIXELS = 8_000_000;
const MAX_CACHED_PIXELS = 12_000_000;

export function evictDocuments() {
  let bytes = [...documents.keys()].reduce((sum, id) => sum + (pdfLoadState(id).total ?? 0), 0);
  for (const [id, document] of documents) {
    if (documents.size <= 2 && bytes <= 64 * 1024 * 1024) break;
    if (documentUsers.get(id)) continue;
    documents.delete(id); bytes -= pdfLoadState(id).total ?? 0;
    void document.destroy().catch(() => undefined);
    for (const [key, raster] of previewPages) {
      if (raster.pdfId !== id) continue;
      raster.canvas.width = raster.canvas.height = 0; previewPages.delete(key);
    }
    documentUsers.delete(id);
    for (const [key, raster] of rasters) {
      if (raster.pdfId !== id) continue;
      raster.canvas.width = raster.canvas.height = 0;
      rasters.delete(key);
    }
  }
}
class LocalPdfRange extends pdfjsLib.PDFDataRangeTransport {
  private aborted = false;
  private ranges: Array<[number, number]> = [];
  constructor(private file: File, private fail: (reason: unknown) => void) { super(file.size, null, true, file.name); }
  requestDataRange(begin: number, end: number) {
    void this.file.slice(begin, end).arrayBuffer().then(buffer => {
      if (this.aborted) return;
      this.onDataRange(begin, new Uint8Array(buffer));
      const ranges = [...this.ranges, [begin, end] as [number, number]].sort((a, b) => a[0] - b[0]);
      this.ranges = [];
      for (const range of ranges) {
        const previous = this.ranges[this.ranges.length - 1];
        if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]);
        else this.ranges.push(range);
      }
      this.onDataProgress(this.ranges.reduce((sum, [a, b]) => sum + b - a, 0), this.file.size);
    }).catch(error => { if (!this.aborted) this.fail(error); });
  }
  abort() { this.aborted = true; }
}

export function loadDocument(pdfId: string): Promise<pdfjsLib.PDFDocumentProxy> {
  const existing = documents.get(pdfId);
  if (existing) {
    documents.delete(pdfId); documents.set(pdfId, existing);
    return Promise.resolve(existing);
  }
  const pending = loadingDocuments.get(pdfId);
  if (pending && !pending.aborted) return pending.promise;

  const file = localPdf(pdfId);
  updatePdfLoad(pdfId, { phase: 'loading', source: file ? 'local' : 'remote', loaded: 0, total: file?.size ?? null, pages: 0 });
  let request: DocumentRequest;
  const promise = Promise.resolve().then(async () => {
    if (request.aborted) throw new DOMException('Document no longer needed', 'AbortError');
    let rejectLocalRead: (reason: unknown) => void = () => undefined;
    const localFailure = new Promise<never>((_, reject) => { rejectLocalRead = reject; });
    const transport = file ? new LocalPdfRange(file, rejectLocalRead) : undefined;
    const task = pdfjsLib.getDocument({
      ...(transport ? { range: transport } : { url: getPdfFileUrl(pdfId) }),
      disableAutoFetch: true, disableStream: true, rangeChunkSize: 128 * 1024,
    });
    request.task = task;
    if (request.aborted) { void task.destroy().catch(() => undefined); throw new DOMException('Document no longer needed', 'AbortError'); }
    task.onProgress = ({ loaded, total }: { loaded: number; total: number }) => {
      if (request.aborted || loadingDocuments.get(pdfId) !== request) return;
      const previous = pdfLoadState(pdfId);
      const length = total > 0 ? total : previous.total;
      updatePdfLoad(pdfId, { loaded: Math.min(length ?? Infinity, Math.max(previous.loaded, loaded)), total: length });
    };
    let document: pdfjsLib.PDFDocumentProxy;
    try { document = await (transport ? Promise.race([task.promise, localFailure]) : task.promise); }
    catch (error) { void task.destroy().catch(() => undefined); throw error; }
    if (request.aborted) { void document.destroy().catch(() => undefined); throw new DOMException('Document no longer needed', 'AbortError'); }
    documents.set(pdfId, document);
    updatePdfLoad(pdfId, { phase: 'opening', pages: document.numPages });
    evictDocuments(); return document;
  }).catch(error => { if (!request.aborted) updatePdfLoad(pdfId, { phase: 'error' }); throw error; })
    .finally(() => { if (loadingDocuments.get(pdfId) === request) loadingDocuments.delete(pdfId); });
  request = { promise, aborted: false };
  loadingDocuments.set(pdfId, request);
  return promise;
}
export function rememberRaster(key: string, raster: Raster) {
  rasters.set(key, raster);
  let pixels = [...rasters.values()].reduce((sum, item) => sum + item.pixels, 0);
  for (const [oldKey, item] of rasters) {
    if (rasters.size <= 3 && pixels <= MAX_CACHED_PIXELS) break;
    if (oldKey === key) continue;
    rasters.delete(oldKey); pixels -= item.pixels;
    item.canvas.width = item.canvas.height = 0;
  }
}

const readerRenders = new Map<string, Promise<Raster | null>>();
interface PreviewRequest { promise: Promise<Raster>; controller: AbortController; users: number }
const previewRenders = new Map<string, PreviewRequest>();
const previewPages = new Map<string, Raster>();
const MAX_PREVIEW_PIXELS = 6_000_000;
export function trackReaderRender(pdfId: string, page: number, promise: Promise<Raster | null>) {
  const key = `${pdfId}|${page}`;
  readerRenders.set(key, promise);
  void promise.finally(() => { if (readerRenders.get(key) === promise) readerRenders.delete(key); });
}
export function cachedPageRaster(pdfId: string, page: number): Raster | undefined {
  const prefix = `${pdfId}|${page}|`;
  let best: Raster | undefined;
  for (const [key, raster] of rasters) {
    if (key.startsWith(prefix) && raster.canvas.width && (!best || raster.pixels > best.pixels)) best = raster;
  }
  const preview = previewPages.get(`${pdfId}|${page}`);
  if (preview?.canvas.width && (!best || preview.pixels > best.pixels)) best = preview;
  return best;
}
function withAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    const abort = () => { signal.removeEventListener('abort', abort); reject(new DOMException('Preview no longer needed', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    promise.then(value => { signal.removeEventListener('abort', abort); resolve(value); }, error => { signal.removeEventListener('abort', abort); reject(error); });
    if (signal.aborted) abort();
  });
}
let previewTail: Promise<unknown> = Promise.resolve();
export async function getPreviewPageRaster(pdfId: string, page: number, signal?: AbortSignal): Promise<Raster> {
  throwIfAborted(signal);
  const cached = cachedPageRaster(pdfId, page);
  if (cached) return cached;
  const key = `${pdfId}|${page}`;
  const reader = readerRenders.get(key);
  if (reader) {
    const raster = await withAbort(reader, signal);
    if (raster?.canvas.width) return raster;
  }
  let pending = previewRenders.get(key);
  if (!pending) {
    const controller = new AbortController();
    let request: PreviewRequest;

    const work = async () => {
      const signal = controller.signal;
      throwIfAborted(signal);
      await waitForReaderIdle(signal);
      const existing = cachedPageRaster(pdfId, page);
      if (existing) return existing;
      retainDocument(pdfId);
      const canvas = window.document.createElement('canvas');
      let render: pdfjsLib.RenderTask | undefined;
      const cancel = () => render?.cancel();
      signal.addEventListener('abort', cancel, { once: true });
      try {
        const document = await withAbort(loadDocument(pdfId), signal);
        const source = await withAbort(document.getPage(page), signal);
        await waitForReaderIdle(signal);
        const size = source.getViewport({ scale: 1 });
        const scale = Math.min(2, Math.sqrt(2_000_000 / (size.width * size.height)));
        const viewport = source.getViewport({ scale });
        canvas.width = Math.ceil(viewport.width); canvas.height = Math.ceil(viewport.height);
        render = source.render({ canvasContext: canvas.getContext('2d', { alpha: false })!, viewport });
        render.onContinue = (resume: () => void) => { void waitForReaderIdle(signal).then(resume).catch(cancel); };
        await render.promise; throwIfAborted(signal);
        const raster = { canvas, pixels: canvas.width * canvas.height, pdfId };
        previewPages.set(key, raster);
        let pixels = [...previewPages.values()].reduce((sum, item) => sum + item.pixels, 0);
        for (const [oldKey, item] of previewPages) {
          if (pixels <= MAX_PREVIEW_PIXELS) break;
          if (oldKey === key) continue;
          pixels -= item.pixels; previewPages.delete(oldKey); item.canvas.width = item.canvas.height = 0;
        }
        return raster;
      } catch (error) { canvas.width = canvas.height = 0; throw error; }
      finally {
        signal.removeEventListener('abort', cancel);
        releaseDocument(pdfId);
      }
    };
    request = { controller, users: 0, promise: previewTail.then(work).finally(() => { if (previewRenders.get(key) === request) previewRenders.delete(key); }) };
    previewTail = request.promise.catch(() => undefined);
    previewRenders.set(key, request); pending = request;
  }
  pending.users++;
  try { return await withAbort(pending.promise, signal); }
  finally {
    pending.users--;
    if (!pending.users && previewRenders.get(key) === pending) {
      previewRenders.delete(key); pending.controller.abort();
    }
  }
}

let readerBusyUntil = 0;
export function prioritizeReader(milliseconds = 180) {
  readerBusyUntil = Math.max(readerBusyUntil, performance.now() + milliseconds);
}
export function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new DOMException('Preview no longer needed', 'AbortError');
}
function delay(milliseconds: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => signal?.removeEventListener('abort', cancel);
    const timer = window.setTimeout(() => { cleanup(); resolve(); }, milliseconds);
    const cancel = () => { window.clearTimeout(timer); cleanup(); reject(new DOMException('Preview no longer needed', 'AbortError')); };
    signal?.addEventListener('abort', cancel, { once: true }); if (signal?.aborted) cancel();
  });
}
export async function waitForReaderIdle(signal?: AbortSignal) {
  for (;;) {
    throwIfAborted(signal);
    if (window.document.hidden) {
      await new Promise<void>((resolve, reject) => {
        const cleanup = () => { window.document.removeEventListener('visibilitychange', visible); signal?.removeEventListener('abort', cancel); };
        const visible = () => { if (!window.document.hidden) { cleanup(); resolve(); } };
        const cancel = () => { cleanup(); reject(new DOMException('Preview no longer needed', 'AbortError')); };
        window.document.addEventListener('visibilitychange', visible); signal?.addEventListener('abort', cancel, { once: true });
        if (signal?.aborted) cancel(); else visible();
      });
      continue;
    }
    if (readerRenders.size) { await withAbort(Promise.all([...readerRenders.values()]), signal); continue; }
    const remaining = readerBusyUntil - performance.now();
    if (remaining > 0) { await delay(remaining, signal); continue; }
    await new Promise<void>((resolve, reject) => {
      let handle: number;
      const cleanup = () => signal?.removeEventListener('abort', cancel);
      const done = () => { cleanup(); resolve(); };
      const cancel = () => {
        if ('cancelIdleCallback' in window) window.cancelIdleCallback(handle); else cancelAnimationFrame(handle);
        cleanup(); reject(new DOMException('Preview no longer needed', 'AbortError'));
      };
      handle = 'requestIdleCallback' in window ? window.requestIdleCallback(done, { timeout: 500 }) : requestAnimationFrame(done);
      signal?.addEventListener('abort', cancel, { once: true }); if (signal?.aborted) cancel();
    });
    if (!readerRenders.size && performance.now() >= readerBusyUntil && !window.document.hidden) return;
  }
}

interface ReaderJob { distance: () => number; run: () => Promise<Raster | null>; resolve: (value: Raster | null) => void }
const readerQueue: ReaderJob[] = [];
let reading = false;
let readerFrame = 0;
function drainReaderQueue() {
  if (reading || readerFrame || !readerQueue.length) return;
  readerFrame = requestAnimationFrame(() => {
    readerFrame = 0;
    readerQueue.sort((a, b) => a.distance() - b.distance());
    const job = readerQueue.shift();
    if (!job) return;
    reading = true;
    void job.run().then(job.resolve, () => job.resolve(null)).finally(() => { reading = false; drainReaderQueue(); });
  });
}
export function scheduleReaderRender(run: () => Promise<Raster | null>, distance: () => number) {
  let job: ReaderJob;
  const promise = new Promise<Raster | null>((resolve) => { job = { run, distance, resolve }; readerQueue.push(job); });
  drainReaderQueue();
  return { promise, cancel: () => {
    const index = readerQueue.indexOf(job);
    if (index >= 0) { readerQueue.splice(index, 1); job.resolve(null); }
  } };
}

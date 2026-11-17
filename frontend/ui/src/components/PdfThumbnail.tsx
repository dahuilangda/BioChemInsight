import React from 'react';
import { fetchArtifact, fetchBoxArtifact } from '../api/client';
import { getPreviewPageRaster } from '../utils/pdfDocument';
import { observeVisibility } from '../utils/visibility';

let active = 0;
const queue: Array<() => void> = [];
const sourceCache = new Map<string, HTMLCanvasElement>();
const galleryCache = new Map<string, HTMLCanvasElement>();
async function scheduled(work: () => Promise<void>) {
  if (active >= 2) await new Promise<void>((resolve) => queue.push(resolve));
  active++;
  try { await work(); }
  finally { active--; queue.shift()?.(); }
}
function remember(cache: Map<string, HTMLCanvasElement>, key: string, canvas: HTMLCanvasElement, maxPixels: number) {
  cache.set(key, canvas);
  let pixels = [...cache.values()].reduce((sum, item) => sum + item.width * item.height, 0);
  for (const [oldKey, item] of cache) {
    if (pixels <= maxPixels) break;
    if (oldKey === key) continue;
    pixels -= item.width * item.height;
    cache.delete(oldKey); item.width = item.height = 0;
  }
}
const PdfThumbnail = React.memo(function PdfThumbnail({ pdfId, page, boxPath, label, fallback, fallbackPath }: {
  pdfId: string; page: number; boxPath?: string; label: string; fallback?: string | null; fallbackPath?: string;
}) {
  const root = React.useRef<HTMLDivElement>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const [near, setNear] = React.useState(false);
  const [state, setState] = React.useState<'loading' | 'ready' | 'error'>('loading');
  const [fallbackImage, setFallbackImage] = React.useState<string | null>(null);
  React.useEffect(() => {
    if (!root.current) return;
    return observeVisibility(root.current, setNear, null, '120px');
  }, []);
  React.useEffect(() => {
    if (!near) return;
    let canceled = false;
    const controller = new AbortController();
    setState('loading'); setFallbackImage(null);
    const key = `${pdfId}|${page}|${boxPath || 'page'}`;
    const cache = boxPath ? sourceCache : galleryCache;
    const paint = (raster: HTMLCanvasElement) => {
      if (canceled || !canvas.current) return;
      canvas.current.width = raster.width; canvas.current.height = raster.height;
      canvas.current.getContext('2d', { alpha: false })!.drawImage(raster, 0, 0);
      setState('ready');
    };
    const cached = cache.get(key);
    if (cached) { cache.delete(key); cache.set(key, cached); paint(cached); }
    else void scheduled(async () => {
      if (canceled) return;
      const raster = window.document.createElement('canvas');
      try {
        // Recheck the cache after waiting for a render slot.
        const existing = cache.get(key);
        if (existing) { paint(existing); return; }
        let rect = { x: 0, y: 0, w: 1, h: 1 };
        if (boxPath) {
          const artifact = await fetchBoxArtifact(boxPath);
          const box = JSON.parse(atob(artifact.content));
          const [x1, y1, x2, y2] = box.box.map(Number);
          const width = Number(box.page_width), height = Number(box.page_height);
          if (![x1, y1, x2, y2, width, height].every(Number.isFinite) || width <= 0 || height <= 0 || x2 <= x1 || y2 <= y1) throw new Error('Invalid source region');
          rect = { x: Math.max(0, x1 / width), y: Math.max(0, y1 / height), w: Math.min(1, x2 / width) - Math.max(0, x1 / width), h: Math.min(1, y2 / height) - Math.max(0, y1 / height) };
          if (rect.w <= 0 || rect.h <= 0) throw new Error('Source region outside page');
        }
        if (canceled) return;
        const pageRaster = await getPreviewPageRaster(pdfId, page, controller.signal);
        if (canceled) return;
        const source = pageRaster.canvas;
        const sw = source.width * rect.w, sh = source.height * rect.h;
        const scale = Math.min(1, 480 / sw, 600 / sh);
        raster.width = Math.max(1, Math.ceil(sw * scale - 1e-6));
        raster.height = Math.max(1, Math.ceil(sh * scale - 1e-6));
        raster.getContext('2d', { alpha: false })!.drawImage(source,
          source.width * rect.x, source.height * rect.y, sw, sh, 0, 0, raster.width, raster.height);
        if (!canceled) { remember(cache, key, raster, boxPath ? 3_000_000 : 1_000_000); paint(raster); }
      } finally {
        if (cache.get(key) !== raster) raster.width = raster.height = 0;
      }
    }).catch(async () => {
      if (canceled) return;
      setState('error');
      // Fetch legacy images only after PDF preview failure.
      if (!fallback && fallbackPath) {
        try {
          const artifact = await fetchArtifact(fallbackPath);
          if (!canceled && artifact.mime_type.startsWith('image/')) setFallbackImage(`data:${artifact.mime_type};base64,${artifact.content}`);
        } catch { /* Optional image fallback. */ }
      }
    });
    return () => {
      canceled = true; controller.abort();
      if (canvas.current) canvas.current.width = canvas.current.height = 0;
    };
  }, [pdfId, page, boxPath, near, fallback, fallbackPath]);
  const image = fallback || fallbackImage;
  return <div ref={root} className={`pdf-thumbnail pdf-thumbnail--${state}`} aria-busy={state === 'loading'}>
    <canvas ref={canvas} width={0} height={0} role="img" aria-label={label} />
    {state === 'loading' && <span className="pdf-thumbnail__placeholder" aria-hidden="true" />}
    {state === 'error' && (image ? <img src={image} alt={label} loading="lazy" decoding="async" /> : <span className="pdf-thumbnail__error">Preview unavailable · open source</span>)}
  </div>;
});

export default PdfThumbnail;

import { api } from '../api/client';

export interface TextRect {
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface SearchPage {
  page: number;
  source: 'text' | 'ocr';
  needs_ocr: boolean;
  items: { text: string; rect: TextRect }[];
}
export interface PdfMatch {
  offset: number;
  page: number;
  rects: TextRect[];
}
const pages = new Map<string, { page: SearchPage; bytes: number }>();
let cachedBytes = 0;
const indexes = new WeakMap<
  SearchPage,
  { text: string; spans: { start: number; end: number; rect: TextRect }[] }
>();

function remember(pdfId: string, page: SearchPage) {
  const key = `${pdfId}:${page.page}`;
  cachedBytes -= pages.get(key)?.bytes ?? 0;
  const bytes = page.items.reduce((sum, item) => sum + 160 + item.text.length * 2, 0);
  pages.delete(key);
  pages.set(key, { page, bytes });
  cachedBytes += bytes;
  while (cachedBytes > 8 * 1024 * 1024 || pages.size > 300) {
    const oldest = pages.keys().next().value!;
    cachedBytes -= pages.get(oldest)!.bytes;
    pages.delete(oldest);
  }
}

export async function readSearchPages(pdfId: string, numbers: number[], signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  const saved = new Map(
    numbers.flatMap((page) => {
      const value = pages.get(`${pdfId}:${page}`)?.page;
      return value ? [[page, value] as const] : [];
    }),
  );
  const missing = numbers.filter((page) => !pages.has(`${pdfId}:${page}`));
  let failed: number[] = [];
  if (missing.length) {
    const response = await api.get<{ pages: SearchPage[]; failed: number[] }>(
      `/pdfs/${encodeURIComponent(pdfId)}/search/pages`,
      { params: { numbers: missing.join(',') }, signal, timeout: 30000 },
    );
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    response.data.pages.forEach((page) => remember(pdfId, page));
    failed = response.data.failed;
    const received = new Map(response.data.pages.map((page) => [page.page, page]));
    return {
      pages: numbers.flatMap((page) => received.get(page) ?? saved.get(page) ?? []),
      failed,
    };
  }
  return { pages: numbers.map((page) => saved.get(page)!), failed };
}

export async function readOcrPage(pdfId: string, page: number, signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  const key = `${pdfId}:${page}`;
  const cached = pages.get(key)?.page;
  if (cached && !cached.needs_ocr) return cached;
  const response = await api.get<SearchPage>(
    `/pdfs/${encodeURIComponent(pdfId)}/search/pages/${page}`,
    {
      params: { ocr: true },
      signal,
      timeout: 150000,
    },
  );
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  remember(pdfId, response.data);
  return response.data;
}

const normalize = (value: string) => value.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ');

export function searchPages(pages: SearchPage[], query: string): PdfMatch[] {
  const needle = normalize(query).trim();
  if (!needle) return [];
  const matches: PdfMatch[] = [];
  for (const page of [...pages].sort((a, b) => a.page - b.page)) {
    let index = indexes.get(page);
    if (!index) {
      const spans: { start: number; end: number; rect: TextRect }[] = [];
      let text = '';
      for (const item of page.items) {
        const value = normalize(item.text);
        if (!value) continue;
        const start = text.length;
        text += value + ' ';
        spans.push({ start, end: text.length - 1, rect: item.rect });
      }
      index = { text, spans };
      indexes.set(page, index);
    }
    const { text, spans } = index;
    let cursor = 0;
    let start = 0;
    while ((start = text.indexOf(needle, start)) >= 0) {
      const end = start + needle.length;
      while (cursor < spans.length && spans[cursor].end <= start) cursor++;
      const rects: TextRect[] = [];
      for (let item = cursor; item < spans.length && spans[item].start < end; item++) {
        rects.push(spans[item].rect);
      }
      if (rects.length) matches.push({ page: page.page, offset: start, rects });
      start = end;
      if (matches.length >= 5000) return matches;
    }
  }
  return matches;
}

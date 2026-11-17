import React from 'react';
import axios from 'axios';
import {
  readOcrPage,
  readSearchPages,
  searchPages,
  type SearchPage,
  type PdfMatch,
} from '../utils/pdfSearch';
import './pdf-search.css';

interface Props {
  pdfId: string;
  count: number;
  currentPage: number;
  onMatches: (matches: PdfMatch[], active: number) => void;
  onNavigate: (match: PdfMatch) => void;
  onClose: () => void;
}

export default function PdfSearch({
  pdfId,
  count,
  currentPage,
  onMatches,
  onNavigate,
  onClose,
}: Props) {
  const [query, setQuery] = React.useState('');
  const [term, setTerm] = React.useState('');
  const [composing, setComposing] = React.useState(false);
  const [indexed, setIndexed] = React.useState<SearchPage[]>([]);
  const indexedPages = React.useRef(new Map<number, SearchPage>());
  const [progress, setProgress] = React.useState(0);
  const [ocrPage, setOcrPage] = React.useState<number | null>(null);
  const [failed, setFailed] = React.useState(0);
  const [retry, setRetry] = React.useState(0);
  const [active, setActive] = React.useState(-1);
  const input = React.useRef<HTMLInputElement>(null);
  const navigate = React.useRef(onNavigate);
  navigate.current = onNavigate;
  const startPage = React.useRef(currentPage);
  const searching = Boolean(query.trim());
  const matches = React.useMemo(() => searchPages(indexed, term), [indexed, term]);
  const selected = React.useRef<PdfMatch | null>(null);
  const lastQuery = React.useRef(term);
  React.useEffect(() => {
    if (composing) return;
    const timer = window.setTimeout(() => setTerm(query), query.trim() ? 150 : 0);
    return () => window.clearTimeout(timer);
  }, [query, composing]);

  React.useEffect(() => {
    input.current?.focus();
  }, []);
  React.useEffect(() => {
    if (!searching || !count) return;
    const controller = new AbortController();
    const { signal } = controller;
    setProgress([...indexedPages.current.values()].filter((page) => !page.needs_ocr).length);
    setFailed(0);
    setOcrPage(null);
    const values = indexedPages.current;
    const order = [
      startPage.current,
      ...Array.from({ length: count }, (_, i) => i + 1).filter((n) => n !== startPage.current),
    ];
    const wait = () =>
      new Promise<void>((resolve) => {
        const finish = () => {
          clearTimeout(timer);
          signal.removeEventListener('abort', finish);
          resolve();
        };
        const timer = setTimeout(finish, 2000);
        signal.addEventListener('abort', finish, { once: true });
      });
    const run = async () => {
      const scans = order.filter((page) => values.get(page)?.needs_ocr);
      let done = [...values.values()].filter((page) => !page.needs_ocr).length;
      const pending = order.filter((page) => !values.has(page));
      const batches = pending.length ? [pending.slice(0, 1)] : [];
      for (let index = 1; index < pending.length; index += 8)
        batches.push(pending.slice(index, index + 8));
      for (const numbers of batches) {
        if (signal.aborted) return;
        try {
          const result = await readSearchPages(pdfId, numbers, signal);
          if (signal.aborted) return;
          for (const page of result.pages) {
            values.set(page.page, page);
            if (page.needs_ocr) scans.push(page.page);
            else done++;
          }
          setIndexed([...values.values()]);
          setProgress(done);
          setFailed((value) => value + result.failed.length);
        } catch {
          if (signal.aborted) return;
          setFailed((value) => value + numbers.length);
        }
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }
      for (const page of scans) {
        if (signal.aborted) return;
        setOcrPage(page);
        try {
          let result: SearchPage | undefined;
          for (let attempt = 0; attempt < 4; attempt++) {
            try {
              result = await readOcrPage(pdfId, page, signal);
              break;
            } catch (error) {
              if (signal.aborted) return;
              if (!axios.isAxiosError(error) || error.response?.status !== 429 || attempt === 3)
                throw error;
              await wait();
            }
          }
          if (signal.aborted) return;
          if (result) {
            values.set(page, result);
            setIndexed([...values.values()]);
            setProgress(++done);
          }
        } catch {
          if (signal.aborted) return;
          setFailed((value) => value + 1);
        }
      }
      if (!signal.aborted) setOcrPage(null);
    };
    const timer = window.setTimeout(() => void run(), 250);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [pdfId, count, searching, retry]);

  React.useEffect(() => {
    const changed = term !== lastQuery.current;
    lastQuery.current = term;
    let index = changed
      ? -1
      : matches.findIndex(
          (match) =>
            match.page === selected.current?.page && match.offset === selected.current?.offset,
        );
    if (index < 0 && matches.length)
      index = matches.findIndex((match) => match.page >= startPage.current);
    if (index < 0 && matches.length) index = 0;
    const match = matches[index];
    const moved =
      match &&
      (changed ||
        !selected.current ||
        match.page !== selected.current.page ||
        match.offset !== selected.current.offset);
    selected.current = match ?? null;
    setActive(index);
    if (moved) navigate.current(match);
  }, [matches, term]);
  React.useEffect(() => {
    onMatches(matches, active);
  }, [matches, active, onMatches]);
  const move = (direction: number) => {
    if (!matches.length) return;
    const next = (active + direction + matches.length) % matches.length;
    setActive(next);
    selected.current = matches[next];
    navigate.current(matches[next]);
  };
  return (
    <div
      className="pdf-search"
      role="search"
      aria-label="Search PDF"
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing || composing) return;
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          onClose();
        }
        if (event.key === 'Enter' && event.target === input.current) {
          event.preventDefault();
          event.stopPropagation();
          move(event.shiftKey ? -1 : 1);
        }
      }}
    >
      <div className="pdf-search__controls">
        <input
          ref={input}
          type="search"
          aria-label="Search PDF text"
          placeholder="Search PDF"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onCompositionStart={() => setComposing(true)}
          onCompositionEnd={() => setComposing(false)}
        />
        <span className="pdf-search__count" aria-live="polite">
          {searching
            ? matches.length
              ? `${active + 1} / ${matches.length}${matches.length === 5000 ? '+' : ''}`
              : progress === count
                ? failed
                  ? 'Search incomplete'
                  : 'No matches'
                : failed && progress + failed >= count && !ocrPage
                  ? 'Search incomplete'
                  : 'Searching…'
            : ''}
        </span>
        <button
          type="button"
          aria-label="Previous PDF match"
          title="Previous match (Shift Enter)"
          disabled={!matches.length}
          onClick={() => move(-1)}
        >
          ↑
        </button>
        <button
          type="button"
          aria-label="Next PDF match"
          title="Next match (Enter)"
          disabled={!matches.length}
          onClick={() => move(1)}
        >
          ↓
        </button>
        <button type="button" aria-label="Close PDF search" title="Close (Esc)" onClick={onClose}>
          ×
        </button>
      </div>
      {searching && (progress < count || failed > 0) && (
        <div className="pdf-search__status" role="status">
          <span>
            {ocrPage
              ? `Reading scanned page ${ocrPage} · ${progress}/${count} pages indexed`
              : `${progress}/${count} pages indexed`}
            {failed > 0 ? ` · ${failed} unavailable` : ''}
          </span>
          {failed > 0 && (
            <button type="button" onClick={() => setRetry((value) => value + 1)}>
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}

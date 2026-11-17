import { useSyncExternalStore } from 'react';

export interface PdfLoadState {
  phase: 'loading' | 'opening' | 'ready' | 'error';
  source: 'local' | 'remote';
  loaded: number;
  total: number | null;
  pages: number;
}
const initial: PdfLoadState = { phase: 'loading', source: 'remote', loaded: 0, total: null, pages: 0 };
const states = new Map<string, PdfLoadState>();
const listeners = new Map<string, Set<() => void>>();
const localFiles = new Map<string, File>();

export function registerLocalPdf(pdfId: string, file: File) {
  localFiles.delete(pdfId); localFiles.set(pdfId, file);
  while (localFiles.size > 2) localFiles.delete(localFiles.keys().next().value!);
}
export function localPdf(pdfId: string) { return localFiles.get(pdfId); }
export function pdfLoadState(pdfId: string) { return states.get(pdfId) ?? initial; }
export function updatePdfLoad(pdfId: string, patch: Partial<PdfLoadState>) {
  const previous = pdfLoadState(pdfId);
  const next = { ...previous, ...patch };
  if (Object.keys(next).every(key => next[key as keyof PdfLoadState] === previous[key as keyof PdfLoadState])) return;
  states.set(pdfId, next);
  for (const listener of listeners.get(pdfId) ?? []) listener();
  for (const id of states.keys()) {
    if (states.size <= 8) break;
    if (id !== pdfId && !listeners.get(id)?.size) states.delete(id);
  }
}
export function usePdfLoadState(pdfId: string) {
  return useSyncExternalStore((listener) => {
    let subscribers = listeners.get(pdfId);
    if (!subscribers) { subscribers = new Set(); listeners.set(pdfId, subscribers); }
    subscribers.add(listener);
    return () => { subscribers!.delete(listener); if (!subscribers!.size) listeners.delete(pdfId); };
  }, () => pdfLoadState(pdfId), () => initial);
}

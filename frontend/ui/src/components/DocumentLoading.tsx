import React from 'react';
import { usePdfLoadState } from '../utils/pdfSession';

function formatBytes(value: number) {
  return value >= 1024 * 1024 ? `${(value / (1024 * 1024)).toFixed(1)} MB` : `${Math.ceil(value / 1024)} KB`;
}

export function DocumentTransferStatus({ pdfId }: { pdfId: string }) {
  const state = usePdfLoadState(pdfId);
  if (state.phase === 'opening' || state.phase === 'ready') return <>Loading results…</>;
  return <>{state.loaded ? `Opening PDF · ${formatBytes(state.loaded)}` : 'Opening PDF…'}</>;
}

export default function DocumentLoading({ pdfId }: { pdfId: string }) {
  const state = usePdfLoadState(pdfId);
  const percent = state.total && state.loaded ? Math.min(100, Math.floor(state.loaded / state.total * 100)) : null;
  const opening = state.phase === 'opening' || percent === 100;
  return (
    <div className="pdf-loading-paper" role="status" aria-live="polite">
      <svg className={`document-wait-ring${percent === null || opening ? ' document-wait-ring--indeterminate' : ''}`} width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
        <circle cx="20" cy="20" r="16" className="document-wait-ring__track" />
        <circle cx="20" cy="20" r="16" className="document-wait-ring__value" strokeDasharray="100.53" strokeDashoffset={percent === null || opening ? 75 : 100.53 * (1 - percent / 100)} />
      </svg>
      <strong>{opening ? 'Preparing page…' : 'Opening PDF…'}</strong>
      {!opening && percent !== null && (
        <span className="document-wait__bytes" role="progressbar" aria-label="PDF data received" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
          {percent}% <small>{formatBytes(state.loaded)} / {formatBytes(state.total!)}</small>
        </span>
      )}
    </div>
  );
}

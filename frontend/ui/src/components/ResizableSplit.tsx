import React from 'react';

interface Props {
  children: React.ReactNode;
  className?: string;
  storageKey: string;
  enabled?: boolean;
  busy?: boolean;
}

/** Resize with DOM updates during dragging; commit React state only on release. */
export default function ResizableSplit({ children, className = '', storageKey, enabled = true, busy = false }: Props) {
  const clamp = (value: number) => Math.min(72, Math.max(38, value));
  const [share, setShare] = React.useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) ?? '55');
      return typeof stored === 'number' && Number.isFinite(stored) ? clamp(stored) : 55;
    } catch { return 55; }
  });
  const root = React.useRef<HTMLDivElement>(null);
  React.useLayoutEffect(() => { if (root.current) root.current.inert = busy; }, [busy]);
  const dragging = React.useRef(false);
  const pending = React.useRef<number | null>(null);
  const frame = React.useRef<number | null>(null);
  React.useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(share)); } catch { /* optional persistence */ }
  }, [share, storageKey]);
  React.useEffect(() => () => { if (frame.current !== null) cancelAnimationFrame(frame.current); }, []);
  const commit = (event: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    root.current?.classList.remove('review-split--resizing');
    if (frame.current !== null) { cancelAnimationFrame(frame.current); frame.current = null; }
    if (pending.current !== null) {
      root.current?.style.setProperty('--pdf-pane-share', `${pending.current}%`);
      setShare(pending.current);
      pending.current = null;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const panes = React.Children.toArray(children);
  return <div ref={root} className={`resizable-split ${className}${enabled ? '' : ' resizable-split--single'}`} style={{ '--pdf-pane-share': `${share}%` } as React.CSSProperties}>
    {enabled && panes[0]}
    {enabled && <div className="resizable-split__divider" role="separator" tabIndex={0}
      aria-label="Resize PDF pane" aria-orientation="vertical" aria-valuemin={38} aria-valuemax={72} aria-valuenow={Math.round(share)}
      onPointerDown={(event) => {
        if ((window.innerWidth <= 900 && !(window.innerWidth >= 600 && window.innerHeight <= 500)) || event.button !== 0) return;
        event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
        dragging.current = true; root.current?.classList.add('review-split--resizing');
      }}
      onPointerMove={(event) => {
        if (!dragging.current) return;
        const bounds = root.current?.getBoundingClientRect();
        if (!bounds?.width) return;
        pending.current = clamp((event.clientX - bounds.left) / bounds.width * 100);
        if (frame.current === null) frame.current = requestAnimationFrame(() => {
          frame.current = null; root.current?.style.setProperty('--pdf-pane-share', `${pending.current}%`);
        });
      }}
      onPointerUp={commit} onPointerCancel={commit} onLostPointerCapture={commit}
      onDoubleClick={() => setShare(55)}
      onKeyDown={(event) => {
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
          event.preventDefault(); setShare((value) => clamp(value + (event.key === 'ArrowLeft' ? -3 : 3)));
        } else if (event.key === 'Home') { event.preventDefault(); setShare(55); }
      }} />}
    {enabled ? panes[panes.length - 1] : panes[0]}
  </div>;
}

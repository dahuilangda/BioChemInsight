import React from 'react';

export default function ReviewRowHandle({ rowKey, label, disabled, onMove, onStep }: {
  rowKey: string; label: string; disabled: boolean;
  onMove: (from: string, to: string, after?: boolean) => void; onStep: (direction: -1 | 1) => void;
}) {
  const drag = React.useRef<{ x: number; y: number; target: HTMLTableRowElement | null; moved: boolean; after: boolean } | null>(null);
  const clear = () => { if (drag.current?.target) delete drag.current.target.dataset.drop; drag.current = null; };
  React.useEffect(() => clear, []);
  return <button type="button" className="review-row-grip" disabled={disabled} aria-label={`Move compound ${label}`}
    title={disabled ? 'Clear sorting and grouping to move rows' : 'Drag to reorder · Alt + ↑ / ↓ to move one row'}
    onClick={(event) => event.stopPropagation()}
    onPointerDown={(event) => { if (disabled || event.button !== 0) return; event.preventDefault(); event.stopPropagation(); event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); drag.current = { x: event.clientX, y: event.clientY, target: null, moved: false, after: false }; }}
    onPointerMove={(event) => {
      const current = drag.current; if (!current) return;
      if (Math.hypot(event.clientX - current.x, event.clientY - current.y) > 5) current.moved = true;
      if (!current.moved) return;
      const row = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLTableRowElement>('tr[data-row-key]') || null;
      if (current.target && current.target !== row) delete current.target.dataset.drop;
      current.target = row && row.dataset.rowKey !== rowKey ? row : null;
      if (current.target) { const rect = current.target.getBoundingClientRect(); current.after = event.clientY > rect.y + rect.height / 2; current.target.dataset.drop = current.after ? 'after' : 'before'; }
      const viewport = event.currentTarget.closest('.table-wrapper');
      if (viewport) { const rect = viewport.getBoundingClientRect(); if (event.clientY < rect.top + 36) viewport.scrollTop -= 14; else if (event.clientY > rect.bottom - 36) viewport.scrollTop += 14; }
    }}
    onPointerUp={(event) => { const current = drag.current; const target = current?.target?.dataset.rowKey; clear(); if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); if (current?.moved && target) onMove(rowKey, target, current.after); }}
    onPointerCancel={clear} onLostPointerCapture={clear}
    onKeyDown={(event) => { if (!disabled && event.altKey && ['ArrowUp', 'ArrowDown'].includes(event.key)) { event.preventDefault(); event.stopPropagation(); onStep(event.key === 'ArrowUp' ? -1 : 1); } }}>⠿</button>;
}

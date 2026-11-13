import React from 'react';

type CellProps = React.ThHTMLAttributes<HTMLTableCellElement> & { 'data-column'?: string };
export default function ReviewGridRow({ order, widths, onMoveColumn, onResizeColumn, children, ...props }: React.HTMLAttributes<HTMLTableRowElement> & {
  order: string[]; widths: Record<string, number>;
  onMoveColumn?: (from: string, to: string, after?: boolean) => void;
  onResizeColumn?: (column: string, width: number) => void;
}) {
  const cells = React.Children.toArray(children).filter(React.isValidElement) as React.ReactElement<CellProps>[];
  const defaults = cells.map((cell) => cell.props['data-column'] || '');
  const sequence = [...order.filter((key) => defaults.includes(key)), ...defaults.filter((key) => !order.includes(key))];
  return <tr {...props}>{cells.sort((a, b) => sequence.indexOf(a.props['data-column'] || '') - sequence.indexOf(b.props['data-column'] || '')).map((cell) => {
    const column = cell.props['data-column'];
    if (!column) return cell;
    const width = widths[column];
    const style = width ? { ...cell.props.style, width, minWidth: width, maxWidth: width } : cell.props.style;
    if (!onMoveColumn) return React.cloneElement(cell, { style });
    const label = typeof cell.props.children === 'string' ? cell.props.children : column.replace(/^(assay|meta):/, '');
    return React.cloneElement(cell, {
      style, draggable: true, tabIndex: 0, title: `${label} · Drag to reorder · Alt + ← / → to move`,
      onKeyDown: (event: React.KeyboardEvent<HTMLTableCellElement>) => { if (event.target === event.currentTarget && event.altKey && ['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); const position = sequence.indexOf(column); const adjacent = sequence[position + (event.key === 'ArrowLeft' ? -1 : 1)]; if (adjacent) { if (event.key === 'ArrowLeft') onMoveColumn(column, adjacent); else onMoveColumn(adjacent, column); } } },
      className: `${cell.props.className || ''} review-grid-heading`,
      onDragStart: (event: React.DragEvent<HTMLTableCellElement>) => { event.dataTransfer.setData('application/x-bci-column', column); event.dataTransfer.effectAllowed = 'move'; },
      onDragOver: (event: React.DragEvent<HTMLTableCellElement>) => { if (event.dataTransfer.types.includes('application/x-bci-column')) { event.preventDefault(); const rect = event.currentTarget.getBoundingClientRect(); event.currentTarget.dataset.drop = event.clientX > rect.x + rect.width / 2 ? 'after' : 'before'; } },
      onDragLeave: (event: React.DragEvent<HTMLTableCellElement>) => { delete event.currentTarget.dataset.drop; },
      onDrop: (event: React.DragEvent<HTMLTableCellElement>) => { event.preventDefault(); delete event.currentTarget.dataset.drop; const from = event.dataTransfer.getData('application/x-bci-column'); const rect = event.currentTarget.getBoundingClientRect(); if (from) onMoveColumn(from, column, event.clientX > rect.x + rect.width / 2); },
      children: <><span className="review-grid-heading__content">{cell.props.children}</span><span className="review-column-resizer" role="separator" aria-orientation="vertical" aria-valuemin={64} aria-valuemax={720} aria-valuenow={width || 160} aria-label={`Resize ${label}`} tabIndex={0}
        onDragStart={(event) => { event.preventDefault(); event.stopPropagation(); }}
        onKeyDown={(event) => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); onResizeColumn?.(column, Math.min(720, Math.max(64, (event.currentTarget.closest('th')?.getBoundingClientRect().width || width || 160) + (event.key === 'ArrowLeft' ? -16 : 16)))); } }}
        onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); const element = event.currentTarget; const start = event.clientX; const initial = element.closest('th')!.getBoundingClientRect().width; element.setPointerCapture(event.pointerId);
          const move = (next: PointerEvent) => onResizeColumn?.(column, Math.min(720, Math.max(64, initial + next.clientX - start)));
          const end = () => { element.removeEventListener('pointermove', move); element.removeEventListener('pointerup', end); element.removeEventListener('pointercancel', end); };
          element.addEventListener('pointermove', move); element.addEventListener('pointerup', end); element.addEventListener('pointercancel', end);
        }} /></>,
    });
  })}</tr>;
}

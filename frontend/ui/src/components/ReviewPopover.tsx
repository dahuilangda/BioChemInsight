import React from 'react';
import { createPortal } from 'react-dom';
import './review-controls.css';

export default function ReviewPopover({
  anchor,
  title,
  onClose,
  children,
  width = 360,
  actions,
  className = '',
}: {
  anchor: HTMLElement | null;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  width?: number;
  actions?: React.ReactNode;
  className?: string;
}) {
  const panel = React.useRef<HTMLDivElement>(null);
  const [position, setPosition] = React.useState({ left: 0, top: 0, width: 0, maxHeight: 400 });
  const closeRef = React.useRef(onClose);
  closeRef.current = onClose;
  React.useLayoutEffect(() => {
    if (!anchor) return;
    const measure = () => {
      const trigger = anchor.getBoundingClientRect();
      const pane = anchor.closest('.review-split__list')?.getBoundingClientRect();
      const leftEdge = Math.max(8, pane?.left ?? 8),
        rightEdge = Math.min(innerWidth - 8, pane?.right ?? innerWidth - 8);
      const panelWidth = Math.max(180, Math.min(width, rightEdge - leftEdge - 8));
      const left = Math.min(Math.max(leftEdge + 4, trigger.left), rightEdge - panelWidth - 4);
      const below = innerHeight - trigger.bottom - 16;
      const above = trigger.top - 16;
      const openAbove = below < 200 && above > below;
      const maxHeight = Math.min(680, Math.max(120, openAbove ? above : below));
      const top = openAbove
        ? Math.max(8, trigger.top - Math.min(panel.current?.offsetHeight || 240, maxHeight) - 6)
        : trigger.bottom + 6;
      setPosition({ left, top, width: panelWidth, maxHeight });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(anchor);
    if (panel.current) observer.observe(panel.current);
    const pane = anchor.closest('.review-split__list');
    if (pane) observer.observe(pane);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure, true);
    };
  }, [anchor, width]);
  React.useEffect(() => {
    if (!anchor) return;
    const pointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (
        panel.current?.contains(target) ||
        anchor.contains(target) ||
        (target instanceof Element && target.closest('[aria-modal="true"]'))
      )
        return;
      closeRef.current();
    };
    const escape = (event: KeyboardEvent) => {
      if (
        event.key === 'Escape' &&
        !event.defaultPrevented &&
        !(event.target instanceof Element && event.target.closest('[aria-modal="true"]'))
      ) {
        event.preventDefault();
        closeRef.current();
        anchor.focus({ preventScroll: true });
      }
    };
    document.addEventListener('keydown', escape);
    document.addEventListener('pointerdown', pointer);
    const frame = requestAnimationFrame(() => {
      const body = panel.current?.querySelector('.review-popover__body');
      const first =
        body?.querySelector<HTMLElement>('[autofocus]') ||
        body?.querySelector<HTMLElement>('input:not([type=checkbox]), select, button');
      first?.focus({ preventScroll: true });
    });
    return () => {
      document.removeEventListener('pointerdown', pointer);
      document.removeEventListener('keydown', escape);
      cancelAnimationFrame(frame);
    };
  }, [anchor]);
  if (!anchor) return null;
  return createPortal(
    <div
      ref={panel}
      role="dialog"
      aria-label={title}
      onBlur={(event) => {
        const next = event.relatedTarget instanceof Node ? event.relatedTarget : null;
        if (next && !event.currentTarget.contains(next) && !anchor.contains(next))
          closeRef.current();
      }}
      className={`review-popover ${className}`}
      style={{ ...position, visibility: position.width ? 'visible' : 'hidden' }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          closeRef.current();
          anchor.focus({ preventScroll: true });
        }
      }}
    >
      <header className="review-popover__header">
        <h3>{title}</h3>
        <div>
          {actions}
          <button
            type="button"
            className="review-icon-button"
            aria-label={`Close ${title.toLowerCase()}`}
            onClick={() => {
              onClose();
              anchor.focus({ preventScroll: true });
            }}
          >
            ×
          </button>
        </div>
      </header>
      <div className="review-popover__body">{children}</div>
    </div>,
    document.body,
  );
}

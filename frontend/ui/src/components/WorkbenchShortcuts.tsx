import React from 'react';
import { createPortal } from 'react-dom';
import './WorkbenchShortcuts.css';

export default function WorkbenchShortcuts({
  onSearch,
  onFilter,
  onSort,
  onHistory,
  onSave,
  onEdit,
  onDraw,
}: {
  onSearch: () => void;
  onFilter: () => void;
  onSort: () => void;
  onHistory: () => void;
  onSave: () => void;
  onEdit: () => void;
  onDraw: () => void;
}) {
  const [open, setOpen] = React.useState(false);
  const button = React.useRef<HTMLButtonElement>(null);
  const close = React.useRef<HTMLButtonElement>(null);
  const actions = React.useRef({ onSearch, onFilter, onSort, onHistory, onSave, onEdit, onDraw });
  actions.current = { onSearch, onFilter, onSort, onHistory, onSave, onEdit, onDraw };
  const modifier = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.repeat ||
        target?.closest('[aria-modal="true"], [role="dialog"]')
      )
        return;
      const editing = Boolean(
        target?.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'),
      );
      const command = event.metaKey || event.ctrlKey,
        key = event.key.toLowerCase();
      if (command && !event.altKey && !event.shiftKey && key === 's') {
        event.preventDefault();
        if (editing && target instanceof HTMLElement) target.blur();
        actions.current.onSave();
        return;
      }
      if (editing || event.altKey) return;
      let action: (() => void) | undefined;
      if (command && key === '/') action = () => setOpen(true);
      else if (command && event.shiftKey && key === 'h') action = actions.current.onHistory;
      else if (command && event.shiftKey && key === 'f') action = actions.current.onFilter;
      else if (command && event.shiftKey && key === 's') action = actions.current.onSort;
      else if (command && !event.shiftKey && key === 'f') action = actions.current.onSearch;
      else if (!command && key === '?') action = () => setOpen(true);
      else if (!command && !event.shiftKey && key === 'b') action = actions.current.onDraw;
      else if (
        !command &&
        (key === 'f2' || key === 'enter') &&
        !target?.closest('button,a,summary,.pdf-pane')
      )
        action = actions.current.onEdit;
      if (action) {
        event.preventDefault();
        action();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);
  React.useEffect(() => {
    if (!open) return;
    close.current?.focus();
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);
  const dismiss = () => {
    setOpen(false);
    button.current?.focus();
  };
  const rows = [
    [`${modifier} / or ?`, 'Show keyboard shortcuts'],
    [`${modifier} F`, 'Search results'],
    [`${modifier} Shift F`, 'Open filters'],
    [`${modifier} Shift S`, 'Open sort options'],
    [`${modifier} Shift H`, 'Open edit history'],
    [`${modifier} S`, 'Save current edits'],
    ['↑ / ↓', 'Select previous / next result and show its source'],
    ['Enter / F2', 'Edit the selected result'],
    ['Tab / Shift Tab', 'Move between editable fields and controls'],
    ['Enter', 'Save the field being edited'],
    ['Esc', 'Cancel a field edit or close the current panel'],
    ['B', 'Start / stop drawing a PDF box'],
    [`${modifier} Enter`, 'Parse the selected box while the PDF has focus'],
    ['↑ ↓ ← →', 'Move a selected box when the PDF has focus'],
    ['Shift + ↑ ↓ ← →', 'Move the box in larger steps'],
    ['Alt + ↑ / ↓', 'Move a row while its drag handle has focus'],
    ['Alt + ← / →', 'Move a column while its heading has focus'],
  ];
  return (
    <>
      <button
        ref={button}
        type="button"
        className="workbench-shortcuts-button"
        aria-label="Keyboard shortcuts"
        title={`Keyboard shortcuts (${modifier} /)`}
        onClick={() => setOpen(true)}
      >
        <span aria-hidden="true">?</span>
      </button>
      {open &&
        createPortal(
          <div className="workbench-help-layer" onClick={dismiss}>
            <section
              role="dialog"
              aria-label="Keyboard shortcuts"
              aria-modal="true"
              className="workbench-help"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  e.stopPropagation();
                  dismiss();
                }
                if (e.key === 'Tab') {
                  e.preventDefault();
                  close.current?.focus();
                }
              }}
            >
              <header>
                <div>
                  <h2>Keyboard shortcuts</h2>
                  <p>PDF selection and result editing.</p>
                </div>
                <button
                  ref={close}
                  type="button"
                  aria-label="Close keyboard shortcuts"
                  onClick={dismiss}
                >
                  ×
                </button>
              </header>
              <div className="workbench-help__flow">
                <strong>Select a PDF region</strong>
                <p>
                  Press B → drag a box on the PDF → choose Structure, Result table or Scaffold /
                  fragment → check the destination → Recognize or Extract rows. Drag the box or its
                  corners to adjust it.
                </p>
              </div>
              <dl>
                {rows.map(([keys, label]) => (
                  <div key={keys + label}>
                    <dt>{label}</dt>
                    <dd>
                      <kbd>{keys}</kbd>
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="workbench-help__note">
                B and navigation shortcuts are inactive while typing.
              </p>
            </section>
          </div>,
          document.body,
        )}
    </>
  );
}

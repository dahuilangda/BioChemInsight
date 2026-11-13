import React from 'react';

export default function ReviewValueInput({ value, label, disabled, onSave }: {
  value: unknown; label: string; disabled?: boolean; onSave: (value: string) => void;
}) {
  const original = value == null ? '' : String(value);
  const [draft, setDraft] = React.useState(original);
  const canceled = React.useRef(false);
  React.useEffect(() => setDraft(original), [original]);
  const conflicting = /\[(?:p\.|source)/.test(original) && original.includes(';');
  return <input className={`review-value-input${conflicting ? ' review-value-input--conflict' : ''}`} type="text" aria-label={label} title={conflicting ? `Conflicting source readings — check the PDF before correcting: ${original}` : 'Click to correct · Enter to save · Escape to cancel'}
    value={draft} disabled={disabled} placeholder="—" onChange={(event) => setDraft(event.target.value)}
    onFocus={() => { canceled.current = false; }}
    onBlur={() => { if (!canceled.current && draft !== original) onSave(draft); }}
    onKeyDown={(event) => {
      if (event.key === 'Enter') event.currentTarget.blur();
      if (event.key === 'Escape') { event.stopPropagation(); canceled.current = true; setDraft(original); event.currentTarget.blur(); }
    }} />;
}

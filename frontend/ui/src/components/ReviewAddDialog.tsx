import React from 'react';

export default function ReviewAddDialog({
  kind,
  onClose,
  onAdd,
  returnFocus,
}: {
  kind: 'compound' | 'measurement';
  onClose: () => void;
  returnFocus: HTMLElement | null;
  onAdd: (values: {
    name: string;
    unit: string;
    target: string;
    sourceKind: string;
  }) => string | void;
}) {
  const [error, setError] = React.useState('');
  const [name, setName] = React.useState('');
  const [unit, setUnit] = React.useState('');
  const [target, setTarget] = React.useState('');
  const [sourceKind, setSourceKind] = React.useState('unknown');
  const restoreFocus = React.useRef(true);
  React.useEffect(() => () => {
    if (restoreFocus.current && returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
  }, [returnFocus]);
  return (
    <div className="review-dialog-layer" onClick={onClose}>
      <form
        className="review-add-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={kind === 'compound' ? 'Add compound' : 'Add measurement'}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            onClose();
          }
          if (event.key === 'Tab') {
            const controls = Array.from(
              event.currentTarget.querySelectorAll<HTMLElement>(
                'button:not(:disabled),input,select',
              ),
            );
            const first = controls[0],
              last = controls[controls.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first?.focus();
            }
          }
        }}
        onSubmit={(event) => {
          event.preventDefault();
          if (name.trim()) {
            const message = onAdd({
              name: name.trim(),
              unit: unit.trim(),
              target: target.trim(),
              sourceKind,
            });
            restoreFocus.current = Boolean(message) || kind !== 'compound';
            setError(message || '');
          }
        }}
      >
        <header>
          <h3>{kind === 'compound' ? 'Add compound' : 'Add measurement'}</h3>
          <button type="button" onClick={onClose} aria-label="Close add dialog">
            ×
          </button>
        </header>
        <p>
          {kind === 'compound'
            ? 'Create a record, then draw or paste its molecular structure.'
            : 'Use the scientific endpoint and the source unit. Experimental and predicted results stay separate.'}
        </p>
        <label>
          {kind === 'compound' ? 'Compound ID' : 'Endpoint'}
          <input
            autoFocus
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        {kind === 'measurement' && (
          <>
            <label>
              Unit
              <input value={unit} onChange={(event) => setUnit(event.target.value)} />
            </label>
            <label>
              Target / system
              <input value={target} onChange={(event) => setTarget(event.target.value)} />
            </label>
            <label>
              Result source
              <select value={sourceKind} onChange={(event) => setSourceKind(event.target.value)}>
                <option value="unknown">Not specified</option>
                <option value="experimental">Experimental</option>
                <option value="predicted">Predicted</option>
                <option value="calculated">Calculated</option>
              </select>
            </label>
          </>
        )}
        {error && <p role="alert">{error}</p>}
        <footer>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" disabled={!name.trim()}>
            Add {kind}
          </button>
        </footer>
      </form>
    </div>
  );
}

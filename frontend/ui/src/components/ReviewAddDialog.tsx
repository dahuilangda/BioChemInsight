import React from 'react';

export default function ReviewAddDialog({ kind, onClose, onAdd }: {
  kind: 'compound' | 'measurement'; onClose: () => void;
  onAdd: (values: { name: string; unit: string; target: string; sourceKind: string }) => void;
}) {
  const [name, setName] = React.useState('');
  const [unit, setUnit] = React.useState('');
  const [target, setTarget] = React.useState('');
  const [sourceKind, setSourceKind] = React.useState('unknown');
  return <div className="review-dialog-layer" onClick={onClose}>
    <form className="review-add-dialog" role="dialog" aria-modal="true" aria-label={kind === 'compound' ? 'Add compound' : 'Add measurement'}
      onClick={(event) => event.stopPropagation()} onSubmit={(event) => { event.preventDefault(); if (name.trim()) onAdd({ name: name.trim(), unit: unit.trim(), target: target.trim(), sourceKind }); }}>
      <header><h3>{kind === 'compound' ? 'Add compound' : 'Add measurement'}</h3><button type="button" onClick={onClose} aria-label="Close add dialog">×</button></header>
      <p>{kind === 'compound' ? 'Create a record, then draw or paste its molecular structure.' : 'Use the scientific endpoint and the source unit. Experimental and predicted results stay separate.'}</p>
      <label>{kind === 'compound' ? 'Compound ID' : 'Endpoint'}<input autoFocus required value={name} onChange={(event) => setName(event.target.value)} /></label>
      {kind === 'measurement' && <><label>Unit<input value={unit} onChange={(event) => setUnit(event.target.value)} /></label><label>Target / system<input value={target} onChange={(event) => setTarget(event.target.value)} /></label><label>Result source<select value={sourceKind} onChange={(event) => setSourceKind(event.target.value)}><option value="unknown">Not specified</option><option value="experimental">Experimental</option><option value="predicted">Predicted</option><option value="calculated">Calculated</option></select></label></>}
      <footer><button type="button" onClick={onClose}>Cancel</button><button type="submit" disabled={!name.trim()}>Add {kind}</button></footer>
    </form>
  </div>;
}

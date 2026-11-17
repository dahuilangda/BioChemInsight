import React from 'react';
import './extraction-options.css';

export type AssemblyAvailability = 'checking' | 'available' | 'unavailable' | 'error';

type ExtractionOptionsProps = {
  assemblyAvailability: AssemblyAvailability;
  includeMarkush: boolean;
  disabled: boolean;
  onMarkushChange: (checked: boolean) => void;
  onAvailabilityRetry: () => void;
};

function ResultIcon({ type }: { type: 'structures' | 'activity' | 'adme' }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {type === 'structures' ? (
        <>
          <path d="m12 5-8 4.5v9L12 23l8-4.5v-9L12 5Z" />
          <path d="m20 9.5 7-4M20 18.5l7 4M12 23v5M7 11.5v5M12 8.5l5 3" />
        </>
      ) : type === 'activity' ? (
        <>
          <path d="M4 5v22h24" />
          <path d="M7 22c5 0 5-2 8-8s4-6 11-6" />
          <circle cx="10" cy="21" r="1.5" fill="currentColor" stroke="none" />
          <circle cx="17" cy="11" r="1.5" fill="currentColor" stroke="none" />
          <circle cx="24" cy="8" r="1.5" fill="currentColor" stroke="none" />
        </>
      ) : (
        <>
          <circle cx="16" cy="16" r="4" />
          <path d="M16 12V4m-3 3 3-3 3 3M20 16h8m-3-3 3 3-3 3M16 20v8m3-3-3 3-3-3M12 16H4m3 3-3-3 3-3" />
        </>
      )}
    </svg>
  );
}

function AssemblyDiagram() {
  return (
    <svg className="extraction-options__diagram" viewBox="0 0 100 36" fill="none" stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="m12 9-8 4.5v9L12 27l8-4.5v-9L12 9ZM20 13.5l6-3.5" />
      <circle cx="28" cy="9" r="3" strokeDasharray="1.5 2" />
      <path d="M39 18h19m-4-4 4 4-4 4" />
      <path d="m73 4-7 4v8l7 4 7-4V8l-7-4Zm7 4 8-4M88 4l5 3" />
      <path d="m73 22-7 4v6l7 4 7-4v-6l-7-4Zm7 4 8-4M88 22l5 3" transform="translate(0 -2)" />
    </svg>
  );
}

export default function ExtractionOptions({ assemblyAvailability, includeMarkush, disabled, onMarkushChange, onAvailabilityRetry }: ExtractionOptionsProps) {
  const [detailsOpen, setDetailsOpen] = React.useState(false);
  const detailsId = React.useId();
  const available = assemblyAvailability === 'available';
  const description = assemblyAvailability === 'checking'
    ? 'Checking Markush availability…'
    : assemblyAvailability === 'error'
    ? 'Could not check Markush availability.'
    : available
    ? 'Combine scaffold and substituents. Review proposed compounds.'
    : 'Markush assembly is unavailable.';

  return (
    <div className="extraction-options" onKeyDown={(event) => {
      if (event.key === 'Escape') setDetailsOpen(false);
    }}>
      <ul className="extraction-options__results" aria-label="Extraction results">
        <li><ResultIcon type="structures" /><span>Structures</span></li>
        <li><ResultIcon type="activity" /><span>Activity</span></li>
        <li><ResultIcon type="adme" /><span>ADME</span></li>
      </ul>
      <div className={`extraction-options__assembly${includeMarkush ? ' extraction-options__assembly--selected' : ''}`}>
        <label className="extraction-options__option">
          <AssemblyDiagram />
          <span className="extraction-options__label">Markush assembly</span>
          {assemblyAvailability === 'checking' && <span className="upload-progress__indicator" role="status" aria-label="Checking Markush availability" />}
          {assemblyAvailability === 'error' && <span className="extraction-options__state extraction-options__state--error">Check failed</span>}
          {assemblyAvailability === 'unavailable' && <span className="extraction-options__state">Unavailable</span>}
          <span className="extraction-options__switch">
            <input type="checkbox" role="switch" aria-label="Markush assembly" aria-describedby={detailsOpen ? detailsId : undefined} checked={includeMarkush} disabled={!available || disabled} onChange={(event) => onMarkushChange(event.target.checked)} />
            <span className="extraction-options__track" aria-hidden="true"><span /></span>
          </span>
        </label>
        <button className="extraction-options__help" type="button" aria-label="About Markush assembly" aria-expanded={detailsOpen} aria-controls={detailsId} onClick={() => setDetailsOpen((open) => !open)}>
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><circle cx="10" cy="10" r="7" /><path d="M10 9v5" /><circle cx="10" cy="6" r=".8" fill="currentColor" stroke="none" /></svg>
        </button>
      </div>
      {detailsOpen && <div id={detailsId} className="extraction-options__details"><span>{description}</span>{assemblyAvailability === 'error' && <button type="button" onClick={onAvailabilityRetry}>Retry</button>}</div>}
    </div>
  );
}

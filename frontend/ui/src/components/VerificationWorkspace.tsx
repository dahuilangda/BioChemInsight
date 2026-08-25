import React from 'react';
import { api } from '../api/client';

interface VerificationItem {
  id: string;
  layer: string;
  auto_verdict: string;
  auto_note: string;
  sheet: string | null;
  human_verdict: string | null;
  human_note: string;
}

const LAYER_ORDER = ['perfect', 'good', 'mid', 'low', 'zero'];

function verdictLabel(verdict: string): string {
  if (verdict === 'match') return 'Auto: MATCH';
  if (verdict === 'mismatch') return 'Auto: MISMATCH';
  if (verdict === 'no_structures') return 'Auto: no structures on page';
  return 'Auto: UNCERTAIN';
}

export default function VerificationWorkspace(): React.ReactElement {
  const [items, setItems] = React.useState<VerificationItem[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [selected, setSelected] = React.useState<VerificationItem | null>(null);
  const [sheetUrl, setSheetUrl] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [filter, setFilter] = React.useState('pending');

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get('/verification/items');
      setItems((data.data?.items ?? []) as VerificationItem[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load verification items');
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const open = React.useCallback((item: VerificationItem) => {
    setSelected(item);
    setSheetUrl(item.sheet ? `/api/verification/sheet/${encodeURIComponent(item.id)}?t=${Date.now()}` : null);
  }, []);

  const decide = React.useCallback(
    async (verdict: 'match' | 'mismatch' | 'uncertain') => {
      if (!selected) return;
      setSaving(true);
      try {
        const data = await api.post(`/verification/items/${encodeURIComponent(selected.id)}/decision`, { verdict });
        const updated = data.data as VerificationItem;
        setItems((current) => current.map((item) => (item.id === updated.id ? updated : item)));
        setSelected(updated);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save decision');
      } finally {
        setSaving(false);
      }
    },
    [selected],
  );

  const visible = React.useMemo(() => {
    if (filter === 'all') return items;
    if (filter === 'pending') return items.filter((item) => !item.human_verdict);
    if (filter === 'done') return items.filter((item) => item.human_verdict);
    return items.filter((item) => item.layer === filter);
  }, [items, filter]);

  const stats = React.useMemo(() => {
    const done = items.filter((item) => item.human_verdict).length;
    const agree = items.filter(
      (item) => item.human_verdict && item.human_verdict === item.auto_verdict,
    ).length;
    return { total: items.length, done, agree, pending: items.length - done };
  }, [items]);

  return (
    <div className="verification">
      <header className="verification__header">
        <h2>Structure Verification</h2>
        <p>
          Sampled patents with an automatic visual verdict (PDF page vs decoded
          molecule). Review each sheet, then confirm or override — {stats.done}/{stats.total} reviewed,
          {' '}{stats.agree} agreeing with the automatic verdict.
        </p>
        <div className="verification__filters">
          {['pending', 'done', 'all', ...LAYER_ORDER].map((key) => (
            <button
              key={key}
              type="button"
              className={filter === key ? 'verification__filter--active' : ''}
              onClick={() => setFilter(key)}
            >
              {key}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="verification__error" role="alert">
          {error}{' '}
          <button type="button" onClick={() => void load()}>
            Retry
          </button>
        </div>
      )}
      {loading && <div className="verification__loading">Loading verification items…</div>}
      {!loading && visible.length === 0 && <div className="verification__empty">No items in this view.</div>}

      <ul className="verification__list">
        {visible.map((item) => (
          <li key={item.id} className="verification__item">
            <button type="button" className="verification__open" onClick={() => open(item)}>
              {item.sheet ? (
                <img
                  src={`/api/verification/sheet/${encodeURIComponent(item.id)}`}
                  alt={`Comparison sheet for ${item.id}`}
                  loading="lazy"
                />
              ) : (
                <div className="verification__nosheet">no sheet</div>
              )}
              <span className="verification__meta">
                <strong>{item.id}</strong>
                <em className={`verification__layer verification__layer--${item.layer}`}>{item.layer}</em>
                <span className={`verification__verdict verification__verdict--${item.auto_verdict}`}>
                  {verdictLabel(item.auto_verdict)}
                </span>
                {item.human_verdict ? (
                  <span className={`verification__human verification__human--${item.human_verdict}`}>
                    Human: {item.human_verdict}
                  </span>
                ) : (
                  <span className="verification__human verification__human--pending">awaiting review</span>
                )}
              </span>
            </button>
          </li>
        ))}
      </ul>

      {selected && (
        <div className="verification__modal" role="dialog" aria-modal="true" aria-label={`Verify ${selected.id}`}>
          <div className="verification__modal-body">
            <header>
              <h3>{selected.id}</h3>
              <button type="button" onClick={() => setSelected(null)} aria-label="Close">
                ×
              </button>
            </header>
            <p className="verification__note">Left: PDF page · Right: decoded molecule. {selected.auto_note}</p>
            {sheetUrl ? (
              <img src={sheetUrl} alt={`Full comparison for ${selected.id}`} />
            ) : (
              <p className="verification__note">This item has no comparison sheet (zero-structure patent).</p>
            )}
            <div className="verification__actions">
              <button type="button" disabled={saving} onClick={() => void decide('match')}>
                ✓ Match
              </button>
              <button type="button" disabled={saving} onClick={() => void decide('mismatch')}>
                ✗ Mismatch
              </button>
              <button type="button" disabled={saving} onClick={() => void decide('uncertain')}>
                ? Uncertain
              </button>
              <span className="verification__decision">
                {selected.human_verdict ? `Your verdict: ${selected.human_verdict}` : 'No human verdict yet'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

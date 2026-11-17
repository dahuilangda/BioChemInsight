import React from 'react';
import ReviewPopover from './ReviewPopover';
import {
  fetchReviewHistory,
  restoreReviewRevision,
  type ReviewRevision,
  type ReviewChange,
  renderSmiles,
} from '../api/client';
import type { StructuresResult } from '../types';
import './ReviewHistory.css';

const text = (value: unknown) =>
  value == null || value === ''
    ? 'Empty'
    : typeof value === 'object'
      ? JSON.stringify(value, null, 2)
      : String(value);
const fieldName = (value: string) =>
  ({
    COMPOUND_ID: 'Compound ID',
    PAGE_NUM: 'Page',
    MOLBLOCK: '2D coordinates',
    SMILES: 'SMILES',
    __row_position: 'Row position',
    STRUCTURE_POSE_SOURCE: 'Coordinate source',
  })[value] || value.replaceAll('_', ' ');
const operationLabels: Record<string, string> = {
  add: 'Added',
  delete: 'Deleted',
  update: 'Modified',
};
function changeSummary(changes: ReviewRevision['changes']) {
  return ['add', 'delete', 'update']
    .map((operation) => {
      const count = changes.filter((change) => change.operation === operation).length;
      return count ? `${count} ${operationLabels[operation].toLowerCase()}` : '';
    })
    .filter(Boolean)
    .join(' · ');
}
function isDetailField(field: string) {
  return (
    /^(USER_|PREVIOUS_|MARKUSH_|LOCAL_EVIDENCE_|MEASUREMENT_)/.test(field) ||
    field === 'BIOACTIVITY_OBSERVATIONS' ||
    field === 'ID_SOURCE' ||
    field === 'STRUCTURE_POSE_SOURCE' ||
    field.endsWith('_FILE') ||
    field.endsWith('_PATH') ||
    field.includes('__')
  );
}
function AuditValue({ value, field }: { value: unknown; field: string }) {
  const [open, setOpen] = React.useState(false);
  const [image, setImage] = React.useState('');
  React.useEffect(() => {
    if (!open || !value || !['MOLBLOCK', 'SMILES'].includes(field)) return;
    let live = true;
    renderSmiles(field === 'SMILES' ? String(value) : '', {
      molblock: field === 'MOLBLOCK' ? String(value) : undefined,
      width: 360,
      height: 200,
    })
      .then((img) => {
        if (live) setImage(img);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [open, value, field]);
  if (['MOLBLOCK', 'SMILES'].includes(field) && value)
    return (
      <details className="audit-structure" onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>View molecular structure</summary>
        {image && <img src={image} alt="Historical molecular structure" />}
        <details className="audit-raw">
          <summary>View raw data</summary>
          <pre>{text(value)}</pre>
        </details>
      </details>
    );
  return <span>{text(value)}</span>;
}
export default function ReviewHistory({
  taskId,
  anchor,
  initialRecordId,
  initialRecordLabel,
  onClose,
  onRestored,
  onBusyChange,
}: {
  taskId: string;
  anchor: HTMLElement | null;
  initialRecordId?: string;
  initialRecordLabel?: string;
  onClose: () => void;
  onRestored: (result: StructuresResult) => void;
  onBusyChange: (busy: boolean) => void;
}) {
  const [history, setHistory] = React.useState<ReviewRevision[]>([]);
  const [base, setBase] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [query, setQuery] = React.useState('');
  const [record, setRecord] = React.useState(initialRecordId || '');
  const [operation, setOperation] = React.useState('');
  const [pending, setPending] = React.useState<{
    event: ReviewRevision;
    scope: 'version' | 'record' | 'field';
    recordId?: string;
    field?: string;
    side: 'before' | 'after';
  } | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [expanded, setExpanded] = React.useState<Set<string>>(new Set());
  const [eventLimit, setEventLimit] = React.useState(30);
  const [changeLimits, setChangeLimits] = React.useState<Record<string, number>>({});

  React.useEffect(() => {
    let live = true;
    fetchReviewHistory(taskId)
      .then((data) => {
        if (live) {
          setHistory(data.revisions);
          setBase(data.current_revision);
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [taskId]);
  const records = new Map(
    [...history]
      .reverse()
      .flatMap((e) =>
        e.changes.map((c) => [c.record_id, String(c.compound_id || c.record_id)] as const),
      ),
  );
  if (initialRecordId) records.set(initialRecordId, initialRecordLabel || initialRecordId);
  React.useEffect(() => {
    setEventLimit(30);
    setChangeLimits({});
  }, [record, query, operation]);
  const filtered = history
    .map((e) => ({
      ...e,
      changes: e.changes.filter(
        (c) =>
          (!record || c.record_id === record) &&
          (!operation || c.operation === operation) &&
          (!query || JSON.stringify(c).toLowerCase().includes(query.toLowerCase())),
      ),
    }))
    .filter((e) => e.changes.length);
  React.useEffect(() => {
    if (!record) return;
    const latest = history.find((event) =>
      event.changes.some((change) => change.record_id === record),
    );
    if (latest) setExpanded((previous) => new Set([...previous, latest.revision]));
  }, [record, history]);
  const pendingOperation = pending?.event.changes.find(
    (change) => change.record_id === pending.recordId,
  )?.operation;
  async function restore() {
    if (!pending) return;
    setBusy(true);
    onBusyChange(true);
    setError('');
    try {
      const result = await restoreReviewRevision(
        taskId,
        pending.event.revision,
        base,
        pending.scope,
        pending.recordId,
        pending.field,
        pending.side,
      );
      onRestored(result);
      onClose();
      anchor?.focus({ preventScroll: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Restore failed. Reload history and try again.');
    } finally {
      setBusy(false);
      onBusyChange(false);
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            {
              task_id: taskId,
              current_revision: base,
              scope: { record_id: record || null, operation: operation || null, search: query },
              revisions: filtered,
            },
            null,
            2,
          ),
        ],
        { type: 'application/json' },
      ),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = `${taskId}-edit-history.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const renderField = (
    event: ReviewRevision,
    field: ReviewChange['fields'][number],
    change: ReviewChange,
  ) => (
    <div className="audit-field" key={field.field}>
      <div>
        <strong>{fieldName(field.field)}</strong>
        {!isDetailField(field.field) &&
          !['RECORD_ID', 'SMILES', 'MOLBLOCK'].includes(field.field) &&
          !field.field.startsWith('USER_') &&
          !field.field.startsWith('__') &&
          change.operation === 'update' && (
            <button
              disabled={busy}
              onClick={() =>
                setPending({
                  event,
                  scope: 'field',
                  recordId: change.record_id,
                  field: field.field,
                  side: 'before',
                })
              }
            >
              Restore value
            </button>
          )}
      </div>
      <div className="audit-values">
        <div className="audit-value">
          <small>Before</small>
          <AuditValue field={field.field} value={field.before} />
        </div>
        <div className="audit-value">
          <small>After</small>
          <AuditValue field={field.field} value={field.after} />
        </div>
      </div>
    </div>
  );
  return (
    <ReviewPopover
      anchor={anchor}
      title="Edit history"
      width={600}
      onClose={onClose}
      actions={
        <button
          type="button"
          className="review-text-button"
          disabled={loading || !filtered.length}
          onClick={download}
        >
          Export log
        </button>
      }
      className="review-history-popover"
    >
      <section
        className="audit-panel"
        onKeyDown={(event) => {
          if (event.key === 'Escape' && pending && !busy) {
            event.preventDefault();
            event.stopPropagation();
            setPending(null);
          }
        }}
      >
        <div className="audit-filters">
          <input
            aria-label="Search edit history"
            placeholder="Search compound, field or value…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select
            aria-label="Filter history by compound"
            value={record}
            onChange={(e) => setRecord(e.target.value)}
          >
            <option value="">All compounds</option>
            {[...records].map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
          <select
            aria-label="Filter history by change"
            value={operation}
            onChange={(event) => setOperation(event.target.value)}
          >
            <option value="">All changes</option>
            <option value="add">Added</option>
            <option value="delete">Deleted</option>
            <option value="update">Modified</option>
          </select>
        </div>
        {error && (
          <p role="alert" className="audit-error">
            {error}
          </p>
        )}
        {pending && (
          <div className="audit-confirm" role="status">
            <strong>
              {pending.scope === 'record' && pendingOperation === 'add'
                ? 'Remove this added record?'
                : pending.scope === 'record' && pendingOperation === 'delete'
                  ? 'Recover this deleted record?'
                  : `Restore ${pending.scope === 'version' ? 'the entire result set' : pending.scope === 'field' ? fieldName(pending.field!) : 'this compound'}?`}
            </strong>
            <p>
              {pending.side === 'before'
                ? 'Use the values before this change.'
                : 'Use the values after this change.'}{' '}
              {pending.scope === 'version'
                ? 'All current records will be replaced.'
                : 'Other compounds will stay as they are.'}{' '}
              A new history entry will be saved.
            </p>
            <button disabled={busy} onClick={() => void restore()}>
              {busy ? 'Restoring…' : 'Confirm restore'}
            </button>
            <button disabled={busy} onClick={() => setPending(null)}>
              Cancel
            </button>
          </div>
        )}
        <div className="audit-timeline" aria-busy={loading}>
          {loading ? (
            <p>Loading history…</p>
          ) : !filtered.length ? (
            <div className="audit-empty">
              <p>
                {query || operation
                  ? 'No changes match these filters.'
                  : record
                    ? `No changes recorded for ${records.get(record) || 'this record'}.`
                    : 'No saved changes yet.'}
              </p>
              {record && (
                <button type="button" className="review-text-button" onClick={() => setRecord('')}>
                  Show all records
                </button>
              )}
            </div>
          ) : (
            filtered.slice(0, eventLimit).map((event) => (
              <article key={event.revision}>
                <div className="audit-event-title">
                  <time dateTime={event.saved_at}>
                    {new Date(
                      event.saved_at.endsWith('Z') ? event.saved_at : `${event.saved_at}Z`,
                    ).toLocaleString()}
                  </time>
                  <span
                    title={
                      event.actor?.kind === 'browser_session'
                        ? 'Browser session; account identity is not verified'
                        : undefined
                    }
                  >
                    {event.actor?.label || 'Author not recorded'}
                  </span>
                </div>
                <div className="audit-event-meta">
                  <span className="audit-event-summary">
                    {changeSummary(event.changes)}
                    {event.restore_of && ` · Restored from ${event.restore_of.slice(0, 8)}`}
                  </span>
                  <code title={event.revision}>{event.revision.slice(0, 8)}</code>
                </div>
                <details
                  open={expanded.has(event.revision)}
                  onToggle={(e) => {
                    const isOpen = e.currentTarget.open;
                    setExpanded((previous) => {
                      const next = new Set(previous);
                      if (isOpen) next.add(event.revision);
                      else next.delete(event.revision);
                      return next;
                    });
                  }}
                >
                  <summary>Review changes</summary>
                  {expanded.has(event.revision) && (
                    <>
                      {event.changes.slice(0, changeLimits[event.revision] || 100).map((change) => (
                        <div className="audit-record" key={change.record_id}>
                          <div className="audit-record-title">
                            <strong>{change.compound_id || 'Unnamed compound'}</strong>
                            <span>
                              {change.operation === 'update' &&
                              change.fields.every((field) => isDetailField(field.field))
                                ? 'Source / review updated'
                                : operationLabels[change.operation] || change.operation}
                            </span>
                            <button
                              disabled={busy}
                              onClick={() =>
                                setPending({
                                  event,
                                  scope: 'record',
                                  recordId: change.record_id,
                                  side: 'before',
                                })
                              }
                            >
                              {change.operation === 'delete'
                                ? 'Recover record'
                                : change.operation === 'add'
                                  ? 'Undo addition'
                                  : 'Restore record'}
                            </button>
                          </div>
                          {change.fields
                            .filter((field) => !isDetailField(field.field))
                            .map((field) => renderField(event, field, change))}
                          {change.fields.some((field) => isDetailField(field.field)) && (
                            <details className="audit-source-details">
                              <summary>Source and review details</summary>
                              {change.fields
                                .filter((field) => isDetailField(field.field))
                                .map((field) => renderField(event, field, change))}
                            </details>
                          )}
                        </div>
                      ))}
                      {event.changes.length > (changeLimits[event.revision] || 100) && (
                        <button
                          className="audit-show-more"
                          onClick={() =>
                            setChangeLimits((previous) => ({
                              ...previous,
                              [event.revision]: (previous[event.revision] || 100) + 100,
                            }))
                          }
                        >
                          Show 100 more compounds
                        </button>
                      )}
                    </>
                  )}
                </details>
                <details className="audit-version-options">
                  <summary>Version options</summary>
                  <footer>
                    <button
                      disabled={busy}
                      onClick={() => setPending({ event, scope: 'version', side: 'before' })}
                    >
                      Restore before this change
                    </button>
                    <button
                      disabled={busy}
                      onClick={() => setPending({ event, scope: 'version', side: 'after' })}
                    >
                      Restore this version
                    </button>
                  </footer>
                </details>
              </article>
            ))
          )}
          {filtered.length > eventLimit && (
            <button
              className="audit-show-more"
              onClick={() => setEventLimit((value) => value + 30)}
            >
              Show 30 more saves
            </button>
          )}
        </div>
      </section>
    </ReviewPopover>
  );
}

import React from 'react';
import {
  fetchAnnotationItems,
  fetchAnnotations,
  fetchArtifact,
  getAnnotationsExportUrl,
  renderSmilesBatch,
  upsertAnnotation,
  type AnnotationItem,
  type AnnotationRecord,
} from '../api/client';
import StructureEditorModal from './StructureEditorModal';

type StatusFilter = 'all' | 'pending' | 'confirmed' | 'corrected' | 'wrong';

interface AnnotationWorkspaceProps {
  taskId: string;
  onClose: () => void;
}

const KIND_LABELS: Record<string, string> = {
  complete: '完整结构',
  raw_markush: 'Markush 骨架',
  raw_fragment: '片段',
  raw_text_substituent: '文本取代基',
};

const STATUS_META: Record<string, { label: string; className: string }> = {
  confirmed: { label: '已确认', className: 'annot-chip--confirmed' },
  corrected: { label: '已更正', className: 'annot-chip--corrected' },
  wrong: { label: '错误', className: 'annot-chip--wrong' },
};

const AnnotationWorkspace: React.FC<AnnotationWorkspaceProps> = ({ taskId, onClose }) => {
  const [items, setItems] = React.useState<AnnotationItem[]>([]);
  const [annotations, setAnnotations] = React.useState<Record<string, AnnotationRecord>>({});
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState<StatusFilter>('all');
  const [onlyRaw, setOnlyRaw] = React.useState(false);
  const [currentKey, setCurrentKey] = React.useState('');
  const [segmentImages, setSegmentImages] = React.useState<Record<string, string>>({});
  const [renders, setRenders] = React.useState<Record<string, string>>({});
  const [editorOpen, setEditorOpen] = React.useState(false);
  const [editorItem, setEditorItem] = React.useState<AnnotationItem | null>(null);
  const [saving, setSaving] = React.useState(false);
  const listRef = React.useRef<HTMLDivElement | null>(null);

  const loadAll = React.useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [itemsRes, annotationsRes] = await Promise.all([
        fetchAnnotationItems(taskId),
        fetchAnnotations(taskId),
      ]);
      setItems(itemsRes.items || []);
      setAnnotations(annotationsRes || {});
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  React.useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const statusOf = React.useCallback(
    (key: string): string => annotations[key]?.status || 'pending',
    [annotations],
  );

  const visibleItems = React.useMemo(() => {
    return items.filter((item) => {
      if (onlyRaw && !item.kind.startsWith('raw_')) return false;
      const st = statusOf(item.row_key);
      if (statusFilter === 'all') return true;
      if (statusFilter === 'pending') return st === 'pending';
      return st === statusFilter;
    });
  }, [items, onlyRaw, statusFilter, statusOf]);

  React.useEffect(() => {
    if (visibleItems.length === 0) return;
    if (!visibleItems.some((it) => it.row_key === currentKey)) {
      setCurrentKey(visibleItems[0].row_key);
    }
  }, [visibleItems, currentKey]);

  const currentItem = React.useMemo(
    () => visibleItems.find((it) => it.row_key === currentKey) || null,
    [visibleItems, currentKey],
  );

  // Fetch the source crop and the parsed render for the current item only.
  const activeKey = currentKeyOrNull(currentItem);
  const correctedMolblock = activeKey ? annotations[activeKey]?.corrected_molblock : undefined;
  const correctedSmiles = activeKey ? annotations[activeKey]?.corrected_smiles : undefined;
  React.useEffect(() => {
    if (!currentItem) return;
    let cancelled = false;
    const key = currentItem.row_key;
    if (currentItem.segment_file && !(key in segmentImages)) {
      fetchArtifact(currentItem.segment_file)
        .then((res) => {
          const dataUri = res ? `data:${res.mime_type || 'image/png'};base64,${res.content}` : '';
          if (!cancelled) {
            setSegmentImages((prev) => ({ ...prev, [key]: dataUri }));
          }
        })
        .catch(() => {
          if (!cancelled) {
            setSegmentImages((prev) => ({ ...prev, [key]: '' }));
          }
        });
    }
    const molblockForRender = correctedMolblock || currentItem.molblock || '';
    const smilesForRender = correctedSmiles || currentItem.smiles || '';
    if ((molblockForRender || smilesForRender) && !(key in renders)) {
      renderSmilesBatch([{ key, smiles: smilesForRender, molblock: molblockForRender, width: 420, height: 360 }])
        .then((batch) => {
          if (!cancelled) {
            setRenders((prev) => ({ ...prev, [key]: batch?.[0]?.image || '' }));
          }
        })
        .catch(() => {
          if (!cancelled) {
            setRenders((prev) => ({ ...prev, [key]: '' }));
          }
        });
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeKey, correctedMolblock, correctedSmiles]);

  const moveToIndex = React.useCallback(
    (index: number) => {
      if (index < 0 || index >= visibleItems.length) return;
      setCurrentKey(visibleItems[index].row_key);
      const node = listRef.current?.querySelector<HTMLElement>(`[data-annot-key="${visibleItems[index].row_key}"]`);
      node?.scrollIntoView({ block: 'nearest' });
    },
    [visibleItems],
  );

  const advance = React.useCallback(() => {
    if (!currentItem) return;
    const idx = visibleItems.findIndex((it) => it.row_key === currentItem.row_key);
    const next = visibleItems
      .slice(idx + 1)
      .find((it) => statusOf(it.row_key) === 'pending');
    if (next) setCurrentKey(next.row_key);
    else if (idx + 1 < visibleItems.length) setCurrentKey(visibleItems[idx + 1].row_key);
  }, [currentItem, visibleItems, statusOf]);

  const saveAnnotation = React.useCallback(
    async (item: AnnotationItem, status: 'confirmed' | 'wrong' | 'corrected', extra?: { corrected_molblock?: string }) => {
      setSaving(true);
      try {
        const updated = await upsertAnnotation(taskId, {
          row_key: item.row_key,
          status,
          original_smiles: item.smiles,
          segment_file: item.segment_file,
          compound_id: item.compound_id,
          kind: item.kind,
          ...extra,
        });
        setAnnotations(updated || {});
        if (extra?.corrected_molblock) {
          setRenders((prev) => {
            const next = { ...prev };
            delete next[item.row_key];
            return next;
          });
        }
        return true;
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        return false;
      } finally {
        setSaving(false);
      }
    },
    [taskId],
  );

  const handleConfirm = React.useCallback(() => {
    if (!currentItem || saving) return;
    void saveAnnotation(currentItem, 'confirmed').then((ok) => {
      if (ok) advance();
    });
  }, [currentItem, saving, saveAnnotation, advance]);

  const handleWrong = React.useCallback(() => {
    if (!currentItem || saving) return;
    void saveAnnotation(currentItem, 'wrong').then((ok) => {
      if (ok) advance();
    });
  }, [currentItem, saving, saveAnnotation, advance]);

  const handleEditorSave = React.useCallback(
    (payload: { smiles: string; molblock?: string }) => {
      const item = editorItem;
      setEditorOpen(false);
      if (!item) return;
      // the editor molblock is the single source of truth: JSME opened on
      // the item's molblock so untouched atoms keep their coordinates, and
      // the server derives the SMILES label from the saved block
      void saveAnnotation(item, 'corrected', {
        corrected_molblock: payload.molblock,
      });
    },
    [editorItem, saveAnnotation],
  );

  // Keyboard: 1 confirm, 2 wrong, E edit, Up/Down navigate.
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && target.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (editorOpen) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (!currentItem) return;
      if (event.key === '1') {
        event.preventDefault();
        handleConfirm();
      } else if (event.key === '2') {
        event.preventDefault();
        handleWrong();
      } else if (event.key === 'e' || event.key === 'E') {
        if (saving) return;
        event.preventDefault();
        setEditorItem(currentItem);
        setEditorOpen(true);
      } else if (event.key === 'ArrowDown' || event.key === 'j') {
        event.preventDefault();
        const idx = visibleItems.findIndex((it) => it.row_key === currentItem.row_key);
        moveToIndex(idx + 1);
      } else if (event.key === 'ArrowUp' || event.key === 'k') {
        event.preventDefault();
        const idx = visibleItems.findIndex((it) => it.row_key === currentItem.row_key);
        moveToIndex(idx - 1);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentItem, editorOpen, handleConfirm, handleWrong, moveToIndex, visibleItems]);

  const counts = React.useMemo(() => {
    const acc = { pending: 0, confirmed: 0, corrected: 0, wrong: 0 };
    for (const item of items) {
      const st = statusOf(item.row_key);
      if (st === 'pending') acc.pending += 1;
      else acc[st as keyof typeof acc] += 1;
    }
    return acc;
  }, [items, statusOf]);

  const annotatedTotal = items.length - counts.pending;
  const progressPct = items.length > 0 ? Math.round((annotatedTotal / items.length) * 100) : 0;

  if (loading) {
    return (
      <section className="annot-workspace">
        <div className="annot-workspace__header">
          <span className="annot-workspace__title">标注工作台（调试模式）</span>
        </div>
        <div className="annot-workspace__body">加载中…</div>
      </section>
    );
  }

  return (
    <section className="annot-workspace">
      <div className="annot-workspace__header">
        <div className="annot-workspace__title-row">
          <span className="annot-workspace__title">标注工作台</span>
          <span className="annot-workspace__hint">
            左：专利原图 · 右：识别结果 · 快捷键 1 正确 / 2 错误 / E 纠正 / ↑↓ 或 J K 切换
          </span>
        </div>
        <div className="annot-workspace__progress">
          <div className="annot-workspace__progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={items.length} aria-valuenow={annotatedTotal}>
            <div className="annot-workspace__progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <span className="annot-workspace__progress-text">
            {annotatedTotal}/{items.length}（已确认 {counts.confirmed} · 已更正 {counts.corrected} · 错误 {counts.wrong}）
          </span>
        </div>
        <div className="annot-workspace__filters">
          {(['all', 'pending', 'confirmed', 'corrected', 'wrong'] as StatusFilter[]).map((value) => (
            <button
              key={value}
              type="button"
              className={`annot-filter ${statusFilter === value ? 'annot-filter--active' : ''}`}
              onClick={() => setStatusFilter(value)}
            >
              {value === 'all' ? '全部' : value === 'pending' ? '待判定' : STATUS_META[value].label}
            </button>
          ))}
          <label className="annot-onlyraw">
            <input type="checkbox" checked={onlyRaw} onChange={(e) => setOnlyRaw(e.target.checked)} />
            仅骨架/片段
          </label>
          <a className="annot-export" href={getAnnotationsExportUrl(taskId)} download>
            导出训练集
          </a>
          <button type="button" className="secondary annot-close" onClick={onClose}>
            退出调试模式
          </button>
        </div>
      </div>
      {error && <div className="annot-error" role="alert">{error}</div>}
      <div className="annot-workspace__body">
        <div className="annot-list" ref={listRef}>
          {visibleItems.length === 0 && <div className="annot-list__empty">没有匹配的条目</div>}
          {visibleItems.map((item) => {
            const st = statusOf(item.row_key);
            return (
              <button
                key={item.row_key}
                type="button"
                data-annot-key={item.row_key}
                className={`annot-list__item ${currentKey === item.row_key ? 'annot-list__item--active' : ''}`}
                aria-current={currentKey === item.row_key}
                onClick={() => setCurrentKey(item.row_key)}
              >
                <span className={`annot-dot annot-dot--${st}`} />
                <span className="annot-list__id">{item.compound_id || KIND_LABELS[item.kind] || item.kind}</span>
                <span className="annot-list__kind">{KIND_LABELS[item.kind] || item.kind}</span>
                <span className="annot-list__smiles">{(annotations[item.row_key]?.corrected_smiles || item.smiles || '').slice(0, 24)}</span>
              </button>
            );
          })}
        </div>
        {currentItem ? (
          <div className="annot-detail">
            <div className="annot-detail__meta">
              <span className="annot-detail__kind">{KIND_LABELS[currentItem.kind] || currentItem.kind}</span>
              {currentItem.compound_id && <span className="annot-detail__cid">{currentItem.compound_id}</span>}
              {STATUS_META[statusOf(currentItem.row_key)] && (
                <span className={`annot-chip ${STATUS_META[statusOf(currentItem.row_key)].className}`}>
                  {STATUS_META[statusOf(currentItem.row_key)].label}
                </span>
              )}
              <span className="annot-detail__page">第 {String(currentItem.page ?? '?')} 页</span>
            </div>
            <div className="annot-detail__panels">
              <figure className="annot-panel">
                <figcaption>专利原图</figcaption>
                {segmentImages[currentItem.row_key] ? (
                  <img src={segmentImages[currentItem.row_key]} alt="专利原图" />
                ) : segmentImages[currentItem.row_key] === '' ? (
                  <div className="annot-panel__loading">原图不可用</div>
                ) : (
                  <div className="annot-panel__loading">加载中…</div>
                )}
              </figure>
              <figure className="annot-panel">
                <figcaption>识别结果{annotations[currentItem.row_key]?.status === 'corrected' ? '（已更正）' : ''}</figcaption>
                {renders[currentItem.row_key] ? (
                  <img src={renders[currentItem.row_key]} alt="识别渲染图" />
                ) : (
                  <div className="annot-panel__placeholder">{(annotations[currentItem.row_key]?.corrected_smiles || currentItem.smiles || '').slice(0, 80)}</div>
                )}
              </figure>
            </div>
            <div className="annot-detail__smiles">
              <code>{(statusOf(currentItem.row_key) === 'corrected' && annotations[currentItem.row_key]?.corrected_smiles) || currentItem.smiles}</code>
            </div>
            <div className="annot-detail__actions">
              <button type="button" className="annot-action annot-action--confirm" disabled={saving} onClick={handleConfirm}>
                ✓ 正确 <kbd>1</kbd>
              </button>
              <button type="button" className="annot-action annot-action--edit" disabled={saving} onClick={() => { setEditorItem(currentItem); setEditorOpen(true); }}>
                ✎ 纠正 <kbd>E</kbd>
              </button>
              <button type="button" className="small-btn danger annot-action--wrong" disabled={saving} onClick={handleWrong}>
                ✗ 错误 <kbd>2</kbd>
              </button>
            </div>
          </div>
        ) : (
          <div className="annot-detail annot-detail--empty">选择左侧条目开始判定</div>
        )}
      </div>
      {editorOpen && editorItem && (
        <StructureEditorModal
          open={editorOpen}
          initialSmiles={annotations[editorItem.row_key]?.corrected_smiles || editorItem.smiles}
          initialMolblock={annotations[editorItem.row_key]?.corrected_molblock || editorItem.molblock}
          onCancel={() => setEditorOpen(false)}
          onSave={handleEditorSave}
        />
      )}
    </section>
  );
};

function currentKeyOrNull(item: AnnotationItem | null): string {
  return item ? item.row_key : '';
}

export default AnnotationWorkspace;

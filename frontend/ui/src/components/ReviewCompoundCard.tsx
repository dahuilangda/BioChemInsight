import React from 'react';
import type { StructureRecord } from '../types';
import { formatCellValue } from '../utils/format';
import { measurementEdits } from '../utils/measurements';
import CompoundIdInput from './CompoundIdInput';
import ReviewValueInput from './ReviewValueInput';

const PdfThumbnail = React.lazy(() => import('./PdfThumbnail'));

export interface CompoundCardRow {
  index: number;
  record: StructureRecord;
  primaryPageNumber: number | null;
  boxCoordsPath: string;
  segmentImage: string | null;
  segmentSource: string;
  structurePreview: string | null;
  isMarkushAssembled: boolean;
  scaffoldPageNumber: number | null;
  scaffoldBoxCoordsPath: string;
  displayedScaffoldImage: string | null;
  assayData: Record<string, unknown>;
  pagePreviewImage: string | null;
  pagePreviewSource: string;
}
interface SourceTarget { page: number; boxPath?: string }
interface Props {
  row: CompoundCardRow;
  pdfId?: string;
  active: boolean;
  canEdit: boolean;
  canEditValues: boolean;
  previewUnavailable: boolean;
  measurementColumns: readonly string[];
  artifactErrors: Readonly<Record<string, string>>;
  onLocate: (row: CompoundCardRow, source?: SourceTarget) => void;
  onSelect: (row: CompoundCardRow) => void;
  onSaveId: (index: number, value: string) => void;
  onSaveMeasurement: (index: number, record: StructureRecord, column: string, value: string) => void;
  onEdit: (index: number) => void;
  onDelete: (index: number) => void;
  onOpenArtifact: (source: unknown, label?: string, options?: { rowIndex?: number | null }) => void;
  onRetryArtifact: (path: string) => void;
}
interface SourcePreviewProps {
  pdfId?: string;
  page: number | null;
  label: string;
  boxPath?: string;
  image?: string | null;
  path?: string;
}
const SourcePreview = React.memo(function SourcePreview({ pdfId, page, label, boxPath, image, path }: SourcePreviewProps) {
  if (!pdfId || !page) return image ? <img src={image} alt={label} loading="lazy" decoding="async" /> : null;
  return (
    <React.Suspense fallback={<span className="preview-loading" aria-label={`Loading ${label}`} />}>
      <PdfThumbnail pdfId={pdfId} page={page} boxPath={boxPath} label={label} fallback={image} fallbackPath={path} />
    </React.Suspense>
  );
});

function OpenIcon() {
  return <svg className="button-icon button-icon--small" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M14 4h6v6M20 4 10 14M20 14v6H4V4h6" /></svg>;
}
function reviewNotice(record: StructureRecord): string | null {
  if (!record.SMILES && record.NAME_STRUCTURE_CANDIDATE) return 'Review the proposed structure.';
  if (!record.SMILES && String(record.RESULT_ONLY).toLowerCase() === 'true') return 'No linked structure.';
  if (['needs_review', 'failed'].includes(String(record.LOCAL_EVIDENCE_STATUS)) || record.MEASUREMENT_LINK_STATUS === 'identifier_unverified') return 'Verify the compound ID.';
  return null;
}

const ReviewCompoundCard = React.memo(function ReviewCompoundCard({
  row, pdfId, active, canEdit, canEditValues, previewUnavailable,
  measurementColumns, artifactErrors, onLocate, onSelect, onSaveId, onSaveMeasurement,
  onEdit, onDelete, onOpenArtifact, onRetryArtifact,
}: Props) {
  const { index, record, segmentImage, segmentSource, structurePreview, displayedScaffoldImage } = row;
  const compoundId = formatCellValue(record.COMPOUND_ID);
  const label = compoundId || String(index + 1);
  const notice = reviewNotice(record);
  const edits = React.useMemo(() => measurementEdits(record), [record]);
  const columns = React.useMemo(() => measurementColumns.filter(column => (
    column !== 'Source annotation' && (formatCellValue(row.assayData[column]) || Object.prototype.hasOwnProperty.call(edits, column))
  )), [measurementColumns, row.assayData, edits]);
  const originalPath = String(record.Segment || record.SEGMENT_FILE || '');
  const sourceError = originalPath ? artifactErrors[originalPath] : undefined;
  const select = () => onSelect(row);
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    const target = event.target as Element;
    if (target.closest('button,a,summary')) return;
    if (!active || !target.closest('input,textarea,select,[contenteditable="true"]')) select();
  };
  const handleFocus = (event: React.FocusEvent<HTMLElement>) => {
    if (!active && (event.target as Element).matches('input,textarea,select,[contenteditable="true"]')) select();
  };
  const handleKey = (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); select();
    }
  };
  const openSource = () => pdfId ? onLocate(row) : onOpenArtifact(segmentSource || segmentImage, `Source — ${label}`);
  const renderValues = (names: readonly string[]) => (
    <dl className="review-card__assays">
      {names.map(column => (
        <div key={column}>
          <dt title={column}>{column}</dt>
          <dd><ReviewValueInput value={row.assayData[column]} label={`${column} for compound ${label}`} disabled={!canEditValues}
            onSave={value => onSaveMeasurement(index, record, column, value)} /></dd>
        </div>
      ))}
    </dl>
  );

  return (
    <article className={`review-card${active ? ' review-card--active' : ''}`} data-row-index={index} tabIndex={0}
      aria-label={`Compound ${label}`} aria-current={active ? 'true' : undefined}
      onFocusCapture={handleFocus} onClick={handleClick} onKeyDown={handleKey}>
      <div className="review-card__header">
        <div className="review-card__identity">
          <span className="review-card__eyebrow">Compound</span>
          <CompoundIdInput initialValue={compoundId} rowIndex={index} disabled={!canEdit} onSave={onSaveId} />
        </div>
        <button type="button" className="review-card__page review-card__page-button" disabled={!row.primaryPageNumber}
          aria-label={`Open source page ${row.primaryPageNumber || ''}`} onClick={() => onLocate(row)}>
          {row.primaryPageNumber ? `Page ${row.primaryPageNumber}` : 'No source page'} <OpenIcon />
        </button>
      </div>
      {notice && <p className="review-card__notice">{notice}</p>}
      <div className="review-card__images">
        <figure className="review-card__figure">
          <figcaption className="review-card__figure-heading"><span>{row.isMarkushAssembled ? 'Scaffold + fragment' : 'Source'}</span><span>PDF ↗</span></figcaption>
          {segmentImage || pdfId && row.primaryPageNumber ? (
            <>
              {row.isMarkushAssembled ? <div className="review-card__assembly-source">
                {(row.scaffoldPageNumber || displayedScaffoldImage) && (
                  <button type="button" className="review-card__image-surface" aria-label={`Locate scaffold for compound ${label}`}
                    onClick={() => {
                      if (pdfId && row.scaffoldPageNumber) onLocate(row, { page: row.scaffoldPageNumber, boxPath: row.scaffoldBoxCoordsPath || undefined });
                      else onOpenArtifact(displayedScaffoldImage, `Scaffold — ${label}`);
                    }}>
                    <SourcePreview pdfId={pdfId} page={row.scaffoldPageNumber} boxPath={row.scaffoldBoxCoordsPath || undefined}
                      image={displayedScaffoldImage} path={String(record.MARKUSH_SCAFFOLD_IMAGE_FILE || '')} label="Assembly scaffold source" />
                    <span className="review-card__source-kind">Scaffold</span>
                  </button>
                )}
                <button type="button" className="review-card__image-surface" aria-label={`Locate fragment for compound ${label}`} onClick={openSource}>
                  <SourcePreview pdfId={pdfId} page={row.primaryPageNumber} boxPath={row.boxCoordsPath || undefined}
                    image={segmentImage} path={segmentSource} label="Source structure" />
                  <span className="review-card__source-kind">Fragment</span>
                </button>
              </div> : (
                <button type="button" className="review-card__image-surface" title="Show source in PDF" aria-label={`Locate source for compound ${label}`} onClick={openSource}>
                  <SourcePreview pdfId={pdfId} page={row.primaryPageNumber} boxPath={row.boxCoordsPath || undefined}
                    image={segmentImage} path={segmentSource} label="Source structure" />
                </button>
              )}
              {!pdfId && <button type="button" className="review-card__enlarge" aria-label={`Enlarge original for compound ${label}`} title="Enlarge original" onClick={openSource}><OpenIcon /></button>}
            </>
          ) : !pdfId && row.pagePreviewImage && !originalPath ? (
            <button type="button" className="review-card__image-surface" onClick={() => onOpenArtifact(row.pagePreviewSource || row.pagePreviewImage, `Source — ${label}`, { rowIndex: index })}>
              <img src={row.pagePreviewImage} alt="Source page" loading="lazy" decoding="async" />
            </button>
          ) : (
            <button type="button" className="review-card__image-placeholder" disabled={!row.primaryPageNumber && !sourceError}
              onClick={() => sourceError ? onRetryArtifact(originalPath) : onLocate(row)}>
              {sourceError ? 'Retry source image' : originalPath ? 'Loading source…' : row.primaryPageNumber ? `View source page ${row.primaryPageNumber}` : 'Source unavailable'}
            </button>
          )}
        </figure>
        <figure className="review-card__figure">
          <figcaption className="review-card__figure-heading"><span>Extracted</span><span>Edit ↗</span></figcaption>
          <button type="button" className={`review-card__image-surface${structurePreview ? '' : ' review-card__image-surface--empty'}`}
            title="Edit molecular structure" aria-label={`Edit structure for compound ${label}`} disabled={!canEdit} onClick={() => onEdit(index)}>
            {structurePreview ? <img src={structurePreview} alt="Extracted structure" loading="lazy" decoding="async" /> : (
              <span className="review-card__empty-structure"><span aria-hidden="true">＋</span>
                {record.SMILES ? previewUnavailable ? 'Preview unavailable' : 'Loading structure…' : record.NAME_STRUCTURE_CANDIDATE ? 'Review proposed structure' : 'Add structure'}
              </span>
            )}
          </button>
        </figure>
      </div>
      {row.isMarkushAssembled && displayedScaffoldImage && !segmentImage && !pdfId && (
        <details className="review-card__scaffold"><summary>Scaffold</summary>
          <button type="button" onClick={() => onOpenArtifact(displayedScaffoldImage, `Scaffold — ${label}`, { rowIndex: index })}>
            <img src={displayedScaffoldImage} alt="Scaffold" loading="lazy" decoding="async" />
          </button>
        </details>
      )}
      {record.ANNOTATION_TEXT && <details className="review-card__source-note"><summary>Original caption</summary><p>{formatCellValue(record.ANNOTATION_TEXT)}</p></details>}
      {columns.length > 0 && renderValues(columns.slice(0, 4))}
      {columns.length > 4 && <details className="review-card__more-values"><summary>+{columns.length - 4} values</summary>{renderValues(columns.slice(4))}</details>}
      <div className="review-card__actions">
        <button type="button" className="small-btn" aria-label="Edit structure" disabled={!canEdit} onClick={() => onEdit(index)}>Edit structure</button>
        <button type="button" className="small-btn danger" disabled={!canEdit} onClick={() => onDelete(index)}>Delete</button>
      </div>
    </article>
  );
});

export default ReviewCompoundCard;

import React from 'react';
import ReviewGridRow from './components/ReviewGridRow';
import ReviewRowHandle from './components/ReviewRowHandle';
import { moveBefore, useReviewLayout } from './utils/reviewLayout';
import ErrorNotice from './components/ErrorNotice';
import ReviewValueInput from './components/ReviewValueInput';
import ReviewAddDialog from './components/ReviewAddDialog';
import { comparableMeasurement, measurementValues, measurementEdits } from './utils/measurements';
import ResizableSplit from './components/ResizableSplit';
import {
  api,
  undoStructureEdit,
  cancelTask,
  fetchArtifact,
  fetchBoxArtifact,
  fetchPdfInfo,
  fetchPdfPage,
  fetchTask,
  fetchTaskAssays,
  fetchTaskDownload,
  fetchTaskStructures,
  fetchTasks,
  getTaskDownloadUrl,
  getPdfPreviewUrl,
  queueAutoDetectTask,
  queueAssayTask,
  queueFullPipelineTask,
  queueStructureTask,
  renderSmiles,
  renderSmilesBatch,
  updateTaskStructures,
  uploadPdf,
  fetchPatentPdf,
} from './api/client';
import type {
  AssayRecord,
  AssayTaskRequest,
  AutoDetectTaskRequest,
  FullPipelineRequest,
  StructureRecord,
  TaskListResponse,
  TaskStatus,
  UploadPDFResponse,
} from './types';
import StructureEditorModal from './components/StructureEditorModal';
import type { PdfTarget } from './components/PdfPane';
// pdf.js is heavy and only needed once the review pane opens; keep it out of
// the initial bundle
const PdfPane = React.lazy(() => import('./components/PdfPane'));
const AnnotationWorkspace = React.lazy(() => import('./components/AnnotationWorkspace'));
const VerificationWorkspace = React.lazy(() => import('./components/VerificationWorkspace'));
import StructureEditorInline, { type StructureEditorInlineHandle } from './components/StructureEditorInline';

type StepId = 1 | 2 | 3 | 4;

type StepKey = 'upload' | 'structures' | 'bioactivity' | 'review';
type StructureFilterStrictness = 'strict' | 'balanced' | 'permissive';
type JobSortBy = 'task_id' | 'updated_at' | 'created_at' | 'type' | 'status' | 'progress' | 'queue_position';
type SortDir = 'asc' | 'desc';

const stepDefinitions: Array<{ id: StepId; label: string; icon: StepKey }> = [
  { id: 1, label: 'Upload PDF', icon: 'upload' },
  { id: 2, label: 'Structures', icon: 'structures' },
  { id: 3, label: 'Activity & ADME', icon: 'bioactivity' },
  { id: 4, label: 'Review', icon: 'review' },
];

const preferredColumnOrder = ['COMPOUND_ID', 'SMILES', 'source_pages', 'IMAGE_FILE', 'SEGMENT_FILE'];
const markdownImageRegex = /!\[[^\]]*\]\((data:image\/[^)]+)\)/i;
const isDataImage = (value: string) => value.startsWith('data:image');
const ARTIFACT_CACHE_TARGET_ENTRIES = 80;
const JOBS_DEFAULT_RETENTION_DAYS = 30;
const STRUCTURE_PREVIEW_BATCH_SIZE = 16;
const REVIEW_PAGE_SIZE_OPTIONS = [25, 50, 100, 200];

function getExplicitStructureSourceTaskId(
  structureTask: TaskStatus | null,
  fullPipelineTask: TaskStatus | null,
): string | undefined {
  if (structureTask?.status === 'completed' && isUsableId(structureTask.task_id)) {
    return structureTask.task_id;
  }
  if (fullPipelineTask?.status === 'completed' && isUsableId(fullPipelineTask.task_id)) {
    return fullPipelineTask.task_id;
  }
  return undefined;
}

function formatDateInputValue(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function getDefaultJobsDateFrom(): string {
  const date = new Date();
  date.setDate(date.getDate() - JOBS_DEFAULT_RETENTION_DAYS);
  return formatDateInputValue(date);
}

function hashPreviewInput(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

const STRUCTURE_IGNORED_COLUMNS = new Set([
  'RECORD_ID', 'BIOACTIVITY_OBSERVATIONS', 'RESULT_ONLY', 'MEASUREMENT_NOTES', 'MEASUREMENT_LINK_STATUS', 'NAME_STRUCTURE_CANDIDATE', 'NAME_STRUCTURE_SOURCE', 'NAME_STRUCTURE_CID', 'STRUCTURE_RECOVERY_STATUS', 'PEPTIDE_GRAPH_VALIDATION', 'PEPTIDE_GRAPH_STATUS',
  'Structure',
  'Segment',
  'structure',
  'segment',
  'IMAGE_FILE',
  'SEGMENT_FILE',
  'Image File',
  'Segment File',
  'source_pages',
  'PAGE_NUM',
  'group_id',
  'BOX_COORDS_FILE',
  'PAGE_IMAGE_FILE',
  'MOLBLOCK',
  'molblock',
  'Molblock',
  'STRUCTURE_TYPE',
  'STRUCTURE_FILTER_REASON',
  'STRUCTURE_FILTER_BORDER_SIDES',
  'STRUCTURE_FILTER_STRICTNESS',
  'STRUCTURE_FILTER_RAW_RESPONSE',
  'IS_COMPLETE_COMPOUND',
  'FILTERED_OUT',
  'CANDIDATE_SOURCE',
  'BOX_REVIEW_CONFIDENCE',
  'BOX_REVIEW_EVIDENCE',
  'BOX_REVIEW_RAW_RESPONSE',
  'BOX_REVIEW_STATUS',
  'MOLNEXTR_QUALITY_ISSUES',
  'RAW_COMPOUND_ID',
  'CANONICAL_COMPOUND_ID',
  'ALIAS_RESOLUTION_SOURCE',
  'ID_CONFIDENCE',
  'ID_EVIDENCE',
  'ID_RAW_RESPONSE',
  'ID_SOURCE',
  'INITIAL_COMPOUND_ID',
  'INITIAL_ID_CONFIDENCE',
  'INITIAL_ID_EVIDENCE',
  'INITIAL_ID_RAW_RESPONSE',
  'VISUAL_ROLE',
  'FRAGMENT_SMILES',
  'MOLNEXTR_EXPECTED_STRUCTURE_TYPE',
  'MOLNEXTR_EXPERT_WEIGHTS',
  'MOLNEXTR_ROUTED_EXPERT',
  'MOLNEXTR_ROUTING_CONFIDENCE',
  'MOLNEXTR_ROUTING_FORCED_COMPLETE',
  'MOLNEXTR_ROUTING_FORCED_DEFAULT',
  'MOLNEXTR_ROUTING_REQUIRED_THRESHOLD',
  'MOLNEXTR_ROUTING_STRATEGY',
  'MOLNEXTR_CONFIDENCE',
  'MARKUSH_ASSEMBLY_STATUS',
  'MARKUSH_SCAFFOLD_REF',
  'MARKUSH_FRAGMENT_REFS',
  'MARKUSH_SCAFFOLD_PAGE',
  'MARKUSH_SCAFFOLD_IMAGE_FILE',
  'MARKUSH_SCAFFOLD_BOX_COORDS_FILE',
  'MARKUSH_SCAFFOLD_PAGE_IMAGE_FILE',
  'MARKUSH_VARIABLE_POSITIONS',
  'MARKUSH_ASSEMBLY_METHOD',
  'MARKUSH_NORMALIZATION_NOTES',
  'MARKUSH_SCAFFOLD_CONFIDENCE',
  'MARKUSH_FRAGMENT_CONFIDENCES',
  'MARKUSH_ASSEMBLY_VISUAL_REVIEW',
  'STRUCTURE_CONFIDENCE_REVIEW',
  'STRUCTURE_POSE_CLEANUP',
]);
const STRUCTURE_COLUMN_LABELS: Record<string, string> = {
  COMPOUND_ID: 'Compound ID',
  COMPOUND_NAME: 'Compound name',
  ANNOTATION_TEXT: 'Source annotation',
};

const StepGlyph: React.FC<{ type: StepKey; className?: string }> = ({ type, className }) => {
  switch (type) {
    case 'upload':
      return (
        <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
          <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
          <path d="M14 3v5h5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
          <path d="M9 12h6M9 15h6M9 18h4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    case 'structures':
      return (
        <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M8.5 4.5 4 7v6l4.5 2.5L13 13V7L8.5 4.5Zm9.5 3-4.5 2.5v6L18 18.5 22.5 16v-6L18 7.5Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinejoin="round"
          />
        </svg>
      );
    case 'bioactivity':
      return (
        <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 20V11" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M11 20V5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M17 20V9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M3 20h18" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      );
    case 'review':
    default:
      return (
        <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
          <path d="M7 5h10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <path d="M7 9h6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <path d="m9 17 2 2 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          <rect x="5" y="3" width="14" height="18" rx="2" ry="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
        </svg>
      );
  }
};

const Icon: React.FC<{
  name: 'first' | 'previous' | 'next' | 'last' | 'open' | 'cancel' | 'download' | 'play' | 'go' | 'search' | 'table' | 'cards' | 'filter' | 'check';
  className?: string;
}> = ({ name, className }) => {
  if (name === 'table' || name === 'cards' || name === 'filter' || name === 'check') {
    const paths = {table: 'M3 3h18v18H3z M3 9h18 M9 9v12 M3 15h18', cards: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z', filter: 'M4 6h16 M7 12h10 M10 18h4', check: 'm5 12 4 4 10-10'};
    return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
  }
  if (name === 'search') {
    return <svg className={className} viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" strokeWidth="1.6" /><path d="m15 15 5 5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>;
  }
  if (name === 'open') {
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <path d="M14 5h5v5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        <path d="m13 11 6-6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M19 14v4a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    );
  }
  if (name === 'cancel') {
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="1.7" />
        <path d="m9 9 6 6M15 9l-6 6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }
  if (name === 'download') {
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 4v10" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="m8 10 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M5 19h14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }
  if (name === 'play') {
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 5.8v12.4L18 12 8 5.8Z" fill="currentColor" />
      </svg>
    );
  }
  if (name === 'go') {
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 12h13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="m13 6 6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (name === 'first' || name === 'last') {
    const flip = name === 'last' ? 'scale(-1 1) translate(-24 0)' : undefined;
    return (
      <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
        <g transform={flip}>
          <path d="M6 5v14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="m18 6-7 6 7 6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </g>
      </svg>
    );
  }
  const flip = name === 'next' ? 'scale(-1 1) translate(-24 0)' : undefined;
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <g transform={flip}>
        <path d="m15 6-7 6 7 6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </g>
    </svg>
  );
};

interface CompoundIdInputProps {
  initialValue: string;
  rowIndex: number;
  disabled?: boolean;
  onSave: (rowIndex: number, value: string) => void;
}

const CompoundIdInput: React.FC<CompoundIdInputProps> = ({ initialValue, rowIndex, disabled, onSave }) => {
  const [value, setValue] = React.useState(initialValue);
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      onSave(rowIndex, value);
      event.stopPropagation();
    }
  };

  const handleBlur = () => {
    onSave(rowIndex, value);
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValue(event.target.value);
  };

  return (
    <input
      ref={inputRef}
      type="text"
      className="page-cell__id-input"
      value={value}
      placeholder="Compound ID"
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
      onChange={handleChange}
      disabled={disabled}
    />
  );
};

function parsePagesInput(input: string): number[] {
  if (!input.trim()) {
    return [];
  }
  const pages = new Set<number>();
  for (const rawPart of input.split(',')) {
    const part = rawPart.trim();
    if (!part) continue;
    if (part.includes('-')) {
      const [startStr, endStr] = part.split('-', 2);
      const start = Number(startStr);
      const end = Number(endStr);
      if (Number.isInteger(start) && Number.isInteger(end)) {
        const min = Math.min(start, end);
        const max = Math.max(start, end);
        for (let page = min; page <= max; page += 1) {
          if (page > 0) pages.add(page);
        }
      }
    } else {
      const value = Number(part);
      if (Number.isInteger(value) && value > 0) {
        pages.add(value);
      }
    }
  }
  return Array.from(pages).sort((a, b) => a - b);
}

function pagesToString(pages: number[]): string {
  return pages.join(', ');
}

function parsePageListParam(value: unknown): number[] {
  if (Array.isArray(value)) {
    const pages = value
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item > 0);
    return Array.from(new Set(pages)).sort((a, b) => a - b);
  }
  if (typeof value === 'string') {
    return parsePagesInput(value);
  }
  return [];
}

function parseStringListParam(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : String(item ?? '').trim()))
    .filter(Boolean);
}

function isTaskInFlight(task: TaskStatus | null): boolean {
  return Boolean(task && (task.status === 'running' || task.status === 'pending'));
}

function isTaskTerminal(task: TaskStatus | null): boolean {
  return Boolean(task && (task.status === 'completed' || task.status === 'failed' || task.status === 'canceled'));
}

function isCancelableTask(task: TaskStatus | null): boolean {
  return Boolean(task && !task.task_id.startsWith('pending-') && (task.status === 'pending' || task.status === 'running'));
}

function getTaskDisplayProgress(task: TaskStatus | null): number {
  if (!task) return 0;
  if (task.status === 'completed' || task.status === 'canceled') return 1;
  const progress = Number(task.progress ?? 0);
  if (!Number.isFinite(progress)) return 0;
  return Math.max(0, Math.min(1, progress));
}

function isHttpNotFound(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const response = (err as { response?: { status?: number } }).response;
  return response?.status === 404;
}

function formatTaskType(type: string): string {
  const labels: Record<string, string> = {
    auto_detect_plan: 'Detection plan',
    structure_extraction: 'Structure extraction',
    bioactivity_extraction: 'Bioactivity extraction',
    full_pipeline: 'Document extraction',
    data_merge: 'Merge',
  };
  return labels[type] ?? type.replace(/_/g, ' ');
}

function formatTaskTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatTaskDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString([], {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

const jobTypeOptions = [
  'auto_detect_plan',
  'structure_extraction',
  'bioactivity_extraction',
  'full_pipeline',
  'data_merge',
];

function readStoredJson<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function storeJson(key: string, value: unknown): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* localStorage can be unavailable; ignore persistence only */
  }
}

const REVIEW_VIEW_STORAGE_KEY = 'bci:review:view';
const REVIEW_COLUMNS_STORAGE_KEY = 'bci:review:columns';

function buildAddonStorageKey(kind: 'structure' | 'assay', pdfId?: string, taskId?: string): string | null {
  if (!isUsableId(pdfId) || !isUsableId(taskId)) return null;
  return `bci:${kind}:pending-addon:${pdfId}:${taskId}`;
}

function loadStoredPages(key: string | null): number[] {
  if (!key || typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return [];
    return parsePageListParam(JSON.parse(raw));
  } catch {
    return [];
  }
}

function storePages(key: string | null, pages: number[]): void {
  if (!key || typeof window === 'undefined') return;
  try {
    if (!pages.length) {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, JSON.stringify(pages));
    }
  } catch {
    /* localStorage can be unavailable; ignore persistence only */
  }
}

function getTaskSelectedPages(task: TaskStatus | null): number[] {
  if (!task) return [];
  const detected = parsePageListParam(task.params?.detected_pages);
  if (detected.length) return detected;
  return parsePageListParam(task.params?.pages);
}

function summarizePageDelta(nextPages: number[], taskPages: number[]): string {
  const nextSet = new Set(nextPages);
  const taskSet = new Set(taskPages);
  const added = nextPages.filter((page) => !taskSet.has(page));
  const removed = taskPages.filter((page) => !nextSet.has(page));
  const parts: string[] = [];
  if (added.length) {
    parts.push(`added ${added.length} page${added.length === 1 ? '' : 's'}`);
  }
  if (removed.length) {
    parts.push(`removed ${removed.length} page${removed.length === 1 ? '' : 's'}`);
  }
  return parts.join(' and ');
}

function parseTaskActivePages(task: TaskStatus | null): number[] {
  if (!isTaskInFlight(task)) return [];
  const text = `${task?.message ?? ''} ${task?.params?.current_page ?? ''}`;
  const pages = new Set<number>();
  const singlePatterns = [
    /\bpage\s+(\d+)\b/gi,
    /\bcurrent[_\s-]*page\s*[:=]?\s*(\d+)\b/gi,
  ];
  singlePatterns.forEach((pattern) => {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text))) {
      const page = Number(match[1]);
      if (Number.isInteger(page) && page > 0) pages.add(page);
    }
  });
  const rangePattern = /\bpages\s+(\d+)\s*[-–]\s*(\d+)\b/gi;
  let rangeMatch: RegExpExecArray | null;
  while ((rangeMatch = rangePattern.exec(text))) {
    const start = Number(rangeMatch[1]);
    const end = Number(rangeMatch[2]);
    if (!Number.isInteger(start) || !Number.isInteger(end)) continue;
    const min = Math.max(1, Math.min(start, end));
    const max = Math.max(start, end);
    for (let page = min; page <= max && page < min + 24; page += 1) {
      pages.add(page);
    }
  }
  return Array.from(pages).sort((a, b) => a - b);
}

function getRecordMergeKey(record: Record<string, unknown>): string {
  // Sequence is deliberately excluded: extras from an add-on run re-index
  // from base.length, so including it made dedup between base and extras
  // impossible even for identical rows.
  return [
    record.COMPOUND_ID ?? '',
    record.PAGE_NUM ?? record.page_num ?? record.page ?? '',
    record.SMILES ?? '',
    record.SEGMENT_FILE ?? record.Segment ?? '',
  ].join('|');
}

function mergeStructureRecordLists(base: StructureRecord[], extras: StructureRecord[]): StructureRecord[] {
  const merged = [...base.map((record) => ({ ...record }))];
  const seen = new Set(merged.map((record) => getRecordMergeKey(record as Record<string, unknown>)));
  extras.forEach((record, index) => {
    const cloned = { ...record };
    const key = getRecordMergeKey(cloned as Record<string, unknown>);
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(cloned);
  });
  return merged;
}

function mergeAssayRecordLists(base: AssayRecord[], extras: AssayRecord[]): AssayRecord[] {
  const byCompound = new Map<string, AssayRecord>();
  base.forEach((record, index) => {
    const key = formatCellValue(record.COMPOUND_ID).trim() || `__base_${index}`;
    byCompound.set(key, { ...record });
  });
  extras.forEach((record, index) => {
    const key = formatCellValue(record.COMPOUND_ID).trim() || `__extra_${index}`;
    byCompound.set(key, { ...(byCompound.get(key) ?? {}), ...record });
  });
  return Array.from(byCompound.values());
}

function parseAssayNames(input: string): string[] {
  if (!input.trim()) return [];
  const seen = new Set<string>();
  const names: string[] = [];
  input
    .split(/[;,\n\r]/)
    .map((name) => name.trim())
    .filter(Boolean)
    .forEach((name) => {
      const key = name.toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      names.push(name);
    });
  return names;
}

function mergeAssayNameLists(base: string[], extras: string[]): string[] {
  if (!extras.length) return [...base];
  const existing = new Set(base.map((name) => name.toLowerCase()));
  const combined = [...base];
  extras.forEach((name) => {
    const key = name.toLowerCase();
    if (existing.has(key)) return;
    existing.add(key);
    combined.push(name);
  });
  return combined;
}

function buildColumns(records: Array<Record<string, unknown>>): string[] {
  if (!records.length) return [];
  const keys = new Set<string>();
  records.forEach((record) => {
    Object.keys(record).forEach((key) => keys.add(key));
  });
  const preferred = preferredColumnOrder.filter((col) => keys.has(col));
  const rest = Array.from(keys).filter((key) => !preferred.includes(key));
  rest.sort();
  return [...preferred, ...rest];
}

const readOnlyColumnPrefixes = ['image', 'segment'];

function isEditableColumn(key: string, value: unknown): boolean {
  if (Array.isArray(value)) return false;
  if (value === null || value === undefined) return true;
  if (typeof value === 'number') return true;
  if (typeof value === 'string') {
    const lowered = key.toLowerCase();
    if (readOnlyColumnPrefixes.some((prefix) => lowered.includes(prefix))) {
      return false;
    }
    return true;
  }
  return false;
}

function formatCellValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(', ');
  }
  if (value === null || value === undefined) {
    return '';
  }
  return String(value);
}

function hasReviewCellValue(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some((item) => hasReviewCellValue(item));
  }
  return formatCellValue(value).trim().length > 0;
}

function isAssayMetadataColumn(key: string): boolean {
  return key === 'RECORD_ID' || /__(method|description|confidence|reason|source_kind)$/i.test(key);
}

function getRecordPrimaryPageNumber(record: Record<string, unknown>): number | null {
  const direct = record.PAGE_NUM ?? record.page_num ?? record.page;
  const directNumber = Number(formatCellValue(direct).trim());
  if (Number.isInteger(directNumber) && directNumber > 0) {
    return directNumber;
  }
  const sourcePages = parsePageListParam(record.source_pages);
  return sourcePages[0] ?? null;
}

function cropPageImageFromBox(
  pageDataUri: string,
  box: number[],
  sourceSize?: { width: number; height: number } | null,
): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!box || box.length !== 4) {
      reject(new Error('Invalid crop box'));
      return;
    }
    const image = new Image();
    image.onload = () => {
      try {
        // Detection boxes are stored in XYXY order (see highlight json / detector output).
        const [rawX1, rawY1, rawX2, rawY2] = box.map((value) => Number(value));
        if (![rawY1, rawX1, rawY2, rawX2].every((value) => Number.isFinite(value))) {
          throw new Error('Invalid crop coordinates');
        }
        const scaleX = sourceSize?.width ? image.naturalWidth / sourceSize.width : 1;
        const scaleY = sourceSize?.height ? image.naturalHeight / sourceSize.height : 1;
        let x1 = rawX1 * scaleX;
        let y1 = rawY1 * scaleY;
        let x2 = rawX2 * scaleX;
        let y2 = rawY2 * scaleY;
        const padding = Math.max(12, Math.round(Math.max(x2 - x1, y2 - y1) * 0.15));
        x1 = Math.max(0, Math.floor(x1 - padding));
        y1 = Math.max(0, Math.floor(y1 - padding));
        x2 = Math.min(image.naturalWidth, Math.ceil(x2 + padding));
        y2 = Math.min(image.naturalHeight, Math.ceil(y2 + padding));
        const width = Math.max(1, x2 - x1);
        const height = Math.max(1, y2 - y1);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          throw new Error('Canvas is unavailable');
        }
        ctx.drawImage(image, x1, y1, width, height, 0, 0, width, height);
        resolve(canvas.toDataURL('image/png'));
      } catch (err) {
        reject(err);
      }
    };
    image.onerror = () => reject(new Error('Failed to load page image'));
    image.src = pageDataUri;
  });
}

function withBoundedArtifactCache(
  previous: Record<string, string>, key: string, value: string,
  visiblePaths: ReadonlySet<string>,
): Record<string, string> {
  if (previous[key]) return previous;
  const entries = [...Object.entries(previous), [key, value]];
  const visible = entries.filter(([path]) => path === key || visiblePaths.has(path));
  const inactive = entries.filter(([path]) => path !== key && !visiblePaths.has(path));
  const remaining = Math.max(0, ARTIFACT_CACHE_TARGET_ENTRIES - visible.length);
  // A large screen can show more images than the usual cache target. Keep
  // every visible image and evict off-screen ones first, avoiding refetch loops.
  return Object.fromEntries([...(remaining ? inactive.slice(-remaining) : []), ...visible]);
}

function isUsableId(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '' && value !== 'undefined' && value !== 'null';
}

function isPendingAssayTaskId(value: unknown): value is string {
  return isUsableId(value) && value.startsWith('pending-') && isUsableId(value.slice('pending-'.length));
}

type SelectionMode = 'structure' | 'assay';

interface ArtifactPreview {
  path: string;
  mime: string;
  data: string;
  rowIndex: number | null;
  kind?: 'image' | 'extracted-structure';
}

function getSelectionModeForStep(step: StepId): SelectionMode {
  return step === 3 ? 'assay' : 'structure';
}

const App: React.FC = () => {
  const [pdfInfo, setPdfInfo] = React.useState<UploadPDFResponse | null>(null);

  const [structurePagesInput, setStructurePagesInput] = React.useState('');
  const [assayPagesInput, setAssayPagesInput] = React.useState('');
  const [structureSelection, setStructureSelection] = React.useState<Set<number>>(new Set());
  const [assaySelection, setAssaySelection] = React.useState<Set<number>>(new Set());
  const [autoDetectStructurePages, setAutoDetectStructurePages] = React.useState(true);
  const [autoDetectAssayPages, setAutoDetectAssayPages] = React.useState(true);
  const [autoDetectAssayNames, setAutoDetectAssayNames] = React.useState(true);

  const [pageImages, setPageImages] = React.useState<Record<number, string>>({});
  const [loadingPages, setLoadingPages] = React.useState<Set<number>>(new Set());

  const [structureTask, setStructureTask] = React.useState<TaskStatus | null>(null);
  const [assayTask, setAssayTask] = React.useState<TaskStatus | null>(null);
  const [structureAddonTask, setStructureAddonTask] = React.useState<TaskStatus | null>(null);
  const [assayAddonTask, setAssayAddonTask] = React.useState<TaskStatus | null>(null);
  const [autoDetectTask, setAutoDetectTask] = React.useState<TaskStatus | null>(null);
  const [fullPipelineTask, setFullPipelineTask] = React.useState<TaskStatus | null>(null);
  const [isFullPipelineSubmitting, setIsFullPipelineSubmitting] = React.useState(false);
  const [structureFilterStrictness, setStructureFilterStrictness] = React.useState<StructureFilterStrictness>('strict');
  const [structureSelectionFeedback, setStructureSelectionFeedback] = React.useState<string | null>(null);
  const [assaySelectionFeedback, setAssaySelectionFeedback] = React.useState<string | null>(null);
  const [pendingStructureAddonPages, setPendingStructureAddonPages] = React.useState<number[]>([]);
  const [pendingAssayAddonPages, setPendingAssayAddonPages] = React.useState<number[]>([]);

  const [structures, setStructures] = React.useState<StructureRecord[]>([]);
  const [editedStructures, setEditedStructures] = React.useState<StructureRecord[]>([]);
  const [filteredStructures, setFilteredStructures] = React.useState<StructureRecord[]>([]);
  const [saveStatus, setSaveStatus] = React.useState<'idle' | 'pending' | 'saving' | 'saved' | 'error'>('idle');
  const [structurePreviewCache, setStructurePreviewCache] = React.useState<Record<string, string>>({});
  const structurePreviewCacheRef = React.useRef(structurePreviewCache);
  const [loadingStructurePreviews, setLoadingStructurePreviews] = React.useState<Set<string>>(new Set());
  const loadingStructurePreviewsRef = React.useRef(loadingStructurePreviews);

  const trackedTasksRef = React.useRef<Record<string, string | undefined>>({});
  trackedTasksRef.current = {
    structure: structureTask?.task_id, structureAddon: structureAddonTask?.task_id,
    assay: assayTask?.task_id, assayAddon: assayAddonTask?.task_id,
    autoDetect: autoDetectTask?.task_id, fullPipeline: fullPipelineTask?.task_id,
  };
  const [loadedStructureTaskId, setLoadedStructureTaskId] = React.useState<string>();
  const editableTaskId = [structureTask, fullPipelineTask].some(
    (task) => task?.status === 'completed' && task.task_id === loadedStructureTaskId,
  ) ? loadedStructureTaskId : undefined;
  const activeSaveTaskRef = React.useRef(editableTaskId);
  activeSaveTaskRef.current = editableTaskId;
  const persistedRowsRef = React.useRef<{ taskId: string; submitted: StructureRecord[]; returned: StructureRecord[] } | null>(null);
  const saveErrorRef = React.useRef<string | null>(null);
  const structureSaveQueueRef = React.useRef<Promise<boolean>>(Promise.resolve(true));
  const [isDownloading, setIsDownloading] = React.useState(false);
  const [isCanceling, setIsCanceling] = React.useState(false);
  const autoSaveTimerRef = React.useRef<number | null>(null);
  const editedStructuresRef = React.useRef<StructureRecord[]>([]);

  const [assayRecords, setAssayRecords] = React.useState<AssayRecord[]>([]);
  const [assayNames, setAssayNames] = React.useState<string[]>([]);
  const [assayNameDraft, setAssayNameDraft] = React.useState('');

  const [isUploading, setIsUploading] = React.useState(false);
  const [uploadProgress, setUploadProgress] = React.useState<number | null>(null);
  const [imageCache, setImageCache] = React.useState<Record<string, string>>({});
  const [loadingArtifacts, setLoadingArtifacts] = React.useState<Set<string>>(new Set());
  const [artifactErrors, setArtifactErrors] = React.useState<Record<string, string>>({});
  const activeArtifactPathsRef = React.useRef<Set<string>>(new Set());
  const artifactEpochRef = React.useRef(0);
  const previewRequestRef = React.useRef(0);
  const [sourceCropCache, setSourceCropCache] = React.useState<Record<string, string>>({});
  const [loadingSourceCrops, setLoadingSourceCrops] = React.useState<Set<string>>(new Set());
  const [pdfTarget, setPdfTarget] = React.useState<PdfTarget | null>(null);
  const [reviewView, setReviewView] = React.useState<'table' | 'cards'>(() => {
    const stored = readStoredJson<string>(REVIEW_VIEW_STORAGE_KEY, 'cards');
    return stored === 'table' || stored === 'pdf' ? 'table' : 'cards';
  });
  // result table shows a minimal set of columns by default; the rest stays
  // available through the "Columns" picker in the review toolbar
  const [activeMetadataColumns, setActiveMetadataColumns] = React.useState<Set<string>>(() => {
    const stored = readStoredJson<string[]>(REVIEW_COLUMNS_STORAGE_KEY, []);
    return new Set(Array.isArray(stored) ? stored : []);
  });
  const [columnsPickerOpen, setColumnsPickerOpen] = React.useState(false);
  const [columnSearch, setColumnSearch] = React.useState('');
  const [reviewFiltersOpen, setReviewFiltersOpen] = React.useState(false);
  const [reviewSearchOpen, setReviewSearchOpen] = React.useState(false);
  const [reviewFocusMode, setReviewFocusMode] = React.useState(false);
  const [reviewAddKind, setReviewAddKind] = React.useState<'compound' | 'measurement' | null>(null);
  const [sarOpen, setSarOpen] = React.useState(false);
  const [sarReport, setSarReport] = React.useState<{ coverage: { open_gaps: number; complete: boolean }; gaps: Array<{ gap_id: string; row_index?: number; page?: number; reason: string }>; next_actions: Array<{ operation: string; reason: string }> } | null>(null);
  const [assembleMarkush, setAssembleMarkush] = React.useState(false);
  const [assemblyAvailable, setAssemblyAvailable] = React.useState(false);
  const [assemblyChecked, setAssemblyChecked] = React.useState(false);
  React.useEffect(() => {
    let active = true;
    void api.get('/capabilities').then(({ data }) => { if (active) setAssemblyAvailable(data?.markush_assembly?.available === true); })
      .catch(() => undefined).finally(() => { if (active) setAssemblyChecked(true); });
    return () => { active = false; };
  }, []);
  const [stageOptionsOpen, setStageOptionsOpen] = React.useState(false);
  const [documentPage, setDocumentPage] = React.useState<number>(1);
  React.useEffect(() => { setDocumentPage(1); }, [pdfInfo?.pdf_id]);
  React.useEffect(() => {
    if (!loadedStructureTaskId) return;
    const page = editedStructures.map(getRecordPrimaryPageNumber).find((value) => value !== null);
    if (page) setDocumentPage(page);
  }, [loadedStructureTaskId]);
  React.useEffect(() => {
    if (!stageOptionsOpen) return;
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') { setStageOptionsOpen(false); (document.querySelector('.stage-options-toggle') as HTMLButtonElement | null)?.focus(); } };
    const outside = (event: PointerEvent) => { if (!(event.target as Element)?.closest('.selector-toolbar,.stage-options-toggle')) setStageOptionsOpen(false); };
    window.addEventListener('keydown', close); window.addEventListener('pointerdown', outside);
    return () => { window.removeEventListener('keydown', close); window.removeEventListener('pointerdown', outside); };
  }, [stageOptionsOpen]);
  const [reviewDensity, setReviewDensity] = React.useState(() => readStoredJson<string>('bci:review:density', 'compact') === 'roomy' ? 'roomy' : 'compact');
  const [showReviewSmiles, setShowReviewSmiles] = React.useState(false);
  const [showPageThumbnails, setShowPageThumbnails] = React.useState(false);
  const sourcePaneVisible = Boolean(pdfInfo?.pdf_id);
  const pdfLocateSequenceRef = React.useRef(0);
  const [activeReviewRowIndex, setActiveReviewRowIndex] = React.useState<number | null>(null);
  const [editorState, setEditorState] = React.useState<{
    open: boolean;
    rowIndex: number | null;
    smiles: string;
    molblock: string;
  }>({ open: false, rowIndex: null, smiles: '', molblock: '' });
  const [currentStep, setCurrentStep] = React.useState<StepId>(1);
  const autoAdvanceRef = React.useRef(false);
  const suppressAutoStepUntilNextJobRef = React.useRef(false);
  const pdfInitializedRef = React.useRef(false);
  const activeSelectionMode = React.useMemo<SelectionMode>(() => getSelectionModeForStep(currentStep), [currentStep]);
  const isGalleryLoading = loadingPages.size > 0;
  const pdfReadyForDetection = Boolean(
    pdfInfo &&
      !isUploading &&
      !isGalleryLoading &&
      Object.keys(pageImages).length > 0,
  );
  const activeStructurePages = React.useMemo(
    () =>
      new Set([
        ...parseTaskActivePages(structureTask),
        ...parseTaskActivePages(structureAddonTask),
        ...(fullPipelineTask && fullPipelineTask.progress < 0.55
          ? parseTaskActivePages(fullPipelineTask)
          : []),
      ]),
    [structureTask, structureAddonTask, fullPipelineTask],
  );
  const activeAssayPages = React.useMemo(
    () =>
      new Set([
        ...parseTaskActivePages(assayTask),
        ...parseTaskActivePages(assayAddonTask),
        ...(fullPipelineTask && fullPipelineTask.progress >= 0.55
          ? parseTaskActivePages(fullPipelineTask)
          : []),
      ]),
    [assayTask, assayAddonTask, fullPipelineTask],
  );

  const [error, setError] = React.useState<string | null>(null);
  const [toast, setToast] = React.useState<string | null>(null);
  const [jobsOpen, setJobsOpen] = React.useState(false);
  const [jobsInfo, setJobsInfo] = React.useState<TaskListResponse | null>(null);
  const [jobsLoading, setJobsLoading] = React.useState(false);
  const [jobsError, setJobsError] = React.useState<string | null>(null);
  const [verificationMode, setVerificationMode] = React.useState(false);
  const [jobsPage, setJobsPage] = React.useState(1);
  const [jobsPageSize, setJobsPageSize] = React.useState(20);
  const [jobsSearchInput, setJobsSearchInput] = React.useState('');
  const [jobsSearch, setJobsSearch] = React.useState('');
  const [jobsStatusFilter, setJobsStatusFilter] = React.useState('all');
  const [jobsTypeFilter, setJobsTypeFilter] = React.useState('all');
  const [jobsDateFrom, setJobsDateFrom] = React.useState(() => getDefaultJobsDateFrom());
  const [jobsDateTo, setJobsDateTo] = React.useState('');
  const [jobsSortBy, setJobsSortBy] = React.useState<JobSortBy>('updated_at');
  const [jobsSortDir, setJobsSortDir] = React.useState<SortDir>('desc');
  const jobsRequestSeqRef = React.useRef(0);
  const jobsQuerySignatureRef = React.useRef('');
  const jobsInfoSignatureRef = React.useRef('');
  const jobsInFlightRef = React.useRef(false);
  const openJobRequestSeqRef = React.useRef(0);
  const [selectedJobId, setSelectedJobId] = React.useState<string | null>(null);
  const [modalArtifact, setModalArtifact] = React.useState<ArtifactPreview | null>(null);
  const [modalBox, setModalBox] = React.useState<number[] | null>(null);
  const [originalImageSize, setOriginalImageSize] = React.useState<{ width: number; height: number } | null>(null);
  const [boxStyle, setBoxStyle] = React.useState<React.CSSProperties>({});
  const imageDialogRef = React.useRef<HTMLDivElement | null>(null);
  const modalImageRef = React.useRef<HTMLImageElement>(null);
  const [isStructureSubmitting, setIsStructureSubmitting] = React.useState(false);
  const [isAssaySubmitting, setIsAssaySubmitting] = React.useState(false);
  const [showScrollTop, setShowScrollTop] = React.useState(false);
  const [modalDragState, setModalDragState] = React.useState<{
    isDragging: boolean;
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    element: HTMLElement | null;
  } | null>(null);
  const [rowHeight, setRowHeight] = React.useState(() => readStoredJson<number>('bci:review:row-height', 104));
  const [modalEditorPosition, setModalEditorPosition] = React.useState<{ x: number; y: number }>({ x: 32, y: 32 });
  const modalEditorPanelRef = React.useRef<HTMLDivElement | null>(null);
  const modalInlineEditorRef = React.useRef<StructureEditorInlineHandle | null>(null);
  const modalEditorDragRef = React.useRef<{ offsetX: number; offsetY: number } | null>(null);
  const tableWrapperRef = React.useRef<HTMLDivElement | null>(null);
  const reviewCardsRef = React.useRef<HTMLDivElement | null>(null);
  const tableBodyRef = React.useRef<HTMLTableSectionElement | null>(null);
  const [visibleRowIndices, setVisibleRowIndices] = React.useState<Set<number>>(() => new Set());
  const [reviewSearch, setReviewSearch] = React.useState('');
  const gridLayout = useReviewLayout(editableTaskId || selectedJobId || 'workspace');
  const reviewRowKey = (row: { index: number; record: StructureRecord }) => String(row.record.RECORD_ID || row.record.BOX_COORDS_FILE || `row:${row.index}`);
  const [reviewRequiredColumn, setReviewRequiredColumn] = React.useState('');
  const [reviewFilterOperator, setReviewFilterOperator] = React.useState('filled');
  const [reviewFilterValue, setReviewFilterValue] = React.useState('');
  const [reviewSortColumn, setReviewSortColumn] = React.useState('');
  const [reviewSortDescending, setReviewSortDescending] = React.useState(false);
  const hiddenMeasurements = new Set(gridLayout.hidden);
  const setHiddenMeasurements = (next: Set<string> | ((previous: Set<string>) => Set<string>)) => gridLayout.update({ hidden: [...(typeof next === 'function' ? next(hiddenMeasurements) : next)] });
  const [renamingMeasurement, setRenamingMeasurement] = React.useState<string | null>(null);
  const [measurementNameDraft, setMeasurementNameDraft] = React.useState('');
  const [reviewPage, setReviewPage] = React.useState(1);
  const [reviewPageSize, setReviewPageSize] = React.useState(25);
  // developer mode: activated only via ?debug=1 in the URL, never a visible control
  const [debugAnnotationMode, setDebugAnnotationMode] = React.useState(
    () => new URLSearchParams(window.location.search).get('debug') === '1',
  );
  const navigateReviewRowRef = React.useRef<(row: { index: number; record: Record<string, unknown> }) => void>(() => undefined);
  const pendingAssayRequestRef = React.useRef<AssayTaskRequest | null>(null);
  const [isAssayWaitingForStructures, setIsAssayWaitingForStructures] = React.useState(false);
  const lastStructurePageRef = React.useRef<number | null>(null);
  const lastAssayPageRef = React.useRef<number | null>(null);
  const appliedStructureDetectedPagesRef = React.useRef('');
  const appliedAssayDetectedPagesRef = React.useRef('');
  const appliedAssayNamesRef = React.useRef('');
  const automaticExtractionRunRef = React.useRef(0);
  const structureAddonSubmittedPagesRef = React.useRef<number[]>([]);
  const assayAddonSubmittedPagesRef = React.useRef<number[]>([]);
  const loadedStructureAddonStorageKeyRef = React.useRef<string | null>(null);
  const loadedAssayAddonStorageKeyRef = React.useRef<string | null>(null);
  const skipNextStructureAddonStoreRef = React.useRef(false);
  const skipNextAssayAddonStoreRef = React.useRef(false);
  const pdfInfoRef = React.useRef<UploadPDFResponse | null>(pdfInfo);
  const pageImagesRef = React.useRef(pageImages);
  const loadingPagesRef = React.useRef(loadingPages);
  const loadingArtifactsRef = React.useRef(loadingArtifacts);
  const sourceCropCacheRef = React.useRef(sourceCropCache);
  const loadingSourceCropsRef = React.useRef(loadingSourceCrops);

  const structureDisplayColumns = React.useMemo(() => {
    const keys = new Set<string>();
    editedStructures.forEach((record) => {
      Object.keys(record).forEach((key) => {
        if (!STRUCTURE_IGNORED_COLUMNS.has(key) && !/^(USER_|PREVIOUS_|LOCAL_EVIDENCE_|MEASUREMENT_REVIEW_)/.test(key) && !key.includes('__')) {
          keys.add(key);
        }
      });
    });
    editedStructures.forEach((record) => Object.keys(measurementValues(record)).forEach((column) => keys.delete(column)));
    assayRecords.forEach((record) => Object.keys(record).filter((key) => key !== 'COMPOUND_ID').forEach((column) => keys.delete(column)));
    keys.delete('SMILES');
    const baseOrder = ['COMPOUND_ID'];
    const preferred = preferredColumnOrder.filter((col) => keys.has(col));
    const rest = Array.from(keys).filter((col) => !baseOrder.includes(col) && !preferred.includes(col));
    rest.sort();
    return Array.from(new Set([...baseOrder, ...preferred, ...rest]));
  }, [editedStructures, assayRecords]);
  const structureColumnsToRender = React.useMemo(
    () => structureDisplayColumns.filter((column) => column !== 'SMILES' && column !== 'COMPOUND_ID'),
    [structureDisplayColumns],
  );
  // only the columns enabled in the "Columns" picker are rendered in the table
  const activeStructureColumns = React.useMemo(
    () => structureColumnsToRender.filter((column) => activeMetadataColumns.has(column)),
    [structureColumnsToRender, activeMetadataColumns],
  );
  const extractImageSource = React.useCallback(
    (value: unknown): string | null => {
      if (typeof value !== 'string' || !value) return null;
      if (isDataImage(value)) return value;
      const match = value.match(markdownImageRegex);
      if (match) {
        return match[1];
      }
      if (imageCache[value]) {
        return imageCache[value];
      }
      return null;
    },
    [imageCache],
  );
  const performAutoSave = React.useCallback((): Promise<boolean> => {
    if (!editableTaskId) return Promise.resolve(false);
    const taskId = editableTaskId;
    const payload = editedStructuresRef.current;
    const save = async (): Promise<boolean> => {
      if (activeSaveTaskRef.current !== taskId) return false;
      const persisted = persistedRowsRef.current;
      if (persisted?.taskId === taskId && (payload === persisted.submitted || payload === persisted.returned)) return true;
      setSaveStatus('saving');
      try {
        const response = await updateTaskStructures(taskId, payload);
        if (activeSaveTaskRef.current !== taskId) return true;
        const nextRecords = response.records.map((row) => ({ ...row }));
        persistedRowsRef.current = { taskId, submitted: payload, returned: nextRecords };
        if (editedStructuresRef.current !== payload) {
          setSaveStatus('pending');
          return true;
        }
        setStructures(response.records);
        editedStructuresRef.current = nextRecords;
        setEditedStructures(nextRecords);
        setFilteredStructures((response.filtered_records ?? []).map((row) => ({ ...row })));
        setSaveStatus('saved');
        const previousSaveError = saveErrorRef.current;
        if (previousSaveError) setError((current) => current === previousSaveError ? null : current);
        saveErrorRef.current = null;
        return true;
      } catch (err) {
        if (activeSaveTaskRef.current === taskId) {
          setSaveStatus('error');
          const message = err instanceof Error ? err.message : 'Failed to save structures';
          saveErrorRef.current = message;
          setError(message);
        }
        return false;
      }
    };
    // Serialize PUTs: a slower old request must never finish after a newer edit.
    const pending = structureSaveQueueRef.current.then(save, save);
    structureSaveQueueRef.current = pending;
    return pending;
  }, [editableTaskId]);


  const flushPendingEdits = React.useCallback(async () => {
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    if (editableTaskId && (saveStatus === 'pending' || saveStatus === 'saving' || saveStatus === 'error')) {
      return performAutoSave();
    }
    return true;
  }, [editableTaskId, performAutoSave, saveStatus]);

  const getMolblockValue = React.useCallback((record: StructureRecord) => {
    const raw =
      (record.MOLBLOCK as string | null | undefined) ??
      (record.molblock as string | null | undefined) ??
      (record.Molblock as string | null | undefined);
    return typeof raw === 'string' ? raw.trimEnd() : '';
  }, []);
  const getStructurePreviewKey = React.useCallback((smilesValue: string, molblockValue: string) => {
    if (molblockValue) {
      return `molblock:${molblockValue.length}:${hashPreviewInput(molblockValue)}`;
    }
    return smilesValue ? `smiles:${smilesValue.length}:${hashPreviewInput(smilesValue)}` : '';
  }, []);
  const structureRows = React.useMemo(
    () =>
      editedStructures
        .map((record, index) => ({ record, index }))
        .filter(({ record }) => {
          // Filter out records that were rejected by visual verification
          // (FILTERED_OUT=True, SMILES cleared).  These are wrong molecules
          // that should not appear in the results list.
          const filteredOut = String(record.FILTERED_OUT ?? '').toLowerCase();
          return filteredOut !== 'true';
        })
        .map(({ record, index }) => {
          const smilesValue = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
        const molblockValue = getMolblockValue(record);
        const previewKey = getStructurePreviewKey(smilesValue, molblockValue);
        const rawPreview = previewKey ? structurePreviewCache[previewKey] : undefined;
        const structurePreview = rawPreview && rawPreview.length > 0 ? rawPreview : null;
        const structureImage =
          extractImageSource(record.Structure) ??
          extractImageSource(record.PAGE_IMAGE_FILE) ??
          extractImageSource(record.IMAGE_FILE) ??
          extractImageSource(record.SEGMENT_FILE) ??
          structurePreview;
        const primaryPageNumber = getRecordPrimaryPageNumber(record as Record<string, unknown>);
        const cardPageImage =
          primaryPageNumber && pageImages[primaryPageNumber]
            ? `data:image/png;base64,${pageImages[primaryPageNumber]}`
            : null;
        const pagePreviewImage = cardPageImage;
        const pagePreviewSource =
          (typeof record.PAGE_IMAGE_FILE === 'string'
            ? record.PAGE_IMAGE_FILE
            : typeof record.IMAGE_FILE === 'string'
            ? record.IMAGE_FILE
            : '');
        const boxCoordsPath = typeof record.BOX_COORDS_FILE === 'string' ? record.BOX_COORDS_FILE : '';
        const sourceCropKey =
          primaryPageNumber && boxCoordsPath && pageImages[primaryPageNumber]
            ? `${boxCoordsPath}|${primaryPageNumber}|${pageImages[primaryPageNumber].length}`
            : '';
        const segmentImage = (sourceCropKey ? sourceCropCache[sourceCropKey] : null)
          || extractImageSource(record.Segment) || extractImageSource(record.SEGMENT_FILE) || null;
        const isMarkushAssembled =
          (record.STRUCTURE_TYPE as string | undefined)?.trim() === 'markush_assembled';
        const scaffoldBoxCoordsPath =
          typeof record.MARKUSH_SCAFFOLD_BOX_COORDS_FILE === 'string'
            ? record.MARKUSH_SCAFFOLD_BOX_COORDS_FILE
            : '';
        const scaffoldPageRaw = record.MARKUSH_SCAFFOLD_PAGE;
        const scaffoldPageParsed =
          typeof scaffoldPageRaw === 'number' ? scaffoldPageRaw : Number(scaffoldPageRaw);
        const scaffoldPageNumber =
          Number.isFinite(scaffoldPageParsed) && scaffoldPageParsed > 0 ? scaffoldPageParsed : null;
        const scaffoldSourceCropKey =
          isMarkushAssembled && scaffoldBoxCoordsPath && scaffoldPageNumber && pageImages[scaffoldPageNumber]
            ? `${scaffoldBoxCoordsPath}|${scaffoldPageNumber}|${pageImages[scaffoldPageNumber].length}`
            : '';
        const scaffoldSegmentImage = scaffoldSourceCropKey
          ? sourceCropCache[scaffoldSourceCropKey] || null
          : null;
        const scaffoldImageFilePath =
          isMarkushAssembled && typeof record.MARKUSH_SCAFFOLD_IMAGE_FILE === 'string'
            ? record.MARKUSH_SCAFFOLD_IMAGE_FILE
            : '';
        const scaffoldImageFromFile = scaffoldImageFilePath && imageCache[scaffoldImageFilePath]
          ? imageCache[scaffoldImageFilePath]
          : null;
        const displayedScaffoldImage = scaffoldSegmentImage ?? scaffoldImageFromFile;
        const scaffoldPagePreviewImage =
          isMarkushAssembled && scaffoldPageNumber && pageImages[scaffoldPageNumber]
            ? `data:image/png;base64,${pageImages[scaffoldPageNumber]}`
            : null;
        const scaffoldPagePreviewSource =
          isMarkushAssembled && typeof record.MARKUSH_SCAFFOLD_PAGE_IMAGE_FILE === 'string'
            ? record.MARKUSH_SCAFFOLD_PAGE_IMAGE_FILE
            : '';
        const fragmentPagePreviewImage =
          isMarkushAssembled && primaryPageNumber && pageImages[primaryPageNumber]
            ? `data:image/png;base64,${pageImages[primaryPageNumber]}`
            : null;
          return {
          id: (record.COMPOUND_ID ?? '').toString(),
          record,
          index,
          structureImage,
          pagePreviewImage: isMarkushAssembled ? scaffoldPagePreviewImage : pagePreviewImage,
          pagePreviewSource: isMarkushAssembled ? scaffoldPagePreviewSource : pagePreviewSource,
          fragmentPagePreviewImage: isMarkushAssembled ? fragmentPagePreviewImage : null,
          fragmentPagePreviewSource: isMarkushAssembled ? pagePreviewSource : '',
          primaryPageNumber,
          sourceCropKey,
          boxCoordsPath,
          segmentImage,
          isMarkushAssembled,
          scaffoldSourceCropKey,
          scaffoldBoxCoordsPath,
          scaffoldPageNumber,
          scaffoldSegmentImage,
          displayedScaffoldImage,
          structurePreviewKey: previewKey,
          structurePreview,
          structureSource:
            structureImage ||
            (typeof record.Structure === 'string'
              ? record.Structure
              : typeof record.PAGE_IMAGE_FILE === 'string'
              ? record.PAGE_IMAGE_FILE
              : typeof record.IMAGE_FILE === 'string'
              ? record.IMAGE_FILE
              : typeof record.SEGMENT_FILE === 'string'
              ? record.SEGMENT_FILE
              : structurePreview ?? ''),
          segmentSource:
            extractImageSource(record.Segment) && typeof record.Segment === 'string'
              ? record.Segment : typeof record.SEGMENT_FILE === 'string' ? record.SEGMENT_FILE : pagePreviewSource,
        };
      }),
    [editedStructures, extractImageSource, getMolblockValue, getStructurePreviewKey, pageImages, sourceCropCache, structurePreviewCache],
  );
  const assayColumnNames = React.useMemo(() => {
    const columns = new Set<string>();
    editedStructures.forEach((record) => Object.keys(measurementValues(record)).forEach((column) => columns.add(column)));
    assayRecords.forEach((record) => {
      Object.keys(record).forEach((key) => {
        if (
          key !== 'COMPOUND_ID' &&
          !isAssayMetadataColumn(key) &&
          record[key] !== undefined &&
          record[key] !== null
        ) {
          columns.add(key);
        }
      });
    });
    const removed = new Set<string>();
    editedStructures.forEach((record) => { try { JSON.parse(String(record.USER_REMOVED_MEASUREMENTS || '[]')).forEach((column: string) => removed.add(column)); } catch { /* older rows */ } });
    const coverage = new Map<string, number>();
    const priority = new Map<string, number>();
    editedStructures.forEach((record) => {
      Object.entries(measurementValues(record)).forEach(([column, value]) => { if (value) coverage.set(column, (coverage.get(column) || 0) + 1); });
      try { for (const item of JSON.parse(String(record.BIOACTIVITY_OBSERVATIONS || '[]'))) { const rank = item.source_kind === 'experimental' ? 0 : item.source_kind === 'predicted' ? 2 : item.source_kind === 'calculated' ? 3 : 1; priority.set(item.column, Math.min(priority.get(item.column) ?? 4, rank)); } } catch {}
    });
    return Array.from(columns).filter((column) => !removed.has(column)).sort((a, b) => (priority.get(a) ?? 1) - (priority.get(b) ?? 1) || (coverage.get(b) || 0) - (coverage.get(a) || 0) || a.localeCompare(b));
  }, [assayRecords, editedStructures]);
  const visibleMeasurementColumns = assayColumnNames.filter((column) => !hiddenMeasurements.has(column)).sort((a, b) => { const ai = gridLayout.columns.indexOf(`assay:${a}`), bi = gridLayout.columns.indexOf(`assay:${b}`); return (ai < 0 ? gridLayout.columns.length + assayColumnNames.indexOf(a) : ai) - (bi < 0 ? gridLayout.columns.length + assayColumnNames.indexOf(b) : bi); });
  const assayDataMap = React.useMemo(() => {
    const map = new Map<string, Record<string, unknown>>();
    assayRecords.forEach((record) => {
      const compoundId = (record.COMPOUND_ID ?? '').toString();
      if (!map.has(compoundId)) {
        map.set(compoundId, {});
      }
      const entry = map.get(compoundId)!;
      assayColumnNames.forEach((column) => {
        if (record[column] !== undefined) {
          entry[column] = record[column];
        }
      });
    });
    return map;
  }, [assayRecords, assayColumnNames]);
      const processedStructureRows = React.useMemo(() => {
    const graphsById = new Map<string, Set<string>>();
    structureRows.forEach(({ id, record }) => {
      const graph=String(record.SMILES || '').trim();
      if (!id || !graph || graph.includes('*')) return;
      const graphs=graphsById.get(id) ?? new Set<string>();graphs.add(graph);graphsById.set(id,graphs);
    });
    const ambiguousIds=new Set([...graphsById].filter(([,graphs]) => graphs.size>1).map(([id]) => id));
    const legacyLinkedIds=new Set<string>();
    const rows = structureRows.map((row) => {
      const status=String(row.record.MEASUREMENT_LINK_STATUS || '');
      const mayLink=!ambiguousIds.has(row.id) && !['identifier_unverified','ambiguous_identity'].includes(status) && !['needs_review','failed'].includes(String(row.record.LOCAL_EVIDENCE_STATUS || '')) && !['markush','fragment','text_substituent'].includes(String(row.record.STRUCTURE_TYPE || ''));
      if(mayLink) legacyLinkedIds.add(row.id);
      return {...row,assayData:{...(mayLink ? assayDataMap.get(row.id) ?? {} : {}),...measurementValues(row.record)} as Record<string,unknown>};
    });
    const usedIds = new Set(rows.map((row) => row.id));
    assayDataMap.forEach((assayData, id) => {
      if (id && (!usedIds.has(id) || !legacyLinkedIds.has(id) && !rows.some((row) => row.id === id && String(row.record.RESULT_ONLY).toLowerCase() === 'true'))) {
        rows.push({
          id,
          record: { COMPOUND_ID: id, RESULT_ONLY: 'true', MEASUREMENT_LINK_STATUS: 'ambiguous_identity' } as StructureRecord,
          index: rows.length,
          structureImage: null,
          pagePreviewImage: null,
          pagePreviewSource: '',
          fragmentPagePreviewImage: null,
          fragmentPagePreviewSource: '',
          primaryPageNumber: null,
          sourceCropKey: '',
          boxCoordsPath: '',
          segmentImage: null,
          isMarkushAssembled: false,
          scaffoldSourceCropKey: '',
          scaffoldBoxCoordsPath: '',
          scaffoldPageNumber: null,
          scaffoldSegmentImage: null,
          displayedScaffoldImage: null,
          structurePreviewKey: '',
          structurePreview: null,
          structureSource: '',
          segmentSource: '',
          assayData,
        });
      }
    });
    const pageDetails = rows.map((row) => {
      const record = row.record;
      const primaryPageValue =
        record.PAGE_NUM ?? (record as Record<string, unknown>).page_num ?? (record as Record<string, unknown>).page;
      const normalizedPrimaryPage = formatCellValue(primaryPageValue).trim();
      const rawSourcePages = formatCellValue((record as Record<string, unknown>).source_pages).trim();
      const hasSourcePages = Boolean(rawSourcePages);
      const pageHeading = normalizedPrimaryPage
        ? `Page ${normalizedPrimaryPage}`
        : hasSourcePages
        ? `Pages ${rawSourcePages}`
        : 'Page —';
      const secondaryPageInfo = '';
      const artifactLabel = normalizedPrimaryPage
        ? `PDF page ${normalizedPrimaryPage}`
        : `PDF page ${formatCellValue(record.COMPOUND_ID ?? '')}`;
      return {
        ...row,
        normalizedPrimaryPage,
        rawSourcePages,
        pageHeading,
        secondaryPageInfo,
        artifactLabel,
      };
    });
    // Don't merge page cells - show each page number individually
    return pageDetails.map((row) => {
      return {
        ...row,
        rowSpan: 1,
        showPageCell: true,
      };
    });
  }, [structureRows, assayDataMap]);

  const gridColumnKeys = ['page', 'preview', 'id', 'source', 'structure', 'smiles', ...structureColumnsToRender.map((column) => `meta:${column}`), ...assayColumnNames.map((column) => `assay:${column}`), 'actions'];
  const displayedGridColumns = gridColumnKeys.filter((key) => key === 'preview' ? showPageThumbnails : key === 'smiles' ? showReviewSmiles : key.startsWith('meta:') ? activeMetadataColumns.has(key.slice(5)) : key.startsWith('assay:') ? !hiddenMeasurements.has(key.slice(6)) : true);
  const arrangedGridColumns = [...gridLayout.columns.filter((key) => displayedGridColumns.includes(key)), ...displayedGridColumns.filter((key) => !gridLayout.columns.includes(key))];
  const gridColumnLabel = (key: string) => ({ page: 'Page', preview: 'PDF preview', id: 'Compound ID', source: 'Source structure', structure: 'Extracted structure', smiles: 'SMILES', actions: 'Actions' }[key] || (key.startsWith('meta:') ? STRUCTURE_COLUMN_LABELS[key.slice(5)] || key.slice(5) : key.slice(6)));
  const moveGridColumn = (from: string, to: string, after = false) => {
    const sequence = [...gridLayout.columns.filter((key) => gridColumnKeys.includes(key)), ...gridColumnKeys.filter((key) => !gridLayout.columns.includes(key))];
    gridLayout.update({ columns: moveBefore(sequence, from, to, after) });
  };
  const moveGridRow = (from: string, to: string, after = false) => {
    if (reviewSortColumn) { setToast('Switch to manual order before moving rows.'); return; }
    const all = processedStructureRows.map(reviewRowKey);
    const sequence = [...gridLayout.rows.filter((key) => all.includes(key)), ...all.filter((key) => !gridLayout.rows.includes(key))];
    gridLayout.update({ rows: moveBefore(sequence, from, to, after) });
  };

  const saveStatusMeta = React.useMemo(() => {
    switch (saveStatus) {
      case 'pending':
        return { label: 'Unsaved', modifier: 'table-actions__status--pending' };
      case 'saving':
        return { label: 'Saving…', modifier: 'table-actions__status--saving' };
      case 'saved':
        return { label: 'Saved', modifier: 'table-actions__status--saved' };
      case 'error':
        return { label: 'Not saved', modifier: 'table-actions__status--error' };
      default:
        return null;
    }
  }, [saveStatus]);
  const tableStyle = React.useMemo(() => ({
    '--review-row-height': `${rowHeight}px`,
  }), [rowHeight]);
  const reviewColumnFilterOptions = React.useMemo(() => {
    const options: Array<{ value: string; label: string }> = [
      { value: 'page', label: 'Page' },
      { value: 'COMPOUND_ID', label: 'Compound ID' },
      { value: 'SMILES', label: 'SMILES' },
      ...structureColumnsToRender.map((column) => ({
        value: `structure:${column}`,
        label: STRUCTURE_COLUMN_LABELS[column] ?? column,
      })),
      ...assayColumnNames.map((column) => ({
        value: `assay:${column}`,
        label: column,
      })),
    ];
    const seen = new Set<string>();
    return options.filter((option) => {
      if (seen.has(option.value)) return false;
      seen.add(option.value);
      return true;
    });
  }, [assayColumnNames, structureColumnsToRender]);
  const getReviewColumnValue = React.useCallback((row: (typeof processedStructureRows)[number], columnKey: string): unknown => {
    const record = row.record as Record<string, unknown>;
    if (columnKey === 'page') {
      return row.primaryPageNumber ?? row.rawSourcePages;
    }
    if (columnKey.startsWith('assay:')) {
      return row.assayData[columnKey.slice('assay:'.length)];
    }
    if (columnKey.startsWith('structure:')) {
      return record[columnKey.slice('structure:'.length)];
    }
    return record[columnKey];
  }, []);
  const normalizedReviewSearch = reviewSearch.trim().toLowerCase();
  const hasReviewFilters = Boolean(normalizedReviewSearch || reviewRequiredColumn);
  const filteredReviewRows = React.useMemo(() => {
    return processedStructureRows.filter((row) => {
      const record = row.record as Record<string, unknown>;
      if (reviewRequiredColumn) {
        const cell = getReviewColumnValue(row, reviewRequiredColumn);
        const text = formatCellValue(cell).toLowerCase();
        const query = reviewFilterValue.trim().toLowerCase();
        const number = comparableMeasurement(text); const limit = comparableMeasurement(query);
        const matches = reviewFilterOperator === 'empty' ? !hasReviewCellValue(cell)
          : reviewFilterOperator === 'contains' ? text.includes(query)
          : reviewFilterOperator === 'equals' ? text === query
          : reviewFilterOperator === 'gt' ? text !== '' && query !== '' && number !== null && limit !== null && number > limit
          : reviewFilterOperator === 'lt' ? text !== '' && query !== '' && number !== null && limit !== null && number < limit
          : hasReviewCellValue(cell);
        if (!matches) return false;
      }
      if (!normalizedReviewSearch) {
        return true;
      }
      const values: unknown[] = [
        row.id,
        row.pageHeading,
        row.rawSourcePages,
        record.COMPOUND_ID,
        record.SMILES,
        ...activeStructureColumns.map((column) => record[column]),
        ...assayColumnNames.map((column) => row.assayData[column]),
      ];
      return values.some((value) => formatCellValue(value).toLowerCase().includes(normalizedReviewSearch));
    }).sort((a, b) => {
      if (!reviewSortColumn) { const keys = gridLayout.rows; const ai = keys.indexOf(reviewRowKey(a)), bi = keys.indexOf(reviewRowKey(b)); return (ai < 0 ? keys.length + a.index : ai) - (bi < 0 ? keys.length + b.index : bi); }
      const av = formatCellValue(getReviewColumnValue(a, reviewSortColumn));
      const bv = formatCellValue(getReviewColumnValue(b, reviewSortColumn));
      if (!av || !bv) return !av === !bv ? 0 : !av ? 1 : -1;
      const an = comparableMeasurement(av), bn = comparableMeasurement(bv);
      const result = an !== null && bn !== null ? an - bn : av.localeCompare(bv, undefined, { numeric: true, sensitivity: 'base' });
      return reviewSortDescending ? -result : result;
    });
  }, [
    assayColumnNames,
    getReviewColumnValue,
    normalizedReviewSearch,
    processedStructureRows,
    reviewRequiredColumn, reviewFilterOperator, reviewFilterValue, reviewSortColumn, reviewSortDescending, gridLayout.rows,
    activeStructureColumns,
  ]);
  const reviewTotalSourceRows = processedStructureRows.length;
  const reviewTotalRows = filteredReviewRows.length;
  const reviewTotalPages = Math.max(1, Math.ceil(reviewTotalRows / reviewPageSize));
  const boundedReviewPage = Math.min(Math.max(1, reviewPage), reviewTotalPages);
  const reviewStartIndex = (boundedReviewPage - 1) * reviewPageSize;
  const reviewEndIndex = Math.min(reviewStartIndex + reviewPageSize, reviewTotalRows);
  const paginatedStructureRows = React.useMemo(
    () => filteredReviewRows.slice(reviewStartIndex, reviewEndIndex),
    [filteredReviewRows, reviewEndIndex, reviewStartIndex],
  );

  const paginatedRowIndexKey = paginatedStructureRows.map((row) => row.index).join(',');

  React.useEffect(() => {
    if (!reviewRequiredColumn) return;
    if (!reviewColumnFilterOptions.some((option) => option.value === reviewRequiredColumn)) {
      setReviewRequiredColumn('');
    }
  }, [reviewColumnFilterOptions, reviewRequiredColumn]);

  React.useEffect(() => {
    if (reviewPage !== boundedReviewPage) {
      setReviewPage(boundedReviewPage);
    }
  }, [boundedReviewPage, reviewPage]);

  React.useEffect(() => {
    setReviewPage(1);
    setActiveReviewRowIndex(null);
  }, [reviewPageSize, reviewRequiredColumn, reviewFilterOperator, reviewFilterValue, reviewSortColumn, reviewSortDescending, reviewSearch, selectedJobId]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    tableWrapperRef.current?.scrollTo({ top: 0, left: 0 });
  }, [boundedReviewPage, currentStep, reviewPageSize]);

  React.useEffect(() => {
    if (processedStructureRows.length === 0) {
      setVisibleRowIndices(new Set());
      return;
    }
    setVisibleRowIndices((prev) => {
      const next = new Set(prev);
      let changed = false;
      const initialCount = Math.min(12, paginatedStructureRows.length);
      for (let idx = 0; idx < initialCount; idx += 1) {
        const rowIndex = paginatedStructureRows[idx]?.index;
        if (typeof rowIndex === 'number' && !next.has(rowIndex)) {
          next.add(rowIndex);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [paginatedRowIndexKey, processedStructureRows.length, loadedStructureTaskId]);

  React.useEffect(() => {
    const body = tableBodyRef.current;
    const cards = reviewCardsRef.current;
    if (!body && !cards) return undefined;

    const intersecting = new Map<Element, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const index = Number(entry.target.getAttribute('data-row-index'));
          if (!Number.isFinite(index)) return;
          if (entry.isIntersecting) intersecting.set(entry.target, index);
          else intersecting.delete(entry.target);
        });
        const next = new Set(intersecting.values());
        setVisibleRowIndices((previous) => previous.size === next.size && [...next].every((index) => previous.has(index)) ? previous : next);
      },
      { root: null, rootMargin: '200px 0px' },
    );

    const rows = [
      ...Array.from(body?.querySelectorAll('[data-row-index]') ?? []),
      ...Array.from(cards?.querySelectorAll('[data-row-index]') ?? []),
    ];
    rows.forEach((row) => observer.observe(row));

    return () => observer.disconnect();
  }, [paginatedRowIndexKey, reviewView, loadedStructureTaskId]);

  // ---- review workspace preferences (persisted per browser) ----
  React.useEffect(() => {
    storeJson(REVIEW_VIEW_STORAGE_KEY, reviewView);
  }, [reviewView]);
  React.useEffect(() => {
    storeJson(REVIEW_COLUMNS_STORAGE_KEY, Array.from(activeMetadataColumns));
  }, [activeMetadataColumns]);

  React.useEffect(() => { storeJson('bci:review:density', reviewDensity); }, [reviewDensity]);
  React.useEffect(() => { storeJson('bci:review:row-height', rowHeight); }, [rowHeight]);

  // toasts clear themselves so the workspace never keeps stale banners around
  React.useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 6000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  // Jobs is a non-modal tray: close from any stage without stealing focus
  // from an editor or an image dialog layered above it.
  React.useEffect(() => {
    if (!jobsOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const panel = document.querySelector<HTMLElement>('.jobs-panel');
    panel?.querySelector<HTMLInputElement>('input[type="search"]')?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !(event.target as HTMLElement)?.closest('[aria-modal="true"]')) {
        event.preventDefault(); setJobsOpen(false);
      }
    };
    const onPointer = (event: PointerEvent) => {
      if (!(event.target as HTMLElement)?.closest('.jobs-panel,.jobs-button,[aria-modal="true"]')) setJobsOpen(false);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onPointer);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onPointer);
      if (panel?.contains(document.activeElement) || document.activeElement === document.body) previous?.focus();
    };
  }, [jobsOpen]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    const closeOutside = (event: PointerEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.columns-picker')) setColumnsPickerOpen(false);
      if (!target.closest('#review-advanced-filters,.review-filters-toggle')) setReviewFiltersOpen(false);
      document.querySelectorAll<HTMLDetailsElement>('.review-toolbar details[open]').forEach((menu) => { if (!menu.contains(target)) menu.open = false; });
    };
    const closeMenus = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      const columns = document.querySelector<HTMLButtonElement>('.columns-picker__toggle[aria-expanded="true"]');
      if (columns) { setColumnsPickerOpen(false); columns.focus(); event.stopPropagation(); return; }
      const menu = document.querySelector<HTMLDetailsElement>('.review-toolbar details.review-export-menu[open]');
      if (menu) { menu.open = false; menu.querySelector<HTMLElement>('summary')?.focus(); event.stopPropagation(); }
    };
    document.addEventListener('pointerdown', closeOutside); document.addEventListener('keydown', closeMenus);
    return () => { document.removeEventListener('pointerdown', closeOutside); document.removeEventListener('keydown', closeMenus); };
  }, [currentStep]);

  // Arrow keys select a record and update its source; editing controls keep
  // their native key behavior. Escape follows the visible UI layer.
  React.useEffect(() => {
    if (currentStep !== 4) return;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest('[aria-modal="true"]')) return;
      if (event.key === 'Escape') {
        if (reviewSearchOpen) { setReviewSearchOpen(false); return; }
        if (reviewFiltersOpen) { setReviewFiltersOpen(false); return; }
        if (jobsOpen) { setJobsOpen(false); return; }
        if (reviewFocusMode) { setReviewFocusMode(false); return; }
      }
      if (target?.closest('input,textarea,select,button,a,.pdf-pane,[contenteditable="true"]')) return;
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      if (!filteredReviewRows.length) return;
      event.preventDefault();
      const last = filteredReviewRows.length - 1;
      const position = filteredReviewRows.findIndex((row) => row.index === activeReviewRowIndex);
      const next = position < 0 ? (event.key === 'ArrowDown' ? 0 : last)
        : Math.min(last, Math.max(0, position + (event.key === 'ArrowDown' ? 1 : -1)));
      navigateReviewRowRef.current(filteredReviewRows[next]);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentStep, filteredReviewRows, activeReviewRowIndex, reviewSearchOpen, reviewFiltersOpen, jobsOpen, reviewFocusMode]);

  // keep the active row visible: follow pagination and scroll it into view
  React.useEffect(() => {
    if (activeReviewRowIndex === null) return;
    const position = filteredReviewRows.findIndex((row) => row.index === activeReviewRowIndex);
    const row = filteredReviewRows[position];
    if (!row) {
      setActiveReviewRowIndex(null);
      return;
    }
    const targetPage = Math.floor(position / reviewPageSize) + 1;
    if (targetPage !== boundedReviewPage) {
      setReviewPage(targetPage);
      return;
    }
    const container = reviewView === 'cards' ? reviewCardsRef.current : tableWrapperRef.current;
    const rowEl = container?.querySelector<HTMLElement>(`[data-row-index="${row.index}"]`);
    if (container && rowEl && rowEl.getClientRects().length) {
      const rowBox = rowEl.getBoundingClientRect(), paneBox = container.getBoundingClientRect();
      if (rowBox.top < paneBox.top + 36) container.scrollTop += rowBox.top - paneBox.top - 36;
      else if (rowBox.bottom > paneBox.bottom) container.scrollTop += rowBox.bottom - paneBox.bottom;
    }
  }, [activeReviewRowIndex, filteredReviewRows, reviewPageSize, boundedReviewPage, reviewView]);

  React.useEffect(() => {
    if (modalArtifact && modalBox && originalImageSize && modalImageRef.current) {
      const renderedImage = modalImageRef.current;
      const scaleX = renderedImage.clientWidth / originalImageSize.width;
      const scaleY = renderedImage.clientHeight / originalImageSize.height;

      // Box json stores XYXY; unpack accordingly.
      const [x1, y1, x2, y2] = modalBox;

      setBoxStyle({
        position: 'absolute',
        border: '2px dashed red',
        top: `${y1 * scaleY}px`,
        left: `${x1 * scaleX}px`,
        width: `${(x2 - x1) * scaleX}px`,
        height: `${(y2 - y1) * scaleY}px`,
      });
    } else {
      setBoxStyle({});
    }
  }, [modalArtifact, modalBox, originalImageSize]);
  const modalRowIndex = modalArtifact?.rowIndex ?? null;
  const modalRowCanEdit =
    modalRowIndex !== null && modalRowIndex >= 0 && modalRowIndex < editedStructures.length;
  const modalCompoundIdValue = modalRowCanEdit
    ? formatCellValue(editedStructures[modalRowIndex].COMPOUND_ID ?? '')
    : '';
  const showModalQuickEdit = Boolean(
    modalArtifact && modalRowCanEdit && modalArtifact.mime !== 'loading',
  );

  React.useEffect(() => {
    if (modalArtifact) {
      setModalEditorPosition({ x: 32, y: 32 });
    } else {
      modalEditorDragRef.current = null;
      setModalEditorPosition({ x: 32, y: 32 });
    }
  }, [modalArtifact]);

  const handleModalEditorPointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = modalEditorPanelRef.current;
    if (!panel) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = panel.getBoundingClientRect();
    modalEditorDragRef.current = { offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
    panel.setPointerCapture(event.pointerId);
  }, []);

  const handleModalEditorPointerMove = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!modalEditorDragRef.current) return;
    event.preventDefault();
    const panel = modalEditorPanelRef.current;
    const { offsetX, offsetY } = modalEditorDragRef.current;
    const width = panel?.offsetWidth ?? 0;
    const height = panel?.offsetHeight ?? 0;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const rawX = event.clientX - offsetX;
    const rawY = event.clientY - offsetY;
    const maxX = Math.max(12, viewportWidth - width - 12);
    const maxY = Math.max(12, viewportHeight - height - 12);
    const clampedX = Math.min(Math.max(12, rawX), maxX);
    const clampedY = Math.min(Math.max(12, rawY), maxY);
    setModalEditorPosition((prev) => (prev.x === clampedX && prev.y === clampedY ? prev : { x: clampedX, y: clampedY }));
  }, []);

  const handleModalEditorPointerUp = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = modalEditorPanelRef.current;
    if (panel && panel.hasPointerCapture(event.pointerId)) {
      panel.releasePointerCapture(event.pointerId);
    }
    modalEditorDragRef.current = null;
  }, []);

  // Image preview drag handlers

  // Modal image drag handlers
  const handleModalImagePointerDown = React.useCallback((event: React.PointerEvent<HTMLElement>, element: HTMLElement) => {
    event.preventDefault();
    event.stopPropagation();

    const rect = element.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = rect.left;
    const startTop = rect.top;

    setModalDragState({
      isDragging: true,
      startX,
      startY,
      startLeft,
      startTop,
      element
    });

    if (element.setPointerCapture) {
      element.setPointerCapture(event.pointerId);
    }
  }, []);

  const handleModalImagePointerMove = React.useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (!modalDragState || !modalDragState.isDragging || !modalDragState.element) return;

    event.preventDefault();

    const { startX, startY, startLeft, startTop, element } = modalDragState;
    const deltaX = event.clientX - startX;
    const deltaY = event.clientY - startY;

    const newLeft = startLeft + deltaX;
    const newTop = startTop + deltaY;

    element.style.position = 'absolute';
    element.style.left = `${newLeft}px`;
    element.style.top = `${newTop}px`;
  }, [modalDragState]);

  const handleModalImagePointerUp = React.useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (!modalDragState || !modalDragState.element) return;

    const element = modalDragState.element;
    if (element.hasPointerCapture && element.hasPointerCapture(event.pointerId)) {
      element.releasePointerCapture(event.pointerId);
    }

    setModalDragState(null);
  }, [modalDragState]);

  React.useEffect(() => {
    if (currentStep !== 4) return () => undefined;
    const cacheSnapshot = structurePreviewCacheRef.current;
    if (loadingStructurePreviewsRef.current.size > 0) return () => undefined;
    const pending = new Map<string, { smiles: string; molblock: string; previewKey: string }>();
    editedStructures.forEach((record, index) => {
      if (!visibleRowIndices.has(index)) return;
      const smilesValue = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
      const molblockValue = getMolblockValue(record);
      const previewKey = getStructurePreviewKey(smilesValue, molblockValue);
      if (!previewKey) return;
      if (Object.prototype.hasOwnProperty.call(cacheSnapshot, previewKey)) return;
      if (loadingStructurePreviewsRef.current.has(previewKey)) return;
      if (!pending.has(previewKey)) {
        pending.set(previewKey, {
          smiles: smilesValue,
          molblock: molblockValue,
          previewKey,
        });
      }
    });
    if (!pending.size) return () => undefined;

    const batch = Array.from(pending.values()).slice(0, STRUCTURE_PREVIEW_BATCH_SIZE);
    const batchKeys = batch.map((payload) => payload.previewKey);
    const nextLoading = new Set(loadingStructurePreviewsRef.current);
    batchKeys.forEach((key) => nextLoading.add(key));
    loadingStructurePreviewsRef.current = nextLoading;
    setLoadingStructurePreviews(nextLoading);

    renderSmilesBatch(
      batch.map((payload) => ({
        key: payload.previewKey,
        smiles: payload.smiles,
        width: 220,
        height: 170,
        molblock: payload.molblock,
      })),
    )
      .then((results) => {
        const resultMap = new Map(results.map((result) => [result.key, result]));
        setStructurePreviewCache((prev) => {
          const next = { ...prev };
          batch.forEach((payload) => {
            if (!Object.prototype.hasOwnProperty.call(next, payload.previewKey)) {
              next[payload.previewKey] = resultMap.get(payload.previewKey)?.image || '';
            }
          });
          structurePreviewCacheRef.current = next;
          return next;
        });
      })
      .catch((error) => {
        console.warn('Failed to generate structure previews', error);
        setStructurePreviewCache((prev) => {
          const next = { ...prev };
          batch.forEach((payload) => {
            if (!Object.prototype.hasOwnProperty.call(next, payload.previewKey)) {
              next[payload.previewKey] = '';
            }
          });
          structurePreviewCacheRef.current = next;
          return next;
        });
      })
      .finally(() => {
        const remaining = new Set(loadingStructurePreviewsRef.current);
        batchKeys.forEach((key) => remaining.delete(key));
        loadingStructurePreviewsRef.current = remaining;
        setLoadingStructurePreviews(remaining);
      });

    return () => undefined;
  }, [currentStep, editedStructures, extractImageSource, getMolblockValue, getStructurePreviewKey, structurePreviewCache, visibleRowIndices]);
  const canViewResults = React.useMemo(
    () =>
      Boolean(
        structures.length > 0 ||
          assayRecords.length > 0 ||
          fullPipelineTask !== null ||
          autoDetectTask !== null ||
          structureTask !== null ||
          structureAddonTask !== null ||
          assayAddonTask !== null ||
          assayTask !== null,
      ),
    [
      structures.length,
      assayRecords.length,
      fullPipelineTask,
      autoDetectTask,
      structureTask,
      structureAddonTask,
      assayAddonTask,
      assayTask,
    ],
  );
  const maxStep = React.useMemo<StepId>(() => {
    if (canViewResults) return 4;
    if (pdfInfo) return 3;
    return 1;
  }, [pdfInfo, canViewResults]);
  const handleStepNavigation = React.useCallback(
    (stepId: StepId) => {
      if (stepId <= maxStep) {
        suppressAutoStepUntilNextJobRef.current = true;
        setCurrentStep(stepId);
      }
    },
    [maxStep],
  );

  const initializedFromUrl = React.useRef(false);
  const lastUrlRef = React.useRef<string>('');
  const requestedStepRef = React.useRef<StepId | null>(null);

  const resetNotifications = () => {
    setError(null);
    setToast(null);
  };

  const jobsQueryParams = React.useMemo(
    () => ({
      page: jobsPage,
      page_size: jobsPageSize,
      search: jobsSearch,
      status: jobsStatusFilter,
      task_type: jobsTypeFilter,
      date_from: jobsDateFrom,
      date_to: jobsDateTo,
      sort_by: jobsSortBy,
      sort_dir: jobsSortDir,
    }),
    [jobsDateFrom, jobsDateTo, jobsPage, jobsPageSize, jobsSearch, jobsSortBy, jobsSortDir, jobsStatusFilter, jobsTypeFilter],
  );

  React.useEffect(() => {
    jobsQuerySignatureRef.current = JSON.stringify(jobsQueryParams);
  }, [jobsQueryParams]);

  const loadJobs = React.useCallback(async (options?: { force?: boolean; silent?: boolean }) => {
    if (jobsInFlightRef.current && !options?.force) {
      return;
    }
    const query = jobsQueryParams;
    const signature = JSON.stringify(query);
    if (!jobsQuerySignatureRef.current) {
      jobsQuerySignatureRef.current = signature;
    }
    const requestSeq = jobsRequestSeqRef.current + 1;
    jobsRequestSeqRef.current = requestSeq;
    jobsInFlightRef.current = true;
    try {
      if (!options?.silent) {
        setJobsLoading(true);
      }
      const nextJobs = await fetchTasks(query);
      if (requestSeq !== jobsRequestSeqRef.current || signature !== jobsQuerySignatureRef.current) {
        return;
      }
      setJobsError(null);
      setJobsInfo((current) => {
        if (current?.revision === nextJobs.revision && jobsInfoSignatureRef.current === signature) {
          return current;
        }
        jobsInfoSignatureRef.current = signature;
        return nextJobs;
      });
    } catch (err) {
      console.warn('Failed to load jobs', err);
      if (requestSeq === jobsRequestSeqRef.current && !options?.silent) {
        setJobsError(err instanceof Error ? err.message : 'Failed to load jobs');
      }
    } finally {
      if (requestSeq === jobsRequestSeqRef.current) {
        jobsInFlightRef.current = false;
        if (!options?.silent) {
          setJobsLoading(false);
        }
      }
    }
  }, [jobsQueryParams]);

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setJobsPage(1);
      setJobsSearch(jobsSearchInput.trim());
    }, 250);
    return () => window.clearTimeout(timer);
  }, [jobsSearchInput]);

  const handleJobsSort = React.useCallback((field: JobSortBy) => {
    setJobsPage(1);
    setJobsSortBy((current) => {
      if (current === field) {
        setJobsSortDir((direction) => (direction === 'asc' ? 'desc' : 'asc'));
        return current;
      }
      setJobsSortDir(field === 'queue_position' ? 'asc' : 'desc');
      return field;
    });
  }, []);

  const renderJobsSortLabel = React.useCallback(
    (field: JobSortBy, label: string) => (
      <button className="jobs-sort-button" type="button" onClick={() => handleJobsSort(field)}>
        <span>{label}</span>
        <span className="jobs-sort-button__mark">{jobsSortBy === field ? (jobsSortDir === 'asc' ? '↑' : '↓') : ''}</span>
      </button>
    ),
    [handleJobsSort, jobsSortBy, jobsSortDir],
  );

  const updateTrackedTaskAfterCancel = React.useCallback((updated: TaskStatus) => {
    const clearIfSame = (current: TaskStatus | null) => (current?.task_id === updated.task_id ? null : current);
    setFullPipelineTask(clearIfSame);
    setAutoDetectTask(clearIfSame);
    setStructureAddonTask(clearIfSame);
    setAssayAddonTask(clearIfSame);
    setStructureTask(clearIfSame);
    setAssayTask(clearIfSame);
  }, []);

  const handleCancelJob = React.useCallback(
    async (job: TaskStatus) => {
      if (!isCancelableTask(job)) return;
      try {
        const updated = await cancelTask(job.task_id);
        updateTrackedTaskAfterCancel(updated);
        setToast('Job canceled.');
        void loadJobs({ force: true });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not cancel job');
      }
    },
    [loadJobs, updateTrackedTaskAfterCancel],
  );

  const coerceStructureFilterStrictness = React.useCallback((value: unknown): StructureFilterStrictness => {
    if (value === 'balanced' || value === 'permissive' || value === 'strict') {
      return value;
    }
    return 'strict';
  }, []);

  const coerceAutoDetectionFlag = React.useCallback((value: unknown) => {
    return typeof value === 'boolean' ? value : true;
  }, []);

  const applyDetectedStructurePages = React.useCallback((value: unknown) => {
    const detectedPages = parsePageListParam(value);
    if (!detectedPages.length) return;
    const signature = pagesToString(detectedPages);
    if (appliedStructureDetectedPagesRef.current === signature) return;
    appliedStructureDetectedPagesRef.current = signature;
    setStructureSelection(new Set<number>(detectedPages));
    setStructurePagesInput(signature);
    lastStructurePageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, []);

  const applyPlannedStructurePages = React.useCallback((value: unknown) => {
    const detectedPages = parsePageListParam(value);
    if (!detectedPages.length) return;
    const signature = pagesToString(detectedPages);
    if (appliedStructureDetectedPagesRef.current === signature) return;
    appliedStructureDetectedPagesRef.current = signature;
    setStructureSelection(new Set<number>(detectedPages));
    setStructurePagesInput(signature);
    lastStructurePageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, []);

  const applyDetectedAssayPages = React.useCallback((value: unknown) => {
    const detectedPages = parsePageListParam(value);
    if (!detectedPages.length) return;
    const signature = pagesToString(detectedPages);
    if (appliedAssayDetectedPagesRef.current === signature) return;
    appliedAssayDetectedPagesRef.current = signature;
    setAssaySelection(new Set<number>(detectedPages));
    setAssayPagesInput(signature);
    lastAssayPageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, []);

  const applyPlannedAssayPages = React.useCallback((value: unknown) => {
    const detectedPages = parsePageListParam(value);
    if (!detectedPages.length) return;
    const signature = pagesToString(detectedPages);
    if (appliedAssayDetectedPagesRef.current === signature) return;
    appliedAssayDetectedPagesRef.current = signature;
    setAssaySelection(new Set<number>(detectedPages));
    setAssayPagesInput(signature);
    lastAssayPageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, []);

  const restoredEvidencePagesRef = React.useRef('');
  React.useEffect(() => {
    if (!loadedStructureTaskId || !editedStructures.length || restoredEvidencePagesRef.current === loadedStructureTaskId) return;
    restoredEvidencePagesRef.current = loadedStructureTaskId;
    const structurePages = new Set<number>();
    const resultPages = new Set<number>();
    for (const record of editedStructures) {
      if (record.SMILES || record.SEGMENT_FILE || record.MARKUSH_SCAFFOLD_IMAGE_FILE) {
        const page = getRecordPrimaryPageNumber(record); if (page) structurePages.add(page);
        const scaffoldPage = Number(record.MARKUSH_SCAFFOLD_PAGE); if (scaffoldPage > 0) structurePages.add(scaffoldPage);
      }
      try { for (const observation of JSON.parse(String(record.BIOACTIVITY_OBSERVATIONS || '[]'))) { const page = Number(observation.page); if (page > 0) resultPages.add(page); } } catch { /* original evidence is preserved */ }
    }
    if (!structureSelection.size && structurePages.size) applyDetectedStructurePages([...structurePages].sort((a, b) => a - b));
    if (!assaySelection.size && resultPages.size) applyDetectedAssayPages([...resultPages].sort((a, b) => a - b));
  }, [loadedStructureTaskId, editedStructures, structureSelection.size, assaySelection.size, applyDetectedStructurePages, applyDetectedAssayPages]);

  const applyDetectedAssayNames = React.useCallback((value: unknown) => {
    const detectedNames = parseStringListParam(value);
    if (!detectedNames.length) return;
    const signature = detectedNames.join('\n');
    if (appliedAssayNamesRef.current === signature) return;
    appliedAssayNamesRef.current = signature;
    setAssayNames((prev) => mergeAssayNameLists(prev, detectedNames));
  }, []);

  const handleStructuresReady = React.useCallback(() => {
    if (autoAdvanceRef.current) return;
    autoAdvanceRef.current = true;
  }, []);

  const clearWorkspace = React.useCallback(() => {
    setStructurePagesInput('');
    setAssayPagesInput('');
    setStructureSelection(new Set<number>());
    setAssaySelection(new Set<number>());
    setAutoDetectStructurePages(true);
    setAutoDetectAssayPages(true);
    setAutoDetectAssayNames(true);
    lastStructurePageRef.current = null;
    lastAssayPageRef.current = null;
    restoredEvidencePagesRef.current = '';
    appliedStructureDetectedPagesRef.current = '';
    appliedAssayDetectedPagesRef.current = '';
    appliedAssayNamesRef.current = '';
    setPageImages({});
    pageImagesRef.current = {};
    setLoadingPages(new Set<number>());
    loadingPagesRef.current = new Set();
    pageFetchQueueRef.current = [];
    activePageFetchesRef.current = 0;
    setStructureTask(null);
    setAssayTask(null);
    setStructureAddonTask(null);
    setAssayAddonTask(null);
    setAutoDetectTask(null);
    setFullPipelineTask(null);
    setIsFullPipelineSubmitting(false);
    setStructureSelectionFeedback(null);
    setAssaySelectionFeedback(null);
    setPendingStructureAddonPages([]);
    setPendingAssayAddonPages([]);
    structureAddonSubmittedPagesRef.current = [];
    assayAddonSubmittedPagesRef.current = [];
    loadedStructureAddonStorageKeyRef.current = null;
    loadedAssayAddonStorageKeyRef.current = null;
    skipNextStructureAddonStoreRef.current = false;
    skipNextAssayAddonStoreRef.current = false;
    setStructureFilterStrictness('strict');
    pdfLocateSequenceRef.current += 1;
    setPdfTarget(null);
    setActiveReviewRowIndex(null);
    setStructures([]);
    setEditedStructures([]);
    setFilteredStructures([]);
    setStructurePreviewCache({});
    structurePreviewCacheRef.current = {};
    setLoadingStructurePreviews(new Set());
    loadingStructurePreviewsRef.current = new Set();
    editedStructuresRef.current = [];
    setLoadedStructureTaskId(undefined);
    setSaveStatus('idle');
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    setAssayRecords([]);
    setAssayNames([]);
    setAssayNameDraft('');
    setImageCache({});
    artifactEpochRef.current += 1;
    setArtifactErrors({});
    activeArtifactPathsRef.current = new Set();
    setLoadingArtifacts(new Set());
    loadingArtifactsRef.current = new Set();
    setSourceCropCache({});
    sourceCropCacheRef.current = {};
    setLoadingSourceCrops(new Set());
    loadingSourceCropsRef.current = new Set();
    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);
    autoAdvanceRef.current = false;
    structurePagesAutoAdvancedRef.current = false;
    assayPagesAutoAdvancedRef.current = false;
    pdfInitializedRef.current = false;
    requestedStepRef.current = null;
    suppressAutoStepUntilNextJobRef.current = false;
    setCurrentStep(1);
  }, []);

  const clearJobResultState = React.useCallback((options?: { clearPageCache?: boolean }) => {
    setStructurePagesInput('');
    setAssayPagesInput('');
    setStructureSelection(new Set<number>());
    setAssaySelection(new Set<number>());
    setStructureSelectionFeedback(null);
    setAssaySelectionFeedback(null);
    setPendingStructureAddonPages([]);
    setPendingAssayAddonPages([]);
    structureAddonSubmittedPagesRef.current = [];
    assayAddonSubmittedPagesRef.current = [];
    loadedStructureAddonStorageKeyRef.current = null;
    loadedAssayAddonStorageKeyRef.current = null;
    skipNextStructureAddonStoreRef.current = false;
    skipNextAssayAddonStoreRef.current = false;
    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);

    setStructureTask(null);
    setAssayTask(null);
    setStructureAddonTask(null);
    setAssayAddonTask(null);
    setAutoDetectTask(null);
    setFullPipelineTask(null);
    setIsFullPipelineSubmitting(false);

    pdfLocateSequenceRef.current += 1;
    setPdfTarget(null);
    setActiveReviewRowIndex(null);
    setStructures([]);
    setEditedStructures([]);
    editedStructuresRef.current = [];
    setLoadedStructureTaskId(undefined);
    setFilteredStructures([]);
    setAssayRecords([]);
    setAssayNames([]);
    setAssayNameDraft('');
    setSaveStatus('idle');
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }

    setStructurePreviewCache({});
    structurePreviewCacheRef.current = {};
    setLoadingStructurePreviews(new Set());
    loadingStructurePreviewsRef.current = new Set();
    setImageCache({});
    artifactEpochRef.current += 1;
    setArtifactErrors({});
    activeArtifactPathsRef.current = new Set();
    setLoadingArtifacts(new Set());
    loadingArtifactsRef.current = new Set();
    setSourceCropCache({});
    sourceCropCacheRef.current = {};
    setLoadingSourceCrops(new Set());
    loadingSourceCropsRef.current = new Set();
    setModalArtifact(null);
    setModalBox(null);
    setOriginalImageSize(null);
    setVisibleRowIndices(new Set());
    autoAdvanceRef.current = false;
    suppressAutoStepUntilNextJobRef.current = false;
    restoredEvidencePagesRef.current = '';
    appliedStructureDetectedPagesRef.current = '';
    appliedAssayDetectedPagesRef.current = '';
    appliedAssayNamesRef.current = '';

    if (options?.clearPageCache) {
      setPageImages({});
      pageImagesRef.current = {};
      setLoadingPages(new Set<number>());
      loadingPagesRef.current = new Set();
      pageFetchQueueRef.current = [];
      activePageFetchesRef.current = 0;
    }
  }, []);


  const [patentNumber, setPatentNumber] = React.useState('');

  const handlePatentFetch = async (event: React.FormEvent<HTMLFormElement>) => {
    resetNotifications();
    event.preventDefault();
    const number = patentNumber.trim();
    if (!number) return;
    if (!(await flushPendingEdits())) return;
    try {
      setIsUploading(true);
      setUploadProgress(null);
      const response = await fetchPatentPdf(number);
      setPdfInfo(response);
      clearWorkspace();
      setToast(`Patent ${number} downloaded: ${response.total_pages} pages.`);
      setCurrentStep(1);
      pdfInitializedRef.current = true;
      autoAdvanceRef.current = false;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Patent download failed');
    } finally {
      setIsUploading(false);
    }
  };

  const uploadSourceFile = async (file: File) => {
    if (isUploading) return;
    resetNotifications();
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Choose a PDF document.');
      return;
    }
    if (!(await flushPendingEdits())) return;
    try {
      setIsUploading(true);
      setUploadProgress(0);
      const response = await uploadPdf(file, (percent) => setUploadProgress(percent));
      setPdfInfo(response);
      clearWorkspace();
      setToast(`Upload complete: ${response.total_pages} page${response.total_pages === 1 ? '' : 's'}.`);
      setCurrentStep(1);
      pdfInitializedRef.current = true;
      autoAdvanceRef.current = false;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadProgress(null), 500);
    }
  };

  const handleFileUpload: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) void uploadSourceFile(file);
  };

  const pages = React.useMemo(() => {
    if (!pdfInfo) return [] as number[];
    return Array.from({ length: pdfInfo.total_pages }, (_, idx) => idx + 1);
  }, [pdfInfo]);

  React.useEffect(() => {
    pdfInfoRef.current = pdfInfo;
    pageFetchQueueRef.current = [];
    activePageFetchesRef.current = 0;
  }, [pdfInfo]);

  React.useEffect(() => {
    pageImagesRef.current = pageImages;
  }, [pageImages]);

  React.useEffect(() => {
    loadingPagesRef.current = loadingPages;
  }, [loadingPages]);

  React.useEffect(() => {
    loadingArtifactsRef.current = loadingArtifacts;
  }, [loadingArtifacts]);

  React.useEffect(() => {
    sourceCropCacheRef.current = sourceCropCache;
  }, [sourceCropCache]);

  React.useEffect(() => {
    loadingSourceCropsRef.current = loadingSourceCrops;
  }, [loadingSourceCrops]);

  React.useEffect(() => {
    structurePreviewCacheRef.current = structurePreviewCache;
  }, [structurePreviewCache]);

  React.useEffect(() => {
    loadingStructurePreviewsRef.current = loadingStructurePreviews;
  }, [loadingStructurePreviews]);

  React.useEffect(() => {
    editedStructuresRef.current = editedStructures;
  }, [editedStructures]);

  React.useEffect(
    () => () => {
      if (autoSaveTimerRef.current) {
        window.clearTimeout(autoSaveTimerRef.current);
      }
    },
    [],
  );

  const MAX_PAGE_FETCH_CONCURRENCY = 2;
  const pageFetchQueueRef = React.useRef<number[]>([]);
  const activePageFetchesRef = React.useRef(0);

  const galleryContainerRefs = React.useRef<Record<SelectionMode, HTMLDivElement | null>>({
    structure: null,
    assay: null,
  });
  const setStructureGalleryRef = React.useCallback((node: HTMLDivElement | null) => {
    galleryContainerRefs.current.structure = node;
  }, []);
  const setAssayGalleryRef = React.useCallback((node: HTMLDivElement | null) => {
    galleryContainerRefs.current.assay = node;
  }, []);

  const processQueue = React.useCallback(() => {
    const pdf = pdfInfoRef.current;
    if (!pdf) return;
    while (
      activePageFetchesRef.current < MAX_PAGE_FETCH_CONCURRENCY &&
      pageFetchQueueRef.current.length > 0
    ) {
      const page = pageFetchQueueRef.current.shift();
      if (page === undefined) continue;
      if (pageImagesRef.current[page]) {
        setLoadingPages((prev) => {
          if (!prev.has(page)) return prev;
          const next = new Set(prev);
          next.delete(page);
          loadingPagesRef.current = next;
          return next;
        });
        continue;
      }
      const requestPdfId = pdf.pdf_id;
      activePageFetchesRef.current += 1;
      void fetchPdfPage(requestPdfId, page, { zoom: 1.6, maxWidth: 680 })
        .then((image) => {
          if (pdfInfoRef.current?.pdf_id !== requestPdfId) {
            return;
          }
          setPageImages((prev) => {
            const next = { ...prev, [page]: image };
            pageImagesRef.current = next;
            return next;
          });
        })
        .catch((err) => {
          setError((prevError) =>
            prevError ?? (err instanceof Error ? err.message : `Failed to load preview for page ${page}.`),
          );
        })
        .finally(() => {
          activePageFetchesRef.current -= 1;
          setLoadingPages((prev) => {
            if (!prev.has(page)) return prev;
            const next = new Set(prev);
            next.delete(page);
            loadingPagesRef.current = next;
            return next;
          });
          processQueue();
        });
    }
  }, []);

  const enqueuePage = React.useCallback(
    (page: number, priority = false) => {
      const pdf = pdfInfoRef.current;
      if (!pdf) return;
      if (page <= 0) return;
      if (pageImagesRef.current[page]) return;
      if (loadingPagesRef.current.has(page)) return;
      if (pageFetchQueueRef.current.includes(page)) return;
      setLoadingPages((prev) => {
        if (prev.has(page)) return prev;
        const next = new Set(prev);
        next.add(page);
        loadingPagesRef.current = next;
        return next;
      });
      if (priority) {
        pageFetchQueueRef.current.unshift(page);
      } else {
        pageFetchQueueRef.current.push(page);
      }
      processQueue();
    },
    [processQueue],
  );

  const loadPageImage = React.useCallback(
    (page: number, priority = false) => {
      enqueuePage(page, priority);
    },
    [enqueuePage],
  );

  React.useEffect(() => {
    if (!pdfInfo || !pages.length || currentStep === 4 || requestedStepRef.current === 4) return;
    const initialCount = Math.min(currentStep === 1 ? 1 : 8, pages.length);
    pages.slice(0, initialCount).forEach((page) => {
      loadPageImage(page, true);
    });
  }, [currentStep, pdfInfo, pages, loadPageImage]);

  React.useEffect(() => {
    if (!pages.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const dataPage = entry.target.getAttribute('data-page');
          if (!dataPage) return;
          const pageNumber = Number(dataPage);
          if (!Number.isNaN(pageNumber)) {
            void loadPageImage(pageNumber);
          }
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: '160px' },
    );
    const activeMode = getSelectionModeForStep(currentStep);
    const container = galleryContainerRefs.current[activeMode];
    container?.querySelectorAll('[data-page]').forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [pages, currentStep, loadPageImage]);

  React.useEffect(() => {
    if (!pdfInfo) return;
    if (currentStep === 4) return;
    const focusPages = new Set<number>();
    structureSelection.forEach((page) => focusPages.add(page));
    assaySelection.forEach((page) => focusPages.add(page));
    Array.from(focusPages)
      .slice(0, 16)
      .forEach((page) => {
      if (page > 0) {
        loadPageImage(page, true);
      }
    });
  }, [currentStep, pdfInfo, structureSelection, assaySelection, loadPageImage]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    paginatedStructureRows.forEach((row) => {
      if (!visibleRowIndices.has(row.index)) return;
      const hasCropArtifact = Boolean(row.record.Segment || row.record.SEGMENT_FILE);
      const pageNumber = row.primaryPageNumber;
      if (hasCropArtifact && !row.isMarkushAssembled && !(showPageThumbnails && reviewView === 'table')) return;
      if (typeof pageNumber === 'number' && pageNumber > 0 && !pageImagesRef.current[pageNumber]) {
        loadPageImage(pageNumber, true);
      }
      if (row.isMarkushAssembled && typeof row.scaffoldPageNumber === 'number' && row.scaffoldPageNumber > 0 && !pageImagesRef.current[row.scaffoldPageNumber]) {
        loadPageImage(row.scaffoldPageNumber, true);
      }
    });
  }, [currentStep, loadPageImage, paginatedStructureRows, visibleRowIndices, sourcePaneVisible, showPageThumbnails, reviewView]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    type CropTask = { cropKey: string; boxPath: string; pageNumber: number };
    const tasks: CropTask[] = [];
    paginatedStructureRows.forEach((row) => {
      if (!visibleRowIndices.has(row.index)) return;
      const candidates: CropTask[] = [];
      if (row.sourceCropKey && row.boxCoordsPath && row.primaryPageNumber && pageImages[row.primaryPageNumber]) {
        candidates.push({ cropKey: row.sourceCropKey, boxPath: row.boxCoordsPath, pageNumber: row.primaryPageNumber });
      }
      if (
        row.isMarkushAssembled &&
        row.scaffoldSourceCropKey &&
        row.scaffoldBoxCoordsPath &&
        row.scaffoldPageNumber &&
        pageImages[row.scaffoldPageNumber]
      ) {
        candidates.push({
          cropKey: row.scaffoldSourceCropKey,
          boxPath: row.scaffoldBoxCoordsPath,
          pageNumber: row.scaffoldPageNumber,
        });
      }
      candidates.forEach((task) => {
        if (Object.prototype.hasOwnProperty.call(sourceCropCacheRef.current, task.cropKey)) return;
        if (loadingSourceCropsRef.current.has(task.cropKey)) return;
        if (tasks.length >= 4) return;
        tasks.push(task);
      });
    });
    if (!tasks.length) return;

    tasks.forEach(({ cropKey, boxPath, pageNumber }) => {
      setLoadingSourceCrops((prev) => {
        if (prev.has(cropKey)) return prev;
        const next = new Set(prev);
        next.add(cropKey);
        return next;
      });
      fetchBoxArtifact(boxPath)
        .then(async (artifact) => {
          const coordsData = JSON.parse(atob(artifact.content));
          const sourceSize =
            Number.isFinite(Number(coordsData?.page_width)) && Number.isFinite(Number(coordsData?.page_height))
              ? { width: Number(coordsData.page_width), height: Number(coordsData.page_height) }
              : null;
          if (!sourceSize) {
            throw new Error('Source page size is unavailable for crop');
          }
          const crop = await cropPageImageFromBox(
            `data:image/png;base64,${pageImages[pageNumber]}`,
            coordsData.box,
            sourceSize,
          );
          setSourceCropCache((prev) => ({ ...prev, [cropKey]: crop }));
        })
        .catch(() => {
          setSourceCropCache((prev) => ({ ...prev, [cropKey]: '' }));
        })
        .finally(() => {
          setLoadingSourceCrops((prev) => {
            if (!prev.has(cropKey)) return prev;
            const next = new Set(prev);
            next.delete(cropKey);
            return next;
          });
        });
    });
  }, [currentStep, pageImages, paginatedStructureRows, visibleRowIndices]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    const pending = new Set<string>();
    const visiblePaths = new Set<string>();
    const addFirstArtifact = (record: StructureRecord, keys: string[]) => {
      for (const key of keys) {
        const value = record[key as keyof StructureRecord];
        if (typeof value !== 'string' || !value) continue;
        if (isDataImage(value) || markdownImageRegex.test(value)) return;
        visiblePaths.add(value);
        if (artifactErrors[value]) continue;
        if (imageCache[value] || loadingArtifactsRef.current.has(value)) return;
        pending.add(value);
        return;
      }
    };
    paginatedStructureRows.forEach((row) => {
      if (!visibleRowIndices.has(row.index)) return;
      const record = row.record;
      addFirstArtifact(record, ['Structure', 'PAGE_IMAGE_FILE', 'IMAGE_FILE', 'Segment File']);
      addFirstArtifact(record, ['Segment', 'SEGMENT_FILE']);
      if ((record.STRUCTURE_TYPE as string | undefined)?.trim() === 'markush_assembled') {
        addFirstArtifact(record, ['MARKUSH_SCAFFOLD_IMAGE_FILE']);
      }
    });
    activeArtifactPathsRef.current = visiblePaths;
    const available = Math.max(0, 6 - loadingArtifactsRef.current.size);
    const epoch = artifactEpochRef.current;
    [...pending].slice(0, available).forEach((path) => {
      const loading = new Set(loadingArtifactsRef.current);
      loading.add(path);
      loadingArtifactsRef.current = loading;
      setLoadingArtifacts(loading);
      fetchArtifact(path)
        .then((artifact) => {
          if (artifactEpochRef.current !== epoch) return;
          const dataUri = `data:${artifact.mime_type};base64,${artifact.content}`;
          setImageCache((previous) => withBoundedArtifactCache(previous, path, dataUri, activeArtifactPathsRef.current));
        })
        .catch((error) => {
          if (artifactEpochRef.current !== epoch) return;
          setArtifactErrors((previous) => ({ ...previous, [path]: error instanceof Error ? error.message : 'Image unavailable' }));
        })
        .finally(() => {
          if (artifactEpochRef.current !== epoch) return;
          const loading = new Set(loadingArtifactsRef.current);
          loading.delete(path);
          loadingArtifactsRef.current = loading;
          setLoadingArtifacts(loading);
        });
    });
  }, [currentStep, paginatedStructureRows, imageCache, visibleRowIndices, artifactErrors, loadingArtifacts]);

  React.useEffect(() => {
    if (currentStep > maxStep) {
      setCurrentStep(maxStep);
    }
  }, [currentStep, maxStep]);

  React.useEffect(() => {
    if (!requestedStepRef.current) return;
    const requested = requestedStepRef.current;
    if (requested <= maxStep) {
      setCurrentStep(requested);
      requestedStepRef.current = null;
    }
  }, [maxStep]);

  React.useEffect(() => {
    if (!structureTask) {
      autoAdvanceRef.current = false;
      return;
    }
    if (structureTask.status === 'running' || structureTask.status === 'pending') {
      autoAdvanceRef.current = false;
    }
    if (structureTask.status === 'completed') {
      handleStructuresReady();
    }
  }, [structureTask?.status, handleStructuresReady]);

  React.useEffect(() => {
    if (structures.length > 0) {
      handleStructuresReady();
    }
  }, [structures.length, handleStructuresReady]);

  const structurePagesAutoAdvancedRef = React.useRef(false);
  const assayPagesAutoAdvancedRef = React.useRef(false);
  const assayAutoAdvancedRef = React.useRef(false);
  React.useEffect(() => {
    if (
      assayRecords.length > 0 &&
      currentStep < 4 &&
      !assayAutoAdvancedRef.current &&
      !suppressAutoStepUntilNextJobRef.current
    ) {
      assayAutoAdvancedRef.current = true;
      setCurrentStep(4);
    }
    if (assayRecords.length === 0) {
      assayAutoAdvancedRef.current = false;
    }
  }, [assayRecords.length, currentStep]);

  React.useEffect(() => {
    if (!pdfInfo) {
      pdfInitializedRef.current = false;
      setCurrentStep(1);
    }
  }, [pdfInfo]);

  React.useEffect(() => {
    const key = buildAddonStorageKey('structure', pdfInfo?.pdf_id, structureTask?.task_id);
    if (!key || loadedStructureAddonStorageKeyRef.current === key) return;
    loadedStructureAddonStorageKeyRef.current = key;
    const storedPages = loadStoredPages(key);
    if (storedPages.length) {
      skipNextStructureAddonStoreRef.current = true;
      setPendingStructureAddonPages((prev) =>
        Array.from(new Set([...prev, ...storedPages])).sort((a, b) => a - b),
      );
      setStructureSelectionFeedback(
        `Restored ${storedPages.length} selected structure page${storedPages.length === 1 ? '' : 's'}.`,
      );
    }
  }, [pdfInfo?.pdf_id, structureTask?.task_id]);

  React.useEffect(() => {
    const key = buildAddonStorageKey('assay', pdfInfo?.pdf_id, assayTask?.task_id);
    if (!key || loadedAssayAddonStorageKeyRef.current === key) return;
    loadedAssayAddonStorageKeyRef.current = key;
    const storedPages = loadStoredPages(key);
    if (storedPages.length) {
      skipNextAssayAddonStoreRef.current = true;
      setPendingAssayAddonPages((prev) =>
        Array.from(new Set([...prev, ...storedPages])).sort((a, b) => a - b),
      );
      setAssaySelectionFeedback(
        `Restored ${storedPages.length} selected bioactivity page${storedPages.length === 1 ? '' : 's'}.`,
      );
    }
  }, [assayTask?.task_id, pdfInfo?.pdf_id]);

  React.useEffect(() => {
    const key = buildAddonStorageKey('structure', pdfInfo?.pdf_id, structureTask?.task_id);
    if (skipNextStructureAddonStoreRef.current && pendingStructureAddonPages.length === 0) {
      skipNextStructureAddonStoreRef.current = false;
      return;
    }
    skipNextStructureAddonStoreRef.current = false;
    storePages(key, pendingStructureAddonPages);
  }, [pdfInfo?.pdf_id, pendingStructureAddonPages, structureTask?.task_id]);

  React.useEffect(() => {
    const key = buildAddonStorageKey('assay', pdfInfo?.pdf_id, assayTask?.task_id);
    if (skipNextAssayAddonStoreRef.current && pendingAssayAddonPages.length === 0) {
      skipNextAssayAddonStoreRef.current = false;
      return;
    }
    skipNextAssayAddonStoreRef.current = false;
    storePages(key, pendingAssayAddonPages);
  }, [assayTask?.task_id, pdfInfo?.pdf_id, pendingAssayAddonPages]);

  const reportStructureSelectionEdit = React.useCallback((nextPages: number[]) => {
    const taskPages = getTaskSelectedPages(structureTask);
    if (!structureTask || !taskPages.length) return;
    const delta = summarizePageDelta(nextPages, taskPages);
    if (!delta) {
      if (isTaskInFlight(structureTask)) {
        setPendingStructureAddonPages([]);
      }
      setStructureSelectionFeedback(null);
      return;
    }
    if (isTaskInFlight(structureTask)) {
      const taskSet = new Set(taskPages);
      const added = nextPages.filter((page) => !taskSet.has(page));
      const nextSet = new Set(nextPages);
      if (added.length) {
        setPendingStructureAddonPages((prev) =>
          Array.from(new Set([...prev.filter((page) => nextSet.has(page)), ...added])).sort((a, b) => a - b),
        );
      } else {
        setPendingStructureAddonPages((prev) => prev.filter((page) => nextSet.has(page)));
      }
      setStructureSelectionFeedback(
        added.length
          ? `Selection updated (${delta}). Added page${added.length === 1 ? '' : 's'} are selected for your next extraction.`
          : `Selection updated (${delta}). Task running; click Extract after it finishes to process the new page set.`,
      );
      return;
    }
    if (structureTask.status === 'completed') {
      setStructureSelectionFeedback(
        `Selection updated (${delta}). Click Extract to process the updated page set.`,
      );
    }
  }, [structureTask]);

  const reportAssaySelectionEdit = React.useCallback((nextPages: number[]) => {
    const taskPages = getTaskSelectedPages(assayTask);
    if (!assayTask || !taskPages.length || assayTask.task_id.startsWith('pending-')) return;
    const delta = summarizePageDelta(nextPages, taskPages);
    if (!delta) {
      if (isTaskInFlight(assayTask)) {
        setPendingAssayAddonPages([]);
      }
      setAssaySelectionFeedback(null);
      return;
    }
    if (isTaskInFlight(assayTask)) {
      const taskSet = new Set(taskPages);
      const added = nextPages.filter((page) => !taskSet.has(page));
      const nextSet = new Set(nextPages);
      if (added.length) {
        setPendingAssayAddonPages((prev) =>
          Array.from(new Set([...prev.filter((page) => nextSet.has(page)), ...added])).sort((a, b) => a - b),
        );
      } else {
        setPendingAssayAddonPages((prev) => prev.filter((page) => nextSet.has(page)));
      }
      setAssaySelectionFeedback(
        added.length
          ? `Selection updated (${delta}). Added page${added.length === 1 ? '' : 's'} are selected for your next extraction.`
          : `Selection updated (${delta}). Task running; click Extract after it finishes to process the new page set.`,
      );
      return;
    }
    if (assayTask.status === 'completed') {
      setAssaySelectionFeedback(
        `Selection updated (${delta}). Click Extract to process the updated page set.`,
      );
    }
  }, [assayTask]);

  const updateStructureSelection = React.useCallback((updater: (prev: Set<number>) => Set<number>) => {
    setStructureSelection((prev) => {
      const next = updater(prev);
      const sorted = Array.from(next).sort((a, b) => a - b);
      setStructurePagesInput(sorted.length ? pagesToString(sorted) : '');
      reportStructureSelectionEdit(sorted);
      return next;
    });
  }, [reportStructureSelectionEdit]);

  const updateAssaySelection = React.useCallback((updater: (prev: Set<number>) => Set<number>) => {
    setAssaySelection((prev) => {
      const next = updater(prev);
      const sorted = Array.from(next).sort((a, b) => a - b);
      setAssayPagesInput(sorted.length ? pagesToString(sorted) : '');
      reportAssaySelectionEdit(sorted);
      return next;
    });
  }, [reportAssaySelectionEdit]);

  const toggleStructurePage = (page: number) => {
    updateStructureSelection((prev) => {
      const next = new Set(prev);
      if (next.has(page)) {
        next.delete(page);
      } else {
        next.add(page);
      }
      return next;
    });
  };

  const toggleAssayPage = (page: number) => {
    updateAssaySelection((prev) => {
      const next = new Set(prev);
      if (next.has(page)) {
        next.delete(page);
      } else {
        next.add(page);
      }
      return next;
    });
  };

  const handleCardClick = (page: number, event: { shiftKey: boolean }) => {
    resetNotifications();
    const shiftPressed = event.shiftKey;
    if (activeSelectionMode === 'assay') {
      setAutoDetectAssayPages(false);
      if (shiftPressed && lastAssayPageRef.current !== null) {
        const start = Math.min(lastAssayPageRef.current, page);
        const end = Math.max(lastAssayPageRef.current, page);
        updateAssaySelection((prev) => {
          const next = new Set(prev);
          for (let idx = start; idx <= end; idx += 1) {
            next.add(idx);
          }
          return next;
        });
      } else {
        toggleAssayPage(page);
      }
      lastAssayPageRef.current = page;
    } else {
      setAutoDetectStructurePages(false);
      if (shiftPressed && lastStructurePageRef.current !== null) {
        const start = Math.min(lastStructurePageRef.current, page);
        const end = Math.max(lastStructurePageRef.current, page);
        updateStructureSelection((prev) => {
          const next = new Set(prev);
          for (let idx = start; idx <= end; idx += 1) {
            next.add(idx);
          }
          return next;
        });
      } else {
        toggleStructurePage(page);
      }
      lastStructurePageRef.current = page;
    }
  };

  const renderPageGallery = (mode: SelectionMode) => (
    <div
      className="gallery-grid"
      ref={mode === 'structure' ? setStructureGalleryRef : setAssayGalleryRef}
    >
      {pages.map((page) => {
        const imageLoaded = Boolean(pageImages[page]);
        const isStructureSelected = structureSelection.has(page);
        const isAssaySelected = assaySelection.has(page);
        const isActiveSelected = mode === 'assay' ? isAssaySelected : isStructureSelected;
        const isProcessing = mode === 'assay' ? activeAssayPages.has(page) : activeStructurePages.has(page);
        const classNames = [
          'page-card',
          isStructureSelected ? 'structure-selected' : '',
          isAssaySelected ? 'assay-selected' : '',
          isActiveSelected ? `active-mode-${mode}` : '',
          isProcessing ? `page-card--processing page-card--processing-${mode}` : '',
        ]
          .filter(Boolean)
          .join(' ');
        return (
          <div
            key={page}
            className={classNames}
            onClick={(event) => handleCardClick(page, event)}
            onKeyDown={(event) => {
              if (event.target !== event.currentTarget) return;
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                handleCardClick(page, event);
              }
            }}
            aria-label={`Page ${page}`}
            aria-pressed={isActiveSelected}
            role="button"
            tabIndex={0}
            data-page={page}
          >
            <div className="page-card__inner">
              <div className="page-card__body">
                {imageLoaded ? (
                  <img src={`data:image/png;base64,${pageImages[page]}`} alt={`Page ${page}`} />
                ) : (
                  <div className="page-card__placeholder">Loading…</div>
                )}
              </div>
              <div className="page-card__footer">
                <div className="page-card__magnify">
                <button
                  className="magnify-btn"
                  onClick={(event) => {
                    event.stopPropagation();
                    pdfLocateSequenceRef.current += 1;
                    setActiveReviewRowIndex(null); setPdfTarget(null); setDocumentPage(page);
                  }}
                  type="button"
                  title="Open full-size image" aria-label={`Enlarge page ${page}`}
                >
                  <Icon name="search" className="button-icon button-icon--small" />
                </button>
              </div>
                <span className="page-card__label">Page {page}</span>
                <div className="flex-gap">
                  {isProcessing && <span className="badge badge-processing">Processing</span>}
                  {isStructureSelected && <span className="badge badge-structure">S</span>}
                  {isAssaySelected && <span className="badge badge-assay">A</span>}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );

  const handleStructureInputChange: React.ChangeEventHandler<HTMLTextAreaElement> = (event) => {
    const value = event.target.value;
    setStructurePagesInput(value);
    const parsed = parsePagesInput(value);
    setStructureSelection(new Set<number>(parsed));
    reportStructureSelectionEdit(parsed);
    lastStructurePageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
  };

  const handleAssayInputChange: React.ChangeEventHandler<HTMLTextAreaElement> = (event) => {
    const value = event.target.value;
    setAssayPagesInput(value);
    const parsed = parsePagesInput(value);
    setAssaySelection(new Set<number>(parsed));
    reportAssaySelectionEdit(parsed);
    lastAssayPageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
  };

  const addAssayNames = React.useCallback((raw: string) => {
    const extras = parseAssayNames(raw);
    if (!extras.length) return;
    setAssayNames((prev) => mergeAssayNameLists(prev, extras));
    if (isTaskInFlight(assayTask) || isTaskInFlight(assayAddonTask)) {
      setAssaySelectionFeedback('Assay names updated. Running tasks keep their submitted names; changes apply to the next run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names updated. Click Extract to re-run with the new names.');
    }
  }, [assayAddonTask, assayTask]);

  const removeAssayName = (index: number) => {
    setAssayNames((prev) => prev.filter((_, idx) => idx !== index));
    if (isTaskInFlight(assayTask) || isTaskInFlight(assayAddonTask)) {
      setAssaySelectionFeedback('Assay names updated. Running tasks keep their submitted names; changes apply to the next run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names updated. Click Extract to re-run with the new names.');
    }
  };

  const clearAssayNames = () => {
    setAssayNames([]);
    setAssayNameDraft('');
    if (isTaskInFlight(assayTask) || isTaskInFlight(assayAddonTask)) {
      setAssaySelectionFeedback('Assay names cleared. Running tasks keep their submitted names; changes apply to the next run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names cleared. Click Extract to re-run.');
    }
  };

  const commitAssayDraft = React.useCallback(() => {
    if (!assayNameDraft.trim()) return;
    addAssayNames(assayNameDraft);
    setAssayNameDraft('');
  }, [assayNameDraft, addAssayNames]);

  const handleAssayNameKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (event) => {
    if (event.key === 'Enter' || event.key === 'Tab' || event.key === ',' || event.key === ';') {
      event.preventDefault();
      commitAssayDraft();
    }
  };

  const handleAssayNamePaste: React.ClipboardEventHandler<HTMLInputElement> = (event) => {
    const data = event.clipboardData.getData('text');
    if (!data) return;
    event.preventDefault();
    addAssayNames(data);
    setAssayNameDraft('');
  };

  const loadExplicitStructureSource = React.useCallback(
    async (sourceTaskId: string, isCurrentRequest?: () => boolean) => {
      if (!isUsableId(sourceTaskId)) return;
      const sourceTask = await fetchTask(sourceTaskId);
      if (isCurrentRequest && !isCurrentRequest()) return;
      if (
        sourceTask.status !== 'completed' ||
        (sourceTask.type !== 'structure_extraction' && sourceTask.type !== 'full_pipeline')
      ) {
        return;
      }

      if (sourceTask.type === 'structure_extraction') {
        setStructureTask(sourceTask);
        setStructureFilterStrictness(coerceStructureFilterStrictness(sourceTask.params?.structure_filter_strictness));
        setAutoDetectStructurePages(coerceAutoDetectionFlag(sourceTask.params?.auto_detect_pages));
        applyDetectedStructurePages(sourceTask.params?.detected_pages ?? sourceTask.params?.pages);
      } else {
        setFullPipelineTask(sourceTask);
        const sourceParams = sourceTask.params as Record<string, unknown>;
        if (Object.prototype.hasOwnProperty.call(sourceParams ?? {}, 'detected_structure_pages')) {
          applyPlannedStructurePages(sourceParams?.detected_structure_pages);
        }
        if (Object.prototype.hasOwnProperty.call(sourceParams ?? {}, 'detected_assay_pages')) {
          applyPlannedAssayPages(sourceParams?.detected_assay_pages);
        }
        applyDetectedAssayNames(sourceParams?.detected_assay_names);
      }

      const structureResults = await fetchTaskStructures(sourceTaskId);
      if (isCurrentRequest && !isCurrentRequest()) return;
      const nextRecords = structureResults.records.map((record) => ({ ...record }));
      const nextFilteredRecords = (structureResults.filtered_records ?? []).map((record) => ({ ...record }));
      setStructures(structureResults.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setLoadedStructureTaskId(sourceTaskId);
      setFilteredStructures(nextFilteredRecords);
      setSaveStatus(structureResults.records.length ? 'saved' : 'idle');
    },
    [
      applyDetectedAssayNames,
      applyDetectedStructurePages,
      applyPlannedAssayPages,
      applyPlannedStructurePages,
      coerceAutoDetectionFlag,
      coerceStructureFilterStrictness,
    ],
  );

  const refreshStructureTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.structure !== taskId) return;
    setStructureTask(updated);
    applyDetectedStructurePages(updated.params?.detected_pages ?? updated.params?.pages);
    if (updated.status === 'completed') {
      const results = await fetchTaskStructures(taskId);
      if (trackedTasksRef.current.structure !== taskId) return;
      const nextRecords = results.records.map((record) => ({ ...record }));
      const nextFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
      setStructures(results.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setLoadedStructureTaskId(taskId);
      setFilteredStructures(nextFilteredRecords);
      setSaveStatus(results.records.length ? 'saved' : 'idle');
      setToast(
        `Structure extraction complete: ${results.records.length} record${
          results.records.length === 1 ? '' : 's'
        }${nextFilteredRecords.length ? `, ${nextFilteredRecords.length} filtered` : ''}. Continue in Step 3 for bioactivity.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Structure extraction failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Structure extraction canceled.');
    }
  }, [applyDetectedStructurePages]);

  const refreshStructureAddonTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.structureAddon !== taskId) return;
    setStructureAddonTask(updated);
    if (updated.status === 'completed') {
      const results = await fetchTaskStructures(taskId);
      if (trackedTasksRef.current.structureAddon !== taskId) return;
      const addRecords = results.records.map((record) => ({ ...record }));
      const addFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
      const mergedRecords = mergeStructureRecordLists(editedStructuresRef.current, addRecords);
      const mergedFilteredRecords = mergeStructureRecordLists(filteredStructures, addFilteredRecords);
      editedStructuresRef.current = mergedRecords;
      setStructures(mergedRecords);
      setEditedStructures(mergedRecords);
      setFilteredStructures(mergedFilteredRecords);
      setSaveStatus(mergedRecords.length ? 'saved' : 'idle');
      const completedPages = getTaskSelectedPages(updated);
      const completedSet = new Set(
        completedPages.length ? completedPages : structureAddonSubmittedPagesRef.current,
      );
      setPendingStructureAddonPages((prev) => prev.filter((page) => !completedSet.has(page)));
      structureAddonSubmittedPagesRef.current = [];
      setStructureSelectionFeedback(null);
      if (structureTask?.status === 'completed') {
        void updateTaskStructures(structureTask.task_id, mergedRecords).catch(() => {
          setSaveStatus('error');
        });
      }
      setToast(
        `Structure add-on complete: added ${addRecords.length} record${addRecords.length === 1 ? '' : 's'}.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Additional structure extraction failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Additional structure extraction canceled.');
    }
  }, [filteredStructures, structureTask]);

  const refreshAssayTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.assay !== taskId) return;
    setAssayTask(updated);
    applyDetectedAssayPages(updated.params?.detected_pages ?? updated.params?.pages);
    applyDetectedAssayNames(updated.params?.detected_assay_names);
    if (updated.status === 'completed') {
      const results = await fetchTaskAssays(taskId);
      if (trackedTasksRef.current.assay !== taskId) return;
      setAssayRecords(results.records);
      setToast(
        `Bioactivity extraction complete: ${results.records.length} record${
          results.records.length === 1 ? '' : 's'
        }. Review in Step 4.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Bioactivity extraction failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Bioactivity extraction canceled.');
    }
  }, [applyDetectedAssayNames, applyDetectedAssayPages]);

  const refreshAssayAddonTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.assayAddon !== taskId) return;
    setAssayAddonTask(updated);
    if (updated.status === 'completed') {
      const results = await fetchTaskAssays(taskId);
      if (trackedTasksRef.current.assayAddon !== taskId) return;
      setAssayRecords((prev) => mergeAssayRecordLists(prev, results.records));
      const completedPages = getTaskSelectedPages(updated);
      const completedSet = new Set(
        completedPages.length ? completedPages : assayAddonSubmittedPagesRef.current,
      );
      setPendingAssayAddonPages((prev) => prev.filter((page) => !completedSet.has(page)));
      assayAddonSubmittedPagesRef.current = [];
      setAssaySelectionFeedback(null);
      setToast(
        `Bioactivity add-on complete: merged ${results.records.length} record${results.records.length === 1 ? '' : 's'}.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Additional bioactivity extraction failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Additional bioactivity extraction canceled.');
    }
  }, []);

  const refreshAutoDetectTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.autoDetect !== taskId) return;
    setAutoDetectTask(updated);
    if (Object.prototype.hasOwnProperty.call(updated.params ?? {}, 'detected_structure_pages')) {
      applyPlannedStructurePages(updated.params?.detected_structure_pages);
      const pages = parsePageListParam(updated.params?.detected_structure_pages);
      if (pages.length > 0 && !structurePagesAutoAdvancedRef.current) {
        structurePagesAutoAdvancedRef.current = true;
        setCurrentStep((prev) => (prev < 2 ? 2 : prev));
      }
    }
    if (Object.prototype.hasOwnProperty.call(updated.params ?? {}, 'detected_assay_pages')) {
      applyPlannedAssayPages(updated.params?.detected_assay_pages);
      const pages = parsePageListParam(updated.params?.detected_assay_pages);
      if (pages.length > 0 && !assayPagesAutoAdvancedRef.current) {
        assayPagesAutoAdvancedRef.current = true;
        setCurrentStep((prev) => (prev < 3 ? 3 : prev));
      }
    }
    applyDetectedAssayNames(updated.params?.detected_assay_names);
    if (updated.status === 'completed') {
      setToast('Detection ready. Review Steps 2-3, then start extraction.');
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Automatic detection failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Automatic detection canceled.');
    }
  }, [applyDetectedAssayNames, applyPlannedAssayPages, applyPlannedStructurePages]);

  const refreshFullPipelineTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    if (trackedTasksRef.current.fullPipeline !== taskId) return;
    setFullPipelineTask(updated);
    const params = updated.params as Record<string, unknown>;

    // Mirror auto-detect: apply detected pages on every poll so step 2/3 stay in sync
    if (Object.prototype.hasOwnProperty.call(params ?? {}, 'detected_structure_pages')) {
      applyPlannedStructurePages(params?.detected_structure_pages);
      const pages = parsePageListParam(params?.detected_structure_pages);
      if (pages.length > 0 && !structurePagesAutoAdvancedRef.current) {
        structurePagesAutoAdvancedRef.current = true;
        setCurrentStep((prev) => (prev < 2 ? 2 : prev));
      }
    }
    if (Object.prototype.hasOwnProperty.call(params ?? {}, 'detected_assay_pages')) {
      applyPlannedAssayPages(params?.detected_assay_pages);
      const pages = parsePageListParam(params?.detected_assay_pages);
      if (pages.length > 0 && updated.progress >= 0.55 && !assayPagesAutoAdvancedRef.current) {
        assayPagesAutoAdvancedRef.current = true;
        setCurrentStep((prev) => (prev < 3 ? 3 : prev));
      }
    }
    if (params?.detected_assay_names) {
      applyDetectedAssayNames(params?.detected_assay_names);
    }

    if (updated.status === 'completed') {
      const [structureResults, assayResults] = await Promise.all([
        fetchTaskStructures(taskId),
        fetchTaskAssays(taskId),
      ]);
      if (trackedTasksRef.current.fullPipeline !== taskId) return;
      const nextRecords = structureResults.records.map((record) => ({ ...record }));
      setStructures(structureResults.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setLoadedStructureTaskId(taskId);
      setFilteredStructures(structureResults.filtered_records ?? []);
      setAssayRecords(assayResults.records);
      setSaveStatus('saved');
      const structureCount = structureResults.records.length;
      const filteredCount = (structureResults.filtered_records ?? []).length;
      const assayCount = assayResults.records.length;
      setToast(
        `Pipeline complete: ${structureCount} structure${structureCount === 1 ? '' : 's'}` +
        `${filteredCount ? `, ${filteredCount} filtered` : ''}` +
        `, ${assayCount} assay record${assayCount === 1 ? '' : 's'}. Review results in Step 4.`,
      );
      setCurrentStep(4);
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Full pipeline failed');
    }
    if (updated.status === 'canceled') {
      setToast(updated.message || 'Full pipeline canceled.');
    }
  }, [applyPlannedStructurePages, applyPlannedAssayPages, applyDetectedAssayNames]);

  const handleOpenJob = React.useCallback(async (job: TaskStatus) => {
    if (!(await flushPendingEdits())) return;
    resetNotifications();
    suppressAutoStepUntilNextJobRef.current = false;
    const requestSeq = openJobRequestSeqRef.current + 1;
    openJobRequestSeqRef.current = requestSeq;
    const isCurrentOpenRequest = () => requestSeq === openJobRequestSeqRef.current;
    const listedPdfId = isUsableId(job.pdf_id) ? job.pdf_id : null;
    const listedPdfChanged = Boolean(listedPdfId && listedPdfId !== pdfInfoRef.current?.pdf_id);

    clearJobResultState({ clearPageCache: listedPdfChanged });
    setSelectedJobId(job.task_id);

    try {
      const updated = await fetchTask(job.task_id);
      if (!isCurrentOpenRequest()) return;
      setSelectedJobId(updated.task_id);

      const updatedPdfId = isUsableId(updated.pdf_id) ? updated.pdf_id : null;
      const updatedPdfChanged = Boolean(updatedPdfId && updatedPdfId !== pdfInfoRef.current?.pdf_id);
      if (updatedPdfChanged && !listedPdfChanged) {
        clearJobResultState({ clearPageCache: true });
        setSelectedJobId(updated.task_id);
      }
      if (updatedPdfId && updatedPdfChanged) {
        try {
          const info = await fetchPdfInfo(updatedPdfId);
          if (!isCurrentOpenRequest()) return;
          setPdfInfo(info);
        } catch (err) {
          console.warn('Failed to load PDF for job', err);
        }
      }
      if (updated.type === 'auto_detect_plan') {
        if (!isCurrentOpenRequest()) return;
        setAutoDetectTask(updated);
        if (Object.prototype.hasOwnProperty.call(updated.params ?? {}, 'detected_structure_pages')) {
          applyPlannedStructurePages(updated.params?.detected_structure_pages);
        }
        if (Object.prototype.hasOwnProperty.call(updated.params ?? {}, 'detected_assay_pages')) {
          applyPlannedAssayPages(updated.params?.detected_assay_pages);
        }
        applyDetectedAssayNames(updated.params?.detected_assay_names);
        setCurrentStep(
          Object.prototype.hasOwnProperty.call(updated.params ?? {}, 'detected_assay_pages') ? 3 : 2,
        );
        setToast(updated.status === 'completed' ? 'Detection job opened.' : 'Detection job added to this workspace.');
        return;
      }

      if (updated.type === 'structure_extraction') {
        if (!isCurrentOpenRequest()) return;
        setStructureTask(updated);
        setStructureFilterStrictness(coerceStructureFilterStrictness(updated.params?.structure_filter_strictness));
        setAutoDetectStructurePages(coerceAutoDetectionFlag(updated.params?.auto_detect_pages));
        applyDetectedStructurePages(updated.params?.detected_pages ?? updated.params?.pages);
        if (updated.status === 'completed') {
          const results = await fetchTaskStructures(updated.task_id);
          if (!isCurrentOpenRequest()) return;
          const nextRecords = results.records.map((record) => ({ ...record }));
          const nextFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
          setStructures(results.records);
          editedStructuresRef.current = nextRecords;
          setEditedStructures(nextRecords);
          setLoadedStructureTaskId(updated.task_id);
          setFilteredStructures(nextFilteredRecords);
          setSaveStatus(results.records.length ? 'saved' : 'idle');
          setCurrentStep(4);
          setToast('Structure job result opened.');
        } else {
          setCurrentStep(2);
          setToast('Structure job added to this workspace.');
        }
        return;
      }

      if (updated.type === 'bioactivity_extraction') {
        if (!isCurrentOpenRequest()) return;
        setAssayTask(updated);
        setAutoDetectAssayPages(coerceAutoDetectionFlag(updated.params?.auto_detect_pages));
        setAutoDetectAssayNames(coerceAutoDetectionFlag(updated.params?.auto_detect_assay_names));
        applyDetectedAssayPages(updated.params?.detected_pages ?? updated.params?.pages);
        applyDetectedAssayNames(updated.params?.detected_assay_names);
        if (updated.status === 'completed') {
          const linkedStructureTaskId =
            typeof updated.params?.structure_task_id === 'string' ? updated.params.structure_task_id : '';
          if (isUsableId(linkedStructureTaskId)) {
            try {
              await loadExplicitStructureSource(linkedStructureTaskId, isCurrentOpenRequest);
            } catch (err) {
              console.warn('Failed to load linked structure job', err);
            }
          }
          const results = await fetchTaskAssays(updated.task_id);
          if (!isCurrentOpenRequest()) return;
          setAssayRecords(results.records);
          setCurrentStep(4);
          setToast('Bioactivity job result opened.');
        } else {
          setCurrentStep(3);
          setToast('Bioactivity job added to this workspace.');
        }
        return;
      }

      if (updated.type === 'full_pipeline') {
        if (!isCurrentOpenRequest()) return;
        setFullPipelineTask(updated);
        const params = updated.params as Record<string, unknown>;
        if (Object.prototype.hasOwnProperty.call(params ?? {}, 'detected_structure_pages')) {
          applyPlannedStructurePages(params?.detected_structure_pages);
        }
        if (Object.prototype.hasOwnProperty.call(params ?? {}, 'detected_assay_pages')) {
          applyPlannedAssayPages(params?.detected_assay_pages);
        }
        applyDetectedAssayNames(params?.detected_assay_names);
        if (updated.status === 'completed') {
          const [structureResults, assayResults] = await Promise.all([
            fetchTaskStructures(updated.task_id),
            fetchTaskAssays(updated.task_id),
          ]);
          if (!isCurrentOpenRequest()) return;
          const nextStructures = structureResults.records.map((record) => ({ ...record }));
          setStructures(structureResults.records);
          editedStructuresRef.current = nextStructures;
          setEditedStructures(nextStructures);
          setLoadedStructureTaskId(updated.task_id);
          setFilteredStructures(structureResults.filtered_records ?? []);
          setAssayRecords(assayResults.records);
          setCurrentStep(4);
          setToast('Full pipeline job result opened.');
        } else {
          setCurrentStep(updated.progress >= 0.55 ? 3 : 2);
          setToast('Full pipeline job added to this workspace.');
        }
        return;
      }

      setToast('This job type has no dedicated result view yet.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open this job.');
    }
  }, [
    applyDetectedAssayNames,
    applyDetectedAssayPages,
    applyDetectedStructurePages,
    applyPlannedAssayPages,
    applyPlannedStructurePages,
    clearJobResultState,
    coerceAutoDetectionFlag,
    coerceStructureFilterStrictness,
    loadExplicitStructureSource,
    flushPendingEdits,
  ]);

  const submitAssayTask = React.useCallback(
    async (request: AssayTaskRequest) => {
      try {
        setIsAssaySubmitting(true);
        const taskStatus = await queueAssayTask(request);
        setAssayTask(taskStatus);
        setAssaySelectionFeedback(null);
        setAssayRecords([]);
        setToast('Bioactivity extraction submitted.');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to submit bioactivity extraction task');
      } finally {
        setIsAssaySubmitting(false);
      }
    },
    [queueAssayTask],
  );

  React.useEffect(() => {
    if (!autoDetectTask || isTaskTerminal(autoDetectTask)) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshAutoDetectTask(autoDetectTask.task_id).catch((err) => {
        if (trackedTasksRef.current.autoDetect !== autoDetectTask.task_id) return;
        if (isHttpNotFound(err)) {
          setAutoDetectTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh automatic detection task');
      }).finally(() => { refreshing = false; });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [autoDetectTask, refreshAutoDetectTask]);

  React.useEffect(() => {
    if (!fullPipelineTask || isTaskTerminal(fullPipelineTask)) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshFullPipelineTask(fullPipelineTask.task_id).catch((err) => {
        if (trackedTasksRef.current.fullPipeline !== fullPipelineTask.task_id) return;
        if (isHttpNotFound(err)) {
          setFullPipelineTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh full pipeline task');
      }).finally(() => { refreshing = false; });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [fullPipelineTask, refreshFullPipelineTask]);

  React.useEffect(() => {
    if (!structureTask || isTaskTerminal(structureTask)) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshStructureTask(structureTask.task_id).catch((err) => {
        if (trackedTasksRef.current.structure !== structureTask.task_id) return;
        if (isHttpNotFound(err)) {
          setStructureTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh structure task');
      }).finally(() => { refreshing = false; });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [structureTask, refreshStructureTask]);

  React.useEffect(() => {
    if (!structureAddonTask || isTaskTerminal(structureAddonTask)) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshStructureAddonTask(structureAddonTask.task_id).catch((err) => {
        if (trackedTasksRef.current.structureAddon !== structureAddonTask.task_id) return;
        if (isHttpNotFound(err)) {
          setStructureAddonTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh additional structure task');
      }).finally(() => { refreshing = false; });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [structureAddonTask, refreshStructureAddonTask]);

  React.useEffect(() => {
    if (!structureTask) return;
    if (structureTask.status === 'completed') {
      if (!pendingAssayRequestRef.current) return;
      const pendingRequest: AssayTaskRequest = {
        ...pendingAssayRequestRef.current,
        structure_task_id: structureTask.task_id,
      };
      pendingAssayRequestRef.current = null;
      setIsAssayWaitingForStructures(false);
      void submitAssayTask(pendingRequest);
      return;
    }
    if (structureTask.status === 'failed' || structureTask.status === 'canceled') {
      if (!pendingAssayRequestRef.current) return;
      pendingAssayRequestRef.current = null;
      setIsAssayWaitingForStructures(false);
      setError('Structure extraction did not complete. Start bioactivity extraction again.');
    }
  }, [structureTask, submitAssayTask]);

  React.useEffect(() => {
    if (
      !assayTask ||
      isTaskTerminal(assayTask) ||
      assayTask.task_id.startsWith('pending-')
    ) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshAssayTask(assayTask.task_id).catch((err) => {
        if (trackedTasksRef.current.assay !== assayTask.task_id) return;
        if (isHttpNotFound(err)) {
          setAssayTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh bioactivity task');
      }).finally(() => { refreshing = false; });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [assayTask, refreshAssayTask]);

  React.useEffect(() => {
    if (!assayAddonTask || isTaskTerminal(assayAddonTask)) {
      return;
    }
    let refreshing = false;
    const interval = window.setInterval(() => {
      if (document.hidden || refreshing) return;
      refreshing = true;
      void refreshAssayAddonTask(assayAddonTask.task_id).catch((err) => {
        if (trackedTasksRef.current.assayAddon !== assayAddonTask.task_id) return;
        if (isHttpNotFound(err)) {
          setAssayAddonTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh additional bioactivity task');
      }).finally(() => { refreshing = false; });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [assayAddonTask, refreshAssayAddonTask]);

  const handleStartStructureExtraction = async () => {
    resetNotifications();
    if (!pdfInfo) {
      setError('Upload a PDF first.');
      return;
    }
    const pagesSelected = Array.from(structureSelection).sort((a, b) => a - b);
    const useSelectedPages = pagesSelected.length > 0;
    const useAutomaticPages = !useSelectedPages && autoDetectStructurePages;
    if (!useSelectedPages && !useAutomaticPages) {
      setError('Select structure pages or enable automatic page detection.');
      return;
    }
    const pageString = pagesToString(pagesSelected);
    if (useSelectedPages) {
      setStructurePagesInput(pageString);
    }
    try {
      setIsStructureSubmitting(true);
      if (useAutomaticPages) {
        appliedStructureDetectedPagesRef.current = '';
      }
      const taskStatus = await queueStructureTask({
        pdf_id: pdfInfo.pdf_id,
        pages: useSelectedPages ? pageString : undefined,
        auto_detect_pages: useAutomaticPages,
        structure_filter_strictness: structureFilterStrictness,
        assemble_markush: assembleMarkush && assemblyAvailable,
      });
      suppressAutoStepUntilNextJobRef.current = false;
      setStructureTask(taskStatus);
      setStructureAddonTask(null);
      setPendingStructureAddonPages([]);
      structureAddonSubmittedPagesRef.current = [];
      setStructureSelectionFeedback(null);
      setStructures([]);
      setEditedStructures([]);
      setFilteredStructures([]);
      editedStructuresRef.current = [];
      setLoadedStructureTaskId(undefined);
      setSaveStatus('idle');
      setToast('Structure extraction submitted.');
      autoAdvanceRef.current = false;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit structure extraction task');
    } finally {
      setIsStructureSubmitting(false);
    }
  };

  const handleStartAssayExtraction = async () => {
    resetNotifications();
    if (!pdfInfo) {
      setError('Upload a PDF first.');
      return;
    }
    const pagesSelected = Array.from(assaySelection).sort((a, b) => a - b);
    const useSelectedPages = pagesSelected.length > 0;
    const useAutomaticPages = !useSelectedPages && autoDetectAssayPages;
    if (!useSelectedPages && !useAutomaticPages) {
      setError('Select bioactivity pages or enable automatic page detection.');
      return;
    }
    let namesList = assayNames;
    if (assayNameDraft.trim()) {
      const pending = parseAssayNames(assayNameDraft);
      const merged = pending.length ? mergeAssayNameLists(assayNames, pending) : assayNames;
      setAssayNames(merged);
      namesList = merged;
      setAssayNameDraft('');
    }
    const useProvidedNames = namesList.length > 0;
    const useAutomaticNames = !useProvidedNames && autoDetectAssayNames;
    if (!useProvidedNames && !useAutomaticNames) {
      setError('Add an assay name or enable automatic assay-name detection.');
      return;
    }
    const pageString = pagesToString(pagesSelected);
    if (useSelectedPages) {
      setAssayPagesInput(pageString);
    }

    const requestBase: AssayTaskRequest = {
      pdf_id: pdfInfo.pdf_id,
      pages: useSelectedPages ? pageString : undefined,
      assay_names: useProvidedNames ? namesList : [],
      auto_detect_pages: useAutomaticPages,
      auto_detect_assay_names: useAutomaticNames,
    };
    if (useAutomaticPages) {
      appliedAssayDetectedPagesRef.current = '';
    }
    if (useAutomaticNames) {
      appliedAssayNamesRef.current = '';
    }

    if (structureTask && structureTask.status !== 'completed') {
      suppressAutoStepUntilNextJobRef.current = false;
      pendingAssayRequestRef.current = requestBase;
      setIsAssayWaitingForStructures(true);
      const now = new Date().toISOString();
      const pendingParams: Record<string, unknown> = { ...requestBase };
      setAssayTask({
        task_id: `pending-${structureTask.task_id}`,
        type: 'assay',
        status: 'pending',
        progress: 0,
        message: 'Waiting for structure extraction…',
        pdf_id: pdfInfo.pdf_id,
        params: pendingParams,
        created_at: now,
        updated_at: now,
      });
      setToast('Bioactivity extraction queued; waiting for structure extraction.');
      return;
    }

    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);
    setAssayAddonTask(null);
    setPendingAssayAddonPages([]);
    assayAddonSubmittedPagesRef.current = [];
    setAssaySelectionFeedback(null);
    const explicitStructureSourceTaskId = getExplicitStructureSourceTaskId(structureTask, fullPipelineTask);
    const finalRequest: AssayTaskRequest = {
      ...requestBase,
      structure_task_id: explicitStructureSourceTaskId,
    };
    suppressAutoStepUntilNextJobRef.current = false;
    await submitAssayTask(finalRequest);
  };

  const handleStartAutomaticExtraction = async () => {
    resetNotifications();
    const runId = automaticExtractionRunRef.current + 1;
    automaticExtractionRunRef.current = runId;
    if (!pdfInfo) {
      setError('Upload a PDF first.');
      return;
    }
    if (!pdfReadyForDetection) {
      setError('Wait for page previews to finish loading.');
      return;
    }

    if (!autoDetectStructurePages && !autoDetectAssayPages && !autoDetectAssayNames) {
      setError('Enable at least one detection target.');
      return;
    }

    let namesList = assayNames;
    if (assayNameDraft.trim()) {
      const pending = parseAssayNames(assayNameDraft);
      const merged = pending.length ? mergeAssayNameLists(assayNames, pending) : assayNames;
      setAssayNames(merged);
      namesList = merged;
      setAssayNameDraft('');
    }

    try {
      setIsStructureSubmitting(true);
      setIsAssaySubmitting(true);
      if (autoDetectStructurePages) {
        appliedStructureDetectedPagesRef.current = '';
      }
      if (autoDetectAssayPages) {
        appliedAssayDetectedPagesRef.current = '';
      }
      if (autoDetectAssayNames) {
        appliedAssayNamesRef.current = '';
      }

      const request: AutoDetectTaskRequest = {
        pdf_id: pdfInfo.pdf_id,
        assay_names: namesList,
        detect_structure_pages: autoDetectStructurePages,
        detect_assay_pages: autoDetectAssayPages,
        detect_assay_names: autoDetectAssayNames,
      };
      const taskStatus = await queueAutoDetectTask(request);
      if (automaticExtractionRunRef.current !== runId) {
        return;
      }

      suppressAutoStepUntilNextJobRef.current = false;
      setAutoDetectTask(taskStatus);
      setStructureSelectionFeedback(null);
      setAssaySelectionFeedback(null);
      setStructureAddonTask(null);
      setAssayAddonTask(null);
      setPendingStructureAddonPages([]);
      setPendingAssayAddonPages([]);
      structureAddonSubmittedPagesRef.current = [];
      assayAddonSubmittedPagesRef.current = [];
      autoAdvanceRef.current = false;
      structurePagesAutoAdvancedRef.current = false;
      assayPagesAutoAdvancedRef.current = false;
      setToast('Automatic detection started. Extraction starts after your review.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit automatic detection task');
    } finally {
      if (automaticExtractionRunRef.current === runId) {
        setIsStructureSubmitting(false);
        setIsAssaySubmitting(false);
      }
    }
  };

  const handleStartStructureDetection = async () => {
    resetNotifications();
    if (!pdfInfo) { setError('Upload a PDF first.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for page previews to finish loading.'); return; }
    try {
      setIsStructureSubmitting(true);
      appliedStructureDetectedPagesRef.current = '';
      structurePagesAutoAdvancedRef.current = false;
      const request: AutoDetectTaskRequest = {
        pdf_id: pdfInfo.pdf_id,
        detect_structure_pages: true,
        detect_assay_pages: false,
        detect_assay_names: false,
      };
      const taskStatus = await queueAutoDetectTask(request);
      suppressAutoStepUntilNextJobRef.current = false;
      setAutoDetectTask(taskStatus);
      setStructureSelectionFeedback(null);
      setStructureAddonTask(null);
      setPendingStructureAddonPages([]);
      structureAddonSubmittedPagesRef.current = [];
      setToast('Structure page detection started.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit structure detection');
    } finally {
      setIsStructureSubmitting(false);
    }
  };

  const handleStartAssayDetection = async () => {
    resetNotifications();
    if (!pdfInfo) { setError('Upload a PDF first.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for page previews to finish loading.'); return; }
    let namesList = assayNames;
    if (assayNameDraft.trim()) {
      const pending = parseAssayNames(assayNameDraft);
      const merged = pending.length ? mergeAssayNameLists(assayNames, pending) : assayNames;
      setAssayNames(merged);
      namesList = merged;
      setAssayNameDraft('');
    }
    try {
      setIsAssaySubmitting(true);
      appliedAssayDetectedPagesRef.current = '';
      appliedAssayNamesRef.current = '';
      assayPagesAutoAdvancedRef.current = false;
      const request: AutoDetectTaskRequest = {
        pdf_id: pdfInfo.pdf_id,
        assay_names: namesList,
        detect_structure_pages: false,
        detect_assay_pages: true,
        detect_assay_names: true,
      };
      const taskStatus = await queueAutoDetectTask(request);
      suppressAutoStepUntilNextJobRef.current = false;
      setAutoDetectTask(taskStatus);
      setAssaySelectionFeedback(null);
      setAssayAddonTask(null);
      setPendingAssayAddonPages([]);
      assayAddonSubmittedPagesRef.current = [];
      setToast('Bioactivity page detection started.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit bioactivity detection');
    } finally {
      setIsAssaySubmitting(false);
    }
  };

  const structureDetectionActive = Boolean(
    isStructureSubmitting ||
      (autoDetectTask && (autoDetectTask.status === 'running' || autoDetectTask.status === 'pending')),
  );

  const assayDetectionActive = Boolean(
    isAssaySubmitting ||
      isAssayWaitingForStructures ||
      (assayTask && (assayTask.status === 'running' || assayTask.status === 'pending')),
  );

  const handleStartFullPipeline = async () => {
    resetNotifications();
    if (!pdfInfo) { setError('Upload a PDF first.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for page previews to finish loading.'); return; }
    if (!(await flushPendingEdits())) return;
    try {
      setIsFullPipelineSubmitting(true);
      structurePagesAutoAdvancedRef.current = false;
      assayPagesAutoAdvancedRef.current = false;
      const request: FullPipelineRequest = {
        pdf_id: pdfInfo.pdf_id,
        structure_filter_strictness: structureFilterStrictness,
        assemble_markush: assembleMarkush && assemblyAvailable,
      };
      const taskStatus = await queueFullPipelineTask(request);
      clearJobResultState();
      setSelectedJobId(taskStatus.task_id);
      suppressAutoStepUntilNextJobRef.current = false;
      setFullPipelineTask(taskStatus);
      setToast('Extraction started. Structures, result tables and source evidence will appear as they are processed.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit full pipeline task');
    } finally {
      setIsFullPipelineSubmitting(false);
    }
  };

  const fullPipelineActive = Boolean(
    isFullPipelineSubmitting ||
      (fullPipelineTask && (fullPipelineTask.status === 'running' || fullPipelineTask.status === 'pending')),
  );

  const automaticExtractionActive = Boolean(
    isStructureSubmitting ||
      isAssaySubmitting ||
      isAssayWaitingForStructures ||
      isFullPipelineSubmitting ||
      (autoDetectTask && (autoDetectTask.status === 'running' || autoDetectTask.status === 'pending')) ||
      (structureTask && (structureTask.status === 'running' || structureTask.status === 'pending')) ||
      (assayTask && (assayTask.status === 'running' || assayTask.status === 'pending')) ||
      (structureAddonTask && (structureAddonTask.status === 'running' || structureAddonTask.status === 'pending')) ||
      (assayAddonTask && (assayAddonTask.status === 'running' || assayAddonTask.status === 'pending')) ||
      (fullPipelineTask && (fullPipelineTask.status === 'running' || fullPipelineTask.status === 'pending')),
  );
  const automaticActionDisabled = !automaticExtractionActive && !pdfReadyForDetection;

  const activeJobCount = jobsInfo
    ? jobsInfo.running_count + jobsInfo.pending_count
    : automaticExtractionActive
    ? 1
    : 0;

  React.useEffect(() => {
    if (!jobsOpen && !automaticExtractionActive) return undefined;
    void loadJobs({ silent: !jobsOpen });
    if (!automaticExtractionActive && activeJobCount === 0) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      if (!document.hidden) void loadJobs({ silent: true });
    }, jobsOpen ? 2500 : 5000);
    return () => window.clearInterval(timer);
  }, [activeJobCount, automaticExtractionActive, jobsOpen, loadJobs]);

  const handleCancelAutomaticExtraction = React.useCallback(async () => {
    if (isCanceling) return;
    setIsCanceling(true);
    setError(null);
    setToast(null);
    const activeTasks = Array.from(new Map([
      fullPipelineTask, autoDetectTask, structureAddonTask, assayAddonTask, structureTask, assayTask,
    ].filter(isCancelableTask).map((task) => [task!.task_id, task!])).values());
    const outcomes = await Promise.all(activeTasks.map(async (task) => {
      try {
        return { taskId: task.task_id, updated: await cancelTask(task.task_id), error: '' };
      } catch (err) {
        return { taskId: task.task_id, updated: null, error: err instanceof Error ? err.message : 'Cancellation failed' };
      }
    }));
    const canceled = new Map(outcomes.filter((outcome) => outcome.updated?.status === 'canceled')
      .map((outcome) => [outcome.taskId, outcome.updated!]));
    const applyCancellation = (current: TaskStatus | null) => current ? canceled.get(current.task_id) ?? current : current;
    automaticExtractionRunRef.current += 1;
    structurePagesAutoAdvancedRef.current = false;
    assayPagesAutoAdvancedRef.current = false;
    pendingAssayRequestRef.current = null;
    setPendingStructureAddonPages([]);
    setPendingAssayAddonPages([]);
    structureAddonSubmittedPagesRef.current = [];
    assayAddonSubmittedPagesRef.current = [];
    setIsStructureSubmitting(false);
    setIsAssaySubmitting(false);
    setIsFullPipelineSubmitting(false);
    setIsAssayWaitingForStructures(false);
    // Failed cancellation requests remain in state so polling and retry continue.
    setFullPipelineTask(applyCancellation);
    setAutoDetectTask(applyCancellation);
    setStructureAddonTask(applyCancellation);
    setAssayAddonTask(applyCancellation);
    setStructureTask(applyCancellation);
    setAssayTask(applyCancellation);
    const failures = outcomes.filter((outcome) => outcome.error);
    if (failures.length) {
      setError(`Could not cancel ${failures.length} job${failures.length === 1 ? '' : 's'}. Still tracking their progress.\n${failures.map((outcome) => outcome.error).join('\n')}`);
    } else {
      setToast(canceled.size ? 'Cancellation accepted.' : 'Jobs finished before cancellation. Checking their results.');
    }
    setIsCanceling(false);
    void loadJobs({ force: true });
  }, [assayAddonTask, assayTask, autoDetectTask, fullPipelineTask, isCanceling, loadJobs, structureAddonTask, structureTask]);


  const scheduleAutoSave = React.useCallback(() => {
    if (!editableTaskId) return;
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
    }
    setSaveStatus((prev) => (prev === 'saving' ? 'saving' : 'pending'));
    autoSaveTimerRef.current = window.setTimeout(() => {
      autoSaveTimerRef.current = null;
      void performAutoSave();
    }, 800);
  }, [editableTaskId, performAutoSave]);

  const handleCellChange = (rowIndex: number, column: string, value: string) => {
    setEditedStructures((prev) => {
      const next = prev.map((row, idx) => (idx === rowIndex ? { ...row, [column]: value } : row));
      return next;
    });

    // COMPOUND_ID saves explicitly (Enter/blur), not via debounce
    if (column !== 'COMPOUND_ID') {
      scheduleAutoSave();
    }
  };

  React.useEffect(() => {
    if (!editableTaskId || (saveStatus !== 'saved' && saveStatus !== 'idle')) return;
    let canceled = false;
    void api.get(`/tasks/${editableTaskId}/sar`).then((response) => { if (!canceled) setSarReport(response.data?.coverage && Array.isArray(response.data?.gaps) ? response.data : null); }).catch(() => { if (!canceled) setSarReport(null); });
    return () => { canceled = true; };
  }, [editableTaskId, saveStatus]);

  const handleAddReviewData = (values: { name: string; unit: string; target: string; sourceKind: string }) => {
    if (!editableTaskId) return;
    if (reviewAddKind === 'compound') {
      const index = editedStructuresRef.current.length;
      const record: StructureRecord = { RECORD_ID: crypto.randomUUID(), COMPOUND_ID: values.name, SMILES: '', PAGE_NUM: documentPage, SOURCE: 'manual', ID_SOURCE: 'manual', USER_LOCKED_FIELDS: JSON.stringify(['COMPOUND_ID']) };
      const updated = [...editedStructuresRef.current, record];
      editedStructuresRef.current = updated; setEditedStructures(updated); setStructures(updated);
      setReviewPage(Math.max(1, Math.ceil(updated.length / reviewPageSize)));
      setActiveReviewRowIndex(index); setEditorState({ open: true, rowIndex: index, smiles: '', molblock: '' });
    } else {
      let column = values.target ? `${values.target} · ${values.name}` : values.name;
      if (values.unit) column += ` (${values.unit.replace('uM', 'µM').replace('μ', 'µ')})`;
      if (values.sourceKind === 'predicted' || values.sourceKind === 'calculated') column += ` · ${values.sourceKind}`;
      if (assayColumnNames.includes(column)) { setError('This measurement column already exists. Edit its cells directly.'); return; }
      const updated = editedStructuresRef.current.map((record) => {
        let removed: string[] = []; try { removed = JSON.parse(String(record.USER_REMOVED_MEASUREMENTS || '[]')); } catch { /* older rows */ }
        return { ...record, [column]: '', USER_MEASUREMENT_EDITS: JSON.stringify({ ...measurementEdits(record), [column]: '' }), USER_REMOVED_MEASUREMENTS: JSON.stringify(removed.filter((name) => name !== column)) };
      });
      editedStructuresRef.current = updated; setEditedStructures(updated);
    }
    setReviewAddKind(null); scheduleAutoSave();
  };

  const removeMeasurement = (column: string) => {
    const updated = editedStructuresRef.current.map((record) => {
      const next = { ...record }; delete next[column];
      const edits = measurementEdits(record); delete edits[column];
      let removed: string[] = []; try { removed = JSON.parse(String(record.USER_REMOVED_MEASUREMENTS || '[]')); } catch { /* older rows */ }
      return { ...next, USER_MEASUREMENT_EDITS: JSON.stringify(edits), USER_REMOVED_MEASUREMENTS: JSON.stringify([...new Set([...removed, column])]) };
    });
    editedStructuresRef.current = updated; setEditedStructures(updated); scheduleAutoSave();
  };

  const restorePreviousReview = async () => {
    if (!editableTaskId || !(await flushPendingEdits())) return;
    try {
      const result = await undoStructureEdit(editableTaskId);
      const records = result.records.map((row) => ({ ...row }));
      editedStructuresRef.current = records; setEditedStructures(records); setStructures(records); setSaveStatus('saved');
      setToast('Previous saved version restored.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Restore failed'); }
  };

  const startSarRecovery = async () => {
    if (!editableTaskId || !(await flushPendingEdits())) return;
    try {
      const result = await api.post<TaskStatus>(`/tasks/${editableTaskId}/sar/recover`, { max_actions: 8 });
      setStructureTask(result.data); setToast('Recovery started as a new task. Your current records and edits are preserved.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Recovery failed to start'); }
  };

  const handleMeasurementSave = (rowIndex: number, record: StructureRecord, column: string, value: string) => {
    if (!activeSaveTaskRef.current) return;
    const current = editedStructuresRef.current;
    const index = rowIndex < current.length ? rowIndex : current.length;
    const original = current[index] ?? record;
    const changed: StructureRecord = { ...original, [column]: value,
      USER_MEASUREMENT_EDITS: JSON.stringify({ ...measurementEdits(original), [column]: value }),
      MEASUREMENT_REVIEW_SOURCE: 'user', MEASUREMENT_REVIEWED_AT: new Date().toISOString() };
    const updated = index === current.length ? [...current, changed] : current.map((row, i) => i === index ? changed : row);
    editedStructuresRef.current = updated;
    setEditedStructures(updated);
    scheduleAutoSave();
  };

  const handleCompoundIdSave = (rowIndex: number, value: string) => {
    const current = editedStructuresRef.current[rowIndex];
    if (!current || formatCellValue(current.COMPOUND_ID) === value) return;
      const updatedStructures = editedStructuresRef.current.map((row, idx) =>
      idx === rowIndex ? { ...row, COMPOUND_ID: value, ID_SOURCE: 'manual', USER_LOCKED_FIELDS: JSON.stringify(['COMPOUND_ID']) } : row
    );
    setEditedStructures(updatedStructures);
    editedStructuresRef.current = updatedStructures;

      if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }

      setSaveStatus('saving');

      void performAutoSave();
  };

  const handleDeleteStructureRow = (rowIndex: number) => {
    if (rowIndex < 0 || rowIndex >= editedStructuresRef.current.length) return;
    if (!window.confirm('Delete this structure record?')) return;

    pdfLocateSequenceRef.current += 1;
    setActiveReviewRowIndex(null);
    setPdfTarget(null);
    const updatedStructures = editedStructuresRef.current.filter((_, idx) => idx !== rowIndex);
    setEditedStructures(updatedStructures);
    editedStructuresRef.current = updatedStructures;
    setStructures((prev) => prev.filter((_, idx) => idx !== rowIndex));

    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }

    setSaveStatus('saving');
    void performAutoSave();
  };


  const locateRowInPdf = async (row: { index: number; record: Record<string, unknown>; normalizedPrimaryPage?: string | number }) => {
    const page = getRecordPrimaryPageNumber(row.record as StructureRecord);
    setActiveReviewRowIndex(row.index);
    if (!page || !pdfInfo?.pdf_id) return;
    const sequence = ++pdfLocateSequenceRef.current;
    setDocumentPage(page);
    setPdfTarget({ page, requestId: sequence });
    const boxPath = typeof row.record.MARKUSH_SCAFFOLD_BOX_COORDS_FILE === 'string' && row.record.STRUCTURE_TYPE === 'markush_assembled' ? row.record.MARKUSH_SCAFFOLD_BOX_COORDS_FILE : typeof row.record.BOX_COORDS_FILE === 'string' ? row.record.BOX_COORDS_FILE : '';
    if (!boxPath) return;
    try {
      const artifact = await fetchBoxArtifact(boxPath);
      if (sequence !== pdfLocateSequenceRef.current) return;
      const box = JSON.parse(atob(artifact.content));
      const b = box.box;
      const width = Number(box.page_width), height = Number(box.page_height);
      if (!Array.isArray(b) || b.length < 4 || width <= 0 || height <= 0) return;
      const [x1, y1, x2, y2] = b.map(Number);
      if (![y1, x1, y2, x2, width, height].every(Number.isFinite) || y2 <= y1 || x2 <= x1) return;
      const x = Math.max(0, Math.min(1, x1 / width)), y = Math.max(0, Math.min(1, y1 / height));
      setPdfTarget({ page, requestId: sequence, rect: { x, y, w: Math.min(1 - x, (x2 - x1) / width), h: Math.min(1 - y, (y2 - y1) / height) } });
    } catch { /* Page navigation remains usable if a source box is unavailable. */ }
  };

  navigateReviewRowRef.current = (row) => { void locateRowInPdf(row); };

  const recognizeManualRegion = async (page: number, rect: NonNullable<PdfTarget['rect']>, purpose: 'structure' | 'table' | 'fragment' = 'structure', replaceSelected = false) => {
    if (!pdfInfo?.pdf_id) throw new Error('Open a PDF first');
    const taskId = activeSaveTaskRef.current;
    const selectedIndex = replaceSelected ? activeReviewRowIndex : null;
    const selectedRecord = selectedIndex === null ? null : editedStructuresRef.current[selectedIndex];
    if (purpose === 'table') {
      const response = await api.post<{ records: StructureRecord[]; gaps: string[] }>(`/pdfs/${pdfInfo.pdf_id}/regions/measurements`, { page, ...rect });
      if (!taskId || taskId !== activeSaveTaskRef.current) throw new Error('The active task changed. Select the region again.');
      let updated = [...editedStructuresRef.current];
      for (const incoming of response.data.records) {
        incoming.RECORD_ID ||= crypto.randomUUID();
        const matches = updated.map((row, index) => ({ row, index })).filter(({ row }) => row.COMPOUND_ID === incoming.COMPOUND_ID);
        if (matches.length === 1 && matches[0].row.SMILES && matches[0].row.LOCAL_EVIDENCE_STATUS !== 'needs_review' && matches[0].row.MEASUREMENT_LINK_STATUS !== 'identifier_unverified' && !['markush', 'fragment', 'text_substituent'].includes(String(matches[0].row.STRUCTURE_TYPE || ''))) {
          const { row, index } = matches[0];
          let existing: unknown[] = []; try { existing = JSON.parse(String(row.BIOACTIVITY_OBSERVATIONS || '[]')); } catch { /* older rows */ }
          const observations = JSON.parse(String(incoming.BIOACTIVITY_OBSERVATIONS || '[]'));
          updated[index] = { ...row, BIOACTIVITY_OBSERVATIONS: JSON.stringify([...existing, ...observations]), RESULT_SOURCE_PAGES: incoming.RESULT_SOURCE_PAGES };
        } else updated.push(incoming);
      }
      editedStructuresRef.current = updated; setEditedStructures(updated); setStructures(updated); scheduleAutoSave();
      setToast(`Added ${response.data.records.length} result records. Check the editable measurement cells.`);
      return;
    }
    const response = await api.post<StructureRecord>(`/pdfs/${pdfInfo.pdf_id}/regions/recognize`, { page, ...rect });
    if (!taskId || taskId !== activeSaveTaskRef.current) throw new Error('The active task changed. Select the region again.');
    const identified: StructureRecord = { ...response.data, RECORD_ID: response.data.RECORD_ID || crypto.randomUUID() };
    const record: StructureRecord = purpose === 'fragment' ? { ...identified, STRUCTURE_TYPE: 'fragment', ASSEMBLY_STATUS: 'pending' } : identified;
    const index = selectedRecord && selectedIndex !== null ? selectedIndex : editedStructuresRef.current.length;
    if (selectedRecord && record.COMPOUND_ID && selectedRecord.COMPOUND_ID && record.COMPOUND_ID !== selectedRecord.COMPOUND_ID) throw new Error('The detected compound ID differs from the selected row. Add it as a new record and check the source.');
    const replacement = selectedRecord ? { ...selectedRecord, ...record, COMPOUND_ID: selectedRecord.COMPOUND_ID, RECORD_ID: selectedRecord.RECORD_ID, PREVIOUS_STRUCTURE_EVIDENCE: JSON.stringify({ SMILES: selectedRecord.SMILES, SEGMENT_FILE: selectedRecord.SEGMENT_FILE, BOX_COORDS_FILE: selectedRecord.BOX_COORDS_FILE }), BIOACTIVITY_OBSERVATIONS: selectedRecord.BIOACTIVITY_OBSERVATIONS, USER_MEASUREMENT_EDITS: selectedRecord.USER_MEASUREMENT_EDITS } : record;
    const updated = selectedRecord ? editedStructuresRef.current.map((row, i) => i === index ? replacement : row) : [...editedStructuresRef.current, replacement];
    editedStructuresRef.current = updated;
    setEditedStructures(updated);
    setStructures(updated);
    setActiveReviewRowIndex(index);
    setPdfTarget({ page, rect, requestId: ++pdfLocateSequenceRef.current });
    setEditorState({ open: true, rowIndex: index, smiles: String(record.SMILES || ''), molblock: getMolblockValue(record) });
    scheduleAutoSave();
  };

  const handleOpenStructureEditor = (rowIndex: number) => {
    const record = editedStructures[rowIndex];
    const smiles = String(record.SMILES || record.NAME_STRUCTURE_CANDIDATE || '');
    const molblock = getMolblockValue(record);
    setEditorState({ open: true, rowIndex, smiles, molblock });
  };

  const handleStructureEditorCancel = () => {
    setEditorState({ open: false, rowIndex: null, smiles: '', molblock: '' });
  };

  const handleStructureEditorSave = ({
    smiles,
    molblock,
    image,
  }: {
    smiles: string;
    molblock?: string;
    image?: string;
  }) => {
    if (editorState.rowIndex === null) {
      setEditorState({ open: false, rowIndex: null, smiles: '', molblock: '' });
      return;
    }
    const molblockValue = typeof molblock === 'string' ? molblock : '';
    setEditedStructures((prev) =>
      prev.map((row, idx) => {
        if (idx !== editorState.rowIndex) return row;
        const updated: StructureRecord = { ...row, SMILES: smiles, MOLBLOCK: molblockValue, STRUCTURE_REVIEW_SOURCE: 'user', USER_LOCKED_FIELDS: JSON.stringify(['SMILES', 'MOLBLOCK']) };
        return updated;
      }),
    );
    if (image) {
      setImageCache((prev) => ({ ...prev, [image]: image }));
    }
    const previewKey = getStructurePreviewKey(smiles, molblockValue);
    if (image && previewKey) {
      setStructurePreviewCache((prev) => {
        const next = { ...prev };
        if (!Object.prototype.hasOwnProperty.call(prev, previewKey)) {
          next[previewKey] = image;
        }
        return next;
      });
    }
    scheduleAutoSave();
    setEditorState({ open: false, rowIndex: null, smiles: '', molblock: '' });
  };

  const handleInlineStructureSave = ({
    smiles,
    molblock,
    image,
  }: {
    smiles: string;
    molblock?: string;
    image?: string;
  }) => {
    if (modalRowIndex === null) {
      return;
    }
    const molblockValue = typeof molblock === 'string' ? molblock : '';
    setEditedStructures((prev) =>
      prev.map((row, idx) => {
        if (idx !== modalRowIndex) return row;
        const updated: StructureRecord = { ...row, SMILES: smiles, MOLBLOCK: molblockValue, STRUCTURE_REVIEW_SOURCE: 'user', USER_LOCKED_FIELDS: JSON.stringify(['SMILES', 'MOLBLOCK']) };
        return updated;
      }),
    );
    if (image) {
      setImageCache((prev) => ({ ...prev, [image]: image }));
    }
    const previewKey = getStructurePreviewKey(smiles, molblockValue);
    if (image && previewKey) {
      setStructurePreviewCache((prev) => {
        const next = { ...prev };
        if (!Object.prototype.hasOwnProperty.call(prev, previewKey)) {
          next[previewKey] = image;
        }
        return next;
      });
    }
    scheduleAutoSave();
  };

  const openArtifact = async (source: unknown, label?: string, options?: { rowIndex?: number | null }) => {
    const request = ++previewRequestRef.current;
    const epoch = artifactEpochRef.current;
    const current = () => request === previewRequestRef.current && epoch === artifactEpochRef.current;
    setModalBox(null);
    setOriginalImageSize(null);
    if (typeof source !== 'string' || !source) return;

    if (options?.rowIndex !== null && options?.rowIndex !== undefined) {
      const record = editedStructures[options.rowIndex];
      const isMarkushAssembled =
        (record?.STRUCTURE_TYPE as string | undefined)?.trim() === 'markush_assembled';
      const isScaffoldClick = isMarkushAssembled && (label || '').startsWith('Scaffold');
      const boxCoordsFile = isScaffoldClick
        ? (record?.MARKUSH_SCAFFOLD_BOX_COORDS_FILE as string | undefined)
        : (record?.BOX_COORDS_FILE as string | undefined);
      if (boxCoordsFile) {
        try {
          const artifact = await fetchBoxArtifact(boxCoordsFile);
          if (!current()) return;
          const coordsData = JSON.parse(atob(artifact.content));
          if (coordsData && coordsData.box) {
            setModalBox(coordsData.box);
          }
        } catch (err) {
          console.error('Failed to fetch or parse box coordinates', err);
        }
      }
    }

    const showDataImage = (dataUri: string, pathLabel?: string) => {
      if (!current()) return;
      const img = new Image();
      img.onload = () => {
        if (!current()) return;
        setOriginalImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = dataUri;

      setModalArtifact({
        path: pathLabel ?? 'Preview',
        mime: 'image/png',
        data: dataUri,
        rowIndex: options?.rowIndex ?? null,
        kind: 'image',
      });
    };

    if (isDataImage(source)) {
      showDataImage(source, label ?? 'Preview');
      return;
    }

    const markdownMatch = source.match(markdownImageRegex);
    if (markdownMatch) {
      showDataImage(markdownMatch[1], label ?? 'Preview');
      return;
    }

    if (imageCache[source]) {
      showDataImage(imageCache[source], label ?? source);
      return;
    }

    setModalArtifact({
      path: label ?? source,
      mime: 'loading',
      data: '',
      rowIndex: options?.rowIndex ?? null,
      kind: 'image',
    });

    try {
      const artifact = await fetchArtifact(source);
      if (!current()) return;
      const dataUri = `data:${artifact.mime_type};base64,${artifact.content}`;
      setImageCache((prev) => ({ ...prev, [source]: dataUri }));
      showDataImage(dataUri, source);
    } catch (err) {
      if (!current()) return;
      setModalArtifact(null);
      setError(err instanceof Error ? err.message : 'Could not load image');
    }
  };

  const closeModal = () => {
    previewRequestRef.current += 1;
    setModalArtifact(null);
    setModalBox(null);
    setOriginalImageSize(null);
    setBoxStyle({});
  };

  const downloadResults = async (taskId: string | undefined) => {
    if (!taskId || isDownloading) return;
    setIsDownloading(true);
    try {
      if (autoSaveTimerRef.current) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      if (editableTaskId && !(await performAutoSave())) return;
      const blob = await fetchTaskDownload(taskId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'biocheminsight_results.zip';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed. Try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  const downloadStructuresCsv = () => { void downloadResults(editableTaskId); };
  const downloadAssayCsv = () => {
    void downloadResults(fullPipelineTask?.status === 'completed' ? fullPipelineTask.task_id : assayTask?.task_id);
  };

  const clearStructureSelection = () => {
    reportStructureSelectionEdit([]);
    setStructureSelection(new Set<number>());
    setStructurePagesInput('');
    lastStructurePageRef.current = null;
  };

  const clearAssaySelection = () => {
    reportAssaySelectionEdit([]);
    setAssaySelection(new Set<number>());
    setAssayPagesInput('');
    lastAssayPageRef.current = null;
  };

  React.useEffect(() => {
    if (initializedFromUrl.current) return;
    initializedFromUrl.current = true;
    const params = new URLSearchParams(window.location.search);
    const pdfParam = params.get('pdf');
    let restoredAssayPagesParam = '';
    let restoredAssayNames: string[] = [];

    const stepParam = params.get('step');
    if (stepParam) {
      const parsedStep = Number(stepParam);
      if (parsedStep === 2 || parsedStep === 3 || parsedStep === 4) {
        requestedStepRef.current = parsedStep as StepId;
      }
    } else {
      const legacyMode = params.get('mode');
      if (legacyMode === 'assay') {
        requestedStepRef.current = 3;
      }
    }

    const structPagesParam = params.get('structPages');
    if (structPagesParam) {
      setAutoDetectStructurePages(false);
      setStructurePagesInput(structPagesParam);
      const parsed = parsePagesInput(structPagesParam);
      setStructureSelection(new Set<number>(parsed));
      lastStructurePageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
    }

    const assayPagesParam = params.get('assayPages');
    if (assayPagesParam) {
      restoredAssayPagesParam = assayPagesParam;
      setAutoDetectAssayPages(false);
      setAssayPagesInput(assayPagesParam);
      const parsed = parsePagesInput(assayPagesParam);
      setAssaySelection(new Set<number>(parsed));
      lastAssayPageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
    }

    const assayNamesParam = params.get('assayNames');
    if (assayNamesParam) {
      restoredAssayNames = parseAssayNames(assayNamesParam);
      setAutoDetectAssayNames(false);
      setAssayNames(restoredAssayNames);
    }

    if (isUsableId(pdfParam)) {
      fetchPdfInfo(pdfParam)
        .then((info: UploadPDFResponse) => {
          setPdfInfo(info);

        })
        .catch(() => setError('Could not restore the PDF. Upload it again.'));
    }

    const structureTaskId = params.get('structureTask');
    if (isUsableId(structureTaskId)) {
      fetchTask(structureTaskId)
        .then((task) => {
          setStructureTask(task);
          setStructureFilterStrictness(coerceStructureFilterStrictness(task.params?.structure_filter_strictness));
          setAutoDetectStructurePages(coerceAutoDetectionFlag(task.params?.auto_detect_pages));
          applyDetectedStructurePages(task.params?.detected_pages ?? task.params?.pages);
          if (task.status === 'completed') {
            return fetchTaskStructures(structureTaskId).then((results) => {
              const nextRecords = results.records.map((record) => ({ ...record }));
              const nextFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
              setStructures(results.records);
              editedStructuresRef.current = nextRecords;
              setEditedStructures(nextRecords);
              setLoadedStructureTaskId(structureTaskId);
              setFilteredStructures(nextFilteredRecords);
              setSaveStatus(results.records.length ? 'saved' : 'idle');
              if (results.records.length > 0 && !suppressAutoStepUntilNextJobRef.current) {
                autoAdvanceRef.current = true;
                setCurrentStep((prev) => (prev < 4 ? 4 : prev));
              }
            });
          }
          return undefined;
        })
        .catch(() => setError('Could not restore the structure task.'));
    }

    const assayTaskId = params.get('assayTask');
    if (isPendingAssayTaskId(assayTaskId)) {
      if (isUsableId(pdfParam)) {
        const hasProvidedAssayPages = restoredAssayPagesParam.trim().length > 0;
        const hasProvidedAssayNames = restoredAssayNames.length > 0;
        const pendingRequest: AssayTaskRequest = {
          pdf_id: pdfParam,
          pages: hasProvidedAssayPages ? restoredAssayPagesParam : undefined,
          assay_names: hasProvidedAssayNames ? restoredAssayNames : [],
          auto_detect_pages: !hasProvidedAssayPages,
          auto_detect_assay_names: !hasProvidedAssayNames,
        };
        pendingAssayRequestRef.current = pendingRequest;
        setIsAssayWaitingForStructures(true);
        const now = new Date().toISOString();
        setAssayTask({
          task_id: assayTaskId,
          type: 'assay',
          status: 'pending',
          progress: 0,
          message: 'Waiting for structure extraction…',
          pdf_id: pdfParam,
          params: { ...pendingRequest },
          created_at: now,
          updated_at: now,
        });
      }
    } else if (isUsableId(assayTaskId)) {
      fetchTask(assayTaskId)
        .then((task) => {
          setAssayTask(task);
          setAutoDetectAssayPages(coerceAutoDetectionFlag(task.params?.auto_detect_pages));
          setAutoDetectAssayNames(coerceAutoDetectionFlag(task.params?.auto_detect_assay_names));
          applyDetectedAssayPages(task.params?.detected_pages ?? task.params?.pages);
          applyDetectedAssayNames(task.params?.detected_assay_names);
          if (task.status === 'completed') {
            const linkedStructureTaskId =
              typeof task.params?.structure_task_id === 'string' ? task.params.structure_task_id : '';
            const structurePromise = isUsableId(linkedStructureTaskId)
              ? loadExplicitStructureSource(linkedStructureTaskId).catch((err) => {
                  console.warn('Failed to load linked structure job', err);
                })
              : Promise.resolve();
            return structurePromise.then(() => fetchTaskAssays(assayTaskId)).then((results) => {
              setAssayRecords(results.records);
              if (results.records.length > 0 && !suppressAutoStepUntilNextJobRef.current) {
                setCurrentStep((prev) => (prev < 4 ? 4 : prev));
              }
            });
          }
          return undefined;
        })
        .catch(() => setError('Could not restore the bioactivity task.'));
    }

    const structureAddonTaskId = params.get('structureAddonTask');
    if (isUsableId(structureAddonTaskId)) {
      fetchTask(structureAddonTaskId)
        .then((task) => {
          setStructureAddonTask(task);
          structureAddonSubmittedPagesRef.current = getTaskSelectedPages(task);
        })
        .catch(() => setError('Could not restore the structure add-on task.'));
    }

    const assayAddonTaskId = params.get('assayAddonTask');
    if (isUsableId(assayAddonTaskId)) {
      fetchTask(assayAddonTaskId)
        .then((task) => {
          setAssayAddonTask(task);
          assayAddonSubmittedPagesRef.current = getTaskSelectedPages(task);
        })
        .catch(() => setError('Could not restore the bioactivity add-on task.'));
    }

    const autoDetectTaskId = params.get('autoDetectTask');
    if (isUsableId(autoDetectTaskId)) {
      fetchTask(autoDetectTaskId)
        .then((task) => {
          setAutoDetectTask(task);
          setAutoDetectStructurePages(coerceAutoDetectionFlag(task.params?.detect_structure_pages));
          setAutoDetectAssayPages(coerceAutoDetectionFlag(task.params?.detect_assay_pages));
          setAutoDetectAssayNames(coerceAutoDetectionFlag(task.params?.detect_assay_names));
          if (Object.prototype.hasOwnProperty.call(task.params ?? {}, 'detected_structure_pages')) {
            applyPlannedStructurePages(task.params?.detected_structure_pages);
          }
          if (Object.prototype.hasOwnProperty.call(task.params ?? {}, 'detected_assay_pages')) {
            applyPlannedAssayPages(task.params?.detected_assay_pages);
          }
          applyDetectedAssayNames(task.params?.detected_assay_names);
        })
        .catch(() => setError('Could not restore the detection task.'));
    }

    const fullPipelineTaskId = params.get('fullPipelineTask');
    if (isUsableId(fullPipelineTaskId)) {
      fetchTask(fullPipelineTaskId)
        .then((task) => {
          setFullPipelineTask(task);
          if (task.status === 'failed') setError(task.error || 'Full pipeline failed.');
          const taskParams = task.params as Record<string, unknown>;
          if (Object.prototype.hasOwnProperty.call(taskParams ?? {}, 'detected_structure_pages')) {
            applyPlannedStructurePages(taskParams?.detected_structure_pages);
          }
          if (Object.prototype.hasOwnProperty.call(taskParams ?? {}, 'detected_assay_pages')) {
            applyPlannedAssayPages(taskParams?.detected_assay_pages);
          }
          applyDetectedAssayNames(taskParams?.detected_assay_names);
          if (task.status === 'completed') {
            return Promise.all([
              fetchTaskStructures(fullPipelineTaskId),
              fetchTaskAssays(fullPipelineTaskId),
            ]).then(([structureResults, assayResults]) => {
              const nextStructures = structureResults.records.map((record) => ({ ...record }));
              setStructures(structureResults.records);
              editedStructuresRef.current = nextStructures;
              setEditedStructures(nextStructures);
              setLoadedStructureTaskId(fullPipelineTaskId);
              setFilteredStructures(structureResults.filtered_records ?? []);
              setAssayRecords(assayResults.records);
              setSaveStatus('saved');
            });
          }
          return undefined;
        })
        .catch(() => setError('Could not restore the full pipeline task.'));
    }
  }, [
    applyDetectedAssayNames,
    applyDetectedAssayPages,
    applyDetectedStructurePages,
    applyPlannedAssayPages,
    applyPlannedStructurePages,
    coerceAutoDetectionFlag,
    coerceStructureFilterStrictness,
    loadExplicitStructureSource,
  ]);

  React.useEffect(() => {
    const params = new URLSearchParams();

    if (isUsableId(pdfInfo?.pdf_id)) {
      params.set('pdf', pdfInfo.pdf_id);
    }
    if (structurePagesInput) {
      params.set('structPages', structurePagesInput);
    }
    if (assayPagesInput) {
      params.set('assayPages', assayPagesInput);
    }
    if (assayNames.length) {
      params.set('assayNames', assayNames.join(','));
    }
    if (currentStep > 1) {
      params.set('step', String(currentStep));
    }
    if (isUsableId(structureTask?.task_id)) {
      params.set('structureTask', structureTask.task_id);
    }
    if (isUsableId(assayTask?.task_id)) {
      params.set('assayTask', assayTask.task_id);
    }
    if (isUsableId(structureAddonTask?.task_id)) {
      params.set('structureAddonTask', structureAddonTask.task_id);
    }
    if (isUsableId(assayAddonTask?.task_id)) {
      params.set('assayAddonTask', assayAddonTask.task_id);
    }
    if (isUsableId(autoDetectTask?.task_id)) {
      params.set('autoDetectTask', autoDetectTask.task_id);
    }
    if (isUsableId(fullPipelineTask?.task_id)) {
      params.set('fullPipelineTask', fullPipelineTask.task_id);
    }

    const query = params.toString();
    const newUrl = query ? `${window.location.pathname}?${query}` : window.location.pathname;

    if (lastUrlRef.current !== newUrl) {
      window.history.replaceState(null, '', newUrl);
      lastUrlRef.current = newUrl;
    }
  }, [
    pdfInfo,
    structurePagesInput,
    assayPagesInput,
    assayNames,
    currentStep,
    structureTask?.task_id,
    assayTask?.task_id,
    structureAddonTask?.task_id,
    assayAddonTask?.task_id,
    autoDetectTask?.task_id,
    fullPipelineTask?.task_id,
  ]);

  const imageModalOpen = modalArtifact !== null;
  React.useEffect(() => {
    if (!imageModalOpen) {
      return;
    }
    const previousFocus = document.activeElement as HTMLElement | null;
    document.body.classList.add('no-scroll');
    imageDialogRef.current?.querySelector<HTMLButtonElement>('.image-dialog__close')?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); closeModal(); }
      if (event.key !== 'Tab') return;
      const controls = Array.from(imageDialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]',
      ) ?? []).filter((element) => element.getClientRects().length > 0);
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first || !imageDialogRef.current?.contains(document.activeElement))) {
        event.preventDefault(); last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first?.focus();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => {
      document.body.classList.remove('no-scroll');
      window.removeEventListener('keydown', handleKey);
      previousFocus?.focus();
    };
  }, [imageModalOpen]);

  React.useEffect(() => {
    const handleResize = () => {
      const table = document.querySelector('.review-table');
      if (table && table instanceof HTMLElement) {
          table.style.tableLayout = 'auto';
          setTimeout(() => {
          if (table instanceof HTMLElement) {
            table.style.tableLayout = 'auto';
          }
        }, 100);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  React.useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 400);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = React.useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const renderReviewPagination = (placement: 'top' | 'bottom') => {
    const firstRow = reviewTotalRows === 0 ? 0 : reviewStartIndex + 1;
    const lastRow = reviewEndIndex;
    const isFirstPage = boundedReviewPage <= 1;
    const isLastPage = boundedReviewPage >= reviewTotalPages;
    return (
      <div aria-label="Result pages" className={`review-toolbar__pager review-pagination review-pagination--${placement}`}>
        <span className="review-pagination__summary" title={`${firstRow}–${lastRow} of ${reviewTotalRows} records`} aria-label={`${firstRow}–${lastRow} of ${reviewTotalRows} records`}>
          {reviewTotalRows ? boundedReviewPage : 0}/{reviewTotalRows ? reviewTotalPages : 0}
        </span>
        <div className="review-toolbar__control review-pagination__buttons">
          <button
            type="button"
            className="icon-btn"
            title="First page"
            aria-label="First page"
            disabled={isFirstPage}
            onClick={() => setReviewPage(1)}
          >
            <Icon name="first" className="icon-btn__svg" />
          </button>
          <button
            type="button"
            className="icon-btn"
            title="Previous page"
            aria-label="Previous page"
            disabled={isFirstPage}
            onClick={() => setReviewPage((value) => Math.max(1, value - 1))}
          >
            <Icon name="previous" className="icon-btn__svg" />
          </button>
          <button
            type="button"
            className="icon-btn"
            title="Next page"
            aria-label="Next page"
            disabled={isLastPage}
            onClick={() => setReviewPage((value) => Math.min(reviewTotalPages, value + 1))}
          >
            <Icon name="next" className="icon-btn__svg" />
          </button>
          <button
            type="button"
            className="icon-btn"
            title="Last page"
            aria-label="Last page"
            disabled={isLastPage}
            onClick={() => setReviewPage(reviewTotalPages)}
          >
            <Icon name="last" className="icon-btn__svg" />
          </button>
        </div>
      </div>
    );
  };

  const renderStageHeading = (mode: SelectionMode) => {
    const structure = mode === 'structure';
    const busy = isTaskInFlight(fullPipelineTask) || (structure ? isStructureSubmitting || isTaskInFlight(structureTask) : isAssaySubmitting || isTaskInFlight(assayTask));
    const count = structure ? structureSelection.size : assaySelection.size;
    const disabled = busy || !pdfInfo || (structure
      ? !autoDetectStructurePages && !structureSelection.size
      : (!autoDetectAssayPages && !assaySelection.size) || (!autoDetectAssayNames && !assayNames.length && !assayNameDraft.trim()) || isAssayWaitingForStructures);
    return <div className="stage-workspace__heading">
      <StepGlyph type={structure ? 'structures' : 'bioactivity'} className="stage-heading-icon" />
      <h2>{structure ? 'Structures' : 'Activity & ADME'}</h2>
      <span className="stage-workspace__document" title={pdfInfo?.filename}>{pdfInfo?.filename}</span>
      <span className="stage-workspace__selection" title="Selected pages">{count ? `${count} selected` : 'Auto'}</span>
      <button type="button" className="stage-options-toggle" aria-label="Page settings" title="Page settings" aria-expanded={stageOptionsOpen} onClick={() => setStageOptionsOpen((open) => !open)}><Icon name="filter" className="button-icon" /></button>
      <button type="button" className="stage-results-button" aria-label="Review results" title="Review results" disabled={!canViewResults} onClick={() => handleStepNavigation(4)}><StepGlyph type="review" className="button-icon" /></button>
      <button type="button" className="stage-run-button" disabled={disabled} aria-label={structure ? 'Extract structures' : 'Extract bioactivity'} title={busy ? 'Running' : structure ? 'Extract structures' : 'Extract bioactivity'} onClick={structure ? handleStartStructureExtraction : handleStartAssayExtraction}><Icon name="play" className="button-icon" /><span>{busy ? 'Running' : 'Run'}</span></button>
    </div>;
  };

  const renderStageContent = (mode: SelectionMode) => (
    <div className="stage-data">{renderStageHeading(mode)}<div className="selector-gallery">{renderPageGallery(mode)}</div></div>
  );

  return (
    <div className={`container container--workspace${currentStep === 1 ? ' container--upload' : ''}${currentStep === 2 || currentStep === 3 ? ' container--workflow' : ''}${currentStep === 4 ? ` container--review${reviewFocusMode ? ' container--review-focus' : ''}` : ''}`}>
      <header className="app-header">
        <div className="app-header__copy">
          <h1><img className="brand-logo" src="/brand-mark.svg" width="26" height="26" alt="" aria-hidden="true" />BioChemInsight <span>Workbench</span></h1>
          <p>
            Structures, activity, and source evidence — side by side.
          </p>
        </div>
      </header>
        <button
          className={`jobs-button${activeJobCount > 0 ? ' jobs-button--active' : ''}`}
          type="button"
          onClick={() => {
            setJobsOpen((open) => !open);
            void loadJobs({ force: true });
          }}
          aria-expanded={jobsOpen}
        >
          <span className="jobs-button__dot" aria-hidden="true" />
          Jobs
          {activeJobCount > 0 && <span className="jobs-button__count">{activeJobCount}</span>}
        </button>

      {jobsOpen && (
        <section className="jobs-panel" aria-label="Background jobs">
          <div className="jobs-panel__header">
            <div>
              <div className="jobs-panel__title-row">
                <h2>Jobs</h2>
                <button
                  type="button"
                  className={`verification-toggle${verificationMode ? ' verification-toggle--active' : ''}`}
                  aria-pressed={verificationMode}
                  onClick={() => setVerificationMode((current) => !current)}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                    <path d="m9 12 2 2 4-4" />
                    <path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Z" />
                  </svg>
                  Structure review
                </button>
              </div>
              <p>
                Running {jobsInfo?.running_count ?? 0}/{jobsInfo?.max_concurrent_tasks ?? '-'} slots · {jobsInfo?.pending_count ?? 0} pending · structure extraction concurrency {jobsInfo?.structure_task_concurrency ?? 1}
              </p>
            </div>
          </div>
          <button type="button" className="jobs-panel__close" aria-label="Close jobs" onClick={() => setJobsOpen(false)}>×</button>
          <div className="jobs-toolbar">
            <label className="jobs-search">
              <span>Search</span>
              <input
                type="search"
                value={jobsSearchInput}
                onChange={(event) => setJobsSearchInput(event.target.value)}
                placeholder="ID, type, message…"
              />
            </label>
            <label>
              <span>Status</span>
              <select
                value={jobsStatusFilter}
                onChange={(event) => {
                  setJobsPage(1);
                  setJobsStatusFilter(event.target.value);
                }}
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="canceled">Canceled</option>
              </select>
            </label>
            <label>
              <span>Type</span>
              <select
                value={jobsTypeFilter}
                onChange={(event) => {
                  setJobsPage(1);
                  setJobsTypeFilter(event.target.value);
                }}
              >
                <option value="all">All</option>
                {jobTypeOptions.map((type) => (
                  <option value={type} key={type}>
                    {formatTaskType(type)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Rows</span>
              <select
                value={jobsPageSize}
                onChange={(event) => {
                  setJobsPage(1);
                  setJobsPageSize(Number(event.target.value));
                }}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </label>
            <label>
              <span>From</span>
              <input
                type="date"
                value={jobsDateFrom}
                onChange={(event) => {
                  setJobsPage(1);
                  setJobsDateFrom(event.target.value);
                }}
                title="Defaults to the last 30 days. Clear to show older jobs."
              />
            </label>
            <label>
              <span>To</span>
              <input
                type="date"
                value={jobsDateTo}
                onChange={(event) => {
                  setJobsPage(1);
                  setJobsDateTo(event.target.value);
                }}
              />
            </label>
          </div>
          <div className="jobs-table-wrap">
            {jobsError ? (
              <div className="jobs-empty">
                Failed to load jobs: {jobsError}
                <button
                  type="button"
                  className="secondary jobs-retry"
                  onClick={() => {
                    setJobsError(null);
                    void loadJobs({ force: true });
                  }}
                >
                  Retry
                </button>
              </div>
            ) : (jobsInfo?.tasks ?? []).length === 0 ? (
              <div className="jobs-empty">No jobs yet. Jobs appear here once an extraction starts.</div>
            ) : (
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th>{renderJobsSortLabel('status', 'Status')}</th>
                    <th>{renderJobsSortLabel('type', 'Type')}</th>
                    <th>{renderJobsSortLabel('task_id', 'Job')}</th>
                    <th>{renderJobsSortLabel('progress', 'Progress')}</th>
                    <th>{renderJobsSortLabel('updated_at', 'Updated')}</th>
                    <th>{renderJobsSortLabel('created_at', 'Created')}</th>
                    <th>{renderJobsSortLabel('queue_position', 'Queue')}</th>
                    <th className="jobs-table__actions-heading">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(jobsInfo?.tasks ?? []).map((job) => {
                    const displayProgressPercent = Math.round(getTaskDisplayProgress(job) * 100);
                    const jobTitle = job.filename || job.pdf_id || job.task_id;
                    return (
                      <tr className={selectedJobId === job.task_id ? 'jobs-table__row--selected' : ''} key={job.task_id}>
                        <td>
                          <span className={`job-status job-status--${job.status}`}>{job.status}</span>
                        </td>
                        <td>{formatTaskType(job.type)}</td>
                        <td className="jobs-table__job">
                          <div className="jobs-table__id" title={jobTitle}>
                            {jobTitle}
                          </div>
                          <div className="jobs-table__message" title={job.message || job.task_id}>
                            {job.message || job.task_id}
                          </div>
                        </td>
                        <td>
                          <div className="jobs-progress">
                            <span>{displayProgressPercent}%</span>
                            <div className="jobs-progress__bar" aria-hidden="true">
                              <span style={{ width: `${displayProgressPercent}%` }} />
                            </div>
                          </div>
                        </td>
                        <td>{formatTaskDateTime(job.updated_at)}</td>
                        <td>{formatTaskDateTime(job.created_at)}</td>
                        <td>{job.queue_position ? `#${job.queue_position}` : '—'}</td>
                        <td>
                          <div className="jobs-table__actions">
                            {isCancelableTask(job) && (
                              <button
                                className="icon-btn job-row__action job-row__action--cancel"
                                type="button"
                                onClick={() => void handleCancelJob(job)}
                                title="Cancel job"
                                aria-label={`Cancel job ${job.task_id}`}
                              >
                                <Icon name="cancel" className="icon-btn__svg" />
                              </button>
                            )}
                            {job.status === 'completed' && job.result_path && (
                              <a
                                className="icon-btn job-row__action"
                                href={getTaskDownloadUrl(job.task_id)}
                                onClick={(event) => {
                                  if (job.task_id === editableTaskId) {
                                    event.preventDefault();
                                    void downloadResults(job.task_id);
                                  }
                                }}
                                target="_blank"
                                rel="noreferrer"
                                title="Download result"
                                aria-label={`Download result for job ${job.task_id}`}
                              >
                                <Icon name="download" className="icon-btn__svg" />
                              </a>
                            )}
                            <button
                              className="icon-btn job-row__action job-row__action--open"
                              type="button"
                              onClick={() => void handleOpenJob(job)}
                              title="Open job"
                              aria-label={`Open job ${job.task_id}`}
                            >
                              <Icon name="open" className="icon-btn__svg" /><span>Open</span>
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
          <div className="jobs-pagination">
            <span>
              {jobsInfo?.total_count ?? 0} job{(jobsInfo?.total_count ?? 0) === 1 ? '' : 's'} · page {jobsInfo?.page ?? jobsPage} of{' '}
              {jobsInfo?.total_pages ?? 1}
            </span>
            <div>
              <button
                className="icon-btn"
                type="button"
                disabled={(jobsInfo?.page ?? jobsPage) <= 1 || jobsLoading}
                onClick={() => setJobsPage(1)}
                title="First page"
                aria-label="First page"
              >
                <Icon name="first" className="icon-btn__svg" />
              </button>
              <button
                className="icon-btn"
                type="button"
                disabled={(jobsInfo?.page ?? jobsPage) <= 1 || jobsLoading}
                onClick={() => setJobsPage((value) => Math.max(1, value - 1))}
                title="Previous page"
                aria-label="Previous page"
              >
                <Icon name="previous" className="icon-btn__svg" />
              </button>
              <button
                className="icon-btn"
                type="button"
                disabled={(jobsInfo?.page ?? jobsPage) >= (jobsInfo?.total_pages ?? 1) || jobsLoading}
                onClick={() => setJobsPage((value) => value + 1)}
                title="Next page"
                aria-label="Next page"
              >
                <Icon name="next" className="icon-btn__svg" />
              </button>
              <button
                className="icon-btn"
                type="button"
                disabled={(jobsInfo?.page ?? jobsPage) >= (jobsInfo?.total_pages ?? 1) || jobsLoading}
                onClick={() => setJobsPage(jobsInfo?.total_pages ?? 1)}
                title="Last page"
                aria-label="Last page"
              >
                <Icon name="last" className="icon-btn__svg" />
              </button>
            </div>
          </div>
        </section>
      )}

      <nav className="step-nav" aria-label="Process steps">
        {stepDefinitions.map((step, index) => {
          const disabled = step.id > maxStep;
          const isActive = step.id === currentStep;
          const isCompleted = step.id < currentStep && step.id <= maxStep;
          const canNavigate = !disabled;
          return (
            <React.Fragment key={step.id}>
              <button
                type="button"
                className={`step-nav__button${isActive ? ' active' : ''}${isCompleted ? ' completed' : ''}`}
                onClick={() => handleStepNavigation(step.id)}
                disabled={!canNavigate}
                aria-current={isActive ? 'step' : undefined}
              >
                <span className="step-nav__icon">
                  <StepGlyph type={step.icon} className="step-nav__icon-glyph" />
                </span>
                <span className="step-nav__text">
                  <span className="step-nav__step">Step {step.id}</span>
                  <span className="step-nav__label">{step.label}</span>
                </span>
              </button>
              {index < stepDefinitions.length - 1 && (
                <span className={`step-nav__connector${step.id < maxStep ? ' active' : ''}`} />
              )}
            </React.Fragment>
          );
        })}
      </nav>

      <div className="content-stack">
        <ResizableSplit className="workbench-split" storageKey="bci:workbench:split">
          <div className="workbench-document" aria-label="Document preview">
            {pdfInfo?.pdf_id ? <React.Suspense fallback={<div className="pdf-pane pdf-pane--loading"><img className="pdf-pane__first-preview" src={getPdfPreviewUrl(pdfInfo.pdf_id, documentPage)} alt="PDF preview" /></div>}><PdfPane pdfId={pdfInfo.pdf_id}
              selectedCompoundLabel={activeReviewRowIndex !== null ? String(editedStructures[activeReviewRowIndex]?.COMPOUND_ID || "Selected row") : undefined}
              onRecognizeRegion={currentStep === 4 && editableTaskId ? recognizeManualRegion : undefined}
              target={currentStep === 4 && activeReviewRowIndex !== null && pdfTarget ? pdfTarget : { page: documentPage }}
              onPageChange={(page) => { setDocumentPage(page); pdfLocateSequenceRef.current += 1; setActiveReviewRowIndex(null); setPdfTarget(null); }} /></React.Suspense> : (
              <div className="upload-document__empty"><span className="upload-document__paper" aria-hidden="true"><StepGlyph type="upload" /><i /><i /><i /></span><h2>Your document, in view</h2><p>Upload a PDF or enter a patent number.<br />Read the source here as you work.</p></div>
            )}
          </div>
          <div className="workbench-content">
      {currentStep === 1 && (
          <section className="upload-controls">
          <h2>Source document</h2>
          <p className="upload-controls__intro">Import a patent to extract structures and bioactivity.</p>
          <form onSubmit={handlePatentFetch} className="patent-import">
            <input
              type="text"
              value={patentNumber}
              onChange={(e) => setPatentNumber(e.target.value)}
              aria-label="Patent number"
              placeholder="Patent number · WO2024163691A1"
              style={{ flex: 1, padding: '6px 10px' }}
            />
            <button type="submit" disabled={isUploading || !patentNumber.trim()}>
              {isUploading ? 'Downloading…' : 'Fetch PDF'}
            </button>
          </form>
          <label className={`source-drop${isUploading ? ' source-drop--busy' : ''}${pdfInfo ? ' source-drop--loaded' : ''}`}
            onDragOver={(event) => { event.preventDefault(); }}
            onDrop={(event) => {
              event.preventDefault();
              const file = event.dataTransfer.files[0];
              if (file) void uploadSourceFile(file);
            }}>
            <input type="file" accept="application/pdf,.pdf" onChange={handleFileUpload} disabled={isUploading} />
            <span className="source-drop__art" aria-hidden="true"><StepGlyph type="upload" /></span>
            <strong>{isUploading ? 'Importing…' : pdfInfo ? 'Replace PDF' : 'Drop a patent PDF here'}</strong>
            <span>or click to browse</span>
            <span className="source-drop__flow" aria-hidden="true">
              <StepGlyph type="structures" /><span>Structures</span><span>→</span>
              <StepGlyph type="bioactivity" /><span>Activity</span><span>→</span>
              <StepGlyph type="review" /><span>Review</span>
            </span>
          </label>
          {isUploading && (
            <div className="upload-progress" role="status" aria-live="polite">
              <div className="upload-progress__bar">
                <span style={{ width: `${uploadProgress ?? 0}%` }} />
              </div>
              <span className="upload-progress__label">Uploading… {uploadProgress ?? 0}%</span>
            </div>
          )}
          {pdfInfo && (
            <div className="planning-panel">
              <div className="launch-card">
                <div className="launch-card__info">
                  <div className="launch-card__file">
                    <span className="launch-card__filename">{pdfInfo.filename}</span>
                    <span className="launch-card__pages">{pdfInfo.total_pages} page{pdfInfo.total_pages === 1 ? '' : 's'}</span>
                  </div>
                  <p className="launch-card__desc">
                    Extract molecular structures, biological activity and ADME data with links to the source.
                  </p>
                  <label className={`assembly-option${assemblyAvailable ? ' assembly-option--available' : ''}`} title={assemblyAvailable ? 'Assemble individual compounds from explicit scaffold attachment sites and matching substituents.' : assemblyChecked ? 'Assembly is not available on this server.' : 'Checking assembly availability…'}>
                    <input type="checkbox" aria-label="Assemble Markush members" checked={assembleMarkush} disabled={!assemblyAvailable || automaticExtractionActive} onChange={(event) => setAssembleMarkush(event.target.checked)} />
                    <span>Assemble Markush members <small>{assemblyChecked ? assemblyAvailable ? 'Optional' : 'Not available' : 'Checking…'}</small></span>
                  </label>
                  <p className="assembly-option__hint">Use scaffold and substituent definitions to propose individual molecules. Uncertain connections remain for review.</p>
                </div>
                <button
                  className={automaticExtractionActive ? 'button--cancel' : 'button--start'}
                  type="button"
                  onClick={automaticExtractionActive ? handleCancelAutomaticExtraction : handleStartFullPipeline}
                  disabled={isCanceling || (!automaticExtractionActive && !pdfReadyForDetection)}
                >
                  {automaticExtractionActive
                    ? (
                      <>
                        <Icon name="cancel" className="button-icon" />
                        {isCanceling ? 'Canceling…' : 'Cancel'}
                      </>
                    )
                    : pdfReadyForDetection
                    ? (
                      <>
                        <Icon name="play" className="button-icon" />
                        Start extraction
                      </>
                    )
                    : (
                      <>
                        <Icon name="play" className="button-icon" />
                        Preparing…
                      </>
                    )}
                </button>
              </div>

              {(fullPipelineTask || autoDetectTask || structureTask || assayTask) && (
                <div className="pipeline-progress">
                  <div className="pipeline-progress__header">
                    <span className="pipeline-progress__label">
                      {fullPipelineTask
                        ? fullPipelineTask.message || fullPipelineTask.status
                        : autoDetectTask && (autoDetectTask.status === 'running' || autoDetectTask.status === 'pending')
                        ? autoDetectTask.message || 'Detecting…'
                        : structureTask && (structureTask.status === 'running' || structureTask.status === 'pending')
                        ? structureTask.message || 'Extracting structures…'
                        : assayTask && (assayTask.status === 'running' || assayTask.status === 'pending')
                        ? assayTask.message || 'Extracting bioactivity…'
                        : 'Processing…'}
                    </span>
                    <span className="pipeline-progress__pct">
                      {Math.round(
                        ((fullPipelineTask?.progress ?? 0) ||
                         (autoDetectTask?.progress ?? 0) ||
                         (structureTask?.progress ?? 0) ||
                         (assayTask?.progress ?? 0)) * 100,
                      )}%
                    </span>
                  </div>
                  <div className="pipeline-progress__bar">
                    <span
                      className="pipeline-progress__fill"
                      style={{
                        width: `${Math.round(
                          ((fullPipelineTask?.progress ?? 0) ||
                           (autoDetectTask?.progress ?? 0) ||
                           (structureTask?.progress ?? 0) ||
                           (assayTask?.progress ?? 0)) * 100,
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {pdfInfo && currentStep === 2 && (
        <section className={`stage-workspace stage-workspace--structures${stageOptionsOpen ? ' stage-workspace--options' : ''}`}>
          <div className="selector-layout">
            <div className="selector-toolbar selector-toolbar--structure">
              <div className="selector-field">
                <div className="selector-field__heading">
                  <label htmlFor="structure-pages">Pages</label>
                  <label className="selector-check" htmlFor="auto-structure-pages">
                    <input
                      id="auto-structure-pages"
                      type="checkbox"
                      checked={autoDetectStructurePages}
                      onChange={(event) => {
                        setAutoDetectStructurePages(event.target.checked);
                        if (event.target.checked) setStructureSelectionFeedback('Selected pages are retained. Clear the selection to discover pages again.');
                      }}
                    />
                    Auto
                  </label>
                </div>
                <textarea
                  id="structure-pages"
                  placeholder="e.g. 12-18, 24, 31"
                  value={structurePagesInput}
                  onChange={handleStructureInputChange}
                  rows={2}
                />
                <div className="selector-compact-row">
                  <span className="tag">
                    {structureSelection.size > 0
                      ? `${structureSelection.size} selected`
                      : 'Auto'}
                  </span>
                  {structureSelection.size > 0 && (
                    <button className="small-btn" onClick={clearStructureSelection} type="button">
                      Clear
                    </button>
                  )}
                </div>
              </div>
              <div className="selector-field selector-field--filter">
                <label htmlFor="structure-filter-strictness">Filter</label>
                <span className="filter-select filter-select--wide">
                  <select
                    id="structure-filter-strictness"
                    value={structureFilterStrictness}
                    title="Structure filter strictness"
                    onChange={(event) =>
                      setStructureFilterStrictness(event.target.value as StructureFilterStrictness)
                    }
                  >
                    <option value="strict">Strict</option>
                    <option value="balanced">Balanced</option>
                    <option value="permissive">Permissive</option>
                  </select>
                </span>
              </div>

              <div className="selector-actions">
{isGalleryLoading && (
                  <small className="selector-actions__hint">Loading page previews…</small>
                )}
                {structureTask && structureTask.status !== 'completed' && structureTask.status !== 'failed' && (
                  <div className="selector-status">
                    <span>Structure task: {structureTask.message || structureTask.status}</span>
                    <div className="progress-bar slim">
                      <span style={{ width: `${Math.round(structureTask.progress * 100)}%` }} />
                    </div>
                  </div>
                )}
                {structureAddonTask && structureAddonTask.status !== 'completed' && structureAddonTask.status !== 'failed' && (
                  <div className="selector-status">
                    <span>Additional structure pages: {structureAddonTask.message || structureAddonTask.status}</span>
                    <div className="progress-bar slim">
                      <span style={{ width: `${Math.round(structureAddonTask.progress * 100)}%` }} />
                    </div>
                  </div>
                )}
                {currentStep === 2 && structures.length === 0 && filteredStructures.length > 0 && loadedStructureTaskId && (
                  <div className="selector-status" role="note">
                    <span>
                      The last extraction accepted no structures — {filteredStructures.length} candidate
                      {filteredStructures.length === 1 ? '' : 's'} were detected and filtered:
                    </span>
                    <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: '0.85em' }}>
                      {filteredStructures.slice(0, 8).map((candidate, index) => (
                        <li key={index} style={{ marginBottom: 2 }}>
                          {`page ${candidate.PAGE_NUM ?? candidate.page_num ?? '?'}: ${(
                            candidate.STRUCTURE_FILTER_REASON
                            || candidate.structure_filter_reason
                            || candidate.STRUCTURE_TYPE
                            || candidate.structure_type
                            || 'filtered'
                          ).toString().slice(0, 160)}`}
                        </li>
                      ))}
                    </ul>
                    {filteredStructures.length > 8 && (
                      <span>… and {filteredStructures.length - 8} more (see filtered_structures.csv in the task download).</span>
                    )}
                    <span>
                      Tip: switch the structure filter strictness (e.g. to Balanced or Permissive) and run again to keep more candidates.
                    </span>
                  </div>
                )}
                {structureSelectionFeedback && (
                  <small className="selector-actions__hint selector-actions__hint--notice">{structureSelectionFeedback}</small>
                )}
              </div>
            </div>

            {renderStageContent('structure')}
          </div>
        </section>
      )}

      {pdfInfo && currentStep === 3 && (
        <section className={`stage-workspace stage-workspace--assays${stageOptionsOpen ? ' stage-workspace--options' : ''}`}>
          <div className="selector-layout">
            <div className="selector-toolbar selector-toolbar--assay">
              <div className="selector-field">
                <div className="selector-field__heading">
                  <label htmlFor="assay-pages">Pages</label>
                  <label className="selector-check" htmlFor="auto-assay-pages">
                    <input
                      id="auto-assay-pages"
                      type="checkbox"
                      checked={autoDetectAssayPages}
                      onChange={(event) => {
                        setAutoDetectAssayPages(event.target.checked);
                        if (event.target.checked) setAssaySelectionFeedback('Selected pages are retained. Clear the selection to discover pages again.');
                      }}
                    />
                    Auto
                  </label>
                </div>
                <textarea
                  id="assay-pages"
                  placeholder="e.g. 42, 76-81"
                  value={assayPagesInput}
                  onChange={handleAssayInputChange}
                  rows={2}
                />
                <div className="selector-compact-row">
                  <span className="tag">
                    {assaySelection.size > 0
                      ? `${assaySelection.size} selected`
                      : 'Auto'}
                  </span>
                  {assaySelection.size > 0 && (
                    <button className="small-btn" onClick={clearAssaySelection} type="button">
                      Clear
                    </button>
                  )}
                </div>
              </div>
              <div className="selector-field">
                <div className="selector-field__heading">
                  <label htmlFor="assay-names">Assay names</label>
                  <label className="selector-check" htmlFor="auto-assay-names">
                    <input
                      id="auto-assay-names"
                      type="checkbox"
                      checked={autoDetectAssayNames}
                      onChange={(event) => setAutoDetectAssayNames(event.target.checked)}
                    />
                    Auto
                  </label>
                </div>
                <div className="assay-name-editor">
                  <input
                    id="assay-names"
                    type="text"
                    placeholder="IC50, EC50…"
                    value={assayNameDraft}
                    onChange={(event) => setAssayNameDraft(event.target.value)}
                    onKeyDown={handleAssayNameKeyDown}
                    onBlur={commitAssayDraft}
                    onPaste={handleAssayNamePaste}
                  />
                  <button
                    className="small-btn add-btn"
                    onClick={commitAssayDraft}
                    type="button"
                    disabled={!assayNameDraft.trim()}
                  >
                    Add
                  </button>
                </div>
                {assayNames.length > 0 && (
                  <div className="assay-name-chip-list">
                    {assayNames.map((name, index) => (
                      <span className="assay-name-chip" key={`${name}-${index}`}>
                        {name}
                        <button
                          type="button"
                          className="assay-name-chip__remove"
                          onClick={() => removeAssayName(index)}
                          aria-label={`Remove ${name}`}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                    <button className="small-btn subtle" type="button" onClick={clearAssayNames}>
                      Clear list
                    </button>
                  </div>
                )}
              </div>

              <div className="selector-actions">
{isGalleryLoading && (
                <small className="selector-actions__hint" role="status">Loading previews…</small>
              )}
              {assayTask && assayTask.status !== 'completed' && assayTask.status !== 'failed' && (
                <div className="selector-status">
                  <span>Bioactivity task: {assayTask.message || assayTask.status}</span>
                  <div className="progress-bar slim">
                    <span style={{ width: `${Math.round(assayTask.progress * 100)}%` }} />
                  </div>
                </div>
              )}
              {assayAddonTask && assayAddonTask.status !== 'completed' && assayAddonTask.status !== 'failed' && (
                <div className="selector-status">
                  <span>Additional bioactivity pages: {assayAddonTask.message || assayAddonTask.status}</span>
                  <div className="progress-bar slim">
                    <span style={{ width: `${Math.round(assayAddonTask.progress * 100)}%` }} />
                  </div>
                </div>
              )}
              {assaySelectionFeedback && (
                <small className="selector-actions__hint selector-actions__hint--notice">{assaySelectionFeedback}</small>
              )}
              </div>
              </div>

            {renderStageContent('assay')}
          </div>
        </section>
      )}

          {currentStep === 4 && (
        <section className="results-workspace" aria-label="Document and results">
          {(pdfInfo || structures.length > 0 || assayRecords.length > 0 || editableTaskId) && (
            <div className={`review-layout${reviewView === 'cards' ? ' review-layout--cards' : ''}${sourcePaneVisible ? ' review-layout--with-source' : ''}${reviewDensity === 'compact' ? ' review-layout--compact' : ''}${showReviewSmiles ? ' review-layout--smiles' : ''}${showPageThumbnails ? ' review-layout--page-thumbs' : ''}`}>
              <div className="review-main">
                {(structureTask && structureTask.status !== 'completed') && (
                  <div className="status-banner" style={{ marginTop: 16 }}>
                    <span>
                      Structure task status: <strong>{structureTask.status}</strong>
                      {structureTask.message ? ` · ${structureTask.message}` : ''}
                    </span>
                    <div style={{ minWidth: 160 }}>
                      <div className="progress-bar">
                        <span style={{ width: `${Math.round(structureTask.progress * 100)}%` }} />
          </div>
        </div>
      </div>
    )}
                {assayTask && assayTask.status !== 'completed' && (
                  <div className="status-banner" style={{ marginTop: 12 }}>
                    <span>
                      Bioactivity task status: <strong>{assayTask.status}</strong>
                      {assayTask.message ? ` · ${assayTask.message}` : ''}
                    </span>
                    <div style={{ minWidth: 160 }}>
                      <div className="progress-bar">
                        <span style={{ width: `${Math.round(assayTask.progress * 100)}%` }} />
                      </div>
                    </div>
                  </div>
                )}
                {verificationMode && (
                  <React.Suspense fallback={<div className="annot-loading">Loading verification…</div>}>
                    <VerificationWorkspace />
                  </React.Suspense>
                )}
                {debugAnnotationMode && fullPipelineTask?.status === 'completed' && fullPipelineTask?.task_id && (
                  <React.Suspense fallback={<div className="annot-loading">Loading…</div>}>
                    <AnnotationWorkspace
                      taskId={fullPipelineTask.task_id}
                      onClose={() => setDebugAnnotationMode(false)}
                    />
                  </React.Suspense>
                )}
                  <div className="review-split__list">
                <div className="review-toolbar">
                  <div className="jobs-toolbar review-toolbar__controls">
                    <details className="review-export-menu review-manage-menu">
                      <summary aria-label="Manage review data" title="Add, restore or recover review data"><svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg><span>Add</span></summary>
                      <div>
                        <button type="button" disabled={!editableTaskId} onClick={() => setReviewAddKind('compound')}>Add compound</button>
                        <button type="button" disabled={!editableTaskId || !editedStructures.length} onClick={() => setReviewAddKind('measurement')}>Add measurement</button>
                        <button type="button" onClick={() => { gridLayout.reset(); setReviewSortColumn(''); setToast('Default column widths and order restored.'); }}>Reset table layout</button>
                        <button type="button" disabled={!editableTaskId} onClick={() => void restorePreviousReview()}>Restore previous version</button>
                        <button type="button" disabled={!editableTaskId} onClick={() => setSarOpen((value) => !value)}>Needs review{sarReport ? ` (${sarReport.coverage.open_gaps})` : ''}</button>
                        <button type="button" disabled={!editableTaskId} onClick={() => void startSarRecovery()}>Find missing data</button>
                      </div>
                    </details>
                    <div className="review-toolbar__views" role="group" aria-label="Result view">
                    <button type="button"
                            className={reviewView === 'table' ? 'is-active' : ''}
                            aria-pressed={reviewView === 'table'} aria-label="Table" title="Table view" onClick={() => setReviewView('table')}><Icon name="table" className="button-icon" /></button>
                    <button type="button"
                            className={reviewView === 'cards' ? 'is-active' : ''}
                            aria-pressed={reviewView === 'cards'} aria-label="Cards" title="Card view" onClick={() => setReviewView('cards')}><Icon name="cards" className="button-icon" /></button>
                  </div>
                    <button type="button" className="review-focus-button" aria-label={reviewFocusMode ? 'Exit focus mode' : 'Focus review'} title={reviewFocusMode ? 'Exit focus mode (Esc)' : 'Focus review'} onClick={() => setReviewFocusMode((value) => !value)}>
                      <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.4"><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4" /></svg>
                    </button>
                    {saveStatusMeta && (
                      <div className={`review-main__status review-save--${saveStatus}`} role="status" title={saveStatusMeta.label} aria-label={saveStatusMeta.label}>
                        {saveStatus === 'error'
                          ? <button type="button" className="small-btn" aria-label="Retry save" title="Retry save" onClick={() => { void performAutoSave(); }}><Icon name="go" className="button-icon" /></button>
                          : <Icon name="check" className="button-icon" />}
                      </div>
                    )}
                    {editableTaskId && <div className="review-toolbar__downloads">
                      {fullPipelineTask?.status !== 'completed' && assayColumnNames.length > 0 ? (
                        <details className="review-export-menu">
                          <summary title="Export" aria-label="Export"><Icon name="download" className="button-icon" /><span>Export</span></summary>
                          <div>
                            <button type="button" onClick={downloadStructuresCsv} disabled={isDownloading}>Structures ZIP</button>
                            <button type="button" onClick={downloadAssayCsv} disabled={isDownloading || assayRecords.length === 0}>Activity & ADME ZIP</button>
                          </div>
                        </details>
                      ) : (
                        <button className="secondary review-toolbar__button" type="button" onClick={downloadStructuresCsv} disabled={isDownloading} aria-label="Export" title={isDownloading ? 'Exporting…' : 'Export results ZIP'}><Icon name="download" className="button-icon" /><span>Export</span></button>
                      )}
                    </div>}
                    <div className="columns-picker">
                    <button
                      type="button"
                      className={`secondary columns-picker__toggle${columnsPickerOpen ? ' is-open' : ''}`}
                      onClick={() => { setColumnsPickerOpen((open) => !open); setColumnSearch(''); setReviewFiltersOpen(false); }}
                      aria-expanded={columnsPickerOpen}
                    >
                      <svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="4" y="5" width="16" height="14" rx="2" /><path d="M10 5v14M15 5v14" /></svg>Columns{activeMetadataColumns.size > 0 ? ` (${activeMetadataColumns.size})` : ''}
                    </button>
                    {columnsPickerOpen && (
                      <div className="columns-picker__panel">
                        <input type="search" className="review-field-search" aria-label="Find a field" placeholder="Find a field…" value={columnSearch} onChange={(event) => setColumnSearch(event.target.value)} />
                        <p className="review-field-hint">Drag headers or use Arrange columns to change field order. Drag a header edge to resize.</p>
                        <details className="review-field-order"><summary>Arrange columns</summary><div>{arrangedGridColumns.filter((key) => gridColumnLabel(key).toLowerCase().includes(columnSearch.trim().toLowerCase())).map((key) => <div key={key}><span title={gridColumnLabel(key)}>{gridColumnLabel(key)}</span><button type="button" disabled={arrangedGridColumns.indexOf(key) === 0} aria-label={`Move ${gridColumnLabel(key)} left`} onClick={() => moveGridColumn(key, arrangedGridColumns[arrangedGridColumns.indexOf(key) - 1])}>‹</button><button type="button" disabled={arrangedGridColumns.indexOf(key) === arrangedGridColumns.length - 1} aria-label={`Move ${gridColumnLabel(key)} right`} onClick={() => moveGridColumn(key, arrangedGridColumns[arrangedGridColumns.indexOf(key) + 1], true)}>›</button></div>)}</div></details>
                        <div className="review-field-actions"><button type="button" onClick={() => setHiddenMeasurements(new Set())}>Show measurements</button><button type="button" onClick={() => setHiddenMeasurements(new Set(assayColumnNames))}>Hide measurements</button></div>
                        <div className="columns-picker__hint">Measurement columns</div>
                        {assayColumnNames.filter((column) => column.toLowerCase().includes(columnSearch.trim().toLowerCase())).map((column) => <label key={column} className="columns-picker__option"><input type="checkbox" checked={!hiddenMeasurements.has(column)} onChange={() => setHiddenMeasurements((previous) => { const next = new Set(previous); if (next.has(column)) next.delete(column); else next.add(column); return next; })} /><span>{column}</span></label>)}
                        <div className="columns-picker__hint">Additional fields</div>
                        {structureColumnsToRender.length === 0 && (
                          <span className="muted">No extra columns available</span>
                        )}
                        {structureColumnsToRender.filter((column) => (STRUCTURE_COLUMN_LABELS[column] || column).toLowerCase().includes(columnSearch.trim().toLowerCase())).map((column) => (
                          <label key={column} className="columns-picker__option">
                            <input
                              type="checkbox"
                              checked={activeMetadataColumns.has(column)}
                              onChange={(event) => {
                                const checked = event.target.checked;
                                setActiveMetadataColumns((prev) => {
                                  const next = new Set(prev);
                                  if (checked) next.add(column);
                                  else next.delete(column);
                                  return next;
                                });
                              }}
                            />
                            <span>{STRUCTURE_COLUMN_LABELS[column] ?? column}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                    <button type="button" className={`secondary review-search-toggle${reviewSearch ? ' has-filter' : ''}`} aria-label="Search results" title="Search results" aria-expanded={reviewSearchOpen} aria-controls="review-search-panel" onClick={() => { setReviewSearchOpen((open) => !open); setReviewFiltersOpen(false); }}><Icon name="search" className="button-icon" /><span>Search</span></button>
                    {reviewSearchOpen && <div className="review-search-panel" id="review-search-panel">
                      <input autoFocus type="search" value={reviewSearch} aria-label="Search results" placeholder="Find compound or activity" onChange={(event) => setReviewSearch(event.target.value)} />
                      <button type="button" aria-label="Close search" title="Close search" onClick={() => setReviewSearchOpen(false)}><Icon name="cancel" className="button-icon" /></button>
                    </div>}
                    <button type="button" className={`secondary review-filters-toggle${hasReviewFilters || reviewSortColumn ? " has-filter" : ""}`} title="Filters" aria-label="Filters"
                      aria-expanded={reviewFiltersOpen} aria-controls="review-advanced-filters"
                      onClick={() => { setReviewFiltersOpen((open) => !open); setReviewSearchOpen(false); }}><Icon name="filter" className="button-icon" /><span>Filter & sort</span></button>
                    <div id="review-advanced-filters" className={`review-toolbar__advanced${reviewFiltersOpen ? ' is-open' : ''}`}>
                    <label>
                      <span>Filter column</span>
                      <select
                        value={reviewRequiredColumn}
                        onChange={(event) => setReviewRequiredColumn(event.target.value)}
                      >
                        <option value="">Any column</option>
                        {reviewColumnFilterOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label><span>Condition</span><select aria-label="Filter condition" value={reviewFilterOperator} onChange={(event) => setReviewFilterOperator(event.target.value)}><option value="filled">Is not empty</option><option value="empty">Is empty</option><option value="contains">Contains</option><option value="equals">Equals</option><option value="gt">Greater than</option><option value="lt">Less than</option></select></label>
                    {!['filled', 'empty'].includes(reviewFilterOperator) && <label><span>Value</span><input aria-label="Filter value" value={reviewFilterValue} onChange={(event) => setReviewFilterValue(event.target.value)} /></label>}
                    <label><span>Sort by</span><select aria-label="Sort column" value={reviewSortColumn} onChange={(event) => setReviewSortColumn(event.target.value)}><option value="">Manual order</option>{reviewColumnFilterOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                    <label><span>Order</span><select aria-label="Sort direction" value={reviewSortDescending ? 'desc' : 'asc'} onChange={(event) => setReviewSortDescending(event.target.value === 'desc')}><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
                    <button type="button" className="secondary" onClick={() => { setReviewRequiredColumn(''); setReviewFilterValue(''); setReviewSortColumn(''); setReviewSearch(''); }}>Reset view</button>
                    <label className="review-row-height">
                      <span>Row height</span>
                      <div className="review-toolbar__control review-row-height__control">
                        <input
                          type="range"
                          min={72}
                          max={240}
                          step={8}
                          value={rowHeight}
                          onChange={(event) => setRowHeight(Number(event.target.value))}
                        />
                        <strong>{rowHeight}px</strong>
                      </div>
                    </label>
                    <label>
                      <span>Rows</span>
                      <select aria-label="Rows per page"
                        value={reviewPageSize}
                        onChange={(event) => setReviewPageSize(Number(event.target.value))}
                      >
                        {REVIEW_PAGE_SIZE_OPTIONS.map((size) => (
                          <option key={size} value={size}>
                            {size}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label><span>Density</span><select aria-label="Result density" value={reviewDensity} onChange={(event) => setReviewDensity(event.target.value)}><option value="compact">Compact</option><option value="roomy">Comfortable</option></select></label>
                    <label className="review-page-thumbs-option"><input type="checkbox" checked={showPageThumbnails} onChange={(event) => setShowPageThumbnails(event.target.checked)} /> Page thumbnails</label>
                    <label className="review-smiles-option"><input type="checkbox" checked={showReviewSmiles} onChange={(event) => setShowReviewSmiles(event.target.checked)} /> SMILES</label>


                    </div>
                    {renderReviewPagination('top')}
                  </div>
                </div>

                {renamingMeasurement && <form className="review-column-rename" onSubmit={(event) => { event.preventDefault(); const name = measurementNameDraft.trim(); if (!name || name === renamingMeasurement) { setRenamingMeasurement(null); return; } if (assayColumnNames.includes(name)) { setError('A column with this name already exists.'); return; } const old = renamingMeasurement; const columnOrder = [...gridLayout.columns.filter((key) => gridColumnKeys.includes(key)), ...gridColumnKeys.filter((key) => !gridLayout.columns.includes(key))]; const widths = { ...gridLayout.widths }; if (widths[`assay:${old}`]) { widths[`assay:${name}`] = widths[`assay:${old}`]; delete widths[`assay:${old}`]; } gridLayout.update({ columns: columnOrder.map((key) => key === `assay:${old}` ? `assay:${name}` : key), widths }); setHiddenMeasurements((previous) => new Set([...previous].map((key) => key === old ? name : key))); if (reviewSortColumn === `assay:${old}`) setReviewSortColumn(`assay:${name}`); if (reviewRequiredColumn === `assay:${old}`) setReviewRequiredColumn(`assay:${name}`); const updated = editedStructuresRef.current.map((record) => { const value = measurementValues(record)[old] ?? record[old] ?? processedStructureRows.find((row) => row.record === record)?.assayData[old] ?? ''; let removed: string[] = []; try { removed = JSON.parse(String(record.USER_REMOVED_MEASUREMENTS || '[]')); } catch {} const edits = measurementEdits(record); delete edits[old]; return { ...record, USER_MEASUREMENT_EDITS: JSON.stringify({ ...edits, [name]: String(value) }), USER_REMOVED_MEASUREMENTS: JSON.stringify([...new Set([...removed, old])].filter((column) => column !== name)) }; }); editedStructuresRef.current = updated; setEditedStructures(updated); setRenamingMeasurement(null); scheduleAutoSave(); }}><label>Column name<input autoFocus aria-label="Column name" value={measurementNameDraft} onChange={(event) => setMeasurementNameDraft(event.target.value)} /></label><button type="submit" className="primary">Save name</button><button type="button" className="secondary" onClick={() => setRenamingMeasurement(null)}>Cancel</button></form>}
                {reviewView !== 'cards' && (
                <div className="table-wrapper" style={tableStyle as React.CSSProperties} ref={tableWrapperRef}>
                  <table className="review-table">
                    <thead>
                      <ReviewGridRow order={gridLayout.columns} widths={gridLayout.widths} onMoveColumn={moveGridColumn} onResizeColumn={(column, width) => gridLayout.update({ widths: { ...gridLayout.widths, [column]: width } })}>
                        <th data-column="page">Page</th>
                        <th data-column="preview" className="review-column--page-preview">PDF preview</th>
                        <th data-column="id">Compound ID</th>
                        <th data-column="source">Source structure</th>
                        <th data-column="structure">Extracted structure</th>
                        <th data-column="smiles" className="review-column--smiles">SMILES</th>
                        {activeStructureColumns.map((column) => (
                          <th
                            key={column} data-column={`meta:${column}`}
                            className="column-header column-header--structure"
                          >
                            <span className="column-header__inner">
                              <span className="column-header__icon" aria-hidden="true">🧬</span>
                              <span>{STRUCTURE_COLUMN_LABELS[column] ?? column}</span>
                            </span>
                          </th>
                        ))}
                        {visibleMeasurementColumns.map((column) => (
                          <th data-column={`assay:${column}`} key={`assay-${column}`} title="Scientific endpoint and source unit · values can be corrected"><button type="button" className="review-column-name" title="Rename column" onClick={() => { setRenamingMeasurement(column); setMeasurementNameDraft(column); }}>{column}</button><button type="button" className="review-remove-measurement" aria-label={`Remove measurement ${column}`} disabled={!editableTaskId} onClick={() => removeMeasurement(column)}>×</button></th>
                        ))}
                        <th data-column="actions" className="column-header">
                          <span className="column-header__inner">Actions</span>
                        </th>
                      </ReviewGridRow>
                    </thead>
                    <tbody ref={tableBodyRef}>
                      {paginatedStructureRows.length === 0 && (
                        <tr>
                          <td
                            className="review-table__cell review-table__cell--empty"
                            colSpan={7 + activeStructureColumns.length + assayColumnNames.length}
                          >
                            {structures.length === 0
                              ? (loadedStructureTaskId
                                ? (filteredStructures.length ? `${filteredStructures.length} candidates were extracted and filtered out; see the notice in Step 2 (adjust strictness and re-run).` : 'No records in this result.')
                                : 'Run a structure extraction to populate this table.')
                              : 'No rows match the current filters.'}
                          </td>
                        </tr>
                      )}
                      {paginatedStructureRows.map((row) => {
                        const {
                          record,
                          pagePreviewImage,
                          pagePreviewSource,
                          fragmentPagePreviewImage,
                          fragmentPagePreviewSource,
                          segmentImage,
                          segmentSource,
                          structurePreviewKey,
                          structurePreview,
                          sourceCropKey,
                          primaryPageNumber,
                          assayData,
                          index,
                          pageHeading,
                          secondaryPageInfo,
                          artifactLabel,
                          normalizedPrimaryPage,
                          rowSpan,
                          showPageCell,
                          isMarkushAssembled,
                          scaffoldSourceCropKey,
                          scaffoldSegmentImage,
                          displayedScaffoldImage,
                          scaffoldPageNumber,
                        } = row;
                        const smilesValue = typeof record.SMILES === 'string' ? record.SMILES : '';
                        const compoundIdRaw = record.COMPOUND_ID;
                        const compoundIdValue =
                          typeof compoundIdRaw === 'string' ? compoundIdRaw : formatCellValue(compoundIdRaw);
                        const canEditStructure = index < editedStructures.length;
                        const isRowVisible = visibleRowIndices.has(index);
const previewPageNumber = isMarkushAssembled ? scaffoldPageNumber : primaryPageNumber;
const isPreviewLoading = Boolean(previewPageNumber && loadingPages.has(previewPageNumber as number));
const isFragmentPreviewLoading = Boolean(isMarkushAssembled && primaryPageNumber && loadingPages.has(primaryPageNumber as number));
                        const isSourceCropLoading = Boolean(sourceCropKey && loadingSourceCrops.has(sourceCropKey));
                        const isStructurePreviewLoading = Boolean(structurePreviewKey && loadingStructurePreviews.has(structurePreviewKey));
                        const sourceCropUnavailable = Boolean(sourceCropKey && Object.prototype.hasOwnProperty.call(sourceCropCache, sourceCropKey) && !segmentImage);
                        const pageCell = (
                          <td data-column="page"
                            className="review-table__cell review-table__cell--page-number"
                            rowSpan={rowSpan}
                          >
                            <div className="page-number"><ReviewRowHandle rowKey={reviewRowKey(row)} label={compoundIdValue || String(index + 1)} disabled={Boolean(reviewSortColumn)} onMove={moveGridRow} onStep={(direction) => { const position = filteredReviewRows.findIndex((item) => item.index === index); const adjacent = filteredReviewRows[position + direction]; if (adjacent) { if (direction < 0) moveGridRow(reviewRowKey(row), reviewRowKey(adjacent)); else moveGridRow(reviewRowKey(adjacent), reviewRowKey(row)); } }} />
                              <button type="button" className="review-page-link" aria-label={`Go to PDF page ${primaryPageNumber}`} onClick={() => void locateRowInPdf(row)}>{primaryPageNumber ?? "—"}</button>

                            </div>
                          </td>
                        );
                        return (
                          <ReviewGridRow order={gridLayout.columns} widths={gridLayout.widths} data-row-key={reviewRowKey(row)} key={`${selectedJobId ?? 'workspace'}-${row.id || 'row'}-${index}`} data-row-index={index}
                              onClick={(event) => { if (!(event.target as HTMLElement).closest('button,input,textarea,select,a,summary')) void locateRowInPdf(row); }}
                              className={`review-row${activeReviewRowIndex === index ? ' review-row--active' : ''}`}>
                            {(!normalizedPrimaryPage || showPageCell) && pageCell}
                            <td data-column="preview" className="review-table__cell review-table__cell--preview">
                              <div className="page-cell">
                                <div className="page-cell__media" style={isMarkushAssembled ? { display: 'flex', gap: 6, alignItems: 'flex-start' } : undefined}>
                                  {isMarkushAssembled ? (
                                    <>
                                      {pagePreviewImage ? (
                                        <button
                                          type="button"
                                          className="page-cell__image"
                                          disabled={!pagePreviewSource && !pagePreviewImage}
                                          onClick={() => { void locateRowInPdf(row); }}
                                        >
                                          <img src={pagePreviewImage} alt="Scaffold page" loading="lazy" style={{ maxWidth: 120 }} />
                                        </button>
                                      ) : (
                                        <div className="page-cell__placeholder page-cell__placeholder--compact" style={{ maxWidth: 120 }}>
                                          {isPreviewLoading ? 'Loading…' : 'No scaffold page'}
                                        </div>
                                      )}
                                      {fragmentPagePreviewImage ? (
                                        <button
                                          type="button"
                                          className="page-cell__image"
                                          disabled={!fragmentPagePreviewSource && !fragmentPagePreviewImage}
                                          onClick={() => { void locateRowInPdf(row); }}
                                        >
                                          <img src={fragmentPagePreviewImage} alt="Fragment page" loading="lazy" style={{ maxWidth: 120 }} />
                                        </button>
                                      ) : (
                                        <div className="page-cell__placeholder page-cell__placeholder--compact" style={{ maxWidth: 120 }}>
                                          {isFragmentPreviewLoading ? 'Loading…' : 'No fragment page'}
                                        </div>
                                      )}
                                    </>
                                  ) : pagePreviewImage ? (
                                    <button
                                      type="button"
                                      className="page-cell__image"
                                      disabled={!pagePreviewSource && !pagePreviewImage}
                                      onClick={() => { void locateRowInPdf(row); }}
                                    >
                                      <img src={pagePreviewImage} alt="PDF page" loading="lazy" />
                                    </button>
                                  ) : (
                                    <div className="page-cell__placeholder">
                                      {isPreviewLoading ? 'Loading…' : 'No preview'}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                            <td data-column="id" className="review-table__cell review-table__cell--id">
                              <CompoundIdInput
                                initialValue={compoundIdValue}
                                rowIndex={index}
                                disabled={!canEditStructure}
                                onSave={handleCompoundIdSave}
                              />
                            </td>
                            <td data-column="source" className="review-table__cell review-table__cell--structure">
                              {isMarkushAssembled ? (
                                <div className="structure-cell-content structure-cell-content--assembly" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                  {displayedScaffoldImage ? (
                                    <button
                                      type="button"
                                      className="structure-image-btn"
                                      style={{ flex: '0 0 auto' }}
                                      onClick={() => { void locateRowInPdf(row); }}
                                    >
                                      <img src={displayedScaffoldImage} alt="Scaffold" loading="lazy" style={{ maxWidth: 90 }} />
                                    </button>
                                  ) : (
                                    <div className="page-cell__placeholder page-cell__placeholder--compact" style={{ maxWidth: 90 }}>
                                      {Boolean(scaffoldSourceCropKey && loadingSourceCrops.has(scaffoldSourceCropKey))
                                        ? 'Loading…'
                                        : scaffoldSourceCropKey
                                        ? 'No scaffold crop'
                                        : ''}
                                    </div>
                                  )}
                                  {segmentImage ? (
                                    <button
                                      type="button"
                                      className="structure-image-btn"
                                      style={{ flex: '0 0 auto' }}
                                      onClick={() => { void locateRowInPdf(row); }}
                                    >
                                      <img src={segmentImage} alt="Fragment" loading="lazy" style={{ maxWidth: 90 }} />
                                    </button>
                                  ) : (
                                    <div className="page-cell__placeholder page-cell__placeholder--compact" style={{ maxWidth: 90 }}>
                                      {isSourceCropLoading ? 'Loading…' : sourceCropUnavailable ? 'Crop unavailable' : isRowVisible ? 'No crop' : 'Scroll to load'}
                                    </div>
                                  )}
                                </div>
                              ) : segmentImage ? (
                                <button
                                  type="button"
                                  className="structure-image-btn"
                                  disabled={!segmentSource && !pagePreviewSource && !segmentImage}
                                  onClick={() => { void locateRowInPdf(row); }}
                                >
                                  {isMarkushAssembled && displayedScaffoldImage && <img src={displayedScaffoldImage} alt="Assembly scaffold source" loading="lazy" />}
                                  <img src={segmentImage} alt="Source structure" loading="lazy" />
                                </button>
                              ) : primaryPageNumber ? (
                                <button type="button" className="review-source-link" onClick={() => void locateRowInPdf(row)} title="Open the source page; this result has no dedicated structure crop">View source · {primaryPageNumber}</button>
                              ) : (
                                <span className="muted">Source not linked</span>
                              )}
                            </td>
                            <td data-column="structure" className="review-table__cell review-table__cell--structure">
                              <div className="structure-cell-content">
                                <button
                                  type="button"
                                  className={structurePreview ? 'structure-image-btn' : 'smiles-chip'}
                                  title="Edit molecular structure"
                                  disabled={!canEditStructure}
                                  onClick={() => handleOpenStructureEditor(index)}
                                >
                                  {structurePreview ? (
                                    <img src={structurePreview} alt="Extracted structure" loading="lazy" />
                                  ) : isStructurePreviewLoading ? (
                                    'Rendering…'
                                  ) : smilesValue ? (
                                    <span className="smiles-text">{smilesValue}</span>
                                  ) : (
                                    record.NAME_STRUCTURE_CANDIDATE ? 'Review proposal' : 'Add SMILES'
                                  )}
                                </button>
                              </div>
                            </td>
                            <td data-column="smiles"
                              className="review-table__cell review-table__cell--smiles"
                              title={smilesValue || undefined}
                            >
                              {smilesValue || <span className="muted">—</span>}
                            </td>
                            {activeStructureColumns.map((column) => {
                              const value = record[column as keyof StructureRecord];
                              const editable = isEditableColumn(column, value) && canEditStructure;
                              return (
                                <td data-column={`meta:${column}`} key={column} className="review-table__cell review-table__cell--structure">
                                  {editable ? (
                                    <input
                                      type="text"
                                      value={formatCellValue(value)}
                                      onChange={(event) => handleCellChange(index, column, event.target.value)}
                                    />
                                  ) : (
                                    formatCellValue(column === 'Structure' ? record.Structure : value)
                                  )}
                                </td>
                              );
                            })}
                            {visibleMeasurementColumns.map((column) => (
                              <td data-column={`assay:${column}`}
                                key={column}
                                className="review-table__cell review-table__cell--assay"
                              >
                                <ReviewValueInput value={assayData[column]} label={`${column} for compound ${compoundIdValue || index + 1}`} disabled={!editableTaskId}
                                  onSave={(value) => handleMeasurementSave(index, record, column, value)} />
                              </td>
                            ))}
                            <td data-column="actions" className="review-table__cell review-table__cell--actions">
                              {canEditStructure ? (
                                <div className="review-row-actions">
                                <button type="button" className="small-btn" aria-label="Edit structure" title="Edit structure" onClick={() => handleOpenStructureEditor(index)}>Edit structure</button>
                                <button
                                  type="button"
                                  className="small-btn danger"
                                  onClick={() => handleDeleteStructureRow(index)}
                                >
                                  Delete
                                </button>
                                </div>
                              ) : (
                                <span className="muted">—</span>
                              )}
                            </td>
                          </ReviewGridRow>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                )}
                {sarOpen && sarReport && <aside className="sar-review-panel" aria-label="SAR evidence gaps"><header><strong>{sarReport.coverage.open_gaps} items need review</strong><button type="button" onClick={() => setSarOpen(false)} aria-label="Close evidence gaps">×</button></header><p>Review unresolved structures, identifiers and result links before using or exporting the data.</p><ul>{sarReport.gaps.slice(0, 20).map((gap) => <li key={gap.gap_id}><button type="button" onClick={() => { const row = processedStructureRows.find((item) => item.index === gap.row_index); if (row) void locateRowInPdf(row); else if (gap.page) setDocumentPage(gap.page); }}>{gap.page ? `Page ${gap.page} · ` : ''}{gap.reason}</button></li>)}</ul><button type="button" disabled={!editableTaskId} onClick={() => void startSarRecovery()}>Find missing data</button></aside>}
                {editedStructures.some((record) => record.MEASUREMENT_NOTES) && <details className="review-source-note"><summary>How to read these results</summary>{Array.from(new Set(editedStructures.map((record) => String(record.MEASUREMENT_NOTES || '')).filter(Boolean))).map((note) => <p key={note}>{note}</p>)}</details>}
                {/* Review cards: compact per-compound cards (desktop toggle + mobile layout) */}
                <div ref={reviewCardsRef} className={`review-cards${reviewView === 'cards' ? ' review-cards--active' : ''}`}>
                  {paginatedStructureRows.map((row) => {
                    const { record, index, pagePreviewImage, pagePreviewSource, segmentImage,
                      segmentSource, structurePreview,
                      isMarkushAssembled, displayedScaffoldImage, assayData } = row;
                    const cardSmiles = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
                    const cardCid = record.COMPOUND_ID;
                    const cardAssays = visibleMeasurementColumns.filter((column) => column !== 'Source annotation' && (formatCellValue(assayData[column]) || Object.prototype.hasOwnProperty.call(measurementEdits(record), column)));
                    const originalSourcePath = typeof record.Segment === 'string' && record.Segment
                      ? record.Segment : typeof record.SEGMENT_FILE === 'string' ? record.SEGMENT_FILE : '';
                    return (
                      <div className={`review-card${activeReviewRowIndex === index ? ' review-card--active' : ''}`}
                        data-row-index={index} key={`card-${index}`} tabIndex={0}
                        aria-label={`Compound ${cardCid ?? index + 1}`}
                        onClick={(event) => { if (!(event.target as HTMLElement).closest('button,input,textarea,select,a,summary')) void locateRowInPdf(row); }}
                        onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); void locateRowInPdf(row); } }}>
                        <div className="review-card__header">
                          <div className="review-card__identity">
                            <span className="review-card__eyebrow">Compound</span>
                            <CompoundIdInput
                              initialValue={typeof cardCid === 'string' ? cardCid : formatCellValue(cardCid)}
                              rowIndex={index}
                              disabled={index >= editedStructures.length}
                              onSave={handleCompoundIdSave}
                            />
                          </div>
                          <button type="button" className="review-card__page review-card__page-button"
                            disabled={!row.primaryPageNumber} aria-label={`Open source page ${row.primaryPageNumber || ''}`}
                            onClick={() => { void locateRowInPdf(row); }}>
                            {row.primaryPageNumber ? `Page ${row.primaryPageNumber}` : 'No source page'} <Icon name="open" className="button-icon button-icon--small" />
                          </button>
                        </div>
                        {(!cardSmiles && record.NAME_STRUCTURE_CANDIDATE) ? <p className="review-card__notice">Review the structure proposal before use.</p> : !cardSmiles && String(record.RESULT_ONLY).toLowerCase() === 'true' ? <p className="review-card__notice">Result row · molecular structure not linked yet.</p> : ['needs_review','failed'].includes(String(record.LOCAL_EVIDENCE_STATUS || '')) || record.MEASUREMENT_LINK_STATUS === 'identifier_unverified' ? <p className="review-card__notice">Check the compound ID before linking results.</p> : null}
                        <div className="review-card__images">
                          <div className="review-card__figure">
                            <div className="review-card__figure-heading"><span>{isMarkushAssembled ? "Scaffold + fragment" : "Source"}</span><span>Locate in PDF</span></div>
                            {segmentImage ? (
                              <>
                                <button type="button" className={`review-card__image-surface${isMarkushAssembled ? " review-card__image-surface--assembly" : ""}`} title="Show source in PDF" onClick={() => { void locateRowInPdf(row); }}>
                                  {isMarkushAssembled && displayedScaffoldImage && <img src={displayedScaffoldImage} alt="Assembly scaffold source" loading="lazy" />}
                                  <img src={segmentImage} alt="Source structure" loading="lazy" />
                                </button>
                                <button type="button" className="review-card__enlarge" aria-label={`Enlarge original for compound ${cardCid ?? ''}`} title="Enlarge original"
                                  onClick={() => openArtifact(segmentSource || segmentImage, `Source - ${cardCid ?? ''}`)}><Icon name="open" className="button-icon button-icon--small" /></button>
                              </>
                            ) : !sourcePaneVisible && pagePreviewImage && !originalSourcePath ? (
                              <button type="button" className="review-card__image-surface" onClick={() => openArtifact(pagePreviewSource || pagePreviewImage, `Preview - ${cardCid ?? ''}`, { rowIndex: index })}>
                                <img src={pagePreviewImage} alt="Source page" loading="lazy" />
                              </button>
                            ) : (
                              <button type="button" className="review-card__image-placeholder"
                                disabled={!row.primaryPageNumber && (!originalSourcePath || !artifactErrors[originalSourcePath])}
                                onClick={() => { if (originalSourcePath && artifactErrors[originalSourcePath]) setArtifactErrors((previous) => { const next = { ...previous }; delete next[originalSourcePath]; return next; }); else void locateRowInPdf(row); }}>
                                {originalSourcePath ? artifactErrors[originalSourcePath] ? 'Retry source image' : 'Loading source…' : row.primaryPageNumber ? `View source page ${row.primaryPageNumber}` : 'Source not linked'}
                              </button>
                            )}
                          </div>
                          <div className="review-card__figure">
                            <div className="review-card__figure-heading"><span>Extracted</span><span>Edit structure</span></div>
                            <button type="button" className={`review-card__image-surface${!structurePreview ? ' review-card__image-surface--empty' : ''}`} title="Edit molecular structure"
                              disabled={index >= editedStructures.length} onClick={() => handleOpenStructureEditor(index)}>
                              {structurePreview ? <img src={structurePreview} alt="Extracted structure" loading="lazy" /> : (
                                <span className="review-card__empty-structure"><span aria-hidden="true">＋</span>{cardSmiles ? 'Preview loading · open editor' : record.NAME_STRUCTURE_CANDIDATE ? 'Review name-derived proposal' : 'Draw structure'}</span>
                              )}
                            </button>
                          </div>
                        </div>
                        {isMarkushAssembled && displayedScaffoldImage && !segmentImage && <details className="review-card__scaffold"><summary>Assembly scaffold</summary><button type="button" onClick={() => openArtifact(displayedScaffoldImage, `Scaffold - ${cardCid ?? ''}`, { rowIndex: index })}><img src={displayedScaffoldImage} alt="Scaffold" loading="lazy" /></button></details>}
                        {record.ANNOTATION_TEXT && <details className="review-card__source-note"><summary>Original caption</summary><p>{formatCellValue(record.ANNOTATION_TEXT)}</p></details>}
                        {cardAssays.length > 0 && (
                          <dl className="review-card__assays">
                            {cardAssays.slice(0, 4).map((column) => (
                              <div key={column}><dt>{column}</dt><dd><ReviewValueInput value={assayData[column]} label={`${column} for compound ${cardCid || index + 1}`} disabled={!editableTaskId} onSave={(value) => handleMeasurementSave(index, record, column, value)} /></dd></div>
                            ))}
                          </dl>
                        )}
                        {cardAssays.length > 4 && <details className="review-card__more-values"><summary>+{cardAssays.length - 4} values</summary><dl className="review-card__assays">{cardAssays.slice(4).map((column) => <div key={column}><dt title={column}>{column}</dt><dd><ReviewValueInput value={assayData[column]} label={`${column} for compound ${cardCid || index + 1}`} disabled={!editableTaskId} onSave={(value) => handleMeasurementSave(index, record, column, value)} /></dd></div>)}</dl></details>}
                        <div className="review-card__actions">
                          <button type="button" className="small-btn" onClick={() => handleOpenStructureEditor(index)}
                            aria-label="Edit structure" disabled={index >= editedStructures.length}>Edit structure</button>
                          <button
                            type="button"
                            className="small-btn danger"
                            onClick={() => handleDeleteStructureRow(index)}
                            disabled={index >= editedStructures.length}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    );
                  })}
                  {paginatedStructureRows.length === 0 && (
                    <div className="review-card">
                      <span>
                        {structures.length === 0
                          ? (loadedStructureTaskId
                            ? (filteredStructures.length ? `${filteredStructures.length} candidates require review in Step 2.` : 'No records in this result.')
                            : 'Run a structure extraction to populate this table.')
                          : 'No rows match the current filters.'}
                      </span>
                    </div>
                  )}
                </div>
                  </div>
              </div>
            </div>
          )}
        </section>
      )}

          </div>
        </ResizableSplit>
      </div>

      {error && (
        <ErrorNotice message={error} onClose={() => setError(null)} />
      )}
      {toast && !error && (
        <div role="status" className="status-banner" style={{ background: '#ecfdf5', borderColor: '#bbf7d0', color: '#065f46' }}>
          <span>{toast}</span>
          <button className="secondary" onClick={() => setToast(null)} type="button">
            Got it
          </button>
        </div>
      )}

      {reviewAddKind && <ReviewAddDialog kind={reviewAddKind} onClose={() => setReviewAddKind(null)} onAdd={handleAddReviewData} />}
      {modalArtifact && (
        <div ref={imageDialogRef} className="modal" onClick={closeModal} role="dialog" aria-modal="true" aria-label="Image preview">
          <button type="button" className="image-dialog__close" aria-label="Close image"
            onClick={(event) => { event.stopPropagation(); closeModal(); }}>
            <Icon name="cancel" className="button-icon" />
          </button>
          <div
            className="modal-content modal-content--image-only"
            onClick={(event) => event.stopPropagation()}
            onPointerDown={(e) => handleModalImagePointerDown(e, e.currentTarget)}
            onPointerMove={handleModalImagePointerMove}
            onPointerUp={handleModalImagePointerUp}
            onPointerCancel={handleModalImagePointerUp}
          >
            {modalArtifact.mime === 'loading' ? (
              <div className="modal-spinner">
                <div className="spinner" />
                <span>Loading…</span>
              </div>
            ) : (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  ref={modalImageRef}
                  src={modalArtifact.data}
                  alt={modalArtifact.path}
                  className="magnify-preview"
                  style={{ display: 'block' }}
                />
                {modalBox && (
                  <div
                    style={boxStyle}
                  />
                )}
              </div>
            )}
          </div>
          {showModalQuickEdit && (
            <div
              ref={modalEditorPanelRef}
              className="floating-panel"
              style={{ top: modalEditorPosition.y, left: modalEditorPosition.x }}
              onPointerMove={handleModalEditorPointerMove}
              onPointerUp={handleModalEditorPointerUp}
              onPointerCancel={handleModalEditorPointerUp}
              onClick={(event) => event.stopPropagation()}
            >
              <div
                className="floating-panel__handle"
                onPointerDown={handleModalEditorPointerDown}
              >
                <span className="floating-panel__title">Quick edit</span>
                <span className="floating-panel__hint">Drag anywhere</span>
              </div>
              <div className="floating-panel__body">
                <label className="floating-panel__field">
                  <span className="floating-panel__field-label">Compound ID</span>
                  <CompoundIdInput
                    initialValue={modalCompoundIdValue}
                    rowIndex={modalRowIndex!}
                    onSave={handleCompoundIdSave}
                  />
                  <div className="inline-editor__actions floating-panel__structure-actions">
                    <button
                      className="inline-editor__icon-btn"
                      type="button"
                      onClick={() => modalInlineEditorRef.current?.save()}
                      title="Save structure"
                      aria-label="Save structure"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M5 12.5 9.2 17 19 7" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>
                    <button
                      className="inline-editor__icon-btn"
                      type="button"
                      onClick={() => modalInlineEditorRef.current?.reset()}
                      title="Reset structure"
                      aria-label="Reset structure"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M6.5 9A6.5 6.5 0 1 1 5 13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                        <path d="M6.5 5v4h4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>
                  </div>
                </label>
                <StructureEditorInline
                  ref={modalInlineEditorRef}
                  smiles={modalRowCanEdit ? String(editedStructures[modalRowIndex!]?.SMILES || '') : ''}
                  molblock={modalRowCanEdit ? getMolblockValue(editedStructures[modalRowIndex!]) : ''}
                  onSave={handleInlineStructureSave}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {showScrollTop && (
        <button type="button" className="scroll-top" onClick={scrollToTop} aria-label="Back to top">
          <svg className="scroll-top__icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6.5 13.5 12 8l5.5 5.5" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M6.5 18 12 12.5 17.5 18" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" opacity="0.72" />
          </svg>
        </button>
      )}

      <StructureEditorModal
        open={editorState.open}
        initialSmiles={editorState.smiles}
        initialMolblock={editorState.molblock}
        onCancel={handleStructureEditorCancel}
        proposalName={editorState.rowIndex !== null && !editedStructures[editorState.rowIndex]?.SMILES && editedStructures[editorState.rowIndex]?.NAME_STRUCTURE_CANDIDATE ? String(editedStructures[editorState.rowIndex]?.COMPOUND_NAME || '') : undefined}
        sourceImage={editorState.rowIndex !== null ? structureRows.find((row) => row.index === editorState.rowIndex)?.segmentImage ?? undefined : undefined}
        compoundLabel={editorState.rowIndex !== null ? String(editedStructures[editorState.rowIndex]?.COMPOUND_ID || '') : ''}
        onSave={handleStructureEditorSave}
      />
    </div>
  );
};

export default App;

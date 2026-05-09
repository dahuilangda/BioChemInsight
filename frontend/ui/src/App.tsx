import React from 'react';
import {
  fetchArtifact,
  fetchPdfInfo,
  fetchPdfPage,
  fetchTask,
  fetchTaskAssays,
  fetchTaskStructures,
  fetchTasks,
  getTaskDownloadUrl,
  queueAutoDetectTask,
  queueAssayTask,
  queueFullPipelineTask,
  queueStructureTask,
  renderSmiles,
  updateTaskStructures,
  uploadPdf,
  reparseStructure,
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
import StructureEditorInline from './components/StructureEditorInline';

type StepId = 1 | 2 | 3 | 4;

type StepKey = 'upload' | 'structures' | 'bioactivity' | 'review';
type StructureFilterStrictness = 'strict' | 'balanced' | 'permissive';

const stepDefinitions: Array<{ id: StepId; label: string; icon: StepKey }> = [
  { id: 1, label: 'Upload PDF', icon: 'upload' },
  { id: 2, label: 'Structures', icon: 'structures' },
  { id: 3, label: 'Bioactivity', icon: 'bioactivity' },
  { id: 4, label: 'Review', icon: 'review' },
];

const preferredColumnOrder = ['COMPOUND_ID', 'SMILES', 'source_pages', 'IMAGE_FILE', 'SEGMENT_FILE'];
const markdownImageRegex = /!\[[^\]]*\]\((data:image\/[^)]+)\)/i;
const isDataImage = (value: string) => value.startsWith('data:image');
const MAX_ARTIFACT_CACHE_ENTRIES = 80;
const STRUCTURE_IGNORED_COLUMNS = new Set([
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
  'RAW_COMPOUND_ID',
  'CANONICAL_COMPOUND_ID',
  'ALIAS_RESOLUTION_SOURCE',
]);
const STRUCTURE_COLUMN_LABELS: Record<string, string> = {
  COMPOUND_ID: 'Compound ID',
};
const FILTERED_PREFERRED_COLUMNS = [
  'source_pages',
  'PAGE_NUM',
];

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
      // 防止事件冒泡
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
    full_pipeline: 'Full pipeline',
    merge: 'Merge',
  };
  return labels[type] ?? type.replace(/_/g, ' ');
}

function formatTaskTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

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

function getRecordMergeKey(record: Record<string, unknown>, fallback: number): string {
  return [
    record.COMPOUND_ID ?? '',
    record.PAGE_NUM ?? record.page_num ?? record.page ?? '',
    record.SMILES ?? '',
    record.SEGMENT_FILE ?? record.Segment ?? '',
    fallback,
  ].join('|');
}

function mergeStructureRecordLists(base: StructureRecord[], extras: StructureRecord[]): StructureRecord[] {
  const merged = [...base.map((record) => ({ ...record }))];
  const seen = new Set(merged.map((record, index) => getRecordMergeKey(record as Record<string, unknown>, index)));
  extras.forEach((record, index) => {
    const cloned = { ...record };
    const key = getRecordMergeKey(cloned as Record<string, unknown>, base.length + index);
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

function withBoundedArtifactCache(
  previous: Record<string, string>,
  key: string,
  value: string,
): Record<string, string> {
  if (previous[key]) return previous;
  const entries = Object.entries(previous);
  const nextEntries =
    entries.length >= MAX_ARTIFACT_CACHE_ENTRIES
      ? entries.slice(entries.length - MAX_ARTIFACT_CACHE_ENTRIES + 1)
      : entries;
  return { ...Object.fromEntries(nextEntries), [key]: value };
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

  const autoSaveTimerRef = React.useRef<number | null>(null);
  const editedStructuresRef = React.useRef<StructureRecord[]>([]);

  const [assayRecords, setAssayRecords] = React.useState<AssayRecord[]>([]);
  const [assayNames, setAssayNames] = React.useState<string[]>([]);
  const [assayNameDraft, setAssayNameDraft] = React.useState('');

  const [isUploading, setIsUploading] = React.useState(false);
  const [uploadProgress, setUploadProgress] = React.useState<number | null>(null);
  const [magnifyCache, setMagnifyCache] = React.useState<Record<number, string>>({});
  const [isMagnifying, setIsMagnifying] = React.useState(false);
  const [magnifiedPage, setMagnifiedPage] = React.useState<number | null>(null);
  const [imageCache, setImageCache] = React.useState<Record<string, string>>({});
  const [loadingArtifacts, setLoadingArtifacts] = React.useState<Set<string>>(new Set());
  const [editorState, setEditorState] = React.useState<{
    open: boolean;
    rowIndex: number | null;
    smiles: string;
    molblock: string;
  }>({ open: false, rowIndex: null, smiles: '', molblock: '' });
  const [reparseState, setReparseState] = React.useState<{ rowIndex: number | null; engine: string | null }>({
    rowIndex: null,
    engine: null,
  });
  const [currentStep, setCurrentStep] = React.useState<StepId>(1);
  const autoAdvanceRef = React.useRef(false);
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
  const [modalArtifact, setModalArtifact] = React.useState<ArtifactPreview | null>(null);
  const [modalBox, setModalBox] = React.useState<number[] | null>(null);
  const [originalImageSize, setOriginalImageSize] = React.useState<{ width: number; height: number } | null>(null);
  const [boxStyle, setBoxStyle] = React.useState<React.CSSProperties>({});
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
  const [rowHeight, setRowHeight] = React.useState(160);
  const [modalEditorPosition, setModalEditorPosition] = React.useState<{ x: number; y: number }>({ x: 32, y: 32 });
  const modalEditorPanelRef = React.useRef<HTMLDivElement | null>(null);
  const modalEditorDragRef = React.useRef<{ offsetX: number; offsetY: number } | null>(null);
  const tableWrapperRef = React.useRef<HTMLDivElement | null>(null);
  const tableBodyRef = React.useRef<HTMLTableSectionElement | null>(null);
  const [visibleRowIndices, setVisibleRowIndices] = React.useState<Set<number>>(() => new Set());
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

  const structureDisplayColumns = React.useMemo(() => {
    const keys = new Set<string>();
    editedStructures.forEach((record) => {
      Object.keys(record).forEach((key) => {
        if (!STRUCTURE_IGNORED_COLUMNS.has(key)) {
          keys.add(key);
        }
      });
    });
    keys.delete('SMILES');
    const baseOrder = ['COMPOUND_ID'];
    const preferred = preferredColumnOrder.filter((col) => keys.has(col));
    const rest = Array.from(keys).filter((col) => !baseOrder.includes(col) && !preferred.includes(col));
    rest.sort();
    return Array.from(new Set([...baseOrder, ...preferred, ...rest]));
  }, [editedStructures]);
  const structureColumnsToRender = React.useMemo(
    () => structureDisplayColumns.filter((column) => column !== 'SMILES' && column !== 'COMPOUND_ID'),
    [structureDisplayColumns],
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
  const getMolblockValue = React.useCallback((record: StructureRecord) => {
    const raw =
      (record.MOLBLOCK as string | null | undefined) ??
      (record.molblock as string | null | undefined) ??
      (record.Molblock as string | null | undefined);
    return typeof raw === 'string' ? raw.trim() : '';
  }, []);
  const getStructurePreviewKey = React.useCallback((smilesValue: string, molblockValue: string) => {
    if (molblockValue) {
      return `molblock:${molblockValue}`;
    }
    return '';
  }, []);
  const structureRows = React.useMemo(
    () =>
      editedStructures.map((record, index) => {
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
        const segmentImage = extractImageSource(record.Segment) ?? extractImageSource(record.SEGMENT_FILE);
        // 使用COMPOUND_ID作为key来匹配活性数据
        return {
          id: (record.COMPOUND_ID ?? '').toString(),
          record,
          index,
          structureImage,
          segmentImage,
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
            segmentImage ||
            (typeof record.Segment === 'string'
              ? record.Segment
              : typeof record.SEGMENT_FILE === 'string'
              ? record.SEGMENT_FILE
              : ''),
        };
      }),
    [editedStructures, extractImageSource, getMolblockValue, getStructurePreviewKey, structurePreviewCache],
  );
  const assayColumnNames = React.useMemo(() => {
    const columns = new Set<string>();
    assayRecords.forEach((record) => {
      Object.keys(record).forEach((key) => {
        if (key !== 'COMPOUND_ID' && record[key] !== undefined && record[key] !== null) {
          columns.add(key);
        }
      });
    });
    return Array.from(columns).sort();
  }, [assayRecords]);
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
  const filteredStructureColumns = React.useMemo(() => {
    const keys = new Set<string>();
    filteredStructures.forEach((record) => {
      Object.keys(record).forEach((key) => {
        if (
          ![
            'COMPOUND_ID',
            'SMILES',
            'MOLBLOCK',
            'FILTERED_OUT',
            'STRUCTURE_FILTER_RAW_RESPONSE',
            'STRUCTURE_TYPE',
            'STRUCTURE_FILTER_REASON',
            'STRUCTURE_FILTER_BORDER_SIDES',
            'STRUCTURE_FILTER_STRICTNESS',
            'IS_COMPLETE_COMPOUND',
            'RAW_COMPOUND_ID',
            'CANONICAL_COMPOUND_ID',
            'ALIAS_RESOLUTION_SOURCE',
            'BOX_COORDS_FILE',
            'IMAGE_FILE',
            'SEGMENT_FILE',
            'PAGE_IMAGE_FILE',
            'Segment',
            'Structure',
          ].includes(key)
        ) {
          keys.add(key);
        }
      });
    });
    const preferred = FILTERED_PREFERRED_COLUMNS.filter((col) => keys.has(col));
    const rest = Array.from(keys).filter((col) => !preferred.includes(col));
    rest.sort();
    return [...preferred, ...rest];
  }, [filteredStructures]);
  const filteredStructureRows = React.useMemo(
    () =>
      filteredStructures.map((record, index) => ({
        index,
        record,
        previewImage:
          extractImageSource(record.PAGE_IMAGE_FILE) ??
          extractImageSource(record.IMAGE_FILE) ??
          extractImageSource(record.SEGMENT_FILE) ??
          null,
        segmentImage:
          extractImageSource(record.SEGMENT_FILE) ??
          extractImageSource(record.Segment) ??
          null,
      })),
    [filteredStructures, extractImageSource],
  );
  const filteredStructureStats = React.useMemo(() => {
    const counts: Record<string, number> = {};
    filteredStructures.forEach((record) => {
      const key = formatCellValue(record.STRUCTURE_TYPE ?? 'unknown').trim() || 'unknown';
      counts[key] = (counts[key] ?? 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [filteredStructures]);
      const processedStructureRows = React.useMemo(() => {
    const rows = structureRows.map((row) => ({
      ...row,
      assayData: assayDataMap.get(row.id) ?? {},
    }));
    const usedIds = new Set(rows.map((row) => row.id));
    assayDataMap.forEach((assayData, id) => {
      if (id && !usedIds.has(id)) {
        rows.push({
          id,
          record: { COMPOUND_ID: id } as StructureRecord,
          index: rows.length,
          structureImage: null,
          segmentImage: null,
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
      // Remove secondary page info (source pages display)
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

  const saveStatusMeta = React.useMemo(() => {
    if (!structures.length) return null;
    switch (saveStatus) {
      case 'pending':
        return { label: 'Pending save…', modifier: 'table-actions__status--pending' };
      case 'saving':
        return { label: 'Saving changes…', modifier: 'table-actions__status--saving' };
      case 'saved':
        return { label: 'All changes saved', modifier: 'table-actions__status--saved' };
      case 'error':
        return { label: 'Auto-save failed. Try editing again.', modifier: 'table-actions__status--error' };
      default:
        return null;
    }
  }, [saveStatus, structures.length]);
  const tableStyle = React.useMemo(() => ({
    '--review-row-height': `${rowHeight}px`,
  }), [rowHeight]);

  React.useEffect(() => {
    if (processedStructureRows.length === 0) {
      setVisibleRowIndices(new Set());
      return;
    }
    setVisibleRowIndices((prev) => {
      if (prev.size) return prev;
      const next = new Set<number>();
      const initialCount = Math.min(12, processedStructureRows.length);
      for (let idx = 0; idx < initialCount; idx += 1) {
        next.add(idx);
      }
      return next;
    });
  }, [processedStructureRows.length]);

  React.useEffect(() => {
    const container = tableWrapperRef.current;
    const body = tableBodyRef.current;
    if (!container || !body) return undefined;

    const observer = new IntersectionObserver(
      (entries) => {
        setVisibleRowIndices((prev) => {
          let changed = false;
          const next = new Set(prev);
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const attr = entry.target.getAttribute('data-row-index');
            if (!attr) return;
            const index = Number(attr);
            if (Number.isNaN(index)) return;
            if (!next.has(index)) {
              next.add(index);
              changed = true;
            }
          });
          return changed ? next : prev;
        });
      },
      { root: container, rootMargin: '200px 0px' },
    );

    const rows = Array.from(body.querySelectorAll<HTMLTableRowElement>('tr[data-row-index]'));
    rows.forEach((row) => observer.observe(row));

    return () => observer.disconnect();
  }, [processedStructureRows]);

  React.useEffect(() => {
    if (modalArtifact && modalBox && originalImageSize && modalImageRef.current) {
      const renderedImage = modalImageRef.current;
      const scaleX = renderedImage.clientWidth / originalImageSize.width;
      const scaleY = renderedImage.clientHeight / originalImageSize.height;

      const [y1, x1, y2, x2] = modalBox;

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
  const showModalQuickEdit = Boolean(modalArtifact && modalRowCanEdit && modalArtifact.mime !== 'loading');

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
    
    // 更新图像位置
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
    const cacheSnapshot = structurePreviewCacheRef.current;
    const pending = new Map<string, { smiles: string; molblock: string; previewKey: string }>();
    editedStructures.forEach((record) => {
      const smilesValue = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
      const molblockValue = getMolblockValue(record);
      const previewKey = getStructurePreviewKey(smilesValue, molblockValue);
      if (!previewKey) return;
      if (Object.prototype.hasOwnProperty.call(cacheSnapshot, previewKey)) return;
      const hasImage =
        extractImageSource(record.Structure) ?? extractImageSource(record.IMAGE_FILE);
      if (hasImage) return;
      if (!pending.has(previewKey)) {
        pending.set(previewKey, {
          smiles: smilesValue,
          molblock: molblockValue,
          previewKey,
        });
      }
    });
    if (!pending.size) return () => undefined;

    let cancelled = false;
    const tasks = Array.from(pending.values()).map(async (payload) => {
      try {
        const image = await renderSmiles(payload.smiles, {
          width: 280,
          height: 220,
          molblock: payload.molblock,
        });
        if (cancelled) return;
        setStructurePreviewCache((prev) => {
          const next = { ...prev };
          if (payload.previewKey && !Object.prototype.hasOwnProperty.call(prev, payload.previewKey)) {
            next[payload.previewKey] = image || '';
          }
          return next;
        });
      } catch (error) {
        if (cancelled) return;
        console.warn('Failed to generate structure preview', error);
        setStructurePreviewCache((prev) => {
          const next = { ...prev };
          if (payload.previewKey && !Object.prototype.hasOwnProperty.call(prev, payload.previewKey)) {
            next[payload.previewKey] = '';
          }
          return next;
        });
      }
    });

    void Promise.all(tasks);

    return () => {
      cancelled = true;
    };
  }, [editedStructures, extractImageSource, getMolblockValue, getStructurePreviewKey]);
  const canViewResults = React.useMemo(
    () =>
      Boolean(
        structures.length > 0 ||
          filteredStructures.length > 0 ||
          assayRecords.length > 0 ||
          structureTask !== null ||
          assayTask !== null,
      ),
    [structures.length, filteredStructures.length, assayRecords.length, structureTask, assayTask],
  );
  const maxStep = React.useMemo<StepId>(() => {
    if (canViewResults) return 4;
    if (pdfInfo) return 3;
    return 1;
  }, [pdfInfo, canViewResults]);
  const handleStepNavigation = React.useCallback(
    (stepId: StepId) => {
      if (stepId <= maxStep) {
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

  const loadJobs = React.useCallback(async () => {
    try {
      setJobsLoading(true);
      const nextJobs = await fetchTasks(80);
      setJobsInfo(nextJobs);
    } catch (err) {
      console.warn('Failed to load jobs', err);
    } finally {
      setJobsLoading(false);
    }
  }, []);

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
    const signature = pagesToString(detectedPages);
    if (appliedStructureDetectedPagesRef.current === signature && structurePagesInput === signature) return;
    appliedStructureDetectedPagesRef.current = signature;
    setStructureSelection(new Set<number>(detectedPages));
    setStructurePagesInput(signature);
    lastStructurePageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, [structurePagesInput]);

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
    const signature = pagesToString(detectedPages);
    if (appliedAssayDetectedPagesRef.current === signature && assayPagesInput === signature) return;
    appliedAssayDetectedPagesRef.current = signature;
    setAssaySelection(new Set<number>(detectedPages));
    setAssayPagesInput(signature);
    lastAssayPageRef.current = detectedPages[detectedPages.length - 1] ?? null;
  }, [assayPagesInput]);

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
    setToast((prev) => prev ?? 'Structure extraction completed. Review selected pages or continue with bioactivity extraction.');
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
    setStructures([]);
    setEditedStructures([]);
    setFilteredStructures([]);
    editedStructuresRef.current = [];
    setSaveStatus('idle');
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    setAssayRecords([]);
    setAssayNames([]);
    setAssayNameDraft('');
    setMagnifyCache({});
    setMagnifiedPage(null);
    setImageCache({});
    setLoadingArtifacts(new Set());
    loadingArtifactsRef.current = new Set();
    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);
    autoAdvanceRef.current = false;
    structurePagesAutoAdvancedRef.current = false;
    assayPagesAutoAdvancedRef.current = false;
    pdfInitializedRef.current = false;
    requestedStepRef.current = null;
    setCurrentStep(1);
  }, []);

  const handleFileUpload: React.ChangeEventHandler<HTMLInputElement> = async (event) => {
    resetNotifications();
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setIsUploading(true);
      setUploadProgress(0);
      const response = await uploadPdf(file, (percent) => setUploadProgress(percent));
      setPdfInfo(response);
      clearWorkspace();
      setToast(`Upload complete: detected ${response.total_pages} page${response.total_pages === 1 ? '' : 's'}`);
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
    structurePreviewCacheRef.current = structurePreviewCache;
  }, [structurePreviewCache]);

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
      activePageFetchesRef.current += 1;
      void fetchPdfPage(pdf.pdf_id, page, { zoom: 1.6, maxWidth: 680 })
        .then((image) => {
          setPageImages((prev) => {
            const next = { ...prev, [page]: image };
            pageImagesRef.current = next;
            return next;
          });
        })
        .catch((err) => {
          setError((prevError) =>
            prevError ?? (err instanceof Error ? err.message : `Failed to load preview for page ${page}`),
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
    if (!pdfInfo || !pages.length) return;
    const initialCount = Math.min(8, pages.length);
    pages.slice(0, initialCount).forEach((page) => {
      loadPageImage(page, true);
    });
  }, [pdfInfo, pages, loadPageImage]);

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
    const focusPages = new Set<number>();
    structureSelection.forEach((page) => focusPages.add(page));
    assaySelection.forEach((page) => focusPages.add(page));
    focusPages.forEach((page) => {
      if (page > 0) {
        loadPageImage(page, true);
      }
    });
  }, [pdfInfo, structureSelection, assaySelection, loadPageImage]);

  React.useEffect(() => {
    if (currentStep !== 4) return;
    const pending = new Set<string>();
    const collectArtifacts = (record: StructureRecord) => {
      ['Structure', 'Segment', 'PAGE_IMAGE_FILE', 'IMAGE_FILE', 'SEGMENT_FILE', 'Image File', 'Segment File'].forEach(
        (key) => {
          const value = record[key as keyof StructureRecord];
          if (typeof value !== 'string') return;
          if (!value) return;
          if (isDataImage(value) || markdownImageRegex.test(value) || imageCache[value]) return;
          if (loadingArtifactsRef.current.has(value)) return;
          pending.add(value);
        },
      );
    };
    editedStructures.forEach((record, index) => {
      if (visibleRowIndices.has(index)) {
        collectArtifacts(record);
      }
    });
    filteredStructures.slice(0, 8).forEach(collectArtifacts);
    if (!pending.size) return;
    Array.from(pending)
      .slice(0, 8)
      .forEach((path) => {
      setLoadingArtifacts((prev) => {
        if (prev.has(path)) return prev;
        const next = new Set(prev);
        next.add(path);
        return next;
      });
      fetchArtifact(path)
        .then((artifact) => {
          const dataUri = `data:${artifact.mime_type};base64,${artifact.content}`;
          setImageCache((prev) => {
            return withBoundedArtifactCache(prev, path, dataUri);
          });
        })
        .catch(() => {
          /* ignore individual fetch errors */
        })
        .finally(() => {
          setLoadingArtifacts((prev) => {
            if (!prev.has(path)) return prev;
            const next = new Set(prev);
            next.delete(path);
            return next;
          });
        });
      });
  }, [currentStep, editedStructures, filteredStructures, imageCache, visibleRowIndices]);

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
    if (assayRecords.length > 0 && currentStep < 4 && !assayAutoAdvancedRef.current) {
      assayAutoAdvancedRef.current = true;
      setCurrentStep(4);
    }
    // Reset the flag when assay records are cleared
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
        `Restored ${storedPages.length} queued structure page${storedPages.length === 1 ? '' : 's'} from this workspace.`,
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
        `Restored ${storedPages.length} queued bioactivity page${storedPages.length === 1 ? '' : 's'} from this workspace.`,
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
          ? `Selection updated (${delta}). Added page${added.length === 1 ? '' : 's'} will be processed automatically after the current structure task finishes.`
          : `Selection updated (${delta}). The current task is already running; click “Run structure extraction” after it finishes if you want to exclude pages already submitted.`,
      );
      return;
    }
    if (structureTask.status === 'completed') {
      setStructureSelectionFeedback(
        `Selection updated (${delta}). Click “Run structure extraction” to process the updated page set.`,
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
          ? `Selection updated (${delta}). Added page${added.length === 1 ? '' : 's'} will be processed automatically after the current bioactivity task finishes.`
          : `Selection updated (${delta}). The current task is already running; click “Run bioactivity extraction” after it finishes if you want to exclude pages already submitted.`,
      );
      return;
    }
    if (assayTask.status === 'completed') {
      setAssaySelectionFeedback(
        `Selection updated (${delta}). Click “Run bioactivity extraction” to process the updated page set.`,
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

  const handleCardClick = (page: number, event: React.MouseEvent<HTMLDivElement>) => {
    resetNotifications();
    const shiftPressed = event.shiftKey;
    if (activeSelectionMode === 'assay') {
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
            role="button"
            tabIndex={0}
            data-page={page}
          >
            <div className="page-card__inner">
              <div className="page-card__magnify">
                <button
                  className="magnify-btn"
                  onClick={(event) => {
                    event.stopPropagation();
                    void handleMagnifyPage(page);
                  }}
                  type="button"
                  title="Open full-size image"
                >
                  🔍
                </button>
              </div>
              <div className="page-card__body">
                {imageLoaded ? (
                  <img src={`data:image/png;base64,${pageImages[page]}`} alt={`Page ${page}`} />
                ) : (
                  <div className="page-card__placeholder">Loading…</div>
                )}
              </div>
              <div className="page-card__footer">
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

  const handleMagnifyPage = async (page: number) => {
    resetNotifications();
    if (!pdfInfo) return;
    setMagnifiedPage(page);
    const cached = magnifyCache[page];
    if (cached) {
      setModalArtifact({
        path: `PDF Page ${page}`,
        mime: 'image/png',
        data: cached,
        rowIndex: null,
      });
      return;
    }

    setIsMagnifying(true);
    setModalArtifact({
      path: `PDF Page ${page}`,
      mime: 'loading',
      data: '',
      rowIndex: null,
    });

    try {
      const image = await fetchPdfPage(pdfInfo.pdf_id, page, { zoom: 2.0, maxWidth: 1024 });
      const dataUri = `data:image/png;base64,${image}`;
      setMagnifyCache((prev) => ({ ...prev, [page]: dataUri }));
      setModalArtifact({
        path: `PDF Page ${page}`,
        mime: 'image/png',
        data: dataUri,
        rowIndex: null,
      });
    } catch (err) {
      setModalArtifact(null);
      setError(err instanceof Error ? err.message : 'Unable to magnify the selected page');
    } finally {
      setIsMagnifying(false);
    }
  };

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
      setAssaySelectionFeedback('Assay names were updated. Running bioactivity tasks keep their submitted names; the update will apply to the next queued or manual run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names were updated. Click “Run bioactivity extraction” to re-run with the updated names.');
    }
  }, [assayAddonTask, assayTask]);

  const removeAssayName = (index: number) => {
    setAssayNames((prev) => prev.filter((_, idx) => idx !== index));
    if (isTaskInFlight(assayTask) || isTaskInFlight(assayAddonTask)) {
      setAssaySelectionFeedback('Assay names were updated. Running bioactivity tasks keep their submitted names; the update will apply to the next queued or manual run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names were updated. Click “Run bioactivity extraction” to re-run with the updated names.');
    }
  };

  const clearAssayNames = () => {
    setAssayNames([]);
    setAssayNameDraft('');
    if (isTaskInFlight(assayTask) || isTaskInFlight(assayAddonTask)) {
      setAssaySelectionFeedback('Assay names were cleared. Running bioactivity tasks keep their submitted names; the change applies to the next run.');
    } else if (assayTask?.status === 'completed') {
      setAssaySelectionFeedback('Assay names were cleared. Click “Run bioactivity extraction” to re-run with automatic or updated names.');
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

  const refreshStructureTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    setStructureTask(updated);
    applyDetectedStructurePages(updated.params?.detected_pages ?? updated.params?.pages);
    if (updated.status === 'completed') {
      const results = await fetchTaskStructures(taskId);
      const nextRecords = results.records.map((record) => ({ ...record }));
      const nextFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
      setStructures(results.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setFilteredStructures(nextFilteredRecords);
      setSaveStatus(results.records.length ? 'saved' : 'idle');
      setToast(
        `Structure extraction complete: ${results.records.length} record${
          results.records.length === 1 ? '' : 's'
        }${nextFilteredRecords.length ? `; filtered ${nextFilteredRecords.length} candidate${nextFilteredRecords.length === 1 ? '' : 's'}` : ''}. Continue in Step 3 to extract bioactivity.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Structure extraction task failed');
    }
  }, [applyDetectedStructurePages]);

  const refreshStructureAddonTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    setStructureAddonTask(updated);
    if (updated.status === 'completed') {
      const results = await fetchTaskStructures(taskId);
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
        `Additional structure pages complete: added ${addRecords.length} record${addRecords.length === 1 ? '' : 's'}.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Additional structure extraction task failed');
    }
  }, [filteredStructures, structureTask]);

  const refreshAssayTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    setAssayTask(updated);
    applyDetectedAssayPages(updated.params?.detected_pages ?? updated.params?.pages);
    applyDetectedAssayNames(updated.params?.detected_assay_names);
    if (updated.status === 'completed') {
      const results = await fetchTaskAssays(taskId);
      setAssayRecords(results.records);
      setToast(
        `Bioactivity extraction complete: ${results.records.length} record${
          results.records.length === 1 ? '' : 's'
        }. Review everything in Step 4.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Bioactivity extraction task failed');
    }
  }, [applyDetectedAssayNames, applyDetectedAssayPages]);

  const refreshAssayAddonTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    setAssayAddonTask(updated);
    if (updated.status === 'completed') {
      const results = await fetchTaskAssays(taskId);
      setAssayRecords((prev) => mergeAssayRecordLists(prev, results.records));
      const completedPages = getTaskSelectedPages(updated);
      const completedSet = new Set(
        completedPages.length ? completedPages : assayAddonSubmittedPagesRef.current,
      );
      setPendingAssayAddonPages((prev) => prev.filter((page) => !completedSet.has(page)));
      assayAddonSubmittedPagesRef.current = [];
      setAssaySelectionFeedback(null);
      setToast(
        `Additional bioactivity pages complete: merged ${results.records.length} record${results.records.length === 1 ? '' : 's'}.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Additional bioactivity extraction task failed');
    }
  }, []);

  const refreshAutoDetectTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
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
      setToast('Automatic detection is ready. Review Step 2 and Step 3 before extraction.');
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Automatic detection task failed');
    }
  }, [applyDetectedAssayNames, applyPlannedAssayPages, applyPlannedStructurePages]);

  const refreshFullPipelineTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
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
      if (pages.length > 0 && !assayPagesAutoAdvancedRef.current) {
        assayPagesAutoAdvancedRef.current = true;
        setCurrentStep((prev) => (prev < 3 ? 3 : prev));
      }
    }
    if (params?.detected_assay_names) {
      applyDetectedAssayNames(params?.detected_assay_names);
    }

    if (updated.status === 'completed') {
      const structData = params?.structure_records;
      if (Array.isArray(structData) && structData.length) {
        setStructures(structData as StructureRecord[]);
        setEditedStructures(structData as StructureRecord[]);
        setFilteredStructures([]);
      }
      const assayData = params?.assay_records;
      if (Array.isArray(assayData) && assayData.length) {
        setAssayRecords(assayData as AssayRecord[]);
      }
      setToast('Full pipeline completed. Review the results in Step 4.');
      setCurrentStep(4);
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Full pipeline task failed');
    }
  }, [applyPlannedStructurePages, applyPlannedAssayPages, applyDetectedAssayNames]);

  const handleOpenJob = React.useCallback(async (job: TaskStatus) => {
    resetNotifications();
    try {
      const updated = await fetchTask(job.task_id);
      if (updated.type === 'auto_detect_plan') {
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
        setToast(updated.status === 'completed' ? 'Detection job opened.' : 'Detection job is now tracked in this workspace.');
        return;
      }

      if (updated.type === 'structure_extraction') {
        setStructureTask(updated);
        setStructureFilterStrictness(coerceStructureFilterStrictness(updated.params?.structure_filter_strictness));
        setAutoDetectStructurePages(coerceAutoDetectionFlag(updated.params?.auto_detect_pages));
        applyDetectedStructurePages(updated.params?.detected_pages ?? updated.params?.pages);
        if (updated.status === 'completed') {
          const results = await fetchTaskStructures(updated.task_id);
          const nextRecords = results.records.map((record) => ({ ...record }));
          const nextFilteredRecords = (results.filtered_records ?? []).map((record) => ({ ...record }));
          setStructures(results.records);
          editedStructuresRef.current = nextRecords;
          setEditedStructures(nextRecords);
          setFilteredStructures(nextFilteredRecords);
          setSaveStatus(results.records.length ? 'saved' : 'idle');
          setCurrentStep(4);
          setToast('Structure job result opened.');
        } else {
          setCurrentStep(2);
          setToast('Structure job is now tracked in this workspace.');
        }
        return;
      }

      if (updated.type === 'bioactivity_extraction') {
        setAssayTask(updated);
        setAutoDetectAssayPages(coerceAutoDetectionFlag(updated.params?.auto_detect_pages));
        setAutoDetectAssayNames(coerceAutoDetectionFlag(updated.params?.auto_detect_assay_names));
        applyDetectedAssayPages(updated.params?.detected_pages ?? updated.params?.pages);
        applyDetectedAssayNames(updated.params?.detected_assay_names);
        if (updated.status === 'completed') {
          const results = await fetchTaskAssays(updated.task_id);
          setAssayRecords(results.records);
          setCurrentStep(4);
          setToast('Bioactivity job result opened.');
        } else {
          setCurrentStep(3);
          setToast('Bioactivity job is now tracked in this workspace.');
        }
        return;
      }

      if (updated.type === 'full_pipeline') {
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
          const structData = params?.structure_records;
          if (Array.isArray(structData)) {
            const nextStructures = structData as StructureRecord[];
            setStructures(nextStructures);
            editedStructuresRef.current = nextStructures.map((record) => ({ ...record }));
            setEditedStructures(nextStructures.map((record) => ({ ...record })));
            setFilteredStructures([]);
          }
          const assayData = params?.assay_records;
          if (Array.isArray(assayData)) {
            setAssayRecords(assayData as AssayRecord[]);
          }
          setCurrentStep(4);
          setToast('Full pipeline job result opened.');
        } else {
          setCurrentStep(updated.progress >= 0.55 ? 3 : 2);
          setToast('Full pipeline job is now tracked in this workspace.');
        }
        return;
      }

      setToast('This job type can be monitored here, but has no dedicated result view yet.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open this job.');
    }
  }, [
    applyDetectedAssayNames,
    applyDetectedAssayPages,
    applyDetectedStructurePages,
    applyPlannedAssayPages,
    applyPlannedStructurePages,
    coerceAutoDetectionFlag,
    coerceStructureFilterStrictness,
  ]);

  const submitAssayTask = React.useCallback(
    async (request: AssayTaskRequest) => {
      try {
        setIsAssaySubmitting(true);
        const taskStatus = await queueAssayTask(request);
        setAssayTask(taskStatus);
        setAssaySelectionFeedback(null);
        setAssayRecords([]);
        setToast('Bioactivity extraction task submitted');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to submit bioactivity extraction task');
      } finally {
        setIsAssaySubmitting(false);
      }
    },
    [queueAssayTask],
  );

  const submitStructureAddonTask = React.useCallback(
    async (pagesToRun: number[]) => {
      if (!pdfInfo || !pagesToRun.length || isTaskInFlight(structureAddonTask)) return;
      const pagesSnapshot = Array.from(new Set(pagesToRun)).sort((a, b) => a - b);
      const pageString = pagesToString(pagesSnapshot);
      try {
        const taskStatus = await queueStructureTask({
          pdf_id: pdfInfo.pdf_id,
          pages: pageString,
          auto_detect_pages: false,
          structure_filter_strictness: structureFilterStrictness,
        });
        structureAddonSubmittedPagesRef.current = pagesSnapshot;
        setStructureAddonTask(taskStatus);
        setStructureSelectionFeedback(`Automatically queued additional structure extraction for pages ${pageString}.`);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to submit additional structure extraction');
      }
    },
    [pdfInfo, structureAddonTask, structureFilterStrictness],
  );

  const submitAssayAddonTask = React.useCallback(
    async (pagesToRun: number[]) => {
      if (!pdfInfo || !pagesToRun.length || isTaskInFlight(assayAddonTask)) return;
      const pagesSnapshot = Array.from(new Set(pagesToRun)).sort((a, b) => a - b);
      const pageString = pagesToString(pagesSnapshot);
      const namesList = assayNames.length ? assayNames : parseStringListParam(assayTask?.params?.detected_assay_names);
      if (!namesList.length) {
        setAssaySelectionFeedback('Additional bioactivity pages are queued, but assay names are needed before they can run.');
        return;
      }
      try {
        const taskStatus = await queueAssayTask({
          pdf_id: pdfInfo.pdf_id,
          pages: pageString,
          assay_names: namesList,
          auto_detect_pages: false,
          auto_detect_assay_names: false,
          structure_task_id: structureTask?.status === 'completed' ? structureTask.task_id : undefined,
        });
        assayAddonSubmittedPagesRef.current = pagesSnapshot;
        setAssayAddonTask(taskStatus);
        setAssaySelectionFeedback(`Automatically queued additional bioactivity extraction for pages ${pageString}.`);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to submit additional bioactivity extraction');
      }
    },
    [assayAddonTask, assayNames, assayTask, pdfInfo, structureTask],
  );

  React.useEffect(() => {
    if (!autoDetectTask || autoDetectTask.status === 'completed' || autoDetectTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshAutoDetectTask(autoDetectTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setAutoDetectTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh automatic detection task');
      });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [autoDetectTask, refreshAutoDetectTask]);

  React.useEffect(() => {
    if (!fullPipelineTask || fullPipelineTask.status === 'completed' || fullPipelineTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshFullPipelineTask(fullPipelineTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setFullPipelineTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh full pipeline task');
      });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [fullPipelineTask, refreshFullPipelineTask]);

  React.useEffect(() => {
    if (!structureTask || structureTask.status === 'completed' || structureTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshStructureTask(structureTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setStructureTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh structure task');
      });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [structureTask, refreshStructureTask]);

  React.useEffect(() => {
    if (!structureAddonTask || structureAddonTask.status === 'completed' || structureAddonTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshStructureAddonTask(structureAddonTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setStructureAddonTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh additional structure task');
      });
    }, 1800);
    return () => window.clearInterval(interval);
  }, [structureAddonTask, refreshStructureAddonTask]);

  React.useEffect(() => {
    if (!structureTask || structureTask.status !== 'completed') return;
    if (!pendingStructureAddonPages.length) return;
    if (isTaskInFlight(structureAddonTask)) return;
    void submitStructureAddonTask(pendingStructureAddonPages);
  }, [pendingStructureAddonPages, structureAddonTask, structureTask, submitStructureAddonTask]);

  React.useEffect(() => {
    if (!structureTask || structureTask.status !== 'completed') return;
    if (!pendingAssayRequestRef.current) return;
    const pendingRequest: AssayTaskRequest = {
      ...pendingAssayRequestRef.current,
      structure_task_id: structureTask.task_id,
    };
    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);
    void submitAssayTask(pendingRequest);
  }, [structureTask, submitAssayTask]);

  React.useEffect(() => {
    if (
      !assayTask ||
      assayTask.status === 'completed' ||
      assayTask.status === 'failed' ||
      assayTask.task_id.startsWith('pending-')
    ) {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshAssayTask(assayTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setAssayTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh bioactivity task');
      });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [assayTask, refreshAssayTask]);

  React.useEffect(() => {
    if (!assayAddonTask || assayAddonTask.status === 'completed' || assayAddonTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshAssayAddonTask(assayAddonTask.task_id).catch((err) => {
        if (isHttpNotFound(err)) {
          setAssayAddonTask(null);
          return;
        }
        setError(err instanceof Error ? err.message : 'Unable to refresh additional bioactivity task');
      });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [assayAddonTask, refreshAssayAddonTask]);

  React.useEffect(() => {
    if (!assayTask || assayTask.status !== 'completed') return;
    if (!pendingAssayAddonPages.length) return;
    if (isTaskInFlight(assayAddonTask)) return;
    void submitAssayAddonTask(pendingAssayAddonPages);
  }, [assayAddonTask, assayTask, pendingAssayAddonPages, submitAssayAddonTask]);

  const handleStartStructureExtraction = async () => {
    resetNotifications();
    if (!pdfInfo) {
      setError('Upload a PDF before starting structure extraction.');
      return;
    }
    const pagesSelected = Array.from(structureSelection).sort((a, b) => a - b);
    const useSelectedPages = pagesSelected.length > 0;
    const useAutomaticPages = !useSelectedPages && autoDetectStructurePages;
    if (!useSelectedPages && !useAutomaticPages) {
      setError('Select structure pages or enable automatic page detection in Step 1.');
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
      });
      setStructureTask(taskStatus);
      setStructureAddonTask(null);
      setPendingStructureAddonPages([]);
      structureAddonSubmittedPagesRef.current = [];
      setStructureSelectionFeedback(null);
      setStructures([]);
      setEditedStructures([]);
      setFilteredStructures([]);
      editedStructuresRef.current = [];
      setSaveStatus('idle');
      setToast('Structure extraction task submitted');
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
      setError('Upload a PDF before starting bioactivity extraction.');
      return;
    }
    const pagesSelected = Array.from(assaySelection).sort((a, b) => a - b);
    const useSelectedPages = pagesSelected.length > 0;
    const useAutomaticPages = !useSelectedPages && autoDetectAssayPages;
    if (!useSelectedPages && !useAutomaticPages) {
      setError('Select bioactivity pages or enable automatic page detection in Step 1.');
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
      setError('Add at least one assay name or enable automatic assay-name detection in Step 1.');
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
      pendingAssayRequestRef.current = requestBase;
      setIsAssayWaitingForStructures(true);
      const now = new Date().toISOString();
      const pendingParams: Record<string, unknown> = { ...requestBase };
      setAssayTask({
        task_id: `pending-${structureTask.task_id}`,
        type: 'assay',
        status: 'pending',
        progress: 0,
        message: 'Waiting for structure extraction to finish…',
        pdf_id: pdfInfo.pdf_id,
        params: pendingParams,
        created_at: now,
        updated_at: now,
      });
      setToast('Bioactivity extraction queued. Waiting for structure extraction to finish.');
      return;
    }

    pendingAssayRequestRef.current = null;
    setIsAssayWaitingForStructures(false);
    setAssayAddonTask(null);
    setPendingAssayAddonPages([]);
    assayAddonSubmittedPagesRef.current = [];
    setAssaySelectionFeedback(null);
    const finalRequest: AssayTaskRequest = {
      ...requestBase,
      structure_task_id:
        structureTask && structureTask.status === 'completed' ? structureTask.task_id : undefined,
    };
    await submitAssayTask(finalRequest);
  };

  const handleStartAutomaticExtraction = async () => {
    resetNotifications();
    const runId = automaticExtractionRunRef.current + 1;
    automaticExtractionRunRef.current = runId;
    if (!pdfInfo) {
      setError('Upload a PDF before starting extraction.');
      return;
    }
    if (!pdfReadyForDetection) {
      setError('Wait for the PDF preview to finish loading before starting detection.');
      return;
    }

    if (!autoDetectStructurePages && !autoDetectAssayPages && !autoDetectAssayNames) {
      setError('Enable at least one automatic detection target.');
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
      setToast('Automatic detection started. Extraction will wait for your review.');
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
    if (!pdfInfo) { setError('Upload a PDF before starting detection.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for the PDF preview to finish loading.'); return; }
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
    if (!pdfInfo) { setError('Upload a PDF before starting detection.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for the PDF preview to finish loading.'); return; }
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
    if (!pdfInfo) { setError('Upload a PDF before starting.'); return; }
    if (!pdfReadyForDetection) { setError('Wait for the PDF preview to finish loading.'); return; }
    try {
      setIsFullPipelineSubmitting(true);
      structurePagesAutoAdvancedRef.current = false;
      assayPagesAutoAdvancedRef.current = false;
      const request: FullPipelineRequest = {
        pdf_id: pdfInfo.pdf_id,
        structure_filter_strictness: structureFilterStrictness,
      };
      const taskStatus = await queueFullPipelineTask(request);
      setFullPipelineTask(taskStatus);
      setToast('Full auto pipeline started. This will run detection, structure extraction, and bioactivity extraction.');
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
    void loadJobs();
    const timer = window.setInterval(() => {
      void loadJobs();
    }, jobsOpen ? 2500 : 5000);
    return () => window.clearInterval(timer);
  }, [automaticExtractionActive, jobsOpen, loadJobs]);

  const handleCancelAutomaticExtraction = React.useCallback(() => {
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
    setFullPipelineTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setAutoDetectTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setStructureAddonTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setAssayAddonTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setStructureTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setAssayTask((current) =>
      current && (current.status === 'running' || current.status === 'pending') ? null : current,
    );
    setToast('Extraction canceled in this workspace.');
  }, []);

  const performAutoSave = React.useCallback(async () => {
    if (!structureTask || structureTask.status !== 'completed') {
      setSaveStatus('idle');
      return;
    }
    const payload = editedStructuresRef.current;
    if (!payload.length) {
      setSaveStatus('saved');
      return;
    }
    // Don't set saving status here since it's already set in handleCompoundIdSave
    try {
      const response = await updateTaskStructures(structureTask.task_id, payload);
      const nextRecords = response.records.map((row) => ({ ...row }));
      const nextFilteredRecords = (response.filtered_records ?? []).map((row) => ({ ...row }));
      setStructures(response.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setFilteredStructures(nextFilteredRecords);
      setSaveStatus('saved');
    } catch (err) {
      console.error('Failed to auto-save structures', err);
      setSaveStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to save structures');
    }
  }, [structureTask]);

  const scheduleAutoSave = React.useCallback(() => {
    if (!structureTask || structureTask.status !== 'completed') return;
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
    }
    setSaveStatus((prev) => (prev === 'saving' ? 'saving' : 'pending'));
    autoSaveTimerRef.current = window.setTimeout(() => {
      autoSaveTimerRef.current = null;
      void performAutoSave();
    }, 800);
  }, [structureTask, performAutoSave]);

  const handleCellChange = (rowIndex: number, column: string, value: string) => {
    setEditedStructures((prev) => {
      const next = prev.map((row, idx) => (idx === rowIndex ? { ...row, [column]: value } : row));
      return next;
    });
    
    // For Compound ID changes, we don't auto-save immediately
    // Instead, we'll save when the user presses Enter or the input loses focus
    if (column !== 'COMPOUND_ID') {
      scheduleAutoSave();
    }
  };

  const handleCompoundIdSave = (rowIndex: number, value: string) => {
    // Update the state first and immediately reflect the change in UI
    const updatedStructures = editedStructures.map((row, idx) => 
      idx === rowIndex ? { ...row, COMPOUND_ID: value } : row
    );
    setEditedStructures(updatedStructures);
    editedStructuresRef.current = updatedStructures;
    
    // Clear any pending auto-save timer
    if (autoSaveTimerRef.current) {
      window.clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    
    // Set save status to saving immediately
    setSaveStatus('saving');
    
    // Perform save in background without blocking UI
    void performAutoSave();
  };

  const handleDeleteStructureRow = (rowIndex: number) => {
    if (rowIndex < 0 || rowIndex >= editedStructuresRef.current.length) return;
    if (!window.confirm('Delete this structure record?')) return;

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

  const handleOpenStructureEditor = (rowIndex: number) => {
    const record = editedStructures[rowIndex];
    const smiles = typeof record.SMILES === 'string' ? record.SMILES : '';
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
        const updated: StructureRecord = { ...row, SMILES: smiles, MOLBLOCK: molblockValue };
        // 只更新SMILES字段，不修改原始的Structure和IMAGE_FILE字段
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

  const handleReparseStructure = async (index: number, engine: string) => {
    const record = editedStructures[index];
    if (!record || !pdfInfo) return;

    // 获取页面号和段落索引
    const pageNum = typeof record.PAGE_NUM === 'number' ? record.PAGE_NUM : 
                   typeof record.page_num === 'number' ? record.page_num : 
                   typeof record.page === 'number' ? record.page : 0;
    
    // 获取段落文件路径
    const segmentFile = typeof record.SEGMENT_FILE === 'string' ? record.SEGMENT_FILE : 
                       typeof record.Segment === 'string' ? record.Segment : 
                       typeof record['Segment File'] === 'string' ? record['Segment File'] : '';

    if (!pageNum) {
      setError('Unable to determine the page for this structure');
      return;
    }

    try {
      setReparseState({ rowIndex: index, engine });
      
      // 调用重新解析API
      const result = await reparseStructure({
        pdf_id: pdfInfo.pdf_id,
        page_num: pageNum,
        segment_idx: index, // 这里可能需要根据实际情况调整
        engine: engine,
        segment_file: segmentFile
      });

      // 更新SMILES值
      const molblockValue = typeof result.molblock === 'string' ? result.molblock : '';
      setEditedStructures((prev) =>
        prev.map((row, idx) => {
          if (idx !== index) return row;
          return { ...row, SMILES: result.smiles, MOLBLOCK: molblockValue };
        }),
      );

      const previewKey = getStructurePreviewKey(result.smiles ?? '', molblockValue);
      if (previewKey) {
        try {
          const image = await renderSmiles(result.smiles ?? '', {
            width: 280,
            height: 220,
            molblock: molblockValue,
          });
          if (image) {
            setStructurePreviewCache((prev) => {
              const next = { ...prev };
              if (!Object.prototype.hasOwnProperty.call(prev, previewKey)) {
                next[previewKey] = image;
              }
              return next;
            });
            setImageCache((prev) => ({ ...prev, [image]: image }));
          }
        } catch (err) {
          console.warn('Failed to generate structure preview', err);
        }
      }

      setReparseState({ rowIndex: null, engine: null });
      scheduleAutoSave();
      setToast(`Structure re-parsed with ${engine}`);
    } catch (err) {
      setReparseState({ rowIndex: null, engine: null });
      setError(err instanceof Error ? err.message : 'Failed to re-parse structure');
    }
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
        const updated: StructureRecord = { ...row, SMILES: smiles, MOLBLOCK: molblockValue };
        // 只更新SMILES字段，不修改原始的Structure和IMAGE_FILE字段
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
    setModalBox(null);
    setOriginalImageSize(null);
    if (typeof source !== 'string' || !source) return;

    if (options?.rowIndex !== null && options?.rowIndex !== undefined) {
      const record = editedStructures[options.rowIndex];
      const boxCoordsFile = record?.BOX_COORDS_FILE as string | undefined;
      if (boxCoordsFile) {
        try {
          const artifact = await fetchArtifact(boxCoordsFile);
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
      const img = new Image();
      img.onload = () => {
        setOriginalImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = dataUri;

      setModalArtifact({
        path: pathLabel ?? 'Preview',
        mime: 'image/png',
        data: dataUri,
        rowIndex: options?.rowIndex ?? null,
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
    });

    try {
      const artifact = await fetchArtifact(source);
      const dataUri = `data:${artifact.mime_type};base64,${artifact.content}`;
      setImageCache((prev) => ({ ...prev, [source]: dataUri }));
      showDataImage(dataUri, source);
    } catch (err) {
      setModalArtifact(null);
      setError(err instanceof Error ? err.message : 'Unable to load image');
    }
  };

  const closeModal = () => {
    setModalArtifact(null);
    setModalBox(null);
    setOriginalImageSize(null);
    setBoxStyle({});
  };

  const downloadStructuresCsv = () => {
    if (!structureTask || structureTask.status !== 'completed') return;
    window.open(getTaskDownloadUrl(structureTask.task_id), '_blank');
  };

  const downloadAssayCsv = () => {
    if (!assayTask || assayTask.status !== 'completed') return;
    window.open(getTaskDownloadUrl(assayTask.task_id), '_blank');
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
          setToast((prev) => prev ?? `Loaded PDF: ${info.filename}`);
        })
        .catch(() => setError('Could not restore the PDF. Please upload it again.'));
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
              setFilteredStructures(nextFilteredRecords);
              setSaveStatus(results.records.length ? 'saved' : 'idle');
              if (results.records.length > 0) {
                autoAdvanceRef.current = true;
                setCurrentStep((prev) => (prev < 4 ? 4 : prev));
              }
            });
          }
          return undefined;
        })
        .catch(() => setError('Could not restore the structure task status.'));
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
          message: 'Waiting for structure extraction to finish…',
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
            return fetchTaskAssays(assayTaskId).then((results) => {
              setAssayRecords(results.records);
              if (results.records.length > 0) {
                setCurrentStep((prev) => (prev < 4 ? 4 : prev));
              }
            });
          }
          return undefined;
        })
        .catch(() => setError('Could not restore the bioactivity task status.'));
    }

    const structureAddonTaskId = params.get('structureAddonTask');
    if (isUsableId(structureAddonTaskId)) {
      fetchTask(structureAddonTaskId)
        .then((task) => {
          setStructureAddonTask(task);
          structureAddonSubmittedPagesRef.current = getTaskSelectedPages(task);
        })
        .catch(() => setError('Could not restore the additional structure task status.'));
    }

    const assayAddonTaskId = params.get('assayAddonTask');
    if (isUsableId(assayAddonTaskId)) {
      fetchTask(assayAddonTaskId)
        .then((task) => {
          setAssayAddonTask(task);
          assayAddonSubmittedPagesRef.current = getTaskSelectedPages(task);
        })
        .catch(() => setError('Could not restore the additional bioactivity task status.'));
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
        .catch(() => setError('Could not restore the automatic detection task status.'));
    }

    const fullPipelineTaskId = params.get('fullPipelineTask');
    if (isUsableId(fullPipelineTaskId)) {
      fetchTask(fullPipelineTaskId)
        .then((task) => {
          setFullPipelineTask(task);
          const taskParams = task.params as Record<string, unknown>;
          if (Object.prototype.hasOwnProperty.call(taskParams ?? {}, 'detected_structure_pages')) {
            applyPlannedStructurePages(taskParams?.detected_structure_pages);
          }
          if (Object.prototype.hasOwnProperty.call(taskParams ?? {}, 'detected_assay_pages')) {
            applyPlannedAssayPages(taskParams?.detected_assay_pages);
          }
          applyDetectedAssayNames(taskParams?.detected_assay_names);
          if (task.status === 'completed') {
            const structData = taskParams?.structure_records;
            if (Array.isArray(structData)) {
              const nextStructures = structData as StructureRecord[];
              setStructures(nextStructures);
              editedStructuresRef.current = nextStructures.map((record) => ({ ...record }));
              setEditedStructures(nextStructures.map((record) => ({ ...record })));
              setFilteredStructures([]);
            }
            const assayData = taskParams?.assay_records;
            if (Array.isArray(assayData)) {
              setAssayRecords(assayData as AssayRecord[]);
            }
          }
        })
        .catch(() => setError('Could not restore the full pipeline task status.'));
    }
  }, [
    applyDetectedAssayNames,
    applyDetectedAssayPages,
    applyDetectedStructurePages,
    applyPlannedAssayPages,
    applyPlannedStructurePages,
    coerceAutoDetectionFlag,
    coerceStructureFilterStrictness,
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

  React.useEffect(() => {
    if (modalArtifact) {
      document.body.classList.add('no-scroll');
    } else {
      document.body.classList.remove('no-scroll');
      setIsMagnifying(false);
      setMagnifiedPage(null);
    }
    return () => {
      document.body.classList.remove('no-scroll');
    };
  }, [modalArtifact]);

  React.useEffect(() => {
    if (modalArtifact) {
      document.body.classList.add('no-scroll');
    } else {
      document.body.classList.remove('no-scroll');
      setIsMagnifying(false);
      setMagnifiedPage(null);
    }
    return () => {
      document.body.classList.remove('no-scroll');
    };
  }, [modalArtifact]);

  // 自动调整表格列宽
  React.useEffect(() => {
    const handleResize = () => {
      // 触发表格重新布局
      const table = document.querySelector('.review-table');
      if (table && table instanceof HTMLElement) {
        // 强制浏览器重新计算表格布局
        table.style.tableLayout = 'auto';
        // 短暂延迟后恢复自动布局
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

  return (
    <div className="container">
      <header className="app-header">
        <div className="app-header__copy">
          <h1>BioChemInsight Workbench</h1>
          <p>
            SAR curation for innovator patents—ingest the PDF once, resolve chemotypes, and review assay evidence in a single workspace.
          </p>
        </div>
        <button
          className={`jobs-button${activeJobCount > 0 ? ' jobs-button--active' : ''}`}
          type="button"
          onClick={() => {
            setJobsOpen((open) => !open);
            void loadJobs();
          }}
          aria-expanded={jobsOpen}
        >
          <span className="jobs-button__dot" aria-hidden="true" />
          Jobs
          {activeJobCount > 0 && <span className="jobs-button__count">{activeJobCount}</span>}
        </button>
      </header>

      {jobsOpen && (
        <section className="jobs-panel" aria-label="Background jobs">
          <div className="jobs-panel__header">
            <div>
              <h2>Jobs</h2>
              <p>
                Running {jobsInfo?.running_count ?? 0} / {jobsInfo?.max_concurrent_tasks ?? '-'} slots, pending {jobsInfo?.pending_count ?? 0}.
                Structure extraction runs {jobsInfo?.structure_task_concurrency ?? 1} at a time.
              </p>
            </div>
            <button className="small-btn" type="button" onClick={() => void loadJobs()} disabled={jobsLoading}>
              {jobsLoading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          <div className="jobs-list">
            {(jobsInfo?.tasks ?? []).length === 0 ? (
              <div className="jobs-empty">No jobs yet.</div>
            ) : (
              (jobsInfo?.tasks ?? []).map((job) => (
                <div className="job-row" key={job.task_id}>
                  <div className="job-row__main">
                    <div className="job-row__title">
                      <span className={`job-status job-status--${job.status}`}>{job.status}</span>
                      <span>{formatTaskType(job.type)}</span>
                      {job.queue_position && <span className="job-row__queue">#{job.queue_position} in queue</span>}
                    </div>
                    <div className="job-row__id" title={job.task_id}>
                      ID {job.task_id}
                    </div>
                    <div className="job-row__message">{job.message || job.task_id}</div>
                  </div>
                  <div className="job-row__meta">
                    <span>{Math.round((job.progress ?? 0) * 100)}%</span>
                    <span>{formatTaskTime(job.updated_at)}</span>
                    <button className="small-btn job-row__open" type="button" onClick={() => void handleOpenJob(job)}>
                      Open
                    </button>
                  </div>
                  <div className="job-row__bar" aria-hidden="true">
                    <span style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
                  </div>
                </div>
              ))
            )}
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
      {currentStep === 1 && (
        <section>
          <h2>1. Upload PDF</h2>
          <p style={{ marginTop: 0, color: '#64748b' }}>
            Upload the source PDF to begin. Page thumbnails are prepared automatically.
          </p>
          <input type="file" accept="application/pdf" onChange={handleFileUpload} />
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
                    Automatic detection, structure extraction and bioactivity parsing.
                  </p>
                </div>
                <button
                  className={automaticExtractionActive ? 'button--cancel' : 'button--start'}
                  type="button"
                  onClick={automaticExtractionActive ? handleCancelAutomaticExtraction : handleStartFullPipeline}
                  disabled={!automaticExtractionActive && !pdfReadyForDetection}
                >
                  {automaticExtractionActive
                    ? 'Cancel'
                    : pdfReadyForDetection
                    ? 'Start'
                    : 'Preparing…'}
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
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ margin: 0 }}>2. Extract Structures</h2>
            <span className="tip-icon" data-tip="Structure pages are detected automatically. Use thumbnails or page ranges such as 1,3,5-8 only when you want to constrain processing.">?</span>
          </div>
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
                      onChange={(event) => setAutoDetectStructurePages(event.target.checked)}
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
                <div className="selector-actions__group structure">
                  <button
                    className="primary structure-action"
                    type="button"
                    onClick={handleStartStructureExtraction}
                    disabled={!pdfInfo || (!autoDetectStructurePages && structureSelection.size === 0) || isStructureSubmitting}
                  >
                    {isStructureSubmitting ? 'Extracting…' : 'Extract'}
                  </button>
                </div>
                <div className="selector-actions__links flex-gap">
                  <button className="small-btn" type="button" onClick={() => setCurrentStep(4)} disabled={!canViewResults}>
                    Results
                  </button>
                </div>
                {isGalleryLoading && (
                  <small className="selector-actions__hint">Rendering previews in the background.</small>
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
                {structureSelectionFeedback && (
                  <small className="selector-actions__hint selector-actions__hint--notice">{structureSelectionFeedback}</small>
                )}
              </div>
            </div>

            <div className="selector-gallery">
              {renderPageGallery('structure')}
            </div>
          </div>
        </section>
      )}

      {pdfInfo && currentStep === 3 && (
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ margin: 0 }}>3. Extract Bioactivity</h2>
            <span className="tip-icon" data-tip="Bioactivity pages and assay names are detected automatically. Add page ranges or assay names only when you want to constrain extraction.">?</span>
          </div>
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
                      onChange={(event) => setAutoDetectAssayPages(event.target.checked)}
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
                    placeholder="Enter/comma to add"
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
                <div className="selector-actions__group assay">
                  <button
                    className="primary assay-action"
                    type="button"
                    onClick={handleStartAssayExtraction}
                    disabled={
                      !pdfInfo ||
                      (!autoDetectAssayPages && assaySelection.size === 0) ||
                      (!autoDetectAssayNames && assayNames.length === 0 && !assayNameDraft.trim()) ||
                      isAssaySubmitting ||
                      isAssayWaitingForStructures
                    }
                  >
                  {isAssaySubmitting
                    ? 'Extracting…'
                    : isAssayWaitingForStructures
                    ? 'Waiting for structures…'
                    : 'Extract'}
                </button>
              </div>
              <div className="selector-actions__links flex-gap">
                <button className="small-btn" type="button" onClick={() => setCurrentStep(4)} disabled={!canViewResults}>
                  Results
                </button>
              </div>
              {isGalleryLoading && (
                <small className="selector-actions__hint">Rendering previews in the background.</small>
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

            <div className="selector-gallery">
              {renderPageGallery('assay')}
            </div>
          </div>
        </section>
      )}

          {currentStep === 4 && (
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2 style={{ margin: 0 }}>4. Review Results</h2>
            <span className="tip-icon" data-tip="Review the extracted structures and assay records, apply edits, and export tables.">?</span>
          </div>
          {(structures.length === 0 && filteredStructures.length === 0 && assayRecords.length === 0) && (
            <p style={{ color: '#94a3b8' }}>Extraction results appear here once the tasks finish.</p>
          )}
          {(structures.length > 0 || filteredStructures.length > 0 || assayRecords.length > 0) && (
            <div className="review-layout">
              <aside className="review-sidebar">
                <div className="side-card">
                  <h3 className="side-card__title">Navigate</h3>
                  <button
                    className="secondary side-card__button"
                    type="button"
                    onClick={() => setCurrentStep(2)}
                  >
                    Structures
                  </button>
                  <button
                    className="secondary side-card__button"
                    type="button"
                    onClick={() => setCurrentStep(3)}
                  >
                    Bioactivity
                  </button>
                </div>
                <div className="side-card">
                  <h3 className="side-card__title">Export</h3>
                  <button className="secondary side-card__button" type="button" onClick={downloadStructuresCsv}>
                    Structures CSV
                  </button>
                  {assayColumnNames.length > 0 && (
                    <button
                      className="secondary side-card__button"
                      type="button"
                      onClick={downloadAssayCsv}
                      disabled={assayRecords.length === 0}
                    >
                      Bioactivity CSV
                    </button>
                  )}
                </div>
              </aside>

              <div className="review-main">
                {saveStatusMeta && (
                  <div className="review-main__status">
                    <span className={`table-actions__status ${saveStatusMeta.modifier}`}>
                      {saveStatusMeta.label}
                    </span>
                  </div>
                )}
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
              </label>
              <StructureEditorInline
                smiles={modalRowCanEdit ? String(editedStructures[modalRowIndex!]?.SMILES || '') : ''}
                molblock={modalRowCanEdit ? getMolblockValue(editedStructures[modalRowIndex!]) : ''}
                onSave={handleInlineStructureSave}
              />
              <small className="floating-panel__note">Changes save automatically.</small>
            </div>
          </div>
        )}
      </div>
    )}
                {assayTask && (
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
                <div className="table-controls">
                  <label className="table-controls__item">
                    <span className="table-controls__label">Row height</span>
                    <input
                      type="range"
                      min={120}
                      max={280}
                      step={10}
                      value={rowHeight}
                      onChange={(event) => setRowHeight(Number(event.target.value))}
                    />
                    <span className="table-controls__value">{rowHeight}px</span>
                  </label>
                </div>
                <div className="table-wrapper" style={tableStyle as React.CSSProperties} ref={tableWrapperRef}>
                  <table className="review-table">
                    <thead>
                      <tr>
                        <th>Page</th>
                        <th>PDF preview</th>
                        <th>Compound ID</th>
                        <th>Source structure</th>
                        <th>Extracted structure</th>
                        {structureColumnsToRender.map((column) => (
                          <th
                            key={column}
                            className="column-header column-header--structure"
                          >
                            <span className="column-header__icon" aria-hidden="true">🧬</span>
                            <span>{STRUCTURE_COLUMN_LABELS[column] ?? column}</span>
                          </th>
                        ))}
                        {assayColumnNames.map((column) => (
                          <th key={`assay-${column}`}>🧪 {column}</th>
                        ))}
                        <th className="column-header">Actions</th>
                      </tr>
                    </thead>
                    <tbody ref={tableBodyRef}>
                      {processedStructureRows.map((row) => {
                        const {
                          record,
                          structureImage,
                          structureSource,
                          segmentImage,
                          segmentSource,
                          structurePreview,
                          assayData,
                          index,
                          pageHeading,
                          secondaryPageInfo,
                          artifactLabel,
                          normalizedPrimaryPage,
                          rowSpan,
                          showPageCell,
                        } = row;
                        const smilesValue = typeof record.SMILES === 'string' ? record.SMILES : '';
                        const compoundIdRaw = record.COMPOUND_ID;
                        const compoundIdValue =
                          typeof compoundIdRaw === 'string' ? compoundIdRaw : formatCellValue(compoundIdRaw);
                        const canEditStructure = index < editedStructures.length;
                        const isRowVisible = visibleRowIndices.has(index);
                        const isPreviewLoading = ['Structure', 'IMAGE_FILE', 'Image File', 'SEGMENT_FILE'].some((key) => {
                          const value = record[key];
                          return typeof value === 'string' && loadingArtifacts.has(value);
                        });
                        const pageCell = (
                          <td
                            className="review-table__cell review-table__cell--page-number"
                            rowSpan={rowSpan}
                          >
                            <div className="page-number">
                              <span className="page-number__label">{pageHeading}</span>
                              {secondaryPageInfo && <span className="page-number__range">{secondaryPageInfo}</span>}
                            </div>
                          </td>
                        );
                        return (
                          <tr key={row.id ?? index} data-row-index={index}>
                            {(!normalizedPrimaryPage || showPageCell) && pageCell}
                            <td className="review-table__cell review-table__cell--preview">
                              <div className="page-cell">
                                <div className="page-cell__media">
                                  {structureImage ? (
                                    isRowVisible ? (
                                      <button
                                        type="button"
                                        className="page-cell__image"
                                        onClick={() =>
                                          openArtifact(structureSource || structureImage || '', artifactLabel, {
                                            rowIndex: canEditStructure ? index : null,
                                          })
                                        }
                                      >
                                        <img src={structureImage} alt="PDF page" loading="lazy" />
                                      </button>
                                    ) : (
                                      <div className="page-cell__placeholder">Scroll to load preview</div>
                                    )
                                  ) : (
                                    <div className="page-cell__placeholder">
                                      {isPreviewLoading ? 'Loading…' : 'No preview'}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                            <td className="review-table__cell review-table__cell--id">
                              <CompoundIdInput
                                initialValue={compoundIdValue}
                                rowIndex={index}
                                disabled={!canEditStructure}
                                onSave={handleCompoundIdSave}
                              />
                            </td>
                            <td className="review-table__cell review-table__cell--structure">
                              {segmentImage ? (
                                isRowVisible ? (
                                  <button
                                    type="button"
                                    className="structure-image-btn"
                                    onClick={() =>
                                      openArtifact(segmentSource || segmentImage || '', `Source structure - ${record.COMPOUND_ID ?? ''}`, {
                                        rowIndex: canEditStructure ? index : null,
                                      })
                                    }
                                  >
                                    <img src={segmentImage} alt="Source structure" loading="lazy" />
                                  </button>
                                ) : (
                                  <div className="page-cell__placeholder page-cell__placeholder--compact">Scroll to load preview</div>
                                )
                              ) : (
                                <span className="muted">None</span>
                              )}
                            </td>
                            <td className="review-table__cell review-table__cell--structure">
                              <div className="structure-cell-content">
                                <button
                                  type="button"
                                  className={structurePreview ? 'structure-image-btn' : 'smiles-chip'}
                                  onClick={() => {
                                    // 如果有SMILES预览，显示Quick edit面板
                                    if (structurePreview) {
                                      setModalArtifact({
                                        path: `Extracted structure - ${record.COMPOUND_ID ?? ''}`,
                                        mime: 'image/png',
                                        data: structurePreview,
                                        rowIndex: canEditStructure ? index : null,
                                      });
                                    } else {
                                      // 否则打开结构编辑器
                                      handleOpenStructureEditor(index);
                                    }
                                  }}
                                  disabled={!canEditStructure}
                                >
                                  {structurePreview ? (
                                    isRowVisible ? (
                                      <img src={structurePreview} alt="Extracted structure" loading="lazy" />
                                    ) : (
                                      <span className="lazy-chip">Scroll to load preview</span>
                                    )
                                  ) : smilesValue ? (
                                    <span className="smiles-text">{smilesValue}</span>
                                  ) : (
                                    'Add SMILES'
                                  )}
                                </button>
                                {!smilesValue && canEditStructure && (
                                  <div className="reparse-buttons">
                                    <button
                                      type="button"
                                      className="reparse-btn"
                                      onClick={() => handleReparseStructure(index, 'MolNexTR')}
                                      disabled={reparseState.rowIndex === index && reparseState.engine === 'MolNexTR'}
                                    >
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolNexTR' ? 'Re-parsing…' : 'MolNexTR'}
                                    </button>
                                    <button
                                      type="button"
                                      className="reparse-btn"
                                      onClick={() => handleReparseStructure(index, 'MolVec')}
                                      disabled={reparseState.rowIndex === index && reparseState.engine === 'MolVec'}
                                    >
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolVec' ? 'Re-parsing…' : 'MolVec'}
                                    </button>
                                    <button
                                      type="button"
                                      className="reparse-btn"
                                      onClick={() => handleReparseStructure(index, 'MolScribe')}
                                      disabled={reparseState.rowIndex === index && reparseState.engine === 'MolScribe'}
                                    >
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolScribe' ? 'Re-parsing…' : 'MolScribe'}
                                    </button>
                                  </div>
                                )}
                              </div>
                            </td>
                            {structureColumnsToRender.map((column) => {
                              const value = record[column as keyof StructureRecord];
                              const editable = isEditableColumn(column, value) && canEditStructure;
                              return (
                                <td key={column} className="review-table__cell review-table__cell--structure">
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
                            {assayColumnNames.map((column) => (
                              <td
                                key={column}
                                className="review-table__cell review-table__cell--assay"
                              >
                                {formatCellValue(assayData[column])}
                              </td>
                            ))}
                            <td className="review-table__cell review-table__cell--actions">
                              {canEditStructure ? (
                                <button
                                  type="button"
                                  className="small-btn danger"
                                  onClick={() => handleDeleteStructureRow(index)}
                                >
                                  Delete
                                </button>
                              ) : (
                                <span className="muted">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {filteredStructureRows.length > 0 && (
                  <div style={{ marginTop: 24 }}>
                    <h3 style={{ marginBottom: 8 }}>Filtered structure candidates</h3>
                    <p style={{ color: '#64748b', marginTop: 0 }}>
                      These candidates were excluded from downstream compound ID and bioactivity matching.
                    </p>
                    <div className="flex-gap" style={{ flexWrap: 'wrap', marginBottom: 12 }}>
                      <span className="tag">
                        Total filtered: {filteredStructureRows.length}
                      </span>
                      {filteredStructureStats.map(([label, count]) => (
                        <span className="tag" key={`filtered-stat-${label}`}>
                          {label}: {count}
                        </span>
                      ))}
                    </div>
                    <div className="table-wrapper">
                      <table className="review-table">
                        <thead>
                          <tr>
                            <th>Page</th>
                            <th>PDF preview</th>
                            <th>Source structure</th>
                            {filteredStructureColumns.map((column) => (
                              <th key={`filtered-${column}`}>
                                {STRUCTURE_COLUMN_LABELS[column] ?? column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {filteredStructureRows.map(({ record, index, previewImage, segmentImage }) => {
                            const pageValue =
                              formatCellValue(
                                record.PAGE_NUM ??
                                  (record as Record<string, unknown>).page_num ??
                                  (record as Record<string, unknown>).page,
                              ) || '—';
                            const previewSource =
                              typeof record.PAGE_IMAGE_FILE === 'string'
                                ? record.PAGE_IMAGE_FILE
                                : typeof record.IMAGE_FILE === 'string'
                                ? record.IMAGE_FILE
                                : typeof record.SEGMENT_FILE === 'string'
                                ? record.SEGMENT_FILE
                                : '';
                            const segmentSource =
                              typeof record.SEGMENT_FILE === 'string'
                                ? record.SEGMENT_FILE
                                : typeof record.Segment === 'string'
                                ? record.Segment
                                : '';
                            return (
                              <tr key={`filtered-row-${index}`}>
                                <td className="review-table__cell">{pageValue}</td>
                                <td className="review-table__cell review-table__cell--preview">
                                  {previewImage ? (
                                    <button
                                      type="button"
                                      className="page-cell__image"
                                      onClick={() => openArtifact(previewSource || previewImage, `Filtered candidate page ${pageValue}`)}
                                    >
                                      <img src={previewImage} alt="Filtered PDF preview" loading="lazy" />
                                    </button>
                                  ) : (
                                    <span className="muted">None</span>
                                  )}
                                </td>
                                <td className="review-table__cell review-table__cell--structure">
                                  {segmentImage ? (
                                    <button
                                      type="button"
                                      className="structure-image-btn"
                                      onClick={() =>
                                        openArtifact(
                                          segmentSource || segmentImage,
                                          `Filtered structure - ${formatCellValue(record.STRUCTURE_TYPE ?? '')}`,
                                        )
                                      }
                                    >
                                      <img src={segmentImage} alt="Filtered structure candidate" loading="lazy" />
                                    </button>
                                  ) : (
                                    <span className="muted">None</span>
                                  )}
                                </td>
                                {filteredStructureColumns.map((column) => (
                                  <td key={`filtered-value-${index}-${column}`} className="review-table__cell">
                                    {formatCellValue(record[column as keyof StructureRecord])}
                                  </td>
                                ))}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      )}

      </div>

      {error && (
        <div className="status-banner" style={{ background: '#fee2e2', borderColor: '#fecaca', color: '#b91c1c' }}>
          <span>{error}</span>
          <button className="secondary" onClick={() => setError(null)} type="button">
            Close
          </button>
        </div>
      )}
      {toast && !error && (
        <div className="status-banner" style={{ background: '#ecfdf5', borderColor: '#bbf7d0', color: '#065f46' }}>
          <span>{toast}</span>
          <button className="secondary" onClick={() => setToast(null)} type="button">
            Got it
          </button>
        </div>
      )}

      {modalArtifact && (
        <div className="modal" onClick={closeModal} role="button" tabIndex={-1}>
          <div 
            className="modal-content" 
            onClick={(event) => event.stopPropagation()}
            onPointerDown={(e) => handleModalImagePointerDown(e, e.currentTarget)}
            onPointerMove={handleModalImagePointerMove}
            onPointerUp={handleModalImagePointerUp}
            onPointerCancel={handleModalImagePointerUp}
          >
            <h3 style={{ marginTop: 0 }}>Image preview</h3>
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
            <div style={{ marginTop: 12, color: '#475569' }}>{modalArtifact.path}</div>
            <div className="flex-gap" style={{ marginTop: 12 }}>
              {!isMagnifying && modalArtifact.mime !== 'loading' && modalArtifact.data && (
                <a className="secondary" href={modalArtifact.data} download={`page-${magnifiedPage ?? ''}.png`}>
                  Download image
                </a>
              )}
            </div>
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
                </label>
                <StructureEditorInline
                  smiles={modalRowCanEdit ? String(editedStructures[modalRowIndex!]?.SMILES || '') : ''}
                  molblock={modalRowCanEdit ? getMolblockValue(editedStructures[modalRowIndex!]) : ''}
                  onSave={handleInlineStructureSave}
                />
                <small className="floating-panel__note">Changes save automatically.</small>
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
        onSave={handleStructureEditorSave}
      />
    </div>
  );
};

export default App;

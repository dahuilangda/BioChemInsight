import React from 'react';
import {
  fetchArtifact,
  fetchPdfInfo,
  fetchPdfPage,
  fetchTask,
  fetchTaskAssays,
  fetchTaskStructures,
  getTaskDownloadUrl,
  queueAssayTask,
  queueStructureTask,
  renderSmiles,
  updateTaskStructures,
  uploadPdf,
  reparseStructure,
} from './api/client';
import type {
  AssayRecord,
  StructureRecord,
  TaskStatus,
  UploadPDFResponse,
} from './types';
import StructureEditorModal from './components/StructureEditorModal';
import StructureEditorInline from './components/StructureEditorInline';

type StepId = 1 | 2 | 3 | 4;

type StepKey = 'upload' | 'structures' | 'bioactivity' | 'review';

const stepDefinitions: Array<{ id: StepId; label: string; icon: StepKey }> = [
  { id: 1, label: 'Upload PDF', icon: 'upload' },
  { id: 2, label: 'Structures', icon: 'structures' },
  { id: 3, label: 'Bioactivity', icon: 'bioactivity' },
  { id: 4, label: 'Review', icon: 'review' },
];

const preferredColumnOrder = ['COMPOUND_ID', 'SMILES', 'source_pages', 'IMAGE_FILE', 'SEGMENT_FILE'];
const markdownImageRegex = /!\[[^\]]*\]\((data:image\/[^)]+)\)/i;
const isDataImage = (value: string) => value.startsWith('data:image');
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
]);
const STRUCTURE_COLUMN_LABELS: Record<string, string> = {
  COMPOUND_ID: 'Compound ID',
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
  }, []);

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

  const [pageImages, setPageImages] = React.useState<Record<number, string>>({});
  const [loadingPages, setLoadingPages] = React.useState<Set<number>>(new Set());

  const [structureTask, setStructureTask] = React.useState<TaskStatus | null>(null);
  const [assayTask, setAssayTask] = React.useState<TaskStatus | null>(null);

  const [structures, setStructures] = React.useState<StructureRecord[]>([]);
  const [editedStructures, setEditedStructures] = React.useState<StructureRecord[]>([]);
  const [saveStatus, setSaveStatus] = React.useState<'idle' | 'pending' | 'saving' | 'saved' | 'error'>('idle');
  const [smilesPreviewCache, setSmilesPreviewCache] = React.useState<Record<string, string>>({});
  const smilesPreviewCacheRef = React.useRef(smilesPreviewCache);

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
  const [editorState, setEditorState] = React.useState<{ open: boolean; rowIndex: number | null; smiles: string }>(
    { open: false, rowIndex: null, smiles: '' },
  );
  const [reparseState, setReparseState] = React.useState<{ rowIndex: number | null; engine: string | null }>({
    rowIndex: null,
    engine: null,
  });
  const [currentStep, setCurrentStep] = React.useState<StepId>(1);
  const autoAdvanceRef = React.useRef(false);
  const pdfInitializedRef = React.useRef(false);
  const activeSelectionMode = React.useMemo<SelectionMode>(() => getSelectionModeForStep(currentStep), [currentStep]);
  const isGalleryLoading = loadingPages.size > 0;

  const [error, setError] = React.useState<string | null>(null);
  const [toast, setToast] = React.useState<string | null>(null);
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
    element: HTMLImageElement | null;
  } | null>(null);
  const [rowHeight, setRowHeight] = React.useState(160);
  const [modalEditorPosition, setModalEditorPosition] = React.useState<{ x: number; y: number }>({ x: 32, y: 32 });
  const modalEditorPanelRef = React.useRef<HTMLDivElement | null>(null);
  const modalEditorDragRef = React.useRef<{ offsetX: number; offsetY: number } | null>(null);
  const lastStructurePageRef = React.useRef<number | null>(null);
  const lastAssayPageRef = React.useRef<number | null>(null);
  const pdfInfoRef = React.useRef<UploadPDFResponse | null>(pdfInfo);
  const pageImagesRef = React.useRef(pageImages);
  const loadingPagesRef = React.useRef(loadingPages);

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
  const structureRows = React.useMemo(
    () =>
      editedStructures.map((record, index) => {
        const smilesValue = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
        const rawPreview = smilesValue ? smilesPreviewCache[smilesValue] : undefined;
        const smilesPreview = rawPreview && rawPreview.length > 0 ? rawPreview : null;
        const structureImage =
          extractImageSource(record.Structure) ??
          extractImageSource(record.PAGE_IMAGE_FILE) ??
          smilesPreview;
        const segmentImage = extractImageSource(record.Segment) ?? extractImageSource(record.SEGMENT_FILE);
        // 使用COMPOUND_ID作为key来匹配活性数据
        return {
          id: (record.COMPOUND_ID ?? '').toString(),
          record,
          index,
          structureImage,
          segmentImage,
          smilesPreview,
          structureSource:
            structureImage ||
            (typeof record.Structure === 'string'
              ? record.Structure
              : typeof record.PAGE_IMAGE_FILE === 'string'
              ? record.PAGE_IMAGE_FILE
              : smilesPreview ?? ''),
          segmentSource:
            segmentImage ||
            (typeof record.Segment === 'string'
              ? record.Segment
              : typeof record.SEGMENT_FILE === 'string'
              ? record.SEGMENT_FILE
              : ''),
        };
      }),
    [editedStructures, extractImageSource, smilesPreviewCache],
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
          smilesPreview: null,
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
  const handleModalImagePointerDown = React.useCallback((event: React.PointerEvent<HTMLImageElement>, element: HTMLImageElement) => {
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
    
    element.setPointerCapture(event.pointerId);
  }, []);

  const handleModalImagePointerMove = React.useCallback((event: React.PointerEvent<HTMLImageElement>) => {
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

  const handleModalImagePointerUp = React.useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    if (!modalDragState || !modalDragState.element) return;
    
    const element = modalDragState.element;
    if (element.hasPointerCapture(event.pointerId)) {
      element.releasePointerCapture(event.pointerId);
    }
    
    setModalDragState(null);
  }, [modalDragState]);

  React.useEffect(() => {
    const cacheSnapshot = smilesPreviewCacheRef.current;
    const pending = new Set<string>();
    editedStructures.forEach((record) => {
      const smilesValue = typeof record.SMILES === 'string' ? record.SMILES.trim() : '';
      if (!smilesValue) return;
      if (Object.prototype.hasOwnProperty.call(cacheSnapshot, smilesValue)) return;
      const hasImage =
        extractImageSource(record.Structure) ?? extractImageSource(record.IMAGE_FILE);
      if (hasImage) return;
      pending.add(smilesValue);
    });
    if (!pending.size) return () => undefined;

    let cancelled = false;
    const tasks = Array.from(pending).map(async (smilesValue) => {
      try {
        const image = await renderSmiles(smilesValue, { width: 280, height: 220 });
        if (cancelled || !image) return;
        setSmilesPreviewCache((prev) =>
          Object.prototype.hasOwnProperty.call(prev, smilesValue)
            ? prev
            : { ...prev, [smilesValue]: image },
        );
      } catch (error) {
        if (cancelled) return;
        console.warn('Failed to generate structure preview', error);
        setSmilesPreviewCache((prev) =>
          Object.prototype.hasOwnProperty.call(prev, smilesValue)
            ? prev
            : { ...prev, [smilesValue]: '' },
        );
      }
    });

    void Promise.all(tasks);

    return () => {
      cancelled = true;
    };
  }, [editedStructures, extractImageSource]);
  const canViewResults = React.useMemo(
    () =>
      Boolean(
        structures.length > 0 ||
          assayRecords.length > 0 ||
          structureTask !== null ||
          assayTask !== null,
      ),
    [structures.length, assayRecords.length, structureTask, assayTask],
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

  const handleStructuresReady = React.useCallback(() => {
    if (autoAdvanceRef.current) return;
    autoAdvanceRef.current = true;
    if (!assayTask && assayRecords.length === 0) {
      setCurrentStep(3);
      setToast((prev) => prev ?? 'Structure extraction completed. Continue with bioactivity extraction.');
    } else {
      setCurrentStep((prev) => (prev < 4 ? 4 : prev));
    }
  }, [assayTask, assayRecords.length]);

  const clearWorkspace = React.useCallback(() => {
    setStructurePagesInput('');
    setAssayPagesInput('');
    setStructureSelection(new Set<number>());
    setAssaySelection(new Set<number>());
    lastStructurePageRef.current = null;
    lastAssayPageRef.current = null;
    setPageImages({});
    pageImagesRef.current = {};
    setLoadingPages(new Set<number>());
    loadingPagesRef.current = new Set();
    pageFetchQueueRef.current = [];
    activePageFetchesRef.current = 0;
    setStructureTask(null);
    setAssayTask(null);
    setStructures([]);
    setEditedStructures([]);
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
    autoAdvanceRef.current = false;
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
      setCurrentStep(2);
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
    smilesPreviewCacheRef.current = smilesPreviewCache;
  }, [smilesPreviewCache]);

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

  const MAX_PAGE_FETCH_CONCURRENCY = 4;
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
    const initialCount = Math.min(16, pages.length);
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
    (['structure', 'assay'] as SelectionMode[]).forEach((mode) => {
      const container = galleryContainerRefs.current[mode];
      container?.querySelectorAll('[data-page]').forEach((node) => observer.observe(node));
    });
    return () => observer.disconnect();
  }, [pages, currentStep, loadPageImage, pageImages]);

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
    const missing = new Set<string>();
            editedStructures.forEach((record) => {
              ['Structure', 'Segment', 'IMAGE_FILE', 'SEGMENT_FILE', 'Image File', 'Segment File', 'PAGE_IMAGE_FILE'].forEach((key) => {        const value = record[key as keyof StructureRecord];
        if (typeof value !== 'string') return;
        if (!value) return;
        if (isDataImage(value) || markdownImageRegex.test(value) || imageCache[value]) return;
        missing.add(value);
      });
    });
    if (!missing.size) return;
    missing.forEach((path) => {
      fetchArtifact(path)
        .then((artifact) => {
          const dataUri = `data:${artifact.mime_type};base64,${artifact.content}`;
          setImageCache((prev) => {
            if (prev[path]) return prev;
            return { ...prev, [path]: dataUri };
          });
        })
        .catch(() => {
          /* ignore individual fetch errors */
        });
    });
  }, [editedStructures, imageCache]);

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
    } else if (!pdfInitializedRef.current && maxStep >= 2) {
      setCurrentStep(2);
      pdfInitializedRef.current = true;
    }
  }, [pdfInfo, maxStep]);

  const updateStructureSelection = React.useCallback((updater: (prev: Set<number>) => Set<number>) => {
    setStructureSelection((prev) => {
      const next = updater(prev);
      const sorted = Array.from(next).sort((a, b) => a - b);
      setStructurePagesInput(sorted.length ? pagesToString(sorted) : '');
      return next;
    });
  }, []);

  const updateAssaySelection = React.useCallback((updater: (prev: Set<number>) => Set<number>) => {
    setAssaySelection((prev) => {
      const next = updater(prev);
      const sorted = Array.from(next).sort((a, b) => a - b);
      setAssayPagesInput(sorted.length ? pagesToString(sorted) : '');
      return next;
    });
  }, []);

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
        const classNames = [
          'page-card',
          isStructureSelected ? 'structure-selected' : '',
          isAssaySelected ? 'assay-selected' : '',
          isActiveSelected ? `active-mode-${mode}` : '',
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
                {isStructureSelected && <span className="badge badge-structure">S</span>}
                {isAssaySelected && <span className="badge badge-assay">A</span>}
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
    lastStructurePageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
  };

  const handleAssayInputChange: React.ChangeEventHandler<HTMLTextAreaElement> = (event) => {
    const value = event.target.value;
    setAssayPagesInput(value);
    const parsed = parsePagesInput(value);
    setAssaySelection(new Set<number>(parsed));
    lastAssayPageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
  };

  const addAssayNames = React.useCallback((raw: string) => {
    const extras = parseAssayNames(raw);
    if (!extras.length) return;
    setAssayNames((prev) => mergeAssayNameLists(prev, extras));
  }, []);

  const removeAssayName = (index: number) => {
    setAssayNames((prev) => prev.filter((_, idx) => idx !== index));
  };

  const clearAssayNames = () => {
    setAssayNames([]);
    setAssayNameDraft('');
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
    if (updated.status === 'completed') {
      const results = await fetchTaskStructures(taskId);
      const nextRecords = results.records.map((record) => ({ ...record }));
      setStructures(results.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
      setSaveStatus(results.records.length ? 'saved' : 'idle');
      setToast(
        `Structure extraction complete: ${results.records.length} record${
          results.records.length === 1 ? '' : 's'
        }. Continue in Step 3 to extract bioactivity.`,
      );
    }
    if (updated.status === 'failed') {
      setError(updated.error || 'Structure extraction task failed');
    }
  }, []);

  const refreshAssayTask = React.useCallback(async (taskId: string) => {
    const updated = await fetchTask(taskId);
    setAssayTask(updated);
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
  }, []);

  React.useEffect(() => {
    if (!structureTask || structureTask.status === 'completed' || structureTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshStructureTask(structureTask.task_id).catch((err) =>
        setError(err instanceof Error ? err.message : 'Unable to refresh structure task'),
      );
    }, 1800);
    return () => window.clearInterval(interval);
  }, [structureTask, refreshStructureTask]);

  React.useEffect(() => {
    if (!assayTask || assayTask.status === 'completed' || assayTask.status === 'failed') {
      return;
    }
    const interval = window.setInterval(() => {
      void refreshAssayTask(assayTask.task_id).catch((err) =>
        setError(err instanceof Error ? err.message : 'Unable to refresh bioactivity task'),
      );
    }, 2000);
    return () => window.clearInterval(interval);
  }, [assayTask, refreshAssayTask]);

  const handleStartStructureExtraction = async () => {
    resetNotifications();
    if (!pdfInfo) {
      setError('Upload a PDF before starting structure extraction.');
      return;
    }
    const pagesSelected = Array.from(structureSelection).sort((a, b) => a - b);
    if (!pagesSelected.length) {
      setError('Select or enter the pages that contain structures.');
      return;
    }
    const pageString = pagesToString(pagesSelected);
    setStructurePagesInput(pageString);
    try {
      setIsStructureSubmitting(true);
      const taskStatus = await queueStructureTask({
        pdf_id: pdfInfo.pdf_id,
        pages: pageString,
      });
      setStructureTask(taskStatus);
      setStructures([]);
      setEditedStructures([]);
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
    if (!pagesSelected.length) {
      setError('Select or enter the pages that contain bioactivity data.');
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
    if (!namesList.length) {
      setError('Provide at least one assay name, e.g. "IC50".');
      return;
    }
    const pageString = pagesToString(pagesSelected);
    setAssayPagesInput(pageString);
    try {
      setIsAssaySubmitting(true);
      const taskStatus = await queueAssayTask({
        pdf_id: pdfInfo.pdf_id,
        pages: pageString,
        assay_names: namesList,
        structure_task_id: structureTask?.status === 'completed' ? structureTask.task_id : undefined,
      });
      setAssayTask(taskStatus);
      setAssayRecords([]);
      setToast('Bioactivity extraction task submitted');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit bioactivity extraction task');
    } finally {
      setIsAssaySubmitting(false);
    }
  };

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
      setStructures(response.records);
      editedStructuresRef.current = nextRecords;
      setEditedStructures(nextRecords);
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

  const handleOpenStructureEditor = (rowIndex: number) => {
    const record = editedStructures[rowIndex];
    const smiles = typeof record.SMILES === 'string' ? record.SMILES : '';
    setEditorState({ open: true, rowIndex, smiles });
  };

  const handleStructureEditorCancel = () => {
    setEditorState({ open: false, rowIndex: null, smiles: '' });
  };

  const handleStructureEditorSave = ({ smiles, image }: { smiles: string; image?: string }) => {
    if (editorState.rowIndex === null) {
      setEditorState({ open: false, rowIndex: null, smiles: '' });
      return;
    }
    setEditedStructures((prev) =>
      prev.map((row, idx) => {
        if (idx !== editorState.rowIndex) return row;
        const updated: StructureRecord = { ...row, SMILES: smiles };
        // 只更新SMILES字段，不修改原始的Structure和IMAGE_FILE字段
        return updated;
      }),
    );
    if (image) {
      setImageCache((prev) => ({ ...prev, [image]: image }));
    }
    if (smiles) {
      setSmilesPreviewCache((prev) => (image ? { ...prev, [smiles]: image } : prev));
    }
    scheduleAutoSave();
    setEditorState({ open: false, rowIndex: null, smiles: '' });
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
      setError('无法确定结构所在的页面');
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
      setEditedStructures((prev) =>
        prev.map((row, idx) => {
          if (idx !== index) return row;
          return { ...row, SMILES: result.smiles };
        }),
      );

      // 如果有SMILES，生成预览图像
      if (result.smiles) {
        try {
          const image = await renderSmiles(result.smiles, { width: 280, height: 220 });
          setSmilesPreviewCache((prev) => ({ ...prev, [result.smiles]: image }));
          setImageCache((prev) => ({ ...prev, [image]: image }));
        } catch (err) {
          console.warn('Failed to generate structure preview', err);
        }
      }

      setReparseState({ rowIndex: null, engine: null });
      scheduleAutoSave();
      setToast(`结构已使用 ${engine} 重新解析`);
    } catch (err) {
      setReparseState({ rowIndex: null, engine: null });
      setError(err instanceof Error ? err.message : '重新解析结构失败');
    }
  };

  const handleInlineStructureSave = ({ smiles, image }: { smiles: string; image?: string }) => {
    if (modalRowIndex === null) {
      return;
    }
    setEditedStructures((prev) =>
      prev.map((row, idx) => {
        if (idx !== modalRowIndex) return row;
        const updated: StructureRecord = { ...row, SMILES: smiles };
        // 只更新SMILES字段，不修改原始的Structure和IMAGE_FILE字段
        return updated;
      }),
    );
    if (image) {
      setImageCache((prev) => ({ ...prev, [image]: image }));
    }
    if (smiles) {
      setSmilesPreviewCache((prev) => (image ? { ...prev, [smiles]: image } : prev));
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
    setStructureSelection(new Set<number>());
    setStructurePagesInput('');
    lastStructurePageRef.current = null;
  };

  const clearAssaySelection = () => {
    setAssaySelection(new Set<number>());
    setAssayPagesInput('');
    lastAssayPageRef.current = null;
  };

  React.useEffect(() => {
    if (initializedFromUrl.current) return;
    initializedFromUrl.current = true;
    const params = new URLSearchParams(window.location.search);

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
      setStructurePagesInput(structPagesParam);
      const parsed = parsePagesInput(structPagesParam);
      setStructureSelection(new Set<number>(parsed));
      lastStructurePageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
    }

    const assayPagesParam = params.get('assayPages');
    if (assayPagesParam) {
      setAssayPagesInput(assayPagesParam);
      const parsed = parsePagesInput(assayPagesParam);
      setAssaySelection(new Set<number>(parsed));
      lastAssayPageRef.current = parsed.length ? parsed[parsed.length - 1] : null;
    }

    const assayNamesParam = params.get('assayNames');
    if (assayNamesParam) {
      setAssayNames(parseAssayNames(assayNamesParam));
    }

    const pdfParam = params.get('pdf');
    if (pdfParam) {
      fetchPdfInfo(pdfParam)
        .then((info: UploadPDFResponse) => {
          setPdfInfo(info);
          setToast((prev) => prev ?? `Loaded PDF: ${info.filename}`);
        })
        .catch(() => setError('Could not restore the PDF. Please upload it again.'));
    }

    const structureTaskId = params.get('structureTask');
    if (structureTaskId) {
      fetchTask(structureTaskId)
        .then((task) => {
          setStructureTask(task);
          if (task.status === 'completed') {
            return fetchTaskStructures(structureTaskId).then((results) => {
              const nextRecords = results.records.map((record) => ({ ...record }));
              setStructures(results.records);
              editedStructuresRef.current = nextRecords;
              setEditedStructures(nextRecords);
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
    if (assayTaskId) {
      fetchTask(assayTaskId)
        .then((task) => {
          setAssayTask(task);
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
  }, []);

  React.useEffect(() => {
    const params = new URLSearchParams();

    if (pdfInfo) {
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
    if (structureTask?.task_id) {
      params.set('structureTask', structureTask.task_id);
    }
    if (assayTask?.task_id) {
      params.set('assayTask', assayTask.task_id);
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
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: '1.8rem', marginBottom: 4 }}>BioChemInsight Workbench</h1>
        <p style={{ marginTop: 0, color: '#475569' }}>
          SAR curation for innovator patents—ingest the PDF once, resolve chemotypes, and review assay evidence in a single workspace.
        </p>
      </header>

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
            <div className="status-banner" style={{ marginTop: 16 }}>
              <span>
                Current file: <strong>{pdfInfo.filename}</strong>
              </span>
              <span className="badge">{pdfInfo.total_pages} page{pdfInfo.total_pages === 1 ? '' : 's'}</span>
            </div>
          )}
        </section>
      )}

      {pdfInfo && currentStep === 2 && (
        <section>
          <h2>2. Extract Structures</h2>
          <p style={{ marginTop: 0, color: '#64748b' }}>
            Select pages that contain chemical structures. Use thumbnails or type page ranges such as <code>1,3,5-8</code>.
          </p>
          <div className="selector-layout">
            <div className="selector-toolbar selector-toolbar--structure">
              <div className="selector-toolbar__header">
                <span className="toolbar-icon">
                  <StepGlyph type="structures" className="toolbar-icon__glyph" />
                </span>
                <h3 className="selector-toolbar__title">Structure selection</h3>
              </div>
              <div className="selector-field">
                <label htmlFor="structure-pages">Structure pages</label>
                <textarea
                  id="structure-pages"
                  placeholder="e.g. 12-18, 24, 31"
                  value={structurePagesInput}
                  onChange={handleStructureInputChange}
                  rows={3}
                />
            <div className="flex-gap" style={{ marginTop: 8 }}>
              <span className="tag">{structureSelection.size} page{structureSelection.size === 1 ? '' : 's'} selected</span>
              <button className="small-btn" onClick={clearStructureSelection} type="button">
                Clear
              </button>
            </div>
              </div>

              <div className="selector-actions">
                <div className="selector-actions__group structure">
                  <button
                    className="primary structure-action"
                    type="button"
                    onClick={handleStartStructureExtraction}
                    disabled={!pdfInfo || structureSelection.size === 0 || isStructureSubmitting}
                  >
                    {isStructureSubmitting ? 'Extracting structures…' : 'Run structure extraction'}
                  </button>
                </div>
                <div className="selector-actions__links flex-gap">
                  <button className="small-btn subtle" type="button" onClick={() => setCurrentStep(3)}>
                    Continue to bioactivity extraction
                  </button>
                  <button className="small-btn" type="button" onClick={() => setCurrentStep(4)} disabled={!canViewResults}>
                    Go to results
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
          <h2>3. Extract Bioactivity</h2>
          <p style={{ marginTop: 0, color: '#64748b' }}>
            Select the assay tables and add assay names so the parser focuses on the right content.
          </p>
          <div className="selector-layout">
            <div className="selector-toolbar selector-toolbar--assay">
              <div className="selector-toolbar__header">
                <span className="toolbar-icon">
                  <StepGlyph type="bioactivity" className="toolbar-icon__glyph" />
                </span>
                <h3 className="selector-toolbar__title">Assay selection</h3>
              </div>
              <div className="selector-field">
                <label htmlFor="assay-pages">Bioactivity pages</label>
                <textarea
                  id="assay-pages"
                  placeholder="e.g. 42, 76-81"
                  value={assayPagesInput}
                  onChange={handleAssayInputChange}
                  rows={3}
                />
            <div className="flex-gap" style={{ marginTop: 8 }}>
              <span className="tag">{assaySelection.size} page{assaySelection.size === 1 ? '' : 's'} selected</span>
              <button className="small-btn" onClick={clearAssaySelection} type="button">
                Clear
              </button>
            </div>
              </div>
              <div className="selector-field">
                <label htmlFor="assay-names">Assay names</label>
                <div className="assay-name-editor">
                  <input
                    id="assay-names"
                    type="text"
                    placeholder="Press Enter or comma to add"
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
                    assaySelection.size === 0 ||
                    (assayNames.length === 0 && !assayNameDraft.trim()) ||
                    isAssaySubmitting
                  }
                >
                  {isAssaySubmitting ? 'Extracting bioactivity…' : 'Run bioactivity extraction'}
                </button>
              </div>
              <div className="selector-actions__links flex-gap">
                <button className="small-btn subtle" type="button" onClick={() => setCurrentStep(2)}>
                  Back to structure extraction
                </button>
                <button className="small-btn" type="button" onClick={() => setCurrentStep(4)} disabled={!canViewResults}>
                  Go to results
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
          <h2>4. Review Results</h2>
          <p style={{ color: '#64748b', marginTop: 0 }}>
            Review the extracted structures and assay records, apply edits, and export tables.
          </p>
          {(structures.length === 0 && assayRecords.length === 0) && (
            <p style={{ color: '#94a3b8' }}>Extraction results appear here once the tasks finish.</p>
          )}
          {(structures.length > 0 || assayRecords.length > 0) && (
            <div className="review-layout">
              <aside className="review-sidebar">
                <div className="side-card">
                  <h3 className="side-card__title">Adjust extractions</h3>
                  <p className="side-card__body">
                    Need to refine page selections before exporting? Jump back to the earlier steps.
                  </p>
                  <button
                    className="secondary side-card__button"
                    type="button"
                    onClick={() => setCurrentStep(2)}
                  >
                    Revisit structure extraction
                  </button>
                  <button
                    className="secondary side-card__button"
                    type="button"
                    onClick={() => setCurrentStep(3)}
                  >
                    Revisit bioactivity extraction
                  </button>
                </div>
                <div className="side-card">
                  <h3 className="side-card__title">Exports</h3>
                  <p className="side-card__body">Download the latest saved tables whenever you need.</p>
                  <button className="secondary side-card__button" type="button" onClick={downloadStructuresCsv}>
                    Download structures CSV
                  </button>
                  {assayColumnNames.length > 0 && (
                    <button
                      className="secondary side-card__button"
                      type="button"
                      onClick={downloadAssayCsv}
                      disabled={assayRecords.length === 0}
                    >
                      Download bioactivity CSV
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
                <div className="table-wrapper" style={tableStyle as React.CSSProperties}>
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
                          <th>🧪 {column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {processedStructureRows.map((row) => {
                        const {
                          record,
                          structureImage,
                          structureSource,
                          segmentImage,
                          segmentSource,
                          smilesPreview,
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
                          <tr key={row.id ?? index}>
                            {(!normalizedPrimaryPage || showPageCell) && pageCell}
                            <td className="review-table__cell review-table__cell--preview">
                              <div className="page-cell">
                                <div className="page-cell__media">
                                  {structureImage ? (
                                    <button
                                      type="button"
                                      className="page-cell__image"
                                      onClick={() =>
                                        openArtifact(structureSource || structureImage || '', artifactLabel, {
                                          rowIndex: canEditStructure ? index : null,
                                        })
                                      }
                                    >
                                      <img src={structureImage} alt="PDF page" />
                                    </button>
                                  ) : (
                                    <div className="page-cell__placeholder">No preview</div>
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
                                <button
                                  type="button"
                                  className="structure-image-btn"
                                  onClick={() =>
                                    openArtifact(segmentSource || segmentImage || '', `Source structure - ${record.COMPOUND_ID ?? ''}`, {
                                      rowIndex: canEditStructure ? index : null,
                                    })
                                  }
                                >
                                  <img src={segmentImage} alt="Source structure" />
                                </button>
                              ) : (
                                <span className="muted">None</span>
                              )}
                            </td>
                            <td className="review-table__cell review-table__cell--structure">
                              <div className="structure-cell-content">
                                <button
                                  type="button"
                                  className={smilesPreview ? 'structure-image-btn' : 'smiles-chip'}
                                  onClick={() => {
                                    // 如果有SMILES预览，显示Quick edit面板
                                    if (smilesPreview) {
                                      setModalArtifact({
                                        path: `Extracted structure - ${record.COMPOUND_ID ?? ''}`,
                                        mime: 'image/png',
                                        data: smilesPreview,
                                        rowIndex: canEditStructure ? index : null,
                                      });
                                    } else {
                                      // 否则打开结构编辑器
                                      handleOpenStructureEditor(index);
                                    }
                                  }}
                                  disabled={!canEditStructure}
                                >
                                  {smilesPreview ? (
                                    <img src={smilesPreview} alt="Extracted structure" />
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
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolNexTR' ? '解析中...' : 'MolNexTR'}
                                    </button>
                                    <button
                                      type="button"
                                      className="reparse-btn"
                                      onClick={() => handleReparseStructure(index, 'MolVec')}
                                      disabled={reparseState.rowIndex === index && reparseState.engine === 'MolVec'}
                                    >
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolVec' ? '解析中...' : 'MolVec'}
                                    </button>
                                    <button
                                      type="button"
                                      className="reparse-btn"
                                      onClick={() => handleReparseStructure(index, 'MolScribe')}
                                      disabled={reparseState.rowIndex === index && reparseState.engine === 'MolScribe'}
                                    >
                                      {reparseState.rowIndex === index && reparseState.engine === 'MolScribe' ? '解析中...' : 'MolScribe'}
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
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
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
            <path d="M12 6 5 13h4v5h6v-5h4L12 6Z" fill="currentColor" />
          </svg>
        </button>
      )}

      <StructureEditorModal
        open={editorState.open}
        initialSmiles={editorState.smiles}
        onCancel={handleStructureEditorCancel}
        onSave={handleStructureEditorSave}
      />
    </div>
  );
};

export default App;

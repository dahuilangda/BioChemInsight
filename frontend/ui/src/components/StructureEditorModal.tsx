import React from 'react';
import OCL from 'openchemlib/full';
import { renderSmiles } from '../api/client';
import './structure-editor.css';

interface StructureEditorModalProps {
  open: boolean;
  initialSmiles: string;
  onCancel: () => void;
  onSave: (payload: { smiles: string; image?: string }) => void;
}

type StructureEditorInstance = InstanceType<typeof OCL.StructureEditor>;

const StructureEditorModal: React.FC<StructureEditorModalProps> = ({
  open,
  initialSmiles,
  onCancel,
  onSave,
}) => {
  const editorContainerRef = React.useRef<HTMLDivElement | null>(null);
  const editorRef = React.useRef<StructureEditorInstance | null>(null);
  const [editorReady, setEditorReady] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(true);
  const [isSaving, setIsSaving] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const panelRef = React.useRef<HTMLDivElement | null>(null);
  const dragRef = React.useRef<{ offsetX: number; offsetY: number } | null>(null);
  const resizeRef = React.useRef<{
    pointerId: number;
    startWidth: number;
    startHeight: number;
    startX: number;
    startY: number;
  } | null>(null);
  const MIN_WIDTH = 640;
  const MIN_HEIGHT = 520;
  const [panelPosition, setPanelPosition] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panelSize, setPanelSize] = React.useState<{ width: number; height: number }>({
    width: MIN_WIDTH,
    height: MIN_HEIGHT,
  });
  const loadStructure = React.useCallback((smiles: string) => {
    const editor = editorRef.current;
    if (!editor) return;

    const trimmed = smiles?.trim?.() ?? '';

    if (!trimmed) {
      try {
        editor.setMolFile('');
      } catch (clearError) {
        console.warn('Failed to clear the structure editor.', clearError);
      }
      return;
    }

    try {
      editor.setSmiles(trimmed);
      return;
    } catch (primaryError) {
      try {
        const molecule = OCL.Molecule.fromSmiles(trimmed);
        const molfile = molecule.toMolfileV3();
        editor.setMolFile(molfile);
        return;
      } catch (secondaryError) {
        const message =
          secondaryError instanceof Error
            ? secondaryError.message
            : typeof secondaryError === 'string'
              ? secondaryError
              : 'Unable to load the provided structure.';
        throw new Error(message);
      }
    }
  }, []);

  React.useEffect(() => {
    if (!open) {
      setEditorReady(false);
      setIsLoading(true);
      setErrorMessage(null);
      dragRef.current = null;
      setPanelPosition({ x: 0, y: 0 });
      return;
    }

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const availableWidth = Math.max(MIN_WIDTH, viewportWidth - 48);
    const availableHeight = Math.max(MIN_HEIGHT, viewportHeight - 48);
    const defaultWidth = Math.min(960, availableWidth);
    const defaultHeight = Math.min(680, availableHeight);
    const nextX = Math.max(24, (viewportWidth - defaultWidth) / 2);
    const nextY = Math.max(24, (viewportHeight - defaultHeight) / 2);
    setPanelPosition({ x: nextX, y: nextY });
    setPanelSize({ width: defaultWidth, height: defaultHeight });

    const container = editorContainerRef.current;
    if (!container) return;

    container.innerHTML = '';
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const editor = new OCL.StructureEditor(container, true, 1.1);
      editorRef.current = editor;
      setEditorReady(true);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Failed to initialize the structure editor.';
      setErrorMessage(message);
      setEditorReady(false);
      setIsLoading(false);
    }

    return () => {
      setEditorReady(false);
      const currentEditor = editorRef.current;
      if (currentEditor && typeof (currentEditor as { destroy?: () => void }).destroy === 'function') {
        try {
          (currentEditor as { destroy: () => void }).destroy();
        } catch (destroyError) {
          console.warn('Structure editor cleanup failed', destroyError);
        }
      }
      editorRef.current = null;
      container.innerHTML = '';
    };
  }, [open]);

  React.useLayoutEffect(() => {
    if (!open || !panelRef.current) return;
    const { innerWidth, innerHeight } = window;
    const rect = panelRef.current.getBoundingClientRect();
    setPanelPosition((prev) => {
      const clampedX = Math.min(Math.max(24, prev.x), Math.max(24, innerWidth - rect.width - 24));
      const clampedY = Math.min(Math.max(24, prev.y), Math.max(24, innerHeight - rect.height - 24));
      if (prev.x === clampedX && prev.y === clampedY) {
        return prev;
      }
      return { x: clampedX, y: clampedY };
    });
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    const handleResize = () => {
      const panel = panelRef.current;
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      setPanelPosition((prev) => {
        const clampedX = Math.min(Math.max(24, prev.x), Math.max(24, window.innerWidth - rect.width - 24));
        const clampedY = Math.min(Math.max(24, prev.y), Math.max(24, window.innerHeight - rect.height - 24));
        if (prev.x === clampedX && prev.y === clampedY) return prev;
        return { x: clampedX, y: clampedY };
      });
    };
    window.addEventListener('resize', handleResize);

    let resizeObserver: ResizeObserver | null = null;
    if (window.ResizeObserver && panelRef.current) {
      resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(panelRef.current);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [open]);

  const handlePanelPointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (!panel) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = panel.getBoundingClientRect();
    dragRef.current = { offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
    resizeRef.current = null;
    panel.setPointerCapture(event.pointerId);
  }, []);

  const handlePanelPointerMove = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const resizeState = resizeRef.current;
    if (resizeState) {
      event.preventDefault();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const deltaX = event.clientX - resizeState.startX;
      const deltaY = event.clientY - resizeState.startY;
      const proposedWidth = resizeState.startWidth + deltaX;
      const proposedHeight = resizeState.startHeight + deltaY;
      const maxWidth = Math.max(MIN_WIDTH, viewportWidth - panelPosition.x - 24);
      const maxHeight = Math.max(MIN_HEIGHT, viewportHeight - panelPosition.y - 24);
      const width = Math.min(Math.max(MIN_WIDTH, proposedWidth), maxWidth);
      const height = Math.min(Math.max(MIN_HEIGHT, proposedHeight), maxHeight);
      setPanelSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
      return;
    }

    if (!dragRef.current) return;
    event.preventDefault();
    const panel = panelRef.current;
    const { offsetX, offsetY } = dragRef.current;
    const width = panel?.offsetWidth ?? panelSize.width;
    const height = panel?.offsetHeight ?? panelSize.height;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const rawX = event.clientX - offsetX;
    const rawY = event.clientY - offsetY;
    const maxX = Math.max(24, viewportWidth - width - 24);
    const maxY = Math.max(24, viewportHeight - height - 24);
    const clampedX = Math.min(Math.max(24, rawX), maxX);
    const clampedY = Math.min(Math.max(24, rawY), maxY);
    setPanelPosition((prev) => (prev.x === clampedX && prev.y === clampedY ? prev : { x: clampedX, y: clampedY }));
  }, [MIN_HEIGHT, MIN_WIDTH, panelPosition.x, panelPosition.y, panelSize.height, panelSize.width]);

  const handlePanelPointerUp = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (panel && panel.hasPointerCapture(event.pointerId)) {
      panel.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    resizeRef.current = null;
  }, []);

  const handleResizePointerDown = React.useCallback((event: React.PointerEvent<HTMLSpanElement>) => {
    const panel = panelRef.current;
    if (!panel) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = panel.getBoundingClientRect();
    resizeRef.current = {
      pointerId: event.pointerId,
      startWidth: rect.width,
      startHeight: rect.height,
      startX: event.clientX,
      startY: event.clientY,
    };
    dragRef.current = null;
    panel.setPointerCapture(event.pointerId);
  }, []);

  React.useEffect(() => {
    if (!open || !editorReady) return;

    setIsLoading(true);
    setErrorMessage(null);
    try {
      loadStructure(initialSmiles ?? '');
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Unable to load the initial structure. Please draw it manually.';
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  }, [open, editorReady, initialSmiles, loadStructure]);

  const handleSave = React.useCallback(async () => {
    const editor = editorRef.current;
    if (!editor) return;
    setIsSaving(true);
    setErrorMessage(null);
    try {
      const smiles = editor.getSmiles()?.trim();
      if (!smiles) {
        setErrorMessage('Draw or import a structure before saving.');
        return;
      }
      let image: string | undefined;
      try {
        image = await renderSmiles(smiles, { width: 240, height: 180 });
      } catch (renderError) {
        console.warn('Failed to render structure preview; SMILES will still be saved.', renderError);
      }
      onSave({ smiles, image });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Unable to save the structure.';
      setErrorMessage(message);
    } finally {
      setIsSaving(false);
    }
  }, [onSave]);

  React.useEffect(() => {
    if (!open) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onCancel();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className="floating-layer floating-layer--structure" role="presentation">
      <button
        type="button"
        className="floating-layer__backdrop"
        aria-label="Close structure editor"
        onClick={onCancel}
      />
      <div
        ref={panelRef}
        className="floating-panel floating-panel--structure"
        style={{ top: panelPosition.y, left: panelPosition.x, width: panelSize.width, height: panelSize.height }}
        onPointerMove={handlePanelPointerMove}
        onPointerUp={handlePanelPointerUp}
        onPointerCancel={handlePanelPointerUp}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="floating-panel__handle" onPointerDown={handlePanelPointerDown}>
          <span className="floating-panel__title">Edit structure</span>
          <button
            type="button"
            className="floating-panel__close"
            onClick={(event) => {
              event.stopPropagation();
              onCancel();
            }}
          >
            ✕
          </button>
        </div>
        <div className="floating-panel__body floating-panel__body--structure">
          <div className="structure-editor">
            <div className="structure-editor__editor-wrapper">
              <div ref={editorContainerRef} className="structure-editor__editor-host" />
              {isLoading && (
                <div className="structure-editor__overlay">
                  <div className="spinner" />
                  <span>Loading structure editor…</span>
                </div>
              )}
            </div>
            {errorMessage && <div className="structure-editor__error">{errorMessage}</div>}
          </div>
          <div className="floating-panel__footer">
            <button className="primary" type="button" onClick={handleSave} disabled={isSaving || isLoading}>
              {isSaving ? 'Saving…' : 'Save'}
            </button>
            <button className="secondary" type="button" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
        <span
          className="floating-panel__resize-handle"
          onPointerDown={handleResizePointerDown}
          aria-label="Resize structure editor"
          role="presentation"
        />
      </div>
    </div>
  );
};

export default StructureEditorModal;

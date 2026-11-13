import React from 'react';
import { prepareJSMEMolblock, renderSmiles } from '../api/client';
import {
  buildJSMEInitOptions,
  getJSMEMolfile,
  getJSMESmiles,
  loadJSME,
  type JsmeApplet,
} from '../utils/jsme';
import './structure-editor.css';

interface StructureEditorModalProps {
  open: boolean;
  initialSmiles: string;
  initialMolblock?: string;
  sourceImage?: string;
  proposalName?: string;
  compoundLabel?: string;
  onCancel: () => void;
  onSave: (payload: { smiles: string; molblock?: string; image?: string }) => void;
}

function viewportBounds() {
  const margin = window.innerWidth <= 640 || window.innerHeight < 560 ? 8 : 24;
  const width = Math.max(1, window.innerWidth - margin * 2);
  const height = Math.max(1, window.innerHeight - margin * 2);
  return { margin, width, height, minWidth: Math.min(640, width), minHeight: Math.min(520, height) };
}

const StructureEditorModal: React.FC<StructureEditorModalProps> = ({
  open,
  initialSmiles,
  initialMolblock,
  sourceImage,
  proposalName,
  compoundLabel,
  onCancel,
  onSave,
}) => {
  const editorHostRef = React.useRef<HTMLDivElement | null>(null);
  const editorMountRef = React.useRef<HTMLDivElement | null>(null);
  const editorRef = React.useRef<JsmeApplet | null>(null);
  const dirtyRef = React.useRef(false);
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
  const editorIdRef = React.useRef(`jsme-editor-modal-${Math.random().toString(36).slice(2)}`);
  const MIN_WIDTH = 640;
  const MIN_HEIGHT = 520;
  const [panelPosition, setPanelPosition] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panelSize, setPanelSize] = React.useState<{ width: number; height: number }>({
    width: MIN_WIDTH,
    height: MIN_HEIGHT,
  });

  const applyEditorSize = React.useCallback(() => {
    const host = editorHostRef.current;
    const mount = editorMountRef.current;
    const editor = editorRef.current;
    if (!host || !mount) return;

    const width = Math.max(1, Math.floor(host.clientWidth));
    const height = Math.max(1, Math.floor(host.clientHeight));
    mount.style.width = '100%';
    mount.style.height = '100%';
    if (!editor?.setSize) return;

    try {
      editor.setSize(width, height);
    } catch {
      try {
        editor.setSize(`${width}`, `${height}`);
      } catch {
        try {
          editor.setSize('100%', '100%');
        } catch {
          // JSME versions differ in setSize signatures.
        }
      }
    }
  }, []);

  React.useEffect(() => {
    if (!open) {
      setIsLoading(true);
      setErrorMessage(null);
      dragRef.current = null;
      setPanelPosition({ x: 0, y: 0 });
      return;
    }

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const bounds = viewportBounds();
    const availableWidth = bounds.width;
    const availableHeight = bounds.height;
    const defaultWidth = Math.min(960, availableWidth);
    const defaultHeight = Math.min(680, availableHeight);
    const nextX = Math.max(bounds.margin, (viewportWidth - defaultWidth) / 2);
    const nextY = Math.max(bounds.margin, (viewportHeight - defaultHeight) / 2);
    setPanelPosition({ x: nextX, y: nextY });
    setPanelSize({ width: defaultWidth, height: defaultHeight });

    const mount = editorMountRef.current;
    if (!mount) return;

    let cancelled = false;
    mount.innerHTML = '';
    mount.id = editorIdRef.current;
    editorRef.current = null;
    setIsLoading(true);
    setErrorMessage(null);

    const boot = async () => {
      try {
        const JSApplet = await loadJSME();
        if (cancelled || !editorMountRef.current) return;

        editorMountRef.current.innerHTML = '';
        editorMountRef.current.id = editorIdRef.current;
        const initialMolblockForEditor = await prepareJSMEMolblock(initialMolblock || '').catch((err) => {
          console.warn('Molblock normalization failed; loading raw block.', err);
          return initialMolblock || '';
        });
        if (cancelled || !editorMountRef.current) return;
        const editor = new JSApplet.JSME(
          editorIdRef.current,
          '100%',
          '100%',
          buildJSMEInitOptions(initialSmiles ?? '', initialMolblockForEditor),
        );
        const markDirty = () => {
          dirtyRef.current = true;
        };
        editor.setCallBack?.('AfterStructureModified', markDirty);
        editor.setAfterStructureModifiedCallback?.(markDirty);
        editorRef.current = editor;
        dirtyRef.current = false;
        setIsLoading(false);
        window.requestAnimationFrame(applyEditorSize);
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'string'
              ? err
            : 'Failed to initialize the structure editor.';
        setErrorMessage(message);
        setIsLoading(false);
      }
    };

    void boot();

    return () => {
      cancelled = true;
      editorRef.current = null;
      if (mount) {
        mount.innerHTML = '';
      }
    };
  }, [MIN_HEIGHT, MIN_WIDTH, applyEditorSize, initialMolblock, initialSmiles, open]);

  React.useEffect(() => {
    if (!open) return;
    const handleResize = () => {
      const panel = panelRef.current;
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      const bounds = viewportBounds();
      const width = Math.min(rect.width, bounds.width);
      const height = Math.min(rect.height, bounds.height);
      setPanelSize((prev) => prev.width === width && prev.height === height ? prev : { width, height });
      setPanelPosition((prev) => {
        const clampedX = Math.min(Math.max(bounds.margin, prev.x), window.innerWidth - width - bounds.margin);
        const clampedY = Math.min(Math.max(bounds.margin, prev.y), window.innerHeight - height - bounds.margin);
        if (prev.x === clampedX && prev.y === clampedY) return prev;
        return { x: clampedX, y: clampedY };
      });
      applyEditorSize();
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
  }, [applyEditorSize, open]);

  const handlePanelPointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (!panel || (event.target as HTMLElement).closest('button, input, select, textarea')) return;
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
      const bounds = viewportBounds();
      const maxWidth = Math.max(1, viewportWidth - panelPosition.x - bounds.margin);
      const maxHeight = Math.max(1, viewportHeight - panelPosition.y - bounds.margin);
      const width = Math.min(Math.max(bounds.minWidth, proposedWidth), maxWidth);
      const height = Math.min(Math.max(bounds.minHeight, proposedHeight), maxHeight);
      setPanelSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
      window.requestAnimationFrame(applyEditorSize);
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
    const { margin } = viewportBounds();
    const maxX = Math.max(margin, viewportWidth - width - margin);
    const maxY = Math.max(margin, viewportHeight - height - margin);
    const clampedX = Math.min(Math.max(margin, rawX), maxX);
    const clampedY = Math.min(Math.max(margin, rawY), maxY);
    setPanelPosition((prev) => (prev.x === clampedX && prev.y === clampedY ? prev : { x: clampedX, y: clampedY }));
  }, [MIN_HEIGHT, MIN_WIDTH, applyEditorSize, panelPosition.x, panelPosition.y, panelSize.height, panelSize.width]);

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

  const handleSave = React.useCallback(async () => {
    const editor = editorRef.current;
    if (!editor) return;
    setIsSaving(true);
    setErrorMessage(null);
    try {
      const editorSmiles = getJSMESmiles(editor);
      const editorMolblock = getJSMEMolfile(editor);
      const smiles = editorSmiles;
      const molblock = (editorMolblock || '').trimEnd();
      if (!molblock) {
        setErrorMessage('Draw or import a structure before saving.');
        return;
      }
      if (!smiles) {
        setErrorMessage('Draw or import a structure before saving.');
        return;
      }
      let image: string | undefined;
      try {
        image = await renderSmiles(smiles, { width: 240, height: 180, molblock });
      } catch (renderError) {
        console.warn('Failed to render structure preview; SMILES will still be saved.', renderError);
      }
      onSave({ smiles, molblock, image });
      dirtyRef.current = false;
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
  }, [initialMolblock, initialSmiles, onSave]);

  React.useEffect(() => {
    if (!open) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancel();
      }
      if (event.key === 'Tab') {
        const controls = Array.from(panelRef.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]',
        ) ?? []).filter((element) => element.getClientRects().length > 0);
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && (document.activeElement === first || !panelRef.current?.contains(document.activeElement))) {
          event.preventDefault(); last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault(); first?.focus();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onCancel]);

  React.useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    panelRef.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [open]);

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
        role="dialog"
        aria-modal="true"
        aria-label="Edit structure"
        className="floating-panel floating-panel--structure"
        style={{ top: panelPosition.y, left: panelPosition.x, width: panelSize.width, height: panelSize.height }}
        onPointerMove={handlePanelPointerMove}
        onPointerUp={handlePanelPointerUp}
        onPointerCancel={handlePanelPointerUp}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="floating-panel__handle" onPointerDown={handlePanelPointerDown}>
          <div className="structure-editor__heading"><span className="floating-panel__title">Edit molecular structure</span><span className="structure-editor__subtitle">{compoundLabel ? `Compound ${compoundLabel} · ` : ""}JSME editor</span></div>
          <button
            type="button"
            className="floating-panel__close"
            aria-label="Close structure editor"
            onClick={(event) => {
              event.stopPropagation();
              onCancel();
            }}
          >
            ×
          </button>
        </div>
        <div className="floating-panel__body floating-panel__body--structure">
          <div className="structure-editor__workspace">
          {(sourceImage || proposalName) && <aside className="structure-editor__source"><span>{proposalName ? 'Source chemical name' : 'Original structure'}</span>{sourceImage && <img src={sourceImage} alt="Original structure reference" />}{proposalName ? <><p>{proposalName}</p><p>Proposed from the printed name. Verify connectivity and stereochemistry before saving.</p></> : <p>Compare the source as you correct atoms, bonds and stereochemistry.</p>}</aside>}
          <div className="structure-editor">
            <div className="structure-editor__editor-wrapper">
              <div ref={editorHostRef} className="structure-editor__editor-host">
                <div ref={editorMountRef} className="structure-editor__jsme-mount" />
              </div>
              {isLoading && (
                <div className="structure-editor__overlay">
                  <div className="spinner" />
                  <span>Loading structure editor…</span>
                </div>
              )}
            </div>
            {errorMessage && <div className="structure-editor__error">{errorMessage}</div>}
          </div>
          </div>
          <div className="floating-panel__footer">
            <span className="structure-editor__hint">Changes are saved to this compound.</span>
            <button className="primary" type="button" onClick={handleSave} disabled={isSaving || isLoading}>
              {isSaving ? 'Saving…' : 'Save structure'}
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

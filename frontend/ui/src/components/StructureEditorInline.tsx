import React from 'react';
import { renderSmiles } from '../api/client';
import {
  buildJSMEInitOptions,
  getJSMEMolfile,
  getJSMESmiles,
  loadJSME,
  type JsmeApplet,
} from '../utils/jsme';
import './structure-editor.css';

interface StructureEditorInlineProps {
  smiles: string;
  molblock?: string;
  disabled?: boolean;
  onSave: (payload: { smiles: string; molblock?: string; image?: string }) => void | Promise<void>;
  onReset?: () => void;
}

const StructureEditorInline: React.FC<StructureEditorInlineProps> = ({
  smiles,
  molblock,
  disabled = false,
  onSave,
  onReset,
}) => {
  const hostRef = React.useRef<HTMLDivElement | null>(null);
  const mountRef = React.useRef<HTMLDivElement | null>(null);
  const editorRef = React.useRef<JsmeApplet | null>(null);
  const editorIdRef = React.useRef(`jsme-editor-inline-${Math.random().toString(36).slice(2)}`);
  const dirtyRef = React.useRef(false);
  const [isLoading, setIsLoading] = React.useState(true);
  const [isSaving, setIsSaving] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const initialSmilesRef = React.useRef(smiles);
  const initialMolblockRef = React.useRef(molblock);
  const [editorVersion, setEditorVersion] = React.useState(0);

  const applyEditorSize = React.useCallback(() => {
    const host = hostRef.current;
    const mount = mountRef.current;
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
    const mount = mountRef.current;
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
        if (cancelled || !mountRef.current) return;

        mountRef.current.innerHTML = '';
        mountRef.current.id = editorIdRef.current;
        const editor = new JSApplet.JSME(
          editorIdRef.current,
          '100%',
          '100%',
          buildJSMEInitOptions(initialSmilesRef.current, initialMolblockRef.current),
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
  }, [applyEditorSize, editorVersion]);

  React.useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const handleResize = () => applyEditorSize();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(host);
    return () => resizeObserver.disconnect();
  }, [applyEditorSize]);

  React.useEffect(() => {
    initialSmilesRef.current = smiles;
    initialMolblockRef.current = molblock;
    if (editorRef.current) {
      dirtyRef.current = false;
      setEditorVersion((version) => version + 1);
    }
  }, [molblock, smiles]);

  const handleReset = React.useCallback(() => {
    dirtyRef.current = false;
    setEditorVersion((version) => version + 1);
    onReset?.();
  }, [onReset]);

  const handleSave = React.useCallback(async () => {
    if (disabled) return;
    const editor = editorRef.current;
    if (!editor) return;
    setIsSaving(true);
    setErrorMessage(null);
    try {
      const editorSmiles = getJSMESmiles(editor);
      const editorMolblock = getJSMEMolfile(editor);
      const nextSmiles = editorSmiles;
      const nextMolblock = editorMolblock || (initialMolblockRef.current || '').trim();
      if (!nextSmiles) {
        setErrorMessage('Draw or import a structure before saving.');
        return;
      }
      let image: string | undefined;
      try {
        image = await renderSmiles(nextSmiles, { width: 240, height: 180, molblock: nextMolblock });
      } catch (renderError) {
        console.warn('Failed to render structure preview; SMILES will still be saved.', renderError);
      }
      await Promise.resolve(onSave({ smiles: nextSmiles, molblock: nextMolblock, image }));
      initialSmilesRef.current = nextSmiles;
      initialMolblockRef.current = nextMolblock;
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
  }, [disabled, onSave]);

  return (
    <div className="structure-editor structure-editor--inline">
      <div className="structure-editor__editor-wrapper structure-editor__editor-wrapper--inline">
        <div ref={hostRef} className="structure-editor__editor-host">
          <div ref={mountRef} className="structure-editor__jsme-mount" />
        </div>
        {isLoading && (
          <div className="structure-editor__overlay">
            <div className="spinner" />
            <span>Loading...</span>
          </div>
        )}
      </div>
      {errorMessage && <div className="structure-editor__error" role="alert">{errorMessage}</div>}
      <div className="inline-editor__actions">
        <button
          className="primary"
          type="button"
          onClick={handleSave}
          disabled={disabled || isLoading || isSaving}
        >
          {isSaving ? 'Saving...' : 'Save'}
        </button>
        <button
          className="secondary"
          type="button"
          onClick={handleReset}
          disabled={isLoading || isSaving}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default StructureEditorInline;

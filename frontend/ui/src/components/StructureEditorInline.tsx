import React from 'react';
import OCL from 'openchemlib/full';
import { renderSmiles } from '../api/client';
import './structure-editor.css';

type StructureEditorInstance = InstanceType<typeof OCL.StructureEditor>;

interface StructureEditorInlineProps {
  smiles: string;
  disabled?: boolean;
  onSave: (payload: { smiles: string; image?: string }) => void | Promise<void>;
  onReset?: () => void;
}

const StructureEditorInline: React.FC<StructureEditorInlineProps> = ({ smiles, disabled = false, onSave, onReset }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const editorRef = React.useRef<StructureEditorInstance | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [isSaving, setIsSaving] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const initialSmilesRef = React.useRef(smiles);

  const loadStructure = React.useCallback((value: string) => {
    const editor = editorRef.current;
    if (!editor) return;

    const trimmed = value?.trim?.() ?? '';

    if (!trimmed) {
      try {
        editor.setMolFile('');
      } catch (err) {
        console.warn('Failed to clear editor', err);
      }
      return;
    }

    try {
      editor.setSmiles(trimmed);
    } catch (primaryError) {
      try {
        const molecule = OCL.Molecule.fromSmiles(trimmed);
        const molfile = molecule.toMolfileV3();
        editor.setMolFile(molfile);
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
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = '';
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const editor = new OCL.StructureEditor(container, true, 1.1);
      editor.setFragment(false);
      editorRef.current = editor;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Failed to initialize the structure editor.';
      setErrorMessage(message);
      setIsLoading(false);
      return () => undefined;
    }

    loadStructure(initialSmilesRef.current);
    setIsLoading(false);

    return () => {
      const editor = editorRef.current;
      if (editor && typeof (editor as { destroy?: () => void }).destroy === 'function') {
        try {
          (editor as { destroy: () => void }).destroy();
        } catch (destroyError) {
          console.warn('Structure editor cleanup failed', destroyError);
        }
      }
      editorRef.current = null;
      container.innerHTML = '';
    };
  }, [loadStructure]);

  React.useEffect(() => {
    if (!editorRef.current) return;
    initialSmilesRef.current = smiles;
    setIsLoading(true);
    setErrorMessage(null);
    try {
      loadStructure(smiles);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'string'
            ? err
            : 'Unable to load the provided structure.';
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  }, [smiles, loadStructure]);

  const handleReset = React.useCallback(() => {
    loadStructure(initialSmilesRef.current);
    onReset?.();
  }, [loadStructure, onReset]);

  const handleSave = React.useCallback(async () => {
    if (disabled) return;
    const editor = editorRef.current;
    if (!editor) return;
    setIsSaving(true);
    setErrorMessage(null);
    try {
      const nextSmiles = editor.getSmiles()?.trim();
      if (!nextSmiles) {
        setErrorMessage('Draw or import a structure before saving.');
        return;
      }
      let image: string | undefined;
      try {
        image = await renderSmiles(nextSmiles, { width: 240, height: 180 });
      } catch (renderError) {
        console.warn('Failed to render structure preview; SMILES will still be saved.', renderError);
      }
      await Promise.resolve(onSave({ smiles: nextSmiles, image }));
      initialSmilesRef.current = nextSmiles;
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
        <div ref={containerRef} className="structure-editor__editor-host" />
        {isLoading && (
          <div className="structure-editor__overlay">
            <div className="spinner" />
            <span>Loading…</span>
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
          {isSaving ? 'Saving…' : 'Save'}
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

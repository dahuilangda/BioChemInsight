import React from 'react';
import OCL from 'openchemlib/full';
import { renderSmiles } from '../api/client';
import { normalizeMolblock } from '../utils/chem';
import './structure-editor.css';

type StructureEditorInstance = InstanceType<typeof OCL.StructureEditor>;

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
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const editorRef = React.useRef<StructureEditorInstance | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [isSaving, setIsSaving] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const initialSmilesRef = React.useRef(smiles);
  const initialMolblockRef = React.useRef(molblock);

  // 响应窗口大小变化的函数
  const handleResize = React.useCallback(() => {
    const editor = editorRef.current;
    if (editor) {
      // 尝试调用编辑器的resize方法
      if (typeof (editor as any).resize === 'function') {
        try {
          (editor as any).resize();
        } catch (err) {
          console.warn('Failed to resize editor', err);
        }
      }
      // 如果没有resize方法，尝试其他方式
      else if (typeof (editor as any).update === 'function') {
        try {
          (editor as any).update();
        } catch (err) {
          console.warn('Failed to update editor', err);
        }
      }
    }
  }, []);

  // 更完善的手动resize处理函数
  const manualResizeEditor = React.useCallback(() => {
    const editor = editorRef.current;
    const container = containerRef.current;
    
    if (!editor || !container) return;
    
    try {
      // 获取容器的实际尺寸
      const containerRect = container.getBoundingClientRect();
      const width = containerRect.width;
      const height = containerRect.height;
      
      // 尝试多种方法来调整编辑器大小
      // 方法1: 直接调用resize（如果存在）
      if (typeof (editor as any).resize === 'function') {
        (editor as any).resize();
        return;
      }
      
    } catch (err) {
      console.warn('Failed to manually resize editor', err);
    }
  }, []);

  const loadStructure = React.useCallback((value: string, molblockValue?: string) => {
    const editor = editorRef.current;
    if (!editor) return;

    const trimmed = value?.trim?.() ?? '';
    const molblockRaw = typeof molblockValue === 'string' ? molblockValue : '';
    const hasMolblock = Boolean(molblockRaw.trim());
    const normalizedMolblock = normalizeMolblock(molblockRaw);
    if (hasMolblock) {
      if (!normalizedMolblock) {
        throw new Error('Failed to normalize molblock.');
      }
      try {
        OCL.Molecule.fromMolfile(normalizedMolblock);
        editor.setMolFile(normalizedMolblock);
        return;
      } catch (molblockError) {
        const message =
          molblockError instanceof Error
            ? molblockError.message
            : typeof molblockError === 'string'
              ? molblockError
              : 'Failed to load molblock.';
        throw new Error(message);
      }
    }

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
      // 使用更适合的参数初始化编辑器
      const editor = new OCL.StructureEditor(container, true, 1.0);
      editor.setFragment(false);
      editorRef.current = editor;
      
      // 确保编辑器正确填充容器
      setTimeout(() => {
        handleResize();
        manualResizeEditor();
      }, 100);
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

    loadStructure(initialSmilesRef.current, initialMolblockRef.current);
    setIsLoading(false);

    // 添加一个定时器来确保编辑器正确渲染
    const resizeTimer = setTimeout(() => {
      handleResize();
      manualResizeEditor();
    }, 200);

    return () => {
      clearTimeout(resizeTimer);
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
  }, [loadStructure, handleResize, manualResizeEditor]);

  React.useEffect(() => {
    if (!editorRef.current) return;
    initialSmilesRef.current = smiles;
    initialMolblockRef.current = molblock;
    setIsLoading(true);
    setErrorMessage(null);
    try {
      loadStructure(smiles, molblock);
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
  }, [smiles, molblock, loadStructure]);

  // 使用ResizeObserver来监听容器大小变化
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    // 创建ResizeObserver实例
    let resizeObserver: ResizeObserver | null = null;
    
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        // 容器大小变化时调整编辑器大小
        setTimeout(() => {
          handleResize();
          manualResizeEditor();
        }, 50);
      });
      
      resizeObserver.observe(container);
    }
    
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [handleResize, manualResizeEditor]);

  const handleReset = React.useCallback(() => {
    loadStructure(initialSmilesRef.current, initialMolblockRef.current);
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
      const nextMolblock = editor.getMolFile?.()?.trim?.() ?? '';
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

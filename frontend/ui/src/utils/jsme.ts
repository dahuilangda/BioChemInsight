import { loadScript } from './script';

const DEFAULT_JSME_SCRIPT_URL = '/jsme/jsme.nocache.js';
const JSME_SCRIPT_URL =
  (import.meta.env.VITE_JSME_SCRIPT_URL as string | undefined)?.trim() || DEFAULT_JSME_SCRIPT_URL;

export interface JsmeApplet {
  smiles?: () => string;
  molFile?: () => string;
  jmeFile?: () => string;
  readGenericMolecularInput?: (input: string) => void;
  readMolFile?: (input: string) => void;
  clear?: () => void;
  setSize?: (width: number | string, height: number | string) => void;
  setCallBack?: (eventName: string, callback: () => void) => void;
  setAfterStructureModifiedCallback?: (callback: () => void) => void;
}

interface JsAppletGlobal {
  JSME: new (
    containerId: string,
    width: string,
    height: string,
    options?: Record<string, unknown>,
  ) => JsmeApplet;
}

declare global {
  interface Window {
    JSApplet?: JsAppletGlobal;
    jsmeOnLoad?: () => void;
  }
}

export async function waitForJSMEReady(timeoutMs = 12000, intervalMs = 120): Promise<JsAppletGlobal> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (window.JSApplet?.JSME) {
      return window.JSApplet;
    }
    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
  throw new Error('JSME initialization timed out.');
}

export async function loadJSME(): Promise<JsAppletGlobal> {
  if (typeof window.jsmeOnLoad !== 'function') {
    window.jsmeOnLoad = () => {};
  }
  await loadScript(JSME_SCRIPT_URL);
  return waitForJSMEReady();
}

export function readStructureIntoJSME(applet: JsmeApplet, smiles: string, molblock?: string): void {
  const normalizedSmiles = smiles.trim();
  const normalizedMolblock = (molblock || '').trim();

  if (normalizedSmiles && typeof applet.readGenericMolecularInput === 'function') {
    try {
      applet.readGenericMolecularInput(normalizedSmiles);
      return;
    } catch {
      // Fall back to MOL input below.
    }
  }

  if (normalizedMolblock && typeof applet.readMolFile === 'function') {
    try {
      applet.readMolFile(normalizedMolblock);
      return;
    } catch {
      // Fall back to generic input below.
    }
  }

  if (normalizedMolblock && typeof applet.readGenericMolecularInput === 'function') {
    try {
      applet.readGenericMolecularInput(normalizedMolblock);
      return;
    } catch {
      // Clear the editor when neither input format can be loaded.
    }
  }

  applet.clear?.();
}

export function getJSMESmiles(applet: JsmeApplet): string {
  return applet.smiles?.().trim() || '';
}

export function getJSMEMolfile(applet: JsmeApplet): string {
  return applet.molFile?.().trim() || '';
}

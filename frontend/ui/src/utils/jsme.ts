import { loadScript } from './script';

const JSME_ASSET_BASE_URL = '/jsme/';
const DEFAULT_JSME_SCRIPT_URL = `${JSME_ASSET_BASE_URL}jsme.nocache.js?v=20240520-2`;
export const DEFAULT_JSME_OPTIONS = 'oldlook,star,query,hydrogens';
const JSME_SCRIPT_URL =
  (import.meta.env.VITE_JSME_SCRIPT_URL as string | undefined)?.trim() || DEFAULT_JSME_SCRIPT_URL;
let jsmeLoadPromise: Promise<JsAppletGlobal> | null = null;

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

function ensureJSMEModuleBase(): void {
  let meta = document.querySelector('meta[data-biocheminsight-jsme-base="true"]') as HTMLMetaElement | null;
  if (!meta) {
    meta = document.createElement('meta');
    meta.dataset.biocheminsightJsmeBase = 'true';
    document.head.appendChild(meta);
  }
  meta.name = 'jsme::gwt:property';
  meta.content = `baseUrl=${JSME_ASSET_BASE_URL}`;
}

export function normalizeJSMEMolblock(molblock?: string): string {
  return (molblock || '').replace(/\r\n/g, '\n').trimEnd();
}

interface V2000Atom {
  x: number;
  y: number;
  z: number;
  symbol: string;
  line: string;
}

interface V2000Bond {
  a1: number;
  a2: number;
}

interface MolfileAlias {
  atomIndex: number;
  label: string;
  lineIndex: number;
}

const V2000_COUNTS_LINE_PATTERN = /^\s*\d+\s+\d+\s+(?:\d+\s+){7,}\d+\s+V2000\s*$/;
const V2000_ATOM_LINE_PATTERN =
  /^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+([A-Za-z*][A-Za-z0-9*#]{0,2})\b/;
const V2000_BOND_LINE_PATTERN = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/;
const V2000_ALIAS_LINE_PATTERN = /^A\s+(\d+)\s*$/;

function getV2000AtomSymbol(line: string): string {
  return line.slice(31, 34).trim() || line.match(V2000_ATOM_LINE_PATTERN)?.[4] || '';
}

function setV2000AtomSymbol(line: string, symbol: string): string {
  if (line.length >= 34) {
    return `${line.slice(0, 31)}${symbol.padEnd(3)}${line.slice(34)}`;
  }
  return line.replace(V2000_ATOM_LINE_PATTERN, (_match, x, y, z) =>
    `${x.toString().padStart(10)}${y.toString().padStart(10)}${z.toString().padStart(10)} ${symbol.padEnd(3)}`,
  );
}

function formatV2000AtomLine(atom: V2000Atom): string {
  return `${atom.x.toFixed(4).padStart(10)}${atom.y.toFixed(4).padStart(10)}${atom.z
    .toFixed(4)
    .padStart(10)} ${atom.symbol.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`;
}

function formatV2000BondLine(a1: number, a2: number, order = 1): string {
  return `${a1.toString().padStart(3)}${a2.toString().padStart(3)}${order
    .toString()
    .padStart(3)}  0`;
}

function updateV2000CountsLine(line: string, atomCount: number, bondCount: number): string {
  return `${atomCount.toString().padStart(3)}${bondCount.toString().padStart(3)}${line.slice(6)}`;
}

function parseV2000Atom(line: string): V2000Atom | null {
  const match = line.match(V2000_ATOM_LINE_PATTERN);
  if (!match) return null;
  return {
    x: Number(match[1]),
    y: Number(match[2]),
    z: Number(match[3]),
    symbol: getV2000AtomSymbol(line),
    line,
  };
}

function parseV2000Bond(line: string): V2000Bond | null {
  const match = line.match(V2000_BOND_LINE_PATTERN);
  if (!match) return null;
  return {
    a1: Number(match[1]),
    a2: Number(match[2]),
  };
}

function getAverageBondLength(atoms: V2000Atom[], bonds: V2000Bond[]): number {
  const lengths = bonds
    .map((bond) => {
      const first = atoms[bond.a1 - 1];
      const second = atoms[bond.a2 - 1];
      if (!first || !second) return 0;
      return Math.hypot(first.x - second.x, first.y - second.y);
    })
    .filter((length) => Number.isFinite(length) && length > 0);
  if (!lengths.length) return 1.0;
  return lengths.reduce((sum, length) => sum + length, 0) / lengths.length;
}

function rotateVector(x: number, y: number, radians: number): { x: number; y: number } {
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
}

function createCF3FluorineAtoms(
  atomIndex: number,
  atoms: V2000Atom[],
  bonds: V2000Bond[],
  bondLength: number,
): V2000Atom[] {
  const center = atoms[atomIndex - 1];
  const neighborBond = bonds.find((bond) => bond.a1 === atomIndex || bond.a2 === atomIndex);
  const neighborIndex = neighborBond ? (neighborBond.a1 === atomIndex ? neighborBond.a2 : neighborBond.a1) : null;
  const neighbor = neighborIndex ? atoms[neighborIndex - 1] : null;

  let dx = 1;
  let dy = 0;
  if (center && neighbor) {
    dx = center.x - neighbor.x;
    dy = center.y - neighbor.y;
  }
  const length = Math.hypot(dx, dy) || 1;
  const unitX = dx / length;
  const unitY = dy / length;
  const cfBondLength = bondLength * 0.85;

  return [0, (2 * Math.PI) / 3, (-2 * Math.PI) / 3].map((angle) => {
    const vector = rotateVector(unitX, unitY, angle);
    return {
      x: center.x + vector.x * cfBondLength,
      y: center.y + vector.y * cfBondLength,
      z: center.z,
      symbol: 'F',
      line: '',
    };
  });
}

function expandCF3AliasesForJSME(molblock: string): string {
  const lines = molblock.split('\n');
  const countsLineIndex = lines.findIndex((line) => V2000_COUNTS_LINE_PATTERN.test(line));
  if (countsLineIndex < 0) return molblock;

  const countMatch = lines[countsLineIndex].match(/^\s*(\d+)\s+(\d+)/);
  if (!countMatch) return molblock;

  const atomCount = Number(countMatch[1]);
  const bondCount = Number(countMatch[2]);
  const atomStart = countsLineIndex + 1;
  const bondStart = atomStart + atomCount;
  const propertyStart = bondStart + bondCount;
  if (atomCount <= 0 || bondCount < 0 || propertyStart > lines.length) return molblock;

  const atoms = lines.slice(atomStart, bondStart).map(parseV2000Atom);
  const bonds = lines.slice(bondStart, propertyStart).map(parseV2000Bond);
  if (atoms.some((atom) => !atom) || bonds.some((bond) => !bond)) return molblock;

  const parsedAtoms = atoms as V2000Atom[];
  const parsedBonds = bonds as V2000Bond[];
  const aliases: MolfileAlias[] = [];
  for (let index = propertyStart; index < lines.length - 1; index += 1) {
    const aliasMatch = lines[index].match(V2000_ALIAS_LINE_PATTERN);
    if (!aliasMatch) continue;
    aliases.push({
      atomIndex: Number(aliasMatch[1]),
      label: lines[index + 1].trim(),
      lineIndex: index,
    });
  }

  const cf3Aliases = aliases.filter((alias) => {
    const atom = parsedAtoms[alias.atomIndex - 1];
    return alias.label.toUpperCase() === 'CF3' && atom?.symbol.toUpperCase() === 'R';
  });
  if (!cf3Aliases.length) return molblock;

  const aliasLinesToRemove = new Set<number>();
  const nextAtoms = [...parsedAtoms];
  const nextBondLines = lines.slice(bondStart, propertyStart);
  const averageBondLength = getAverageBondLength(parsedAtoms, parsedBonds);

  cf3Aliases.forEach((alias) => {
    const atom = nextAtoms[alias.atomIndex - 1];
    nextAtoms[alias.atomIndex - 1] = {
      ...atom,
      symbol: 'C',
      line: setV2000AtomSymbol(atom.line, 'C'),
    };
    const fluorines = createCF3FluorineAtoms(alias.atomIndex, parsedAtoms, parsedBonds, averageBondLength);
    fluorines.forEach((fluorine) => {
      const nextAtomIndex = nextAtoms.length + 1;
      nextAtoms.push({ ...fluorine, line: formatV2000AtomLine(fluorine) });
      nextBondLines.push(formatV2000BondLine(alias.atomIndex, nextAtomIndex));
    });
    aliasLinesToRemove.add(alias.lineIndex);
    aliasLinesToRemove.add(alias.lineIndex + 1);
  });

  const nextPropertyLines = lines
    .slice(propertyStart)
    .filter((_line, offset) => !aliasLinesToRemove.has(propertyStart + offset));

  return [
    ...lines.slice(0, countsLineIndex),
    updateV2000CountsLine(lines[countsLineIndex], nextAtoms.length, nextBondLines.length),
    ...nextAtoms.map((atom) => atom.line),
    ...nextBondLines,
    ...nextPropertyLines,
  ].join('\n');
}

function prepareMolblockForJSME(molblock: string): string {
  return expandCF3AliasesForJSME(molblock);
}

export function buildJSMEInitOptions(smiles: string, molblock?: string): Record<string, string> {
  const normalizedMolblock = normalizeJSMEMolblock(molblock);
  const normalizedSmiles = smiles.trim();
  const options: Record<string, string> = {
    options: DEFAULT_JSME_OPTIONS,
  };

  if (normalizedMolblock) {
    options.mol = prepareMolblockForJSME(normalizedMolblock);
  } else if (normalizedSmiles) {
    options.smiles = normalizedSmiles;
  }

  return options;
}

export async function loadJSME(): Promise<JsAppletGlobal> {
  if (window.JSApplet?.JSME) {
    return window.JSApplet;
  }
  if (jsmeLoadPromise) {
    return jsmeLoadPromise;
  }

  ensureJSMEModuleBase();
  const previousOnLoad = typeof window.jsmeOnLoad === 'function' ? window.jsmeOnLoad : undefined;
  jsmeLoadPromise = new Promise<JsAppletGlobal>((resolve, reject) => {
    const resolveReady = () => {
      void waitForJSMEReady(3000, 50).then(resolve, reject);
    };
    window.jsmeOnLoad = () => {
      try {
        previousOnLoad?.();
      } finally {
        resolveReady();
      }
    };
    void loadScript(JSME_SCRIPT_URL)
      .then(() => waitForJSMEReady())
      .then(resolve)
      .catch((error) => {
        jsmeLoadPromise = null;
        reject(error);
      });
  });
  return jsmeLoadPromise;
}

export function getJSMESmiles(applet: JsmeApplet): string {
  return applet.smiles?.().trim() || '';
}

export function getJSMEMolfile(applet: JsmeApplet): string {
  return applet.molFile?.().trimEnd() || '';
}

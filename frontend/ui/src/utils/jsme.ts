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

interface AliasTemplateAtom {
  symbol: string;
  x: number;
  y: number;
  z: number;
}

interface AliasTemplateBond {
  from: number;
  to: number;
  order?: number;
}

interface AliasExpansionDefinition {
  anchor: string;
  atoms: AliasTemplateAtom[];
  bonds: AliasTemplateBond[];
  referenceAtomIndex?: number;
  scale?: number;
}

const V2000_COUNTS_LINE_PATTERN = /^\s*\d+\s+\d+\s+(?:\d+\s+){7,}\d+\s+V2000\s*$/;
const V2000_ATOM_LINE_PATTERN =
  /^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+([A-Za-z*][A-Za-z0-9*#]{0,2})\b/;
const V2000_BOND_LINE_PATTERN = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/;
const V2000_ALIAS_LINE_PATTERN = /^A\s+(\d+)\s*$/;
const SQRT3_OVER_2 = Math.sqrt(3) / 2;

function normalizeAliasLabel(label: string): string {
  return label.replace(/[\s_\-\[\]\(\)]/g, '').toUpperCase();
}

function resolveAliasDefinition(label: string): AliasExpansionDefinition | undefined {
  const compactLabel = label.replace(/[\s_\-\[\]\(\)]/g, '');
  if (compactLabel === 'MeS') {
    return JSME_ALIAS_DEFINITIONS.SME;
  }
  if (compactLabel === 'Mes') {
    return undefined;
  }
  return JSME_ALIAS_DEFINITIONS[compactLabel.toUpperCase()];
}

function createAliasAtom(symbol: string, x: number, y: number, z = 0): AliasTemplateAtom {
  return { symbol, x, y, z };
}

function createLinearAliasDefinition(
  anchor: string,
  terminal: string[],
  scale = 0.85,
  bondOrder = 1,
): AliasExpansionDefinition {
  const atoms = [createAliasAtom(anchor, 0, 0)];
  terminal.forEach((symbol, index) => {
    atoms.push(createAliasAtom(symbol, index + 1, 0));
  });
  const bonds = terminal.map((_symbol, index) => ({
    from: index + 1,
    to: index + 2,
    order: bondOrder,
  }));
  return {
    anchor,
    atoms,
    bonds,
    referenceAtomIndex: terminal.length > 0 ? 2 : undefined,
    scale,
  };
}

function createTrigonalAliasDefinition(
  anchor: string,
  branches: Array<{ symbol: string; x: number; y: number; order?: number }>,
  scale = 0.85,
): AliasExpansionDefinition {
  return {
    anchor,
    atoms: [createAliasAtom(anchor, 0, 0), ...branches.map((branch) => createAliasAtom(branch.symbol, branch.x, branch.y))],
    bonds: branches.map((branch, index) => ({
      from: 1,
      to: index + 2,
      order: branch.order ?? 1,
    })),
    referenceAtomIndex: branches.length > 0 ? 2 : undefined,
    scale,
  };
}

function createChainAliasDefinition(anchor: string, length: number, scale = 0.85): AliasExpansionDefinition {
  const atoms = [createAliasAtom(anchor, 0, 0)];
  for (let index = 0; index < length; index += 1) {
    atoms.push(createAliasAtom('C', index + 1, 0));
  }
  const bonds = Array.from({ length }, (_value, index) => ({
    from: index + 1,
    to: index + 2,
    order: 1,
  }));
  return {
    anchor,
    atoms,
    bonds,
    referenceAtomIndex: length > 0 ? 2 : undefined,
    scale,
  };
}

function createExplicitAliasDefinition(
  anchor: string,
  atoms: AliasTemplateAtom[],
  bonds: AliasTemplateBond[],
  scale = 0.85,
  referenceAtomIndex = 2,
): AliasExpansionDefinition {
  return {
    anchor,
    atoms: [createAliasAtom(anchor, 0, 0), ...atoms],
    bonds,
    referenceAtomIndex: atoms.length > 0 ? referenceAtomIndex : undefined,
    scale,
  };
}

function createSelfAnchoredAliasDefinition(
  atoms: AliasTemplateAtom[],
  bonds: AliasTemplateBond[],
  scale = 0.85,
  referenceAtomIndex = 2,
): AliasExpansionDefinition {
  return {
    anchor: atoms[0]?.symbol ?? 'C',
    atoms,
    bonds,
    referenceAtomIndex: atoms.length > 1 ? referenceAtomIndex : undefined,
    scale,
  };
}

function shiftAliasAtoms(atoms: AliasTemplateAtom[], dx: number, dy: number): AliasTemplateAtom[] {
  return atoms.map((atom) => ({ ...atom, x: atom.x + dx, y: atom.y + dy }));
}

function offsetAliasBonds(bonds: AliasTemplateBond[], offset: number): AliasTemplateBond[] {
  return bonds.map((bond) => ({
    from: bond.from + offset,
    to: bond.to + offset,
    order: bond.order,
  }));
}

function createRingFragment(symbols: string[], coordinates: Array<[number, number]>, bondOrder = 4): {
  atoms: AliasTemplateAtom[];
  bonds: AliasTemplateBond[];
} {
  const atoms = symbols.map((symbol, index) => createAliasAtom(symbol, coordinates[index][0], coordinates[index][1]));
  const bonds = symbols.map((_symbol, index) => ({
    from: index + 1,
    to: index === symbols.length - 1 ? 1 : index + 2,
    order: bondOrder,
  }));
  return { atoms, bonds };
}

const PHENYL_FRAGMENT = createRingFragment(
  ['C', 'C', 'C', 'C', 'C', 'C'],
  [
    [0, 0],
    [1, 0],
    [1.5, SQRT3_OVER_2],
    [1, 2 * SQRT3_OVER_2],
    [0, 2 * SQRT3_OVER_2],
    [-0.5, SQRT3_OVER_2],
  ],
);

const PYRIDYL_FRAGMENT = createRingFragment(
  ['C', 'N', 'C', 'C', 'C', 'C'],
  [
    [0, 0],
    [1, 0],
    [1.5, SQRT3_OVER_2],
    [1, 2 * SQRT3_OVER_2],
    [0, 2 * SQRT3_OVER_2],
    [-0.5, SQRT3_OVER_2],
  ],
);

const FURYL_FRAGMENT = createRingFragment(
  ['C', 'O', 'C', 'C', 'C'],
  [
    [0, 0],
    [1, 0],
    [1.45, 0.95],
    [0.5, 1.6],
    [-0.35, 0.8],
  ],
);

const THIENYL_FRAGMENT = createRingFragment(
  ['C', 'S', 'C', 'C', 'C'],
  [
    [0, 0],
    [1, 0],
    [1.45, 0.95],
    [0.5, 1.6],
    [-0.35, 0.8],
  ],
);

const BENZYL_FRAGMENT = createSelfAnchoredAliasDefinition(
  [
    createAliasAtom('C', 0, 0),
    ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 1, 0),
  ],
  [
    { from: 1, to: 2, order: 1 },
    ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 1),
  ],
);

const BENZOYL_FRAGMENT = createSelfAnchoredAliasDefinition(
  [
    createAliasAtom('C', 0, 0),
    createAliasAtom('O', 1, 0),
    ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 2, 0),
  ],
  [
    { from: 1, to: 2, order: 2 },
    ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 2),
    { from: 1, to: 3, order: 1 },
  ],
);

const CBZ_FRAGMENT = createSelfAnchoredAliasDefinition(
  [
    createAliasAtom('C', 0, 0),
    createAliasAtom('O', 1, 0),
    createAliasAtom('O', 2, 0),
    createAliasAtom('C', 3, 0),
    ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 4, 0),
  ],
  [
    { from: 1, to: 2, order: 2 },
    { from: 2, to: 3, order: 1 },
    { from: 3, to: 4, order: 1 },
    ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 4),
  ],
);

const TOSYL_FRAGMENT = createSelfAnchoredAliasDefinition(
  [
    createAliasAtom('S', 0, 0),
    createAliasAtom('O', 1, 0),
    createAliasAtom('O', 1, 1),
    createAliasAtom('C', -1, 0),
    ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 4, 0),
  ],
  [
    { from: 1, to: 2, order: 2 },
    { from: 1, to: 3, order: 2 },
    { from: 1, to: 4, order: 1 },
    ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 4),
    { from: 4, to: 5, order: 1 },
  ],
);

const PHENYL_ALIAS = createSelfAnchoredAliasDefinition(PHENYL_FRAGMENT.atoms, PHENYL_FRAGMENT.bonds);
const PYRIDYL_ALIAS = createSelfAnchoredAliasDefinition(PYRIDYL_FRAGMENT.atoms, PYRIDYL_FRAGMENT.bonds);
const FURYL_ALIAS = createSelfAnchoredAliasDefinition(FURYL_FRAGMENT.atoms, FURYL_FRAGMENT.bonds);
const THIENYL_ALIAS = createSelfAnchoredAliasDefinition(THIENYL_FRAGMENT.atoms, THIENYL_FRAGMENT.bonds);

const O_PHENYL_ALIAS = createExplicitAliasDefinition(
  'O',
  shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 1, 0),
  [{ from: 1, to: 2, order: 1 }, ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 1)],
);

const S_PHENYL_ALIAS = createExplicitAliasDefinition(
  'S',
  shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 1, 0),
  [{ from: 1, to: 2, order: 1 }, ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 1)],
);

const JSME_ALIAS_DEFINITIONS: Record<string, AliasExpansionDefinition> = {
  CF3: createTrigonalAliasDefinition('C', [
    { symbol: 'F', x: 1, y: 0 },
    { symbol: 'F', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'F', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  F3C: createTrigonalAliasDefinition('C', [
    { symbol: 'F', x: 1, y: 0 },
    { symbol: 'F', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'F', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  OCF3: createExplicitAliasDefinition(
    'O',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('F', 2, 0),
      createAliasAtom('F', 0.5, SQRT3_OVER_2),
      createAliasAtom('F', 0.5, -SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
  F3CO: createExplicitAliasDefinition(
    'O',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('F', 2, 0),
      createAliasAtom('F', 0.5, SQRT3_OVER_2),
      createAliasAtom('F', 0.5, -SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
  NCF3: createExplicitAliasDefinition(
    'N',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('F', 2, 0),
      createAliasAtom('F', 0.5, SQRT3_OVER_2),
      createAliasAtom('F', 0.5, -SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
  F3CN: createExplicitAliasDefinition(
    'N',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('F', 2, 0),
      createAliasAtom('F', 0.5, SQRT3_OVER_2),
      createAliasAtom('F', 0.5, -SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
  CCL3: createTrigonalAliasDefinition('C', [
    { symbol: 'Cl', x: 1, y: 0 },
    { symbol: 'Cl', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'Cl', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  CF2H: createTrigonalAliasDefinition('C', [
    { symbol: 'F', x: 1, y: 0 },
    { symbol: 'F', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  HF2C: createTrigonalAliasDefinition('C', [
    { symbol: 'F', x: 1, y: 0 },
    { symbol: 'F', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  F2C: createTrigonalAliasDefinition('C', [
    { symbol: 'F', x: 1, y: 0 },
    { symbol: 'F', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  CN: createLinearAliasDefinition('C', ['N'], 0.85, 3),
  NC: createLinearAliasDefinition('C', ['N'], 0.85, 3),
  NO2: createTrigonalAliasDefinition('N', [
    { symbol: 'O', x: 1, y: 0, order: 2 },
    { symbol: 'O', x: -0.5, y: SQRT3_OVER_2, order: 1 },
  ]),
  O2N: createTrigonalAliasDefinition('N', [
    { symbol: 'O', x: 1, y: 0, order: 2 },
    { symbol: 'O', x: -0.5, y: SQRT3_OVER_2, order: 1 },
  ]),
  CHO: createLinearAliasDefinition('C', ['O'], 0.85, 2),
  OHC: createLinearAliasDefinition('C', ['O'], 0.85, 2),
  AC: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('C', 0.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
    ],
  ),
  OAC: createExplicitAliasDefinition(
    'O',
    [createAliasAtom('C', 1, 0), createAliasAtom('O', 2, 0), createAliasAtom('C', 1.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 1 },
    ],
  ),
  NHAC: createExplicitAliasDefinition(
    'N',
    [createAliasAtom('C', 1, 0), createAliasAtom('O', 2, 0), createAliasAtom('C', 1.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 1 },
    ],
  ),
  CO2H: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 0.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
    ],
  ),
  HO2C: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 0.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
    ],
  ),
  COOH: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 0.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
    ],
  ),
  CO2ME: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 0.5, SQRT3_OVER_2), createAliasAtom('C', 1.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
    ],
  ),
  MEO2C: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 0.5, SQRT3_OVER_2), createAliasAtom('C', 1.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
    ],
  ),
  CO2ET: createExplicitAliasDefinition(
    'C',
    [
      createAliasAtom('O', 1, 0),
      createAliasAtom('O', 0.5, SQRT3_OVER_2),
      createAliasAtom('C', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
    ],
  ),
  COOET: createExplicitAliasDefinition(
    'C',
    [
      createAliasAtom('O', 1, 0),
      createAliasAtom('O', 0.5, SQRT3_OVER_2),
      createAliasAtom('C', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
    ],
  ),
  ETO2C: createExplicitAliasDefinition(
    'C',
    [
      createAliasAtom('O', 1, 0),
      createAliasAtom('O', 0.5, SQRT3_OVER_2),
      createAliasAtom('C', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
    ],
  ),
  OME: createLinearAliasDefinition('O', ['C']),
  MEO: createLinearAliasDefinition('O', ['C']),
  OCH3: createLinearAliasDefinition('O', ['C']),
  H3CO: createLinearAliasDefinition('O', ['C']),
  CH3O: createLinearAliasDefinition('O', ['C']),
  SME: createLinearAliasDefinition('S', ['C']),
  NME: createLinearAliasDefinition('N', ['C']),
  MEN: createLinearAliasDefinition('N', ['C']),
  NME2: createTrigonalAliasDefinition('N', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  ME2N: createTrigonalAliasDefinition('N', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  ME: createChainAliasDefinition('C', 0),
  ET: createChainAliasDefinition('C', 1),
  C2H5: createChainAliasDefinition('C', 1),
  PR: createChainAliasDefinition('C', 2),
  NPR: createChainAliasDefinition('C', 2),
  NC3H7: createChainAliasDefinition('C', 2),
  C3H7: createChainAliasDefinition('C', 2),
  BU: createChainAliasDefinition('C', 3),
  NBU: createChainAliasDefinition('C', 3),
  NC4H9: createChainAliasDefinition('C', 3),
  C4H9: createChainAliasDefinition('C', 3),
  C5H11: createChainAliasDefinition('C', 4),
  NC5H11: createChainAliasDefinition('C', 4),
  PENT: createChainAliasDefinition('C', 4),
  NPENT: createChainAliasDefinition('C', 4),
  OET: createChainAliasDefinition('O', 2),
  ETO: createChainAliasDefinition('O', 2),
  IPR: createTrigonalAliasDefinition('C', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  IPRO: createTrigonalAliasDefinition('O', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  OIPR: createTrigonalAliasDefinition('O', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  IBU: createExplicitAliasDefinition(
    'C',
    [createAliasAtom('C', 1, 0), createAliasAtom('C', 2, 0), createAliasAtom('C', 1.5, SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
    ],
  ),
  TBU: createTrigonalAliasDefinition('C', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  OTBUT: createTrigonalAliasDefinition('O', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  OTB: createTrigonalAliasDefinition('O', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  OTBU: createTrigonalAliasDefinition('O', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  BOC: createExplicitAliasDefinition(
    'C',
    [
      createAliasAtom('O', 1, 0),
      createAliasAtom('O', 0.5, SQRT3_OVER_2),
      createAliasAtom('C', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
      createAliasAtom('C', 1.5, 2 * SQRT3_OVER_2),
      createAliasAtom('C', 1.5, 0),
    ],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 1 },
      { from: 3, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
      { from: 4, to: 6, order: 1 },
      { from: 4, to: 7, order: 1 },
    ],
  ),
  NHBOC: createExplicitAliasDefinition(
    'N',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('O', 2, 0),
      createAliasAtom('O', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
      createAliasAtom('C', 3.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, 2 * SQRT3_OVER_2),
      createAliasAtom('C', 2.5, 0),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
      { from: 5, to: 6, order: 1 },
      { from: 5, to: 7, order: 1 },
      { from: 5, to: 8, order: 1 },
    ],
  ),
  NBOC: createExplicitAliasDefinition(
    'N',
    [
      createAliasAtom('C', 1, 0),
      createAliasAtom('O', 2, 0),
      createAliasAtom('O', 1.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, SQRT3_OVER_2),
      createAliasAtom('C', 3.5, SQRT3_OVER_2),
      createAliasAtom('C', 2.5, 2 * SQRT3_OVER_2),
      createAliasAtom('C', 2.5, 0),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
      { from: 5, to: 6, order: 1 },
      { from: 5, to: 7, order: 1 },
      { from: 5, to: 8, order: 1 },
    ],
  ),
  PH: PHENYL_ALIAS,
  PHENYL: PHENYL_ALIAS,
  ARYL: PHENYL_ALIAS,
  AR: PHENYL_ALIAS,
  OPH: O_PHENYL_ALIAS,
  SPH: S_PHENYL_ALIAS,
  BN: BENZYL_FRAGMENT,
  BZ: BENZOYL_FRAGMENT,
  OBN: createExplicitAliasDefinition(
    'O',
    [createAliasAtom('C', 1, 0), ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 2, 0)],
    [{ from: 1, to: 2, order: 1 }, { from: 2, to: 3, order: 1 }, ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 2)],
  ),
  OBZ: createExplicitAliasDefinition(
    'O',
    [createAliasAtom('C', 1, 0), createAliasAtom('O', 2, 0), ...shiftAliasAtoms(PHENYL_FRAGMENT.atoms, 2, 0)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 1 },
      ...offsetAliasBonds(PHENYL_FRAGMENT.bonds, 3),
    ],
  ),
  CBZ: CBZ_FRAGMENT,
  TS: TOSYL_FRAGMENT,
  TOS: TOSYL_FRAGMENT,
  SO2PH: TOSYL_FRAGMENT,
  TF: createExplicitAliasDefinition(
    'S',
    [
      createAliasAtom('O', 1, 0),
      createAliasAtom('O', 1, 1),
      createAliasAtom('C', -1, 0),
      createAliasAtom('F', -2, 0),
      createAliasAtom('F', -0.5, SQRT3_OVER_2),
      createAliasAtom('F', -0.5, -SQRT3_OVER_2),
    ],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 2 },
      { from: 1, to: 4, order: 1 },
      { from: 4, to: 5, order: 1 },
      { from: 4, to: 6, order: 1 },
      { from: 4, to: 7, order: 1 },
    ],
  ),
  OTF: createExplicitAliasDefinition(
    'O',
    [
      createAliasAtom('S', 1, 0),
      createAliasAtom('O', 2, 0),
      createAliasAtom('O', 2, 1),
      createAliasAtom('C', 1, -1),
      createAliasAtom('F', 2, -1),
      createAliasAtom('F', 0.5, -1.8),
      createAliasAtom('F', 0.5, -0.2),
    ],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 2 },
      { from: 2, to: 5, order: 1 },
      { from: 5, to: 6, order: 1 },
      { from: 5, to: 7, order: 1 },
      { from: 5, to: 8, order: 1 },
    ],
  ),
  SO2ME: createExplicitAliasDefinition(
    'S',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 1, 1), createAliasAtom('C', -1, 0)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 2 },
      { from: 1, to: 4, order: 1 },
    ],
  ),
  OMS: createExplicitAliasDefinition(
    'O',
    [createAliasAtom('S', 1, 0), createAliasAtom('O', 2, 0), createAliasAtom('O', 2, 1), createAliasAtom('C', 1, -1)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 2 },
      { from: 2, to: 4, order: 2 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
  PY: PYRIDYL_ALIAS,
  PYRIDYL: PYRIDYL_ALIAS,
  '2PYRIDYL': PYRIDYL_ALIAS,
  '3PYRIDYL': PYRIDYL_ALIAS,
  FURYL: FURYL_ALIAS,
  '2FURYL': FURYL_ALIAS,
  THIENYL: THIENYL_ALIAS,
  '2THIENYL': THIENYL_ALIAS,
  C6H5: PHENYL_ALIAS,
  C6H4: PHENYL_ALIAS,
  PTOL: createSelfAnchoredAliasDefinition(
    [...PHENYL_FRAGMENT.atoms, createAliasAtom('C', 2, 2.6)],
    [...PHENYL_FRAGMENT.bonds, { from: 4, to: 7, order: 1 }],
  ),
  TOL: createSelfAnchoredAliasDefinition(
    [...PHENYL_FRAGMENT.atoms, createAliasAtom('C', 2, 2.6)],
    [...PHENYL_FRAGMENT.bonds, { from: 4, to: 7, order: 1 }],
  ),
  SO3H: createExplicitAliasDefinition(
    'S',
    [createAliasAtom('O', 1, 0), createAliasAtom('O', 1, 1), createAliasAtom('O', -1, 0)],
    [
      { from: 1, to: 2, order: 2 },
      { from: 1, to: 3, order: 2 },
      { from: 1, to: 4, order: 1 },
    ],
  ),
  'B(OH)2': createTrigonalAliasDefinition('B', [
    { symbol: 'O', x: 1, y: 0 },
    { symbol: 'O', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  BOH2: createTrigonalAliasDefinition('B', [
    { symbol: 'O', x: 1, y: 0 },
    { symbol: 'O', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  B0H2: createTrigonalAliasDefinition('B', [
    { symbol: 'O', x: 1, y: 0 },
    { symbol: 'O', x: -0.5, y: SQRT3_OVER_2 },
  ]),
  ME3SI: createTrigonalAliasDefinition('Si', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  TMS: createTrigonalAliasDefinition('Si', [
    { symbol: 'C', x: 1, y: 0 },
    { symbol: 'C', x: -0.5, y: SQRT3_OVER_2 },
    { symbol: 'C', x: -0.5, y: -SQRT3_OVER_2 },
  ]),
  OTMS: createExplicitAliasDefinition(
    'O',
    [createAliasAtom('Si', 1, 0), createAliasAtom('C', 2, 0), createAliasAtom('C', 0.5, SQRT3_OVER_2), createAliasAtom('C', 0.5, -SQRT3_OVER_2)],
    [
      { from: 1, to: 2, order: 1 },
      { from: 2, to: 3, order: 1 },
      { from: 2, to: 4, order: 1 },
      { from: 2, to: 5, order: 1 },
    ],
  ),
};

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

function expandKnownAliasesForJSME(molblock: string): string {
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

  const expandableAliases = aliases.filter((alias) => {
    const atom = parsedAtoms[alias.atomIndex - 1];
    const definition = resolveAliasDefinition(alias.label);
    return Boolean(definition && atom && ['R', '*'].includes(atom.symbol.toUpperCase()));
  });
  if (!expandableAliases.length) return molblock;

  const aliasLinesToRemove = new Set<number>();
  const nextAtoms = [...parsedAtoms];
  const nextBondLines = lines.slice(bondStart, propertyStart);
  const averageBondLength = getAverageBondLength(parsedAtoms, parsedBonds);

  expandableAliases.forEach((alias) => {
    const definition = resolveAliasDefinition(alias.label);
    if (!definition) return;
    const atom = nextAtoms[alias.atomIndex - 1];
    const center = parsedAtoms[alias.atomIndex - 1];
    const neighborBond = parsedBonds.find((bond) => bond.a1 === alias.atomIndex || bond.a2 === alias.atomIndex);
    const neighborIndex = neighborBond ? (neighborBond.a1 === alias.atomIndex ? neighborBond.a2 : neighborBond.a1) : null;
    const neighbor = neighborIndex ? parsedAtoms[neighborIndex - 1] : null;
    let dx = 1;
    let dy = 0;
    if (center && neighbor) {
      dx = center.x - neighbor.x;
      dy = center.y - neighbor.y;
    }
    const length = Math.hypot(dx, dy) || 1;
    const unitX = dx / length;
    const unitY = dy / length;
    const referenceIndex = definition.referenceAtomIndex ?? (definition.atoms.length > 1 ? 2 : 1);
    const referenceAtom = definition.atoms[referenceIndex - 1] ?? definition.atoms[0];
    const referenceVectorX = referenceAtom.x - definition.atoms[0].x;
    const referenceVectorY = referenceAtom.y - definition.atoms[0].y;
    const referenceLength = Math.hypot(referenceVectorX, referenceVectorY) || 1;
    const templateAngle = Math.atan2(referenceVectorY, referenceVectorX);
    const targetAngle = Math.atan2(unitY, unitX);
    const rotation = targetAngle - templateAngle;
    const bondScale = averageBondLength * (definition.scale ?? 0.85) / referenceLength;
    const templateToGlobalIndex = new Map<number, number>();

    nextAtoms[alias.atomIndex - 1] = {
      ...atom,
      symbol: definition.anchor,
      line: setV2000AtomSymbol(atom.line, definition.anchor),
    };

    templateToGlobalIndex.set(1, alias.atomIndex);

    definition.atoms.slice(1).forEach((templateAtom, templateOffset) => {
      const relativeX = templateAtom.x - definition.atoms[0].x;
      const relativeY = templateAtom.y - definition.atoms[0].y;
      const rotated = rotateVector(relativeX, relativeY, rotation);
      const transformedAtom: V2000Atom = {
        x: center.x + rotated.x * bondScale,
        y: center.y + rotated.y * bondScale,
        z: center.z,
        symbol: templateAtom.symbol,
        line: '',
      };
      const nextAtomIndex = nextAtoms.length + 1;
      nextAtoms.push({ ...transformedAtom, line: formatV2000AtomLine(transformedAtom) });
      templateToGlobalIndex.set(templateOffset + 2, nextAtomIndex);
    });

    definition.bonds.forEach((bond) => {
      const from = templateToGlobalIndex.get(bond.from);
      const to = templateToGlobalIndex.get(bond.to);
      if (!from || !to) return;
      nextBondLines.push(formatV2000BondLine(from, to, bond.order ?? 1));
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
  return expandKnownAliasesForJSME(molblock);
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

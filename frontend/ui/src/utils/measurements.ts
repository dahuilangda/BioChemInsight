import type { StructureRecord } from '../types';

function parseObject(value: unknown): Record<string, string> {
  if (typeof value !== 'string') return {};
  try { const parsed = JSON.parse(value); return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}; }
  catch { return {}; }
}

export function measurementValues(record: StructureRecord): Record<string, string> {
  let observations: unknown = [];
  try { observations = JSON.parse(String(record.BIOACTIVITY_OBSERVATIONS || '[]')); } catch { /* Legacy rows retain their original text. */ }
  const values: Record<string, string> = {};
  const grouped: Record<string, Array<{ value: string; page?: number; table_index?: number }>> = {};
  if (Array.isArray(observations)) for (const item of observations) {
    if (item && item.association_status !== 'needs_review' && typeof item.column === 'string' && typeof item.value === 'string' && item.subject_id === String(record.COMPOUND_ID || '')) (grouped[item.column] ??= []).push(item);
  }
  for (const [column, items] of Object.entries(grouped)) {
    const distinct = new Set(items.map((item) => item.value));
    values[column] = distinct.size === 1 ? items[0].value : items.map((item) => `${item.value} [${item.page ? `p. ${item.page}` : 'source'}${item.table_index !== undefined ? `, table ${item.table_index + 1}` : ''}]`).join('; ');
  }
  return { ...values, ...parseObject(record.USER_MEASUREMENT_EDITS) };
}

export function measurementEdits(record: StructureRecord): Record<string, string> {
  return parseObject(record.USER_MEASUREMENT_EDITS);
}

/** Only exact points or point ± uncertainty support a plain numeric comparison.
 * Censored values and categorical bins remain text instead of invented points. */
export function comparableMeasurement(value: string): number | null {
  const match = value.trim().replace(/,/g, '').match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:\s*±\s*\d+(?:\.\d*)?)?$/i);
  if (!match) return null;
  const number = Number(match[1]); return Number.isFinite(number) ? number : null;
}

import { comparableMeasurement } from './measurements';
const naturalOrder = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
export interface FilterRule {
  column: string;
  operator: string;
  value: string;
}
export interface OrderRule {
  column: string;
  descending: boolean;
}
export interface ReviewQuery {
  filters: FilterRule[];
  match: 'all' | 'any';
  sorts: OrderRule[];
  groups: OrderRule[];
}
export const emptyQuery: ReviewQuery = { filters: [], match: 'all', sorts: [], groups: [] };
export function compareValues(a: unknown, b: unknown, descending = false): number {
  const left = a == null ? '' : String(a),
    right = b == null ? '' : String(b);
  if (!left || !right) return left ? -1 : right ? 1 : 0;
  const x = comparableMeasurement(left),
    y = comparableMeasurement(right);
  if ((x === null) !== (y === null)) return x === null ? 1 : -1;
  const result = x !== null && y !== null ? x - y : naturalOrder.compare(left, right);
  return descending ? -result : result;
}
export function matchesFilter(value: unknown, rule: FilterRule): boolean {
  const text = value == null ? '' : String(value).trim().toLowerCase(),
    query = rule.value.trim().toLowerCase();
  const x = comparableMeasurement(text),
    y = comparableMeasurement(query);
  switch (rule.operator) {
    case 'empty':
      return !text;
    case 'filled':
      return Boolean(text);
    case 'contains':
      return text.includes(query);
    case 'not_contains':
      return !text.includes(query);
    case 'equals':
      return text === query;
    case 'not_equals':
      return text !== query;
    case 'gt':
      return x !== null && y !== null && x > y;
    case 'gte':
      return x !== null && y !== null && x >= y;
    case 'lt':
      return x !== null && y !== null && x < y;
    case 'lte':
      return x !== null && y !== null && x <= y;
    default:
      return true;
  }
}
export function activeFilter(rule: FilterRule): boolean {
  return (
    Boolean(rule.column) &&
    (['filled', 'empty'].includes(rule.operator) || Boolean(rule.value.trim()))
  );
}
export function applyReviewQuery<T>(
  rows: T[],
  query: ReviewQuery,
  value: (row: T, column: string) => unknown,
  manual: (a: T, b: T) => number,
): T[] {
  const filters = query.filters.filter(activeFilter);
  return rows
    .filter(
      (row) =>
        !filters.length ||
        (query.match === 'any'
          ? filters.some((rule) => matchesFilter(value(row, rule.column), rule))
          : filters.every((rule) => matchesFilter(value(row, rule.column), rule))),
    )
    .sort((a, b) => {
      for (const rule of query.groups) {
        const av = value(a, rule.column) ?? '',
          bv = value(b, rule.column) ?? '';
        let result = compareValues(av, bv, rule.descending);
        if (!result && String(av) !== String(bv))
          result = (String(av) < String(bv) ? -1 : 1) * (rule.descending ? -1 : 1);
        if (result) return result;
      }
      for (const rule of query.sorts) {
        const result = compareValues(value(a, rule.column), value(b, rule.column), rule.descending);
        if (result) return result;
      }
      return manual(a, b);
    });
}
export function groupPath<T>(
  row: T,
  groups: OrderRule[],
  value: (row: T, column: string) => unknown,
): string[] {
  const path: unknown[] = [];
  return groups.map((rule) => {
    path.push([rule.column, String(value(row, rule.column) ?? '')]);
    return JSON.stringify(path);
  });
}

export function readQuery(value: unknown): ReviewQuery {
  if (!value || typeof value !== 'object') return emptyQuery;
  const query = value as Record<string, unknown>;
  const object = (rule: unknown): rule is Record<string, unknown> =>
    Boolean(rule) && typeof rule === 'object';
  const orders = (rules: unknown): OrderRule[] =>
    Array.isArray(rules)
      ? rules
          .filter(object)
          .filter((rule) => typeof rule.column === 'string' && typeof rule.descending === 'boolean')
          .map((rule) => ({ column: String(rule.column), descending: Boolean(rule.descending) }))
      : [];
  return {
    filters: Array.isArray(query.filters)
      ? query.filters
          .filter(object)
          .filter(
            (rule) =>
              typeof rule.column === 'string' &&
              typeof rule.operator === 'string' &&
              typeof rule.value === 'string',
          )
          .map((rule) => ({
            column: String(rule.column),
            operator: String(rule.operator),
            value: String(rule.value),
          }))
      : [],
    match: query.match === 'any' ? 'any' : 'all',
    sorts: orders(query.sorts),
    groups: orders(query.groups).slice(0, 3),
  };
}

import React from 'react';
import { emptyQuery, readQuery, type ReviewQuery } from './reviewQuery';
export const reviewPageSizes = [25, 50, 100, 200];
interface Layout {
  pageSize: number;
  query: ReviewQuery;
  scope: string;
  columns: string[];
  rows: string[];
  hidden: string[];
  widths: Record<string, number>;
}
function read(scope: string): Layout {
  try {
    const value = JSON.parse(localStorage.getItem(`bci:grid-layout:${scope}`) || '{}');
    return {
      scope,
      pageSize: reviewPageSizes.includes(value.pageSize) ? value.pageSize : 25,
      query: readQuery(value.query),
      hidden: Array.isArray(value.hidden)
        ? value.hidden.filter((v: unknown) => typeof v === 'string')
        : [],
      columns: Array.isArray(value.columns)
        ? value.columns.filter((v: unknown) => typeof v === 'string')
        : [],
      rows: Array.isArray(value.rows)
        ? value.rows.filter((v: unknown) => typeof v === 'string')
        : [],
      widths: Object.fromEntries(
        Object.entries(value.widths || {}).filter(
          ([, width]) => typeof width === 'number' && width >= 64 && width <= 720,
        ),
      ) as Record<string, number>,
    };
  } catch {
    return {
      scope,
      pageSize: 25,
      query: emptyQuery,
      columns: [],
      rows: [],
      hidden: [],
      widths: {},
    };
  }
}
export function moveBefore(order: string[], from: string, to: string, after = false): string[] {
  if (from === to || !order.includes(from) || !order.includes(to)) return order;
  const next = order.filter((key) => key !== from);
  next.splice(next.indexOf(to) + (after ? 1 : 0), 0, from);
  return next;
}
export function useReviewLayout(scope: string) {
  const [stored, setStored] = React.useState(() => read(scope));
  React.useEffect(() => setStored(read(scope)), [scope]);
  React.useEffect(() => {
    if (stored.scope === scope) {
      try {
        localStorage.setItem(`bci:grid-layout:${scope}`, JSON.stringify(stored));
      } catch {
        /* Storage limits must not block editing. */
      }
    }
  }, [scope, stored]);
  const layout = stored.scope === scope ? stored : read(scope);
  const update = (change: Partial<Layout>) =>
    setStored((previous) => ({
      ...(previous.scope === scope ? previous : read(scope)),
      ...change,
      scope,
    }));
  return {
    ...layout,
    update,
    reset: () =>
      update({ pageSize: 25, query: emptyQuery, columns: [], rows: [], hidden: [], widths: {} }),
  };
}

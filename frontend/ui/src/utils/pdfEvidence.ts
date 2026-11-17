export function evidencePages(records: ReadonlyArray<Record<string, unknown>>): number[] {
  const pages = new Set<number>();
  const add = (value: unknown) => {
    if (Array.isArray(value)) { value.forEach(add); return; }
    if (typeof value === 'number' || typeof value === 'string') {
      const text = String(value).trim();
      if (text.startsWith('[')) {
        try { const parsed = JSON.parse(text); if (Array.isArray(parsed)) { parsed.forEach(add); return; } } catch { /* Plain page list. */ }
      }
      for (const token of text.split(/[,;\s]+/)) {
        const range = /^(\d+)[-–](\d+)$/.exec(token);
        if (range) {
          const first = Number(range[1]), last = Number(range[2]);
          if (first > 0 && last >= first && last - first <= 2000) for (let page = first; page <= last; page++) pages.add(page);
        } else {
          const page = Number(token);
          if (Number.isInteger(page) && page > 0) pages.add(page);
        }
      }
    }
  };
  for (const record of records) {
    add(record.PAGE_NUM ?? record.page_num ?? record.page);
    add(record.source_pages); add(record.MARKUSH_SCAFFOLD_PAGE);
    if (record.BIOACTIVITY_OBSERVATIONS) {
      try {
        const observations = typeof record.BIOACTIVITY_OBSERVATIONS === 'string' ? JSON.parse(record.BIOACTIVITY_OBSERVATIONS) : record.BIOACTIVITY_OBSERVATIONS;
        if (Array.isArray(observations)) for (const observation of observations) {
          if (!observation || typeof observation !== 'object') continue;
          add(observation.page ?? observation.page_num ?? observation.PAGE_NUM); add(observation.source_pages);
        }
      } catch { /* Optional metadata. */ }
    }
  }
  return [...pages];
}

export const normalizeMolblock = (value?: string): string => {
  if (!value) return '';
  let normalized = value.trim();
  if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith("'") && normalized.endsWith("'"))
  ) {
    normalized = normalized.slice(1, -1).trim();
  }
  normalized = normalized
    .replace(/\\r\\n/g, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\n')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\t/g, ' ');
  normalized = normalized.replace(/[^\x20-\x7E\n]/g, '');

  if (normalized.includes('$MOL')) {
    normalized = normalized.split('$MOL', 2)[1]?.replace(/^\n+/, '') ?? normalized;
  }
  if (normalized.includes('$$$$')) {
    normalized = normalized.split('$$$$', 2)[0]?.trimEnd() ?? normalized;
  }
  if (normalized.includes('M  END')) {
    normalized = normalized.split('M  END', 2)[0] + 'M  END';
  } else if (normalized.includes('M  V30 END CTAB')) {
    normalized = `${normalized.trimEnd()}\nM  END`;
  }

  const lines = normalized.split('\n');
  let versionIdx = lines.findIndex((line) => line.includes('V2000') || line.includes('V3000'));
  if (versionIdx === -1) {
    const countsPattern = /^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+/;
    for (let idx = 0; idx < lines.length; idx += 1) {
      const line = lines[idx];
      if (countsPattern.test(line) && !line.includes('.')) {
        if (lines.some((entry) => entry.includes('M  V30'))) {
          lines[idx] = `${line.trimEnd()} V3000`;
        } else {
          lines[idx] = `${line.trimEnd()} V2000`;
        }
        versionIdx = idx;
        break;
      }
    }
  }

  if (versionIdx !== -1) {
    const headerStart = Math.max(0, versionIdx - 3);
    const header = lines.slice(headerStart, versionIdx);
    while (header.length < 3) {
      header.unshift('');
    }
    normalized = [...header, ...lines.slice(versionIdx)].join('\n');
  } else if (lines.some((line) => line.includes('M  V30'))) {
    const header = ['', '', ''];
    const countsLine = '  0  0  0  0  0  0            999 V3000';
    normalized = [...header, countsLine, ...lines].join('\n');
  }

  return normalized;
};

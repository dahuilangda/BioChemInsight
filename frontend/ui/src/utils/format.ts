export function formatCellValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(', ');
  return value == null ? '' : String(value);
}

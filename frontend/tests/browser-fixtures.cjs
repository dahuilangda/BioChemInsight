// Small valid PDF fixtures use only standard PDF objects; no model or SDK.
function makePdf(pageCount = 8, paddingPerPage = 0) {
  const objects = [];
  const fontId = 3 + pageCount * 2;
  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  objects.push(`<< /Type /Pages /Count ${pageCount} /Kids [${Array.from({ length: pageCount }, (_, index) => `${3 + index} 0 R`).join(' ')}] >>`);
  for (let index = 0; index < pageCount; index++) {
    const contentId = 3 + pageCount + index;
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 ${fontId} 0 R >> >> /Contents ${contentId} 0 R >>`);
  }
  for (let index = 0; index < pageCount; index++) {
    const content = `BT /F1 20 Tf 40 745 Td (Patent source - page ${index + 1}) Tj ET\n0.4 w 100 80 180 120 re S\n100 420 180 120 re S\n${paddingPerPage ? '%' + 'x'.repeat(paddingPerPage) + '\n' : ''}`;
    objects.push(`<< /Length ${Buffer.byteLength(content)} >>\nstream\n${content}endstream`);
  }
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  let pdf = '%PDF-1.4\n'; const offsets = [0];
  objects.forEach((object, index) => { offsets.push(Buffer.byteLength(pdf)); pdf += `${index + 1} 0 obj\n${object}\nendobj\n`; });
  const xref = Buffer.byteLength(pdf);
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach((offset) => { pdf += `${String(offset).padStart(10, '0')} 00000 n \n`; });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(pdf);
}
const thumbnail = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l5sAAAAASUVORK5CYII=';
module.exports = { makePdf, thumbnail };

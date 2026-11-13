// Real HTTP PDF ranges, actual PDF.js worker, no inference dependency.
const { chromium } = require('playwright');
const http = require('http');
const assert = require('assert/strict');
const { makePdf } = require('./browser-fixtures.cjs');
(async () => {
  const pdf = makePdf(64, 100000);
  let transferred = 0, fullRequests = 0; const ranges = [];
  const server = http.createServer((req, res) => {
    if (!req.url.startsWith('/api/pdfs/qa/file')) {
      const proxy = http.request({ hostname: '127.0.0.1', port: 5173, path: req.url, method: req.method, headers: req.headers }, upstream => { res.writeHead(upstream.statusCode, upstream.headers); upstream.pipe(res); });
      req.pipe(proxy); return;
    }
    const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
    const start = range ? Number(range[1]) : 0, end = range ? Math.min(Number(range[2] || pdf.length - 1), pdf.length - 1) : pdf.length - 1;
    if (range) ranges.push([start, end]); else fullRequests++;
    res.writeHead(range ? 206 : 200, {
      'Content-Type': 'application/pdf', 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1,
      'Access-Control-Allow-Origin': '*', 'Access-Control-Expose-Headers': 'Accept-Ranges,Content-Length,Content-Range',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${pdf.length}` } : {}),
    });
    res.flushHeaders(); let offset = start;
    const send = () => {
      if (res.destroyed) return;
      if (offset > end) return res.end();
      const chunk = pdf.subarray(offset, Math.min(offset + 16384, end + 1));
      offset += chunk.length; transferred += chunk.length; res.write(chunk);
      timer = setTimeout(send, 2);
    };
    let timer = setTimeout(send, 2); res.on('close', () => clearTimeout(timer));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('console', m => { if (m.type() === 'error') console.error(m.text()); });

  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.continue();
    let json = {};
    if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: '64-page source.pdf', total_pages: 64 };
    else if (path === '/api/tasks') json = { tasks: [], running_count: 0, pending_count: 0, total_pages: 1 };
    else if (path === '/api/tasks/qa') json = { task_id: 'qa', type: 'structure_extraction', status: 'completed', params: {} };
    else if (path.endsWith('/structures')) json = { records: [{ COMPOUND_ID: '1', PAGE_NUM: 1 }], filtered_records: [] };
    return route.fulfill({ json });
  });
  const ready = number => page.waitForFunction(n => document.querySelector('.pdf-pane')?.getAttribute('data-page') === String(n) && document.querySelector('.pdf-pane canvas')?.width > 0 && !document.querySelector('.pdf-pane__painting'), number);
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}/?pdf=qa&structureTask=qa&step=4`); await ready(1);
    const firstPageBytes = transferred;
    const input = page.getByRole('textbox', { name: 'PDF page', exact: true });
    await input.fill('32'); await input.press('Enter'); await ready(32);
    assert.ok(ranges.length >= 2, 'PDF.js must request byte ranges');
    assert.ok(firstPageBytes < pdf.length / 4, 'opening a page must not fetch the whole document');
    assert.ok(transferred < pdf.length / 2, 'jumping to a distant page must remain demand-driven');
    assert.equal(fullRequests, 1);
    console.log(JSON.stringify({ documentBytes: pdf.length, firstPageBytes, afterPage32Bytes: transferred, rangeRequests: ranges.length, fullRequests }));
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exit(1); });

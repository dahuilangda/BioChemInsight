const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let nativeCalls = 0, ocrCalls = 0, delayNative = false, failNative = false;
    const task = { task_id: 'search', type: 'structure_extraction', status: 'completed', pdf_id: 'qa', params: {} };
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url()), path = url.pathname;
      if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(3) });
      if (path.endsWith('/search/pages')) {
        nativeCalls++;
        const numbers = url.searchParams.get('numbers').split(',').map(Number);
        if (delayNative) await new Promise(resolve => setTimeout(resolve, 500));
        const failed = failNative ? numbers.filter(number => number === 3) : [];
        return route.fulfill({ json: { pages: numbers.filter(number => !failed.includes(number)).map(number => ({ page: number, source: 'text', needs_ocr: number === 2,
          items: number === 2 ? [] : [{ text: 'Native alpha beta', rect: { x: .1, y: .2, w: .5, h: .05 } }] })), failed } });
      }
      if (path.includes('/search/pages/')) {
        const number = Number(path.split('/').pop());
        const ocr = url.searchParams.get('ocr') === 'true';
        if (ocr) { ocrCalls++; await new Promise(resolve => setTimeout(resolve, 120)); } else nativeCalls++;
        return route.fulfill({ json: { page: number, source: ocr ? 'ocr' : 'text', needs_ocr: number === 2 && !ocr,
          items: number === 2 && !ocr ? [] : [{ text: number === 2 ? 'Scanned alpha alpha' : 'Native alpha beta', rect: { x: .1, y: .2, w: .5, h: .05 } }] } });
      }
      let json = {};
      if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Search.pdf', total_pages: 3 };
      else if (path.endsWith('/structures')) json = { task, records: [{ RECORD_ID: 'one', COMPOUND_ID: '1', SMILES: 'CC', PAGE_NUM: 1 }], filtered_records: [] };
      else if (path === '/api/tasks/search') json = task;
      else if (path === '/api/tasks') json = { tasks: [task], total_pages: 1 };
      else if (path.includes('/pages/') || path.includes('/preview')) json = { image: thumbnail };
      else if (path.endsWith('/render-batch')) json = { results: [] };
      return route.fulfill({ json });
    });
    await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:3000'}/?pdf=qa&structureTask=search&step=4`);
    await page.getByRole('button', { name: 'Search PDF', exact: true }).click();
    const input = page.getByRole('searchbox', { name: 'Search PDF text' });
    await input.fill('alpha');
    await page.waitForFunction(() => document.querySelector('.pdf-search__count')?.textContent === '1 / 4');
    assert.equal(nativeCalls, 2); assert.equal(ocrCalls, 1);
    await input.press('Enter');
    await page.waitForFunction(() => document.querySelector('.pdf-pane')?.dataset.page === '2');
    await page.waitForFunction(() => document.querySelector('.pdf-search-hit.is-active'));
    await input.press('Enter');
    assert.equal(await page.locator('.pdf-search__count').textContent(), '3 / 4');
    await input.fill('scanned');
    await page.waitForFunction(() => document.querySelector('.pdf-search__count')?.textContent === '1 / 1');
    assert.equal(ocrCalls, 1);
    await input.press('Escape');
    assert.equal(await page.locator('.pdf-search-hit').count(), 0);
    await page.getByRole('button', { name: 'Search PDF', exact: true }).press('Control+f');
    await input.fill('missing');
    await page.waitForFunction(() => document.querySelector('.pdf-search__count')?.textContent === 'No matches');
    assert.equal(nativeCalls, 2); assert.equal(ocrCalls, 1);
    await page.setViewportSize({ width: 390, height: 844 });
    const bounds = await page.locator('.pdf-search').boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 391);
    await page.getByRole('button', { name: 'Close PDF search' }).press('Enter');
    assert.equal(await input.count(), 0);
    await page.reload();
    delayNative = true;
    const callsBefore = nativeCalls, ocrBefore = ocrCalls;
    await page.getByRole('button', { name: 'Search PDF', exact: true }).click();
    await input.fill('alpha');
    await page.waitForFunction(() => document.querySelector('.pdf-search__status'));
    await page.waitForTimeout(300);
    await page.getByRole('button', { name: 'Close PDF search' }).click();
    await page.waitForTimeout(650);
    assert.equal(nativeCalls, callsBefore + 1);
    assert.equal(ocrCalls, ocrBefore);
    delayNative = false; failNative = true;
    await page.getByRole('button', { name: 'Search PDF', exact: true }).click();
    await input.fill('missing');
    await page.waitForFunction(() => document.querySelector('.pdf-search__count')?.textContent === 'Search incomplete');
    const retryCalls = nativeCalls;
    failNative = false;
    await page.locator('.pdf-search').getByRole('button', { name: 'Retry', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.pdf-search__count')?.textContent === 'No matches');
    assert.equal(nativeCalls, retryCalls + 1);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ crossPage: true, scannedPage: true, cached: true, keyboard: true, mobile: true, cancellation: true, partialFailure: true, retry: true }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

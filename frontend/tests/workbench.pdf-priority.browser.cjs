const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  let release;
  const readerFirst = new Promise(resolve => release = resolve);
  // Hold the lazy thumbnail module so the reader has a painted page first.
  await page.route('**/src/components/PdfThumbnail.tsx*', async route => { await readerFirst; await route.continue(); });
  let downloads = 0, images = 0, sharedModuleUrl;
  page.on('request', request => { if (new URL(request.url()).pathname === '/src/utils/pdfDocument.ts') sharedModuleUrl = request.url(); });
  await page.route('**/api/**', async route => {
    const u = new URL(route.request().url()), p = u.pathname;
    if (!p.startsWith('/api/')) return route.continue();
    if (p.endsWith('/file')) { downloads++; return route.fulfill({ contentType: 'application/pdf', body: makePdf(60) }); }
    let json = {};
    if (p === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Reader priority.pdf', total_pages: 60 };
    else if (p === '/api/tasks') json = { tasks: [], total_pages: 1, revision: 'priority' };
    else if (p === '/api/tasks/qa') json = { task_id: 'qa', type: 'structure_extraction', status: 'completed', params: {} };
    else if (p.endsWith('/structures')) json = { records: Array.from({length:4},(_,i)=>({COMPOUND_ID:String(i+1),PAGE_NUM:1,BOX_COORDS_FILE:`/qa/${i}.json`,SEGMENT_FILE:'/qa/image.png'})), filtered_records: [] };
    else if (p === '/api/artifacts' && u.searchParams.get('path').endsWith('.json')) json = {mime_type:'application/json',content:Buffer.from(JSON.stringify({box:[100,260,280,380],page_width:600,page_height:800})).toString('base64')};
    else if (p.includes('/pages/') || p.includes('/preview') || p === '/api/artifacts') {images++;json={mime_type:'image/png',content:thumbnail,image:thumbnail};}
    await route.fulfill({ json });
  });
  const ready = n => page.waitForFunction(n => document.querySelector(`.pdf-pane [data-pdf-page="${n}"] canvas`)?.width > 0 && document.querySelector('.pdf-pane')?.dataset.page === String(n), n);
  try {
    await page.goto(`${process.env.WORKBENCH_MODULE_URL || 'http://127.0.0.1:5174'}/?pdf=qa&structureTask=qa&step=4`);
    await ready(1);
    await page.evaluate(async moduleUrl => {
      const shared = await import(moduleUrl);
      const doc = await shared.loadDocument('qa'), source = await doc.getPage(1);
      const proto = Object.getPrototypeOf(source), render = proto.render;
      window.__pdfCalls = [];
      proto.render = function (...args) { window.__pdfCalls.push(this.pageNumber); return render.apply(this, args); };
    }, sharedModuleUrl);
    release();
    const canceled = await page.evaluate(async moduleUrl => {
      const shared = await import(moduleUrl);
      return Promise.all(Array.from({length:20}, (_,i) => {
        const controller = new AbortController();
        const pending = shared.getPreviewPageRaster('qa', 30+i, controller.signal);
        controller.abort();
        return pending.then(()=>false, error=>error.name==='AbortError');
      }));
    }, sharedModuleUrl);
    assert.ok(canceled.every(Boolean), 'abandoned background previews must cancel immediately');
    await page.locator('.review-card .pdf-thumbnail--ready').first().waitFor();
    const calls = await page.evaluate(() => window.__pdfCalls);
    assert.equal(calls.filter(n => n === 1).length, 0, 'cards must copy the painted PDF page without another PDF.js render');
    const firstCanvas = await page.locator('.pdf-pane [data-pdf-page="1"] canvas').evaluate(c => c.width);
    await page.setViewportSize({width:375,height:812});
    await ready(1);
    await page.locator('.workbench-document__toggle').click();
    await page.waitForTimeout(220);
    assert.equal(await page.locator('.pdf-pane').getAttribute('data-page'), '1', 'collapsing must preserve the page');
    await page.locator('.workbench-document__toggle').click(); await ready(1);
    await page.setViewportSize({width:1280,height:800}); await ready(1);
    assert.equal(await page.locator('.pdf-pane [data-pdf-page="1"] canvas').evaluate(c=>c.width), firstCanvas, 'reopening at the same width reuses its raster');
    const input = page.getByRole('textbox',{name:'PDF page',exact:true});
    for (const n of [14,28,50]) { await input.fill(String(n)); await input.press('Enter'); }
    await ready(50);
    await page.getByRole('button',{name:'Zoom in',exact:true}).click();
    await ready(50);
    await page.waitForTimeout(240);
    assert.equal(await page.locator('.pdf-pane').getAttribute('data-page'), '50', 'zoom preserves the reading location');
    assert.equal(downloads,1); assert.equal(images,0);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({paintedPageCopiesWithoutPdfRender:true,canceledBackgroundPreviews:canceled.length,pdfDownloads:downloads,sourceImageDownloads:images,collapseRetainsPage:true,rapidJumpFinalPage:50}));
  } finally { release(); await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});

// PDF-backed previews, restrained motion and phone navigation use real PDF.js.
const { chromium } = require('playwright');
const assert = require('assert/strict');
const fs = require('fs');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const reads = { pdf: 0, bitmap: 0, crop: 0, boxes: 0 };
  const rows = Array.from({ length: 12 }, (_, i) => ({ COMPOUND_ID: `BCI-${i+1}`, PAGE_NUM: i + 1, SMILES: 'CCO', BOX_COORDS_FILE: `/qa/box-${i}.json`, SEGMENT_FILE: '/qa/crop.png', BIOACTIVITY_OBSERVATIONS: JSON.stringify([{ subject_id: `BCI-${i+1}`, column: 'IC50 (nM)', value: '24' }]) }));
  await page.route('**/api/**', async route => {
    const u = new URL(route.request().url()), p = u.pathname;
    if (!p.startsWith('/api/')) return route.continue();
    if (p.endsWith('/file')) { reads.pdf++; return route.fulfill({ contentType: 'application/pdf', body: makePdf(120) }); }
    let json = {};
    if (p === '/api/pdfs/qa') { await new Promise(resolve=>setTimeout(resolve,180)); json = { pdf_id: 'qa', filename: 'Evidence.pdf', total_pages: 120 }; }
    else if (p.includes('/pages/') || p.includes('/preview')) { reads.bitmap++; json = { image: thumbnail }; }
    else if (p === '/api/tasks') json = { tasks: [], total_pages: 1, revision: 'experience' };
    else if (p === '/api/tasks/qa') json = { task_id: 'qa', type: 'structure_extraction', status: 'completed', progress: 1, params: {}, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
    else if (p.endsWith('/structures')) json = { records: rows, filtered_records: [] };
    else if (p === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map(item => ({ ...item, image: `data:image/png;base64,${thumbnail}` })) };
    else if (p === '/api/artifacts') {
      if (u.searchParams.get('path').endsWith('.json')) {
        reads.boxes++;
        json = { mime_type: 'application/json', content: Buffer.from(JSON.stringify({ box: [100, 260, 280, 380], page_width: 600, page_height: 800 })).toString('base64') };
      } else { reads.crop++; json = { mime_type: 'image/png', content: thumbnail }; }
    }
    await route.fulfill({ json });
  });
  fs.mkdirSync('/tmp/biochem-experience', { recursive: true });
  try {
    await page.goto('http://127.0.0.1:5173/?pdf=qa&structureTask=qa&step=4');
    await page.locator('.review-card .pdf-thumbnail--ready').first().waitFor({timeout:10000}).catch(async e => { await page.screenshot({path:'/tmp/biochem-experience/failure.png'}); console.log(await page.locator('body').innerText(), errors, reads, await page.locator('.pdf-thumbnail').evaluateAll(es=>es.slice(0,3).map(e=>({html:e.outerHTML,b:e.getBoundingClientRect().toJSON()})))); throw e; });
    const pixels = await page.locator('.review-card .pdf-thumbnail--ready canvas').first().evaluate(c => {
      const bytes = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let ink = 0;
      for (let i = 0; i < bytes.length; i += 4) if (bytes[i] < 240 && bytes[i+1] < 240 && bytes[i+2] < 240) ink++;
      return { width: c.width, height: c.height, ink };
    });
    assert.ok(Math.abs(pixels.width / pixels.height - 1.5) < .02); assert.ok(pixels.width >= 150); assert.ok(pixels.ink > 50, 'source crop must contain actual PDF marks');
    await page.screenshot({ path: '/tmp/biochem-experience/desktop.png' });
    for (const width of [320, 375, 390, 600, 768, 900]) {
      await page.setViewportSize({ width, height: 850 });
      await page.waitForTimeout(250);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${width}: document overflow`);
      for (const button of await page.locator('.step-nav__button').all()) {
        const fits = await button.evaluate(b => {
          const label = b.querySelector('.step-nav__label'), br = b.getBoundingClientRect();
          const range = document.createRange(); range.selectNodeContents(label);
          return [...range.getClientRects()].every(r => r.left >= br.left - 1 && r.right <= br.right + 1);
        });
        assert.ok(fits, `${width}: navigation label escapes button`);
      }
      if (width <= 768) {
        await page.locator('.workbench-document__toggle').click();
        assert.equal(await page.locator('.workbench-document__body').isVisible(), false);
        const bar = await page.locator('.workbench-document').boundingBox();
        assert.ok(bar.height <= 50, 'collapsed document leaves room for data');
        await page.screenshot({ path: `/tmp/biochem-experience/mobile-${width}-collapsed.png` });
        await page.locator('.workbench-document__toggle').click();
        assert.equal(await page.locator('.workbench-document__body').isVisible(), true);
      }
    }
    await page.setViewportSize({ width: 375, height: 812 });
    await page.locator('.step-nav').getByRole('button', { name: 'Structures', exact: true }).click();
    await page.locator('.page-card .pdf-thumbnail--ready').first().waitFor({timeout:15000}).catch(async e=>{await page.screenshot({path:'/tmp/biochem-experience/gallery-failure.png'});console.log(await page.locator('body').innerText(),errors);throw e;});
    assert.equal(await page.locator('.page-card[data-page="120"] canvas').getAttribute('width'), '0', 'distant page stays unrendered');
    assert.ok(await page.locator('.page-card .pdf-thumbnail--ready').count() < 20, 'gallery only paints nearby pages');
    await page.screenshot({ path: '/tmp/biochem-experience/gallery-mobile.png' });
    const gallery = page.locator('.selector-gallery');
    await gallery.evaluate(e => e.scrollTop = e.scrollHeight);
    await page.locator('.page-card[data-page="120"] .pdf-thumbnail--ready').waitFor();
    const buffers = await page.locator('.page-card canvas').evaluateAll(cs => cs.filter(c => c.width > 0).length);
    assert.ok(buffers < 20, 'offscreen display buffers are released');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    const motion = await page.locator('.workbench-content > *').first().evaluate(e => getComputedStyle(e).animationName);
    assert.equal(motion, 'none');
    assert.equal(reads.pdf, 1, 'all source views share a PDF document');
    assert.equal(reads.bitmap, 0, 'no redundant page bitmaps');
    assert.equal(reads.crop, 0, 'valid metadata avoids crop image downloads');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ reads, cropPixels: pixels, widths: [320,375,390,600,768,900,1440], liveGalleryBuffers: buffers, reducedMotion: true }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

// Space, selection and tray checks across the whole product; isolated HTTP data.
const { chromium } = require('playwright');
const assert = require('assert/strict');
const fs = require('fs');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage(); page.setDefaultTimeout(12000);
  const errors = []; let submissions = 0;
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname;
    if (url.searchParams.get('max_width') === '1024') await new Promise(resolve => setTimeout(resolve, 250));
    if (!path.startsWith('/api/')) return route.continue();
    let json = {};
    if (route.request().method() === 'POST') submissions++;
    if (path === '/api/tasks') json = { tasks: [], running_count: 0, pending_count: 0, total_count: 0, total_pages: 1, revision: 'stages' };
    else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'WO2024163691A1.pdf', total_pages: 48 };
    else if (path.includes('/pages/')) json = { page: Number(path.split('/').pop()), image: thumbnail };
    else if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(48) });
    return route.fulfill({ json });
  });
  try {
    for (const [width, height] of [[320,640],[375,812],[768,900],[1280,800],[1920,1080],[3840,1600],[812,375]]) {
      await page.setViewportSize({ width, height });
      for (const step of [1,2,3]) {
        await page.goto(`http://127.0.0.1:5173/?pdf=qa&step=${step}`);
        await page.getByRole('button', { name: 'Jobs', exact: true }).waitFor();
        if (step === 1) await page.locator('.workbench-document .pdf-pane canvas').first().waitFor();
        else {
          await page.locator('.page-card').first().waitFor();
          const gallery = await page.locator('.selector-gallery').boundingBox();
          const data = await page.locator('.stage-data').boundingBox();
          assert.ok(gallery.height >= data.height - 64, `gallery must fill results pane ${width}x${height} step ${step}`);
          assert.ok(gallery.height > 40, 'gallery must remain usable');
          if (width === 375 && step === 2) {
            await page.getByRole('button', { name: 'Enlarge page 2', exact: true }).click();
            await page.waitForFunction(() => document.querySelector('.workbench-document .pdf-pane')?.getAttribute('data-page') === '2');
            assert.equal(await page.getByRole('dialog', { name: 'Image preview', exact: true }).count(), 0, 'page cards must navigate the live PDF');
          }
          const card = page.locator('.page-card').first();
          await card.click();
          await page.locator('.stage-workspace__selection').filter({ hasText: '1 selected' }).waitFor();
          await page.getByRole('button', { name: 'Page settings', exact: true }).click();
          const auto = page.locator(step === 2 ? '#auto-structure-pages' : '#auto-assay-pages');
          assert.equal(await auto.isChecked(), false, 'manual selection must disable Auto');
          await auto.check();
          await page.locator('.stage-workspace__selection').filter({ hasText: '1 selected' }).waitFor();
          await page.locator('.selector-toolbar').getByRole('button', { name: 'Clear', exact: true }).click();
          await page.locator('.stage-workspace__selection').filter({ hasText: 'Auto' }).waitFor();
          await card.focus(); await page.keyboard.press('Space');
          await page.locator('.stage-workspace__selection').filter({ hasText: '1 selected' }).waitFor();
        }
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width, `overflow ${width} step ${step}`);
        const jobs = page.getByRole('button', { name: 'Jobs', exact: true });
        await jobs.click();
        await page.getByRole('region', { name: 'Background jobs' }).waitFor();
        assert.equal(await page.locator('.jobs-search input').evaluate(node => node === document.activeElement), true);
        const panel = await page.locator('.jobs-panel').boundingBox();
        assert.ok(panel.x >= 0 && panel.x + panel.width <= width + 1);
        await page.keyboard.press('Escape');
        await page.locator('.jobs-panel').waitFor({ state: 'hidden' });
        assert.equal(await jobs.evaluate(node => node === document.activeElement), true);
        if (process.env.QA_OUTPUT_DIR) {
          fs.mkdirSync(process.env.QA_OUTPUT_DIR, { recursive: true });
          await page.screenshot({ path: `${process.env.QA_OUTPUT_DIR}/stage-${step}-${width}x${height}.png` });
        }
      }
    }
    assert.equal(submissions, 0, 'selection and Auto must not launch background inference');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ stages: [1,2,3], resolutions: 7, selectionKeyboardAndAuto: true, jobsFocusAndEscape: true, implicitSubmissions: submissions, errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

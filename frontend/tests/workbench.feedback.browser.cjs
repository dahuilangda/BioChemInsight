const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');

async function checkFeedback(browser, table) {
  const page = await browser.newPage({ viewport: { width: 375, height: 812 } });
  let releaseSave;
  const save = new Promise(resolve => { releaseSave = resolve; });
  let records = [{ RECORD_ID: 'row-0', COMPOUND_ID: '6a', SMILES: 'CCO', PAGE_NUM: 1 }];
  const task = { task_id: 'feedback', type: 'structure_extraction', status: 'completed', pdf_id: 'qa', params: {} };
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(1) });
    let json = {};
    if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Feedback.pdf', total_pages: 1 };
    else if (path === '/api/tasks/feedback') json = task;
    else if (path === '/api/tasks') json = { tasks: [task], total_pages: 1, total_count: 1 };
    else if (path.endsWith('/structures')) {
      if (route.request().method() === 'PUT') {
        await save;
        records = route.request().postDataJSON().records;
      }
      json = { task, records, filtered_records: [] };
    } else if (path.endsWith('/render-batch')) json = { results: route.request().postDataJSON().items.map(item => ({ ...item, image: `data:image/png;base64,${thumbnail}` })) };
    return route.fulfill({ json });
  });
  try {
    await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=feedback&step=4`);
    if (table) await page.getByRole('button', { name: 'Table', exact: true }).click();
    const input = page.locator(`${table ? 'tbody' : '.review-card'} .page-cell__id-input`).first();
    await input.waitFor();
    const toggle = page.locator('.workbench-document__toggle');
    if (await toggle.getAttribute('aria-expanded') === 'true') await toggle.click();
    await input.fill('changed');
    assert.equal(await toggle.getAttribute('aria-expanded'), 'false');
    await input.press('Enter');
    const saving = page.getByRole('status', { name: 'Saving…', exact: true });
    await saving.waitFor();
    assert.equal(await saving.getAttribute('aria-busy'), 'true');
    assert.equal(await saving.locator('.review-save-spinner').count(), 1);
    assert.equal(await saving.locator('svg').count(), 0);
    releaseSave();
    const saved = page.getByRole('status', { name: 'Saved', exact: true });
    await saved.waitFor();
    assert.equal(await saved.locator('.review-save-spinner').count(), 0);
    assert.equal(await saved.locator('svg').count(), 1);
    assert.equal(records[0].COMPOUND_ID, 'changed');
    assert.equal(await toggle.getAttribute('aria-expanded'), 'false');
    if (!table) {
      await page.getByRole('button', { name: 'Open source page 1', exact: true }).click();
      assert.equal(await toggle.getAttribute('aria-expanded'), 'true');
    }
    const add = page.locator('[data-review-command="add"]');
    for (const dismiss of ['escape', 'close']) {
      await add.click();
      await page.getByRole('button', { name: 'Add compound', exact: true }).click();
      const dialog = page.getByRole('dialog', { name: 'Add compound', exact: true });
      await dialog.waitFor();
      if (dismiss === 'escape') await page.keyboard.press('Escape');
      else await page.getByRole('button', { name: 'Close add dialog', exact: true }).click();
      await dialog.waitFor({ state: 'detached' });
      assert.equal(await add.evaluate(element => document.activeElement === element), true);
    }
  } finally {
    releaseSave();
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch();
  try {
    for (const table of [false, true]) await checkFeedback(browser, table);
    console.log(JSON.stringify({ views: ['cards', 'table'], savingIndicator: true, dialogFocusReturn: true, collapsedSourcePreserved: true, explicitSourceNavigation: true }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });

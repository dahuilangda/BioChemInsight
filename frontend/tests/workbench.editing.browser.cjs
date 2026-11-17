const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf } = require('./browser-fixtures.cjs');

const baseUrl = process.env.WORKBENCH_URL || 'http://127.0.0.1:5173';
const image = 'data:image/svg+xml;base64,' + Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="70"><path d="M15 45 45 15 75 45" stroke="black" fill="none"/></svg>').toString('base64');

async function checkEditing(browser, width) {
  const page = await browser.newPage({ viewport: { width, height: 812 } });
  const errors = [];
  let releaseSave;
  let heldSave = null;
  let records = [1, 2].map((page, index) => ({ RECORD_ID: `row-${index}`, COMPOUND_ID: `6${index ? 'b' : 'a'}`, PAGE_NUM: page, SMILES: 'CCO' }));
  const task = { task_id: 'editing', type: 'structure_extraction', status: 'completed', pdf_id: 'qa', progress: 1, params: {} };
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(2) });
    let json = {};
    if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Editing.pdf', total_pages: 2 };
    else if (path === '/api/tasks/editing') json = task;
    else if (path === '/api/tasks') json = { tasks: [task], total_count: 1, total_pages: 1 };
    else if (path.endsWith('/structures')) {
      if (route.request().method() === 'PUT') {
        if (heldSave) await heldSave;
        records = route.request().postDataJSON().records;
      }
      json = { task, records, filtered_records: [] };
    } else if (path.endsWith('/render-batch')) json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
    return route.fulfill({ json });
  });
  try {
    await page.goto(`${baseUrl}/?pdf=qa&structureTask=editing&step=4`);
    const toggle = page.locator('.workbench-document__toggle');
    const collapseSource = async () => {
      const viewport = page.viewportSize();
      if (!(await toggle.isVisible())) await page.setViewportSize({ width: 375, height: viewport.height });
      if (await toggle.getAttribute('aria-expanded') === 'true') await toggle.click();
      if (page.viewportSize().width !== viewport.width) await page.setViewportSize(viewport);
    };
    for (const view of ['Table', 'Cards']) {
      await page.getByRole('button', { name: view, exact: true }).click();
      const rows = page.locator(view === 'Table' ? 'tbody tr[data-row-index]' : '.review-card');
      await rows.first().waitFor();
      await collapseSource();
      await rows.nth(1).locator('.page-cell__id-input').focus();
      assert.equal(await toggle.getAttribute('aria-expanded'), 'false', `${view}: editing expanded the source`);
      if (view === 'Cards') await rows.first().locator('.review-card__eyebrow').click();
      else await rows.first().locator('td[data-column="id"]').click();
      assert.equal(await toggle.getAttribute('aria-expanded'), 'false', `${view}: selection expanded the source`);
      const pageLink = view === 'Cards'
        ? rows.first().getByRole('button', { name: 'Open source page 1', exact: true })
        : rows.first().getByRole('button', { name: 'Go to PDF page 1', exact: true });
      await pageLink.click();
      assert.equal(await toggle.getAttribute('aria-expanded'), 'true', `${view}: page link did not open the source`);
      await collapseSource();
      const source = view === 'Cards'
        ? rows.first().getByRole('button', { name: 'Locate source for compound 6a', exact: true })
        : rows.first().locator('.structure-image-btn').first();
      await source.click();
      assert.equal(await toggle.getAttribute('aria-expanded'), 'true', `${view}: source image did not open the source`);
    }
    for (const close of ['Escape', 'Cancel', 'Close add dialog', 'backdrop']) {
      const add = page.locator('[data-review-command="add"]');
      await add.click();
      await page.getByRole('button', { name: 'Add compound', exact: true }).click();
      const dialog = page.getByRole('dialog', { name: 'Add compound', exact: true });
      await dialog.waitFor();
      if (close === 'Escape') await page.keyboard.press('Escape');
      else if (close === 'backdrop') await page.locator('.review-dialog-layer').click({ position: { x: 4, y: 4 } });
      else await dialog.getByRole('button', { name: close, exact: true }).click();
      await dialog.waitFor({ state: 'detached' });
      assert.equal(await add.evaluate((element) => document.activeElement === element), true, `${close}: focus did not return to Add`);
    }
    const add = page.locator('[data-review-command="add"]');
    await add.click();
    await page.getByRole('button', { name: 'Add measurement', exact: true }).click();
    const measurement = page.getByRole('dialog', { name: 'Add measurement', exact: true });
    await measurement.getByRole('textbox', { name: 'Endpoint', exact: true }).fill('Test endpoint');
    await measurement.getByRole('button', { name: 'Add measurement', exact: true }).click();
    await measurement.waitFor({ state: 'detached' });
    assert.equal(await add.evaluate((element) => document.activeElement === element), true, 'measurement creation lost focus');
    await page.getByRole('status', { name: 'Saved', exact: true }).waitFor();
    heldSave = new Promise((resolve) => { releaseSave = resolve; });
    const input = page.locator('.review-card .page-cell__id-input').first();
    await input.fill('corrected');
    await input.press('Enter');
    const saving = page.getByRole('status', { name: 'Saving…', exact: true });
    await saving.waitFor();
    assert.equal(await saving.locator('.review-save-spinner').count(), 1);
    assert.equal(await saving.locator('svg').count(), 0);
    releaseSave();
    heldSave = null;
    const saved = page.getByRole('status', { name: 'Saved', exact: true });
    await saved.waitFor();
    assert.equal(await saved.locator('.review-save-spinner').count(), 0);
    assert.equal(await saved.locator('svg').count(), 1);
    assert.equal(records[0].COMPOUND_ID, 'corrected');
    assert.deepEqual(errors, []);
  } finally {
    releaseSave?.();
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch();
  try {
    for (const width of [375, 1440]) await checkEditing(browser, width);
    console.log(JSON.stringify({ collapsedSourceRespected: true, explicitSourceNavigation: true, dialogFocusRestored: true, saveStateIcons: true, widths: [375, 1440] }));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exit(1); });

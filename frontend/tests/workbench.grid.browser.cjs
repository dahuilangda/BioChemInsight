// Real browser drag/drop, resize, view persistence and editing ownership.
const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
const sourcePdf = makePdf(3);
const figure = 'data:image/svg+xml;base64,' + Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="100"><path d="M20 70L60 30L100 70" stroke="#334155" fill="none"/></svg>').toString('base64');
let revision = 0;
let records = ['A', 'B', 'C'].map((id, index) => ({ RECORD_ID: `record-${id}`, COMPOUND_ID: id, PAGE_NUM: index + 1, SMILES: 'CCO', SEGMENT_FILE: '/fixture.png', BIOACTIVITY_OBSERVATIONS: JSON.stringify([{ subject_id: id, column: 'EC50 (nM)', value: String(index + 1), source_kind: 'experimental' }]) }));
(async () => {
 const browser = await chromium.launch();
 const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
 const page = await context.newPage(); page.setDefaultTimeout(12000);
 const errors = []; page.on('pageerror', (error) => errors.push(error.message));
 await page.route('**/api/**', async (route) => {
  const url = new URL(route.request().url()), path = url.pathname; if (!path.startsWith('/api/')) return route.continue();
  if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: sourcePdf });
  let json = {};
  const task = { task_id: 'grid', type: 'structure_extraction', status: 'completed', pdf_id: 'grid-pdf', params: { records_revision: String(revision) } };
  if (path === '/api/tasks/grid') json = task;
  else if (path.endsWith('/structures')) {
   if (route.request().method() === 'PUT') { const body = route.request().postDataJSON(); assert.equal(body.base_revision, String(revision)); records = body.records; revision++; }
   json = { task: { ...task, params: { records_revision: String(revision) } }, records, filtered_records: [] };
  } else if (path === '/api/pdfs/grid-pdf') json = { pdf_id: 'grid-pdf', total_pages: 3, filename: 'Grid.pdf' };
  else if (/\/pages\/\d+$/.test(path)) json = { page: Number(path.split('/').pop()), image: thumbnail, width:600, height:800 };
  else if (path.includes('/previews/')) return route.fulfill({contentType:'image/png',body:Buffer.from(thumbnail,'base64')});
  else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image: figure })) };
  else if (path === '/api/artifacts') json = { mime_type: 'image/svg+xml', content: figure.split(',')[1] };
  else if (path.endsWith('/sar')) json = { coverage: {open_gaps:0,complete:true}, gaps:[], next_actions:[] };
  else if (path === '/api/tasks') json = { tasks: [task], total_count:1, total_pages:1, revision:'grid' };
  await route.fulfill({json});
 });
 const open = async () => { await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5174'}/?pdf=grid-pdf&structureTask=grid&step=4`); await page.getByRole('button',{name:'Table',exact:true}).click(); await page.locator('tbody tr[data-row-index]').first().waitFor(); };
 const columnOrder = () => page.locator('thead th[data-column]').evaluateAll((cells) => cells.map((cell) => cell.dataset.column));
 const rowOrder = () => page.locator('tbody tr .page-cell__id-input').evaluateAll((inputs) => inputs.map((input) => input.value));
 try {
  await open();
  await page.locator('th[data-column="structure"]').dragTo(page.locator('th[data-column="source"]'));
  const columns = await columnOrder(); assert.ok(columns.indexOf('structure') < columns.indexOf('source'));
  const bodyColumns = await page.locator('tbody tr').first().locator('td').evaluateAll((cells) => cells.map((cell) => cell.dataset.column));
  assert.deepEqual(bodyColumns, columns, 'header and cell order must stay aligned, including hidden fields');
  await page.getByRole('button', {name:'Move compound C',exact:true}).dragTo(page.getByRole('button', {name:'Move compound A',exact:true}));
  await page.waitForFunction(() => document.querySelector('tbody tr .page-cell__id-input')?.value === 'C');
  assert.deepEqual(await rowOrder(), ['C','A','B']);
  const cancelGrip = await page.getByRole('button',{name:'Move compound C',exact:true}).boundingBox();
  await page.mouse.move(cancelGrip.x + 8, cancelGrip.y + 10); await page.mouse.down(); await page.mouse.move(cancelGrip.x + 8, 12, {steps:5}); await page.mouse.up();
  assert.deepEqual(await rowOrder(), ['C','A','B'], 'dropping outside the grid must leave order unchanged');
  // Dropping in the lower half must also support moving to the very end.
  const sourceGrip = await page.getByRole('button',{name:'Move compound C',exact:true}).boundingBox();
  const lastRow = await page.locator('tbody tr[data-row-index="1"]').boundingBox();
  await page.mouse.move(sourceGrip.x + 8, sourceGrip.y + 12); await page.mouse.down();
  await page.mouse.move(sourceGrip.x + 8, lastRow.y + lastRow.height - 12, {steps:8}); await page.mouse.up();
  await page.waitForFunction(() => document.querySelector('tbody tr:last-child .page-cell__id-input')?.value === 'C');
  assert.deepEqual(await rowOrder(), ['A','B','C']);
  await page.getByRole('button',{name:'Move compound C',exact:true}).focus(); await page.keyboard.press('Alt+ArrowUp'); await page.keyboard.press('Alt+ArrowUp');
  assert.deepEqual(await rowOrder(), ['C','A','B'], 'keyboard row movement must match pointer ordering');
  await page.locator('th[data-column="structure"]').focus(); await page.keyboard.press('Alt+ArrowRight');
  let keyboardColumns = await columnOrder(); assert.ok(keyboardColumns.indexOf('source') < keyboardColumns.indexOf('structure'));
  await page.keyboard.press('Alt+ArrowLeft'); assert.deepEqual(await columnOrder(), columns);
  // A reordered row must still save to its original record, not the visual position.
  const value = page.getByRole('textbox', { name:'EC50 (nM) for compound C',exact:true }); await value.fill('0.125'); await value.press('Enter');
  await page.waitForResponse((response) => response.url().endsWith('/structures') && response.request().method() === 'PUT');
  assert.equal(JSON.parse(records.find((record) => record.RECORD_ID === 'record-C').USER_MEASUREMENT_EDITS)['EC50 (nM)'], '0.125');
  assert.equal(records[0].COMPOUND_ID, 'A', 'view reordering must not rearrange stored records');
  const header = page.locator('th[data-column="source"]'); const before = await header.boundingBox();
  const grip = header.getByRole('separator'); const handle = await grip.boundingBox();
  await page.mouse.move(handle.x + handle.width/2, handle.y + handle.height/2); await page.mouse.down(); await page.mouse.move(handle.x + 70, handle.y + handle.height/2, { steps:6 }); await page.mouse.up();
  const resized = await header.boundingBox(); assert.ok(resized.width > before.width + 40, 'column resizing must change the real table width');
  await open(); assert.deepEqual(await rowOrder(), ['C','A','B']); assert.deepEqual(await columnOrder(), columns);
  assert.ok((await page.locator('th[data-column="source"]').boundingBox()).width > before.width + 40, 'widths survive reload');
  await page.getByRole('button',{name:'Sort',exact:true}).click(); await page.getByRole('combobox',{name:'Sort column',exact:true}).selectOption('COMPOUND_ID');
  assert.equal(await page.getByRole('button',{name:'Move compound C',exact:true}).isDisabled(), true, 'automatic sort must not compete with manual row order');
  await page.getByRole('button',{name:'Clear sorts',exact:true}).click(); await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Columns',exact:true}).click();
  const menu = await page.locator('.columns-picker__panel').boundingBox(), pane = await page.locator('.review-split__list').boundingBox();
  assert.ok(menu.x >= pane.x && menu.x + menu.width <= pane.x + pane.width, 'field menu must fit inside the results pane');
  await page.keyboard.press('Escape'); assert.equal(await page.locator('.columns-picker__panel').count(), 0);
  await page.getByRole('button',{name:'Columns',exact:true}).click();
  await page.locator('.columns-picker__option').filter({hasText:'EC50 (nM)'}).getByRole('checkbox').uncheck();
  await open(); assert.equal(await page.locator('th[data-column="assay:EC50 (nM)"]').count(), 0, 'hidden fields survive reload');
  await page.getByRole('button',{name:'Columns',exact:true}).click();
  await page.locator('.columns-picker__option').filter({hasText:'EC50 (nM)'}).getByRole('checkbox').check();
  await page.keyboard.press('Escape'); assert.deepEqual(await columnOrder(), columns, 'unhiding restores a field to its previous position');
  const heights = await page.locator('.review-toolbar__controls > button,.columns-picker__toggle,.review-manage-menu > summary,.review-toolbar__downloads > details > summary,.review-toolbar__views').evaluateAll((buttons) => buttons.filter((button) => button.getBoundingClientRect().height > 0).map((button) => button.getBoundingClientRect().height));
  assert.ok(heights.every((height) => Math.abs(height - 32) < 1), `command sizes must align: ${heights}`);
  await page.waitForFunction(() => document.querySelector('.pdf-page canvas')?.width > 0);
  await page.screenshot({path:'/tmp/biochem-grid-desktop.png'});
  await page.setViewportSize({width:375,height:812}); await page.getByRole('button',{name:'Columns',exact:true}).click();
  await page.locator('.review-field-order > summary').click();
  await page.getByRole('button',{name:'Move Compound ID left',exact:true}).click();
  const mobileColumns = await columnOrder(); assert.equal(mobileColumns[0], 'id', 'touch-accessible column controls must also reorder core fields');
  const mobileMenu = await page.locator('.columns-picker__panel').boundingBox(); assert.ok(mobileMenu.x >= 0 && mobileMenu.x + mobileMenu.width <= 375);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth),375);
  await page.screenshot({path:'/tmp/biochem-grid-mobile.png'});
  assert.deepEqual(errors, []); console.log(JSON.stringify({ columnDrag:true, rowDrag:true, editOwnership:true, resize:true, reloadPersistence:true, sortedRowGuard:true, uniformToolbar:true, menusFit:true }));
 } catch (error) { console.error(await page.evaluate(() => ({ layout:localStorage.getItem('bci:grid-layout:grid'), body:document.body.innerText.slice(-1000) }))); await page.screenshot({path:'/tmp/bci-grid-failure.png'}); throw error; } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exit(1); });

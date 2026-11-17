// Exercise linked navigation and raster/request reuse with an actual PDF.js
// worker. Domain results are isolated HTTP fixtures, not inference outputs.
const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
const fixturePdf = makePdf(8);
const image = 'data:image/svg+xml;base64,' + Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="180" height="130"><path d="M20 80L60 40L100 80L140 40" fill="none" stroke="black" stroke-width="2"/></svg>').toString('base64');
const records = [
  { COMPOUND_ID: 'A', PAGE_NUM: 1, box: [100, 260, 280, 380] },
  { COMPOUND_ID: 'B', PAGE_NUM: 1, box: [100, 600, 280, 720] },
  { COMPOUND_ID: 'C', PAGE_NUM: 3, box: [100, 260, 280, 380] },
  { COMPOUND_ID: 'D', PAGE_NUM: 5, box: [100, 600, 280, 720] },
  { COMPOUND_ID: 'E', PAGE_NUM: 2, box: [300, 260, 480, 380] },
].map((record) => ({ ...record, SMILES: 'CCO', ANNOTATION_TEXT: 'IC50 = 5.30 ± 0.10 μM', BIOACTIVITY_OBSERVATIONS: JSON.stringify([{ subject_id: record.COMPOUND_ID, column: 'IC50 (µM)', endpoint: 'IC50', unit: 'µM', value: '5.30 ± 0.10' }]), SEGMENT_FILE: '/qa/crop.svg', BOX_COORDS_FILE: `/qa/box-${record.COMPOUND_ID}.json` }));
(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 });
  const page = await context.newPage(); page.setDefaultTimeout(15000);
  const errors = [], boxReads = {}, pages = [];
  let recognizedCompoundId = 'manual';
  let pdfReads = 0, pagePngReads = 0, manualRequest = null, saves = [], revision = 0;
  const snapshots = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url()), path = url.pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) { pdfReads++; return route.fulfill({ contentType: 'application/pdf', body: fixturePdf }); }
    let json = {};
    if (path.endsWith('/regions/measurements')) { json = { records: [{ RECORD_ID: 'result-qa', COMPOUND_ID: 'A', SMILES: '', PAGE_NUM: 3, BIOACTIVITY_OBSERVATIONS: JSON.stringify([{subject_id:'A', column:'Permeability (cm/s)',value:'1.4e-6',page:3,table_index:0}]) }, {COMPOUND_ID:'proposal',SMILES:'',PAGE_NUM:3,COMPOUND_NAME:'printed source name',NAME_STRUCTURE_CANDIDATE:'CCO',RESULT_ONLY:'true'}], gaps: [] }; }
    else if (path.endsWith('/regions/recognize')) { manualRequest = route.request().postDataJSON(); json = { COMPOUND_ID: recognizedCompoundId, SMILES: 'CCO', PAGE_NUM: manualRequest.page, SOURCE: 'manual' }; }
    else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Linked source.pdf', total_pages: 8 };
    else if (path.includes('/pages/')) { pagePngReads++; json = { page: 1, image: thumbnail }; }
    else if (path === '/api/tasks') json = { tasks: [{ task_id: 'qa', type: 'structure_extraction', status: 'completed', progress: 1, result_path: '/qa/results.csv', filename: 'Linked source.pdf', pdf_id: 'qa', updated_at: new Date().toISOString(), created_at: new Date().toISOString() }], running_count: 0, pending_count: 0, total_count: 1, total_pages: 1, revision: 'navigation' };
    else if (path === '/api/tasks/qa') json = { task_id: 'qa', type: 'structure_extraction', status: 'completed', progress: 1, params: {}, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
    else if (path.endsWith('/sar')) json = { coverage: { open_gaps: 1, complete: false }, gaps: [{ gap_id: 'qa-gap', reason: 'Review source association', page: 1 }], next_actions: [] };
    else if (path.endsWith('/structures/undo')) { const previous = snapshots.pop(); records.splice(0, records.length, ...previous); revision++; json = { records, filtered_records: [], task: {task_id:'qa', params: { records_revision: String(revision) } } }; }
    else if (path.endsWith('/structures')) {
      if (route.request().method() === 'PUT') { const data = route.request().postDataJSON(); assert.equal(data.base_revision, String(revision)); snapshots.push(JSON.parse(JSON.stringify(records))); records.splice(0, records.length, ...data.records); saves.push(data); revision++; }
      json = { records, filtered_records: [], task: {task_id:'qa', params: { records_revision: String(revision) } } };
    }
    else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
    else if (path === '/api/artifacts') {
      const source = url.searchParams.get('path');
      const row = records.find((record) => record.BOX_COORDS_FILE === source);
      if (row) {
        boxReads[row.COMPOUND_ID] = (boxReads[row.COMPOUND_ID] || 0) + 1;
        if (row.COMPOUND_ID === 'D') await new Promise((resolve) => setTimeout(resolve, 350));
        json = { mime_type: 'application/json', content: Buffer.from(JSON.stringify({ box: row.box, page_width: 600, page_height: 800 })).toString('base64') };
      } else json = { mime_type: 'image/svg+xml', content: image.split(',')[1] };
    }
    await route.fulfill({ json });
  });
  const canvasReady = async (number) => {
    await page.waitForFunction((wanted) => {
      const pane = document.querySelector('.pdf-pane');
      return pane?.getAttribute('data-page') === String(wanted) && pane.querySelector(`[data-pdf-page="${wanted}"] canvas`)?.width > 0 && !pane.querySelector('.pdf-pane__painting');
    }, number);
  };
  const rasterCount = () => page.locator('.pdf-pane').getAttribute('data-render-count').then(Number);
  const selectCard = async (index) => { await page.locator(`.review-card[data-row-index="${index}"]`).getByRole('img', { name: 'Source structure', exact: true }).click(); };
  try {
    await page.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=qa&step=4`);
    try { await canvasReady(1); } catch (error) { await page.screenshot({ path: '/tmp/bci-nav-failure.png' }); console.error('UI', await page.locator('body').innerText(), errors); throw error; }
    await page.getByRole('button', { name: 'Jobs', exact: true }).click();
    const openJob = page.getByRole('button', { name: 'Open job qa', exact: true });
    await openJob.waitFor();
    const openBounds = await openJob.boundingBox(), trayBounds = await page.locator('.jobs-panel').boundingBox();
    assert.ok(openBounds.x >= trayBounds.x && openBounds.x + openBounds.width <= trayBounds.x + trayBounds.width, 'job actions must be visible without horizontal scrolling');
    await page.getByRole('button', { name: 'Close jobs', exact: true }).click();
    const startRenders = await rasterCount();
    const source = await page.locator('.pdf-pane').boundingBox(), data = await page.locator('.review-cards').boundingBox();
    assert.ok(source.height >= 800 * .75, 'reading area should receive most of the viewport');
    assert.ok(source.x + source.width <= data.x, 'PDF and records must be side by side');
    const sharpness = await page.locator('.pdf-pane [data-pdf-page="1"] canvas').evaluate((canvas) => canvas.width / canvas.getBoundingClientRect().width);
    assert.ok(sharpness >= 1.9, 'normal-sized PDF text must use the high-DPI raster');
    await selectCard(0); await page.locator('.pdf-pane__highlight').waitFor();
    assert.equal(await rasterCount(), startRenders);
    await selectCard(1);
    await page.waitForFunction(() => document.querySelector('.pdf-pane__highlight')?.style.top === '75%');
    assert.equal(await rasterCount(), startRenders, 'same-page selections must not repaint the PDF');
    await selectCard(0); await page.waitForFunction(() => document.querySelector('.pdf-pane__highlight')?.style.top === '32.5%');
    assert.equal(boxReads.A, 1, 'box metadata must be reused');
    await selectCard(2); await canvasReady(3); pages.push(3);
    const afterPage3 = await rasterCount();
    await selectCard(0); await canvasReady(1);
    assert.ok(await rasterCount() >= afterPage3);
    assert.equal(pdfReads, 1, 'record selection must not refetch the PDF');
    assert.equal(pagePngReads, 0, 'the review must not request redundant page PNGs');

    // A late metadata response must never pull the reader back to an old row.
    const slowBox = page.waitForResponse((response) => response.url().includes('box-D.json'));
    await selectCard(3); await selectCard(4); await slowBox;
    await canvasReady(2);
    await page.waitForFunction(() => document.querySelector('.pdf-pane__highlight')?.style.left === '50%');
    assert.equal(await page.locator('.review-card--active .page-cell__id-input').inputValue(), 'E');

    await selectCard(0); await canvasReady(1);
    const divider = page.getByRole('separator', { name: 'Resize PDF pane' });
    const handle = await divider.boundingBox(), oldWidth = (await page.locator('.pdf-pane').boundingBox()).width;
    const beforeResize = await rasterCount();
    await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
    await page.mouse.down(); await page.mouse.move(handle.x - 170, handle.y + handle.height / 2, { steps: 14 }); await page.mouse.up();
    assert.ok((await page.locator('.pdf-pane').boundingBox()).width < oldWidth - 100);
    await page.waitForFunction(() => { const c=document.querySelector('.pdf-pane [data-pdf-page="1"] canvas'); return c?.width / c?.getBoundingClientRect().width >= 1.9; });
    await canvasReady(1);
    const resizePageRenders = await rasterCount() - beforeResize;
    assert.ok(await rasterCount() <= beforeResize + 4, 'only nearby pages should sharpen after resizing');
    assert.ok(Number(await page.evaluate(() => localStorage.getItem('bci:workbench:split'))) < 60);
    await divider.dblclick(); await canvasReady(1);
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('bci:workbench:split')) === 55);
    await page.getByRole('button', { name: 'Focus review', exact: true }).click();
    assert.equal(await page.locator('.app-header').isVisible(), false);
    assert.ok((await page.locator('.pdf-pane').boundingBox()).height >= 800 * .85);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.app-header').isVisible(), true);

    // The table uses the same selection/navigation model as cards.
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await page.locator('tr[data-row-index="2"]').getByRole('button', {name:'Go to PDF page 3', exact:true}).click(); await canvasReady(3);
    await page.getByRole('button', { name: 'Cards', exact: true }).click();
    assert.equal(await page.locator('.review-card--active .page-cell__id-input').inputValue(), 'C');

    await page.locator('.pdf-pane__viewport').evaluate((viewport) => { viewport.scrollTop = viewport.querySelector('[data-pdf-page="4"]').offsetTop + 60; });
    await canvasReady(4);
    assert.equal(await page.getByRole('textbox', { name: 'PDF page', exact: true }).inputValue(), '4');
    await selectCard(2); await canvasReady(3);
    // Extracted images open JSME; source images continue to navigate.
    await page.locator('.review-card[data-row-index="0"]').getByAltText('Extracted structure', { exact: true }).click();
    await page.getByRole('dialog', { name: 'Edit structure', exact: true }).waitFor();
    await page.getByRole('dialog').getByRole('button', { name: 'Close structure editor', exact: true }).click();
    assert.equal(await page.locator('.review-card').first().getByRole('textbox', { name: 'IC50 (µM) for compound A', exact: true }).inputValue(), '5.30 ± 0.10');
    await selectCard(2); await canvasReady(3);
    await page.getByRole('button', { name: 'Draw box', exact: true }).click();
    const layer = await page.locator('[data-pdf-page="3"] .pdf-region-layer').boundingBox();
    await page.mouse.move(layer.x + layer.width * .1, layer.y + 30);
    await page.mouse.down();
    await page.mouse.move(layer.x + layer.width * .4, layer.y + 110, { steps: 4 });
    await page.mouse.up();
    await page.getByRole('button', { name: 'Recognize', exact: true }).click();
    await page.getByRole('dialog', { name: 'Edit structure', exact: true }).waitFor();
    assert.equal(manualRequest.page, 3);
    assert.ok(Math.abs(manualRequest.x - .1) < .01);
    assert.ok(Math.abs(manualRequest.w - .3) < .01);
    await page.getByRole('dialog').getByRole('button', { name: 'Close structure editor', exact: true }).click();

    const activity = page.locator('.review-card').first().getByRole('textbox', { name: 'IC50 (µM) for compound A', exact: true });
    const savedActivity = page.waitForResponse((response) => response.url().endsWith('/structures') && response.request().method() === 'PUT' && response.request().postDataJSON().records[0].USER_MEASUREMENT_EDITS?.includes('7.2'));
    await activity.fill('7.2'); await activity.press('Enter');
    await savedActivity;
    await page.waitForTimeout(1000);
    assert.equal(JSON.parse(records[0].USER_MEASUREMENT_EDITS)['IC50 (µM)'], '7.2');
    await page.getByRole('button',{name:'More actions',exact:true}).click();
    await page.getByRole('button', { name: 'Undo last save', exact: true }).click();
    await page.waitForFunction(() => document.body.innerText.includes('Previous saved version restored'));
    assert.equal(await activity.inputValue(), '5.30 ± 0.10');
    assert.ok(saves.length > 0, 'edits must persist through the API');
    await page.locator('.review-manage-menu').evaluate((element) => element.removeAttribute('open'));

    await page.getByLabel('Add data', { exact: true }).click();
    await page.getByRole('button', { name: 'Add measurement', exact: true }).click();
    await page.getByRole('dialog', { name: 'Add measurement', exact: true }).getByLabel('Endpoint', { exact: true }).fill('Residence time');
    await page.getByRole('dialog', { name: 'Add measurement', exact: true }).getByLabel('Unit', { exact: true }).fill('min');
    await page.getByRole('dialog', { name: 'Add measurement', exact: true }).getByRole('button', { name: 'Add measurement', exact: true }).click();
    const addedMeasurement = page.locator('.review-card').first().getByRole('textbox', { name: 'Residence time (min) for compound A', exact: true });
    await addedMeasurement.waitFor(); assert.equal(await addedMeasurement.inputValue(), '');
    await page.locator('.review-manage-menu').evaluate((element) => element.removeAttribute('open'));

    await selectCard(2); await canvasReady(3);
    await page.getByRole('button', { name: 'Draw box', exact: true }).click();
    await page.getByLabel('Selection purpose', { exact: true }).selectOption('table');
    const tableLayer = await page.locator('[data-pdf-page="3"] .pdf-region-layer').boundingBox();
    await page.mouse.move(tableLayer.x + tableLayer.width * .1, tableLayer.y + 25); await page.mouse.down();
    await page.mouse.move(tableLayer.x + tableLayer.width * .6, tableLayer.y + 130, { steps: 4 }); await page.mouse.up();
    await page.getByRole('button', { name: 'Extract rows', exact: true }).click();
    const tableValue = page.locator('.review-card').first().getByRole('textbox', { name: 'Permeability (cm/s) for compound A', exact: true });
    await tableValue.waitFor(); assert.equal(await tableValue.inputValue(), '1.4e-6');
    await page.locator('.review-card__empty-structure').filter({ hasText: 'Review proposed structure' }).click();
    await page.getByRole('dialog', { name: 'Edit structure', exact: true }).getByText('printed source name', { exact: true }).waitFor();
    await page.getByRole('dialog').getByRole('button', { name: 'Close structure editor', exact: true }).click();
    assert.equal(records.find((record) => record.COMPOUND_ID === 'proposal')?.SMILES || '', '', 'reviewing a proposal must not accept it automatically');


    await page.getByRole('button', { name: 'Table', exact: true }).click();
    assert.equal((await page.locator('.review-page-link').first().innerText()).trim(), '1');
    await page.getByRole('button', { name: 'Sort', exact: true }).click();
    await page.getByRole('combobox', { name: 'Sort column', exact: true }).selectOption('COMPOUND_ID');
    await page.getByRole('combobox', { name: 'Sort direction', exact: true }).selectOption('desc');
    assert.equal(await page.locator('tbody tr .page-cell__id-input').first().inputValue(), 'proposal', 'sorting must change visible order without changing record identity');
    await page.getByRole('button',{name:'Filters',exact:true}).click();
    await page.getByRole('combobox',{name:'Filter column',exact:true}).selectOption('assay:IC50 (µM)');
    await page.getByRole('combobox', { name: 'Filter condition', exact: true }).selectOption('gt');
    await page.getByRole('textbox', { name: 'Filter value', exact: true }).fill('6');
    await page.getByRole('cell', { name: 'No rows match the current filters.', exact: true }).waitFor();
    await page.getByRole('textbox', { name: 'Filter value', exact: true }).fill('5');
    assert.equal(await page.locator('tbody tr .page-cell__id-input').count(), 5, 'point ± uncertainty values must support numeric filtering');
    await page.getByRole('button',{name:'Clear filters',exact:true}).click();
    await page.getByRole('button',{name:'Sort',exact:true}).click();
    await page.getByRole('button', { name: 'Clear sorts', exact: true }).click();
    await page.getByRole('button', { name: 'Columns', exact: true }).click();
    await page.locator('.columns-picker__option').filter({ hasText: 'Residence time (min)' }).getByRole('checkbox').uncheck();
    assert.equal(await page.locator('th').filter({ hasText: 'Residence time (min)' }).count(), 0);
    await page.locator('.columns-picker__option').filter({ hasText: 'Residence time (min)' }).getByRole('checkbox').check();
    await page.getByRole('button', { name: 'Columns', exact: true }).click();
    await page.locator('th .review-column-name').filter({ hasText: 'Residence time (min)' }).click();
    await page.getByRole('textbox', { name: 'Column name', exact: true }).fill('Residence time (s)');
    await page.getByRole('button', { name: 'Save name', exact: true }).click();
    await page.locator('th .review-column-name').filter({ hasText: 'Residence time (s)' }).waitFor();
    assert.equal(await page.locator('th .review-column-name').filter({ hasText: 'Residence time (min)' }).count(), 0);
    await page.getByRole('button', { name: 'Cards', exact: true }).click();
    recognizedCompoundId = 'A';
    await selectCard(0);
    await canvasReady(1);
    const beforeReplacementCount = await page.locator('.review-card').count();
    await page.getByRole('button', { name: 'Draw box', exact: true }).click();
    await page.getByLabel('Selection purpose', { exact: true }).selectOption('structure');
    await page.getByRole('checkbox', { name: 'Replace structure for A', exact: true }).check();
    const replacementLayer = await page.locator('[data-pdf-page="1"] .pdf-region-layer').boundingBox();
    await page.mouse.move(replacementLayer.x + 25, replacementLayer.y + 30); await page.mouse.down();
    await page.mouse.move(replacementLayer.x + 140, replacementLayer.y + 120, { steps: 4 }); await page.mouse.up();
    await page.getByRole('button', { name: 'Recognize', exact: true }).click();
    await page.getByRole('dialog', { name: 'Edit structure', exact: true }).waitFor();
    assert.equal(await page.locator('.review-card').count(), beforeReplacementCount, 'replacing a selected structure must not append another row');
    assert.equal(await page.locator('.review-card').first().getByRole('textbox', { name: 'IC50 (µM) for compound A', exact: true }).inputValue(), '5.30 ± 0.10', 'replacing a structure must preserve measurements');
    await page.getByRole('dialog').getByRole('button', { name: 'Close structure editor', exact: true }).click();

    await selectCard(3); await canvasReady(5);
    await page.setViewportSize({ width: 375, height: 812 });
    await page.waitForFunction(() => { const viewport = document.querySelector('.pdf-pane__viewport')?.getBoundingClientRect(); const marker = document.querySelector('.pdf-pane__highlight')?.getBoundingClientRect(); const paper = document.querySelector('.pdf-pane__paper')?.getBoundingClientRect(); return viewport && marker && paper && paper.width <= viewport.width && marker.top < viewport.bottom && marker.bottom > viewport.top; });
    await page.getByRole('button', { name: 'Cards', exact: true }).click();
    assert.equal(await page.locator('.pdf-pane').count(), 1);
    await selectCard(1); await canvasReady(1);
    assert.equal(await page.locator('.review-cards').isVisible(), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.pdf-pane').isVisible(), true);
    assert.equal(await page.locator('.review-cards').isVisible(), true);
    assert.equal(pdfReads, 1, 'mobile pane switches must retain the parsed document');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 375);
    assert.ok((await page.locator('.review-card').first().boundingBox()).height > 250, 'cards must keep their header, figures and actions visible');
    await page.screenshot({ path: '/tmp/biochem-review-cards-mobile.png', fullPage: true });
    await page.getByRole('button', { name: 'Jobs', exact: true }).click();
    await openJob.waitFor();
    const mobileOpen = await openJob.boundingBox();
    assert.ok(mobileOpen.x >= 0 && mobileOpen.x + mobileOpen.width <= 375, 'mobile job actions must be visible on opening');
    await page.screenshot({ path: '/tmp/biochem-jobs-mobile.png', fullPage: true });
    await page.getByRole('button', { name: 'Close jobs', exact: true }).click();
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ linkedCardsAndTable: true, samePageRepaints: 0, pdfDownloads: pdfReads, pagePngDownloads: pagePngReads, metadataReads: boxReads, resizePageRenders, highDpi: sharpness, mobileAndFocusMode: true }));
  } catch (error) {
    await page.screenshot({path:'/tmp/bci-nav-failure.png'});
    console.log('PDF failure', await page.locator('.pdf-pane').evaluate(e=>({page:e.dataset.page, renders:e.dataset.renderCount, viewport:{height:e.querySelector('.pdf-pane__viewport').clientHeight,scroll:e.querySelector('.pdf-pane__viewport').scrollTop}, pages:[...e.querySelectorAll('.pdf-page')].map(p=>({number:p.dataset.pdfPage,top:p.offsetTop, canvas:p.querySelector('canvas').width}))})), errors);
    throw error;
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exit(1); });

const { makePdf, thumbnail } = require('./browser-fixtures.cjs');
const fixturePdf = makePdf(1);
// UI persistence regressions use an async HTTP fixture; no models are invoked.
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const assert = require('assert/strict');
const image = 'data:image/svg+xml;base64,' + Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"><path d="M10 40L40 20L70 40" fill="none" stroke="black"/></svg>').toString('base64');
const makeZip = (rows) => execFileSync('python', ['-c',
  'import sys,json,io,csv,zipfile; rows=json.load(sys.stdin); text=io.StringIO(); w=csv.DictWriter(text,fieldnames=["COMPOUND_ID","SMILES"],extrasaction="ignore"); w.writeheader(); w.writerows(rows); data=io.BytesIO(); z=zipfile.ZipFile(data,"w"); z.writestr("structures.csv",text.getvalue()); z.close(); sys.stdout.buffer.write(data.getvalue())',
], { input: JSON.stringify(rows) });

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    for (const kind of ['structure_extraction', 'full_pipeline']) {
      const page = await browser.newPage();
      page.setDefaultTimeout(15000);
      let rows = [{ COMPOUND_ID: '53', SMILES: 'CCO' }, { COMPOUND_ID: '54', SMILES: 'CCN' }];
      let inFlight = 0, maxInFlight = 0, failNext = false, downloadedRows = null, writeDelay = 350;
      const task = { task_id: 'edit-job', type: kind, status: 'completed', progress: 1, result_path: '/qa/structures.csv', pdf_id: 'qa', params: {}, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      await page.route('**/api/**', async (route) => {
        const path = new URL(route.request().url()).pathname;
        if (!path.startsWith('/api/')) return route.continue();
        if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:fixturePdf});
        if (path.endsWith('/structures') && route.request().method() === 'PUT') {
          inFlight++; maxInFlight = Math.max(maxInFlight, inFlight);
          const submitted = route.request().postDataJSON().records;
          await new Promise((resolve) => setTimeout(resolve, writeDelay));
          inFlight--;
          if (failNext) { failNext = false; return route.fulfill({ status: 503, json: { detail: 'Save service unavailable' } }); }
          rows = submitted;
          return route.fulfill({ json: { task, records: rows, filtered_records: [] } });
        }
        if (path.endsWith('/download')) {
          downloadedRows = structuredClone(rows);
          return route.fulfill({ contentType: 'application/zip', body: makeZip(rows) });
        }
        let json = {};
        if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Save validation.pdf', total_pages: 1 };
        else if (path.includes('/pages/')) json = { image: thumbnail, page: 1 };
        else if (path.endsWith('/structures')) json = { task, records: rows, filtered_records: [] };
        else if (path.endsWith('/assays')) json = { task, records: [] };
        else if (path === '/api/tasks') json = { tasks: [task], running_count: 0, pending_count: 0, total_count: 1, total_pages: 1, revision: 'qa' };
        else if (path === '/api/tasks/edit-job') json = task;
        else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
        return route.fulfill({ json });
      });
      const parameter = kind === 'full_pipeline' ? 'fullPipelineTask' : 'structureTask';
      const url = `${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&${parameter}=edit-job&step=4`;
      await page.goto(url);
      const input = () => page.locator('.review-card .page-cell__id-input').first();
      await input().waitFor();
      failNext = true;
      await input().fill('first-edit'); await input().press('Enter');
      await page.getByRole('button', { name: 'Retry save', exact: true }).waitFor();
      await page.getByRole('button', { name: 'Retry save', exact: true }).click();
      await page.getByRole('status', { name: 'Saved', exact: true }).waitFor();
      assert.equal(rows[0].COMPOUND_ID, 'first-edit');
      assert.equal(await page.getByRole('alert').count(), 0, 'successful retry must clear its failure');

      // Reopening a job while its current edit is still being saved must
      // flush that edit before replacing the workspace with a fresh GET.
      await page.locator('.jobs-button').click();
      await page.getByRole('button', { name: 'Open job edit-job', exact: true }).waitFor();
      writeDelay = 1200;
      const switchingSave = page.waitForRequest((request) => request.method() === 'PUT');
      await input().fill('switch-edit'); await input().press('Enter'); await switchingSave;
      const reopened = page.waitForResponse((response) => response.request().method() === 'GET' && response.url().endsWith('/structures'));
      await page.getByRole('button', { name: 'Open job edit-job', exact: true }).click();
      await reopened;
      assert.equal(rows[0].COMPOUND_ID, 'switch-edit');
      await input().waitFor();
      assert.equal(await input().inputValue(), 'switch-edit');
      writeDelay = 350;
      if (await page.locator('.jobs-panel').isVisible()) await page.getByRole('button', { name: 'Close jobs', exact: true }).click();

      const firstRequest = page.waitForRequest((request) => request.method() === 'PUT');
      await input().fill('older-edit'); await input().press('Enter'); await firstRequest;
      await input().fill('latest-edit'); await input().press('Enter');
      const download = page.waitForEvent('download');
      if (kind === 'structure_extraction') {
        await page.locator('.jobs-button').click();
        await page.getByRole('link', { name: 'Download result for job edit-job', exact: true }).click();
      } else {
        await page.getByRole('button', { name: 'Export', exact: true }).click();
      }
      const file = await download;
      if (kind === 'structure_extraction') await page.locator('.jobs-button').click();
      assert.equal(await file.failure(), null);
      assert.equal(downloadedRows[0].COMPOUND_ID, 'latest-edit', 'export must flush recent edits');
      assert.equal(rows[0].COMPOUND_ID, 'latest-edit');
      assert.equal(maxInFlight, 1, 'PUTs for successive edits must remain ordered');

      page.on('dialog', (dialog) => dialog.accept());
      for (let remaining = 1; remaining >= 0; remaining--) {
        const saved = page.waitForResponse((response) => response.request().method() === 'PUT');
        await page.locator('.review-card').first().getByRole('button', { name: 'Delete', exact: true }).click();
        assert.equal((await saved).status(), 200);
        await page.getByRole('status', { name: 'Saved', exact: true }).waitFor();
        assert.equal(rows.length, remaining);
      }
      await page.reload();
      await page.getByText('No records in this result.', { exact: true }).waitFor();
      const emptyDownload = page.waitForEvent('download');
      await page.getByRole('button', { name: 'Export', exact: true }).click();
      await emptyDownload;
      assert.deepEqual(downloadedRows, [], 'deleted rows must not return after reload/export');
      await page.close();
      console.log(`${kind}: retry, ordered saves, export flush, last-row deletion passed`);
    }
    const racePage = await browser.newPage();
    racePage.setDefaultTimeout(12000);
    let oldGets = 0, oldResultReads = 0, releaseOld;
    const oldResponse = new Promise((resolve) => { releaseOld = resolve; });
    const newTask = { task_id: 'new-job', type: 'full_pipeline', status: 'completed', progress: 1, pdf_id: 'qa', params: {}, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
    await racePage.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (!path.startsWith('/api/')) return route.continue();
        if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:fixturePdf});
      let json = {};
      if (path === '/api/tasks/old-job') {
        oldGets++;
        if (oldGets > 1) await oldResponse;
        json = { ...newTask, task_id: 'old-job', status: oldGets > 1 ? 'completed' : 'running' };
      } else if (path.endsWith('/structures')) {
        if (path.includes('/old-job/')) oldResultReads++;
        json = { records: [{ COMPOUND_ID: path.includes('/old-job/') ? 'old' : 'new', SMILES: 'CCO' }], filtered_records: [] };
      } else if (path.endsWith('/assays')) json = { records: [] };
      else if (path === '/api/tasks/new-job') json = newTask;
      else if (path === '/api/tasks') json = { tasks: [newTask], running_count: 1, pending_count: 0, total_count: 1, total_pages: 1, revision: 'race' };
      else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Race validation.pdf', total_pages: 1 };
      else if (path.includes('/pages/')) json = { page: 1, image: thumbnail };
      else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
      return route.fulfill({ json });
    });
    await racePage.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&fullPipelineTask=old-job&step=1`);
    await racePage.getByRole('button', { name: 'Cancel', exact: true }).waitFor();
    await racePage.waitForRequest((request) => request.method() === 'GET' && request.url().endsWith('/tasks/old-job'));
    await racePage.locator('.jobs-button').click();
    await racePage.getByRole('button', { name: 'Open job new-job', exact: true }).click();
    const currentId = () => racePage.locator('.review-card .page-cell__id-input');
    await currentId().waitFor();
    assert.equal(await currentId().inputValue(), 'new');
    const finishedOld = racePage.waitForResponse((response) => response.url().endsWith('/tasks/old-job'));
    releaseOld(); await finishedOld;
    await racePage.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await currentId().inputValue(), 'new');
    assert.equal(oldResultReads, 0, 'a stale poll must not reload the previous task');
    assert.equal(new URL(racePage.url()).searchParams.get('fullPipelineTask'), 'new-job');
    await racePage.close();
    console.log('A delayed poll from the previous job cannot replace the current workspace');

    const loadingPage = await browser.newPage();
    loadingPage.setDefaultTimeout(12000);
    let releaseResults;
    const heldResults = new Promise((resolve) => { releaseResults = resolve; });
    await loadingPage.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (!path.startsWith('/api/')) return route.continue();
        if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:fixturePdf});
      let json = {};
      if (path.endsWith('/structures')) { await heldResults; json = { records: [{ COMPOUND_ID: 'loaded', SMILES: 'CCO' }], filtered_records: [] }; }
      else if (path.endsWith('/assays')) json = { records: [] };
      else if (path === '/api/tasks/new-job') json = newTask;
      else if (path === '/api/tasks') json = { tasks: [], running_count: 0, pending_count: 0, total_pages: 1, revision: 'loading' };
      else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Loading validation.pdf', total_pages: 1 };
      else if (path.includes('/pages/')) json = { page: 1, image: thumbnail };
      else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
      return route.fulfill({ json });
    });
    const loadingRequest = loadingPage.waitForRequest((request) => request.url().endsWith('/structures'));
    await loadingPage.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&fullPipelineTask=new-job&step=4`);
    await loadingRequest;
    await loadingPage.locator('.review-toolbar').waitFor();
    assert.equal(await loadingPage.getByRole('button', { name: 'Export', exact: true }).count(), 0, 'export must wait until rows are loaded');
    releaseResults();
    await loadingPage.getByRole('button', { name: 'Export', exact: true }).waitFor();
    await loadingPage.close();
    console.log('Completed task metadata alone cannot expose an empty save/export before results load');

    const restartPage = await browser.newPage();
    restartPage.setDefaultTimeout(12000);
    const saveTargets = [];
    const oldComplete = { ...newTask, task_id: 'previous-result', type: 'structure_extraction' };
    await restartPage.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (!path.startsWith('/api/')) return route.continue();
        if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:fixturePdf});
      if (path.endsWith('/download')) return route.fulfill({ contentType: 'application/zip', body: makeZip([]) });
      let json = {};
      if (path.endsWith('/structures') && route.request().method() === 'PUT') {
        saveTargets.push(path.split('/')[3]);
        assert.deepEqual(route.request().postDataJSON().records, []);
        json = { records: [], filtered_records: [] };
      } else if (path === '/api/tasks/full-pipeline') json = { ...newTask, status: 'pending', progress: 0 };
      else if (path === '/api/tasks/previous-result') json = oldComplete;
      else if (path === '/api/tasks/new-job') json = newTask;
      else if (path.endsWith('/structures')) json = { records: path.includes('/previous-result/') ? [{ COMPOUND_ID: 'previous', SMILES: 'CCO' }] : [], filtered_records: [] };
      else if (path.endsWith('/assays')) json = { records: [] };
      else if (path === '/api/tasks') json = { tasks: [], running_count: 0, pending_count: 0, total_pages: 1, revision: 'restart' };
      else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Restart validation.pdf', total_pages: 1 };
      else if (path.includes('/pages/')) json = { page: 1, image: thumbnail };
      else if (path === '/api/chem/render-batch') json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image })) };
      return route.fulfill({ json });
    });
    await restartPage.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&structureTask=previous-result&step=4`);
    await restartPage.locator('.review-card .page-cell__id-input').waitFor();
    await restartPage.locator('.step-nav__button').first().click();
    await restartPage.getByRole('button', { name: 'Start extraction', exact: true }).click();
    await restartPage.getByText('No records in this result.', { exact: true }).waitFor();
    assert.equal(await restartPage.locator('.review-card .page-cell__id-input').count(), 0);
    const emptyRestartDownload = restartPage.waitForEvent('download');
    await restartPage.getByRole('button', { name: 'Export', exact: true }).click();
    await emptyRestartDownload;
    assert.deepEqual(saveTargets, ['new-job'], 'a new run must never save its rows into the previous task');
    assert.equal(new URL(restartPage.url()).searchParams.get('structureTask'), null);
    await restartPage.close();
    console.log('A restarted pipeline clears previous results and saves only to the new task');

    const cancelPage = await browser.newPage();
    cancelPage.setDefaultTimeout(12000);
    let failCancellation = true;
    let state = 'running';
    await cancelPage.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (!path.startsWith('/api/')) return route.continue();
        if(path.endsWith('/file'))return route.fulfill({contentType:'application/pdf',body:fixturePdf});
      if (path.endsWith('/cancel')) {
        if (failCancellation) return route.fulfill({ status: 503, json: { detail: 'Cancellation service unavailable' } });
        state = 'canceled';
      }
      let json = {};
      if (path.includes('/tasks/cancel-job')) json = { task_id: 'cancel-job', type: 'full_pipeline', status: state, progress: .1, params: {}, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      else if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Cancel validation.pdf', total_pages: 1 };
      else if (path.includes('/pages/')) json = { page: 1, image: thumbnail };
      else if (path === '/api/tasks') json = { tasks: [], running_count: state === 'running' ? 1 : 0, pending_count: 0, revision: 'qa' };
      return route.fulfill({ json });
    });
    await cancelPage.goto(`${process.env.WORKBENCH_URL || 'http://127.0.0.1:5173'}/?pdf=qa&fullPipelineTask=cancel-job&step=1`);
    await cancelPage.getByRole('button', { name: 'Cancel', exact: true }).click();
    await cancelPage.getByRole('alert').waitFor();
    await cancelPage.getByRole('button', { name: 'Cancel', exact: true }).waitFor();
    await cancelPage.waitForRequest((request) => request.method() === 'GET' && request.url().endsWith('/tasks/cancel-job'));
    failCancellation = false;
    await cancelPage.getByRole('button', { name: 'Cancel', exact: true }).click();
    await cancelPage.getByRole('button', { name: 'Start extraction', exact: true }).waitFor();
    assert.equal(await cancelPage.getByRole('alert').count(), 0);
    assert.ok(cancelPage.url().includes('fullPipelineTask=cancel-job'));
    await cancelPage.close();
    console.log('Failed cancellation keeps polling and allows retry; accepted cancellation retains job handle');
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exit(1); });

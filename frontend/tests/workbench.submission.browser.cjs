const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');

const baseUrl = process.env.WORKBENCH_URL || 'http://127.0.0.1:5173';

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function checkSubmission(browser, scenario) {
  const page = await browser.newPage();
  const response = deferred();
  const errors = [];
  let submitted = 0;
  let cancelRequests = 0;
  let state = 'pending';
  const task = () => ({ task_id: 'submission', type: 'full_pipeline', status: state, pdf_id: 'qa', progress: 0, params: {} });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(1) });
    let json = {};
    if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Submission.pdf', total_pages: 1 };
    else if (path === '/api/capabilities') json = { markush_assembly: { available: true } };
    else if (path === '/api/tasks/full-pipeline') {
      submitted += 1;
      await response.promise;
      if (scenario === 'submit_error') return route.fulfill({ status: 503, json: { detail: 'Submission unavailable' } });
      if (scenario === 'finished') state = 'completed';
      json = task();
    } else if (path.endsWith('/cancel')) {
      cancelRequests += 1;
      if (scenario === 'cancel_retry' && cancelRequests === 1) return route.fulfill({ status: 503, json: { detail: 'Cancellation unavailable' } });
      if (scenario === 'cancel_unconfirmed' && cancelRequests === 1) return route.fulfill({ json: task() });
      state = 'canceled';
      json = task();
    } else if (path === '/api/tasks/submission') json = task();
    else if (path.endsWith('/structures')) json = { task: task(), records: [], filtered_records: [] };
    else if (path.endsWith('/assays')) json = { task: task(), records: [] };
    else if (path === '/api/tasks') json = { tasks: [], total_pages: 1, total_count: 0, running_count: 0, pending_count: state === 'pending' ? 1 : 0 };
    return route.fulfill({ json });
  });
  try {
    await page.goto(`${baseUrl}/?pdf=qa&step=1`);
    const request = page.waitForRequest((item) => item.url().endsWith('/api/tasks/full-pipeline'));
    await page.getByRole('button', { name: 'Start extraction', exact: true }).click();
    await request;
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    const canceling = page.getByRole('button', { name: 'Canceling…', exact: true });
    await canceling.waitFor();
    assert.equal(await canceling.isDisabled(), true);
    assert.equal(await page.getByRole('button', { name: 'Start extraction', exact: true }).count(), 0);
    assert.equal(submitted, 1);
    assert.equal(cancelRequests, 0);
    assert.equal(await page.getByText('Jobs finished before cancellation. Checking their results.', { exact: true }).count(), 0);
    response.resolve();
    if (scenario === 'submit_error') {
      await page.getByRole('alert').getByText('Submission unavailable', { exact: true }).waitFor();
      await page.getByRole('button', { name: 'Start extraction', exact: true }).waitFor();
      assert.equal(cancelRequests, 0);
    } else if (scenario === 'finished') {
      await page.getByText('The job finished before cancellation.', { exact: true }).waitFor();
      assert.equal(cancelRequests, 0);
      assert.equal(state, 'completed');
    } else {
      if (scenario === 'cancel_retry' || scenario === 'cancel_unconfirmed') {
        await page.getByRole('alert').getByText(scenario === 'cancel_retry' ? 'Cancellation unavailable' : 'Cancellation was not accepted. The job is still being tracked.', { exact: true }).waitFor();
        const cancel = page.getByRole('button', { name: 'Cancel', exact: true });
        assert.equal(await cancel.isEnabled(), true);
        assert.equal(new URL(page.url()).searchParams.get('fullPipelineTask'), 'submission');
        await cancel.click();
      }
      await page.getByRole('button', { name: 'Start extraction', exact: true }).waitFor();
      assert.equal(state, 'canceled');
      assert.equal(cancelRequests, scenario === 'cancel_retry' || scenario === 'cancel_unconfirmed' ? 2 : 1);
    }
    assert.equal(submitted, 1);
    assert.deepEqual(errors, []);
  } finally {
    response.resolve();
    await page.close();
  }
}


async function checkCancellationDuringSave(browser) {
  const page = await browser.newPage();
  const save = deferred();
  let submitted = 0;
  let records = [{ RECORD_ID: 'row-0', COMPOUND_ID: '6a', PAGE_NUM: 1, SMILES: 'CCO' }];
  const task = { task_id: 'previous', type: 'structure_extraction', status: 'completed', pdf_id: 'qa', progress: 1, params: {} };
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(1) });
    let json = {};
    if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Previous.pdf', total_pages: 1 };
    else if (path === '/api/tasks/previous') json = task;
    else if (path === '/api/tasks') json = { tasks: [task], total_pages: 1, total_count: 1 };
    else if (path.endsWith('/structures')) {
      if (route.request().method() === 'PUT') {
        await save.promise;
        records = route.request().postDataJSON().records;
      }
      json = { task, records, filtered_records: [] };
    } else if (path.endsWith('/render-batch')) json = { results: route.request().postDataJSON().items.map((item) => ({ ...item, image: `data:image/png;base64,${thumbnail}` })) };
    else if (path === '/api/tasks/full-pipeline') {
      submitted += 1;
      json = { ...task, task_id: 'unexpected', type: 'full_pipeline', status: 'pending' };
    }
    return route.fulfill({ json });
  });
  try {
    await page.goto(`${baseUrl}/?pdf=qa&structureTask=previous&step=4`);
    const input = page.locator('.review-card .page-cell__id-input').first();
    await input.waitFor();
    const saving = page.waitForRequest((request) => request.method() === 'PUT');
    await input.fill('corrected');
    await input.press('Enter');
    await saving;
    await page.locator('.step-nav').getByRole('button', { name: 'Upload PDF', exact: true }).click();
    await page.getByRole('button', { name: 'Start extraction', exact: true }).click();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'Canceling…', exact: true }).isDisabled(), true);
    assert.equal(submitted, 0);
    save.resolve();
    await page.getByText('Submission canceled.', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Start extraction', exact: true }).waitFor();
    assert.equal(submitted, 0);
    assert.equal(records[0].COMPOUND_ID, 'corrected');
  } finally {
    save.resolve();
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch();
  try {
    for (const scenario of ['cancel', 'cancel_retry', 'cancel_unconfirmed', 'submit_error', 'finished']) await checkSubmission(browser, scenario);
    await checkCancellationDuringSave(browser);
    console.log(JSON.stringify({ canceledBeforeQueue: true, pendingCancellation: true, duplicateSubmissionBlocked: true, cancellationRetry: true, unconfirmedCancellation: true, submissionFailure: true, finishedBeforeCancellation: true }));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exit(1); });

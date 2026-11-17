const { chromium } = require('playwright');
const assert = require('assert/strict');
const fs = require('fs');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');

const baseUrl = process.env.WORKBENCH_URL || 'http://127.0.0.1:5173';
const output = process.env.QA_OUTPUT_DIR || '/tmp/biochem-extraction';

async function openWorkspace(browser, { available = true, checked = false, viewport, delayCapabilities = false, capabilityFailure = false } = {}) {
  const page = await browser.newPage({ viewport: viewport || { width: 1440, height: 900 } });
  const errors = [];
  const submissions = [];
  let capabilityRequests = 0;
  let resolveCapabilities;
  const capabilitiesReady = delayCapabilities ? new Promise(resolve => { resolveCapabilities = resolve; }) : Promise.resolve();
  const task = () => ({ task_id: 'assembly-task', type: 'full_pipeline', status: 'pending', progress: 0, params: submissions.at(-1) });
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(2) });
    let json = {};
    if (path === '/api/capabilities') {
      await capabilitiesReady;
      capabilityRequests++;
      if (capabilityFailure && capabilityRequests === 1) return route.fulfill({ status: 503, json: { detail: 'Service unavailable' } });
      json = { markush_assembly: { available, requires_review: true } };
    } else if (path === '/api/pdfs/qa') {
      json = { pdf_id: 'qa', filename: 'Very long source patent identifier with attached supporting information.pdf', total_pages: 2 };
    } else if (path === '/api/pdfs' && route.request().method() === 'POST') {
      json = { pdf_id: 'replacement', filename: 'Replacement.pdf', total_pages: 2 };
    } else if (path.includes('/pages/')) {
      json = { image: thumbnail, page: 1 };
    } else if (path === '/api/tasks/full-pipeline') {
      submissions.push(route.request().postDataJSON());
      json = task();
    } else if (path === '/api/tasks/assembly-task') {
      json = task();
    } else if (path === '/api/tasks') {
      json = { tasks: [], total_count: 0, total_pages: 1, running_count: 0, pending_count: 0 };
    }
    return route.fulfill({ json });
  });
  await page.goto(`${baseUrl}/?pdf=qa&step=1`);
  await page.locator('.launch-card').waitFor();
  const option = page.getByRole('switch', { name: 'Markush assembly', exact: true });
  if (!delayCapabilities) await page.waitForFunction(() => !document.querySelector('[aria-label="Checking Markush availability"]'));
  if (checked) await option.check();
  return { page, option, submissions, errors, resolveCapabilities };
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    for (const available of [true, false]) {
      const { page, option, errors } = await openWorkspace(browser, { available });
      assert.equal(await option.isEnabled(), available);
      assert.equal(await option.isChecked(), false);
      assert.equal(await page.locator('.extraction-options__results li').count(), 3);
      assert.equal(await page.locator('.extraction-options__details').count(), 0);
      assert.equal(await page.locator('.source-import').getAttribute('open'), null);
      assert.equal(await page.locator('.patent-import').isVisible(), false);
      assert.equal(await page.locator('.launch-card').getByText('Optional', { exact: true }).count(), 0);

      const help = page.getByRole('button', { name: 'About Markush assembly' });
      await help.click();
      assert.equal(await help.getAttribute('aria-expanded'), 'true');
      assert.match(await page.locator('.extraction-options__details').innerText(), available ? /Review proposed compounds/ : /unavailable/);
      await page.keyboard.press('Escape');
      assert.equal(await help.getAttribute('aria-expanded'), 'false');
      assert.equal(await page.locator('.extraction-options__details').count(), 0);

      for (const width of [320, 375, 600, 768, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${width}: document overflow`);
        const bounds = await page.locator('.extraction-options__assembly').evaluate(element => {
          const parent = element.getBoundingClientRect();
          return [...element.querySelectorAll('svg, input, button, .extraction-options__label, .extraction-options__state')].every(child => {
            const box = child.getBoundingClientRect();
            return box.left >= parent.left - 1 && box.right <= parent.right + 1 && box.top >= parent.top - 1 && box.bottom <= parent.bottom + 1;
          });
        });
        assert.ok(bounds, `${width}: assembly control overflow`);
        assert.ok(await page.locator('.launch-card').evaluate(element => element.scrollWidth <= element.clientWidth + 1), `${width}: launch card content is clipped`);
        if (width === 375 || width === 1440) {
          await page.screenshot({ path: `${output}/${available ? 'available' : 'unavailable'}-${width}.png` });
        }
      }
      assert.deepEqual(errors, []);
      await page.close();
    }

    for (const enabled of [false, true]) {
      const { page, option, submissions, errors } = await openWorkspace(browser);
      if (enabled) {
        await option.focus();
        await page.keyboard.press('Space');
        assert.equal(await option.isChecked(), true);
      }
      await page.getByRole('button', { name: 'Start extraction', exact: true }).click();
      await page.getByRole('button', { name: 'Cancel', exact: true }).waitFor();
      assert.equal(submissions.length, 1);
      assert.equal(submissions[0].assemble_markush, enabled);
      assert.equal(await option.isEnabled(), false);
      assert.deepEqual(errors, []);
      await page.close();
    }

    {
      const { page, option, resolveCapabilities } = await openWorkspace(browser, { delayCapabilities: true });
      await page.getByRole('status', { name: 'Checking Markush availability' }).waitFor();
      assert.equal(await option.isEnabled(), false);
      resolveCapabilities();
      await page.waitForFunction(() => !document.querySelector('[aria-label="Checking Markush availability"]'));
      assert.equal(await option.isEnabled(), true);
      await page.locator('.source-import > summary').click();
      assert.equal(await page.locator('.patent-import').isVisible(), true);
      await page.locator('.source-drop input').setInputFiles({ name: 'Replacement.pdf', mimeType: 'application/pdf', buffer: makePdf(2) });
      await page.getByText('Replacement.pdf', { exact: true }).waitFor();
      assert.equal(await page.locator('.source-import').getAttribute('open'), null);
      await page.close();
    }
    {
      const { page, option, errors } = await openWorkspace(browser, { capabilityFailure: true, viewport: { width: 320, height: 667 } });
      await page.getByText('Check failed', { exact: true }).waitFor();
      assert.equal(await option.isEnabled(), false);
      assert.equal(await page.getByText('Unavailable', { exact: true }).count(), 0);
      await page.getByRole('button', { name: 'About Markush assembly' }).click();
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('.extraction-options__switch input')?.disabled === false);
      assert.equal(await page.getByText('Check failed', { exact: true }).count(), 0);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log(JSON.stringify({ outputs: 3, widths: [320, 375, 600, 768, 1440], keyboardToggle: true, requestFlags: [false, true], busyLock: true, capabilityStates: true, sourceReplacement: true, capabilityRetry: true }));
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});

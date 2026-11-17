const { chromium } = require('playwright');
const assert = require('assert/strict');
const { makePdf, thumbnail } = require('./browser-fixtures.cjs');

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const records = Array.from({ length: 60 }, (_, index) => ({
      RECORD_ID: `row-${index}`, COMPOUND_ID: `Compound ${index}`, SMILES: 'CCO',
      PAGE_NUM: index % 3 + 1,
    }));
    const task = { task_id: 'pagination', type: 'structure_extraction', status: 'completed', pdf_id: 'qa', params: {} };
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith('/file')) return route.fulfill({ contentType: 'application/pdf', body: makePdf(3) });
      let json = {};
      if (path === '/api/pdfs/qa') json = { pdf_id: 'qa', filename: 'Pagination.pdf', total_pages: 3 };
      else if (path.endsWith('/structures')) json = { task, records, filtered_records: [] };
      else if (path === '/api/tasks/pagination') json = task;
      else if (path === '/api/tasks') json = { tasks: [task], total_pages: 1 };
      else if (path.includes('/pages/') || path.includes('/preview')) json = { image: thumbnail };
      else if (path.endsWith('/render-batch')) json = { results: route.request().postDataJSON().items.map(item => ({ ...item, image: `data:image/png;base64,${thumbnail}` })) };
      return route.fulfill({ json });
    });
    const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:3000';
    const pager = page.locator('.review-pagination--top');
    const summary = pager.locator('.review-pagination__summary');
    const row = index => page.locator(`.review-card[data-row-index="${index}"]:visible, tbody tr[data-row-index="${index}"]:visible`);
    const select = async index => { await row(index).locator('.page-cell__id-input').click(); await page.waitForFunction(i => document.querySelector(`.review-card[data-row-index="${i}"]`)?.getAttribute('aria-current') === 'true' || document.querySelector(`tbody tr[data-row-index="${i}"]`)?.getAttribute('aria-selected') === 'true', index); };
    const next = () => pager.getByRole('button', { name: 'Next page', exact: true }).click();
    const previous = () => pager.getByRole('button', { name: 'Previous page', exact: true }).click();
    const expectPage = async number => {
      await page.waitForFunction(n => document.querySelector('.review-pagination--top .review-pagination__summary')?.textContent.trim() === `${n}/3`, number);
      await row((number - 1) * 25).waitFor();
      await page.waitForTimeout(180);
      assert.equal((await summary.textContent()).trim(), `${number}/3`);
    };
    await page.goto(`${base}/?pdf=qa&structureTask=pagination&step=4`);
    for (const view of ['Table', 'Cards']) {
      await page.getByRole('button', { name: view, exact: true }).click();
      await expectPage(1);
      if (view === 'Table') {
        const wrapper = page.locator('.review-split__list > .table-wrapper');
        await wrapper.evaluate(element => { element.scrollTop = 185; element.scrollLeft = 120; });
        await page.waitForTimeout(50);
        const gap = await wrapper.evaluate(element => element.querySelector('th').getBoundingClientRect().top - element.getBoundingClientRect().top - element.clientTop);
        assert.ok(Math.abs(gap) <= 1, `Sticky header gap: ${gap}`);
        await wrapper.evaluate(element => { element.scrollTop = 0; element.scrollLeft = 0; });
      }
      await select(0);
      await next(); await expectPage(2);
      await next(); await expectPage(3);
      assert.ok(await pager.getByRole('button', { name: 'Next page', exact: true }).isDisabled());
      await previous(); await expectPage(2);
      await select(25);
      await previous(); await expectPage(1);
      assert.ok(await pager.getByRole('button', { name: 'Previous page', exact: true }).isDisabled());
      await next(); await expectPage(2);
      await select(25);
      await previous(); await expectPage(1);
      await page.evaluate(() => document.activeElement?.blur());
      await page.keyboard.press('ArrowDown');
      await select(0);
      await next(); await expectPage(2);
      await page.evaluate(() => document.activeElement?.blur());
      await page.keyboard.press('ArrowDown');
      await page.waitForFunction(() => document.querySelector('.review-card[data-row-index="25"]')?.getAttribute('aria-current') === 'true' || document.querySelector('tbody tr[data-row-index="25"]')?.getAttribute('aria-selected') === 'true');
      await expectPage(2);
      await page.keyboard.press('ArrowUp'); await expectPage(1);
      await select(0);
    }
    // Changing presentation must not return to a selection on another page.
    await next(); await expectPage(2);
    await page.getByRole('button', { name: 'Table', exact: true }).click(); await expectPage(2);
    await page.getByRole('button', { name: 'Cards', exact: true }).click(); await expectPage(2);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ selectedRowThenNext: true, selectedRowThenPrevious: true, pageLimits: true, tableAndCards: true, keyboardStartsOnVisiblePage: true, keyboardCrossesPages: true, viewSwitchPreservesPage: true }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
